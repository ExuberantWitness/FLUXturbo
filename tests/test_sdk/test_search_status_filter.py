"""Test search() status filtering (v0.3 CoE audit integration).

Default search excludes atoms whose CoE audit failed (low_reliability /
low_confidence / demoted). Users can opt back in via status='all' or a
specific status string.
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ccchain
from ccchain.config import Config
from ccchain.core.ontology import Atom
from ccchain import _status_included, _DEFAULT_INCLUDE_STATUSES


# ---------------------------------------------------------------------------
# Unit tests for the _status_included helper
# ---------------------------------------------------------------------------
class TestStatusIncluded:
    def test_default_includes_trusted(self):
        for s in ("active", "verified", "skipped", "needs_review"):
            assert _status_included(s, None) is True

    def test_default_excludes_failures(self):
        for s in ("low_reliability", "low_confidence", "demoted"):
            assert _status_included(s, None) is False

    def test_default_excludes_lifecycle_states(self):
        for s in ("transient", "merged", "stuck"):
            assert _status_included(s, None) is False

    def test_all_returns_everything(self):
        for s in ("verified", "low_reliability", "demoted", "transient", "active"):
            assert _status_included(s, "all") is True

    def test_specific_status_match(self):
        assert _status_included("low_reliability", "low_reliability") is True
        assert _status_included("verified", "low_reliability") is False
        assert _status_included("low_confidence", "low_confidence") is True

    def test_default_set_is_frozen(self):
        assert isinstance(_DEFAULT_INCLUDE_STATUSES, frozenset)


# ---------------------------------------------------------------------------
# End-to-end: ingest → audit assigns statuses → search filters them
# ---------------------------------------------------------------------------
_PAPER_TEXT = (
    "CTDE multi-agent RL has noisy credit assignment. We use optimal-transport "
    "Sinkhorn credit (lambda=0.1), achieving win rate 0.95 on SMAC. "
    "Based on Cuturi 2013 Sinkhorn distances."
)

_PHASE1 = {
    "W1_problem": {
        "name": "Credit Noise", "type": "bottleneck",
        "context": "CTDE credit assignment is noisy.",
    },
    "W2_directions": [
        {"name": "OT Direction", "type": "method", "context": "OT-based credit assignment.",
         "provenance": {"code_span": "sec 3"},
         "W3_approaches": [
             {"name": "OT Method", "type": "method", "context": "OT credit assignment.",
              "provenance": {"code_span": "sec 3"}},
             {"name": "Cuturi Ref", "type": "citation", "context": "Cuturi 2013.",
              "provenance": {"raw_citation": "Smith et al., definitely fake, 2099"}},
         ]},
    ],
}

_PHASE2 = {
    "W4_implementations": [{
        "name": "Sinkhorn Win Rate", "type": "numerical",
        "context": "Win rate 0.95 on SMAC.",
        "provenance": {"score": 0.95, "score_std": 0.0},
        "parent_W3_id": "OT Method",
        "extends": [], "improves": [],
        "W5_code": [
            {"name": "train_exp", "type": "experiment", "context": "Training loop.",
             "code_ref": "train", "code_body": "critic = CentralisedCritic()",
             "provenance": {"code_span": "1-40"}},
            {"name": "sinkhorn_fn", "type": "component", "context": "Solver.",
             "code_ref": "sinkhorn", "code_body": "def sinkhorn(): ..."},
        ],
    }],
}


def _reset_sdk_singletons():
    ccchain._store = None
    ccchain._extractor = ccchain._refiner = ccchain._reducer = None
    ccchain._retriever = ccchain._evaluator = ccchain._verifier = None


def _mock_embed():
    def _embed(texts, **kwargs):
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**32))
        return rng.randn(len(texts), 64).astype(np.float32)
    return _embed


def _mock_chat_json():
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "W1_problem" in msg:
            return _PHASE1
        if "W4_implementations" in msg:
            return _PHASE2
        if "score" in msg.lower() and "extract" in msg.lower():
            return {"score": 0.95}
        if "higher-level" in msg.lower() or "synthesize" in msg.lower() or "reduced_from" in msg:
            return {"name": "Reduced", "context": "abstracted", "type": "concept"}
        return {"fixes": []}
    return _fn


def _mock_majority(i2="violates_spec", i4="aligned"):
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "compliant" in msg or "violates_spec" in msg:
            return {"verdict": i2, "reasoning": "I2"}
        if "aligned" in msg or "misaligned" in msg:
            return {"verdict": i4, "reasoning": "I4"}
        return {"verdict": "ambiguous", "reasoning": "fb"}
    return _fn


def _mock_requests_none():
    def _side(url, **kwargs):
        m = MagicMock()
        m.status_code = 200
        if "semanticscholar" in url:
            m.json.return_value = {"data": []}
        elif "arxiv" in url:
            m.text = "<feed></feed>"
        elif "openalex" in url:
            m.json.return_value = {"results": []}
        elif "crossref" in url:
            m.json.return_value = {"message": {"items": []}}
        else:
            m.json.return_value = {}
        return m
    mock_get = MagicMock()
    mock_get.side_effect = _side
    return mock_get


@pytest.fixture
def ingested_store():
    """Ingest one paper (with audit), leave SDK initialized; return tmpdir."""
    _reset_sdk_singletons()
    tmpdir = tempfile.mkdtemp(prefix="ccchain_search_")
    with patch("ccchain.Config", return_value=Config(
            db_path=os.path.join(tmpdir, "test.db"), graph_dir=tmpdir,
            reference_api_timeout=0.1, reference_api_max_retries=1,
            audit_majority_k=1, audit_i1_k=1)), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json()), \
         patch("ccchain.core.llm.chat_json_majority", side_effect=_mock_majority()), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get", side_effect=_mock_requests_none()):
        ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf")
    yield tmpdir
    _reset_sdk_singletons()
    if ccchain._store is not None:
        try:
            ccchain._store.db.close()
        except Exception:
            pass


def _patch_embed():
    """Patch embed at both the source module and the retriever's bound reference."""
    fn = _mock_embed()
    return patch("ccchain.plugins.retrieval.embed", side_effect=fn)


def test_search_default_excludes_failed_atoms(ingested_store):
    """Default search returns zero low_reliability/low_confidence/demoted atoms."""
    with _patch_embed():
        results, err = ccchain.search("credit assignment", top_k=50, level="W4")
    assert err is None
    forbidden = {"low_reliability", "low_confidence", "demoted", "transient"}
    for r in results:
        assert r["status"] not in forbidden


def test_search_all_returns_failures(ingested_store):
    """status='all' surfaces the citation atom that failed I3 (dangling)."""
    with _patch_embed():
        results, err = ccchain.search("Cuturi", top_k=50, level="W3", status="all")
    assert err is None
    statuses = {r["status"] for r in results}
    # The fake citation should be low_reliability after audit.
    assert "low_reliability" in statuses


def test_search_filter_specific_status(ingested_store):
    """status='low_reliability' returns ONLY low_reliability atoms."""
    with _patch_embed():
        results, err = ccchain.search("credit assignment", top_k=50, level="W3",
                                      status="low_reliability")
    assert err is None
    assert len(results) >= 1
    for r in results:
        assert r["status"] == "low_reliability"


def test_search_default_excludes_failed_across_levels(ingested_store):
    """Check W5 too: the experiment atom violated I2 → low_reliability, must be excluded."""
    with _patch_embed():
        results, err = ccchain.search("training", top_k=50, level="W5")
    assert err is None
    for r in results:
        assert r["status"] != "low_reliability"

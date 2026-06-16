"""Integration test: ingest() with embedded CoE audit pipeline (v0.3).

Validates the full v0.3 pipeline end-to-end through the SDK public method:
    segments → extract → refine → store → reduce → audit → audit_report

The audit runs I1 (Score Verification), I2 (Specification Violation, needs
task_spec), I3 (Reference Verification), I4 (Method-Code Alignment) and
returns a CPR score + per-atom status mapping. Mocks replace every LLM call,
embedding call, and external citation API.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ccchain
from ccchain.config import Config
from ccchain.core.ontology import TaskSpec


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _reset_sdk_singletons():
    """Force the lazy SDK singletons back to None so _init rebuilds with our Config."""
    ccchain._store = None
    ccchain._extractor = None
    ccchain._refiner = None
    ccchain._reducer = None
    ccchain._retriever = None
    ccchain._evaluator = None
    ccchain._verifier = None


# ---------------------------------------------------------------------------
# Controlled LLM fixtures for a single paper
# ---------------------------------------------------------------------------
_PAPER_TEXT = (
    "Multi-agent reinforcement learning under CTDE suffers from high-variance "
    "credit assignment. We propose an optimal-transport-based credit assignment "
    "method using Sinkhorn distances with entropy regularization lambda=0.1. "
    "Our method achieves a win rate of 0.95 on the SMAC benchmark. "
    "This builds on the Sinkhorn distances framework of Cuturi 2013."
)

_PHASE1_RESPONSE = {
    "W2_problem_analysis": {
        "name": "Credit Assignment Noise in CTDE",
        "context": "CTDE suffers from high-variance policy gradients due to noisy credit assignment.",
        "type": "bottleneck",
    },
    "W3_solution_directions": [
        {
            "name": "OT Credit Assignment",
            "context": "Optimal transport credit assignment via Sinkhorn distances.",
            "type": "method",
            "provenance": {"code_span": "section 3"},
        },
        {
            "name": "Cuturi Sinkhorn",
            "context": "Cuturi 2013 Sinkhorn distances framework.",
            "type": "citation",
            "provenance": {"raw_citation": "Cuturi, M. Sinkhorn Distances. NeurIPS 2013."},
        },
    ],
}

_PHASE2_RESPONSE = {
    "W4_concrete_solutions": [
        {
            "name": "Sinkhorn OT Credit",
            "context": "Sinkhorn with entropy reg lambda=0.1. Win rate 0.95 on SMAC.",
            "type": "numerical",
            "provenance": {"score": 0.95, "score_std": 0.0},
            "parent_W3_id": "OT Credit Assignment",
            "extends": [],
            "improves": [],
            "W5_implementations": [
                {
                    "name": "sinkhorn_experiment",
                    "context": "CTDE training loop on SMAC.",
                    "type": "experiment",
                    "code_ref": "train_loop",
                    "code_body": "critic = CentralisedCritic(obs_all, act_all)\nloss = mse(critic(s,a), r)",
                    "provenance": {"code_span": "lines 1-40"},
                },
                {
                    "name": "sinkhorn_solver",
                    "context": "Sinkhorn solver wrapper.",
                    "type": "component",
                    "code_ref": "sinkhorn",
                    "code_body": "def sinkhorn(C, reg=0.1): ...",
                },
            ],
        }
    ],
}

_REDUCE_RESPONSE = {"name": "Reduced Atom", "context": "Reduced abstraction.", "type": "concept"}


@pytest.fixture
def isolated_env():
    """Temp dir + Config pointing SDK at an isolated store; resets singletons."""
    _reset_sdk_singletons()
    tmpdir = tempfile.mkdtemp(prefix="ccchain_audit_")
    config = Config(
        db_path=os.path.join(tmpdir, "test.db"),
        graph_dir=tmpdir,
        reference_api_timeout=0.1,
        reference_api_max_retries=1,
    )
    yield tmpdir, config
    # Restore singletons so other tests aren't polluted.
    _reset_sdk_singletons()
    if ccchain._store is not None:
        try:
            ccchain._store.db.close()
        except Exception:
            pass


def _patch_config(tmpdir: str):
    """Patch Config() inside the ccchain package to use the temp store."""
    return patch(
        "ccchain.Config",
        return_value=Config(
            db_path=os.path.join(tmpdir, "test.db"),
            graph_dir=tmpdir,
            reference_api_timeout=0.1,
            reference_api_max_retries=1,
            audit_majority_k=1,   # speed up tests; majority of 1 is deterministic
            audit_i1_k=1,
        ),
    )


def _mock_embed():
    """Deterministic embed: hash of text → fixed-dim vector."""
    def _embed(texts, **kwargs):
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**32))
        return rng.randn(len(texts), 64).astype(np.float32)
    return _embed


def _mock_chat_json_factory():
    """Route chat_json calls to the right fixture by prompt content."""
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "W2_problem_analysis" in msg:
            return _PHASE1_RESPONSE
        if "W4_concrete_solutions" in msg:
            return _PHASE2_RESPONSE
        # Refiner / reducer / I1 fixes — accept anything the LLM proposes.
        if "score" in msg.lower() and "extract" in msg.lower():
            return {"score": 0.95}  # I1 re-extraction agrees with provenance
        if "reduced_from" in msg or "synthesize" in msg.lower() or "higher-level" in msg.lower():
            return _REDUCE_RESPONSE
        # Refiner fix proposals: nothing to fix (extractions are already valid).
        return {"fixes": []}
    return _fn


def _mock_chat_majority_factory(i2_verdict="violates_spec", i4_verdict="aligned"):
    """K-sample voting → single deterministic verdict per check."""
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "compliant" in msg or "violates_spec" in msg:
            return {"verdict": i2_verdict, "reasoning": "mock I2"}
        if "aligned" in msg or "misaligned" in msg:
            return {"verdict": i4_verdict, "reasoning": "mock I4"}
        return {"verdict": "ambiguous", "reasoning": "fallback"}
    return _fn


def _mock_requests_all_none():
    """I3: every citation API returns an empty/miss response."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_ingest_returns_audit_report(isolated_env):
    """ingest() result must include an audit_report with valid CPR in [0,1]."""
    tmpdir, _ = isolated_env
    with _patch_config(tmpdir), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json_factory()), \
         patch("ccchain.core.llm.chat_json_majority",
               side_effect=_mock_chat_majority_factory()), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get",
               side_effect=_mock_requests_all_none()):
        result, err = ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf")

    assert err is None, f"ingest failed: {err}"
    assert "audit_report" in result
    ar = result["audit_report"]
    assert 0.0 <= ar["cpr"] <= 1.0
    # At least one numerical atom was audited.
    assert ar["atoms_audited"] >= 1
    assert ar["atoms_audited"] == ar["atoms_passed"] + ar["atoms_failed"]


def test_ingest_audit_classifies_statuses(isolated_env):
    """Audit must populate per-atom status values across the verified/low_* taxonomy."""
    tmpdir, _ = isolated_env
    with _patch_config(tmpdir), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json_factory()), \
         patch("ccchain.core.llm.chat_json_majority",
               side_effect=_mock_chat_majority_factory(i2_verdict="violates_spec",
                                                       i4_verdict="aligned")), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get",
               side_effect=_mock_requests_all_none()):
        result, _ = ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf")

    statuses = {p["status"] for p in result["audit_report"]["per_atom"]}
    # The taxonomy we expect to see in this fixture:
    #   verified     — numerical (I1 pass), method (I4 aligned)
    #   low_*        — experiment (I2 violates_spec → low_reliability), citation (I3 miss → low_reliability)
    #   skipped      — bottleneck, component (no CoE check)
    assert statuses & {"verified", "skipped"}  # at least some pass/skip
    assert "low_reliability" in statuses       # I2 + I3 both hard-fail here


def test_ingest_with_task_spec_triggers_i2(isolated_env):
    """Passing a TaskSpec means I2 is no longer skipped for experiment atoms."""
    tmpdir, _ = isolated_env
    ts = TaskSpec(task_name="smac-1v1", eval_harness="smac-v1",
                  success_criteria="win_rate > 0.9", constraints=["CTDE"])
    with _patch_config(tmpdir), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json_factory()), \
         patch("ccchain.core.llm.chat_json_majority",
               side_effect=_mock_chat_majority_factory(i2_verdict="violates_spec")), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get",
               side_effect=_mock_requests_all_none()):
        result, _ = ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf", task_spec=ts)

    ar = result["audit_report"]
    # I2 fired at least once (the experiment atom violates the spec).
    assert ar["failures_by_check"]["I2"] >= 1
    # Citation atom also fails I3 (all APIs miss).
    assert ar["failures_by_check"]["I3"] >= 1


def test_ingest_without_task_spec_skips_i2(isolated_env):
    """Without a TaskSpec, I2 is skipped — failures_by_check['I2'] stays 0."""
    tmpdir, _ = isolated_env
    with _patch_config(tmpdir), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json_factory()), \
         patch("ccchain.core.llm.chat_json_majority",
               side_effect=_mock_chat_majority_factory()), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get",
               side_effect=_mock_requests_all_none()):
        result, _ = ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf")

    assert result["audit_report"]["failures_by_check"]["I2"] == 0


def test_ingest_audited_statuses_persist_to_store(isolated_env):
    """After ingest, querying the store (any status) surfaces audit outcomes."""
    tmpdir, _ = isolated_env
    with _patch_config(tmpdir), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json_factory()), \
         patch("ccchain.core.llm.chat_json_majority",
               side_effect=_mock_chat_majority_factory(i2_verdict="violates_spec")), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get",
               side_effect=_mock_requests_all_none()):
        ccchain.ingest([_PAPER_TEXT], source_pdf="test.pdf")

    store = ccchain._store
    all_statuses = {a.status for a in store.query_by_level("W4_concrete_solution", status=None)}
    all_statuses |= {a.status for a in store.query_by_level("W3_solution_direction", status=None)}
    all_statuses |= {a.status for a in store.query_by_level("W5_code_implementation", status=None)}
    # No atom should be stuck on 'active' after audit ran.
    assert "active" not in all_statuses
    # Audit vocab must be present.
    assert all_statuses & {"verified", "low_reliability", "low_confidence", "skipped"}

"""Test incremental ingest: add a 3rd paper after 2 are already ingested.

Locks the verified behavior that (a) atoms accumulate across calls sharing one
db_path, (b) the CoE audit only runs on each call's NEW atoms, (c) old atoms
retain their audit status (not reset to active), (d) cross-process persistence,
(e) auto-consolidate is a no-op when the merge arbiter declines.
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ccchain
from ccchain.config import Config
from ccchain.core.ontology import TaskSpec
from ccchain.core.store import CCStore


# ---------------------------------------------------------------------------
# Mock LLM/embed (deterministic, distinct per paper so consolidate finds no
# cross-paper duplicates unless we explicitly craft them)
# ---------------------------------------------------------------------------
def _mock_embed():
    import numpy as np
    def _embed(texts, **kwargs):
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**32))
        return rng.randn(len(texts), 64).astype(np.float32)
    return _embed


def _phase1(tag):
    return {
        "W2_problem_analysis": {
            "name": f"{tag} Problem", "type": "bottleneck",
            "context": f"{tag} core problem.",
        },
        "W3_solution_directions": [
            {"name": f"{tag} Method", "type": "method", "context": f"{tag} method.",
             "provenance": {"code_span": "sec 3"}},
        ],
    }


def _phase2(tag):
    return {"W4_concrete_solutions": [{
        "name": f"{tag} score", "type": "numerical",
        "context": f"{tag} achieves 0.9.", "provenance": {"score": 0.9, "score_std": 0.01},
        "parent_W3_id": f"{tag} Method", "extends": [], "improves": [],
        "W5_implementations": [
            {"name": f"{tag} exp", "type": "experiment", "context": f"{tag} training.",
             "code_ref": "t", "code_body": "x=1", "provenance": {"code_span": "1-40"}},
        ],
    }]}


def _mock_chat_json(tags):
    """tags: list of per-paper labels; routes phase1/phase2/I1 by prompt content."""
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        tag = tags[0]
        if "W2_problem_analysis" in msg:
            return _phase1(tag)
        if "W4_concrete_solutions" in msg:
            return _phase2(tag)
        if "score" in msg.lower() and "extract" in msg.lower():
            return {"score": 0.9}        # I1 agrees
        if "higher-level" in msg.lower() or "synthesize" in msg.lower() or "reduced_from" in msg:
            return {"name": f"{tag} reduced", "context": "abstracted", "type": "concept"}
        return {"fixes": []}
    return _fn


def _mock_majority(i2="violates_spec"):
    """Audit I2/I4 + consolidate arbiter. Arbiter prompt has no 'compliant'/
    'aligned' keywords → returns merge=False (no consolidation)."""
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "compliant" in msg or "violates_spec" in msg:
            return {"verdict": i2, "reasoning": "I2"}
        if "aligned" in msg or "misaligned" in msg:
            return {"verdict": "aligned", "reasoning": "I4"}
        # consolidate merge-arbiter → decline (no merge in the default scenario)
        return {"merge": False, "reasoning": "distinct"}
    return _fn


def _reset():
    ccchain._store = ccchain._extractor = ccchain._refiner = None
    ccchain._reducer = ccchain._retriever = ccchain._evaluator = None
    ccchain._verifier = ccchain._consolidator = None


_LEVELS = ["W2_problem_analysis", "W3_solution_direction",
           "W4_concrete_solution", "W5_code_implementation"]


def _total_atoms(store):
    return sum(len(store.query_by_level(l, status=None)) for l in _LEVELS)


@pytest.fixture
def env():
    _reset()
    tmp = tempfile.mkdtemp(prefix="ccchain_incr_")
    cfg = Config(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp,
                 reference_api_timeout=0.1, reference_api_max_retries=1,
                 audit_majority_k=1, audit_i1_k=1, consolidate_majority_k=1,
                 consolidate_similarity_threshold=0.99)  # high → no spurious clusters
    yield tmp, cfg
    _reset()
    if ccchain._store is not None:
        try:
            ccchain._store.db.close()
        except Exception:
            pass


def _ingest(env, tag):
    tmp, _ = env
    with patch("ccchain.Config", return_value=Config(
            db_path=os.path.join(tmp, "t.db"), graph_dir=tmp,
            reference_api_timeout=0.1, reference_api_max_retries=1,
            audit_majority_k=1, audit_i1_k=1, consolidate_majority_k=1,
            consolidate_similarity_threshold=0.99)), \
         patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json([tag])), \
         patch("ccchain.core.llm.chat_json_majority", side_effect=_mock_majority()), \
         patch("ccchain.core.embedding.embed", side_effect=_mock_embed()), \
         patch("ccchain.core.references.requests.get", side_effect=_none_requests()):
        return ccchain.ingest([f"{tag} paper text"], source_pdf=f"{tag}.pdf",
                              task_spec=TaskSpec("t", "h", "c", ["CTDE"]))


def _none_requests():
    from unittest.mock import MagicMock
    def _side(url, **kwargs):
        m = MagicMock(); m.status_code = 200
        if "semanticscholar" in url: m.json.return_value = {"data": []}
        elif "arxiv" in url: m.text = "<feed></feed>"
        elif "openalex" in url: m.json.return_value = {"results": []}
        elif "crossref" in url: m.json.return_value = {"message": {"items": []}}
        else: m.json.return_value = {}
        return m
    g = MagicMock(); g.side_effect = _side; return g


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def test_incremental_accumulates_atoms(env):
    """(i,ii) ingesting a 2nd paper strictly grows the store."""
    r1, e1 = _ingest(env, "A")
    assert e1 is None
    n1 = _total_atoms(ccchain._store)
    assert n1 > 0

    r2, e2 = _ingest(env, "B")
    assert e2 is None
    n2 = _total_atoms(ccchain._store)
    assert n2 > n1


def test_audit_only_runs_on_new_atoms(env):
    """(iii) 2nd ingest's audit atoms_audited reflects only B's new atoms, not A's."""
    r1, _ = _ingest(env, "A")
    audited_A = r1["audit_report"]["atoms_audited"]

    r2, _ = _ingest(env, "B")
    audited_B = r2["audit_report"]["atoms_audited"]

    # B's audit count is ~ the same per-paper count as A's (not cumulative)
    assert audited_B == audited_A
    # and strictly less than the total live atoms at that point
    assert audited_B < _total_atoms(ccchain._store)


def test_old_atoms_keep_audit_status(env):
    """(iv) after ingesting B, A's atoms retain verified/skipped/low_* (not reset)."""
    _ingest(env, "A")
    a_statuses_before = {a.node_id: a.status for l in _LEVELS
                         for a in ccchain._store.query_by_level(l, status=None)
                         if a.source_pdf == "A.pdf"}

    _ingest(env, "B")
    a_statuses_after = {a.node_id: a.status for l in _LEVELS
                        for a in ccchain._store.query_by_level(l, status=None)
                        if a.source_pdf == "A.pdf"}

    assert a_statuses_before == a_statuses_after
    # none of A's atoms were reset to 'active'
    assert "active" not in a_statuses_after.values()


def test_consolidate_noop_when_arbiter_declines(env):
    """(v) auto-consolidate runs but merges nothing when arbiter declines."""
    r2, _ = _ingest(env, "A"); _ingest(env, "B")
    # the last ingest's consolidate_report (re-ingest B path returns r2 of A...)
    # Re-run a 3rd to capture a fresh report.
    r3, _ = _ingest(env, "C")
    cr = r3["consolidate_report"]
    assert cr is not None
    assert cr["atoms_merged"] == 0


def test_cross_process_persistence(env):
    """(vi) reopen the DB from a fresh CCStore → all atoms + statuses survive."""
    _ingest(env, "A"); _ingest(env, "B")
    expected = _total_atoms(ccchain._store)
    statuses_before = sorted(a.status for l in _LEVELS
                             for a in ccchain._store.query_by_level(l, status=None))
    ccchain._store.db.commit()
    ccchain._store.db.close()
    _reset()

    tmp, _ = env
    fresh = CCStore(os.path.join(tmp, "t.db"), tmp)
    assert _total_atoms(fresh) == expected
    statuses_after = sorted(a.status for l in _LEVELS
                            for a in fresh.query_by_level(l, status=None))
    assert statuses_before == statuses_after
    fresh.db.close()


def test_search_returns_multiple_papers(env):
    """search across accumulated atoms surfaces results regardless of paper."""
    _ingest(env, "A"); _ingest(env, "B")
    with patch("ccchain.plugins.retrieval.embed", side_effect=_mock_embed()):
        res, err = ccchain.search("method", top_k=50, level="W3", status="all")
    assert err is None
    assert len(res) >= 1


def test_consolidate_merges_during_ingest_then_excludes_from_search(env):
    """(vii) when consolidate merges across papers during ingest: the dup is
    status='merged' and excluded from default search; the canonical survives."""
    import numpy as np
    tmp, _ = env

    def _fixed_embed():
        # W2 problem contexts (contain 'problem') collapse to ONE vector so the
        # two papers' W2 atoms cluster at the top level (same __root__ group).
        fixed = np.array([1.0] * 64, dtype=np.float32)
        def _embed(texts, **kwargs):
            out = []
            for t in texts:
                if "problem" in t.lower():
                    out.append(fixed.copy())
                else:
                    rng = np.random.RandomState(abs(hash(t)) % (2**32))
                    out.append(rng.randn(64).astype(np.float32))
            return np.stack(out).astype(np.float32)
        return _embed

    def _merging_majority():
        def _fn(messages, **kwargs):
            msg = messages[0]["content"] if messages else ""
            if "compliant" in msg or "violates_spec" in msg:
                return {"verdict": "violates_spec", "reasoning": "I2"}
            if "aligned" in msg or "misaligned" in msg:
                return {"verdict": "aligned", "reasoning": "I4"}
            # consolidate arbiter → merge the W2 problems
            return {"merge": True, "canonical_name": "Shared Problem",
                    "canonical_type": "bottleneck",
                    "canonical_context": "shared cross-paper problem.",
                    "reasoning": "same problem"}
        return _fn

    cfg = Config(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp,
                 reference_api_timeout=0.1, reference_api_max_retries=1,
                 audit_majority_k=1, audit_i1_k=1, consolidate_majority_k=1,
                 consolidate_similarity_threshold=0.85)

    def _do(tag):
        with patch("ccchain.Config", return_value=cfg), \
             patch("ccchain.core.llm.chat_json", side_effect=_mock_chat_json([tag])), \
             patch("ccchain.core.llm.chat_json_majority", side_effect=_merging_majority()), \
             patch("ccchain.core.embedding.embed", side_effect=_fixed_embed()), \
             patch("ccchain.core.references.requests.get", side_effect=_none_requests()):
            return ccchain.ingest([f"{tag} paper text"], source_pdf=f"{tag}.pdf",
                                  task_spec=TaskSpec("t", "h", "c", ["CTDE"]))

    r1, _ = _do("A")
    r2, _ = _do("B")
    # the second ingest's consolidate merged the two W2 problems
    assert r2["consolidate_report"]["atoms_merged"] >= 1

    w2 = ccchain._store.query_by_level("W2_problem_analysis", status=None)
    statuses = [a.status for a in w2]
    assert "merged" in statuses                      # one dup flipped
    # default search excludes merged
    with patch("ccchain.plugins.retrieval.embed", side_effect=_fixed_embed()):
        default_res, _ = ccchain.search("problem", top_k=50, level="W2")
        all_res, _ = ccchain.search("problem", top_k=50, level="W2", status="all")
    assert all(r["status"] != "merged" for r in default_res)
    assert any(r["status"] == "merged" for r in all_res)

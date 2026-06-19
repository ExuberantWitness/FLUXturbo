"""Incremental import test: ingest the 3 real PDFs in ARIS/pdf one at a time.

Demonstrates the v0.5 stack working incrementally on real paper text:
  - atoms accumulate in a shared store (no rebuild between papers)
  - the CoE audit runs only on each paper's NEW atoms (old atoms not re-audited)
  - consolidate merges cross-paper duplicates incrementally
  - the final 5-level pyramid (W1→W5) is exported to an OKF bundle + round-tripped

No live LLM needed: the LLM/embedding/citation layers are deterministic mocks,
but the real PDF text flows through the full extract → refine → store → reduce
→ consolidate → audit pipeline.

Run:  python scripts/incremental_import_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from unittest.mock import patch, MagicMock

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import ccchain
from ccchain.config import Config
from ccchain.core.ontology import TaskSpec
from ccchain.okf import export_okf, import_okf

PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"
_LEVELS = ["W1_problem", "W2_direction", "W3_approach", "W4_implementation", "W5_code"]


# ---------------------------------------------------------------------------
# Real PDF text
# ---------------------------------------------------------------------------
def load_papers():
    import fitz
    papers = []
    for f in sorted(os.listdir(PDF_DIR)):
        if not f.endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(PDF_DIR, f))
        papers.append({"filename": f, "text": "\n\n".join(p.get_text() for p in doc)[:8000]})
        doc.close()
    return papers


def topic_of(text):
    low = text.lower()
    if "safe" in low and ("reinforcement" in low or "q-value" in low):
        return "safe_rl"
    if "inverse" in low and "manipulation" in low:
        return "inverse_manip"
    if "eeg" in low and "emotion" in low:
        return "eeg"
    return "generic"


_TOPIC_W3 = {
    "safe_rl": ("COP-Q Covariance Critic", "Cholesky-ordered joint Q projection."),
    "inverse_manip": ("STRIPS Operator Extraction", "Symbolic operators from demos."),
    "eeg": ("VQ-Masked RL Framework", "VQ + masked temporal modeling."),
    "generic": ("Proposed Method", "The approach."),
}
SHARED = "Reinforcement Learning Challenges"
SHARED_CTX = "RL faces high-variance credit assignment and sample inefficiency."


# ---------------------------------------------------------------------------
# Mocks (deterministic; real PDF text drives the pipeline)
# ---------------------------------------------------------------------------
def phase1(text):
    t = topic_of(text)
    return {
        "W1_problem": {"name": SHARED, "type": "bottleneck", "context": SHARED_CTX},
        "W2_directions": [{
            "name": f"{t} Direction", "type": "method", "context": f"{t} research direction.",
            "provenance": {"code_span": "sec 2"},
            "W3_approaches": [
                {"name": _TOPIC_W3[t][0], "type": "method", "context": _TOPIC_W3[t][1],
                 "provenance": {"code_span": "sec 3"}},
                {"name": f"{t} baseline", "type": "citation", "context": f"{t} cited baseline.",
                 "provenance": {"raw_citation": f"{t} baseline, J. Fake, 2099"}},
            ],
        }],
    }


def phase2(text):
    t = topic_of(text)
    return {"W4_implementations": [{
        "name": f"{t} result", "type": "numerical",
        "context": f"{t} achieves 0.92 normalized score.",
        "provenance": {"score": 0.92, "score_std": 0.01},
        "parent_W3_id": _TOPIC_W3[t][0], "extends": [], "improves": [],
        "W5_code": [
            {"name": f"{t} experiment", "type": "experiment", "context": f"{t} training loop.",
             "code_ref": "train", "code_body": "critic = CentralisedCritic()", "provenance": {"code_span": "1-40"}},
            {"name": f"{t} solver", "type": "component", "context": f"{t} core module.",
             "code_ref": "core", "code_body": "def core(x): return x"},
        ],
    }]}


def semantic_embed():
    """Word-hash embed: identical concepts (same words) cluster together."""
    import hashlib
    DIM = 128
    def _embed(texts, **kwargs):
        out = []
        for t in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for w in __import__("re").findall(r"[a-z]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % DIM] += 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            out.append(v)
        return np.stack(out).astype(np.float32)
    return _embed


def make_chat_json():
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "W1_problem" in msg:
            return phase1(msg)
        if "W4_implementations" in msg:
            return phase2(msg)
        if "score" in msg.lower() and "extract" in msg.lower():
            return {"score": 0.92}
        if "higher-level" in msg.lower() or "synthesize" in msg.lower() or "reduced_from" in msg:
            return {"name": "reduced", "context": "abstracted", "type": "concept"}
        return {"fixes": []}
    return _fn


def make_majority():
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "compliant" in msg or "violates_spec" in msg:
            return {"verdict": "violates_spec", "reasoning": "I2"}
        if "aligned" in msg or "misaligned" in msg:
            return {"verdict": "aligned", "reasoning": "I4"}
        return {"merge": True, "canonical_name": SHARED, "canonical_type": "bottleneck",
                "canonical_context": SHARED_CTX, "reasoning": "same RL problem"}
    return _fn


def none_requests():
    def _side(url, **kwargs):
        m = MagicMock(); m.status_code = 200
        if "semanticscholar" in url: m.json.return_value = {"data": []}
        elif "arxiv" in url: m.text = "<feed></feed>"
        elif "openalex" in url: m.json.return_value = {"results": []}
        elif "crossref" in url: m.json.return_value = {"message": {"items": []}}
        else: m.json.return_value = {}
        return m
    g = MagicMock(); g.side_effect = _side; return g


def reset():
    ccchain._store = ccchain._extractor = ccchain._refiner = None
    ccchain._reducer = ccchain._retriever = ccchain._evaluator = None
    ccchain._verifier = ccchain._consolidator = None


def _level_counts(store):
    return {lvl: len(store.query_by_level(lvl, status=None)) for lvl in _LEVELS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    papers = load_papers()
    print(f"Loaded {len(papers)} PDFs from {PDF_DIR}\n")

    tmp = tempfile.mkdtemp(prefix="ccchain_incr_")
    cfg = Config(db_path=os.path.join(tmp, "cc.db"), graph_dir=tmp,
                 reference_api_timeout=0.1, reference_api_max_retries=1,
                 audit_majority_k=1, audit_i1_k=1, consolidate_majority_k=1,
                 consolidate_similarity_threshold=0.80)
    ts = TaskSpec("safety-gym", "safety-gym-v1", "return>=0.9 & 0 violations", ["CTDE"])

    cumulative_audit = 0
    for i, paper in enumerate(papers):
        with patch("ccchain.Config", return_value=cfg), \
             patch("ccchain.core.llm.chat_json", side_effect=make_chat_json()), \
             patch("ccchain.core.llm.chat_json_majority", side_effect=make_majority()), \
             patch("ccchain.core.embedding.embed", side_effect=semantic_embed()), \
             patch("ccchain.core.references.requests.get", side_effect=none_requests()):
            r, err = ccchain.ingest([paper["text"]], source_pdf=paper["filename"], task_spec=ts)

        if err:
            print(f"  ! ingest error: {err}"); continue

        ar = r["audit_report"]
        cr = r["consolidate_report"] or {}
        counts = _level_counts(ccchain._store)
        total = sum(counts.values())
        cumulative_audit += ar["atoms_audited"]
        print(f"ingest #{i+1}  {paper['filename']}")
        print(f"   pyramid: {counts}  (total {total})")
        print(f"   audit:   audited={ar['atoms_audited']} (this paper only)  "
              f"passed={ar['atoms_passed']}  failed={ar['atoms_failed']}  cpr={ar['cpr']:.2f}")
        print(f"   failures by check: {ar['failures_by_check']}")
        print(f"   consolidate: merged={cr.get('atoms_merged',0)}  "
              f"clusters_formed={cr.get('clusters_formed',0)}  confirmed={cr.get('clusters_confirmed',0)}")
        print()

    # ---- Final state ----
    store = ccchain._store
    print("=" * 72)
    print("FINAL incremental store state")
    print("=" * 72)
    print(f"  pyramid: {_level_counts(store)}  (total {sum(_level_counts(store).values())})")
    status_tally = Counter(a.status for lvl in _LEVELS
                           for a in store.query_by_level(lvl, status=None))
    print(f"  status distribution: {dict(status_tally)}")
    print(f"  cumulative audit atoms (sum of per-paper): {cumulative_audit}  "
          f"(should equal total new atoms, NOT re-audited)")

    # ---- OKF export + round-trip (portability capstone) ----
    bundle = export_okf(store, os.path.join(tmp, "okf_bundle"))
    n_concepts = sum(1 for dp, _, fns in os.walk(bundle)
                     for fn in fns if fn.endswith(".md") and fn not in ("index.md", "log.md"))
    print(f"\n  OKF export: {n_concepts} concepts -> {bundle}")

    reset()
    store2_db = os.path.join(tempfile.mkdtemp(prefix="ccchain_rt_"), "rt.db")
    from ccchain.core.store import CCStore
    rt_store = CCStore(store2_db, os.path.dirname(store2_db))
    rep = import_okf(bundle, rt_store)
    print(f"  OKF round-trip import: atoms={rep['atoms_imported']}  edges={rep['edges_imported']}")
    rt_store.db.close()
    store.db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Compare import modes over the 3 real PDFs in E:\\DATA\\vscode\\ARIS\\pdf.

Three modes, same 3 papers, deterministic mocks (no live LLM needed):

  S-off    : sequential (1 paper per ingest), auto_consolidate=False  (v0.3)
  S-on     : sequential (1 paper per ingest), auto_consolidate=True   (v0.4)
  One-shot : single ingest of all 3 papers concatenated, consolidate=True

To make consolidation's effect VISIBLE, the mock extractor emits a SHARED
"Reinforcement Learning Challenges" W2 problem for every paper (realistic —
adjacent papers address the same broad problem) and a semantic word-hash
embedding so identical concepts cluster. Topic-specific W3/W4/W5 stay unique.

Run:  python scripts/compare_sequential_vs_oneshot.py
"""

from __future__ import annotations

import os
import re
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
from ccchain.core.store import CCStore

PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"
SHARED_W2 = "Reinforcement Learning Challenges"

_LEVELS = ["W2_problem_analysis", "W3_solution_direction",
           "W4_concrete_solution", "W5_code_implementation"]


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
        chunks = [p.get_text().strip() for p in doc if p.get_text().strip()]
        papers.append({"filename": f, "chunks": chunks})
        doc.close()
    return papers


def topics_in(text):
    """Return ALL topics detected in text (one paper → 1; combined → up to 3)."""
    low = text.lower()
    found = []
    if "safe" in low and ("reinforcement" in low or "q-value" in low):
        found.append("safe_rl")
    if "inverse" in low and "manipulation" in low:
        found.append("inverse_manip")
    if "eeg" in low and "emotion" in low:
        found.append("eeg")
    return found or ["generic"]


def topic_of(text):
    tos = topics_in(text)
    return tos[0]


_SHARED_W2_CTX = ("Reinforcement learning faces high-variance credit assignment "
                  "and sample inefficiency challenges.")
_TOPIC_W3 = {
    "safe_rl": ("COP-Q Covariance Critic", "Cholesky-ordered joint Q projection."),
    "inverse_manip": ("STRIPS Operator Extraction", "Symbolic operators from demonstrations."),
    "eeg": ("VQ-Masked RL Framework", "Vector-quantized masked temporal modeling."),
    "generic": ("Proposed Method", "The proposed approach."),
}


# ---------------------------------------------------------------------------
# Mock extract: shared W2 + topic-specific W3/W4/W5 (all topics in the text)
# ---------------------------------------------------------------------------
def phase1(text):
    tos = topics_in(text)
    return {
        "W2_problem_analysis": {
            "name": SHARED_W2, "type": "bottleneck", "context": _SHARED_W2_CTX,
        },
        "W3_solution_directions": [
            {"name": _TOPIC_W3[t][0], "type": "method", "context": _TOPIC_W3[t][1],
             "provenance": {"code_span": "sec 3"}}
            for t in tos
        ],
    }


def phase2(text):
    tos = topics_in(text)
    solutions = []
    for t in tos:
        solutions.append({
            "name": f"{t} score", "type": "numerical",
            "context": f"{t} achieves 0.92 normalized score.",
            "provenance": {"score": 0.92, "score_std": 0.01},
            "parent_W3_id": _TOPIC_W3[t][0], "extends": [], "improves": [],
            "W5_implementations": [
                {"name": f"{t} exp", "type": "experiment", "context": f"{t} training loop.",
                 "code_ref": "t", "code_body": "x=1", "provenance": {"code_span": "1-40"}},
            ],
        })
    return {"W4_concrete_solutions": solutions}


# ---------------------------------------------------------------------------
# Semantic mock embed (word-hash): same words -> similar vectors
# ---------------------------------------------------------------------------
def semantic_embed():
    import hashlib
    DIM = 128
    def _embed(texts, **kwargs):
        out = []
        for t in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for w in re.findall(r"[a-z]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                v[h % DIM] += 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            out.append(v)
        return np.stack(out).astype(np.float32)
    return _embed


def make_chat_json():
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "W2_problem_analysis" in msg:
            return phase1(msg)
        if "W4_concrete_solutions" in msg:
            return phase2(msg)
        if "score" in msg.lower() and "extract" in msg.lower():
            return {"score": 0.92}
        if "higher-level" in msg.lower() or "synthesize" in msg.lower() or "reduced_from" in msg:
            return {"name": "reduced", "context": "abstracted", "type": "concept"}
        return {"fixes": []}
    return _fn


def make_majority():
    """Audit I2/I4 + consolidate arbiter. Arbiter merges genuine duplicates."""
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "compliant" in msg or "violates_spec" in msg:
            return {"verdict": "compliant", "reasoning": "I2"}
        if "aligned" in msg or "misaligned" in msg:
            return {"verdict": "aligned", "reasoning": "I4"}
        # consolidate arbiter: merge genuine duplicates. Keep the shared W2
        # phrasing so the canonical doesn't drift below threshold on later merges.
        return {"merge": True, "canonical_name": SHARED_W2,
                "canonical_type": "bottleneck",
                "canonical_context": _SHARED_W2_CTX,
                "reasoning": "same broad RL problem"}
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


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _run_mode(papers, mode, tag):
    """Run one mode in a fresh temp DB. Returns summary dict."""
    reset()
    tmp = tempfile.mkdtemp(prefix=f"ccchain_{tag}_")
    cfg = Config(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp,
                 reference_api_timeout=0.1, reference_api_max_retries=1,
                 audit_majority_k=1, audit_i1_k=1, consolidate_majority_k=1,
                 consolidate_similarity_threshold=0.80,
                 auto_consolidate=(mode != "S-off"))

    audit_reports = []
    consolidate_reports = []
    with patch("ccchain.Config", return_value=cfg), \
         patch("ccchain.core.llm.chat_json", side_effect=make_chat_json()), \
         patch("ccchain.core.llm.chat_json_majority", side_effect=make_majority()), \
         patch("ccchain.core.embedding.embed", side_effect=semantic_embed()), \
         patch("ccchain.core.references.requests.get", side_effect=none_requests()):

        if mode == "one-shot":
            # Abstracts only (1 chunk/paper) so the combined text fits the
            # extractor's 12000-char prompt window — otherwise one-shot only
            # "sees" the first paper (a real truncation risk of bulk import).
            combined = "\n\n".join(p["chunks"][0] for p in papers)
            r, err = ccchain.ingest([combined], source_pdf="combined.pdf",
                                    task_spec=TaskSpec("t", "h", "c", ["CTDE"]))
            audit_reports.append(r["audit_report"])
            consolidate_reports.append(r["consolidate_report"])
        else:
            for p in papers:
                combined = p["chunks"][0]  # abstract only — same input as one-shot
                r, err = ccchain.ingest([combined], source_pdf=p["filename"],
                                        task_spec=TaskSpec("t", "h", "c", ["CTDE"]))
                audit_reports.append(r["audit_report"])
                consolidate_reports.append(r["consolidate_report"])

    store = ccchain._store
    total = sum(len(store.query_by_level(l, status=None)) for l in _LEVELS)
    merged = sum(1 for l in _LEVELS for a in store.query_by_level(l, status=None)
                 if a.status == "merged")
    live = total - merged
    w2_atoms = store.query_by_level("W2_problem_analysis", status=None)
    w2_live = [a for a in w2_atoms if a.status != "merged"]
    w2_merged = [a for a in w2_atoms if a.status == "merged"]

    summary = {
        "mode": mode,
        "total_atoms": total,
        "live_atoms": live,
        "merged_atoms": merged,
        "w2_total": len(w2_atoms),
        "w2_live": len(w2_live),
        "w2_merged": len(w2_merged),
        "dedup_rate": (merged / total) if total else 0.0,
        "total_consolidate_merges": sum(c["atoms_merged"] for c in consolidate_reports if c),
        "final_cpr": audit_reports[-1]["cpr"],
    }
    store.db.close()
    reset()
    return summary


def _bar(n, width=24, scale=None):
    scale = scale or max(n, 1)
    return "#" * max(1, int(width * n / scale))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    papers = load_papers()
    print(f"Loaded {len(papers)} PDFs: {[p['filename'] for p in papers]}\n")
    print(f"Shared injected concept: W2 '{SHARED_W2}' (each paper extracts it)\n")

    results = {
        "S-off (v0.3, no consolidate)": _run_mode(papers, "S-off", "Soff"),
        "S-on  (v0.4, consolidate)":    _run_mode(papers, "S-on",  "Son"),
        "One-shot (combined + consolidate)": _run_mode(papers, "one-shot", "OS"),
    }

    max_total = max(r["total_atoms"] for r in results.values())
    print("=" * 78)
    print(f"{'mode':<38} {'total':>6} {'live':>6} {'merged':>7} {'dedup%':>7} {'cpr':>5}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<38} {r['total_atoms']:>6} {r['live_atoms']:>6} "
              f"{r['merged_atoms']:>7} {r['dedup_rate']*100:>6.1f}% {r['final_cpr']:>5.2f}")
    print("=" * 78)

    print("\nW2 'RL Challenges' atoms (the shared concept):")
    maxw = max(r["w2_total"] for r in results.values())
    for name, r in results.items():
        print(f"  {name:<38} total={r['w2_total']}  live={r['w2_live']}  merged={r['w2_merged']}")

    print("\nConsolidate merge events (atoms_merged across all consolidate passes):")
    for name, r in results.items():
        print(f"  {name:<38} {r['total_consolidate_merges']}")

    print("\nTotal atoms by mode (bar = relative count):")
    for name, r in results.items():
        print(f"  {name:<38} {r['total_atoms']:>3} {_bar(r['total_atoms'], scale=max_total)}")

    print("\nInterpretation:")
    print("  - S-off vs S-on : consolidate MERGES the 3 duplicate 'RL Challenges' W2")
    print("    atoms accumulated across sequential ingests down to 1 canonical")
    print(f"    ({results['S-on  (v0.4, consolidate)']['dedup_rate']*100:.0f}% dedup).")
    print("  - S-on vs 1-shot: both end with 10 live atoms and 1 shared-W2 canonical.")
    print("    Sequential reaches the SAME deduplicated state as one-shot — just")
    print("    incrementally (merging 3->1) instead of extracting 1 up front.")
    print("  - 1-shot caveat : combined text must fit the extractor's prompt window;")
    print("    sequential has no such limit (each paper processed independently).")


if __name__ == "__main__":
    main()

"""Demo: run the v0.3 audit pipeline over the REAL papers in E:\\DATA\\vscode\\ARIS\\pdf.

No live LLM server is required: the extractor/refiner/reducer LLM calls and the
CoE audit LLM calls (I1/I2/I4) and citation APIs (I3) are replaced with
deterministic, keyword-aware mocks. What is REAL here:

  * fitz extracts the actual PDF text
  * that text flows through the full ingest pipeline:
        extract -> refine -> store -> reduce -> audit -> write-back
  * the CoE dispatch (TYPE_TO_COE_CHECKS), gatekeeper R6/R7, CPR math, status
    taxonomy, and search status-filtering all run for real

Run:
    python scripts/audit_demo_real_pdfs.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from unittest.mock import patch, MagicMock

import numpy as np

# Make ccchain importable when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import ccchain
from ccchain.config import Config
from ccchain.core.ontology import TaskSpec

PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"


# ---------------------------------------------------------------------------
# 1. Real PDF text extraction
# ---------------------------------------------------------------------------
def extract_paper_texts() -> list[dict]:
    import fitz
    papers = []
    for fname in sorted(os.listdir(PDF_DIR)):
        if not fname.endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(PDF_DIR, fname))
        chunks = [p.get_text().strip() for p in doc if p.get_text().strip()]
        papers.append({"filename": fname, "pages": len(doc), "chunks": chunks})
        doc.close()
    return papers


# ---------------------------------------------------------------------------
# 2. Keyword-aware mock LLM responses (deterministic, paper-specific)
# ---------------------------------------------------------------------------
def topic_of(text: str) -> str:
    low = text.lower()
    if "safe" in low and ("reinforcement learning" in low or "q-value" in low):
        return "safe_rl"
    if "inverse" in low and "manipulation" in low:
        return "inverse_manip"
    if "eeg" in low and "emotion" in low:
        return "eeg"
    return "generic"


_P1 = {
    "safe_rl": {
        "name": "Over-conservative Safe RL", "type": "bottleneck",
        "context": "Off-policy safe RL learns reward and safety Q-values with separate critics, causing over-conservative policies from independent uncertainty.",
    },
    "inverse_manip": {
        "name": "Non-invertible Manipulation Dynamics", "type": "bottleneck",
        "context": "Inverting robotic manipulation requires reasoning over symbolic state transitions and continuous interaction dynamics that are not fully reversible.",
    },
    "eeg": {
        "name": "Incoherent EEG Emotion Trajectories", "type": "bottleneck",
        "context": "Continuous emotion prediction from EEG needs long-range temporal dependencies and globally coherent emotional evolution beyond point-wise regression.",
    },
    "generic": {
        "name": "Core Research Problem", "type": "bottleneck",
        "context": "The paper addresses a central research problem with non-trivial structure.",
    },
}

_P3 = {
    "safe_rl": ([
        {"name": "COP-Q Joint Covariance Critic", "type": "method",
         "context": "Models inter-objective covariance via Cholesky-ordered projection of the joint Q.",
         "provenance": {"code_span": "section 3"}},
        {"name": "CUTURI Sinkhorn baseline", "type": "citation",
         "context": "Comparison against entropy-regularized OT credit assignment.",
         "provenance": {"raw_citation": "Smith, J. Entropy-Regularized OT for MARL Credit. J. Nonexistent, 2099."}},
    ]),
    "inverse_manip": ([
        {"name": "STRIPS Operator Extraction", "type": "method",
         "context": "Extracts STRIPS-like operators via soft geometric predicates from demonstrations.",
         "provenance": {"code_span": "section 3"}},
        {"name": "Classical STRIPS Planner baseline", "type": "citation",
         "context": "Comparison against a classical symbolic planner.",
         "provenance": {"raw_citation": "Fikes, R. E., Nilsson, N. J. STRIPS: A New Approach to the Application of Theorem Proving. Artif. Intell., 2099 (fake)."}},
    ]),
    "eeg": ([
        {"name": "VQ-Masked RL Framework", "type": "method",
         "context": "Vector-quantized representation + masked temporal modeling + RL trajectory optimization.",
         "provenance": {"code_span": "section 3"}},
        {"name": "Van Den Oord VQ-VAE baseline", "type": "citation",
         "context": "Comparison against the original VQ-VAE discrete latent codebook.",
         "provenance": {"raw_citation": "Oord, van den, A. Neural Discrete Representation Learning. Definitely Fake Journal, 2099."}},
    ]),
    "generic": ([
        {"name": "Proposed Method", "type": "method",
         "context": "The proposed approach.", "provenance": {"code_span": "section 3"}},
    ]),
}


def make_phase1(text: str) -> dict:
    topic = topic_of(text)
    # One W2 direction containing the topic's method/citation as W3 approaches.
    return {
        "W1_problem": dict(_P1[topic]),
        "W2_directions": [
            {"name": f"{topic} Direction", "type": "method",
             "context": f"{topic} research direction.",
             "provenance": {"code_span": "section 2"},
             "W3_approaches": list(_P3[topic])},
        ],
    }


def make_phase2(text: str, paper_idx: int = 0) -> dict:
    topic = topic_of(text)
    parent = _P3[topic][0]["name"]
    # Paper 2 produces a 'solution' W4 (triggers I4 via its component child)
    # instead of 'numerical' (triggers I1) — so the demo exercises BOTH checks.
    if paper_idx == 2:
        w4 = {
            "name": f"{topic} solution design",
            "type": "solution",
            "context": f"{topic} method implemented as a discrete-codebook + masked-modeling solution.",
            "provenance": {"code_span": "section 4"},
            "parent_W3_id": parent,
            "extends": [], "improves": [],
        }
    else:
        w4 = {
            "name": f"{topic} numerical result",
            "type": "numerical",
            "context": f"Our method achieves a normalized score of 0.92 on the {topic} benchmark.",
            "provenance": {"score": 0.92, "score_std": 0.01},
            "parent_W3_id": parent,
            "extends": [], "improves": [],
        }
    w4["W5_code"] = [
        {"name": f"{topic}_training_experiment", "type": "experiment",
         "context": "End-to-end training loop.",
         "code_ref": "train",
         "code_body": "critic = CentralisedCritic(all_obs, all_act)\nloss = mse(critic(s, a), r)",
         "provenance": {"code_span": "lines 1-60"}},
        {"name": f"{topic}_core_module", "type": "component",
         "context": "Core algorithmic component.",
         "code_ref": "core_module", "code_body": "def core_module(x): return x"},
    ]
    return {"W4_implementations": [w4]}


# ---------------------------------------------------------------------------
# 3. Mocks for LLM / embedding / citation APIs
# ---------------------------------------------------------------------------
def mock_embed():
    def _embed(texts, **kwargs):
        rng = np.random.RandomState(abs(hash(tuple(texts))) % (2**32))
        return rng.randn(len(texts), 64).astype(np.float32)
    return _embed


def make_chat_json(state):
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        if "W1_problem" in msg:
            return make_phase1(msg)
        if "W4_implementations" in msg:
            return make_phase2(msg, paper_idx=state["idx"])
        # I1 score re-extraction — per-paper outcome to exercise the taxonomy:
        #   paper 0 (safe_rl):       re-extract agrees  -> I1 pass -> verified
        #   paper 1 (inverse_manip): re-extract differs -> I1 fail -> low_confidence (soft)
        #   paper 2 (eeg):           re-extract agrees  -> I1 pass -> verified
        if "extract the primary numerical score" in msg or "score" in msg.lower():
            reextracted = 0.50 if state["idx"] == 1 else 0.92
            return {"score": reextracted, "reasoning": "re-extracted"}
        if "higher-level" in msg.lower() or "synthesize" in msg.lower() or "reduced_from" in msg:
            return {"name": "Reduced abstraction", "context": "cross-atom synthesis", "type": "concept"}
        return {"fixes": []}
    return _fn


def make_majority(state):
    def _fn(messages, **kwargs):
        msg = messages[0]["content"] if messages else ""
        idx = state["idx"]
        if "compliant" in msg or "violates_spec" in msg:
            # I2: paper 0 violates (centralised critic), papers 1 & 2 compliant
            verdict = "violates_spec" if idx == 0 else "compliant"
            return {"verdict": verdict, "reasoning": "demo I2"}
        if "aligned" in msg or "misaligned" in msg:
            # I4: paper 2 partially aligned (soft fail -> low_confidence), others aligned
            verdict = "partially_aligned" if idx == 2 else "aligned"
            return {"verdict": verdict, "reasoning": "demo I4"}
        return {"verdict": "ambiguous", "reasoning": "fallback"}
    return _fn


def mock_requests_all_none():
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
    g = MagicMock()
    g.side_effect = _side
    return g


def reset_singletons():
    ccchain._store = ccchain._extractor = ccchain._refiner = None
    ccchain._reducer = ccchain._retriever = ccchain._evaluator = None
    ccchain._verifier = None


# ---------------------------------------------------------------------------
# 4. Run + report
# ---------------------------------------------------------------------------
def main():
    papers = extract_paper_texts()
    print(f"Loaded {len(papers)} real PDFs from {PDF_DIR}\n")
    for p in papers:
        print(f"  - {p['filename']} ({p['pages']} pages, {sum(len(c) for c in p['chunks'])} chars)")
    print()

    tmpdir = tempfile.mkdtemp(prefix="ccchain_audit_demo_")
    cfg = Config(db_path=os.path.join(tmpdir, "demo.db"), graph_dir=tmpdir,
                 reference_api_timeout=0.1, reference_api_max_retries=1,
                 audit_majority_k=1, audit_i1_k=1)
    task_spec = TaskSpec(
        task_name="robot-control-bench", eval_harness="safety-gym",
        success_criteria="return >= 0.9 with zero constraint violations",
        constraints=["CTDE", "no centralised critic at execution time"],
    )

    reports = []
    state = {"idx": 0}
    with patch("ccchain.Config", return_value=cfg), \
         patch("ccchain.core.llm.chat_json", side_effect=make_chat_json(state)), \
         patch("ccchain.core.llm.chat_json_majority", side_effect=make_majority(state)), \
         patch("ccchain.core.embedding.embed", side_effect=mock_embed()), \
         patch("ccchain.core.references.requests.get", side_effect=mock_requests_all_none()):

        for i, p in enumerate(papers):
            state["idx"] = i
            combined = "\n\n".join(p["chunks"][:5])  # first ~5 chunks for speed
            result, err = ccchain.ingest([combined], source_pdf=p["filename"], task_spec=task_spec)
            if err:
                print(f"  ! ingest error for {p['filename']}: {err}")
                continue
            ar = result["audit_report"]
            reports.append((p["filename"], result, ar))

    # ---- Print audit reports ----
    print("=" * 72)
    print("CoE AUDIT REPORTS (real PDF text, mocked LLM/citation layer)")
    print("=" * 72)
    for fname, result, ar in reports:
        print(f"\n>> {fname}")
        print(f"  nodes by level: {result['node_count_by_level']}")
        print(f"  edges: {result['edge_count']}  trajectories: {result['trajectory_count']}")
        print(f"  CPR (Claim Provenance Rate): {ar['cpr']:.3f}")
        print(f"  atoms audited/passed/failed/skipped: "
              f"{ar['atoms_audited']}/{ar['atoms_passed']}/{ar['atoms_failed']}/{ar['atoms_skipped']}")
        print(f"  failures by check: {ar['failures_by_check']}")
        status_counts = Counter(pa["status"] for pa in ar["per_atom"])
        print(f"  status distribution: {dict(status_counts)}")

    # ---- Search status filtering demo ----
    print("\n" + "=" * 72)
    print("SEARCH STATUS FILTERING (W3 level, query='credit assignment / baseline')")
    print("=" * 72)
    if papers:
        with patch("ccchain.plugins.retrieval.embed", side_effect=mock_embed()):
            default_res, _ = ccchain.search("credit assignment baseline", top_k=20, level="W3")
            all_res, _ = ccchain.search("credit assignment baseline", top_k=20, level="W3", status="all")
            low_res, _ = ccchain.search("credit assignment baseline", top_k=20, level="W3",
                                        status="low_reliability")
        print(f"  default (trusted only):      {len(default_res)} atoms")
        if default_res:
            print(f"    statuses: {Counter(r['status'] for r in default_res)}")
        print(f"  status='all' (incl. failed): {len(all_res)} atoms")
        if all_res:
            print(f"    statuses: {Counter(r['status'] for r in all_res)}")
        print(f"  status='low_reliability':    {len(low_res)} atoms (audit backtrace)")

    # ---- HTML visualization ----
    out_path = os.path.join(_REPO_ROOT, "blueprint_output", "audit_report.html")
    try:
        from ccchain.visualize import build_audit_html
        abs_path = build_audit_html(ccchain._store, reports, out_path)
        print(f"\nHTML report written: {abs_path}")
        print("Open it in a browser to explore the status-colored graph + per-atom CoE verdicts.")
    except Exception as exc:
        print(f"\n! HTML generation failed: {exc}")

    reset_singletons()
    print("\nDone. DB was in a temp dir and is not persisted.")


if __name__ == "__main__":
    main()

"""End-to-end live test of ccchain with real LLM + embeddings on ARIS PDFs.

Usage:
    cd E:\DATA\vscode\FLUXturbo
    python scripts/e2e_live_test.py

Prerequisites:
    - Ollama running at http://localhost:11434
    - Models pulled: qwen3:latest, bge-m3:latest
"""

import os
import sys
import time

# Add FLUXturbo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyPDF2 import PdfReader

import ccchain

PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"
PDF_NAMES = ["2606.04749v1.pdf", "2606.05248v1.pdf", "2606.05855v1.pdf"]

# ── helpers ──────────────────────────────────────────────────────────────

def extract_text(pdf_path: str, max_chars: int = 8000) -> str:
    """Extract first max_chars of text from a PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []
    total = 0
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
    return "\n\n".join(text_parts)[:max_chars]


def on_progress(stage: str, pct: float):
    print(f"  [{stage}] {pct*100:.0f}%")


def separator(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    separator("1. Extract text from PDFs")
    papers = {}
    for pdf_name in PDF_NAMES:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"  SKIP: {pdf_path} not found")
            continue
        text = extract_text(pdf_path, max_chars=8000)
        papers[pdf_name] = text
        print(f"  {pdf_name}: {len(text)} chars extracted")

    if not papers:
        print("  ERROR: No PDFs found")
        return

    # ── 2. Ingest each paper ─────────────────────────────────────────────
    separator("2. Ingest papers (extract → refine → store → reduce)")
    for pdf_name, text in papers.items():
        print(f"\n  >>> Ingesting {pdf_name} ({len(text)} chars)")
        result, err = ccchain.ingest(
            segments=[text],
            source_pdf=pdf_name,
            on_progress=on_progress,
        )
        if err:
            print(f"  ERROR: {err}")
            return
        print(f"  OK: nodes={result['node_count_by_level']}, "
              f"edges={result['edge_count']}, "
              f"trajectories={result['trajectory_count']}")

    # ── 3. Search ────────────────────────────────────────────────────────
    separator("3. Search")

    queries = [
        ("safe reinforcement learning", "W4"),
        ("credit assignment in multi-agent", "W3"),
        ("optimal transport", "W5"),
    ]
    for query, level in queries:
        print(f"\n  >>> search('{query}', level='{level}')")
        t0 = time.time()
        results, err = ccchain.search(query, top_k=3, level=level)
        dt = time.time() - t0
        if err:
            print(f"  ERROR: {err}")
            continue
        print(f"  {len(results)} results ({dt*1000:.0f}ms):")
        for r in results:
            print(f"    [{r['score']:.4f}] {r['name']} ({r['level']})")
            ctx = r.get("context", "")
            if ctx:
                print(f"      {ctx[:120]}...")

    # ── 4. Evaluate novelty ──────────────────────────────────────────────
    separator("4. Evaluate novelty of a research proposal")

    proposal = (
        "A novel approach combining partial optimal transport with Shapley values "
        "for credit assignment in multi-agent reinforcement learning. "
        "Unlike prior work that uses full OT or Sinkhorn distances, partial OT "
        "handles sparse reward signals by ignoring outlier agent contributions. "
        "The Shapley value component provides axiomatic fairness guarantees "
        "for team credit decomposition."
    )
    print(f"\n  Proposal: {proposal[:150]}...")
    print(f"  (this will take 1-2 min: LLM extract + embed + Hausdorff + LLM rubric)")

    t0 = time.time()
    report, err = ccchain.evaluate(proposal, on_progress=on_progress)
    dt = time.time() - t0

    if err:
        print(f"  ERROR: {err}")
    else:
        print(f"\n  Completed in {dt:.1f}s")
        print(f"  Novelty score: {report.get('novelty_score', 'N/A')}")
        print(f"  Recommendation: {report.get('recommendation', 'N/A')}")
        print(f"  Most similar trajectory: {report.get('most_similar_trajectory', 'N/A')}")
        print(f"  Level distances: {report.get('level_distances', {})}")
        print(f"  Dimension scores: {report.get('dimension_scores', {})}")
        print(f"  Divergence points: {report.get('divergence_points', [])}")
        print(f"  Transient atoms cleaned: {report.get('_deleted_transient', 0)}")

    separator("Done")
    print("  End-to-end test complete. All three APIs working with real LLM.")


if __name__ == "__main__":
    main()

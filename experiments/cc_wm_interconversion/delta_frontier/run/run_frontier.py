"""run/run_frontier.py — fit the credibility–fidelity frontier (the δ-frontier law).

Multi-fidelity sweep: synthetic WMs at increasing noise σ (lower σ = higher fidelity).
At each level measure δ (defect), g (credibility = mean(1−δ)), c (cost = 1/σ), then fit
  g ≤ G(δ)  (monotone decreasing) + hard bound (low-δ high-g ⇒ high c).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import paths  # noqa
import warnings; warnings.filterwarnings("ignore")
from wms.synthetic import SyntheticLossyWM
from delta.delta import measure_delta
from frontier.fit import credibility_from_delta, fit_frontier

OUT = paths.OUTPUT_DIR
SIGMAS = [0.02, 0.08, 0.2, 0.4, 0.8, 1.5]


def main():
    points = []
    print(f"=== multi-fidelity sweep (synthetic, σ ∈ {SIGMAS}) ===")
    for i, sigma in enumerate(SIGMAS):
        wm = SyntheticLossyWM(dim=6, noise_std=sigma, seed=i)
        res = measure_delta(wm, wm.sample_transitions(n_frames=2500), seed=i)
        deltas = [c["delta_iia"] for c in res["delta_per_claim"]]
        delta = sum(deltas) / len(deltas)
        g = credibility_from_delta(deltas)
        c = 1.0 / (sigma + 1e-3)        # lower noise ⇒ higher cost to achieve
        points.append({"level": f"σ={sigma}", "delta": round(delta, 4),
                       "g": round(g, 4), "c": round(c, 3)})
        print(f"  σ={sigma:<5}  δ={delta:.3f}  g={g:.3f}  c={c:.2f}")

    frontier = fit_frontier(points)
    print("\n=== credibility–fidelity frontier fit ===")
    print(f"  G(δ) = {frontier['g_of_delta_fit']['a']}·exp(−{frontier['g_of_delta_fit']['b']}·δ)")
    print(f"  monotonicity  ρ(g, δ) = {frontier['monotonicity_spearman_g_vs_delta']}  (want ≈ −1: g ↓ as δ ↑)")
    print(f"  hard-bound    ρ(c, δ) = {frontier['hard_bound_spearman_c_vs_delta']}  (want < 0: c ↑ as δ ↓)")
    print(f"  FRONTIER LAW {'HOLDS ✅' if frontier['law_holds'] else 'WEAK ❌'}  "
          f"(g≤G(δ) monotone + low-δ⇒high-c hard bound)")

    (OUT / "frontier_fit.json").write_text(json.dumps(
        {"frontier": frontier, "points": points}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"  → {OUT/'frontier_fit.json'}")


if __name__ == "__main__":
    main()

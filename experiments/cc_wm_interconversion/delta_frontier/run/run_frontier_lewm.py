"""run/run_frontier_lewm.py — fit the δ-frontier on a REAL world model (le-wm).

Sweep fidelity by adding noise to le-wm's real latent z (NoisyWrapper). Shows the frontier
law G(δ)=a·exp(−b·δ) holds starting from a real WM's representation space — not just synthetic.
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
import torch
from wms.lewmdm import LeWMAdapter
from wms.noisy import NoisyWrapper
from delta.delta import measure_delta
from frontier.fit import credibility_from_delta, fit_frontier

OUT = paths.OUTPUT_DIR
NOISE = [0.0, 0.3, 0.7, 1.2, 2.0, 3.5]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = LeWMAdapter("reacher", dev)
    data = base.sample_transitions("reacher", n_frames=2500)
    points = []
    print(f"=== real-WM frontier sweep (le-wm reacher, noise on z) ===")
    for i, n in enumerate(NOISE):
        wm = NoisyWrapper(base, noise_std=n, seed=i)
        res = measure_delta(wm, data, seed=i)
        deltas = [c["delta_iia"] for c in res["delta_per_claim"]]
        delta = sum(deltas) / len(deltas)
        g = credibility_from_delta(deltas)
        c = 1.0 / (n + 0.1)            # higher fidelity (lower noise) ⇒ higher cost
        points.append({"level": f"noise={n}", "delta": round(delta, 4),
                       "g": round(g, 4), "c": round(c, 3)})
        print(f"  noise={n:<4}  δ={delta:.3f}  g={g:.3f}  c={c:.2f}")

    fr = fit_frontier(points)
    print("\n=== real-WM frontier fit ===")
    print(f"  G(δ) = {fr['g_of_delta_fit']['a']}·exp(−{fr['g_of_delta_fit']['b']}·δ)")
    print(f"  ρ(g,δ) = {fr['monotonicity_spearman_g_vs_delta']}  |  ρ(c,δ) = {fr['hard_bound_spearman_c_vs_delta']}")
    print(f"  REAL-WM FRONTIER LAW {'HOLDS ✅' if fr['law_holds'] else 'WEAK ❌'}")

    (OUT / "frontier_lewm.json").write_text(json.dumps(
        {"frontier": fr, "points": points, "base_wm": base.name}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"  → {OUT/'frontier_lewm.json'}")


if __name__ == "__main__":
    main()

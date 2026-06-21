"""run/run_delta_calib.py — Phase A: calibrate δ on le-wm reacher + gate check.

Gate (go/no-go for Phase B):
  - δ_iia(qpos) low (faithfully encoded), δ_iia(qvel) high (not encoded)
  - null IIA ≈ 0 (anti-trivial), consistency ρ(δ_iia, 1−R²) high
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # delta_frontier root
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import paths  # noqa (sys.path setup)
import torch
from wms.lewmdm import LeWMAdapter
from delta.delta import measure_delta

OUT = paths.OUTPUT_DIR


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = LeWMAdapter(env="reacher", device=dev)
    data = wm.sample_transitions("reacher", n_frames=2500)
    print(f"[calib] {wm.name}  latent_dim={wm.latent_dim}  cost={wm.cost()}")

    res = measure_delta(wm, data, seed=0)
    (OUT / "delta_calib_lewm_reacher.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ── per-claim δ table ─────────────────────────────────────────────
    print("\n=== δ per claim (sorted by δ_iia) ===")
    print(f"{'claim':16s} {'δ_iia':>7s} {'IIA':>7s} {'null':>7s} {'dose_mono':>10s} {'R²_probe':>9s}")
    for c in sorted(res["delta_per_claim"], key=lambda x: x["delta_iia"]):
        print(f"{c['name']:16s} {c['delta_iia']:7.3f} {c['iia']:7.3f} "
              f"{c['iia_null_mean']:7.3f} {c.get('dose_monotonicity',0):10.2f} {c['r2_probe']:9.3f}")

    # ── gate ──────────────────────────────────────────────────────────
    by = {c["name"]: c for c in res["delta_per_claim"]}
    d_qpos0 = by.get("qpos[0]", {}).get("delta_iia", float("nan"))
    d_qvel0 = by.get("qvel[0]", {}).get("delta_iia", float("nan"))
    null = res["anti_trivial"]["null_iia_mean"]
    rho = res["consistency_spearman_delta_vs_1minusR2"]
    gate = {
        "delta_iia_qpos0_low": d_qpos0 < 0.3,
        "delta_iia_qvel0_high": d_qvel0 > 0.5,
        "null_iia_near_zero": null < 0.2,
        "consistency_rho_high": rho > 0.5,
    }
    print("\n=== Phase A gate ===")
    print(f"  composite δ = {res['delta_composite']}")
    print(f"  δ_iia(qpos[0]) = {d_qpos0:.3f}  (want <0.3)  → {gate['delta_iia_qpos0_low']}")
    print(f"  δ_iia(qvel[0]) = {d_qvel0:.3f}  (want >0.5)  → {gate['delta_iia_qvel0_high']}")
    print(f"  null IIA mean  = {null:.3f}  (want <0.2)  → {gate['null_iia_near_zero']}")
    print(f"  consistency ρ  = {rho:.3f}  (want >0.5)  → {gate['consistency_rho_high']}")
    print(f"  anti-trivial illusions caught: {res['anti_trivial']['n_correlational_illusions_caught']}")
    print(f"  dynamics: {res['dynamics']}")
    print(f"\n  GATE {'PASS ✅' if all(gate.values()) else 'FAIL ❌'}  → {'proceed to Phase B' if all(gate.values()) else 'debug δ'}")
    print(f"  → {OUT/'delta_calib_lewm_reacher.json'}")


if __name__ == "__main__":
    main()

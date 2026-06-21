"""run/run_exactness.py — Phase B / C1: validate the δ=0 exactness anchor.

Synthetic linear WM (z = state, dynamics = symbolic claim) ⇒ δ_iia ≈ 0 by construction.
Contrast with le-wm reacher (δ>0 for the unobservable velocity). First step of the
three-regime closure (δ=0 exact / δ small / δ≥δ_min).
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
import torch
from wms.synthetic import SyntheticLinearWM
from delta.delta import measure_delta

OUT = paths.OUTPUT_DIR


def main():
    print("=== C1: synthetic linear δ=0 anchor ===")
    syn = SyntheticLinearWM(dim=6)
    data = syn.sample_transitions(n_frames=2500)
    res = measure_delta(syn, data, seed=0)
    max_d = max(c["delta_iia"] for c in res["delta_per_claim"])
    print(f"  composite δ = {res['delta_composite']}  | max δ_iia = {max_d:.4f}")
    for c in sorted(res["delta_per_claim"], key=lambda x: x["delta_iia"]):
        print(f"    {c['name']:8s} δ_iia={c['delta_iia']:.4f}  IIA={c['iia']:.3f}  R²={c['r2_probe']:.3f}")
    exactness_pass = max_d < 0.05
    print(f"  EXACTNESS (max δ_iia < 0.05): {'PASS ✅' if exactness_pass else 'FAIL ❌'}")

    print("\n=== contrast: le-wm reacher (δ>0 for unobservable velocity) ===")
    lewm_d = None
    try:
        from wms.lewmdm import LeWMAdapter
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lewm = LeWMAdapter("reacher", dev)
        ldata = lewm.sample_transitions("reacher", n_frames=2500)
        lres = measure_delta(lewm, ldata, seed=0)
        lewm_d = {c["name"]: round(c["delta_iia"], 3) for c in lres["delta_per_claim"]}
        print(f"  le-wm δ: qpos≈{lewm_d.get('qpos[0]')}  qvel≈{lewm_d.get('qvel[0]')}  "
              f"(velocity δ≫0 ⇒ unobservable, not exact)")
    except Exception as e:
        print(f"  (le-wm skipped: {str(e)[:80]})")

    (OUT / "exactness_c1.json").write_text(json.dumps({
        "C1_synthetic_delta0": {"composite": res["delta_composite"], "max_delta_iia": round(max_d, 4),
                                "exactness_pass": exactness_pass,
                                "per_claim": res["delta_per_claim"]},
        "contrast_lewm_reacher": lewm_d,
        "note": "δ=0 when WM≡symbolic claim (synthetic); δ>0 for unobservable quantities (le-wm velocity).",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n→ {OUT/'exactness_c1.json'}")


if __name__ == "__main__":
    main()

"""delta/delta.py — unified alignment defect δ(WM, CC) + anti-triviality evidence.

Per claim (a physical quantity the CC asserts z encodes):
  - δ_iia : causal interchange-intervention defect (PRIMARY; locked §12-1)
  - R²    : correlational linear-probe R² (corroborating)
  - null  : random-direction IIA ensemble (anti-trivial: should be ≈0)
  - dose_mono : dose-response monotonicity (anti-illusion: faithful ⇒ monotone)

Anti-triviality is built into the causal interchange (swapping the subspace tests USE not
correlation) + the null ensemble + dose-response. A "correlational illusion" = high R² but
high δ_iia (probe found a correlating direction the model doesn't actually use) — caught here.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from delta import extract, intervene


def measure_delta(wm, data, seed: int = 0) -> dict:
    import time
    t0 = time.time()
    z = wm.encode(data["pixels"])
    print(f"[delta] encode {time.time()-t0:.1f}s", flush=True)
    state = wm.probe_targets(data)
    t1 = time.time()
    analysis = extract.analyze(z, state, seed)
    print(f"[delta] analyze (probe+sindy) {time.time()-t1:.1f}s", flush=True)
    probes = {p["name"]: p for p in analysis["probes"]}

    per_claim = []
    for col, arr in state.items():
        if col in ("episode_idx", "step_idx", "action"):
            continue
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[0] != z.shape[0]:
            continue
        for j in range(a.shape[1]):
            nm = f"{col}[{j}]"
            d = intervene.measure_delta_iia(z, a[:, j], nm, seed=seed + abs(hash(nm)) % 99)
            d["r2_probe"] = probes.get(nm, {}).get("r2", float("nan"))
            per_claim.append(d)

    deltas = [c["delta_iia"] for c in per_claim if c["delta_iia"] == c["delta_iia"]]
    r2s = [c["r2_probe"] for c in per_claim if c["r2_probe"] == c["r2_probe"]]
    # consistency: δ_iia should track (1 − R²) across claims
    pairs = [(c["delta_iia"], 1 - c["r2_probe"]) for c in per_claim
             if c["delta_iia"] == c["delta_iia"] and c["r2_probe"] == c["r2_probe"]]
    rho = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation) if len(pairs) > 2 else float("nan")
    # anti-trivial: null ensemble mean (should be ≈0) + correlational-illusion count
    null_mean = float(np.mean([c["iia_null_mean"] for c in per_claim]))
    illusions = [c["name"] for c in per_claim
                 if c["r2_probe"] > 0.5 and c["delta_iia"] > 0.5]   # high R² but high δ ⇒ correlational illusion

    return {
        "wm": wm.name, "n_claims": len(per_claim),
        "delta_composite": round(float(np.mean(deltas)), 4) if deltas else float("nan"),
        "delta_per_claim": [{k: v for k, v in c.items() if k != "dose_response"} for c in per_claim],
        "consistency_spearman_delta_vs_1minusR2": round(rho, 4),
        "anti_trivial": {
            "null_iia_mean": round(null_mean, 4),
            "n_correlational_illusions_caught": len(illusions),
            "illusions": illusions,
            "note": "null IIA≈0 + causal-interchange ⇒ δ is non-trivial; illusions = high R² but high δ_iia.",
        },
        "dynamics": analysis["dynamics"],
    }

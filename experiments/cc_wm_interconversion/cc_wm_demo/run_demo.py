"""run_demo.py — orchestrate the CC⇌WM round-trip on le-wm reacher.

Phases: 0 load+encode → 1 probe/extract → 2 build+render CC → 3 compile-back → 4 fidelity.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, glob
# Windows console (GBK) can't print R²/→ — force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
import numpy as np
import torch

import config as C
from wm import load_model, encode_pixels, load_reacher_data
import extract, cc
from compile import fine_tune
from fidelity import pred_mse, roundtrip_report

OUT = C.OUTPUT_DIR


def ensure_h5() -> Path:
    """Locate reacher.h5; if only tar.zst present, decompress."""
    cands = list(C.DATA_DIR.rglob("*.h5"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_size, reverse=True)
        print(f"[data] found h5: {cands[0]}")
        return cands[0]
    zst = list((C.DATA_DIR / "lewm-reacher-data").glob("*.tar.zst")) if (C.DATA_DIR / "lewm-reacher-data").exists() else []
    if not zst:
        sys.exit("[data] no h5 and no reacher.tar.zst — run the HF download first (see README).")
    zst = zst[0]
    print(f"[data] decompressing {zst.name} ...")
    # try tar --zstd (git-bash) then python fallback
    rc = subprocess.run(["tar", "--zstd", "-xf", str(zst), "-C", str(zst.parent)],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        print("[data] tar --zstd failed, trying python zstandard ...")
        import zstandard as zstd  # noqa
        import io, tarfile
        dctx = zstd.ZstdDecompressor()
        with open(zst, "rb") as f:
            data = dctx.decompressobj().decompress(f.read())
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            tf.extractall(str(zst.parent))
    cands = list((C.DATA_DIR / "lewm-reacher-data").rglob("*.h5"))
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    print(f"[data] decompressed → {cands[0] if cands else 'NO H5 FOUND'}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--skip-roundtrip", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== device: {dev} ===")

    # ── Phase 0: data + model + encode ──────────────────────────────────
    h5 = ensure_h5()
    data = load_reacher_data(h5, max_frames=args.frames)
    model = load_model(dev)
    print("[phase0] encoding pixels → latents ...")
    z = encode_pixels(model, data["pixels"], dev)
    print(f"[phase0] z.shape={z.shape}")
    state = {k: v for k, v in data.items() if k not in ("pixels",)}

    # ── Phase 1: probe + components + dynamics ──────────────────────────
    probes = extract.probe_all(z, state)
    print("\n[phase1] probing results (top 8 by R²):")
    for p in sorted(probes, key=lambda q: (q["r2"] if q["r2"] == q["r2"] else -1), reverse=True)[:8]:
        print(f"    {p['name']:18s} R²={p['r2']:.3f}  ctrl={p['ctrl_r2']:.3f}  sel={p['selectivity']:.3f}")
    pca = extract.pca_components(z)
    dyn = extract.linear_dynamics(z)
    # temporal (2-frame) probes — partial-observability check (velocity needs context)
    temporal_probes = []
    ts = 500
    for col, arr in state.items():
        if col in ("episode_idx", "step_idx", "action"):
            continue
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == z.shape[0]:
            for j in range(a.shape[1]):
                temporal_probes.append(extract.probe_temporal(z, a[:, j], f"{col}[{j}]", ts))
                ts += 1
    print("[phase1] temporal(2-frame) probes for low-R² quantities:")
    for p in sorted(probes, key=lambda q: q["r2"] if q["r2"] == q["r2"] else 9)[:4]:
        tp = next((t for t in temporal_probes if t["name"] == p["name"]), None)
        if tp:
            print(f"    {p['name']:18s} single-frame R²={p['r2']:.3f}  2-frame R²={tp['r2']:.3f}")
    analysis = {"probes": probes, "pca": {"explained_variance": pca["explained_variance"],
                                          "n_components": pca["n_components"]},
                "dynamics": dyn, "temporal_probes": temporal_probes}

    # ── Phase 2: build + render CC ──────────────────────────────────────
    atoms, edges = cc.build_cc(analysis)
    store = cc.render_cc(atoms, edges, OUT / "extracted_cc.html", OUT / "extracted_cc.json")
    (OUT / "analysis_WM.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str))

    if args.skip_roundtrip:
        print("\n[--skip-roundtrip] done after Phase 2.")
        return

    # ── Phase 3: compile-back (adaptive target columns by per-column mean R²) ─
    valid = [p for p in probes if p["r2"] == p["r2"]]
    col_r2 = {}
    for p in valid:
        col = p["name"].split("[")[0]
        col_r2.setdefault(col, []).append(p["r2"])
    col_mean = {c: float(np.mean(v)) for c, v in col_r2.items()}
    preserve_col = max(col_mean, key=col_mean.get)
    fix_col = min({c: m for c, m in col_mean.items() if c != preserve_col}, key=col_mean.get)
    print(f"\n[phase3] per-column mean R²: {dict((k, round(v,3)) for k,v in col_mean.items())}")
    print(f"[phase3] compile-back: preserve='{preserve_col}'  fix='{fix_col}'")
    if fix_col not in data or preserve_col not in data:
        print(f"[phase3] WARN: target col missing in data; available={list(data)}")
    model2, heads, stats = fine_tune(model, data, dev, epochs=args.epochs, batch_size=args.bs,
                                     lam_aux=args.lam, fix_targets=(fix_col,),
                                     preserve_targets=(preserve_col,))
    torch.save({"state_dict": model2.state_dict(), "fix": fix_col, "preserve": preserve_col},
               OUT / "compiled_wm.pt")

    # ── Phase 4: re-encode WM' + fidelity ──────────────────────────────
    z2 = encode_pixels(model2, data["pixels"], dev)
    probes2 = extract.probe_all(z2, state)
    mse1 = pred_mse(model, data, dev)
    mse2 = pred_mse(model2, data, dev)
    tp_fix = [p for p in analysis.get("temporal_probes", []) if p["name"].startswith(fix_col + "[")]
    rep = roundtrip_report(probes, probes2, mse1, mse2, preserve_col, fix_col, tp_fix,
                           OUT / "roundtrip_report.md")
    (OUT / "roundtrip_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))

    # ── re-render CC' to show bottleneck resolved ──────────────────────
    analysis2 = {"probes": probes2, "pca": analysis["pca"], "dynamics": dyn}
    atoms2, edges2 = cc.build_cc(analysis2, wm_label="LeWM-reacher-prime")
    cc.render_cc(atoms2, edges2, OUT / "extracted_cc_prime.html", OUT / "extracted_cc_prime.json",
                 label="WM'→CC (after compile-back)", db_path=OUT / "cc_prime.db",
                 graph_dir=OUT / "cc_graph_prime")

    print("\n=== DONE. Artifacts in", OUT, "===")
    print(f"  extracted_cc.html       (WM→CC, {len(atoms)} atoms)")
    print(f"  extracted_cc_prime.html (WM'→CC, {len(atoms2)} atoms)")
    print(f"  roundtrip_report.md     (3-layer fidelity)")


if __name__ == "__main__":
    main()

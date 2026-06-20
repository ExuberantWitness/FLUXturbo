"""cc_wm.py — the formal CC⇌WM interconversion interface (TRUE bidirectional).

  extract(wm, data)      → cc   (WM→CC: causal probing → claim chain)
  compile(cc, wm, data)  → wm'  (CC→WM: READS the cc graph → constraints → LoRA finetune)
  roundtrip(wm, data)    → report  (WM→CC→WM'(from CC)→CC' + fidelity)

The compile step is genuinely CC-driven: it parses the cc's numerical (preserve) and
bottleneck (fix) atoms into auxiliary decode heads targeting the corresponding state
quantities — no hand-specified target function.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# cc_wm_demo FIRST so `config`/`wm`/`extract`/`cc`/`fidelity` resolve to demo
_HERE = str(Path(__file__).resolve().parent)
for _p in (_HERE, "E:/DATA/vscode/cc_wm_demo"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wm as demo_wm              # cc_wm_demo/wm.py
import extract as demo_extract   # cc_wm_demo/extract.py
import cc as demo_cc             # cc_wm_demo/cc.py
import fidelity as demo_fidelity  # cc_wm_demo/fidelity.py (pred_mse)
import das, fidelity_sota         # this package
from semantic_compile import install_lora_projector  # this package (LoRA)


# ── helpers ───────────────────────────────────────────────────────────
_TARGET_RE = re.compile(r"(\w+)\[(\d+)\]")


def _parse_target(atom):
    """(col, idx) from a numerical/bottleneck/concept atom's provenance or name.
    atom is a ccchain Atom dataclass (attributes, not dict)."""
    prov = getattr(atom, "provenance", None) or {}
    for key in ("target", "worst_target"):
        t = prov.get(key)
        if t:
            m = _TARGET_RE.search(str(t))
            if m:
                return m.group(1), int(m.group(2))
    m = _TARGET_RE.search(getattr(atom, "name", "") or "")
    return (m.group(1), int(m.group(2))) if m else None


def _norm(arr):
    a = np.asarray(arr, float); mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd


# ── WM → CC ───────────────────────────────────────────────────────────
def extract(wm, data, dev="cuda", max_frames=2500) -> dict:
    z = demo_wm.encode_pixels(wm, data["pixels"][:max_frames], dev)
    state = {k: (v[:max_frames] if len(v) > max_frames else v) for k, v in data.items() if k != "pixels"}
    probes = demo_extract.probe_all(z, state)
    das_res = {}
    for col in ("qpos", "qvel", "observation"):
        if col in state:
            a = np.asarray(state[col])
            for j in range(a.shape[1]):
                nm = f"{col}[{j}]"; das_res[nm] = das.das_verify(z, a[:, j], nm, seed=abs(hash(nm)) % 99)
    analysis = {"probes": probes, "pca": demo_extract.pca_components(z),
                "dynamics": demo_extract.linear_dynamics(z), "temporal_probes": [], "das": das_res}
    atoms, edges = demo_cc.build_cc(analysis)
    return {"atoms": atoms, "edges": edges, "analysis": analysis, "z": z, "state": state}


# ── CC → WM (reads the cc graph) ──────────────────────────────────────
def compile(cc, wm, data, dev="cuda", epochs=3, lr=5e-4, max_batches=40, seed=0) -> dict:
    """Parse cc atoms → preserve/fix decode targets → LoRA(projector)+multi-head finetune."""
    state = cc["state"]
    role_by_target = {}                                  # (col,idx) -> role ; 'fix' overrides 'preserve'
    for a in cc["atoms"]:
        t = _parse_target(a)
        if not t:
            continue
        if a.type == "numerical":
            role_by_target.setdefault(t, "preserve")
        elif a.type in ("bottleneck", "concept"):
            role_by_target[t] = "fix"                   # fix overrides
    tgts = sorted(role_by_target.items())
    if not tgts:
        return {"wm": wm, "compiled_targets": [], "aux_final": float("nan"), "per_target_final": []}

    cols = [(c, i) for (c, i), _ in tgts]
    roles = [r for _, r in tgts]
    Y = np.stack([state[c][:, i] for c, i in cols], axis=1).astype(np.float32)
    Yn = _norm(Y)

    head = nn.Linear(192, Y.shape[1]).to(dev)
    lora_params, _ = install_lora_projector(wm, r=8)
    opt = torch.optim.AdamW(list(lora_params) + list(head.parameters()), lr=lr, weight_decay=1e-3)
    wm.train()
    rng = np.random.default_rng(seed)
    from wm import prepare_pixels
    pixels = data["pixels"][:len(state["qpos"])]
    N = len(state["qpos"]); hist = {"aux": [], "per": []}
    for ep in range(epochs):
        order = rng.permutation(N - 2); bi = 0; last_per = None
        for bs in range(0, len(order), 16):
            if bi >= max_batches:
                break
            ix = order[bs:bs + 16]
            pix = prepare_pixels(pixels[ix], dev)
            info = {"pixels": pix.unsqueeze(1)}; info = wm.encode(info)
            z_ = info["emb"][:, 0, :]
            yb = torch.tensor(Yn[ix], dtype=torch.float32, device=dev)
            pred = head(z_)
            aux = ((pred - yb) ** 2).mean()
            opt.zero_grad(); aux.backward(); opt.step()
            hist["aux"].append(float(aux.item()))
            last_per = ((pred - yb) ** 2).mean(0).detach().cpu().numpy()
            bi += 1
        hist["per"].append(last_per.tolist() if last_per is not None else [])
        print(f"[cc→wm] epoch {ep+1}/{epochs}  aux={np.mean(hist['aux'][-max_batches:]):.4f}")
    wm.eval()
    per_final = hist["per"][-1] if hist["per"] else []
    per_auxR2 = [round(1.0 - x, 3) for x in per_final]
    return {"wm": wm, "compiled_targets": [{"col": c, "idx": i, "role": r} for (c, i), r in
                                           [((cols[k][0], cols[k][1]), roles[k]) for k in range(len(cols))]],
            "aux_final": float(np.mean(hist["aux"][-10:])),
            "per_target_auxR2": dict(zip([f"{c}[{i}]" for c, i in cols], per_auxR2))}


# ── round-trip (closed loop) ──────────────────────────────────────────
def roundtrip(wm, data, dev="cuda", epochs=3) -> dict:
    cc = extract(wm, data, dev)
    z_pre = cc["z"]; state = cc["state"]
    mse_pre = demo_fidelity.pred_mse(wm, data, dev)          # before compile (wm original)
    compiled = compile(cc, wm, data, dev, epochs=epochs)     # mutates wm → wm'
    wm_p = compiled["wm"]
    z_post = demo_wm.encode_pixels(wm_p, data["pixels"][:len(z_pre)], dev)
    probes_post = demo_extract.probe_all(z_post, state)
    mse_post = demo_fidelity.pred_mse(wm_p, data, dev)
    fid = fidelity_sota.roundtrip_fidelity_sota(
        cc["analysis"]["probes"], probes_post, z_pre, z_post, mse_pre, mse_post)
    return {"cc": cc, "compiled": compiled, "probes_post": probes_post, "fidelity": fid}

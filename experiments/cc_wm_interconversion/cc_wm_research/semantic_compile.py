"""semantic_compile.py — CC→WM compile-back (SOTA): LoRA-fidelity + controlled compile.

Make-or-break #3: demonstrate compile-back is EFFECTIVE on a *compilable* target
(an observable-but-nonlinearly-encoded quantity, e.g. qpos²) while PRESERVING the WM
(frozen encoder/predictor, LoRA only on the projector = maximal fidelity).

Contrast with the demo's velocity case (architecturally uncompilable — unobservable
single-frame), which this module's "compilable vs uncompilable" probe distinguishes.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a low-rank trainable delta: y = Wx + B(Ax)."""
    def __init__(self, base: nn.Linear, r: int = 8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t())


def install_lora_projector(model, r: int = 8):
    """Freeze everything; replace projector MLP linears with LoRA. Returns trainable params."""
    model.requires_grad_(False)
    proj = model.projector
    replaced = []
    for i, layer in enumerate(proj.net):
        if isinstance(layer, nn.Linear):
            proj.net[i] = LoRALinear(layer, r=r)
            replaced.append(i)
    return [p for p in proj.parameters() if p.requires_grad], replaced


def _norm(arr):
    a = np.asarray(arr, float); mu, sd = a.mean(0), a.std(0) + 1e-6
    return (a - mu) / sd, mu, sd


def compile_controlled(model, data, target_fn, device="cuda",
                       epochs: int = 3, batch_size: int = 16, lr=5e-4, lam=1.0,
                       max_batches: int = 40, seed=0):
    """Force the latent z to linearly encode target_fn(state) via LoRA(projector)+aux head.

    target_fn: dict(state)->np.ndarray (N, d_target)  [the compiled constraint target]
    """
    import importlib, sys
    sys.path.insert(0, "E:/DATA/vscode/cc_wm_demo")
    from wm import prepare_pixels  # reuse demo preprocessing

    rng = np.random.default_rng(seed)
    target = target_fn(data)
    tgt, mu, sd = _norm(target)
    head = nn.Linear(192, tgt.shape[1]).to(device)
    lora_params, replaced = install_lora_projector(model, r=8)
    opt = torch.optim.AdamW(list(lora_params) + list(head.parameters()), lr=lr, weight_decay=1e-3)
    model.train()
    pixels = data["pixels"]; N = len(pixels)
    hist = {"aux": []}
    for ep in range(epochs):
        order = rng.permutation(N - 2); bi = 0
        for bs in range(0, len(order), batch_size):
            if bi >= max_batches: break
            idx = order[bs:bs + batch_size]
            pix = prepare_pixels(pixels[idx], device)
            info = {"pixels": pix.unsqueeze(1)}
            info = model.encode(info)
            z = info["emb"][:, 0, :]
            tgt_b = torch.tensor(tgt[idx], dtype=torch.float32, device=device)
            aux = ((head(z) - tgt_b) ** 2).mean()
            opt.zero_grad(); aux.backward(); opt.step()
            hist["aux"].append(float(aux.item())); bi += 1
        print(f"[compile] epoch {ep+1}/{epochs}  aux={np.mean(hist['aux'][-max_batches:]):.4f}")
    model.eval()
    return model, head, {"aux_final": float(np.mean(hist["aux"][-10:])), "lora_layers": replaced,
                         "target_mean": mu.tolist(), "target_std": sd.tolist()}


def nonlinear_targets(state):
    """Compilable targets: high-variance NONLINEAR functions of OBSERVABLE state.
    sin(5·qpos) is observable (derivable from the pixel) but NOT linearly decodable from z
    (z linearly encodes qpos, not sin(5·qpos)) → genuine headroom for a compilable fix,
    unlike qpos² (low variance, trivially high R²) or velocity (unobservable single-frame)."""
    q = np.asarray(state["qpos"], float)
    return np.sin(5.0 * q)   # (N,2): high-variance, genuinely nonlinear in the encoded qpos


def velocity_targets(state):
    """UNCOMPILABLE target: qvel is NOT observable from a single frame (not in the pixel),
    so no encoder/projector change can make z decode it → aux loss stays high (negative R²).
    Contrast with nonlinear_targets (compilable, observable)."""
    return np.asarray(state["qvel"], float)

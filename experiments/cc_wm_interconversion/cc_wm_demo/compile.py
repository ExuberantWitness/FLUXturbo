"""compile.py — CC→WM compile-back: short fine-tune injecting CC-derived constraints.

The extracted CC said: "latent encodes qpos (R² high) but NOT qvel (bottleneck)".
Compile that back as auxiliary losses:
  - preserve: a linear head z→qpos (keep the latent decoding qpos)  [fidelity]
  - fix:      a linear head z→qvel (force the latent to encode qvel) [resolve bottleneck]
Fine-tune the encoder/projector (+ heads) with JEPA pred_loss + λ·aux → WM'.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import config as C
from wm import prepare_pixels


def _norm(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr).all(1)] if arr.ndim == 2 else arr[np.isfinite(arr)]
    mean, std = arr.mean(0), arr.std(0) + 1e-6
    return (arr - mean) / std, mean, std


def fine_tune(model, data: dict, device: str = C.DEVICE, epochs: int = 3,
              batch_size: int = 64, lr: float = 5e-5, lam_aux: float = 1.0,
              fix_targets=("qvel",), preserve_targets=("qpos",),
              use_pred_loss: bool = True, max_batches: int = 40, seed: int = 0):
    """Returns (model', heads, stats). Mutates model in place (enables grad)."""
    rng = np.random.default_rng(seed)
    D = C.EMBED_DIM
    # normalized targets
    norm = {}
    for t in set(fix_targets) | set(preserve_targets):
        if t in data:
            n, mean, std = _norm(data[t])
            norm[t] = (torch.tensor(n, device=device), n.shape[1])
    heads = {t: nn.Linear(D, dim).to(device) for t, (_, dim) in norm.items()}

    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    for h in heads.values():
        params += list(h.parameters())
    if use_pred_loss:
        params += list(model.predictor.parameters()) + list(model.pred_proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)

    model.train()
    model.requires_grad_(True)
    pixels = data["pixels"]
    N = len(pixels)
    has_action = "action" in data
    hist = {"loss": [], "pred": [], "aux": []}

    for ep in range(epochs):
        order = rng.permutation(N - 4)
        bi = 0
        for bstart in range(0, len(order), batch_size):
            if bi >= max_batches:
                break
            idx = order[bstart:bstart + batch_size]
            # consecutive windows of length 4 starting at idx
            wins = np.stack([np.arange(i, i + 4) for i in idx])   # (B,4)
            pix = prepare_pixels(pixels[wins].reshape(-1, *pixels.shape[1:]), device)  # (B*4,3,224,224)
            info = {"pixels": pix.view(len(idx), 4, *pix.shape[1:])}
            if has_action:
                act = torch.tensor(data["action"][wins], dtype=torch.float32, device=device)
                act = torch.nan_to_num(act, 0.0)
                info["action"] = act
            info = model.encode(info)
            emb = info["emb"]          # (B,4,D)
            z_cur = emb[:, 2, :]        # use the 3rd frame's latent for aux decode (B,D)

            aux = torch.tensor(0.0, device=device)
            for t, head in heads.items():
                tgt, _ = norm[t]
                tgt_b = tgt[wins[:, 2]]            # target at the decoded frame
                aux = aux + ((head(z_cur) - tgt_b) ** 2).mean()

            pred_l = torch.tensor(0.0, device=device)
            if use_pred_loss and has_action:
                act_emb = info.get("act_emb")
                ctx_emb = emb[:, :3]
                ctx_act = act_emb[:, :3] if act_emb is not None else None
                pred = model.predict(ctx_emb, ctx_act)
                pred = model.pred_proj(pred.reshape(-1, D)).reshape(len(idx), 3, D)
                pred_l = ((pred[:, -1, :] - emb[:, 3, :].detach()) ** 2).mean()

            loss = pred_l + lam_aux * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist["loss"].append(float(loss.item()))
            hist["pred"].append(float(pred_l.item()))
            hist["aux"].append(float(aux.item()))
            bi += 1
        print(f"[compile] epoch {ep+1}/{epochs}  loss={np.mean(hist['loss'][-max_batches:]):.4f} "
              f"pred={np.mean(hist['pred'][-max_batches:]):.4f} aux={np.mean(hist['aux'][-max_batches:]):.4f}")

    model.eval()
    return model, heads, {"target_dims": {t: int(norm[t][1]) for t in norm}, "history": hist}

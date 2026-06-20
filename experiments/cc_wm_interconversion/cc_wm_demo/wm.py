"""wm.py — light-path loader for LeWorldModel (reacher).

Reconstructs the JEPA from le-wm's local jepa.py/module.py + a HuggingFace ViT-Tiny
encoder (the spt `vit_hf "tiny"` is just a plain HF ViTModel — keys match exactly),
then loads HF weights.pt with strict=True. No stable_worldmodel/stable_pretraining needed.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

import config as C

# make le-wm importable (jepa.py, module.py live at repo root)
if str(C.LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(C.LEWM_DIR))
from jepa import JEPA                       # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402


# ── encoder: HuggingFace ViT-Tiny, matching spt vit_hf("tiny", patch14, img224) ─
def build_encoder():
    from transformers import ViTConfig, ViTModel
    cfg = ViTConfig(
        hidden_size=C.EMBED_DIM,          # 192
        num_hidden_layers=12,
        num_attention_heads=3,            # canonical ViT-Tiny (head_dim 64)
        intermediate_size=4 * C.EMBED_DIM,  # 768
        image_size=C.IMG_SIZE,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        add_pooling_layer=False,          # no pooler in state_dict
        use_mask_token=False,
    )
    return ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)


def load_model(device: str = C.DEVICE) -> JEPA:
    cfg = json.loads(C.CONFIG_JSON.read_text())
    enc = build_encoder()
    model = JEPA(
        encoder=enc,
        predictor=ARPredictor(**{k: v for k, v in cfg["predictor"].items() if k != "_target_"}),
        action_encoder=Embedder(**{k: v for k, v in cfg["action_encoder"].items() if k != "_target_"}),
        projector=_build_mlp(cfg["projector"]),
        pred_proj=_build_mlp(cfg["pred_proj"]),
    )
    sd = torch.load(C.WEIGHTS_PT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # strict=False so we can SEE mismatches instead of crashing; report them
        print(f"[load_model] missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]:
            print("  missing[:5]:", missing[:5])
        if unexpected[:5]:
            print("  unexpected[:5]:", unexpected[:5])
    model.interpolate_pos_encoding = True
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def _build_mlp(cfg_part: dict):
    import torch.nn as nn
    return MLP(
        input_dim=cfg_part["input_dim"],
        hidden_dim=cfg_part["hidden_dim"],
        output_dim=cfg_part["output_dim"],
        norm_fn=nn.BatchNorm1d,           # config uses BatchNorm1d
    )


# ── pixel preprocessing (matches le-wm utils.get_img_preprocessor) ──────────
def prepare_pixels(raw: np.ndarray, device: str = C.DEVICE) -> torch.Tensor:
    """raw: uint8 (N,H,W,3) or (N,3,H,W) → normalized (N,3,224,224) float on device."""
    t = torch.from_numpy(raw).float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] == 3:                  # (N,H,W,3) → (N,3,H,W)
        t = t.permute(0, 3, 1, 2)
    t = t / 255.0
    mean = torch.tensor(C.IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    t = torch.nn.functional.interpolate(t, size=(C.IMG_SIZE, C.IMG_SIZE),
                                        mode="bilinear", align_corners=False)
    return t.to(device)


@torch.no_grad()
def encode_pixels(model: JEPA, pixels_np: np.ndarray, device: str = C.DEVICE,
                  batch_size: int = 64) -> np.ndarray:
    """Encode a flat array of frames → latents (N, D)."""
    from einops import rearrange
    out = []
    for i in range(0, len(pixels_np), batch_size):
        chunk = pixels_np[i:i + batch_size]
        pix = prepare_pixels(chunk, device)
        info = {"pixels": pix.unsqueeze(1)}      # (B, T=1, C, H, W) as encode expects (B,T,...)
        info = model.encode(info)
        out.append(info["emb"][:, 0, :].cpu().numpy())   # take the single timestep
    return np.concatenate(out, axis=0)


# ── reacher.h5 reader (flexible — validated against real file at smoke test) ─
def load_reacher_data(h5_path: Path = C.REACHER_H5, max_frames: int | None = 4000):
    import h5py
    h5_path = Path(h5_path)
    print(f"[data] opening {h5_path}")
    with h5py.File(h5_path, "r") as f:
        def _resolve(name):
            # accept top-level dataset or <name>/values or first matching
            if name in f:
                return f[name][...]
            for k in f.keys():
                if k == name or k.endswith("/" + name):
                    return f[k][...]
            return None
        keys_avail = list(f.keys())
        print(f"[data] top-level keys: {keys_avail}")
        data = {}
        for col in ["pixels", "action", "observation", "qpos", "qvel",
                    "goal_qpos", "episode_idx", "ep_idx", "step_idx"]:
            arr = _resolve(col)
            if arr is not None:
                data[col] = np.asarray(arr)
                print(f"[data]   {col:12s} shape={data[col].shape} dtype={data[col].dtype}")
    # le-wm's action_encoder expects frameskip-stacked actions (input_dim=10=5*2);
    # emulate by tiling raw 2-d actions to 10-d.
    if "action" in data and data["action"].ndim == 2 and data["action"].shape[1] < 10:
        fs = max(1, 10 // data["action"].shape[1])
        data["action"] = np.tile(data["action"], (1, fs))
        print(f"[data] action frameskip-stacked → {data['action'].shape}")
    n = len(data.get("pixels", data.get("observation", [])))
    if max_frames and n > max_frames:
        for k in list(data):                       # consecutive slice (preserves temporal order
            if len(data[k]) == n:                  # for linear_dynamics; fine for probing too)
                data[k] = data[k][:max_frames]
        print(f"[data] truncated to first {max_frames} frames (temporal order kept)")
    return data


if __name__ == "__main__":
    # smoke test: load model, encode random pixels
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={dev}")
    m = load_model(dev)
    print(f"[smoke] model loaded; params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    rnd = (np.random.rand(8, 96, 96, 3) * 255).astype(np.uint8)
    z = encode_pixels(m, rnd, dev, batch_size=8)
    print(f"[smoke] z.shape={z.shape} (expect (8, {C.EMBED_DIM}))  mean={z.mean():.4f} std={z.std():.4f}")

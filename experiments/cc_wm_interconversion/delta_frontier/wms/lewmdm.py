"""wms/lewmdm.py — LeWorldModel (JEPA) adapter, wrapping cc_wm_demo/wm.py.

Implements the WorldModel protocol for le-wm (reacher for Phase A; cube/pusht/tworooms
need their own checkpoints in Phase C). Reuses the validated light-path loader + dm_control
data generation from cc_wm_demo.
"""
from __future__ import annotations
import numpy as np
import torch
import paths  # sets sys.path (cc_wm_demo first)
import wm as demo_wm          # cc_wm_demo/wm.py
from paths import DEMO_DIR


class LeWMAdapter:
    def __init__(self, env: str = "reacher", device: str = "cuda"):
        self.name = f"LeWM-{env}"
        self.env = env
        self.device = device
        self.model = demo_wm.load_model(device)
        self.latent_dim = 192

    # ── perception ────────────────────────────────────────────────────
    def encode(self, obs: np.ndarray) -> np.ndarray:
        return demo_wm.encode_pixels(self.model, obs, self.device)

    @torch.no_grad()
    def predict_next_latent(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """z (N,D), action (N,A_stacked) → predicted next-z (N,D) via JEPA.predict."""
        m = self.model
        zt = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(1)      # (N,1,D)
        at = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(1)  # (N,1,A)
        info = {"emb": zt}
        info["act_emb"] = m.action_encoder(at)
        pred = m.predict(zt, info["act_emb"])[:, 0, :]                                   # (N,D)
        return pred.cpu().numpy()

    # ── data ──────────────────────────────────────────────────────────
    def sample_transitions(self, env_spec: str = "reacher", n_frames: int = 2500, seed: int = 0):
        h5 = DEMO_DIR / "data" / "reacher_local.h5"
        if not h5.exists():
            raise FileNotFoundError(f"{h5} missing — run cc_wm_demo/gen_data.py first")
        return demo_wm.load_reacher_data(h5, max_frames=n_frames)

    def probe_targets(self, data: dict) -> dict:
        return {k: np.asarray(v) for k, v in data.items()
                if k in ("qpos", "qvel", "observation")}

    # ── cost axis ─────────────────────────────────────────────────────
    def cost(self) -> dict:
        return {"params_M": sum(p.numel() for p in self.model.parameters()) / 1e6}

    @property
    def module(self):
        return self.model

"""wms/base.py — the WorldModel protocol (the universality abstraction).

Every WM in the battery (le-wm, Newton, DreamerV3, TD-MPC2, Othello-GPT, FLOWVLM)
implements this interface, so the δ-pipeline and frontier-fit are WM-agnostic.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class WorldModel(Protocol):
    """A world model + its data interface for δ-frontier measurement."""

    name: str
    latent_dim: int
    device: str

    # ── perception ────────────────────────────────────────────────────
    def encode(self, obs: np.ndarray) -> np.ndarray:
        """observations (N, ...) → latents z (N, D)."""
        ...

    def predict_next_latent(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """latent rollout z_t → ẑ_{t+1} (for δ_rt behavioral reconstruction). Optional."""
        ...

    # ── data ──────────────────────────────────────────────────────────
    def sample_transitions(self, env_spec: str, n_frames: int, seed: int = 0) -> dict:
        """Collect transitions + ground-truth physical quantities.
        Returns dict with at least: 'pixels', 'action', and probe-target columns
        (e.g. qpos, qvel, observation)."""
        ...

    def probe_targets(self, data: dict) -> dict:
        """{column: (N,k) array} of ground-truth quantities to probe δ against."""
        ...

    # ── cost axis (c) ─────────────────────────────────────────────────
    def cost(self) -> dict:
        """{params_M, encode_fps, ...} for the frontier cost axis."""
        ...

    # ── the underlying module (for compile / in-model intervention) ───
    @property
    def module(self):
        """the torch nn.Module (encoder/predictor) — used by compile & pyvene hooks."""
        ...

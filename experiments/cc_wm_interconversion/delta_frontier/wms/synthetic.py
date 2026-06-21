"""wms/synthetic.py — constructive δ=0 anchor (Phase B / C1).

A synthetic LINEAR world model where the latent z IS the state and the dynamics z_{t+1}=A·z_t
are exactly the symbolic claim. By construction every component is perfectly/causally encoded
(swapping z's i-th direction swaps the decoded i-th coordinate) → δ_iia ≈ 0 for all claims.

This validates the EXACTNESS upper end of δ (the δ=0 regime) fast, without Newton.
"""
from __future__ import annotations
import numpy as np
import paths  # noqa
from wms.base import WorldModel  # noqa (protocol only)


class SyntheticLinearWM:
    """z = state (identity encoder); linear dynamics; δ should be ≈0 by construction."""

    def __init__(self, dim: int = 6, device: str = "cpu", seed: int = 0):
        self.name = f"SyntheticLinear-d{dim}"
        self.env = "linear"
        self.device = device
        self.latent_dim = dim
        rng = np.random.default_rng(seed)
        # stable A (spectral radius < 1) so rollouts don't blow up
        A = rng.standard_normal((dim, dim)) * 0.3
        ev = np.linalg.eigvals(A); sr = np.max(np.abs(ev))
        if sr > 0.9:
            A = A * (0.9 / sr)
        self.A = A
        self._states = None

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, float)  # identity: z = state

    def predict_next_latent(self, z, action=None):
        return np.asarray(z, float) @ self.A.T

    def sample_transitions(self, env_spec: str = "linear", n_frames: int = 2500, seed: int = 0):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_frames, self.latent_dim))
        # roll forward one step so 'pixels' (states) and 'next' are consistent with A
        z_next = z @ self.A.T + 0.0 * rng.standard_normal(z.shape)  # deterministic-ish
        self._states = z
        return {"pixels": z, "state": z, "state_next": z_next}

    def probe_targets(self, data: dict) -> dict:
        s = np.asarray(data["state"])
        return {f"z[{i}]": s[:, i:i + 1] for i in range(self.latent_dim)}

    def cost(self) -> dict:
        return {"params_M": 0.0, "synthetic": True}

    @property
    def module(self):
        return None


class SyntheticLossyWM:
    """C2 anchor: z = state + Gaussian noise ⇒ each component encoded but imperfectly
    (R²<1) → δ small (>0 but well below the unobservable regime). Middle of three regimes."""

    def __init__(self, dim: int = 6, noise_std: float = 0.25, device: str = "cpu", seed: int = 0):
        self.name = f"SyntheticLossy-d{dim}-σ{noise_std}"
        self.env = "lossy"
        self.device = device
        self.latent_dim = dim
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((dim, dim)) * 0.3
        ev = np.linalg.eigvals(A); sr = np.max(np.abs(ev))
        if sr > 0.9:
            A = A * (0.9 / sr)
        self.A = A

    def encode(self, obs):
        rng = np.random.default_rng(1)
        return np.asarray(obs, float) + self.noise_std * rng.standard_normal(np.asarray(obs).shape)

    def predict_next_latent(self, z, action=None):
        return np.asarray(z, float) @ self.A.T

    def sample_transitions(self, env_spec="lossy", n_frames=2500, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_frames, self.latent_dim))
        return {"pixels": z, "state": z, "state_next": z @ self.A.T}

    def probe_targets(self, data):
        s = np.asarray(data["state"])
        return {f"z[{i}]": s[:, i:i + 1] for i in range(self.latent_dim)}

    def cost(self):
        return {"params_M": 0.0, "synthetic": True, "noise_std": self.noise_std}

    @property
    def module(self):
        return None


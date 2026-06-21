"""wms/noisy.py — wraps a base WM, adding Gaussian noise to its latent z.

Lets us sweep fidelity on a REAL world model's representation space (e.g. le-wm) without
retraining — demonstrating the δ-frontier law holds starting from a real WM's latents.
"""
from __future__ import annotations
import numpy as np


class NoisyWrapper:
    def __init__(self, base_wm, noise_std: float, seed: int = 0):
        self.base = base_wm
        self.noise_std = noise_std
        self.name = f"{base_wm.name}+noiseσ{noise_std}"
        self.latent_dim = base_wm.latent_dim
        self.device = base_wm.device
        self._rng = np.random.default_rng(seed)

    def encode(self, obs):
        z = self.base.encode(obs)
        if self.noise_std > 0:
            z = z + self.noise_std * self._rng.standard_normal(z.shape)
        return z

    def predict_next_latent(self, z, action=None):
        return self.base.predict_next_latent(z, action)

    def sample_transitions(self, *a, **k):
        return self.base.sample_transitions(*a, **k)

    def probe_targets(self, data):
        return self.base.probe_targets(data)

    def cost(self):
        c = self.base.cost(); c["noise_std"] = self.noise_std; return c

    @property
    def module(self):
        return self.base.module

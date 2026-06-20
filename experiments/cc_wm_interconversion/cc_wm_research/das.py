"""das.py — Distributed Alignment Search / interchange intervention for JEPA latents.

Upgrades "z decodes q (R²)" → "z CAUSALLY encodes q": find the q-relevant subspace,
swap it between two samples, and check the decoded q follows the swap (IIA), with a
null-direction control. Geiger et al. causal-abstraction lineage (arXiv:2303.02536).

A clean, defensible linear-subspace DAS:
  - direction w = normalized Ridge-probe weight (z→q).
  - swap: z_a' = z_a with its projection on w replaced by z_b's projection.
  - IIA = R²(decoded q from z_a' , q_b)   [should be HIGH if causal]
  - null: same swap on a random unit direction → IIA_null ≈ 0.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def probe_direction(z, q, seed=0):
    """Fit Ridge z→q (standardized); return (direction w unit-norm, probe on standardized z, scaler)."""
    z = np.asarray(z, float); q = np.asarray(q, float).ravel()
    m = np.isfinite(z).all(1) & np.isfinite(q); z, q = z[m], q[m]
    zsc = StandardScaler().fit_transform(z)
    qn = (q - q.mean()) / (q.std() + 1e-12)
    probe = Ridge(alpha=1.0).fit(zsc, qn)
    w = probe.coef_
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, probe, zsc, qn


def interchange_intervention(z, q, w, zsc, qn, probe, n_pairs=500, seed=0):
    """Swap the w-subspace between random pairs; measure IIA + null-direction control."""
    rng = np.random.default_rng(seed)
    n = len(zsc); k = min(n_pairs, n)
    idx_a = rng.choice(n, k, replace=False); idx_b = rng.choice(n, k, replace=False)
    za, zb = zsc[idx_a], zsc[idx_b]
    qa, qb = qn[idx_a], qn[idx_b]
    # project on w (scalar per sample)
    pa = za @ w; pb = zb @ w
    za_swap = za + (pb - pa)[:, None] * w[None, :]            # replace a's w-projection with b's
    dec_swap = probe.predict(za_swap)
    iia = _r2(qb, dec_swap)                                    # decoded should follow b's q
    # null: random direction of equal norm
    v = rng.standard_normal(len(w)); v /= (np.linalg.norm(v) + 1e-12)
    pa_n = za @ v; pb_n = zb @ v
    za_swap_n = za + (pb_n - pa_n)[:, None] * v[None, :]
    dec_swap_n = probe.predict(za_swap_n)
    iia_null = _r2(qb, dec_swap_n)                             # should be ≈0 (no causal effect)
    return {"iia": float(iia), "iia_null": float(iia_null),
            "causal_effect": float(iia - iia_null)}


def das_verify(z, q, name, seed=0, n_pairs=500):
    """Full DAS causal check for one scalar target. Returns the causal-verification record."""
    try:
        w, probe, zsc, qn = probe_direction(z, q, seed)
        res = interchange_intervention(z, q, w, zsc, qn, probe, n_pairs=n_pairs, seed=seed)
    except Exception as e:
        res = {"iia": float("nan"), "iia_null": float("nan"), "causal_effect": float("nan"),
               "error": str(e)[:80]}
    res["name"] = name
    res["causal"] = bool(res.get("causal_effect", 0) > 0.3)    # IIA ≫ null → causally encoded
    return res

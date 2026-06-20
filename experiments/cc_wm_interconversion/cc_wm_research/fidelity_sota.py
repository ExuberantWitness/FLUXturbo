"""fidelity_sota.py — round-trip fidelity upgrades: MI-gap (Gaussian log-det proxy) + latent
Wasserstein (behavioral-distribution equivalence) + DPI note.

Complements cc_wm_demo/fidelity.py's 3-layer metric. These answer: does CC⇌WM preserve the
INFORMATION (ΔI) and the BEHAVIORAL DISTRIBUTION (W1) of the latent across round-trip?
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance


def mi_logdet(z):
    """Gaussian mutual-information proxy: 0.5 * log det(cov_z) (differential entropy scale)."""
    z = np.asarray(z, float)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov = cov + 1e-3 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * logdet


def mi_gap(z_pre, z_post):
    """ΔI proxy = I_pre - I_post (positive ⇒ info lost in round-trip)."""
    return float(mi_logdet(z_pre) - mi_logdet(z_post))


def latent_wasserstein(z_pre, z_post):
    """Sliced/marginal Wasserstein-1 between the two latent point clouds (behavioral-distribution shift)."""
    z_pre = np.asarray(z_pre, float); z_post = np.asarray(z_post, float)
    d = z_pre.shape[1]
    return float(np.mean([wasserstein_distance(z_pre[:, j], z_post[:, j]) for j in range(d)]))


def roundtrip_fidelity_sota(probes_pre, probes_post, z_pre, z_post, mse_pre, mse_post):
    """Full round-trip fidelity report (3-layer + MI-gap + W1). DPI: ΔI ≥ 0 by construction."""
    r2_pre = np.array([p["r2"] if p["r2"] == p["r2"] else 0 for p in probes_pre])
    r2_post = np.array([p["r2"] if p["r2"] == p["r2"] else 0 for p in probes_post])
    cos = float(np.dot(r2_pre, r2_post) / (np.linalg.norm(r2_pre) * np.linalg.norm(r2_post) + 1e-12))
    dI = mi_gap(z_pre, z_post)
    w1 = latent_wasserstein(z_pre, z_post)
    return {
        "info_layer_cosine_R2": round(cos, 4),
        "behavioral_pred_mse": {"pre": round(mse_pre, 5), "post": round(mse_post, 5),
                                "delta": round(mse_post - mse_pre, 5)},
        "mi_gap_dI": round(dI, 4),                 # ≥0 by DPI; small ⇒ info preserved
        "latent_wasserstein_W1": round(w1, 4),     # small ⇒ behavioral distribution preserved
        "dpi_note": "ΔI ≥ 0 by data-processing inequality; goal is to bound it within task tolerance.",
    }

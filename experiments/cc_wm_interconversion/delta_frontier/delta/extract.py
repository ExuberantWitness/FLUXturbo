"""delta/extract.py — WM→CC analysis: probe + (pysindy) latent-dynamics mechanism.

Returns the analysis dict consumed by cc_wm_demo/cc.build_cc. Mechanism extraction uses
pysindy sparse-ODE on (z, ż) if available, else falls back to linear ż=Az lstsq.
"""
from __future__ import annotations
import numpy as np
import paths  # noqa
from delta.probe import probe_all, probe_temporal, pca_components


def _linear_dynamics(z):
    """fallback: ż = A·z lstsq (from cc_wm_demo/extract.linear_dynamics)."""
    z = np.asarray(z, float)
    if len(z) < 100:
        return {"A_shape": None, "r2_linear": 0.0}
    X, Y = z[:-1], z[1:]
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ A
    ss_res = ((Y - pred) ** 2).sum(); ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    r2 = float((1 - ss_res / ss_tot).mean())
    return {"A_shape": list(A.shape), "r2_linear": r2, "method": "linear_lstsq"}


def _sindy_dynamics(z):
    """pysindy sparse ODE on PCA-reduced z (≤8 dims) — full 192-d poly² (≈18.7k feats) hangs STLSQ.
    Returns {equation_complexity, r2} on the dominant subspace."""
    try:
        import pysindy as ps
        from sklearn.decomposition import PCA
    except Exception:
        return None
    z = np.asarray(z, float)
    if len(z) < 200:
        return None
    try:
        k = min(8, z.shape[1])
        zr = PCA(n_components=k, random_state=0).fit_transform(z)   # (N, ≤8)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.05),
                         feature_library=ps.PolynomialLibrary(degree=2),
                         differentiation_method=ps.FiniteDifference())
        model.fit(zr, t=1)
        c = model.coefficients()
        r2 = model.score(zr)
        n_active = int((np.abs(c) > 1e-6).sum())
        return {"method": f"pysindy_STLSQ_poly2_on_PCA{k}", "n_active_terms": n_active,
                "r2_sindy": float(r2), "complexity": n_active}
    except Exception:
        return None


def analyze(z: np.ndarray, state: dict, seed: int = 0) -> dict:
    probes = probe_all(z, state, seed)
    temporal = []
    for col, arr in state.items():
        if col in ("episode_idx", "step_idx", "action"):
            continue
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == z.shape[0]:
            for j in range(a.shape[1]):
                temporal.append(probe_temporal(z, a[:, j], f"{col}[{j}]", 500 + j))
    sindy = _sindy_dynamics(z)
    dynamics = sindy if sindy else _linear_dynamics(z)
    return {"probes": probes, "temporal_probes": temporal,
            "pca": pca_components(z), "dynamics": dynamics}

"""extract.py — WM→CC analysis: probe latents for physical quantities (+ components/mechanism/boundary).

Returns plain dicts (no ccchain types here); cc.py maps them to Atom/Edge/Rho.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def probe_quantity(z: np.ndarray, y: np.ndarray, name: str, seed: int) -> dict:
    """Linear probe: how well does latent z linearly encode scalar y?
    Returns {name, r2 (test), ctrl_r2 (shuffled-y control), selectivity, n}."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(z).all(1) & np.isfinite(y)
    z, y = z[mask], y[mask]
    if len(y) < 50 or y.std() < 1e-8:
        return {"name": name, "r2": float("nan"), "ctrl_r2": float("nan"),
                "selectivity": 0.0, "n": int(len(y))}
    z = StandardScaler().fit_transform(z)
    y = (y - y.mean()) / (y.std() + 1e-12)
    # real probe
    z_tr, z_te, y_tr, y_te = train_test_split(z, y, test_size=0.2, random_state=seed)
    r2 = _r2(y_te, Ridge(alpha=1.0).fit(z_tr, y_tr).predict(z_te))
    # control probe: shuffled target (Hewitt&Liang control task) — split aligned
    y_ctrl = np.random.default_rng(seed + 1).permutation(y)
    zc_tr, zc_te, yc_tr, yc_te = train_test_split(z, y_ctrl, test_size=0.2, random_state=seed)
    ctrl_r2 = _r2(yc_te, Ridge(alpha=1.0).fit(zc_tr, yc_tr).predict(zc_te))
    return {"name": name, "r2": float(r2), "ctrl_r2": float(ctrl_r2),
            "selectivity": float(r2 - ctrl_r2), "n": int(len(y))}


def probe_temporal(z: np.ndarray, y: np.ndarray, name: str, seed: int) -> dict:
    """Probe scalar y[t] from a 2-frame temporal feature [z[t-1], z[t]].

    Single-frame z cannot encode velocity (unobservable in one image); a 2-frame
    context can (velocity ≈ position difference). Contrast with probe_quantity.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(z), len(y)) - 1
    feat = np.concatenate([z[:n], z[1:n + 1]], axis=1)   # (n, 2D)
    return probe_quantity(feat, y[1:n + 1], name, seed)


def probe_all(z: np.ndarray, state: dict, seed: int = 0) -> list[dict]:
    """Probe every scalar component of every state array in `state`."""
    results = []
    s = seed
    for col, arr in state.items():
        if col in ("pixels", "episode_idx", "ep_idx", "step_idx", "_subset_idx", "action"):
            continue
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[0] != z.shape[0]:
            continue
        for j in range(a.shape[1]):
            results.append(probe_quantity(z, a[:, j], f"{col}[{j}]", s))
            s += 1
    return results


def pca_components(z: np.ndarray, k: int = 6) -> dict:
    """Find low-dim subspaces (component atoms) via PCA + k-means on PCA coords."""
    z = StandardScaler().fit_transform(np.asarray(z, dtype=np.float64))
    n_comp = min(k, z.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(z)
    coords = pca.transform(z)
    km = KMeans(n_clusters=min(k, n_comp), n_init=4, random_state=0).fit(coords.T)  # cluster dimensions
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "n_components": int(n_comp),
        "coords": coords,
    }


def linear_dynamics(z_seq: np.ndarray) -> dict:
    """Fit ż = A·z on consecutive latent pairs → mechanism (eigenvalues)."""
    z = np.asarray(z_seq, dtype=np.float64)
    if len(z) < 100:
        return {"A_shape": None, "eig": []}
    X, Y = z[:-1], z[1:]                      # (N-1, D), (N-1, D)
    # least squares A = pinv(X) @ Y
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (D, D)
    eig = np.linalg.eigvals(A)
    # one-step linear prediction R² (mean over dims) as mechanism confidence
    pred = X @ A
    r2_lin = float(np.mean([_r2(Y[:, d], pred[:, d]) for d in range(Y.shape[1])]))
    return {"A_shape": list(A.shape), "eig_top": [complex(round(e.real, 3), round(e.imag, 3)) for e in eig[np.argsort(-np.abs(eig))[:5]]],
            "r2_linear": r2_lin}

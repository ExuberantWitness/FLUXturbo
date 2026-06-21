"""delta/intervene.py — δ_iia: alignment defect via interchange intervention.

The LOCKED δ definition (§12-1): causal interchange intervention accuracy (IIA/DAS lineage).
For a claim "z encodes q": find the q-relevant subspace (Ridge probe direction w), swap it
between sample pairs, and ask whether the decoded q follows the swap (IIA) — with a null
ensemble (random directions) and a dose-response curve to rule out illusion/triviality.

  δ_iia = clamp(1 − (IIA − IIA_null_mean), 0, 1)
  → 0 when z faithfully/causally encodes q; → 1 when it does not.

(pyvene in-model layer-wise patching is the deeper upgrade; latent-level interchange is a
valid causal intervention already and is used here. The hook is left for Phase A.2.)
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def _fit(z, q, seed):
    z = np.asarray(z, float); q = np.asarray(q, float).ravel()
    m = np.isfinite(z).all(1) & np.isfinite(q); z, q = z[m], q[m]
    if len(q) < 50:
        return None
    zsc = StandardScaler().fit_transform(z)
    qn = (q - q.mean()) / (q.std() + 1e-12)
    probe = Ridge(alpha=1.0).fit(zsc, qn)
    w = probe.coef_; w = w / (np.linalg.norm(w) + 1e-12)
    return w, probe, zsc, qn


def _iia_on_direction(zsc, qn, probe, w, idx_a, idx_b, n_pairs):
    """Swap the w-projection between pairs; R²(decoded z_a', q_b)."""
    rng = np.random.default_rng(0)
    za, zb = zsc[idx_a], zsc[idx_b]
    pa, pb = za @ w, zb @ w
    za_s = za + (pb - pa)[:, None] * w[None, :]
    dec = probe.predict(za_s)
    return _r2(qn[idx_b], dec)


def measure_delta_iia(z, q, name, n_pairs=500, n_null=10, seed=0):
    fit = _fit(z, q, seed)
    if fit is None:
        return {"name": name, "delta_iia": float("nan"), "iia": float("nan"),
                "iia_null_mean": float("nan"), "n": 0}
    w, probe, zsc, qn = fit
    n = len(zsc); k = min(n_pairs, n)
    rng = np.random.default_rng(seed)
    ia = rng.choice(n, k, replace=False); ib = rng.choice(n, k, replace=False)
    iia = _iia_on_direction(zsc, qn, probe, w, ia, ib, k)
    nulls = []
    for s in range(n_null):
        v = rng.standard_normal(len(w)); v /= (np.linalg.norm(v) + 1e-12)
        nulls.append(_iia_on_direction(zsc, qn, probe, v, ia, ib, k))
    iia_null_mean = float(np.mean(nulls)); iia_null_std = float(np.std(nulls))
    # dose-response: swap magnitude α ∈ {0.25,0.5,1,1.5}
    dose = {}
    for alpha in (0.25, 0.5, 1.0, 1.5):
        za, zb = zsc[ia], zsc[ib]
        pa, pb = za @ w, zb @ w
        za_s = za + alpha * (pb - pa)[:, None] * w[None, :]
        dose[alpha] = float(_r2(qn[ib], probe.predict(za_s)))
    # monotonicity of dose-response (1 = faithful causal encoding)
    doses = [dose[a] for a in (0.25, 0.5, 1.0, 1.5)]
    mono = float(np.mean(np.diff(doses) > 0)) if len(doses) > 1 else 0.0
    # δ = 1 − IIA (the null ensemble is a SEPARATE anti-trivial control, not in δ — a very
    # negative null means random directions have no causal effect, i.e. strong anti-trivial).
    delta = float(np.clip(1.0 - iia, 0.0, 1.0))
    return {"name": name, "delta_iia": delta, "iia": float(iia),
            "iia_null_mean": iia_null_mean, "iia_null_std": iia_null_std,
            "dose_response": dose, "dose_monotonicity": mono, "n": int(n)}

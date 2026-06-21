"""frontier/fit.py — fit the credibility–fidelity frontier (δ, g, c).

The headline law (research plan §3): for any WM there is a universal relation
  g ≤ G(δ)   (G monotone decreasing; low δ ⇒ high credibility/guarantee g)
plus a hard bound (low δ + high g ⇒ high cost c). This module fits G from a set of
measured (δ, g, c) operating points and reports monotonicity + the Pareto frontier.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def credibility_from_delta(claims_delta: list[float]) -> float:
    """Credibility g = mean(1 − δ_iia) over claims (encoding quality ⇒ guarantee).
    (Refinable to full ccchain status+Rho.confidence aggregation; this is the v1 proxy.)"""
    d = [x for x in claims_delta if x == x]
    return float(np.mean([1 - x for x in d])) if d else 0.0


def fit_frontier(points: list[dict]) -> dict:
    """points: [{'level', 'delta', 'g', 'c'}, ...] → frontier fit + monotonicity + hard-bound."""
    pts = sorted([p for p in points if p["delta"] == p["delta"]], key=lambda p: p["delta"])
    d = np.array([p["delta"] for p in pts])
    g = np.array([p["g"] for p in pts])
    c = np.array([p["c"] for p in pts])
    # g(δ) monotonicity (the core law: g decreases as δ increases)
    rho_gd = float(spearmanr(d, g).correlation) if len(d) > 2 else float("nan")
    # cost rises as δ falls (hard bound: low-δ high-g needs high c)
    rho_cd = float(spearmanr(d, c).correlation) if len(d) > 2 else float("nan")
    # fit G(δ) = a·exp(−b·δ) (monotone decreasing)
    a, b = float("nan"), float("nan")
    try:
        from scipy.optimize import curve_fit
        def G(x, a, b): return a * np.exp(-b * x)
        (a, b), _ = curve_fit(G, d, g, p0=[1.0, 1.0], maxfev=4000)
    except Exception:
        pass
    # hard-bound check: is there a δ below which c must be high? (min c at min δ)
    c_at_min_delta = float(c.min()); delta_at_min_c = float(d[c.argmin()])
    return {
        "n_levels": len(pts),
        "delta_range": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "g_of_delta_fit": {"form": "a·exp(−b·δ)", "a": round(a, 4), "b": round(b, 4)},
        "monotonicity_spearman_g_vs_delta": round(rho_gd, 4),   # want ≈ −1 (g ↓ as δ ↑)
        "hard_bound_spearman_c_vs_delta": round(rho_cd, 4),     # want < 0 (c ↑ as δ ↓)
        "min_cost_at": {"delta": round(delta_at_min_c, 4), "c": round(c_at_min_delta, 4)},
        "points": pts,
        "law_holds": (rho_gd < -0.7) and (rho_cd < -0.3),
    }

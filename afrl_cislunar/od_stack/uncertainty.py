"""
Uncertainty analysis for the results stage.

Everything here works from the contracts only (StateEstimate with a 6x6 covariance,
a propagator with `propagate(x0, t) -> Trajectory`), so it does not care which IOD,
estimator or propagator produced the state.

    propagate_covariance   6x6 covariance at the epoch -> covariance at any time
    confidence_ellipsoid   semi-axes of the X% confidence region (no truth needed)
    nees                   normalized estimation error squared (needs truth)
    consistency            "is the reported covariance honest?" checks (needs truth)

Units: SI (m, m/s) in, km / m/s in the dictionaries meant for people.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .contracts import StateEstimate, Trajectory


def _sqrt_psd(P: np.ndarray) -> np.ndarray:
    """Matrix square root L with L @ L.T == P. Cholesky when possible, otherwise
    eigen-decomposition with tiny negative eigenvalues clipped (covariances from
    least squares are often numerically just-barely indefinite)."""
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh((P + P.T) / 2)
        return V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))


def propagate_covariance(x0: StateEstimate, t: np.ndarray, propagator) -> np.ndarray:
    """Covariance at every time in t, shape (M, 6, 6), using symmetric sigma points.

    The 12 points x0 +/- sqrt(6) * (columns of sqrt(P)) are propagated with the SAME
    propagator the stack used, and the covariance is rebuilt from their spread. For a
    small covariance this equals the linear (state transition matrix) result; for a
    large one it also picks up some of the nonlinearity. No STM code is needed, so a
    new propagator gets covariance propagation for free.
    """
    if x0.cov is None:
        raise ValueError("State has no covariance")
    n = 6
    L = _sqrt_psd(np.asarray(x0.cov, float)) * np.sqrt(n)
    s0 = np.concatenate([x0.r, x0.v])
    t = np.asarray(t, float)
    devs = []
    for k in range(n):
        for sign in (+1.0, -1.0):
            s = s0 + sign * L[:, k]
            traj = propagator.propagate(StateEstimate(t=x0.t, r=s[:3], v=s[3:], method=x0.method), t)
            devs.append(np.hstack([traj.r, traj.v]))
    pts = np.stack(devs)                       # (12, M, 6)
    mean = pts.mean(axis=0)                    # (M, 6)
    d = pts - mean
    return np.einsum("kmi,kmj->mij", d, d) / (2 * n)


def sigma_position_m(P: np.ndarray) -> np.ndarray:
    """1-sigma position size sqrt(trace(P_rr)) for one (6,6) or many (M,6,6) covariances."""
    P = np.asarray(P)
    return np.sqrt(np.trace(P[..., :3, :3], axis1=-2, axis2=-1))


def confidence_ellipsoid(P_block: np.ndarray, level: float = 0.95):
    """Semi-axis lengths and directions of the `level` confidence ellipsoid of a
    k-dimensional Gaussian with covariance P_block (k = 2 or 3 typically).
    Returns (semi_axes sorted large -> small, axes as columns)."""
    k = P_block.shape[0]
    w, V = np.linalg.eigh((P_block + P_block.T) / 2)
    order = np.argsort(w)[::-1]
    scale = chi2.ppf(level, k)
    return np.sqrt(np.clip(w[order], 0, None) * scale), V[:, order]


def nees(err: np.ndarray, P: np.ndarray) -> float:
    """Normalized estimation error squared: err^T P^-1 err. If the covariance is honest
    and the error Gaussian, this follows a chi-square with len(err) degrees of freedom."""
    try:
        return float(err @ np.linalg.solve(P, err))
    except np.linalg.LinAlgError:
        return float("nan")


def nees_series(est: Trajectory, truth: Trajectory, P_t: np.ndarray) -> np.ndarray:
    """Position-only NEES (3 degrees of freedom) at every time."""
    e = est.r - truth.r
    return np.array([nees(e[i], P_t[i, :3, :3]) for i in range(len(e))])


def consistency(x0: StateEstimate, truth_r: np.ndarray, truth_v: np.ndarray) -> dict:
    """Is the error at the epoch compatible with the covariance the estimator reported?"""
    e = np.concatenate([x0.r - truth_r, x0.v - truth_v])
    P = np.asarray(x0.cov)
    n6, n3 = nees(e, P), nees(e[:3], P[:3, :3])
    sig = np.sqrt(np.clip(np.diag(P), 0, None))
    return {
        "nees_6d": n6,
        "nees_6d_p_value": float(chi2.sf(n6, 6)) if np.isfinite(n6) else None,
        "nees_position_3d": n3,
        "inside_95pct_ellipsoid_6d": bool(n6 <= chi2.ppf(0.95, 6)),
        "inside_95pct_ellipsoid_position": bool(n3 <= chi2.ppf(0.95, 3)),
        "max_component_error_in_sigma": float(np.max(np.abs(e) / np.where(sig > 0, sig, np.nan))),
    }


def confidence_summary(P: np.ndarray, level: float = 0.95) -> dict:
    """Numbers for summary.json: confidence-region sizes at the epoch (no truth needed)."""
    a_r, _ = confidence_ellipsoid(P[:3, :3], level)
    a_v, _ = confidence_ellipsoid(P[3:, 3:], level)
    return {
        "level": level,
        "position_semi_axes_km": [round(float(x) / 1e3, 3) for x in a_r],
        "velocity_semi_axes_m_s": [round(float(x), 5) for x in a_v],
        "position_1sigma_per_axis_km": [round(float(x) / 1e3, 3) for x in np.sqrt(np.diag(P)[:3])],
        "velocity_1sigma_per_axis_m_s": [round(float(x), 5) for x in np.sqrt(np.diag(P)[3:])],
    }

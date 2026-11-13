"""CR3BP equations of motion (nondimensional, rotating frame)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

TOL = dict(method="DOP853", rtol=1e-12, atol=1e-12)   # same settings as obs_generator.py


def cr3bp_eom(t, s, mu):
    """Same model as cr3bp.nondim_cr3bp: Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0)."""
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3
    return [vx, vy, vz, ax, ay, az]


def propagate_cr3bp(x0: np.ndarray, tau0: float, taus: np.ndarray, mu: float) -> np.ndarray:
    """Propagate x0 (given at tau0) to every time in taus, forward and/or backward.
    Returns an array of shape (len(taus), 6)."""
    taus = np.asarray(taus, dtype=float)
    out = np.empty((len(taus), 6))
    for mask, t_end in ((taus >= tau0, taus.max()), (taus < tau0, taus.min())):
        if not mask.any():
            continue
        if t_end == tau0:
            out[mask] = x0
            continue
        sol = solve_ivp(cr3bp_eom, [tau0, t_end], x0, args=(mu,), dense_output=True, **TOL)
        if not sol.success:
            raise RuntimeError(f"CR3BP propagation failed: {sol.message}")
        out[mask] = sol.sol(taus[mask]).T
    return out


def cr3bp_eom_stm(t, s, mu):
    """CR3BP state plus its 6x6 state transition matrix (row-major, 42 values in all)."""
    x, y, z = s[0], s[1], s[2]
    out = np.empty(42)
    out[:6] = cr3bp_eom(t, s[:6], mu)
    G = np.diag([1.0, 1.0, 0.0])                    # centrifugal term
    for p, m in (((-mu, 0.0, 0.0), 1 - mu), ((1 - mu, 0.0, 0.0), mu)):
        d = np.array([x - p[0], y, z])
        r = np.sqrt(d @ d)
        G += m * (3.0 * np.outer(d, d) / r**5 - np.eye(3) / r**3)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = G
    A[3, 4], A[4, 3] = 2.0, -2.0                    # Coriolis
    out[6:] = (A @ s[6:].reshape(6, 6)).ravel()
    return out


def propagate_cr3bp_stm(x0: np.ndarray, tau0: float, taus: np.ndarray, mu: float, rtol: float = 1e-10,
                        atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Like propagate_cr3bp, but also returns d state(tau) / d x0 for each time: shapes (M, 6) and (M, 6, 6)."""
    taus = np.asarray(taus, dtype=float)
    states, stms = np.empty((len(taus), 6)), np.empty((len(taus), 6, 6))
    s0 = np.concatenate([x0, np.eye(6).ravel()])
    for mask, t_end in ((taus >= tau0, taus.max()), (taus < tau0, taus.min())):
        if not mask.any():
            continue
        if t_end == tau0:
            states[mask], stms[mask] = x0, np.eye(6)
            continue
        sol = solve_ivp(cr3bp_eom_stm, [tau0, t_end], s0, args=(mu,), dense_output=True, method="DOP853",
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"CR3BP propagation failed: {sol.message}")
        y = sol.sol(taus[mask]).T
        states[mask], stms[mask] = y[:, :6], y[:, 6:].reshape(-1, 6, 6)
    return states, stms

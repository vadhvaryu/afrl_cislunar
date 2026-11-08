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

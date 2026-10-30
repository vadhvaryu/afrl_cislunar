"""
Shared CR3BP tools for IOD and estimation (rotating, nondimensional frame unless noted).

    stm_propagate(x0_rot, tau0, tau1, mu)   state and 6x6 state transition matrix, either direction
    obs_to_rotating(obs, sys)                per observation: tau, observer position, line of sight (rotating)
    make_synthetic_obs(x0_rot, t_s, ...)     Observations at any times, same geometry as obs_generator.py
    los_derivatives(obs, idx, t)             line of sight and its 1st/2nd time derivatives by Lagrange fit (EMI)

Conventions as obs_generator.py and frames.py: Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0);
the rotating frame has turned by angle tau = t / TU since t0; RA/Dec are measured in the
non-rotating EMI frame (x toward the Moon at t0, z normal to the Earth-Moon plane).

None of these functions read the truth file. make_synthetic_obs takes the true initial state
as an argument; it is the caller's job (a test, a benchmark) to keep that away from IOD methods.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import Observations
from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem, radec_to_los, rot_z

DAY = 86400.0


# ----------------------------------------------------------------------------- dynamics

def cr3bp_jacobian(s, mu: float) -> np.ndarray:
    """6x6 Jacobian of the CR3BP equations at state s; same as cr3bp.cr3bp_jacobian (tested)."""
    x, y, z = s[0], s[1], s[2]
    dx1, dx2 = x + mu, x - 1 + mu
    r1 = np.sqrt(dx1**2 + y**2 + z**2)
    r2 = np.sqrt(dx2**2 + y**2 + z**2)
    a1, b1 = (1 - mu) / r1**3, 3 * (1 - mu) / r1**5
    a2, b2 = mu / r2**3, 3 * mu / r2**5
    uxy = b1 * dx1 * y + b2 * dx2 * y
    uxz = b1 * dx1 * z + b2 * dx2 * z
    uyz = (b1 + b2) * y * z
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0] = 1 - a1 - a2 + b1 * dx1**2 + b2 * dx2**2
    A[4, 1] = 1 - a1 - a2 + (b1 + b2) * y**2
    A[5, 2] = -a1 - a2 + (b1 + b2) * z**2
    A[3, 1] = A[4, 0] = uxy
    A[3, 2] = A[5, 0] = uxz
    A[4, 2] = A[5, 1] = uyz
    A[3, 4], A[4, 3] = 2.0, -2.0
    return A


def _eom_stm(t, s, mu):
    phi = s[6:].reshape(6, 6)
    out = np.empty(42)
    out[:6] = cr3bp_eom(t, s[:6], mu)
    out[6:] = (cr3bp_jacobian(s[:6], mu) @ phi).ravel()
    return out


def stm_propagate(x0_rot, tau0: float, tau1: float, mu: float):
    """Propagate x0_rot from tau0 to tau1 (forward or backward) with the state transition
    matrix. Returns (x(tau1), Phi(tau1, tau0)). DOP853, rtol = atol = 1e-12 (dynamics.TOL)."""
    x0 = np.asarray(x0_rot, float)
    if tau1 == tau0:
        return x0.copy(), np.eye(6)
    sol = solve_ivp(_eom_stm, [tau0, tau1], np.concatenate([x0, np.eye(6).ravel()]), args=(mu,), **TOL)
    if not sol.success:
        raise RuntimeError(f"CR3BP STM propagation failed: {sol.message}")
    return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)


# ----------------------------------------------------------------------------- observations

def obs_to_rotating(obs: Observations, sys: EarthMoonSystem):
    """(tau (N,), observer position (N, 3), unit line of sight (N, 3)), rotating nondimensional.

    The observer's EMI position o maps to R(tau)^T o / DU + Earth, so Earth's center maps to
    (-mu, 0, 0). A direction maps by R(tau)^T only (rho_rot = rot_z(-tau) rho_EMI)."""
    tau = sys.tau(obs.t)
    R_o = np.array([rot_z(t).T @ (o / sys.du_m) + sys.earth_rot for t, o in zip(tau, obs.observer)])
    u = np.array([rot_z(t).T @ l for t, l in zip(tau, obs.los)])
    return tau, R_o, u


def _line_of_sight_inertial(r_rot, tau, observer_rot):
    """obs_generator.line_of_sight_inertial, reimplemented (obs_generator runs on import)."""
    l = r_rot - observer_rot
    c, s = np.cos(tau), np.sin(tau)
    return np.column_stack((c * l[:, 0] - s * l[:, 1], s * l[:, 0] + c * l[:, 1], l[:, 2]))


def _ra_dec_deg(l):
    """obs_generator.ra_dec, reimplemented: RA in [0, 360), Dec, degrees."""
    ra = np.degrees(np.arctan2(l[:, 1], l[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(l[:, 2] / np.linalg.norm(l, axis=1)))
    return ra, dec


def make_synthetic_obs(x0_rot, times_s, noise_arcsec: float, seed: int, sys: EarthMoonSystem,
                       tau0: float = 0.0) -> Observations:
    """Angles-only observations of the CR3BP trajectory through x0_rot (given at tau0), from
    Earth's center, at times_s (seconds since t0). Same geometry, noise model and draw order
    as obs_generator.py: Gaussian noise in degrees, all RA draws first, then all Dec draws."""
    t = np.asarray(times_s, float)
    taus = sys.tau(t)
    lo, hi = min(tau0, taus.min()), max(tau0, taus.max())
    Y = np.empty((len(t), 6))
    for mask, end in ((taus >= tau0, hi), (taus < tau0, lo)):
        if not mask.any():
            continue
        if end == tau0:
            Y[mask] = x0_rot
            continue
        sol = solve_ivp(cr3bp_eom, [tau0, end], np.asarray(x0_rot, float), args=(sys.mu,), dense_output=True, **TOL)
        Y[mask] = sol.sol(taus[mask]).T
    observer_rot = sys.earth_rot
    ra, dec = _ra_dec_deg(_line_of_sight_inertial(Y[:, :3], taus, observer_rot))
    rng = np.random.default_rng(seed)
    s_deg = noise_arcsec / 3600.0
    ra_m = (ra + rng.normal(0, s_deg, len(t))) % 360.0
    dec_m = dec + rng.normal(0, s_deg, len(t))
    ra_r, dec_r = np.radians(ra_m), np.radians(dec_m)
    return Observations(t=t, los=radec_to_los(ra_r, dec_r), observer=np.zeros((len(t), 3)),
                        sigma=np.full(len(t), np.radians(s_deg)), ra=ra_r, dec=dec_r)


def los_derivatives(obs: Observations, idx, t_eval: float):
    """Line of sight (EMI unit vector) and its first and second time derivatives at t_eval,
    from the Lagrange polynomial through the observations idx (degree len(idx) - 1), in 1/s.
    The fitted vector is not exactly unit length between nodes; it is normalized at t_eval and
    its derivatives are projected accordingly (rho_hat' is perpendicular to rho_hat)."""
    idx = list(idx)
    t = obs.t[idx] - t_eval
    L = obs.los[idx]
    n = len(idx)
    # Polynomial coefficients per component (exact interpolation through n points)
    V = np.vander(t, n, increasing=True)
    c = np.linalg.solve(V, L)                        # (n, 3): L(t) = sum_k c_k t^k
    f = c[0]
    f1 = c[1] if n > 1 else np.zeros(3)
    f2 = 2 * c[2] if n > 2 else np.zeros(3)
    # u = f / m with m = |f|:  m' = u.f',  u' = (f' - m'u) / m,
    # m'' = u.f'' + m|u'|^2,   u'' = (f'' - m''u - 2m'u') / m
    m = np.linalg.norm(f)
    u = f / m
    m1 = u @ f1
    du = (f1 - m1 * u) / m
    m2 = u @ f2 + m * (du @ du)
    du2 = (f2 - m2 * u - 2 * m1 * du) / m
    return u, du, du2

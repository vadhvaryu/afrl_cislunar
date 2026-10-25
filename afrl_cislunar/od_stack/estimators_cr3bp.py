"""
Estimators that do not use Orekit's batch least squares (convergence study, A7).

    multiple_shooting           CR3BP multiple shooting fitted to the angles (scipy least_squares)
    multiple_shooting_then_bls  multiple_shooting, then orekit_bls from its answer (polish + Orekit covariance)
    ukf                         unscented Kalman filter + Rauch-Tung-Striebel smoother in the CR3BP

All take (Observations, StateEstimate from IOD) and return a StateEstimate at the IOD epoch with a
6x6 covariance (EMI, SI), and put `rms_normalized_residual` (post-fit RA/Dec residual RMS in sigma
units, same definition as Orekit's) in info so the 3-sigma residual gate applies to them as well.
The dynamics are the CR3BP (same model as the truth and as orekit_cr3bp); the observer is Earth's
center (as obs_generator.py v1).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .contracts import Observations, StateEstimate
from .cr3bp_tools import _eom_stm
from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem, emi_to_rotating, rot_z, rotating_to_emi


# ----------------------------------------------------------------------------- shared measurement model

def _failure(msg: str, last_state=None):
    """estimation.EstimationFailure (imported here: estimation.py imports this module)."""
    from .estimation import EstimationFailure

    return EstimationFailure(msg, last_state)


def _radec_and_jac(r_emi: np.ndarray, observer: np.ndarray):
    """RA, Dec [rad] of r - observer and their gradients with respect to r (EMI)."""
    d = r_emi - observer
    x, y, z = d
    rho2_xy = x * x + y * y
    rho_xy = np.sqrt(rho2_xy)
    rho2 = rho2_xy + z * z
    ra, dec = np.arctan2(y, x), np.arcsin(z / np.sqrt(rho2))
    g_ra = np.array([-y, x, 0.0]) / rho2_xy
    g_dec = np.array([-x * z, -y * z, rho2_xy]) / (rho2 * rho_xy)
    return ra, dec, g_ra, g_dec


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _propagate_many(x0, tau0, taus, mu, stm=False):
    """States (and STMs) at taus (all on one side of tau0, sorted away from it)."""
    taus = np.asarray(taus, float)
    if len(taus) == 0:
        return np.empty((0, 6)), np.empty((0, 6, 6))
    end = taus[np.argmax(np.abs(taus - tau0))]
    if end == tau0:
        return np.tile(x0, (len(taus), 1)), np.tile(np.eye(6), (len(taus), 1, 1))
    y0 = np.concatenate([x0, np.eye(6).ravel()]) if stm else np.asarray(x0, float)
    sol = solve_ivp(_eom_stm if stm else cr3bp_eom, [tau0, end], y0, args=(mu,), dense_output=True, **TOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.sol(taus).T
    return Y[:, :6], (Y[:, 6:].reshape(-1, 6, 6) if stm else None)


def _propagate_both_ways(x0, tau0, taus, mu, stm=False):
    taus = np.asarray(taus, float)
    X, P = np.empty((len(taus), 6)), np.empty((len(taus), 6, 6))
    for mask in (taus >= tau0, taus < tau0):
        if mask.any():
            x, p = _propagate_many(x0, tau0, taus[mask], mu, stm)
            X[mask] = x
            if stm:
                P[mask] = p
    return X, (P if stm else None)


def _angle_rows(sys: EarthMoonSystem, obs: Observations, i: int, x_rot: np.ndarray):
    """Normalized residuals (2,) for observation i at rotating state x_rot, and their gradient
    (2, 6) with respect to x_rot."""
    tau = float(sys.tau(obs.t[i]))
    R = rot_z(tau)
    r = R @ (x_rot[:3] - sys.earth_rot) * sys.du_m
    ra, dec, g_ra, g_dec = _radec_and_jac(r, obs.observer[i])
    s = obs.sigma[i]
    res = np.array([_wrap(obs.ra[i] - ra), obs.dec[i] - dec]) / s
    dr_dx = np.zeros((3, 6))
    dr_dx[:, :3] = R * sys.du_m
    jac = -np.vstack([g_ra @ dr_dx, g_dec @ dr_dx]) / s
    return res, jac


def _covariance_and_rms(sys, obs, x_rot_epoch, tau_epoch):
    """Exact 6-state angles-only covariance at the epoch (rotating nondimensional -> EMI SI),
    and the residual RMS, from one propagation with the STM."""
    X, P = _propagate_both_ways(x_rot_epoch, tau_epoch, sys.tau(obs.t), sys.mu, stm=True)
    H, res = [], []
    for i in range(len(obs)):
        r, j = _angle_rows(sys, obs, i, X[i])
        res.append(r)
        H.append(j @ P[i])
    H, res = np.vstack(H), np.concatenate(res)
    rms = float(np.sqrt(np.mean(res**2)))
    try:
        C_rot = np.linalg.inv(H.T @ H)                # residuals are already sigma-normalized
    except np.linalg.LinAlgError:
        return None, rms
    J = _rot_to_emi_jacobian(sys, tau_epoch)
    return J @ C_rot @ J.T, rms


def _rot_to_emi_jacobian(sys, tau):
    """d(r, v)_EMI,SI / d(x_rot) (linear map, so exact)."""
    J = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = 1.0
        r1, v1 = rotating_to_emi(tau, e + np.r_[sys.earth_rot, 0, 0, 0], sys)
        J[:, k] = np.concatenate([r1, v1])
    return J


# ----------------------------------------------------------------------------- multiple shooting

class CR3BPMultipleShooting:
    """Multiple shooting on angles in the CR3BP (rotating, nondimensional).

    The observation span is cut into `segments` equal pieces. Unknowns: the state at the start
    of each piece (6 per segment). Residuals: RA/Dec at every observation, from the segment it
    falls in, divided by sigma; plus the continuity defects at the joins (position in DU,
    velocity in DU/TU) times sqrt(weight). Solved with scipy least_squares (trf), analytic
    Jacobians from the state transition matrix. Initial guess: the IOD state propagated to each
    segment start.

    Continuity weight schedule (continuity_weights, default 1e6 -> 1e10 -> 1e14), each stage
    starting from the previous one. A single weight of 1e6 is far too soft here: with angle
    residuals in sigma units, one sigma (10") is ~18 km of position at lunar distance, so a
    1e6 weight leaves ~60 km gaps between segments (measured). The loose first stage lets the
    segments move independently (the point of multiple shooting); the last enforces continuity
    (defect < max_defect, default 1e-8 DU ~ 4 m).

    Output: the state at the IOD epoch (propagated from the segment that contains it) and the
    6x6 covariance (H^T W H)^-1 of the angles with respect to that state, computed exactly
    through the STM at the solution. This is the limit of the epoch block of (J^T W J)^-1 as the
    continuity weight grows, without depending on the weight.
    Fails (EstimationFailure) if the solver fails or continuity is not met (defect > 1e-6).
    """

    name = "multiple_shooting"

    def __init__(self, sys: EarthMoonSystem, segments: int = 4, continuity_weights=(1e6, 1e10, 1e14),
                 max_nfev: int = 100, max_defect: float = 1e-8, **_):
        self.sys = sys
        self.nseg = int(segments)
        self.weights = [float(w) for w in continuity_weights]
        self.cw = float(np.sqrt(self.weights[-1]))
        self.max_nfev = int(max_nfev)
        self.max_defect = float(max_defect)

    def _layout(self, obs):
        sys = self.sys
        tau_obs = sys.tau(obs.t)
        nodes = np.linspace(tau_obs.min(), tau_obs.max(), self.nseg + 1)
        seg = np.clip(np.searchsorted(nodes, tau_obs, side="right") - 1, 0, self.nseg - 1)
        return nodes, seg, tau_obs

    def _fun_jac(self, z, obs, nodes, seg, tau_obs):
        mu, n = self.sys.mu, self.nseg
        Xn = z.reshape(n, 6)
        m_ang, m_con = 2 * len(obs), 6 * (n - 1)
        F = np.zeros(m_ang + m_con)
        J = np.zeros((m_ang + m_con, 6 * n))
        for k in range(n):
            idx = np.flatnonzero(seg == k)
            taus = list(tau_obs[idx]) + ([nodes[k + 1]] if k < n - 1 else [])
            X, P = _propagate_many(Xn[k], nodes[k], np.array(taus), mu, stm=True)
            for j, i in enumerate(idx):
                r, g = _angle_rows(self.sys, obs, i, X[j])
                F[2 * i:2 * i + 2] = r
                J[2 * i:2 * i + 2, 6 * k:6 * k + 6] = g @ P[j]
            if k < n - 1:
                row = m_ang + 6 * k
                F[row:row + 6] = self.cw * (X[-1] - Xn[k + 1])
                J[row:row + 6, 6 * k:6 * k + 6] = self.cw * P[-1]
                J[row:row + 6, 6 * (k + 1):6 * (k + 2)] = -self.cw * np.eye(6)
        return F, J

    def solve(self, obs: Observations, x0: StateEstimate):
        sys = self.sys
        nodes, seg, tau_obs = self._layout(obs)
        tau_e = float(sys.tau(x0.t))
        x_rot = emi_to_rotating(tau_e, x0.r, x0.v, sys)
        X0, _ = _propagate_both_ways(x_rot, tau_e, nodes[:-1], sys.mu)
        cache = {}

        def fj(z):
            key = (z.tobytes(), self.cw)
            if key not in cache:
                cache.clear()
                cache[key] = self._fun_jac(z, obs, nodes, seg, tau_obs)
            return cache[key]

        z, nfev = X0.ravel(), 0
        try:
            for w in self.weights:
                self.cw = float(np.sqrt(w))
                sol = least_squares(lambda z: fj(z)[0], z, jac=lambda z: fj(z)[1], method="trf", x_scale="jac",
                                    max_nfev=self.max_nfev, xtol=1e-14, ftol=1e-14, gtol=1e-12)
                z, nfev = sol.x, nfev + sol.nfev
        except (RuntimeError, ValueError, FloatingPointError) as e:
            raise _failure(f"{self.name}: {e}") from None
        sol.nfev = nfev
        F, _ = fj(sol.x)
        defect = float(np.abs(F[2 * len(obs):]).max() / self.cw) if self.nseg > 1 else 0.0
        Xn = sol.x.reshape(self.nseg, 6)
        k = int(np.clip(np.searchsorted(nodes, tau_e, side="right") - 1, 0, self.nseg - 1))
        x_e = _propagate_both_ways(Xn[k], nodes[k], np.array([tau_e]), sys.mu)[0][0]
        return sol, x_e, tau_e, defect

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        sys = self.sys
        sol, x_e, tau_e, defect = self.solve(obs, x0)
        r, v = rotating_to_emi(tau_e, x_e, sys)
        info = {**x0.info, "iterations": int(sol.nfev), "solver_status": int(sol.status), "max_defect": defect}
        if not np.all(np.isfinite(x_e)) or defect > self.max_defect:
            last = StateEstimate(t=x0.t, r=r, v=v, method=f"{x0.method} + {self.name}",
                                 info={**info, "state_kind": "last_iterate"})
            raise _failure(f"{self.name}: not converged (status {sol.status}, max defect {defect:.2g})", last)
        cov, rms = _covariance_and_rms(sys, obs, x_e, tau_e)
        info["rms_normalized_residual"] = rms
        return StateEstimate(t=x0.t, r=r, v=v, cov=cov, method=f"{x0.method} + {self.name}", info=info)


class MultipleShootingThenBLS:
    """multiple_shooting, then orekit_bls from its answer (Orekit's covariance and iterations)."""

    name = "multiple_shooting_then_bls"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, **kw):
        from .estimation import OrekitBatchLS

        self.ms = CR3BPMultipleShooting(sys, **kw)
        kw.pop("gate_rms", None)
        self.bls = OrekitBatchLS(sys, orekit_data=orekit_data, **kw)

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        first = self.ms.refine(obs, x0)
        fit = self.bls.refine(obs, StateEstimate(t=first.t, r=first.r, v=first.v, method=x0.method, info=dict(x0.info)))
        fit.method = f"{x0.method} + {self.name}"
        fit.info.update(ms_iterations=first.info["iterations"], bls_iterations=fit.info["iterations"],
                        iterations=first.info["iterations"] + fit.info["iterations"])
        return fit


# ----------------------------------------------------------------------------- unscented filter + smoother

class CR3BPUnscented:
    """Unscented Kalman filter with a Rauch-Tung-Striebel smoother (Sarkka 2008) in the CR3BP.

    1. Starting covariance P0 at the IOD epoch: the IOD's own covariance if it gives one;
       otherwise the stand-in's known error size (info position_error_km_set / velocity_error_m_s_set,
       spread evenly over 3 axes); otherwise initial_sigma_km / initial_sigma_m_s (per axis).
    2. Unscented transform back to the first observation (12 symmetric sigma points, weights 1/12,
       the same scheme as uncertainty.propagate_covariance).
    3. Forward filter through every observation (RA/Dec in sigma units, RA wrapped around the
       measured value); process noise Q is tiny (deterministic dynamics), only for numerical health.
    4. RTS smoother back to the first observation; the smoothed state at the epoch is reported
       (propagated from the nearest earlier observation if the epoch is not an observation time).
    """

    name = "ukf"

    def __init__(self, sys: EarthMoonSystem, initial_sigma_km: float = 10_000.0, initial_sigma_m_s: float = 100.0,
                 q_nondim: float = 1e-18, **_):
        self.sys = sys
        self.sig_km, self.sig_ms = float(initial_sigma_km), float(initial_sigma_m_s)
        self.q = float(q_nondim)

    def _p0_rot(self, x0: StateEstimate, tau):
        if x0.cov is not None:
            P_emi = np.asarray(x0.cov, float)
        else:
            d_km = x0.info.get("position_error_km_set")
            w = x0.info.get("velocity_error_m_s_set")
            sr = (float(d_km) if d_km is not None else np.sqrt(3) * self.sig_km) * 1e3 / np.sqrt(3)
            sv = (float(w) if w is not None else np.sqrt(3) * self.sig_ms) / np.sqrt(3)
            sr, sv = max(sr, 1.0), max(sv, 1e-4)                    # keep it positive definite
            P_emi = np.diag([sr**2] * 3 + [sv**2] * 3)
        Ji = np.linalg.inv(_rot_to_emi_jacobian(self.sys, tau))
        return Ji @ P_emi @ Ji.T

    @staticmethod
    def _points(m, P):
        n = len(m)
        try:
            L = np.linalg.cholesky((P + P.T) / 2)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh((P + P.T) / 2)
            L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
        L = L * np.sqrt(n)
        return np.vstack([m + L[:, k] for k in range(n)] + [m - L[:, k] for k in range(n)])

    def _prop(self, X, tau_a, tau_b):
        return np.array([_propagate_both_ways(x, tau_a, np.array([tau_b]), self.sys.mu)[0][0] for x in X])

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        sys, n = self.sys, 6
        tau_e = float(sys.tau(x0.t))
        taus = sys.tau(obs.t)
        m = emi_to_rotating(tau_e, x0.r, x0.v, sys)
        P = self._p0_rot(x0, tau_e)
        Q = self.q * np.eye(n)
        try:
            # 2. back to the first observation
            X = self._prop(self._points(m, P), tau_e, taus[0])
            m, P = X.mean(0), (X - X.mean(0)).T @ (X - X.mean(0)) / (2 * n) + Q
            filt, pred = [], []              # (m, P) filtered; (m-, P-, D) predicted for k+1 from k
            for k in range(len(obs)):
                if k > 0:
                    Xk = self._points(m, P)
                    Xp = self._prop(Xk, taus[k - 1], taus[k])
                    mp = Xp.mean(0)
                    Pp = (Xp - mp).T @ (Xp - mp) / (2 * n) + Q
                    D = (Xk - m).T @ (Xp - mp) / (2 * n)
                    pred.append((mp, Pp, D))
                    m, P = mp, Pp
                # 3. update with observation k
                Xs = self._points(m, P)
                Z = np.array([_angle_rows(sys, obs, k, x)[0] for x in Xs])   # residuals (obs - h)/sigma
                Zh = -Z                                                          # h - obs, in sigma units
                zm = Zh.mean(0)
                S = (Zh - zm).T @ (Zh - zm) / (2 * n) + np.eye(2)
                C = (Xs - m).T @ (Zh - zm) / (2 * n)
                K = C @ np.linalg.inv(S)
                m = m + K @ (0.0 - zm)
                P = P - K @ S @ K.T
                filt.append((m, P))
            # 4. RTS smoother
            ms, Ps = filt[-1]
            smooth = [None] * len(obs)
            smooth[-1] = (ms, Ps)
            for k in range(len(obs) - 2, -1, -1):
                mk, Pk = filt[k]
                mp, Pp, D = pred[k]
                G = D @ np.linalg.inv(Pp)
                ms = mk + G @ (smooth[k + 1][0] - mp)
                Ps = Pk + G @ (smooth[k + 1][1] - Pp) @ G.T
                smooth[k] = (ms, Ps)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
            raise _failure(f"{self.name}: {str(e)[:100]}") from None
        k = int(np.searchsorted(taus, tau_e, side="right") - 1)
        k = min(max(k, 0), len(obs) - 1)
        me, Pe = smooth[k]
        if taus[k] != tau_e:
            Xe = self._prop(self._points(me, Pe), taus[k], tau_e)
            me, Pe = Xe.mean(0), (Xe - Xe.mean(0)).T @ (Xe - Xe.mean(0)) / (2 * n)
        if not np.all(np.isfinite(me)):
            raise _failure(f"{self.name}: diverged")
        J = _rot_to_emi_jacobian(sys, tau_e)
        r, v = rotating_to_emi(tau_e, me, sys)
        try:
            _, rms = _covariance_and_rms(sys, obs, me, tau_e)
        except RuntimeError:
            rms = float("nan")
        return StateEstimate(t=x0.t, r=r, v=v, cov=J @ Pe @ J.T, method=f"{x0.method} + {self.name}",
                             info={**x0.info, "iterations": 1, "rms_normalized_residual": rms})

"""
Per-run variables for the convergence study: what is different about a starting guess
that batch least squares (BLS) fails to converge from?

    fc = FeatureCalculator(sys, truth_path, threshold_km=1000)
    row.update(fc(obs, x_iod, x_final))        # x_final: BLS solution, last iterate, or None
    row["outcome"] = outcome_category(row)

Everything is graded against the truth (synthetic data only). All columns are plain
floats / bools / strings so they go straight into a CSV.

Variables (column names in brackets):
  1. Periselene of the starting guess over [0, 8] and [0, 12] days, and its time
     [peri_guess_8d_km, peri_guess_8d_day, peri_guess_12d_km, peri_guess_12d_day].
     The guess is propagated in the CR3BP (same equations and DOP853 1e-12 tolerances as
     propagators.CR3BPPropagator) backward to day 0 and forward to day 12, sampled every
     hour, and the minimum Moon distance is refined with scipy minimize_scalar on the
     dense-output solution.
  2. Truth periselene over the same windows (FeatureCalculator.truth_periselene) and
     [peri_delta_8d_km, peri_delta_12d_km] = guess - truth.
  3. Periselene of the final state [peri_final_8d_km, peri_final_12d_km, final_state_kind].
  4. Position error vs the observed line of sight at obs index 2 (EMI)
     [dr_along_los_km, dr_across_los_km, angle_dr_los_deg].
  5. RIC components of dr and dv (Earth-centered, from the truth state: R = r/|r|,
     C = r x v / |r x v|, I = C x R) [dr_R_km, dr_I_km, dr_C_km, dv_R_m_s, dv_I_m_s, dv_C_m_s],
     and the Moon-relative radial component of dr [dr_moon_radial_km].
  6. Velocity error along the velocity [dv_along_v_m_s, angle_dv_v_deg].
  7. Jacobi constant change [delta_jacobi] = C(guess) - C(truth), C = 2*Omega - v^2 in the
     rotating nondimensional frame (C = -2 * cr3bp.jacobi; cr3bp.jacobi is the energy).
  8. Flags for the guess over [0, 12] days [hits_moon, hits_earth, leaves_l1_region]:
     Moon distance < 1,737.4 km, Earth distance < 6,378 km, rotating-frame distance from
     L1 > 0.3 DU at any time. Propagation stops at an impact.
  9. Starting residuals: RA/Dec residual RMS of the guess at the observation times, in
     sigma units [start_residual_rms_sigma].
 10. outcome_category(row): converged / iteration_limit / wrong_orbit / other_error.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from .contracts import Observations, StateEstimate, Trajectory
from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi
from .results import residuals
from .sources import dense_truth, load_synthetic_truth

DAY = 86400.0
R_MOON_KM = 1737.4
R_EARTH_KM = 6378.0
L1_REGION_DU = 0.3
WINDOWS_DAYS = (8.0, 12.0)


def l1_x(mu: float) -> float:
    """x coordinate of L1 (rotating frame, nondimensional): where the x-acceleration
    of a particle at rest on the x-axis vanishes, between Earth and Moon."""
    def ax(x):
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3
    return brentq(ax, -mu + 1e-6, 1 - mu - 1e-6, xtol=1e-15)


def jacobi_constant(x_rot: np.ndarray, mu: float) -> float:
    """C = 2*Omega - v^2, Omega = (x^2 + y^2)/2 + (1 - mu)/r1 + mu/r2 (rotating, nondimensional)."""
    x, y, z = x_rot[:3]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return float(x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - np.dot(x_rot[3:], x_rot[3:]))


def _angle_deg(a, b, tiny: float = 0.0) -> float:
    """Angle between a and b; NaN when |a| <= tiny (no error -> no direction)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= tiny or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


class _Arc:
    """One CR3BP trajectory through a state, covering [tau_lo, tau_hi] (or less if it
    hits Earth or the Moon), with dense output in both directions."""

    def __init__(self, x_rot0, tau0, tau_lo, tau_hi, sys: EarthMoonSystem):
        mu = sys.mu
        self.mu = mu
        self.moon = np.array([1 - mu, 0.0, 0.0])
        r_moon, r_earth = R_MOON_KM / sys.du_km, R_EARTH_KM / sys.du_km

        def hit_moon(t, s, mu):
            return np.linalg.norm(s[:3] - self.moon) - r_moon

        def hit_earth(t, s, mu):
            return np.linalg.norm(s[:3] - np.array([-mu, 0, 0])) - r_earth

        hit_moon.terminal = hit_earth.terminal = True
        self.sols, self.hits_moon, self.hits_earth = [], False, False
        self.lo, self.hi = tau0, tau0
        for t_end in (tau_lo, tau_hi):
            if t_end == tau0:
                continue
            sol = solve_ivp(cr3bp_eom, [tau0, t_end], x_rot0, args=(mu,), dense_output=True,
                            events=(hit_moon, hit_earth), **TOL)
            self.hits_moon |= bool(len(sol.t_events[0]))
            self.hits_earth |= bool(len(sol.t_events[1]))
            if sol.sol is None:                                   # stopped on the first step
                continue
            self.sols.append(sol)
            self.lo, self.hi = min(self.lo, sol.t[-1]), max(self.hi, sol.t[-1])
        self.x0, self.tau0 = np.asarray(x_rot0, float), tau0

    def state(self, tau: float) -> np.ndarray:
        for sol in self.sols:
            a, b = sorted((sol.t[0], sol.t[-1]))
            if a <= tau <= b:
                return sol.sol(tau)
        return self.x0

    def moon_dist(self, tau: float) -> float:
        return float(np.linalg.norm(self.state(tau)[:3] - self.moon))

    def periselene(self, tau_lo: float, tau_hi: float, step: float):
        """(min Moon distance [DU], tau of it) over [tau_lo, tau_hi] clipped to the arc:
        hourly samples, then minimize_scalar around the smallest one."""
        a, b = max(tau_lo, self.lo), min(tau_hi, self.hi)
        if b <= a:
            return float("nan"), float("nan")
        taus = np.append(np.arange(a, b, step), b)
        d = np.array([self.moon_dist(t) for t in taus])
        i = int(np.argmin(d))
        lo, hi = taus[max(i - 1, 0)], taus[min(i + 1, len(taus) - 1)]
        best = (d[i], taus[i])
        if hi > lo:
            r = minimize_scalar(self.moon_dist, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
            if r.fun < best[0]:
                best = (float(r.fun), float(r.x))
        return best

    def max_dist_from(self, point: np.ndarray, tau_lo: float, tau_hi: float, step: float) -> float:
        a, b = max(tau_lo, self.lo), min(tau_hi, self.hi)
        taus = np.append(np.arange(a, b, step), b)
        return float(max(np.linalg.norm(self.state(t)[:3] - point) for t in taus))


class FeatureCalculator:
    """Built once per worker (reads the truth once); called once per run."""

    def __init__(self, sys: EarthMoonSystem, truth_path: str, los_index: int = 2):
        self.sys = sys
        self.los_index = int(los_index)
        _, self.tru_rot, self.tru_tau = load_synthetic_truth(truth_path, sys)
        self.l1 = np.array([l1_x(sys.mu), 0.0, 0.0])
        self.step = 3600.0 / sys.tu_s                                 # 1 hour, nondimensional
        self.windows = [(0.0, sys.tau(w * DAY)) for w in WINDOWS_DAYS]
        truth0 = self._truth_state(float(self.tru_tau[0]) * sys.tu_s)
        arc = self._arc(truth0)
        self.truth_periselene = {}
        for w, (a, b) in zip(WINDOWS_DAYS, self.windows):
            d, tau = arc.periselene(a, b, self.step)
            self.truth_periselene[f"{w:g}d"] = {"km": d * sys.du_km, "day": tau * sys.tu_s / DAY}

    def _truth_state(self, t: float) -> StateEstimate:
        tr = dense_truth(self.tru_rot, self.tru_tau, np.array([t]), self.sys)
        return StateEstimate(t=t, r=tr.r[0], v=tr.v[0], method="truth")

    def _arc(self, x: StateEstimate) -> _Arc:
        tau0 = float(self.sys.tau(x.t))
        x_rot = emi_to_rotating(tau0, x.r, x.v, self.sys)
        return _Arc(x_rot, tau0, min(a for a, _ in self.windows), max(b for _, b in self.windows), self.sys)

    def periselene(self, x: StateEstimate, prefix: str) -> tuple[dict, _Arc]:
        arc = self._arc(x)
        out = {}
        for w, (a, b) in zip(WINDOWS_DAYS, self.windows):
            d, tau = arc.periselene(a, b, self.step)
            out[f"{prefix}_{w:g}d_km"] = d * self.sys.du_km
            out[f"{prefix}_{w:g}d_day"] = tau * self.sys.tu_s / DAY
        return out, arc

    def __call__(self, obs: Observations, x_iod: StateEstimate, x_final: StateEstimate | None) -> dict:
        sys = self.sys
        f, arc = self.periselene(x_iod, "peri_guess")
        for w in WINDOWS_DAYS:
            f[f"peri_delta_{w:g}d_km"] = f[f"peri_guess_{w:g}d_km"] - self.truth_periselene[f"{w:g}d"]["km"]

        # Flags over [0, 12] days
        a, b = self.windows[-1]
        f["hits_moon"], f["hits_earth"] = arc.hits_moon, arc.hits_earth
        f["leaves_l1_region"] = arc.max_dist_from(self.l1, a, b, self.step) > L1_REGION_DU

        # Error geometry at the IOD epoch
        tr = self._truth_state(x_iod.t)
        dr, dv = x_iod.r - tr.r, x_iod.v - tr.v
        u = obs.los[self.los_index]
        along = float(dr @ u)
        f["dr_along_los_km"] = along / 1e3
        f["dr_across_los_km"] = float(np.linalg.norm(dr - along * u)) / 1e3
        f["angle_dr_los_deg"] = _angle_deg(dr, u, tiny=1.0)              # < 1 m: none
        R = tr.r / np.linalg.norm(tr.r)
        h = np.cross(tr.r, tr.v)
        C = h / np.linalg.norm(h)
        I = np.cross(C, R)
        for name, e in (("R", R), ("I", I), ("C", C)):
            f[f"dr_{name}_km"] = float(dr @ e) / 1e3
            f[f"dv_{name}_m_s"] = float(dv @ e)
        tau = float(sys.tau(x_iod.t))
        r_moon, _ = rotating_to_emi(tau, np.array([1 - sys.mu, 0, 0, 0, 0, 0]), sys)
        m = (tr.r - r_moon) / np.linalg.norm(tr.r - r_moon)
        f["dr_moon_radial_km"] = float(dr @ m) / 1e3
        vhat = tr.v / np.linalg.norm(tr.v)
        f["dv_along_v_m_s"] = float(dv @ vhat)
        f["angle_dv_v_deg"] = _angle_deg(dv, tr.v, tiny=1e-3)          # < 1 mm/s: none
        f["delta_jacobi"] = (jacobi_constant(emi_to_rotating(tau, x_iod.r, x_iod.v, sys), sys.mu)
                             - jacobi_constant(emi_to_rotating(tau, tr.r, tr.v, sys), sys.mu))

        # Starting residuals: the guess propagated to the observation times
        f["start_residual_rms_sigma"] = self._residual_rms(obs, arc)

        # Where the final state (solution or last iterate) sits
        if x_final is None:
            f.update(final_state_kind="none", peri_final_8d_km=float("nan"), peri_final_12d_km=float("nan"))
        else:
            pf, _ = self.periselene(x_final, "peri_final")
            f.update(final_state_kind=x_final.info.get("state_kind", "solution"),
                     peri_final_8d_km=pf["peri_final_8d_km"], peri_final_12d_km=pf["peri_final_12d_km"])
        return f

    def _residual_rms(self, obs: Observations, arc: _Arc) -> float:
        taus = self.sys.tau(obs.t)
        if arc.hits_moon or arc.hits_earth:
            if taus.min() < arc.lo or taus.max() > arc.hi:
                return float("nan")                         # crashed before an observation time
        rv = [rotating_to_emi(t, arc.state(t), self.sys) for t in taus]
        traj = Trajectory(t=obs.t, r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv]))
        return residuals(obs, traj)["rms_normalized"]


def outcome_category(row: dict) -> str:
    """converged | iteration_limit | wrong_orbit | other_error."""
    if row.get("converged"):
        return "converged"
    if row.get("bls_finished"):
        return "wrong_orbit"
    if "maximal count" in str(row.get("error", "")):
        return "iteration_limit"
    return "other_error"

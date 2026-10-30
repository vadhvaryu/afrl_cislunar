"""
CR3BP differential-corrections IOD from three angles-only observations.

Method: S. Jo, J. Baker-McEvilly and D. Canales, "Differential Corrections Algorithm for
Initial Orbit Determination in the Cislunar Region using Angle-Only Measurements",
arXiv:2507.22350 (2025), CC BY 4.0, Section 3 (Eqs. 10-14, 22, 24 and Algorithm 1).
No public code exists; this is implemented from the paper.

Formulation (rotating, nondimensional CR3BP frame; Earth at (-mu,0,0), Moon at (1-mu,0,0)):
    three observations i < j < k with observer positions R_o(t_n) and lines of sight rho_n
    free variables   chi = [alpha_i, alpha_j, alpha_k, v_j]        (ranges, middle velocity)
    positions        r(t_n) = R_o(t_n) + alpha_n rho_n
    constraints      F = [r(t_i) - r~(t_i); r(t_k) - r~(t_k)]      (Eqs. 12-13)
                     r~ = middle state [r(t_j), v_j] propagated back to t_i / forward to t_k
    Jacobian (Eq. 22) [rho_i, -Phi-_11 rho_j, 0,     -Phi-_12]    Phi- = Phi(t_i, t_j)
                      [0,     -Phi+_11 rho_j, rho_k, -Phi+_12]    Phi+ = Phi(t_k, t_j)
    Newton update    chi <- chi - DF^-1 F (full steps, as in the paper); stop when
                     ||F|| < tol (1e-12 DU) or after max_iter (50) iterations.
    Initial guess (Sec. 3.2) alpha_i = alpha_j = alpha_k = alpha0 and the central-difference
                     velocity v_j = (r(t_k) - r(t_i)) / (t_k - t_i)   (Eq. 24).

What this implementation adds or assumes (the paper leaves these open). Each item says why.
  * Range prior. The paper takes alpha0 from photometry (Eq. 23). We have no brightness data,
    so alpha0 is multi-started over a fixed grid, by default 200,000 to 500,000 km every
    10,000 km (31 values; `alpha0_km` overrides it). The grid is a prior: the object is assumed
    to be 0.5 to 1.3 Earth-Moon distances from the observer. It was set by the task, before
    running on any data.
  * Two starts per alpha0 (`seeds`):
      "paper"    exactly the paper: Eq. 24 seed, full Newton steps.
      "lagrange" (extension) the same equal-range positions but for ALL observations, turned
                 into the Earth-centered inertial frame; the velocity at t_j is the derivative
                 of the degree N-1 interpolating polynomial (Eq. 24 is the 3-point case),
                 converted back to rotating. Newton first; if Newton fails and
                 `trust_region=True`, scipy's trust-region-reflective least squares on the
                 same F and Eq. 22 Jacobian (ranges and velocity boxed by the admissibility
                 bounds, at most `tr_max_nfev` = 40 evaluations), then a Newton polish to tol.
    Why: on the generator's spacing (obs at 0, 4, 8 days, about 2/3 of the halo period) the
    paper's method never converges (0 of 31 starts, and none on a 2,000 km grid within
    +/-100,000 km of the true range): the true middle velocity (z component -0.38 DU/TU) is
    invisible to a chord between two points at similar height, and full Newton steps from
    there overshoot to negative ranges or an impact. Pure Newton from the Lagrange seed
    converges from a single 2,000 km pocket; the trust region widens that to about
    0..+80,000 km. On arcs of hours to a day the paper's own seed works (and is kept).
    `tr_max_nfev` = 40 is a runtime choice: on data/synthetic_obs.csv 30 and 50 evaluations
    converge the same starts.
  * Admissibility (`min_range_km`, `max_range_km`, `max_speed_km_s`). A start fails when a
    range is <= 0 (as the task asks) or outside [20,000 km, 1,500,000 km], when the
    rotating-frame speed exceeds 3 km/s, or when a propagation hits the Earth's or the Moon's
    surface (terminal events). Without these, wild Newton iterates wander into near-Earth,
    high-speed families whose propagation takes seconds and never converges.
  * Convergence. ||F|| < tol (1e-12 DU = 0.4 mm). If the iterations run out, a start still
    counts when its best ||F|| is below `accept_tol` (1e-9 DU = 0.4 m); info keeps ||F||.
  * Disambiguation. The paper confirms a candidate with a 4th measurement. We have more
    observations, so every converged solution is propagated (CR3BP) to all observation times
    and its RA/Dec residual RMS in sigma (results.residuals) is computed; solutions within
    `dedup_km` (100 km) of each other at t_j are merged; the lowest RMS wins. All distinct
    solutions with their RMS are kept in info["solutions"] (chosen first).
    info["n_fitting_solutions"] counts solutions within `fit_gate_sigma` (3); more than one
    means the data cannot tell them apart (always so with only three observations; also seen
    on a 1-day arc with 10" noise, where a wrong branch can have the lower RMS).
  * Verification (`verify=True`, report only, never changes the answer): in the spirit of the
    paper's 4th-measurement check, the DC is re-solved on every other consecutive triple
    ((0,1,2), (1,2,3), (2,3,4) for 5 observations and the default (0,2,4)), seeded with the
    chosen solution's ranges at those times and its velocity at the triple's middle time;
    info["verification"] says whether each converged and how far its solution, propagated to
    t_j, lies from the chosen one. With noise, short sub-triples can disagree by hundreds of
    km or fail; that measures noise sensitivity, not necessarily a wrong answer.
  * Model. Exact CR3BP geometry: no light time or aberration; observer positions come from
    Observations.observer (geocenter in our data). The DC fits three lines of sight exactly,
    so measurement noise enters only through the RMS used to rank solutions. With noise on
    very short arcs (5 obs over 6 h at 10") no admissible exact fit may exist: IODFailure.
  * Honesty. Nothing here reads the truth file. `truth_path` and other unknown keyword
    arguments are accepted and ignored.

Output: StateEstimate in EMI/SI at the middle chosen observation's time, method "cr3bp_dc".
"""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import Observations, StateEstimate, Trajectory
from .cr3bp_tools import _eom_stm, obs_to_rotating
from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem, rot_z, rotating_to_emi
from .iod import IODFailure

R_EARTH_KM, R_MOON_KM = 6378.137, 1737.4
Z_HAT = np.array([0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------- propagation

class _Impact(RuntimeError):
    pass


def _impact_events(mu, du_km):
    """Terminal events at the Earth's and the Moon's surfaces."""
    re, rm = R_EARTH_KM / du_km, R_MOON_KM / du_km

    def earth(t, s, *a):
        return (s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2 - re**2

    def moon(t, s, *a):
        return (s[0] - 1 + mu) ** 2 + s[1] ** 2 + s[2] ** 2 - rm**2

    earth.terminal = moon.terminal = True
    return [earth, moon]


def stm_propagate_guarded(x0, tau0, tau1, mu, du_km=384338.0):
    """cr3bp_tools.stm_propagate (same equations, DOP853 1e-12) that stops at an impact.
    Returns (x(tau1), Phi(tau1, tau0)); raises _Impact or RuntimeError."""
    x0 = np.asarray(x0, float)
    if tau1 == tau0:
        return x0.copy(), np.eye(6)
    sol = solve_ivp(_eom_stm, [tau0, tau1], np.concatenate([x0, np.eye(6).ravel()]), args=(mu,),
                    events=_impact_events(mu, du_km), **TOL)
    if sol.status == 1:
        raise _Impact("trajectory hits the Earth or the Moon")
    if not sol.success:
        raise RuntimeError(f"CR3BP STM propagation failed: {sol.message}")
    return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)


def propagate_guarded(x0, tau0, taus, mu, du_km=384338.0):
    """dynamics.propagate_cr3bp (same model and tolerances) that stops at an impact."""
    taus = np.asarray(taus, float)
    out = np.empty((len(taus), 6))
    for mask, end in ((taus >= tau0, taus.max()), (taus < tau0, taus.min())):
        if not mask.any():
            continue
        if end == tau0:
            out[mask] = x0
            continue
        sol = solve_ivp(cr3bp_eom, [tau0, end], np.asarray(x0, float), args=(mu,), dense_output=True,
                        events=_impact_events(mu, du_km), **TOL)
        if sol.status == 1:
            raise _Impact("trajectory hits the Earth or the Moon")
        if not sol.success:
            raise RuntimeError(f"CR3BP propagation failed: {sol.message}")
        out[mask] = sol.sol(taus[mask]).T
    return out


# ----------------------------------------------------------------------------- one triple

def central_difference_guess(tau, R, u, alphas):
    """Paper Eq. 24: chi0 = [alphas, v_j], v_j = (r(t_k) - r(t_i)) / (t_k - t_i), rotating frame."""
    alphas = np.asarray(alphas, float)
    v = ((R[2] + alphas[2] * u[2]) - (R[0] + alphas[0] * u[0])) / (tau[2] - tau[0])
    return np.concatenate([alphas, v])


def lagrange_guess(tau_all, R_all, u_all, j, alpha0, mu=0.0121505):
    """Extension of Eq. 24 to all N observations: chi0 = [alpha0, alpha0, alpha0, v_j].

    Eq. 24 is the derivative at t_j of the quadratic through three equal-range positions. Here
    the equal-range positions R_o(t_n) + alpha0 rho_n of ALL observations are turned into the
    Earth-centered inertial frame (rotating vectors times R(tau_n)), the degree N-1 polynomial
    through them is differentiated at t_j, and that inertial velocity is converted back to the
    rotating frame: v_j = R(tau_j)^T v_in - z x (r_j - r_Earth). Inertial positions avoid the
    rotating frame's apparent curvature, and more points follow the curvature of the arc."""
    tau_all = np.asarray(tau_all, float)
    e = np.array([-mu, 0.0, 0.0])
    P = np.array([rot_z(t) @ (R_all[n] + alpha0 * u_all[n] - e) for n, t in enumerate(tau_all)])
    dt = tau_all - tau_all[j]
    c = np.linalg.solve(np.vander(dt, len(dt), increasing=True), P)
    r_j = R_all[j] + alpha0 * u_all[j]
    v = rot_z(tau_all[j]).T @ c[1] - np.cross(Z_HAT, r_j - e)
    return np.concatenate([[alpha0] * 3, v])


def _residual_and_jacobian(chi, tau, R, u, mu, du_km):
    """F (6,) and DF (6, 6) of paper Eq. 22, plus the middle rotating state."""
    a = chi[:3]
    x_j = np.concatenate([R[1] + a[1] * u[1], chi[3:]])
    x_i, P_m = stm_propagate_guarded(x_j, tau[1], tau[0], mu, du_km)   # Phi- = Phi(t_i, t_j)
    x_k, P_p = stm_propagate_guarded(x_j, tau[1], tau[2], mu, du_km)   # Phi+ = Phi(t_k, t_j)
    F = np.concatenate([R[0] + a[0] * u[0] - x_i[:3], R[2] + a[2] * u[2] - x_k[:3]])
    DF = np.zeros((6, 6))
    DF[:3, 0] = u[0]
    DF[:3, 1] = -P_m[:3, :3] @ u[1]
    DF[:3, 3:] = -P_m[:3, 3:]
    DF[3:, 1] = -P_p[:3, :3] @ u[1]
    DF[3:, 2] = u[2]
    DF[3:, 3:] = -P_p[:3, 3:]
    return F, DF, x_j


def solve_dc(tau, R, u, alpha0=None, mu=0.0121505, chi0=None, tol=1e-12,
             accept_tol=1e-9, max_iter=50, bounds=(0.0, np.inf, np.inf), du_km=384338.0):
    """The paper's differential corrections on one triple (rotating, nondimensional).

    tau (3,), R (3, 3) observer positions, u (3, 3) unit lines of sight, for t_i < t_j < t_k.
    Starts from chi0, or from alpha0 (all three ranges equal) with the Eq. 24 velocity.
    bounds = (min range, max range, max speed) in DU
    and DU/TU. Never raises; returns a dict: converged, reason, iterations, F_norm, x_j (rotating
    state at tau[1] of the lowest-||F|| iterate) and alphas."""
    tau, R, u = np.asarray(tau, float), np.asarray(R, float), np.asarray(u, float)
    chi = np.array(chi0, float) if chi0 is not None else central_difference_guess(tau, R, u, [alpha0] * 3)
    lo, hi, vmax = bounds
    best = (np.inf, None, None)                   # (||F||, chi, x_j)
    reason, it = "max_iter", 0
    for it in range(max_iter + 1):
        if not np.all(np.isfinite(chi)):
            reason = "diverged"
            break
        if np.any(chi[:3] <= 0):
            reason = "nonpositive_range"
            break
        if np.any(chi[:3] < lo) or np.any(chi[:3] > hi) or np.linalg.norm(chi[3:]) > vmax:
            reason = "out_of_bounds"
            break
        try:
            F, DF, x_j = _residual_and_jacobian(chi, tau, R, u, mu, du_km)
        except _Impact:
            reason = "impact"
            break
        except (RuntimeError, ValueError, FloatingPointError):
            reason = "propagation_failed"
            break
        nF = float(np.linalg.norm(F))
        if not (np.isfinite(nF) and np.all(np.isfinite(DF))):
            reason = "propagation_failed"
            break
        if nF < best[0]:
            best = (nF, chi.copy(), x_j)
        if nF < tol:
            reason = "tol"
            break
        if it == max_iter:
            break
        try:
            chi = chi - np.linalg.solve(DF, F)
        except np.linalg.LinAlgError:
            reason = "singular_jacobian"
            break
    nF, chi_b, x_b = best
    converged = reason == "tol" or (chi_b is not None and nF < accept_tol)
    return {"converged": bool(converged), "reason": reason, "iterations": int(it), "F_norm": float(nF),
            "x_j": None if x_b is None else np.asarray(x_b), "alphas": None if chi_b is None else chi_b[:3].copy()}


def solve_trust_region(tau, R, u, chi0, mu=0.0121505, bounds=(0.0, np.inf, np.inf), du_km=384338.0,
                       max_nfev=40):
    """Globalized fallback (not in the paper): minimize ||F||^2 for the same F and the same
    Eq. 22 Jacobian with scipy's trust-region reflective least squares, the ranges boxed to
    [min, max] and each velocity component to [-max speed, max speed]. A trial point whose
    trajectory hits a primary gets a large constant residual, so the step is rejected.
    Returns the final chi, or None."""
    from scipy.optimize import least_squares

    tau, R, u = np.asarray(tau, float), np.asarray(R, float), np.asarray(u, float)
    lo, hi, vmax = bounds
    vbox = min(vmax, 1e3)
    lb = np.r_[[lo] * 3, [-vbox] * 3]
    ub = np.r_[[min(hi, 1e3)] * 3, [vbox] * 3]
    cache = {}

    def ev(c):
        key = c.tobytes()
        if key not in cache:
            cache.clear()
            try:
                F, DF, _ = _residual_and_jacobian(c, tau, R, u, mu, du_km)
                if not (np.all(np.isfinite(F)) and np.all(np.isfinite(DF))):
                    raise RuntimeError("non-finite")
            except (RuntimeError, ValueError, FloatingPointError):     # includes _Impact
                F, DF = np.full(6, 10.0), np.eye(6)
            cache[key] = (F, DF)
        return cache[key]

    x0 = np.clip(np.asarray(chi0, float), lb + 1e-12, ub - 1e-12)
    try:
        res = least_squares(lambda c: ev(c)[0], x0, jac=lambda c: ev(c)[1], bounds=(lb, ub), method="trf",
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_nfev)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return res.x


# ----------------------------------------------------------------------------- the IOD method

class CR3BPDifferentialCorrection:
    """Jo, Baker-McEvilly & Canales (2025) differential-corrections IOD, multi-started in range
    (paper seed plus an all-observation seed with a trust-region fallback) and disambiguated
    by the residuals of all observations. See the module docstring for every assumption."""

    name = "cr3bp_dc"

    def __init__(self, sys: EarthMoonSystem, obs_indices=(0, 2, 4), alpha0_km=None,
                 alpha0_range_km=(200_000.0, 500_000.0), alpha0_step_km=10_000.0,
                 seeds=("paper", "lagrange"), trust_region: bool = True, tr_max_nfev: int = 40,
                 tol: float = 1e-12, accept_tol: float = 1e-9, max_iter: int = 50,
                 min_range_km: float = 20_000.0, max_range_km: float = 1_500_000.0, max_speed_km_s: float = 3.0,
                 dedup_km: float = 100.0, verify: bool = True, fit_gate_sigma: float = 3.0, **_):
        self.sys = sys
        self.idx = [int(n) for n in obs_indices]
        if len(self.idx) != 3:
            raise ValueError("cr3bp_dc uses exactly 3 observations")
        if alpha0_km is None:
            lo, hi = alpha0_range_km
            alpha0_km = np.arange(lo, hi + 0.5 * alpha0_step_km, alpha0_step_km)
        self.alpha0_km = [float(a) for a in np.atleast_1d(alpha0_km)]
        self.seeds = [seeds] if isinstance(seeds, str) else list(seeds)
        if not set(self.seeds) <= {"paper", "lagrange"}:
            raise ValueError(f"unknown seeds {self.seeds}")
        self.trust_region, self.tr_max_nfev = bool(trust_region), int(tr_max_nfev)
        self.tol, self.accept_tol, self.max_iter = float(tol), float(accept_tol), int(max_iter)
        du = sys.du_km
        self.bounds = (float(min_range_km) / du, float(max_range_km) / du, float(max_speed_km_s) * 1e3 / sys.vu_ms)
        self.dedup_km, self.verify, self.gate = float(dedup_km), bool(verify), float(fit_gate_sigma)

    # ------------------------------------------------------------------ helpers
    def _solve(self, tau, R, u, triple, **kw):
        return solve_dc(tau[triple], R[triple], u[triple], mu=self.sys.mu, tol=self.tol, accept_tol=self.accept_tol,
                        max_iter=self.max_iter, bounds=self.bounds, du_km=self.sys.du_km, **kw)

    def _start(self, tau, R, u, a0, seed):
        """One start: (result dict, how it converged)."""
        tri = self.idx
        if seed == "paper":
            return self._solve(tau, R, u, tri, alpha0=a0), "newton"
        chi0 = lagrange_guess(tau, R, u, tri[1], a0, self.sys.mu)
        res = self._solve(tau, R, u, tri, chi0=chi0)
        if res["converged"] or not self.trust_region:
            return res, "newton"
        chi = solve_trust_region(tau[tri], R[tri], u[tri], chi0, self.sys.mu, self.bounds, self.sys.du_km,
                                 self.tr_max_nfev)
        if chi is None:
            return res, "newton"
        polished = self._solve(tau, R, u, tri, chi0=chi)     # Newton polish to tol
        return polished, "trust_region"

    def _add(self, sols, res, start):
        """Add a converged result to sols unless one within dedup_km at t_j is already there."""
        for s in sols:
            if np.linalg.norm(s["x_j"][:3] - res["x_j"][:3]) * self.sys.du_km < self.dedup_km:
                s["starts"].append(start)
                if res["F_norm"] < s["F_norm"]:
                    s.update({k: res[k] for k in ("x_j", "alphas", "F_norm", "iterations")})
                return
        sols.append({**res, "starts": [start]})

    def _rms(self, obs: Observations, tau, tau_j, x_j) -> float:
        """RA/Dec residual RMS (sigma) of the CR3BP trajectory through x_j at every observation."""
        from .results import residuals

        try:
            X = propagate_guarded(x_j, tau_j, tau, self.sys.mu, self.sys.du_km)
        except RuntimeError:
            return float("inf")
        rv = [rotating_to_emi(t, x, self.sys) for t, x in zip(tau, X)]
        traj = Trajectory(t=obs.t, r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv]))
        rms = residuals(obs, traj)["rms_normalized"]
        return float(rms) if np.isfinite(rms) else float("inf")

    def _verify(self, tau, R, u, best):
        """Re-solve every other consecutive triple, seeded from the chosen solution."""
        out = []
        j = self.idx[1]
        for a in range(len(tau) - 2):
            tri = [a, a + 1, a + 2]
            if tri == self.idx:
                continue
            entry = {"triple": tri, "converged": False}
            try:
                X = propagate_guarded(best["x_j"], tau[j], tau[tri], self.sys.mu, self.sys.du_km)
            except RuntimeError:
                out.append(entry)
                continue
            chi0 = np.concatenate([np.linalg.norm(X[:, :3] - R[tri], axis=1), X[1, 3:]])
            res = self._solve(tau, R, u, tri, chi0=chi0)
            entry.update(converged=res["converged"], iterations=res["iterations"], reason=res["reason"])
            if res["converged"]:
                try:
                    xb = propagate_guarded(res["x_j"], tau[tri[1]], tau[[j]], self.sys.mu, self.sys.du_km)[0]
                    entry["position_diff_km"] = float(np.linalg.norm(xb[:3] - best["x_j"][:3]) * self.sys.du_km)
                    entry["velocity_diff_m_s"] = float(np.linalg.norm(xb[3:] - best["x_j"][3:]) * self.sys.vu_ms)
                except RuntimeError:
                    entry["converged"] = False
            out.append(entry)
        return out

    # ------------------------------------------------------------------ IOD interface
    def estimate(self, obs: Observations) -> StateEstimate:
        t_start = time.perf_counter()
        if max(self.idx) >= len(obs) or min(self.idx) < 0:
            raise IODFailure(f"{self.name}: needs observations {self.idx}, got {len(obs)}")
        i, j, k = self.idx
        if not obs.t[i] < obs.t[j] < obs.t[k]:
            raise IODFailure(f"{self.name}: observation times must increase over {self.idx}")
        tau, R, u = obs_to_rotating(obs, self.sys)

        sols, outcomes = [], {}
        for seed in self.seeds:
            for a0 in self.alpha0_km:
                res, how = self._start(tau, R, u, a0 * 1e3 / self.sys.du_m, seed)
                key = f"{seed}:{how if res['converged'] else res['reason']}"
                outcomes[key] = outcomes.get(key, 0) + 1
                if res["converged"]:
                    self._add(sols, res, f"{seed}/{how}/{a0:.0f}")
        if not sols:
            raise IODFailure(f"{self.name}: no start converged on {self.idx} (outcomes {outcomes})")

        for s in sols:
            s["rms_sigma"] = self._rms(obs, tau, tau[j], s["x_j"])
        sols.sort(key=lambda s: s["rms_sigma"])
        best = sols[0]
        r, v = rotating_to_emi(tau[j], best["x_j"], self.sys)

        def num(x):                                    # JSON-safe float (inf/nan -> None)
            return float(x) if np.isfinite(x) else None

        info = {
            "obs_indices": self.idx,
            "alpha0_km": self.alpha0_km,
            "seeds": self.seeds,
            "n_starts": len(self.alpha0_km) * len(self.seeds),
            "start_outcomes": outcomes,
            "n_distinct_solutions": len(sols),
            "n_fitting_solutions": int(sum(s["rms_sigma"] <= self.gate for s in sols)),
            "fit_gate_sigma": self.gate,
            "solutions": [{
                "rms_sigma": num(s["rms_sigma"]),
                "ranges_km": [float(a * self.sys.du_km) for a in s["alphas"]],
                "x_rot": [float(c) for c in s["x_j"]],
                "F_norm_du": float(s["F_norm"]),
                "iterations": int(s["iterations"]),
                "starts": s["starts"],
            } for s in sols],
        }
        if self.verify:
            info["verification"] = self._verify(tau, R, u, best)
        info["runtime_s"] = time.perf_counter() - t_start
        return StateEstimate(t=float(obs.t[j]), r=r, v=v, method=self.name, info=info)

"""
Laplace-type initial orbit determination with the Moon included (Part B).

    PerturbedLaplace  "perturbed_laplace"  Laplace's method with Earth + Moon gravity, solved numerically
    LaplaceMoon       "laplace_moon"       Orekit's IodLaplace with the Moon as the central body
    GoodingMoon       "gooding_moon"       Orekit's IodGooding with the Moon as the central body

Shared conventions (contracts.py, frames.py): EMI frame (Earth-centered, non-rotating, x toward
the Moon at t0, z normal to the Earth-Moon plane), SI units, time in seconds since t0. The Moon is
the CR3BP Moon, exactly as orekit_models.py models it: a circle of radius DU about Earth's
center, angular rate 1/TU, on +x at t0:

    r_M(t) = DU [cos(t/TU), sin(t/TU), 0],   v_M = (DU/TU) [-sin, cos, 0],   a_M = -r_M / TU^2.

None of these methods reads the truth file. Candidate solutions are ranked by how well they fit
the observations (RA/Dec residuals after a CR3BP propagation), never by comparison with truth.

Why the Moon matters here (finding, see tests/test_iod_laplace.py): with the observer at Earth's
center, Earth's own gravity is parallel to the line of sight and drops out of Laplace's range
equation entirely (this is why the classical two-body Laplace/Gauss fail with no parallax). The
only range information left is the Moon's tidal pull, a small signal in the line of sight's
second derivative. The method therefore needs a short, dense arc with small angle noise.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import brentq

from .contracts import Observations, StateEstimate, Trajectory
from .cr3bp_tools import los_derivatives
from .frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi
from .iod import IODFailure, _OrekitAnglesOnlyIOD

ARCSEC = math.pi / (180 * 3600)


# ----------------------------------------------------------------------------- shared helpers

def moon_state(t: float, sys: EarthMoonSystem):
    """CR3BP Moon in EMI at t seconds: (position [m], velocity [m/s], acceleration [m/s^2])."""
    th = t / sys.tu_s
    c, s = math.cos(th), math.sin(th)
    r = sys.du_m * np.array([c, s, 0.0])
    v = sys.vu_ms * np.array([-s, c, 0.0])
    return r, v, -r / sys.tu_s**2


def earth_moon_accel(r: np.ndarray, r_moon: np.ndarray, sys: EarthMoonSystem) -> np.ndarray:
    """Acceleration [m/s^2] of an object at r (EMI, Earth-centered) under Earth + Moon gravity,
    including the indirect term (the Moon's pull on Earth), as Orekit's ThirdBodyAttraction does:
        a = -GM_E r/|r|^3 - GM_M [ (r - r_M)/|r - r_M|^3 + r_M/|r_M|^3 ].
    With the circular CR3BP Moon this is exactly the CR3BP written in Earth-centered axes."""
    d = r - r_moon
    return (-sys.gm_earth * r / np.linalg.norm(r) ** 3
            - sys.gm_moon * (d / np.linalg.norm(d) ** 3 + r_moon / np.linalg.norm(r_moon) ** 3))


def poly_derivatives(t: np.ndarray, Y: np.ndarray, t_eval: float):
    """Value, first and second time derivative at t_eval of the Lagrange polynomial through
    the rows of Y (one row per time in t). Used for the observer position (not normalized)."""
    tt = np.asarray(t, float) - t_eval
    n = len(tt)
    c = np.linalg.solve(np.vander(tt, n, increasing=True), np.asarray(Y, float))
    zero = np.zeros(c.shape[1])
    return c[0], (c[1] if n > 1 else zero), (2 * c[2] if n > 2 else zero)


def observer_derivatives(obs: Observations, idx, t_eval: float):
    """Observer R, R', R'' (EMI) at t_eval. Exactly zero for the geocenter. Otherwise a
    Lagrange fit through the observer positions of the chosen observations: fine for an
    observer that moves smoothly on the scale of the arc (a space-based sensor, or a dense
    arc), wrong for a ground station sampled days apart (Earth's rotation aliases)."""
    P = obs.observer[list(idx)]
    if np.all(P == 0):
        z = np.zeros(3)
        return z, z.copy(), z.copy()
    return poly_derivatives(obs.t[list(idx)], P, t_eval)


EARTH_RADIUS_M = 6378.137e3
MOON_RADIUS_M = 1737.4e3


def _propagate_with_collisions(x0, tau0: float, taus: np.ndarray, sys: EarthMoonSystem) -> np.ndarray:
    """dynamics.propagate_cr3bp plus terminal events at the Earth and Moon surfaces: a wild IOD
    candidate that dives into a primary would otherwise make DOP853 at 1e-12 crawl for minutes.
    Same equations and tolerances. Raises RuntimeError on a collision before the last time."""
    from scipy.integrate import solve_ivp

    from .dynamics import TOL, cr3bp_eom

    mu = sys.mu
    r_e, r_m = EARTH_RADIUS_M / sys.du_m, MOON_RADIUS_M / sys.du_m

    def hit_earth(t, s, mu):
        return (s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2 - r_e**2

    def hit_moon(t, s, mu):
        return (s[0] - 1 + mu) ** 2 + s[1] ** 2 + s[2] ** 2 - r_m**2

    hit_earth.terminal = hit_moon.terminal = True
    out = np.empty((len(taus), 6))
    for mask, end in ((taus >= tau0, taus.max()), (taus < tau0, taus.min())):
        if not mask.any():
            continue
        if end == tau0:
            out[mask] = x0
            continue
        sol = solve_ivp(cr3bp_eom, [tau0, end], np.asarray(x0, float), args=(mu,), dense_output=True,
                        events=(hit_earth, hit_moon), **TOL)
        if sol.status != 0:
            raise RuntimeError(f"CR3BP propagation stopped: {sol.message}")
        out[mask] = sol.sol(taus[mask]).T
    return out


def fit_residuals(obs: Observations, t: float, r: np.ndarray, v: np.ndarray, sys: EarthMoonSystem) -> dict:
    """Propagate the EMI state (t, r, v) in the CR3BP to every observation time and return the
    RA/Dec residual RMS: 'rms_arcsec' (plain RA and Dec differences, as results.residuals and
    Orekit's AngularRaDec use) and 'rms_normalized' (in sigma; None when sigma is zero).
    A candidate that starts inside or hits the Earth or the Moon, or fails to propagate, gets
    rms = inf."""
    from .results import residuals

    if np.linalg.norm(r) <= EARTH_RADIUS_M or np.linalg.norm(r - moon_state(t, sys)[0]) <= MOON_RADIUS_M:
        return {"rms_arcsec": math.inf, "rms_normalized": None}       # starts inside a primary
    try:
        tau0 = float(sys.tau(t))
        taus = np.asarray(sys.tau(obs.t), float)
        X = _propagate_with_collisions(emi_to_rotating(tau0, r, v, sys), tau0, taus, sys)
        rv = [rotating_to_emi(tau, x, sys) for tau, x in zip(taus, X)]
        traj = Trajectory(t=obs.t, r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv]))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN when sigma = 0
            res = residuals(obs, traj)
        rms = float(np.sqrt(np.mean(np.concatenate([res["ra_arcsec"], res["dec_arcsec"]]) ** 2)))
        norm = res["rms_normalized"]
        return {"rms_arcsec": rms if np.isfinite(rms) else math.inf,
                "rms_normalized": float(norm) if np.isfinite(norm) else None}
    except Exception:
        return {"rms_arcsec": math.inf, "rms_normalized": None}


def _json_safe(x):
    """Replace non-finite floats (inf RMS of a rejected candidate) by None, recursively."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _resolve_indices(obs_indices, n_obs: int, name: str) -> list:
    """(i, j, k, ...) or 'all'. Raises IODFailure when the observations do not cover them."""
    idx = list(range(n_obs)) if isinstance(obs_indices, str) and obs_indices == "all" else [int(i) for i in obs_indices]
    if len(idx) < 3:
        raise IODFailure(f"{name}: needs at least 3 observations, got indices {idx}")
    if max(idx) >= n_obs or min(idx) < -n_obs:
        raise IODFailure(f"{name}: observation indices {idx} out of range for {n_obs} observations")
    return [i % n_obs for i in idx]


# ----------------------------------------------------------------------------- method 1

class PerturbedLaplace:
    """Laplace's angles-only IOD generalized to Earth + Moon gravity (an inertial Earth-centered
    frame, CR3BP-consistent), with the range found numerically and every root kept.

    This is our own generalization written in the spirit of Liu, Hou & Armellin (2025), "perturbed
    Laplace" IOD for cislunar space (Advances in Space Research, doi:10.1016/j.asr.2025.05.094).
    That paper is paywalled and was not consulted, so nothing here should be read as its algorithm.

    Algorithm, at the middle epoch t_j of the chosen observations:
      1. rho_hat, rho_hat', rho_hat'' from cr3bp_tools.los_derivatives (Lagrange polynomial through
         the chosen observations, degree len(idx) - 1).
      2. Observer R, R', R'' (zero for the geocenter; Lagrange fit otherwise, see observer_derivatives).
      3. With r = R + rho rho_hat, r'' = a(r) (earth_moon_accel) gives
             rho'' rho_hat + 2 rho' rho_hat' + rho rho_hat'' = a(R + rho rho_hat) - R''.
         Dotting with n = rho_hat x rho_hat' removes rho'' and rho':
             g(rho) = rho (n . rho_hat'') - n . (a(R + rho rho_hat) - R'') = 0.
         g is scanned on a log grid over [rho_min_km, rho_max_km] (plus the point of closest
         approach of the line of sight to the Moon), every sign change is refined with brentq.
      4. For each root, dotting with m = rho_hat x rho_hat'' gives rho' = m.(a - R'') / (2 m.rho_hat'),
         and v = R' + rho' rho_hat + rho rho_hat'.
      5. Each candidate is propagated in the CR3BP (same equations and tolerances as
         dynamics.propagate_cr3bp, plus collision stops) to all observations; the one with the smallest
         RA/Dec residual RMS is returned. All roots and their RMS are stored in info["roots"].

    Assumptions:
      - The dynamics are the CR3BP (circular Moon, radius DU, rate 1/TU, on +x at t0; GM values
        from sys). No solar or other perturbations.
      - The line-of-sight derivatives from a Lagrange fit are accurate. This holds only for a
        short, dense arc: on the 2-day spacing of data/synthetic_obs.csv the fit's truncation error
        in rho_hat'' is larger than (and of opposite sign to) the lunar signal, so no root exists.
      - The observer at Earth's center makes Earth's gravity drop out of g (it is parallel to
        rho_hat): the range comes only from the Moon's term, which makes the method very
        sensitive to angle noise. The Moon's term is a bump centred where the line of sight passes
        closest to the Moon, and the roots usually sit on its two flanks; for this L1 halo that
        point is itself only 3,000-8,000 km from the truth, so landing within ~10,000 km is partly
        geometry, not proof that the range was measured. Measured (10 seeds, tests/report): 10"
        noise gives no root or a >10,000 km answer on most 6-hour and 1-day arcs.
      - A nonzero observer is supported through a Lagrange fit of its positions (see
        observer_derivatives); that is untested for ground stations and wrong for sparse arcs.
      - Roots are only searched between rho_min_km and rho_max_km (default 1,000 to 1,000,000 km);
        a root closer than one grid step to another may be missed.

    Options: obs_indices (default (0, 2, 4); 'all' for every observation, e.g. a degree-4 fit
    through 5), rho_min_km, rho_max_km, n_grid. Raises IODFailure when g has no root,
    or when every root gives an orbit that hits the Earth or the Moon.
    """

    name = "perturbed_laplace"

    def __init__(self, sys: EarthMoonSystem, obs_indices=(0, 2, 4), rho_min_km: float = 1e3,
                 rho_max_km: float = 1e6, n_grid: int = 4000, **_):
        self.sys = sys
        self.obs_indices = obs_indices
        self.rho_min, self.rho_max = float(rho_min_km) * 1e3, float(rho_max_km) * 1e3
        self.n_grid = int(n_grid)

    def geometry(self, obs: Observations):
        """Everything g(rho) needs at the middle epoch: (idx, t_j, u, u', u'', R, R', R'', r_M)."""
        idx = _resolve_indices(self.obs_indices, len(obs), self.name)
        t_j = float(obs.t[idx[len(idx) // 2]])
        u, du, d2u = los_derivatives(obs, idx, t_j)
        R, dR, d2R = observer_derivatives(obs, idx, t_j)
        r_m = moon_state(t_j, self.sys)[0]
        return idx, t_j, u, du, d2u, R, dR, d2R, r_m

    def roots(self, obs: Observations):
        """All ranges rho [m] with g(rho) = 0, and the geometry they came from."""
        geo = self.geometry(obs)
        _, _, u, du, d2u, R, _, d2R, r_m = geo
        n = np.cross(u, du)
        n_d2u = float(n @ d2u)

        def g(rho):
            return rho * n_d2u - n @ (earth_moon_accel(R + rho * u, r_m, self.sys) - d2R)

        grid = np.geomspace(self.rho_min, self.rho_max, self.n_grid)
        rho_ca = float(u @ (r_m - R))                       # closest approach of the LOS to the Moon
        if self.rho_min < rho_ca < self.rho_max:
            grid = np.unique(np.append(grid, rho_ca))
        gv = np.array([g(x) for x in grid])
        roots = []
        for i in np.nonzero(np.sign(gv[:-1]) * np.sign(gv[1:]) <= 0)[0]:
            if gv[i] == 0:
                roots.append(float(grid[i]))
            elif gv[i + 1] != 0:
                roots.append(float(brentq(g, grid[i], grid[i + 1], xtol=1e-3, rtol=1e-14)))
        return roots, geo

    def state_for_range(self, rho: float, geo) -> tuple:
        """(r, v) in EMI at t_j for a range rho on the line of sight (step 4)."""
        _, _, u, du, d2u, R, dR, d2R, r_m = geo
        r = R + rho * u
        m = np.cross(u, d2u)
        rho_dot = float(m @ (earth_moon_accel(r, r_m, self.sys) - d2R)) / (2.0 * float(m @ du))
        return r, dR + rho_dot * u + rho * du

    def estimate(self, obs: Observations) -> StateEstimate:
        roots, geo = self.roots(obs)
        idx, t_j = geo[0], geo[1]
        if not roots:
            raise IODFailure(f"{self.name}: no range root between {self.rho_min / 1e3:g} and "
                             f"{self.rho_max / 1e3:g} km (line-of-sight derivatives too poor for this arc?)")
        cands = []
        for rho in roots:
            r, v = self.state_for_range(rho, geo)
            fit = fit_residuals(obs, t_j, r, v, self.sys)
            cands.append({"rho_km": rho / 1e3, "r": r, "v": v, **fit})
        best = min(range(len(cands)), key=lambda i: cands[i]["rms_arcsec"])
        c = cands[best]
        if not math.isfinite(c["rms_arcsec"]):
            raise IODFailure(f"{self.name}: every root ({[round(k['rho_km']) for k in cands]} km) gives an "
                             "orbit that hits the Earth or the Moon, or fails to propagate")
        return StateEstimate(
            t=t_j, r=c["r"], v=c["v"], method=self.name,
            info=_json_safe({"obs_indices": [int(i) for i in idx],
                  "epoch_index": int(idx[len(idx) // 2]),
                  "chosen_root": int(best),
                  "rms_arcsec": c["rms_arcsec"],
                  "roots": [{"rho_km": k["rho_km"], "rms_arcsec": k["rms_arcsec"],
                             "rms_normalized": k["rms_normalized"]} for k in cands]}),
        )


# ----------------------------------------------------------------------------- methods 2 and 3

class _MoonCenteredOrekitIOD(_OrekitAnglesOnlyIOD):
    """Orekit's two-body angles-only IOD run about the MOON instead of Earth.

    Observer positions (and, for Laplace, velocity and acceleration) are expressed relative to the
    CR3BP Moon, which is a pure translation, so the lines of sight are unchanged. The frame passed
    to Orekit is labelled GCRF (any inertial label works for two-body IOD). Orekit's result is
    relative to the Moon; the Moon's position and velocity at the result epoch are added back to
    return to EMI. Candidates (several range guesses for Gooding) are ranked by their RA/Dec
    residual RMS after a CR3BP propagation over all observations (fit_residuals).

    Assumptions: the object moves on a two-body orbit about the Moon (mu = sys.gm_moon) between
    the three observations, i.e. Earth's tidal pull (and the Moon-relative frame's acceleration
    for the object) is ignored. That is a poor model for an L1 halo (~60,000-90,000 km from the
    Moon, where Earth's tide is comparable to lunar gravity), so expect partial success at best.
    """

    def _relative_observers(self, obs: Observations, idx):
        """Observer positions relative to the Moon at each chosen observation."""
        return [obs.observer[n] - moon_state(float(obs.t[n]), self.sys)[0] for n in idx]

    def _candidates(self, dates, los, obs: Observations, idx):
        """Yield (label, Orekit orbit or exception) per attempt."""
        raise NotImplementedError

    def estimate(self, obs: Observations) -> StateEstimate:
        idx = _resolve_indices(self.idx, len(obs), self.name)
        dates = [self.t0.shiftedBy(float(obs.t[n])) for n in idx]
        los = [self._vec(obs.los[n]) for n in idx]
        attempts, best, scored = [], None, []
        for label, orbit in self._candidates(dates, los, obs, idx):
            if isinstance(orbit, Exception) or orbit is None:
                msg = "no solution returned" if orbit is None else str(orbit).splitlines()[0]
                attempts.append({"attempt": label, "error": msg})
                continue
            pv = orbit.getPVCoordinates(self.frame)
            p, v = pv.getPosition(), pv.getVelocity()
            t = float(orbit.getDate().durationFrom(self.t0))
            r_m, v_m, _ = moon_state(t, self.sys)
            r = np.array([p.getX(), p.getY(), p.getZ()]) + r_m
            vel = np.array([v.getX(), v.getY(), v.getZ()]) + v_m
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(vel))):
                attempts.append({"attempt": label, "error": "non-finite state"})
                continue
            dup = next((a for a in scored if a[0] == t and np.allclose(a[1], r, rtol=0, atol=1.0)
                        and np.allclose(a[2], vel, rtol=0, atol=1e-6)), None)
            fit = dup[3] if dup else fit_residuals(obs, t, r, vel, self.sys)
            scored.append((t, r, vel, fit))
            attempts.append({"attempt": label, "t_s": t, "range_from_moon_km": float(np.linalg.norm(r - r_m) / 1e3), **fit})
            if best is None or fit["rms_arcsec"] < best[3]["rms_arcsec"]:
                best = (t, r, vel, fit, label)
        if best is None or not math.isfinite(best[3]["rms_arcsec"]):
            raise IODFailure(f"{self.name}: no attempt gave an orbit that propagates through the arc "
                             f"without hitting the Earth or the Moon: {_json_safe(attempts)}")
        t, r, vel, fit, label = best
        return StateEstimate(t=t, r=r, v=vel, method=self.name,
                             info=_json_safe({"obs_indices": [int(i) for i in idx], "central_body": "Moon",
                                   "chosen_attempt": label, "rms_arcsec": fit["rms_arcsec"],
                                   "attempts": attempts}))


class LaplaceMoon(_MoonCenteredOrekitIOD):
    """Orekit IodLaplace on observations re-expressed relative to the Moon.

    Orekit's Laplace needs the observer's position, velocity and acceleration at the middle
    observation; relative to the Moon these are R - r_M, R' - v_M, R'' - a_M (R from
    observer_derivatives: exactly zero at the geocenter). Orekit fits its own 3-point Lagrange
    line-of-sight derivatives.

    Two manipulations are needed, both found by reading Orekit 13.1.9's IodLaplace source:
      1. Units. Orekit returns null when |D| = 2|det(L, L', L'')| < 1e-14, an absolute threshold.
         In seconds, a cislunar line of sight turns so slowly that D ~ 1e-17 s^-3, so Orekit
         always gives up. The problem is passed in CR3BP units instead (lengths / DU, times / TU,
         mu = GM_Moon TU^2 / DU^3), which leaves the two-body equations unchanged, and the result
         is scaled back. Set time_unit_s / length_unit_m to change this (1.0, 1.0 = plain SI).
      2. Root choice. Orekit keeps the LARGEST positive real root r of its degree-8 polynomial.
         When the object is much closer to the central body than the observer (an object near
         the Moon seen from Earth, ~384,000 km away), that is a spurious root near the observer's
         distance. With all_roots=True (default) every positive real root of the same polynomial
         (Orekit's coefficients, re-implemented in numpy here) is also turned into a candidate
         with Orekit's own range/range-rate formulas; candidates are ranked by CR3BP residual RMS.
         info["attempts"] says which candidate is Orekit's own answer.

    See the class above for the two-body-about-the-Moon assumption. Options: obs_indices (3 indices),
    time_unit_s, length_unit_m, all_roots."""

    name = "laplace_moon"

    def __init__(self, sys, time_unit_s: float | None = None, length_unit_m: float | None = None,
                 all_roots: bool = True, **kw):
        super().__init__(sys, **kw)
        self.T = float(time_unit_s) if time_unit_s else sys.tu_s
        self.L = float(length_unit_m) if length_unit_m else sys.du_m
        self.all_roots = bool(all_roots)

    def _candidates(self, dates, los, obs, idx):
        from org.orekit.estimation.iod import IodLaplace
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.utils import PVCoordinates

        T, L = self.T, self.L
        t_j = float(obs.t[idx[1]])
        R, dR, d2R = observer_derivatives(obs, idx, t_j)
        r_m, v_m, a_m = moon_state(t_j, self.sys)
        Ro, dRo, d2Ro = R - r_m, dR - v_m, d2R - a_m           # observer relative to the Moon, SI
        mu = self.sys.gm_moon * T**2 / L**3
        # Scaled dates: t' = t_first + (t - t_first) / T (only differences matter to Laplace).
        t_first = float(obs.t[idx[0]])
        sdates = [self.t0.shiftedBy(t_first + (float(obs.t[n]) - t_first) / T) for n in idx]
        pva = PVCoordinates(self._vec(Ro / L), self._vec(dRo * T / L), self._vec(d2Ro * T**2 / L))

        def unscale(orbit):
            """Scaled Orekit orbit -> SI orbit at the true middle date (Moon-centered)."""
            if orbit is None:
                return None
            pv = orbit.getPVCoordinates(self.frame)
            p, v = pv.getPosition(), pv.getVelocity()
            return CartesianOrbit(PVCoordinates(self._vec(np.array([p.getX(), p.getY(), p.getZ()]) * L),
                                                self._vec(np.array([v.getX(), v.getY(), v.getZ()]) * L / T)),
                                  self.frame, dates[1], self.sys.gm_moon)

        try:
            yield "orekit", unscale(IodLaplace(mu).estimate(
                self.frame, pva, sdates[0], los[0], sdates[1], los[1], sdates[2], los[2]))
        except Exception as e:
            yield "orekit", e
        if not self.all_roots:
            return
        # Every positive real root of Orekit's polynomial, with Orekit's own formulas (scaled units).
        ts = [(float(obs.t[n]) - t_first) / T for n in idx]
        l1, l2, l3 = (obs.los[n] for n in idx)
        t2, t3 = ts[1], ts[2]
        Ld = l1 * (t2 - t3) / (t2 * t3) + l2 * (2 * t2 - t3) / (t2 * (t2 - t3)) + l3 * t2 / (t3 * (t3 - t2))
        Ldd = l1 * 2 / (t2 * t3) + l2 * 2 / (t2 * (t2 - t3)) + l3 * 2 / (t3 * (t3 - t2))
        P, dP, d2P = Ro / L, dRo * T / L, d2Ro * T**2 / L
        det = lambda a, b, c: float(np.linalg.det(np.array([a, b, c])))
        D = 2 * det(l2, Ld, Ldd)
        if D == 0:
            return
        D1, D2 = det(l2, Ld, d2P), det(l2, Ld, P)
        Rn, RdotL = np.linalg.norm(P), float(P @ l2)
        coeff = [1.0, 0, 4 * D1 * RdotL / D - 4 * D1**2 / D**2 - Rn**2, 0, 0,
                 4 * mu * D2 * (RdotL / D - 2 * D1 / D**2), 0, 0, -4 * mu**2 * D2**2 / D**2]
        for k, root in enumerate(np.roots(coeff)):
            if abs(root.imag) > 1e-8 * abs(root) or root.real <= 0:
                continue
            r3 = root.real**3
            rho = -2 * D1 / D - 2 * mu * D2 / (D * r3)
            rho_dot = -det(l2, d2P, Ldd) / D - mu * det(l2, P, Ldd) / (D * r3)
            pos, vel = (P + rho * l2) * L, (dP + rho_dot * l2 + rho * Ld) * L / T
            yield f"polynomial root r={root.real * L / 1e3:.0f} km (numpy replica)", CartesianOrbit(
                PVCoordinates(self._vec(pos), self._vec(vel)), self.frame, dates[1], self.sys.gm_moon)


class GoodingMoon(_MoonCenteredOrekitIOD):
    """Orekit IodGooding(GM_Moon) on observations re-expressed relative to the Moon.

    Gooding needs initial guesses of the ranges (from the OBSERVER, i.e. from Earth's center for
    this data) at the first and last observation. Every pair in range_guesses_km is tried and the
    solution with the smallest CR3BP residual RMS is kept (all attempts are in info["attempts"]).
    Zero-revolution transfers only; short way by default (directions=(True, False) adds the long
    way, which Orekit 13.1.9 failed on in every case we tried). See the class above for the
    two-body-about-the-Moon assumption. Options: obs_indices (3 indices), range_guesses_km (list
    of (rho1, rho3) in km), directions."""

    name = "gooding_moon"

    def __init__(self, sys, range_guesses_km=((350_000.0, 350_000.0), (300_000.0, 300_000.0),
                                               (400_000.0, 400_000.0), (250_000.0, 450_000.0),
                                               (450_000.0, 250_000.0)), directions=(True,), **kw):
        super().__init__(sys, **kw)
        self.guesses = [(float(a) * 1e3, float(b) * 1e3) for a, b in range_guesses_km]
        self.directions = tuple(bool(d) for d in directions)

    def _candidates(self, dates, los, obs, idx):
        from org.orekit.estimation.iod import IodGooding

        O = [self._vec(o) for o in self._relative_observers(obs, idx)]
        for rho1, rho3 in self.guesses:
            for prograde in self.directions:
                label = f"rho_guess_km=({rho1 / 1e3:g}, {rho3 / 1e3:g}) {'short' if prograde else 'long'} way"
                try:
                    yield label, IodGooding(self.sys.gm_moon).estimate(
                        self.frame, O[0], O[1], O[2], los[0], dates[0], los[1], dates[1], los[2], dates[2],
                        rho1, rho3, 0, bool(prograde))
                except Exception as e:
                    yield label, e


IOD_LAPLACE_METHODS = {
    PerturbedLaplace.name: PerturbedLaplace,
    LaplaceMoon.name: LaplaceMoon,
    GoodingMoon.name: GoodingMoon,
}

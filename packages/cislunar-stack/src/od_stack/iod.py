"""
IOD methods. Each one takes Observations and returns a StateEstimate.

Adding a method (e.g. the collocation IOD from the paper):
    1. Write a class with a `name`, a `config_block` (its settings section under iod: in
       config.yaml), and an `estimate(obs) -> StateEstimate` method.
    2. Add it to IOD_METHODS at the bottom of this file.
    3. Select it in config.yaml (iod.method) and add its settings block.

The Orekit methods below are two-body (Earth only). They are baselines: expect
large errors in cislunar space, and Gauss/Laplace fail outright when the observer
is at Earth's center (no parallax).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import Observations, StateEstimate
from .frames import EarthMoonSystem
from .orekit_setup import start_orekit


class IODFailure(RuntimeError):
    pass


class IODMethod(Protocol):
    name: str

    def estimate(self, obs: Observations) -> StateEstimate: ...


class _OrekitAnglesOnlyIOD:
    """Shared plumbing: pick 3 observations, convert numpy <-> Orekit, convert the result back."""

    name = "orekit-base"
    config_block = "orekit"           # reads iod.orekit in config.yaml

    def __init__(self, sys: EarthMoonSystem, obs_indices=(0, 2, 4), orekit_data: str | None = None, **_):
        self.sys = sys
        self.idx = list(obs_indices)
        if len(self.idx) != 3:
            raise ValueError("Orekit angles-only IOD uses exactly 3 observations")
        start_orekit(orekit_data)

        from org.orekit.frames import FramesFactory
        from org.orekit.time import AbsoluteDate

        # The EMI frame is not a standard Orekit frame. For two-body IOD any
        # inertial frame label works, so GCRF stands in for EMI here. Moving to real
        # ephemerides will need the true EMI -> GCRF rotation and a real epoch.
        self.frame = FramesFactory.getGCRF()
        self.t0 = AbsoluteDate.J2000_EPOCH

    def _vec(self, a):
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        return Vector3D(float(a[0]), float(a[1]), float(a[2]))

    def estimate(self, obs: Observations) -> StateEstimate:
        i, j, k = self.idx
        dates = [self.t0.shiftedBy(float(obs.t[n])) for n in (i, j, k)]
        los = [self._vec(obs.los[n]) for n in (i, j, k)]
        obsp = [self._vec(obs.observer[n]) for n in (i, j, k)]
        try:
            orbit = self._solve(dates, los, obsp)
        except Exception as e:  # Java exceptions arrive here
            raise IODFailure(f"{self.name}: {str(e).splitlines()[0]}") from e
        if orbit is None:
            raise IODFailure(f"{self.name}: no solution returned")

        pv = orbit.getPVCoordinates(self.frame)
        p, v = pv.getPosition(), pv.getVelocity()
        return StateEstimate(
            t=float(orbit.getDate().durationFrom(self.t0)),
            r=np.array([p.getX(), p.getY(), p.getZ()]),
            v=np.array([v.getX(), v.getY(), v.getZ()]),
            method=self.name,
            info={"obs_indices": self.idx},
        )

    def _solve(self, dates, los, obsp):
        raise NotImplementedError


class OrekitGooding(_OrekitAnglesOnlyIOD):
    """Gooding's method. Needs a guess of the range at the first and last observation."""

    name = "gooding"

    def __init__(self, sys, range_guess_km=(350_000.0, 350_000.0), **kw):
        super().__init__(sys, **kw)
        self.rho1, self.rho3 = (float(x) * 1e3 for x in range_guess_km)

    def _solve(self, dates, los, obsp):
        from org.orekit.estimation.iod import IodGooding

        return IodGooding(self.sys.gm_earth).estimate(
            self.frame, obsp[0], obsp[1], obsp[2],
            los[0], dates[0], los[1], dates[1], los[2], dates[2],
            self.rho1, self.rho3)


class OrekitGauss(_OrekitAnglesOnlyIOD):
    name = "gauss"

    def _solve(self, dates, los, obsp):
        from org.orekit.estimation.iod import IodGauss

        return IodGauss(self.sys.gm_earth).estimate(
            self.frame, obsp[0], dates[0], los[0], obsp[1], dates[1], los[1], obsp[2], dates[2], los[2])


class OrekitLaplace(_OrekitAnglesOnlyIOD):
    name = "laplace"

    def _solve(self, dates, los, obsp):
        from org.orekit.estimation.iod import IodLaplace
        from org.orekit.utils import PVCoordinates
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        # Laplace takes the observer's state at the middle observation.
        return IodLaplace(self.sys.gm_earth).estimate(
            self.frame, PVCoordinates(obsp[1], Vector3D.ZERO),
            dates[0], los[0], dates[1], los[1], dates[2], los[2])


class TruthPlusError:
    """TEST STAND-IN, not a real IOD method: returns the true state plus a known,
    random error. It reads the answer key, so never use it for real results.

    Purpose: Orekit's two-body IOD methods cannot fit a halo orbit, so this lets
    the propagation and results stages be built and checked now, and shows how a
    given IOD error grows over time. Replace it with a real method when one works.
    """

    name = "truth_plus_error"
    config_block = "truth_plus_error"

    def __init__(self, sys: EarthMoonSystem, truth_path: str | None = None, epoch_index: int = 2,
                 position_error_km: float = 1000.0, velocity_error_m_s: float = 10.0, seed: int = 0, **_):
        if not truth_path:
            raise ValueError("truth_plus_error needs data.truth in the config")
        from .sources import load_synthetic_truth

        self.sys = sys
        _, self.truth_rot, self.truth_tau = load_synthetic_truth(truth_path, sys)
        self.k = int(epoch_index)
        self.dr, self.dv = float(position_error_km) * 1e3, float(velocity_error_m_s)
        self.rng = np.random.default_rng(seed)

    def _random_dir(self):
        u = self.rng.normal(size=3)
        return u / np.linalg.norm(u)

    def estimate(self, obs: Observations) -> StateEstimate:
        from .sources import dense_truth

        k = _check_epoch(self.k, obs)
        tr = dense_truth(self.truth_rot, self.truth_tau, np.array([obs.t[k]]), self.sys)
        return StateEstimate(
            t=float(obs.t[k]),
            r=tr.r[0] + self.dr * self._random_dir(),
            v=tr.v[0] + self.dv * self._random_dir(),
            method=self.name,
            info={"obs_indices": [k], "test_stand_in": True},
        )


def _check_epoch(k: int, obs: Observations) -> int:
    if not 0 <= k < len(obs):
        raise IODFailure(f"iod.epoch_index = {k}, but there are only {len(obs)} observations (0-{len(obs) - 1})")
    return k


class RangeSearch:
    """CR3BP range search: angles give the direction, so try a grid of distances (range) and
    radial speeds (range-rate) at the epoch, propagate each candidate in the CR3BP, and score
    how well its predicted angles match ALL observations. The grid alone is coarse (~tens of
    thousands of km, mostly because the sideways velocity comes from a rough estimate of how
    fast the line of sight turns), so by default the best few distinct grid points are then
    polished: a least-squares fit of all six state components to the angles, in the same
    CR3BP, from each start. The polished candidate with the smallest angle misfit wins. Polishing
    continues past `candidates` starts (up to `max_candidates`) until one fits the angles at the
    residual-check level (RMS <= 3 sigma): on a DRO the right valley was sometimes only the 5th-best
    grid point on the coarse grid (benchmarks/collocation_iod/RESULTS.md)."""

    ACCEPT_RMS_SIGMA = 3.0
    DISTINCT_KM = 1000.0          # polished solutions farther apart than this are different orbits

    name = "range_search"
    config_block = "range_search"

    def __init__(self, sys: EarthMoonSystem, epoch_index: int = 2, range_km=(150000, 650000, 20000),
                 range_rate_m_s=(-800, 800, 100), refine: bool = True, candidates: int = 12,
                 max_candidates: int = 20, **_):
        self.sys, self.k = sys, int(epoch_index)
        self.refine, self.n_candidates = bool(refine), int(candidates)
        self.max_candidates = max(int(max_candidates), self.n_candidates)
        self.ranges = np.arange(range_km[0], range_km[1] + 0.5 * range_km[2], range_km[2]) * 1e3
        self.rates = np.arange(range_rate_m_s[0], range_rate_m_s[1] + 0.5 * range_rate_m_s[2], range_rate_m_s[2])

    def _direction_rate(self, obs: Observations, k: int) -> np.ndarray:
        """How fast the line of sight turns at the epoch: quadratic fit through the 3 nearest observations."""
        near = sorted(sorted(range(len(obs)), key=lambda i: abs(obs.t[i] - obs.t[k]))[:3])
        if len(near) < 2:
            raise IODFailure("range_search needs at least 2 observations")
        dt = obs.t[near] - obs.t[k]
        deg = min(2, len(near) - 1)
        rate = np.array([np.polyval(np.polyder(np.polyfit(dt, obs.los[near, c], deg)), 0.0) for c in range(3)])
        return rate - (rate @ obs.los[k]) * obs.los[k]

    def _cost(self, r, v, obs: Observations, k: int) -> float:
        from scipy.integrate import solve_ivp

        from .dynamics import cr3bp_eom
        from .frames import emi_to_rotating, rotating_to_emi

        taus = self.sys.tau(obs.t)
        x0 = emi_to_rotating(taus[k], r, v, self.sys)
        cost = 0.0
        for mask, t_end in ((taus > taus[k], taus.max()), (taus < taus[k], taus.min())):
            if not mask.any():
                continue
            sol = solve_ivp(cr3bp_eom, [taus[k], t_end], x0, args=(self.sys.mu,), dense_output=True,
                            method="DOP853", rtol=1e-9, atol=1e-11)
            for i in np.flatnonzero(mask):
                rr, _ = rotating_to_emi(taus[i], sol.sol(taus[i]), self.sys)
                d = rr - obs.observer[i]
                # true angle between predicted and measured directions (not just its sine, which
                # would score a direction pointing the exact opposite way as a perfect match)
                ang = np.arctan2(np.linalg.norm(np.cross(d, obs.los[i])), d @ obs.los[i])
                cost += float(ang / obs.sigma[i]) ** 2
        return cost

    def _starts(self, costs: np.ndarray) -> list[tuple[int, int]]:
        """Grid points to polish: the best, then the next best that are not next to an earlier pick
        (each one is a different valley of the cost, not the same valley twice)."""
        picks = []
        for flat in np.argsort(costs, axis=None):
            i, j = np.unravel_index(flat, costs.shape)
            if not np.isfinite(costs[i, j]) or len(picks) == self.max_candidates:
                break
            if all(abs(i - a) > 2 or abs(j - b) > 2 for a, b in picks):
                picks.append((int(i), int(j)))
        return picks

    def _polish(self, x_rot: np.ndarray, obs: Observations, k: int):
        """Least-squares fit of the 6 rotating-frame state components at the epoch to all the angles, with the
        exact Jacobian from the state transition matrix. Returns (state, cost) or None if it fails."""
        from scipy.optimize import least_squares

        from .dynamics import propagate_cr3bp_stm
        from .frames import rot_z

        taus = self.sys.tau(obs.t)
        R = np.array([rot_z(t) for t in taus]) * self.sys.du_m
        sigma = np.maximum(obs.sigma, 1e-12)
        # two directions perpendicular to each measured line of sight
        e1 = np.cross(obs.los, [0.0, 0.0, 1.0])
        e1[np.linalg.norm(e1, axis=1) < 1e-8] = [1.0, 0.0, 0.0]
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(obs.los, e1)
        E = np.stack([e1, e2], axis=1)                      # (N, 2, 3)
        cache = {}

        def model(x):
            key = x.tobytes()
            if key not in cache:
                states, stms = propagate_cr3bp_stm(x, taus[k], taus, self.sys.mu)
                d = np.einsum("nij,nj->ni", R, states[:, :3] - self.sys.earth_rot) - obs.observer
                n = np.linalg.norm(d, axis=1)
                u = d / n[:, None]
                res = (np.einsum("nij,nj->ni", E, u) / sigma[:, None]).ravel()
                P = (np.eye(3)[None] - np.einsum("ni,nj->nij", u, u)) / n[:, None, None]
                jac = np.einsum("nij,njk,nkl,nlm->nim", E, P, R, stms[:, :3, :]) / sigma[:, None, None]
                cache.clear()
                cache[key] = (res, jac.reshape(-1, 6), np.all(np.einsum("ni,ni->n", u, obs.los) > 0))
            return cache[key]

        try:
            with np.errstate(over="ignore", invalid="ignore"):    # wild starts may blow up; they are dropped
                sol = least_squares(lambda x: model(x)[0], x_rot, jac=lambda x: model(x)[1], method="lm",
                                    x_scale="jac", max_nfev=40)   # good starts need <15; wild ones crawl
                ok = model(sol.x)[2]                        # every prediction on the measured side of the sky
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        if not ok or not np.isfinite(sol.x).all():
            return None
        return sol.x, float(2 * sol.cost)

    def _alternatives(self, polished, r_best, tau, accept, n_obs) -> list[dict]:
        """Other, clearly different orbits that also fit the angles at the residual-check level. On 1-day arcs
        there were often two, about 11,000 km apart, and the true one was sometimes the second best: the data
        cannot tell them apart, so the caller should know."""
        from .frames import rotating_to_emi

        out, seen = [], [r_best]
        for x, c in sorted(polished, key=lambda p: p[1]):
            r, v = rotating_to_emi(tau, x, self.sys)
            rng = np.linalg.norm(r - self._observer)
            if (c > accept or not self.ranges[0] <= rng <= self.ranges[-1]          # outside the searched distances
                    or any(np.linalg.norm(r - q) <= self.DISTINCT_KM * 1e3 for q in seen)):
                continue
            seen.append(r)
            out.append({"position_m": r.tolist(), "velocity_m_s": v.tolist(),
                        "rms_sigma": float(np.sqrt(c / (2 * n_obs))),
                        "distance_from_chosen_km": float(np.linalg.norm(r - r_best) / 1e3)})
        return out

    def estimate(self, obs: Observations) -> StateEstimate:
        from .frames import emi_to_rotating, rotating_to_emi

        k = _check_epoch(self.k, obs)
        if len(obs) < 3:
            raise IODFailure("range_search needs at least 3 observations")
        self._observer = obs.observer[k]
        L, Ld = obs.los[k], self._direction_rate(obs, k)
        costs = np.full((len(self.ranges), len(self.rates)), np.inf)
        for i, rho in enumerate(self.ranges):
            for j, rhod in enumerate(self.rates):
                try:
                    costs[i, j] = self._cost(obs.observer[k] + rho * L, rhod * L + rho * Ld, obs, k)
                except (RuntimeError, ValueError, FloatingPointError):
                    pass
        starts = self._starts(costs)
        if not starts:
            raise IODFailure("range_search: no candidate could be evaluated")
        i, j = starts[0]
        r, v = obs.observer[k] + self.ranges[i] * L, self.rates[j] * L + self.ranges[i] * Ld
        info = {"obs_indices": [k], "grid_points": int(costs.size),
                "grid_best_range_km": float(self.ranges[i] / 1e3), "grid_best_cost": float(costs[i, j]),
                "polished": False}
        if self.refine:
            tau = float(self.sys.tau(obs.t[k]))
            polished, accept = [], self.ACCEPT_RMS_SIGMA ** 2 * 2 * len(obs)      # cost = sum of squares, 2 per obs
            tried = 0
            for i, j in starts:
                if tried >= self.n_candidates and any(c <= accept for _, c in polished):
                    break
                tried += 1
                start = emi_to_rotating(tau, obs.observer[k] + self.ranges[i] * L,
                                        self.rates[j] * L + self.ranges[i] * Ld, self.sys)
                p = self._polish(start, obs, k)
                if p is not None:
                    polished.append(p)
            info["polish_costs"] = sorted(c for _, c in polished)
            info["polished_starts"] = tried
            if polished:
                x_best, c_best = min(polished, key=lambda p: p[1])
                if c_best < info["grid_best_cost"]:
                    r, v = rotating_to_emi(tau, x_best, self.sys)
                    info.update(polished=True, polished_cost=c_best,
                                alternatives=self._alternatives(polished, r, tau, accept, len(obs)))
        info["best_range_km"] = float(np.linalg.norm(r - obs.observer[k]) / 1e3)
        return StateEstimate(t=float(obs.t[k]), r=r, v=v, method=self.name, info=info)


IOD_METHODS = {
    "range_search": RangeSearch,
    "truth_plus_error": TruthPlusError,
    "gooding": OrekitGooding,
    "gauss": OrekitGauss,
    "laplace": OrekitLaplace,
}

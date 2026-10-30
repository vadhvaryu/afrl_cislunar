"""
Grid-of-hypotheses IOD for angles-only cislunar tracking, refined by an estimator.

    range_grid     (RangeRateGrid)  (range, range-rate) hypotheses at the middle observation,
                                    scored cheaply in the CR3BP, best K refined by an estimator
    range_grid_fd  (RangeGridFD)    range-only hypotheses with a finite-difference velocity,
                                    refined by an estimator (the original brute-force baseline)

Idea credited, not implemented: AFRL's constrained admissible region multiple-hypothesis filter
(CAR-MHF) seeds a filter bank with a (range, range-rate) grid over the admissible region, spaced
7,500 km and 0.15 km/s in its cislunar work. Its code is not public for cislunar; this module only
borrows the grid-of-hypotheses idea and its spacing. There is no admissible-region constraint
(energy bound), no filter bank, and no hypothesis weighting here: each hypothesis is scored by its
CR3BP residual and the best few are refined by batch least squares (or multiple shooting).

Neither method reads the truth file. Both return the refined state at the middle observation
(EMI, SI), so the stack's own estimator re-converges from it in a few iterations.

Assumptions (both methods)
    * Observations are sorted in time; the epoch is observation j = len(obs) // 2.
    * At least 3 observations (the finite-difference velocity needs j - 1 and j + 1).
    * Range is measured from the observer; the default grid (200,000-500,000 km) assumes an object
      between roughly half and 1.3 lunar distances from Earth (true here: ~379,000 km at day 4).
    * The prefilter uses the CR3BP (same model as the truth generator) with a fixed-step RK4
      (default step 2e-3 TU, ~12.5 min); it only ranks hypotheses, so it need not be exact.
      Trajectories that pass through a primary score as infinity.
    * A refined fit counts when the refiner finishes and its post-fit RA/Dec residual RMS is at
      most gate_rms (default 3 sigma). The project's convergence study found good fits below
      1 sigma and bad local minima at hundreds to thousands of sigma, so 3 separates them.
      Exception: Orekit's AngularRaDec includes light-time, the synthetic angles do not
      (~0.2-0.8"), so on near noise-free data (sigma ~0.001") Orekit fits sit at hundreds of
      sigma; raise gate_rms there (e.g. 1,000 sigma = 1" at 0.001") or use multiple_shooting.
    * Fits closer than dedupe_km (100 km) are the same solution; the lowest RMS wins.
      On short arcs (hours to a day) range is weakly observable: several distinct fits pass the
      gate with statistically equal RMS and "lowest RMS" is then arbitrary (measured on 5 obs
      over 6 h at 10": fits 6,200 km and 12,200 km from truth at 0.58 and 0.51 sigma, and the
      wrong one wins). info["ambiguous"] / info["distinct_solutions"] expose this.
"""

from __future__ import annotations

import time

import numpy as np

from .contracts import Observations, StateEstimate
from .cr3bp_tools import los_derivatives
from .frames import EarthMoonSystem, emi_to_rotating, rot_z

KM = 1e3


def _fail(msg: str):
    """iod.IODFailure, imported here so iod.py can import this module at its bottom."""
    from .iod import IODFailure

    return IODFailure(msg)


# ----------------------------------------------------------------------------- vectorized CR3BP prefilter

def _accel(X: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion (dynamics.cr3bp_eom) for M states at once, X (M, 6)."""
    x, y, z, vx, vy, vz = X.T
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    a1, a2 = (1 - mu) / r1**3, mu / r2**3
    out = np.empty_like(X)
    out[:, 0], out[:, 1], out[:, 2] = vx, vy, vz
    out[:, 3] = 2 * vy + x - a1 * (x + mu) - a2 * (x - 1 + mu)
    out[:, 4] = -2 * vx + y - a1 * y - a2 * y
    out[:, 5] = -a1 * z - a2 * z
    return out


def propagate_batch(X0: np.ndarray, tau0: float, taus: np.ndarray, mu: float, h_max: float = 2e-3) -> np.ndarray:
    """Fixed-step RK4 for M rotating states X0 (M, 6) given at tau0, to every time in taus,
    forward and/or backward. Each interval between output times is cut into equal steps no
    longer than h_max. Returns (len(taus), M, 6); diverged states come back as nan/inf."""
    taus = np.asarray(taus, float)
    out = np.empty((len(taus), len(X0), 6))
    with np.errstate(all="ignore"):
        for idx in (np.flatnonzero(taus >= tau0), np.flatnonzero(taus < tau0)):
            idx = idx[np.argsort(np.abs(taus[idx] - tau0))]
            X, t = np.array(X0, float), tau0
            for i in idx:
                n = int(np.ceil(abs(taus[i] - t) / h_max))
                if n:
                    h = (taus[i] - t) / n
                    for _ in range(n):
                        k1 = _accel(X, mu)
                        k2 = _accel(X + 0.5 * h * k1, mu)
                        k3 = _accel(X + 0.5 * h * k2, mu)
                        k4 = _accel(X + h * k3, mu)
                        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    t = taus[i]
                out[i] = X
    return out


def residual_rms_batch(sys: EarthMoonSystem, obs: Observations, t_epoch: float, r: np.ndarray, v: np.ndarray,
                       h_max: float = 2e-3) -> np.ndarray:
    """RA/Dec residual RMS (sigma units, same definition as results.residuals) of M EMI states
    (r, v: (M, 3), at t_epoch), each propagated in the CR3BP to every observation time.
    Non-finite (e.g. impact) gives inf."""
    tau0 = float(sys.tau(t_epoch))
    X0 = np.array([emi_to_rotating(tau0, ri, vi, sys) for ri, vi in zip(r, v)])
    taus = sys.tau(obs.t)
    X = propagate_batch(X0, tau0, taus, sys.mu, h_max)                       # (N, M, 6)
    sq = np.zeros(len(X0))
    with np.errstate(all="ignore"):
        for i, tau in enumerate(taus):
            pos = (X[i, :, :3] - sys.earth_rot) @ rot_z(tau).T * sys.du_m      # EMI, m
            d = pos - obs.observer[i]
            ra_c = np.arctan2(d[:, 1], d[:, 0])
            dec_c = np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1))
            dra = (obs.ra[i] - ra_c + np.pi) % (2 * np.pi) - np.pi
            sq += (dra / obs.sigma[i]) ** 2 + ((obs.dec[i] - dec_c) / obs.sigma[i]) ** 2
        rms = np.sqrt(sq / (2 * len(obs)))
    rms[~np.isfinite(rms)] = np.inf
    return rms


# ----------------------------------------------------------------------------- geometry

def _observer_rate(obs: Observations, t_eval: float) -> np.ndarray:
    """Observer velocity at t_eval from a Lagrange polynomial fit through all observer positions
    (zero for a fixed observer such as the geocenter)."""
    if np.allclose(obs.observer, obs.observer[0]):
        return np.zeros(3)
    t = obs.t - t_eval
    c = np.linalg.solve(np.vander(t, len(t), increasing=True), obs.observer)
    return c[1]


def _fd_velocities(obs: Observations, j: int, rho_m: np.ndarray) -> np.ndarray:
    """Central-difference velocity at obs j for each range rho (same range at j-1 and j+1)."""
    p_minus = obs.observer[j - 1] + rho_m[:, None] * obs.los[j - 1]
    p_plus = obs.observer[j + 1] + rho_m[:, None] * obs.los[j + 1]
    return (p_plus - p_minus) / (obs.t[j + 1] - obs.t[j - 1])


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(np.floor((hi - lo) / step + 1e-9))
    return lo + step * np.arange(n + 1)


def _check(obs: Observations, name: str) -> int:
    if len(obs) < 3:
        raise _fail(f"{name}: needs at least 3 observations, got {len(obs)}")
    if np.any(np.diff(obs.t) <= 0):
        raise _fail(f"{name}: observations must be sorted by time")
    return len(obs) // 2


# ----------------------------------------------------------------------------- shared refinement

class _GridRefineBase:
    name = "grid-base"

    def __init__(self, sys: EarthMoonSystem, refine_with: str = "orekit_bls", refine_options: dict | None = None,
                 gate_rms: float = 3.0, dedupe_km: float = 100.0, prefilter_step_tau: float = 2e-3,
                 orekit_data: str | None = None, **_):
        from .estimation import ESTIMATORS

        if refine_with not in ESTIMATORS or refine_with == "none":
            raise ValueError(f"{self.name}: unknown refine_with {refine_with!r}")
        self.sys = sys
        self.refine_with = refine_with
        self.refiner = ESTIMATORS[refine_with](sys, orekit_data=orekit_data, **(refine_options or {}))
        self.gate = float(gate_rms)
        self.dedupe_m = float(dedupe_km) * KM
        self.h = float(prefilter_step_tau)

    def _refine_all(self, obs: Observations, t_epoch: float, starts: list[dict]):
        """Refine each start (dict with r, v and descriptive fields). Returns (records, passing),
        passing = [(rms, record index, fit)] for fits that finished and passed the gate."""
        from .estimation import EstimationFailure

        records, passing = [], []
        for k, s in enumerate(starts):
            x0 = StateEstimate(t=t_epoch, r=s["r"], v=s["v"], method=self.name, info={})
            rec = {key: val for key, val in s.items() if key not in ("r", "v")}
            rec.update(start_r_m=[float(a) for a in s["r"]], start_v_m_s=[float(a) for a in s["v"]])
            t0 = time.perf_counter()
            try:
                fit = self.refiner.refine(obs, x0)
            except (EstimationFailure, RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                rec.update(finished=False, passed_gate=False, error=str(e).splitlines()[0][:160])
                last = getattr(e, "last_state", None)
                if last is not None and "rms_normalized_residual" in last.info:
                    rec["refined_rms"] = float(last.info["rms_normalized_residual"])
            else:
                rms = float(fit.info.get("rms_normalized_residual", np.inf))
                ok = bool(np.isfinite(rms) and rms <= self.gate and np.all(np.isfinite(fit.r)) and np.all(np.isfinite(fit.v)))
                rec.update(finished=True, passed_gate=ok, refined_rms=rms,
                           iterations=int(fit.info.get("iterations", 0)),
                           refined_r_m=[float(a) for a in fit.r], refined_v_m_s=[float(a) for a in fit.v])
                if ok:
                    passing.append((rms, k, fit))
            rec["refine_runtime_s"] = round(time.perf_counter() - t0, 3)
            records.append(rec)
        return records, passing

    def _dedupe(self, passing):
        """Distinct solutions (more than dedupe_km apart), lowest RMS first."""
        kept = []
        for rms, k, fit in sorted(passing, key=lambda p: p[0]):
            if all(np.linalg.norm(fit.r - f.r) > self.dedupe_m for _, _, f in kept):
                kept.append((rms, k, fit))
        return kept

    def _result(self, obs, j, records, passing, info):
        distinct = self._dedupe(passing)
        info.update(refine_with=self.refine_with, gate_rms=self.gate, epoch_index=j, candidates=records,
                    n_finished=sum(r["finished"] for r in records), n_passed_gate=len(passing),
                    n_distinct_solutions=len(distinct), ambiguous=len(distinct) > 1,
                    distinct_solutions=[{"candidate": k, "rms": rms, "r_m": [float(a) for a in f.r]}
                                        for rms, k, f in distinct])
        if not distinct:
            best = min((r.get("refined_rms", np.inf) for r in records), default=np.inf)
            raise _fail(f"{self.name}: no refined hypothesis passed the {self.gate:g}-sigma residual gate "
                        f"({len(records)} refined, {info['n_finished']} finished, best RMS {best:.3g})")
        rms, k, fit = distinct[0]
        # Several starts usually reach the same solution; tie-breaking among them is numerical noise.
        reached = sorted(i for _, i, f in passing if np.linalg.norm(f.r - fit.r) <= self.dedupe_m)
        info.update(chosen_candidate=k, chosen_rms=rms, candidates_reaching_winner=reached,
                    winner_was_best_prefilter=bool(0 in reached))
        return StateEstimate(t=float(obs.t[j]), r=np.array(fit.r, float), v=np.array(fit.v, float), cov=fit.cov,
                             method=self.name, info=info)


# ----------------------------------------------------------------------------- method 1

class RangeRateGrid(_GridRefineBase):
    """(range, range-rate) hypotheses at the middle observation, CR3BP prefilter, refine the best K.

    At the middle observation j, the line of sight rho_hat and its rate rho_hat' come from
    cr3bp_tools.los_derivatives (Lagrange fit through all observations). Hypothesis (rho, rho_dot):
        r = R + rho rho_hat,    v = R' + rho_dot rho_hat + rho rho_hat'
    R = observer position at j, R' = rate of a Lagrange fit of the observer positions (zero for
    the geocenter). Grid (defaults): rho 200,000-500,000 km every 7,500 km (41 values) x rho_dot
    -1.5..+1.5 km/s every 0.15 km/s (21 values), the spacing of AFRL's CAR-MHF cislunar grid. Plus,
    per rho, the earlier baseline's velocity (central difference of the same rho along the lines
    of sight at j-1 and j+1), so 41 x 22 = 902 hypotheses.

    Spacing: 0.15 km/s leaves at most 75 m/s of range-rate error from the grid, inside the
    150 m/s basin measured for Orekit BLS on this halo (and 3,750 km of range error, inside the
    10,000 km basin). But the grid only searches velocity along the line of sight: the transverse
    part, rho rho_hat', is only as good as the Lagrange fit. On the 8-day, 2-day-spaced arc that
    fit is poor (measured: ~220 m/s transverse error at the true range), so the cell nearest the
    truth is not guaranteed to be inside the basin; refining several distinct hypotheses is what
    makes the method work, not the grid cell alone.

    Prefilter: all hypotheses are propagated at once (vectorized fixed-step RK4 in the CR3BP) to
    every observation time and scored by RA/Dec residual RMS. The best top_k (default 5) that are
    more than min_separation_km (20,000 km) apart in position are refined by `refine_with`
    ("orekit_bls" default, "orekit_bls_robust", "multiple_shooting"); fits that finish and pass
    the residual gate are de-duplicated within 100 km and the lowest RMS wins.

    info: prefilter runtime, the top prefilter rows, every refined candidate (prefilter score,
    refined RMS and state, runtime), the distinct passing solutions, the candidates whose fits
    reached the winning solution, and winner_was_best_prefilter (the best prefilter hypothesis
    is among them; several starts usually reach the same fit, so "which one won" is a numerical
    tie). Raises iod.IODFailure if no refined fit passes.
    """

    name = "range_grid"

    def __init__(self, sys: EarthMoonSystem, rho_min_km: float = 200_000.0, rho_max_km: float = 500_000.0,
                 rho_step_km: float = 7_500.0, rho_dot_min_km_s: float = -1.5, rho_dot_max_km_s: float = 1.5,
                 rho_dot_step_km_s: float = 0.15, include_fd: bool = True, top_k: int = 5,
                 min_separation_km: float = 20_000.0, info_top_rows: int = 20, **kw):
        super().__init__(sys, **kw)
        self.rho_m = _grid(rho_min_km, rho_max_km, rho_step_km) * KM
        self.rho_dot_m_s = _grid(rho_dot_min_km_s, rho_dot_max_km_s, rho_dot_step_km_s) * KM
        self.include_fd = bool(include_fd)
        self.top_k = int(top_k)
        self.sep_m = float(min_separation_km) * KM
        self.info_rows = int(info_top_rows)

    def hypotheses(self, obs: Observations):
        """All hypotheses at the middle observation: (r (M, 3), v (M, 3), rho_m (M,), rho_dot_m_s (M,)
        with nan for the finite-difference ones, kind (M,) 'grid' or 'fd'), plus j."""
        j = _check(obs, self.name)
        u, du, _ = los_derivatives(obs, range(len(obs)), float(obs.t[j]))
        R, Rd = obs.observer[j], _observer_rate(obs, float(obs.t[j]))
        rho, rdot = np.meshgrid(self.rho_m, self.rho_dot_m_s, indexing="ij")
        rho, rdot = rho.ravel(), rdot.ravel()
        r = R + rho[:, None] * u
        v = Rd + rdot[:, None] * u + rho[:, None] * du
        kind = np.array(["grid"] * len(rho), dtype=object)
        if self.include_fd:
            r = np.vstack([r, R + self.rho_m[:, None] * obs.los[j]])
            v = np.vstack([v, _fd_velocities(obs, j, self.rho_m)])
            rho = np.concatenate([rho, self.rho_m])
            rdot = np.concatenate([rdot, np.full(len(self.rho_m), np.nan)])
            kind = np.concatenate([kind, np.array(["fd"] * len(self.rho_m), dtype=object)])
        return r, v, rho, rdot, kind, j

    def prefilter(self, obs: Observations):
        """Score every hypothesis. Returns dict with arrays r, v, rho_m, rho_dot_m_s, kind, score,
        order (best first), the epoch index j and the runtime."""
        t0 = time.perf_counter()
        r, v, rho, rdot, kind, j = self.hypotheses(obs)
        score = residual_rms_batch(self.sys, obs, float(obs.t[j]), r, v, self.h)
        order = np.argsort(score, kind="stable")
        return dict(r=r, v=v, rho_m=rho, rho_dot_m_s=rdot, kind=kind, score=score, order=order, j=j,
                    runtime_s=time.perf_counter() - t0)

    def select(self, pf: dict) -> list[int]:
        """Best top_k hypotheses, each more than min_separation_km from those already chosen."""
        chosen = []
        for i in pf["order"]:
            if not np.isfinite(pf["score"][i]):
                break
            if all(np.linalg.norm(pf["r"][i] - pf["r"][c]) > self.sep_m for c in chosen):
                chosen.append(int(i))
                if len(chosen) == self.top_k:
                    break
        return chosen

    @staticmethod
    def _row(pf, i):
        rd = pf["rho_dot_m_s"][i]
        return {"kind": str(pf["kind"][i]), "rho_km": float(pf["rho_m"][i] / KM),
                "rho_dot_km_s": None if np.isnan(rd) else float(rd / KM), "prefilter_rms": float(pf["score"][i])}

    def estimate(self, obs: Observations) -> StateEstimate:
        t_start = time.perf_counter()
        pf = self.prefilter(obs)
        chosen = self.select(pf)
        if not chosen:
            raise _fail(f"{self.name}: every hypothesis diverged in the prefilter")
        starts = [{**self._row(pf, i), "prefilter_rank": int(np.flatnonzero(pf["order"] == i)[0]),
                   "r": pf["r"][i], "v": pf["v"][i]} for i in chosen]
        records, passing = self._refine_all(obs, float(obs.t[pf["j"]]), starts)
        info = {"n_hypotheses": int(len(pf["score"])), "n_rho": int(len(self.rho_m)),
                "n_rho_dot": int(len(self.rho_dot_m_s)), "include_fd": self.include_fd,
                "prefilter_runtime_s": round(pf["runtime_s"], 3),
                "prefilter_top": [self._row(pf, i) for i in pf["order"][:self.info_rows]]}
        x = self._result(obs, pf["j"], records, passing, info)
        x.info["runtime_s"] = round(time.perf_counter() - t_start, 3)
        return x


# ----------------------------------------------------------------------------- method 2

class RangeGridFD(_GridRefineBase):
    """The original range-only baseline: range hypotheses with a finite-difference velocity.

    At the middle observation j, for each alpha0 in 200,000-500,000 km every 10,000 km (31 values):
        r = R_j + alpha0 rho_hat_j
        v = (R_{j+1} + alpha0 rho_hat_{j+1} - R_{j-1} - alpha0 rho_hat_{j-1}) / (t_{j+1} - t_{j-1})
    i.e. the same range along the neighbouring lines of sight, central difference. Assumes the
    range changes little between j-1 and j+1 (the range-rate is not searched) and that the chord
    velocity approximates the instantaneous one (poor on a 4-day chord, fine on hours).

    prefilter_top (default 8): brute force refines all 31 hypotheses, which with Orekit BLS here
    takes several minutes per call (measured ~10-20 s per BLS fit on a shared machine), so by
    default only the best 8 by CR3BP residual (same vectorized prefilter as range_grid) are
    refined. prefilter_top=None restores the full brute force. Refined fits that finish and pass
    the gate are de-duplicated within 100 km; the lowest RMS wins. Raises iod.IODFailure if none.
    """

    name = "range_grid_fd"

    def __init__(self, sys: EarthMoonSystem, alpha_min_km: float = 200_000.0, alpha_max_km: float = 500_000.0,
                 alpha_step_km: float = 10_000.0, prefilter_top: int | None = 8, **kw):
        super().__init__(sys, **kw)
        self.alpha_m = _grid(alpha_min_km, alpha_max_km, alpha_step_km) * KM
        self.prefilter_top = None if prefilter_top is None else int(prefilter_top)

    def hypotheses(self, obs: Observations):
        j = _check(obs, self.name)
        r = obs.observer[j] + self.alpha_m[:, None] * obs.los[j]
        return r, _fd_velocities(obs, j, self.alpha_m), j

    def estimate(self, obs: Observations) -> StateEstimate:
        t_start = time.perf_counter()
        r, v, j = self.hypotheses(obs)
        t0 = time.perf_counter()
        score = residual_rms_batch(self.sys, obs, float(obs.t[j]), r, v, self.h)
        pf_time = time.perf_counter() - t0
        order = np.argsort(score, kind="stable")
        idx = order if self.prefilter_top is None else order[:self.prefilter_top]
        starts = [{"kind": "fd", "alpha_km": float(self.alpha_m[i] / KM), "prefilter_rms": float(score[i]),
                   "prefilter_rank": int(n), "r": r[i], "v": v[i]} for n, i in enumerate(idx)]
        records, passing = self._refine_all(obs, float(obs.t[j]), starts)
        info = {"n_hypotheses": int(len(r)), "prefilter_top": self.prefilter_top,
                "prefilter_runtime_s": round(pf_time, 3)}
        x = self._result(obs, j, records, passing, info)
        x.info["runtime_s"] = round(time.perf_counter() - t_start, 3)
        return x

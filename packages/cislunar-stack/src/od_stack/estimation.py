"""
Estimation stage: refine the IOD state using ALL observations.
Each estimator takes (Observations, StateEstimate) and returns a better StateEstimate.

    IOD (3 obs, rough) --StateEstimate--> estimator (all obs) --StateEstimate + covariance--> propagator

Add a method (e.g. a Kalman filter): write a class with `name` and
`refine(obs, x0) -> StateEstimate`, add it to ESTIMATORS, select it in config.yaml.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import Observations, StateEstimate
from .frames import EarthMoonSystem
from .orekit_setup import start_orekit


class EstimationFailure(RuntimeError):
    """The estimator found no acceptable fit. ``state`` holds the rejected fit, if there was one."""

    def __init__(self, message: str, state: StateEstimate | None = None):
        super().__init__(message)
        self.state = state


class NoEstimation:
    """Pass the IOD state straight through. If the IOD estimated impulsive burns (collocation in impulsive
    mode), keep them as the state's maneuvers, so the prediction follows the IOD's own trajectory."""

    name = "none"

    def __init__(self, sys: EarthMoonSystem, **_):
        pass

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        impulses = (x0.info or {}).get("impulses")
        if not impulses:
            return x0
        return StateEstimate(t=x0.t, r=x0.r, v=x0.v, cov=x0.cov, method=x0.method,
                             info={**x0.info, "maneuvers": [{"t_s": i["t_s"], "delta_v_m_s": i["delta_v_m_s"]}
                                                            for i in impulses]})


class OrekitBatchLS:
    """Orekit BatchLSEstimator (Levenberg-Marquardt) on RA/Dec angles, with the
    CR3BP-equivalent force model from orekit_models.py. Estimates the 6 Cartesian
    state components at the IOD epoch and returns their covariance."""

    name = "orekit_bls"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, max_iterations: int = 50,
                 max_evaluations: int = 100, convergence_threshold: float = 1e-3,
                 optimizer: str = "levenberg_marquardt", covariance: bool = True,
                 builder: dict | None = None, orbit_type: str = "cartesian", station_offset_m: float = 1.0,
                 residual_gate_sigma: float | None = 3.0, **_):
        self.sys = sys
        self.max_iter, self.max_eval, self.threshold = int(max_iterations), int(max_evaluations), float(convergence_threshold)
        self.optimizer, self.want_cov = optimizer, bool(covariance)
        self.builder_settings, self.orbit_type, self.station_offset_m = builder or {}, orbit_type, float(station_offset_m)
        # A converged fit whose residual RMS (in measurement sigmas) exceeds this is a wrong local solution, not
        # a noisy right one: correct fits stayed at or below 1.6 sigma at 1-100 arcsec noise, wrong ones were at
        # hundreds (benchmarks/collocation_iod). None turns the check off.
        self.gate = None if residual_gate_sigma is None else float(residual_gate_sigma)
        start_orekit(orekit_data)

    def _optimizer(self):
        if self.optimizer == "gauss_newton":
            from org.hipparchus.linear import QRDecomposer
            from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

            return GaussNewtonOptimizer(QRDecomposer(1e-11), False)
        from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer

        return LevenbergMarquardtOptimizer()

    def _cartesian_covariance(self, estimator, fitted_orbit):
        """6x6 covariance in x, y, z, vx, vy, vz (SI), whatever orbit type was estimated.
        The singularity threshold is 1e-16, not Orekit's usual 1e-12: with 1 m position scaling, short arcs
        trip 1e-12 although their normal matrix is well conditioned (condition number ~300-20,000 on 1-day
        and 6-hour arcs), and the covariance then computed matched the actual errors (benchmarks/collocation_iod)."""
        C = estimator.getPhysicalCovariances(1e-16)
        if self.orbit_type == "cartesian":
            return np.array([[C.getEntry(i, j) for j in range(6)] for i in range(6)])
        from org.orekit.orbits import OrbitType, PositionAngleType
        from org.orekit.propagation import StateCovariance

        sub = C.getSubMatrix(0, 5, 0, 5)
        sc = StateCovariance(sub, fitted_orbit.getDate(), fitted_orbit.getFrame(),
                             OrbitType.valueOf(self.orbit_type.upper()), PositionAngleType.TRUE)
        M = sc.changeCovarianceType(fitted_orbit, OrbitType.CARTESIAN, PositionAngleType.TRUE).getMatrix()
        return np.array([[M.getEntry(i, j) for j in range(6)] for i in range(6)])

    def _measurements(self, obs: Observations):
        from jpype import JArray, JDouble
        from org.orekit.estimation.measurements import AngularRaDec, ObservableSatellite

        from .orekit_models import frame_and_epoch, geocenter_station

        if np.any(np.linalg.norm(obs.observer, axis=1) > 1.0):
            raise NotImplementedError("Only an observer at Earth's center is supported so far; "
                                      "ground stations need a GroundStation per observer position.")
        frame, t0 = frame_and_epoch()
        station, sat = geocenter_station(self.station_offset_m), ObservableSatellite(0)
        out = []
        for t, ra, dec, s in zip(obs.t, obs.ra, obs.dec, obs.sigma):
            ra = math.atan2(math.sin(ra), math.cos(ra))     # same branch Orekit uses
            out.append(AngularRaDec(station, frame, t0.shiftedBy(float(t)), JArray(JDouble)([ra, float(dec)]),
                                    JArray(JDouble)([float(s), float(s)]), JArray(JDouble)([1.0, 1.0]), sat))
        return out

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        from org.orekit.estimation.leastsquares import BatchLSEstimator

        from .orekit_models import frame_and_epoch, propagator_builder

        frame, _ = frame_and_epoch()
        try:        # e.g. a hyperbolic starting state cannot be written as equinoctial elements
            builder = propagator_builder(self.sys, x0.t, x0.r, x0.v, self.builder_settings, self.orbit_type)
        except Exception as e:
            raise EstimationFailure(f"{self.name}: cannot start from this state: {str(e).splitlines()[0]}") from e
        estimator = BatchLSEstimator(self._optimizer(), builder)
        estimator.setParametersConvergenceThreshold(self.threshold)
        estimator.setMaxIterations(self.max_iter)
        estimator.setMaxEvaluations(self.max_eval)
        try:
            measurements = self._measurements(obs)
        except RuntimeError as e:          # e.g. orekit-data not set
            raise EstimationFailure(f"{self.name}: {e}") from None
        for m in measurements:
            estimator.addMeasurement(m)

        try:
            fitted = estimator.estimate()[0]
        except Exception as e:
            raise EstimationFailure(f"{self.name}: {str(e).splitlines()[0]}") from e

        pv = fitted.getInitialState().getPVCoordinates(frame)
        p, v = pv.getPosition(), pv.getVelocity()
        cov = None
        if self.want_cov:
            try:
                cov = self._cartesian_covariance(estimator, fitted.getInitialState().getOrbit())
            except Exception:
                cov = None    # the observations leave some direction of the state undetermined
        rms = float(estimator.getOptimum().getRMS())
        fit = StateEstimate(
            t=x0.t,
            r=np.array([p.getX(), p.getY(), p.getZ()]),
            v=np.array([v.getX(), v.getY(), v.getZ()]),
            cov=cov,
            method=f"{x0.method} + {self.name}",
            info={**x0.info, "optimizer": self.optimizer, "orbit_type": self.orbit_type,
                  "iterations": int(estimator.getIterationsCount()),
                  "evaluations": int(estimator.getEvaluationsCount()),
                  "rms_normalized_residual": rms, "residual_gate_sigma": self.gate},
        )
        if self.gate is not None and not rms <= self.gate:
            raise EstimationFailure(f"{self.name}: residual check failed: RMS {rms:.3g} sigma > {self.gate:g} sigma "
                                    "(the fit converged to an orbit that does not match the observations)", fit)
        return fit


SPEED_OF_LIGHT_M_S = 299_792_458.0


def cluster_impulses(impulses, min_burn_m_s: float, gap_s: float = 0.5 * 86400, keep_fraction: float = 0.3):
    """Group an IOD's impulses (collocation spreads one burn over neighboring mesh points) into burns:
    impulses less than gap_s apart join one burn, at their Δv-weighted mean time with their summed Δv.
    Keeps burns of at least min_burn_m_s and at least keep_fraction of the largest."""
    items = sorted(impulses, key=lambda i: i["t_s"])
    groups = []
    for i in items:
        if groups and i["t_s"] - groups[-1][-1]["t_s"] <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    burns = []
    for g in groups:
        dv = np.sum([i["delta_v_m_s"] for i in g], axis=0)
        w = np.array([np.linalg.norm(i["delta_v_m_s"]) for i in g])
        burns.append((float(np.sum(w * [i["t_s"] for i in g]) / w.sum()), dv))
    if not burns:
        return []
    biggest = max(np.linalg.norm(dv) for _, dv in burns)
    return [(t, dv) for t, dv in burns if np.linalg.norm(dv) >= max(min_burn_m_s, keep_fraction * biggest)]


class CR3BPBatchLS:
    """Batch least squares in the CR3BP itself (no Java), on RA/Dec angles, optionally with impulsive burns.

    Fits the state at the IOD epoch and, when `maneuvers` is "from_iod", the velocity change (and, with
    estimate_burn_times, the time) of each burn the IOD reports: collocation in impulsive mode lists its
    impulses, which are grouped into burns as the starting guess. Exact Jacobian from the state transition
    matrix (dynamics.propagate_with_burns). Angles include light time, as in Orekit's model. Any observer
    position works (not only Earth's center). Covariance of the state and of each burn comes from the
    Jacobian at the solution."""

    name = "cr3bp_bls"

    def __init__(self, sys: EarthMoonSystem, max_iterations: int = 50, covariance: bool = True,
                 residual_gate_sigma: float | None = 3.0, maneuvers: str = "from_iod",
                 estimate_burn_times: bool = True, min_burn_m_s: float = 0.5, **_):
        self.sys, self.max_iter, self.want_cov = sys, int(max_iterations), bool(covariance)
        self.gate = None if residual_gate_sigma is None else float(residual_gate_sigma)
        self.use_burns, self.fit_times, self.min_burn = maneuvers == "from_iod", bool(estimate_burn_times), float(min_burn_m_s)

    def _burn_guess(self, x0: StateEstimate):
        from .frames import rot_z
        if not self.use_burns:
            return []
        out = []
        for t_s, dv in cluster_impulses((x0.info or {}).get("impulses") or [], self.min_burn):
            tau = float(self.sys.tau(t_s))
            out.append((tau, rot_z(tau).T @ dv / self.sys.vu_ms))
        return out

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        from scipy.optimize import least_squares

        from .dynamics import propagate_with_burns
        from .frames import emi_to_rotating, rot_z, rotating_to_emi

        taus = self.sys.tau(obs.t)
        tau0 = float(self.sys.tau(x0.t))
        burns = self._burn_guess(x0)
        k = len(burns)
        R = np.array([rot_z(t) for t in taus])
        Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])    # omega x (.) for omega = z
        sigma = np.maximum(obs.sigma, 1e-12)
        e1 = np.cross(obs.los, [0.0, 0.0, 1.0])
        e1[np.linalg.norm(e1, axis=1) < 1e-8] = [1.0, 0.0, 0.0]
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        E = np.stack([e1, np.cross(obs.los, e1)], axis=1)
        nt = k if self.fit_times else 0

        def unpack(p):
            times = p[6 + 3 * k:] if self.fit_times else [b[0] for b in burns]
            return p[:6], [(times[j], p[6 + 3 * j:9 + 3 * j]) for j in range(k)]

        def model(p):
            x, bl = unpack(p)
            states, sens = propagate_with_burns(x, tau0, taus, self.sys.mu, bl, partials=True)
            rho = states[:, :3] - self.sys.earth_rot
            r = np.einsum("nij,nj->ni", R, rho) * self.sys.du_m
            v = np.einsum("nij,nj->ni", R, states[:, 3:] + rho @ Z.T) * self.sys.vu_ms
            light = np.linalg.norm(r - obs.observer, axis=1) / SPEED_OF_LIGHT_M_S
            d = r - v * light[:, None] - obs.observer                    # where the object was when the light left
            n = np.linalg.norm(d, axis=1)
            u = d / n[:, None]
            res = (np.einsum("nij,nj->ni", E, u) / sigma[:, None]).ravel()
            Pu = (np.eye(3)[None] - np.einsum("ni,nj->nij", u, u)) / n[:, None, None]
            cols = sens if self.fit_times else sens[:, :, :6 + 3 * k]
            jac = np.einsum("nij,njk,nkl,nlm->nim", E, Pu, R * self.sys.du_m, cols[:, :3, :]) / sigma[:, None, None]
            return res, jac.reshape(-1, jac.shape[-1])

        p0 = np.concatenate([emi_to_rotating(tau0, x0.r, x0.v, self.sys), *[dv for _, dv in burns],
                             [t for t, _ in burns] if self.fit_times else []])
        lo = np.full(p0.size, -np.inf)
        hi = np.full(p0.size, np.inf)
        if nt:
            lo[-nt:], hi[-nt:] = taus.min(), taus.max()
            p0[-nt:] = np.clip(p0[-nt:], taus.min() + 1e-9, taus.max() - 1e-9)
        cache = {}

        def cached(p):
            key = p.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = model(p)
            return cache[key]

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                sol = least_squares(lambda p: cached(p)[0], p0, jac=lambda p: cached(p)[1], method="trf",
                                    bounds=(lo, hi), x_scale="jac", max_nfev=self.max_iter * 4, xtol=1e-12, ftol=1e-12)
        except (RuntimeError, ValueError, FloatingPointError) as e:
            raise EstimationFailure(f"{self.name}: {e}") from e
        if not np.isfinite(sol.x).all():
            raise EstimationFailure(f"{self.name}: the fit diverged")
        res, jac = cached(sol.x)
        rms = float(np.sqrt(np.mean(res ** 2)))
        x, bl = unpack(sol.x)
        r, v = rotating_to_emi(tau0, x, self.sys)

        cov = cov_par = None
        if self.want_cov:
            try:
                cov_par = np.linalg.inv(jac.T @ jac)
                R0 = rot_z(tau0)
                T = np.zeros((6, 6))
                T[:3, :3] = R0 * self.sys.du_m
                T[3:, :3] = R0 @ Z * self.sys.vu_ms
                T[3:, 3:] = R0 * self.sys.vu_ms
                cov = T @ cov_par[:6, :6] @ T.T
            except np.linalg.LinAlgError:
                cov = cov_par = None
        maneuvers = []
        for j, (tb, dv) in enumerate(bl):
            Rb = rot_z(tb)
            m = {"t_s": float(tb * self.sys.tu_s), "delta_v_m_s": (Rb @ dv * self.sys.vu_ms).tolist()}
            if cov_par is not None:
                C = Rb @ cov_par[6 + 3 * j:9 + 3 * j, 6 + 3 * j:9 + 3 * j] @ Rb.T * self.sys.vu_ms ** 2
                m["sigma_delta_v_m_s"] = float(np.sqrt(np.trace(C)))
                if self.fit_times:
                    m["sigma_t_s"] = float(np.sqrt(cov_par[6 + 3 * k + j, 6 + 3 * k + j]) * self.sys.tu_s)
            maneuvers.append(m)
        fit = StateEstimate(t=x0.t, r=r, v=v, cov=cov, method=f"{x0.method} + {self.name}",
                            info={**{k_: v_ for k_, v_ in (x0.info or {}).items() if k_ != "impulses"},
                                  "iterations": int(sol.nfev), "rms_normalized_residual": rms,
                                  "residual_gate_sigma": self.gate, "maneuvers": maneuvers,
                                  "burns_from_iod": k})
        if self.gate is not None and not rms <= self.gate:
            raise EstimationFailure(f"{self.name}: residual check failed: RMS {rms:.3g} sigma > {self.gate:g} sigma "
                                    "(the fit converged to an orbit that does not match the observations)", fit)
        return fit


ESTIMATORS = {
    "none": NoEstimation,
    "orekit_bls": OrekitBatchLS,
    "cr3bp_bls": CR3BPBatchLS,
}

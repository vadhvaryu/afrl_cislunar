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
    """Pass the IOD state straight through."""

    name = "none"

    def __init__(self, sys: EarthMoonSystem, **_):
        pass

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        return x0


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


ESTIMATORS = {
    "none": NoEstimation,
    "orekit_bls": OrekitBatchLS,
}

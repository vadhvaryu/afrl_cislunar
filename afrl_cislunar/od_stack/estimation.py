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
    """last_state: where the estimator was when it gave up (or None), for diagnostics only."""

    def __init__(self, msg: str, last_state: StateEstimate | None = None):
        super().__init__(msg)
        self.last_state = last_state


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
                 max_evaluations: int = 100, convergence_threshold: float = 1e-3, **_):
        self.sys = sys
        self.max_iter, self.max_eval, self.threshold = int(max_iterations), int(max_evaluations), float(convergence_threshold)
        start_orekit(orekit_data)

    def _measurements(self, obs: Observations):
        from jpype import JArray, JDouble
        from org.orekit.estimation.measurements import AngularRaDec, ObservableSatellite

        from .orekit_models import frame_and_epoch, geocenter_station

        if np.any(np.linalg.norm(obs.observer, axis=1) > 1.0):
            raise NotImplementedError("Only an observer at Earth's center is supported so far; "
                                      "ground stations need a GroundStation per observer position.")
        frame, t0 = frame_and_epoch()
        station, sat = geocenter_station(), ObservableSatellite(0)
        out = []
        for t, ra, dec, s in zip(obs.t, obs.ra, obs.dec, obs.sigma):
            ra = math.atan2(math.sin(ra), math.cos(ra))     # same branch Orekit uses
            out.append(AngularRaDec(station, frame, t0.shiftedBy(float(t)), JArray(JDouble)([ra, float(dec)]),
                                    JArray(JDouble)([float(s), float(s)]), JArray(JDouble)([1.0, 1.0]), sat))
        return out

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
        from org.orekit.estimation.leastsquares import BatchLSEstimator

        from .orekit_models import frame_and_epoch, propagator_builder

        frame, _ = frame_and_epoch()
        estimator = BatchLSEstimator(LevenbergMarquardtOptimizer(), propagator_builder(self.sys, x0.t, x0.r, x0.v))
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
            raise EstimationFailure(f"{self.name}: {str(e).splitlines()[0]}",
                                    self._last_iterate(estimator, x0)) from e

        pv = fitted.getInitialState().getPVCoordinates(frame)
        p, v = pv.getPosition(), pv.getVelocity()
        try:
            C = estimator.getPhysicalCovariances(1e-12)
            cov = np.array([[C.getEntry(i, j) for j in range(6)] for i in range(6)])
        except Exception:
            cov = None        # happens at very high noise; worth investigating
        return StateEstimate(
            t=x0.t,
            r=np.array([p.getX(), p.getY(), p.getZ()]),
            v=np.array([v.getX(), v.getY(), v.getZ()]),
            cov=cov,
            method=f"{x0.method} + {self.name}",
            info={**x0.info, "iterations": int(estimator.getIterationsCount()),
                  "evaluations": int(estimator.getEvaluationsCount()),
                  "rms_normalized_residual": float(estimator.getOptimum().getRMS())},
        )


    @staticmethod
    def _last_iterate(estimator, x0: StateEstimate) -> StateEstimate | None:
        """The state Orekit last evaluated (its parameter drivers keep it after a failure).
        It may be a rejected Levenberg-Marquardt trial point; use it for diagnostics only."""
        try:
            p = [d.getValue() for d in estimator.getOrbitalParametersDrivers(True).getDrivers()]
            return StateEstimate(t=x0.t, r=np.array(p[:3]), v=np.array(p[3:6]), method=f"{x0.method} + last iterate",
                                 info={"state_kind": "last_iterate",
                                       "iterations": int(estimator.getIterationsCount()),
                                       "evaluations": int(estimator.getEvaluationsCount())})
        except Exception:
            return None


ESTIMATORS = {
    "none": NoEstimation,
    "orekit_bls": OrekitBatchLS,
}

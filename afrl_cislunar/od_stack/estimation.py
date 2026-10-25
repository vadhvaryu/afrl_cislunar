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
    state components at the IOD epoch and returns their covariance.

    gate_rms: if set, a fit whose post-fit residual RMS (in sigma) is above it raises
    EstimationFailure instead of being returned. Orekit stops when the parameter step is
    tiny, which also happens at a bad local minimum (the convergence study found such fits
    at ~1,600 sigma while good fits sit below 1 sigma)."""

    name = "orekit_bls"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, max_iterations: int = 50,
                 max_evaluations: int = 100, convergence_threshold: float = 1e-3, position_scale_m: float = 1.0,
                 gate_rms: float | None = None, **_):
        self.sys = sys
        self.max_iter, self.max_eval, self.threshold = int(max_iterations), int(max_evaluations), float(convergence_threshold)
        self.position_scale = float(position_scale_m)
        self.gate_rms = None if gate_rms is None else float(gate_rms)
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
        estimator = BatchLSEstimator(LevenbergMarquardtOptimizer(),
                                     propagator_builder(self.sys, x0.t, x0.r, x0.v, position_scale_m=self.position_scale))
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
        fit = StateEstimate(
            t=x0.t,
            r=np.array([p.getX(), p.getY(), p.getZ()]),
            v=np.array([v.getX(), v.getY(), v.getZ()]),
            cov=cov,
            method=f"{x0.method} + {self.name}",
            info={**x0.info, "iterations": int(estimator.getIterationsCount()),
                  "evaluations": int(estimator.getEvaluationsCount()),
                  "rms_normalized_residual": float(estimator.getOptimum().getRMS())},
        )
        rms = fit.info["rms_normalized_residual"]
        if self.gate_rms is not None and not rms <= self.gate_rms:
            # Converged to a point that does not fit the data (a bad local minimum): report a failure.
            fit.info["state_kind"] = "rejected_fit"
            raise EstimationFailure(f"{self.name}: rejected by residual gate (RMS {rms:.3g} > {self.gate_rms:g} sigma)", fit)
        return fit


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


def subset(obs: Observations, idx) -> Observations:
    idx = list(idx)
    return Observations(t=obs.t[idx], los=obs.los[idx], observer=obs.observer[idx], sigma=obs.sigma[idx],
                        ra=obs.ra[idx], dec=obs.dec[idx])


class OrekitBatchLSRobust:
    """Orekit BLS made harder to fool, for poor starting guesses. Same fit, same covariance
    as orekit_bls when it works; it only changes how the fit is started.

    Steps, in order; the first that gives a fit passing the residual gate wins:
      1. continuation  fit the short middle arc (short_arc, default obs 1-3, days 2-6) from
                       the IOD state, then all observations from that solution. A shorter arc
                       has a smoother cost landscape. NOTE: 3 observations = 6 angles = 6
                       unknowns, so the short-arc fit can always reach ~zero residual and is
                       never gated; only the full fit is.
      2. plain         the plain full fit from the IOD state (so this estimator never does
                       worse than orekit_bls on a case orekit_bls gets right).
      3. restarts      move the starting position along the line of sight at the epoch by each
                       of restart_offsets_km and repeat step 1 from there. All restarts are run;
                       the passing fit with the lowest RMS is kept.
    gate_rms: a full fit counts only if its residual RMS (sigma) is at most this.
    info["robust_step"] says which step won; info["robust_attempts"] lists every attempt.
    Raises EstimationFailure (last_state = lowest-RMS fit or last iterate) if none passes.
    """

    name = "orekit_bls_robust"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, short_arc=(1, 2, 3),
                 restart_offsets_km=(10_000, -10_000, 20_000, -20_000), robust_gate_rms: float = 3.0, **kw):
        kw.pop("gate_rms", None)
        self.inner = OrekitBatchLS(sys, orekit_data=orekit_data, **kw)
        self.short_arc = list(short_arc) if short_arc else []
        self.offsets_m = [float(d) * 1e3 for d in (restart_offsets_km or [])]
        self.gate = float(robust_gate_rms)

    def _full(self, obs, x, step, attempts):
        """Full fit from x; returns the fit if it passes the gate, else None."""
        try:
            fit = self.inner.refine(obs, x)
        except EstimationFailure as e:
            attempts.append({"step": step, "ok": False, "error": str(e).split(": ", 1)[-1][:80],
                             "iterations": (e.last_state.info.get("iterations") if e.last_state else None)})
            return None, e.last_state
        rms = fit.info["rms_normalized_residual"]
        ok = rms <= self.gate
        attempts.append({"step": step, "ok": ok, "rms": rms, "iterations": fit.info["iterations"]})
        return (fit if ok else None), fit

    def _continuation(self, obs, x, step, attempts):
        start = x
        if self.short_arc and len(obs) > len(self.short_arc):
            try:
                short = self.inner.refine(subset(obs, self.short_arc), x)
                start = StateEstimate(t=x.t, r=short.r, v=short.v, method=x.method, info=dict(x.info))
                attempts.append({"step": f"{step}:short_arc", "ok": True,
                                 "rms": short.info["rms_normalized_residual"], "iterations": short.info["iterations"]})
            except EstimationFailure as e:
                attempts.append({"step": f"{step}:short_arc", "ok": False, "error": str(e).split(": ", 1)[-1][:80]})
        return self._full(obs, start, step, attempts)

    def refine(self, obs: Observations, x0: StateEstimate) -> StateEstimate:
        attempts, seen = [], []

        def done(fit, step):
            fit.method = f"{x0.method} + {self.name}"
            fit.info.update(robust_step=step, robust_attempts=attempts,
                            iterations=int(sum(a.get("iterations") or 0 for a in attempts)))
            return fit

        fit, last = self._continuation(obs, x0, "continuation", attempts)
        seen.append(last)
        if fit is not None:
            return done(fit, "continuation")
        if self.short_arc:
            fit, last = self._full(obs, x0, "plain", attempts)
            seen.append(last)
            if fit is not None:
                return done(fit, "plain")
        k = int(np.argmin(np.abs(obs.t - x0.t)))
        u = obs.los[k]
        passing = []
        for off in self.offsets_m:
            x = StateEstimate(t=x0.t, r=x0.r + off * u, v=x0.v, method=x0.method, info=dict(x0.info))
            fit, last = self._continuation(obs, x, f"restart_{off / 1e3:+,.0f}km", attempts)
            seen.append(last)
            if fit is not None:
                passing.append((fit.info["rms_normalized_residual"], f"restart_{off / 1e3:+,.0f}km", fit))
        if passing:
            _, step, fit = min(passing, key=lambda p: p[0])
            return done(fit, step)
        fits = [s for s in seen if s is not None and "rms_normalized_residual" in s.info]
        best = min(fits, key=lambda s: s.info["rms_normalized_residual"]) if fits else next(
            (s for s in seen if s is not None), None)
        if best is not None:
            best.info.update(state_kind=best.info.get("state_kind", "rejected_fit"), robust_attempts=attempts,
                             iterations=int(sum(a.get("iterations") or 0 for a in attempts)))
        raise EstimationFailure(f"{self.name}: no fit passed the residual gate ({len(attempts)} attempts)", best)


ESTIMATORS = {
    "none": NoEstimation,
    "orekit_bls": OrekitBatchLS,
    "orekit_bls_robust": OrekitBatchLSRobust,
}


# Estimators that do not use Orekit's batch least squares (they import from this module).
from .estimators_cr3bp import CR3BPMultipleShooting, CR3BPUnscented, MultipleShootingThenBLS  # noqa: E402

ESTIMATORS.update({
    "multiple_shooting": CR3BPMultipleShooting,
    "multiple_shooting_then_bls": MultipleShootingThenBLS,
    "ukf": CR3BPUnscented,
})

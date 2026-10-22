"""
IOD methods. Each one takes Observations and returns a StateEstimate.

Adding a method (e.g. the collocation IOD from the paper):
    1. Write a class with a `name` and an `estimate(obs) -> StateEstimate` method.
    2. Add it to IOD_METHODS at the bottom of this file.
    3. Select it in config.yaml (iod.method).

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

    def __init__(self, sys: EarthMoonSystem, truth_path: str | None = None, epoch_index: int = 2,
                 position_error_km: float = 1000.0, velocity_error_m_s: float = 10.0, seed: int = 0, **_):
        if not truth_path:
            raise ValueError("truth_plus_error needs data.truth in the config")
        from .sources import load_synthetic_truth

        self.truth, _, _ = load_synthetic_truth(truth_path, sys)
        self.k = int(epoch_index)
        self.dr, self.dv = float(position_error_km) * 1e3, float(velocity_error_m_s)
        self.rng = np.random.default_rng(seed)

    def _random_dir(self):
        u = self.rng.normal(size=3)
        return u / np.linalg.norm(u)

    def estimate(self, obs: Observations) -> StateEstimate:
        k = self.k
        return StateEstimate(
            t=float(self.truth.t[k]),
            r=self.truth.r[k] + self.dr * self._random_dir(),
            v=self.truth.v[k] + self.dv * self._random_dir(),
            method=self.name,
            info={"obs_indices": [k], "test_stand_in": True},
        )


IOD_METHODS = {
    "truth_plus_error": TruthPlusError,
    "gooding": OrekitGooding,
    "gauss": OrekitGauss,
    "laplace": OrekitLaplace,
}

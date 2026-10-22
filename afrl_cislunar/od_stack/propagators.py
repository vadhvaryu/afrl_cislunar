"""
Propagators. Each one takes a StateEstimate and a list of times, and returns a Trajectory.

Adding a propagator (e.g. Orekit numerical with Earth + Moon + Sun ephemerides):
write a class with `name` and `propagate(x0, t) -> Trajectory`, then add it to
PROPAGATORS and select it in config.yaml (propagator.method).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import StateEstimate, Trajectory
from .dynamics import propagate_cr3bp
from .frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi
from .orekit_setup import start_orekit


class Propagator(Protocol):
    name: str

    def propagate(self, x0: StateEstimate, t: np.ndarray) -> Trajectory: ...


class CR3BPPropagator:
    """Same model as the truth generator, so plotted error comes only from IOD."""

    name = "cr3bp"

    def __init__(self, sys: EarthMoonSystem, **_):
        self.sys = sys

    def propagate(self, x0: StateEstimate, t: np.ndarray) -> Trajectory:
        tau0 = float(self.sys.tau(x0.t))
        x_rot0 = emi_to_rotating(tau0, x0.r, x0.v, self.sys)
        taus = self.sys.tau(t)
        x_rot = propagate_cr3bp(x_rot0, tau0, taus, self.sys.mu)
        rv = [rotating_to_emi(tau, x, self.sys) for tau, x in zip(taus, x_rot)]
        return Trajectory(t=np.asarray(t, float), r=np.array([a for a, _ in rv]),
                          v=np.array([b for _, b in rv]), label=f"{x0.method} + cr3bp")


class OrekitKeplerianPropagator:
    """Two-body (Earth only) baseline, using Orekit."""

    name = "keplerian"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, **_):
        self.sys = sys
        start_orekit(orekit_data)

    def propagate(self, x0: StateEstimate, t: np.ndarray) -> Trajectory:
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.frames import FramesFactory
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.propagation.analytical import KeplerianPropagator
        from org.orekit.time import AbsoluteDate
        from org.orekit.utils import PVCoordinates

        frame, t0 = FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH
        pv0 = PVCoordinates(Vector3D(*map(float, x0.r)), Vector3D(*map(float, x0.v)))
        prop = KeplerianPropagator(CartesianOrbit(pv0, frame, t0.shiftedBy(float(x0.t)), self.sys.gm_earth))
        r, v = [], []
        for ti in t:
            pv = prop.propagate(t0.shiftedBy(float(ti))).getPVCoordinates(frame)
            p, w = pv.getPosition(), pv.getVelocity()
            r.append([p.getX(), p.getY(), p.getZ()])
            v.append([w.getX(), w.getY(), w.getZ()])
        return Trajectory(t=np.asarray(t, float), r=np.array(r), v=np.array(v), label=f"{x0.method} + keplerian")


class OrekitCR3BPPropagator:
    """Orekit NumericalPropagator: Earth gravity + the CR3BP circular Moon (orekit_models.py).
    Same physics as `cr3bp`, but inside Orekit, so it is the model the Batch Least Squares
    stage fits with, and the place to swap in real ephemerides (Sun, real Moon) later."""

    name = "orekit_cr3bp"

    def __init__(self, sys: EarthMoonSystem, orekit_data: str | None = None, **_):
        self.sys = sys
        start_orekit(orekit_data)

    def propagate(self, x0: StateEstimate, t: np.ndarray) -> Trajectory:
        from .orekit_models import frame_and_epoch, propagator_builder

        frame, t0 = frame_and_epoch()
        t = np.asarray(t, float)
        r, v = np.empty((len(t), 3)), np.empty((len(t), 3))
        # One run forward and one backward from the state's epoch, each kept as a dense ephemeris.
        for mask, t_end in ((t >= x0.t, t.max()), (t < x0.t, t.min())):
            if not mask.any():
                continue
            builder = propagator_builder(self.sys, x0.t, x0.r, x0.v)
            prop = builder.buildPropagator(builder.getSelectedNormalizedParameters())
            gen = prop.getEphemerisGenerator()
            prop.propagate(t0.shiftedBy(float(x0.t)), t0.shiftedBy(float(t_end)))
            eph = gen.getGeneratedEphemeris()
            for i in np.flatnonzero(mask):
                pv = eph.getPVCoordinates(t0.shiftedBy(float(t[i])), frame)
                p, w = pv.getPosition(), pv.getVelocity()
                r[i], v[i] = (p.getX(), p.getY(), p.getZ()), (w.getX(), w.getY(), w.getZ())
        return Trajectory(t=t, r=r, v=v, label=f"{x0.method} + {self.name}")


PROPAGATORS = {
    "cr3bp": CR3BPPropagator,
    "orekit_cr3bp": OrekitCR3BPPropagator,
    "keplerian": OrekitKeplerianPropagator,
}

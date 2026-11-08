"""Optional Java object conversions; callers own JVM and frame/epoch setup.

This module does not start Java, load Orekit data, or reinterpret coordinates.
The supplied frame must describe the axes in which ``r`` and ``v`` are expressed.
In particular, labeling synthetic EMI coordinates as GCRF is not a physical
EMI-to-GCRF transformation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_cartesian_orbit(estimate: Any, *, frame: Any, reference_date: Any, gm: float):
    """Convert a stack-like SI state to an Orekit ``CartesianOrbit``.

    ``estimate`` must expose ``t`` (seconds from ``reference_date``), ``r``
    (meters), and ``v`` (meters/second). ``gm`` is the central body's GM in
    m^3/s^2. Start the JVM before calling; the function has no configuration
    or Orekit-data dependency of its own.
    """
    r, v = np.asarray(estimate.r, dtype=float), np.asarray(estimate.v, dtype=float)
    t, gm = float(estimate.t), float(gm)
    if r.shape != (3,) or v.shape != (3,) or not np.isfinite(np.r_[r, v, t, gm]).all():
        raise ValueError("Expected a finite SI state with three position and velocity components")
    if gm <= 0:
        raise ValueError("gm must be positive, in m^3/s^2")
    try:
        import jpype
    except ImportError as exc:
        raise ImportError("Install collocation-iod[orekit] to use the Orekit bridge") from exc
    if not jpype.isJVMStarted():
        raise RuntimeError("Start Orekit's JVM before calling to_cartesian_orbit")

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import CartesianOrbit
    from org.orekit.utils import PVCoordinates

    pv = PVCoordinates(Vector3D(*map(float, r)), Vector3D(*map(float, v)))
    return CartesianOrbit(pv, frame, reference_date.shiftedBy(t), gm)


def from_cartesian_orbit(orbit: Any, *, frame: Any, reference_date: Any):
    """Return ``(seconds, position_m, velocity_m_s)`` in an explicit frame.

    All transformations are performed by Orekit using the supplied objects;
    callers are responsible for mapping these axes to the solver's CR3BP frame.
    """
    pv = orbit.getPVCoordinates(frame)
    p, v = pv.getPosition(), pv.getVelocity()
    return (
        float(orbit.getDate().durationFrom(reference_date)),
        np.array([p.getX(), p.getY(), p.getZ()], dtype=float),
        np.array([v.getX(), v.getY(), v.getZ()], dtype=float),
    )

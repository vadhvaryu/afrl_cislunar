"""
Orekit building blocks shared by the Batch Least Squares stage and the Orekit propagator.

Frame note: the EMI frame (see contracts.py) is passed to Orekit labeled as GCRF.
That is self-consistent here because nothing in these models depends on Earth's real
orientation: the "Moon" is the CR3BP circular Moon, and the observer sits at Earth's
center. Real ephemerides or a real ground station will need the true EMI -> GCRF
rotation and a real UTC epoch.
"""

from __future__ import annotations

import math

from .frames import EarthMoonSystem
from .orekit_setup import start_orekit

_cache: dict = {}


def frame_and_epoch():
    from org.orekit.frames import FramesFactory
    from org.orekit.time import AbsoluteDate

    return FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH


def _circular_moon_class():
    """Defined lazily: Java interfaces can only be implemented once the JVM runs."""
    if "moon" in _cache:
        return _cache["moon"]
    from jpype import JImplements, JOverride
    from org.hipparchus.geometry.euclidean.threed import FieldVector3D, Vector3D
    from org.orekit.time import FieldAbsoluteDate

    @JImplements("org.orekit.utils.ExtendedPositionProvider")
    class CircularMoon:
        """Moon on the CR3BP circle: radius DU around Earth, rate 1/TU, on the +x axis at t0."""

        def __init__(self, frame, t0, rate, radius):
            self.frame, self.t0, self.rate, self.radius = frame, t0, rate, radius

        @JOverride
        def getPosition(self, date, frame):
            if not frame.getName() == self.frame.getName():
                raise ValueError("CircularMoon only works in the EMI frame (passed to Orekit as GCRF)")
            if isinstance(date, FieldAbsoluteDate):     # used for partial derivatives
                dt = date.durationFrom(self.t0)
                th = dt.multiply(self.rate)
                return FieldVector3D(th.cos().multiply(self.radius), th.sin().multiply(self.radius),
                                     dt.getField().getZero())
            th = date.durationFrom(self.t0) * self.rate
            return Vector3D(self.radius * math.cos(th), self.radius * math.sin(th), 0.0)

    _cache["moon"] = CircularMoon
    return CircularMoon


def cr3bp_force_models(sys: EarthMoonSystem) -> list:
    """Forces that, added to Earth's central gravity, reproduce the CR3BP exactly:
    the Moon's pull on the spacecraft minus its pull on Earth (Orekit's ThirdBodyAttraction)."""
    from org.orekit.forces.gravity import ThirdBodyAttraction

    frame, t0 = frame_and_epoch()
    moon = _circular_moon_class()(frame, t0, 1.0 / sys.tu_s, sys.du_m)
    return [ThirdBodyAttraction(moon, "Moon", sys.gm_moon)]


BUILDER_DEFAULTS = {"force_model": "cr3bp", "integrator": "dormand_prince_853", "min_step_s": 0.001,
                    "max_step_s": 3600.0, "position_tolerance_m": 0.01, "position_scale_m": 1.0}


def propagator_builder(sys: EarthMoonSystem, t: float, r, v, settings: dict | None = None,
                       orbit_type: str = "cartesian"):
    """Stages 4-5 of the chain: build the Orekit Orbit from a state, then the
    NumericalPropagatorBuilder (force model + integrator) around it.

    orbit_type also decides which 6 numbers Batch Least Squares adjusts. Cartesian is the
    default because orbital elements around Earth mean little in cislunar space."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngleType
    from org.orekit.propagation.conversion import (DormandPrince54IntegratorBuilder,
                                                   DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder)
    from org.orekit.utils import PVCoordinates

    s = {**BUILDER_DEFAULTS, **(settings or {})}
    frame, t0 = frame_and_epoch()
    orbit = CartesianOrbit(PVCoordinates(Vector3D(*map(float, r)), Vector3D(*map(float, v))),
                           frame, t0.shiftedBy(float(t)), sys.gm_earth)
    if orbit_type != "cartesian":
        orbit = OrbitType.valueOf(orbit_type.upper()).convertType(orbit)

    integrators = {"dormand_prince_853": DormandPrince853IntegratorBuilder,
                   "dormand_prince_54": DormandPrince54IntegratorBuilder}
    integrator = integrators[s["integrator"]](float(s["min_step_s"]), float(s["max_step_s"]),
                                              float(s["position_tolerance_m"]))
    builder = NumericalPropagatorBuilder(orbit, integrator, PositionAngleType.TRUE, float(s["position_scale_m"]))
    if s["force_model"] == "cr3bp":
        for force in cr3bp_force_models(sys):
            builder.addForceModel(force)
    else:
        raise ValueError(f"unknown force model {s['force_model']}")
    return builder


def geocenter_station(offset_m: float = 1.0):
    """An Orekit GroundStation at (almost) Earth's center, matching obs_generator.py v1.

    Placed 1 m from the center: exactly at the center, Orekit's station geometry is
    undefined and returns NaN. 1 m shifts the line of sight by ~0.0006 arcsec,
    far below the 10 arcsec noise. Needs orekit-data (leap seconds) to be loaded.
    """
    key = ("station", float(offset_m))
    if key in _cache:
        return _cache[key]
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.estimation.measurements import GroundStation
    from org.orekit.frames import TopocentricFrame

    frame, _ = frame_and_epoch()
    radius = 6378137.0
    # A non-rotating sphere: no Earth-rotation effects can enter.
    topo = TopocentricFrame(OneAxisEllipsoid(radius, 0.0, frame), GeodeticPoint(0.0, 0.0, -radius + float(offset_m)), "geocenter")
    try:
        try:
            station = GroundStation(topo)
        except Exception as e:
            if "Earth Orientation" not in str(e):
                raise
            station = GroundStation(topo, _zero_eop(), _no_displacements())
    except Exception as e:
        if "UTC-TAI" in str(e):
            raise RuntimeError("Orekit needs orekit-data for this stage: set orekit.data_path in "
                               "config.yaml or the OREKIT_DATA environment variable") from None
        raise
    _cache[key] = station
    return station


def _zero_eop():
    """Earth-orientation values set to zero. Only used when no EOP files are loaded; they
    cannot affect results here (the station is at Earth's center of a non-rotating sphere)."""
    from java.util import ArrayList
    from org.orekit.data import DataContext
    from org.orekit.frames import EOPEntry, EOPHistory, ITRFVersion
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import IERSConventions

    tt = TimeScalesFactory.getTT()
    entries = ArrayList()
    for mjd in range(51500, 51700):
        date = AbsoluteDate(2000, 1, 1, tt).shiftedBy((mjd - 51544) * 86400.0)
        entries.add(EOPEntry(mjd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ITRFVersion.ITRF_2014, date))
    return EOPHistory(IERSConventions.IERS_2010, 3, entries, True, DataContext.getDefault().getTimeScales())


def _no_displacements():
    from jpype import JArray
    from org.orekit.models.earth.displacement import StationDisplacement

    return JArray(StationDisplacement)([])

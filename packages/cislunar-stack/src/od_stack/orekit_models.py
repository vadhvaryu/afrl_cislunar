"""
Orekit building blocks shared by the Batch Least Squares stage, the Orekit propagator and
the synthetic-observation generator.

Frames. Without a calendar epoch (system.epoch_utc: null) the EMI frame is passed to Orekit
labeled as GCRF with t0 = J2000. That is self-consistent while the observer sits at Earth's
center or at a point fixed in EMI, because nothing then depends on Earth's real orientation.

With system.epoch_utc set, EMI becomes a real Orekit frame: a fixed rotation of GCRF
(frames.emi_axes_in_gcrf) with t0 = the epoch. A ground site is then an Orekit station on the
real, rotating WGS-84 Earth (ITRF), and Orekit handles EMI <-> GCRF <-> ITRF. Without Earth
orientation files (only leap seconds are bundled) Orekit uses zero EOP corrections: under a
second of Earth rotation, about 0.2 arcsec of line of sight at lunar distance.
"""

from __future__ import annotations

import math

import numpy as np

from .frames import EarthMoonSystem, emi_axes_in_gcrf, parse_epoch
from .orekit_setup import start_orekit

_cache: dict = {}
_reference: dict = {"epoch_utc": None}


def configure_reference(epoch_utc=None) -> None:
    """Choose the calendar epoch for t = 0 (None: legacy EMI-as-GCRF, t0 = J2000).
    Clears cached frames and stations when it changes."""
    epoch = None if epoch_utc is None else parse_epoch(epoch_utc).isoformat()
    if epoch != _reference["epoch_utc"]:
        for key in [k for k in _cache if k != "moon"]:
            del _cache[key]
        _reference["epoch_utc"] = epoch


def reference_epoch():
    return _reference["epoch_utc"]


def frame_and_epoch():
    """(EMI frame, date of t = 0) as Orekit objects."""
    if "frame" in _cache:
        return _cache["frame"]
    from org.orekit.frames import FramesFactory
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    if _reference["epoch_utc"] is None:
        _cache["frame"] = (FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH)
        return _cache["frame"]
    from jpype import JArray, JDouble
    from org.hipparchus.geometry.euclidean.threed import Rotation
    from org.orekit.frames import Frame, Transform

    dt = parse_epoch(_reference["epoch_utc"])
    t0 = AbsoluteDate(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second + dt.microsecond * 1e-6, TimeScalesFactory.getUTC())
    R = emi_axes_in_gcrf(dt)                    # r_gcrf = R @ r_emi, so r_emi = R.T @ r_gcrf
    rotation = Rotation(JArray(JArray(JDouble))([[float(a) for a in row] for row in R.T]), 1e-10)
    emi = Frame(FramesFactory.getGCRF(), Transform(t0, rotation), "EMI", True)
    _cache["frame"] = (emi, t0)
    return _cache["frame"]

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



def inertial_station(position_m=(0.0, 0.0, 0.0), offset_m: float = 1.0):
    """An Orekit GroundStation at a point fixed in EMI: Earth's center (obs_generator.py v1)
    or a fixed_emi observer. It sits on a non-rotating sphere attached to the EMI frame, so no
    Earth rotation enters.

    At the center itself Orekit's station geometry is undefined (NaN), so the geocenter
    station is placed offset_m (1 m) away: about 0.0006 arcsec, far below any noise.
    Needs orekit-data (leap seconds)."""
    p = np.asarray(position_m, dtype=float)
    key = ("inertial", tuple(p), float(offset_m))
    if key in _cache:
        return _cache[key]
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.frames import TopocentricFrame

    frame, _ = frame_and_epoch()
    radius = 6378137.0
    r = float(np.linalg.norm(p))
    if r < 1.0:
        point = GeodeticPoint(0.0, 0.0, -radius + float(offset_m))
    else:
        point = GeodeticPoint(math.asin(p[2] / r), math.atan2(p[1], p[0]), r - radius)
    topo = TopocentricFrame(OneAxisEllipsoid(radius, 0.0, frame), point, "fixed observer")
    _cache[key] = _station(topo)
    return _cache[key]


def geocenter_station(offset_m: float = 1.0):
    return inertial_station((0.0, 0.0, 0.0), offset_m)


def ground_site(latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0):
    """(TopocentricFrame, GroundStation) for a site on the real, rotating Earth (WGS-84,
    ITRF). Needs system.epoch_utc, so that Earth's rotation is known."""
    if _reference["epoch_utc"] is None:
        raise RuntimeError("A ground site needs system.epoch_utc (the calendar date and time of t = 0)")
    key = ("site", float(latitude_deg), float(longitude_deg), float(altitude_m))
    if key in _cache:
        return _cache[key]
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.utils import Constants, IERSConventions

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    topo = TopocentricFrame(earth, GeodeticPoint(math.radians(latitude_deg), math.radians(longitude_deg),
                                                 float(altitude_m)), "ground site")
    _cache[key] = (topo, _station(topo))
    return _cache[key]


def observer_station(observer: dict, offset_m: float = 1.0):
    """The Orekit GroundStation for a simulation.observer block."""
    kind = observer.get("type", "geocenter")
    if kind == "ground_site":
        return ground_site(observer["latitude_deg"], observer["longitude_deg"], observer.get("altitude_m", 0.0))[1]
    position = np.asarray(observer.get("position_km", (0.0, 0.0, 0.0)), dtype=float) * 1e3
    return inertial_station(position if kind == "fixed_emi" else (0.0, 0.0, 0.0), offset_m)


def observer_positions(observer: dict, t_s) -> np.ndarray:
    """(N, 3) observer positions in EMI [m] at times t_s [s since t0]."""
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    kind = observer.get("type", "geocenter")
    if kind != "ground_site":
        p = np.asarray(observer.get("position_km", (0.0, 0.0, 0.0)), dtype=float) * 1e3
        return np.tile(p if kind == "fixed_emi" else np.zeros(3), (len(t_s), 1))
    topo, _ = ground_site(observer["latitude_deg"], observer["longitude_deg"], observer.get("altitude_m", 0.0))
    frame, t0 = frame_and_epoch()
    out = np.empty((len(t_s), 3))
    for i, t in enumerate(t_s):
        p = topo.getPVCoordinates(t0.shiftedBy(float(t)), frame).getPosition()
        out[i] = (p.getX(), p.getY(), p.getZ())
    return out


def elevations_deg(observer: dict, t_s, r_emi) -> np.ndarray:
    """Elevation above the site's horizon [deg] of EMI positions r_emi [m] at times t_s."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    topo, _ = ground_site(observer["latitude_deg"], observer["longitude_deg"], observer.get("altitude_m", 0.0))
    frame, t0 = frame_and_epoch()
    return np.array([math.degrees(topo.getElevation(Vector3D(*map(float, r)), frame, t0.shiftedBy(float(t))))
                     for t, r in zip(np.atleast_1d(t_s), np.atleast_2d(r_emi))])


def _station(topo):
    from org.orekit.estimation.measurements import GroundStation

    try:
        try:
            return GroundStation(topo)
        except Exception as e:
            if "Earth Orientation" not in str(e):
                raise
            return GroundStation(topo, _zero_eop(), _no_displacements())
    except Exception as e:
        if "UTC-TAI" in str(e):
            raise RuntimeError("Orekit needs orekit-data for this stage: set orekit.data_path in "
                               "config.yaml or the OREKIT_DATA environment variable") from None
        raise


def _zero_eop():
    """Earth-orientation values set to zero, spanning the run's epoch. Only used when no EOP
    files are loaded; they cannot affect a station that does not rotate with Earth."""
    from java.util import ArrayList
    from org.orekit.data import DataContext
    from org.orekit.frames import EOPEntry, EOPHistory, ITRFVersion
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import IERSConventions

    _, t0 = frame_and_epoch()
    tt = TimeScalesFactory.getTT()
    j2000_mjd = 51544
    centre = j2000_mjd + int(t0.durationFrom(AbsoluteDate(2000, 1, 1, tt)) // 86400)
    entries = ArrayList()
    for mjd in range(centre - 400, centre + 400):
        date = AbsoluteDate(2000, 1, 1, tt).shiftedBy((mjd - j2000_mjd) * 86400.0)
        entries.add(EOPEntry(mjd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ITRFVersion.ITRF_2014, date))
    return EOPHistory(IERSConventions.IERS_2010, 3, entries, True, DataContext.getDefault().getTimeScales())


def _no_displacements():
    from jpype import JArray
    from org.orekit.models.earth.displacement import StationDisplacement

    return JArray(StationDisplacement)([])

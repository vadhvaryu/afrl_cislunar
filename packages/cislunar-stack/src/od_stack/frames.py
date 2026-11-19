"""
The one place where CR3BP rotating, nondimensional states are converted to and
from the EMI frame in SI units. Uses the same conventions as obs_generator.py:
Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0), and the rotating frame has turned
by angle tau (nondimensional time) since t0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_HAT = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class EarthMoonSystem:
    mu: float = 0.0121505      # Earth-Moon mass ratio  (obs_generator.py: MU)
    du_km: float = 384338.0    # distance unit [km]     (obs_generator.py: DU)
    tu_s: float = 375099.0     # time unit [s]          (obs_generator.py: TU)

    @property
    def du_m(self) -> float:
        return self.du_km * 1e3

    @property
    def vu_ms(self) -> float:
        return self.du_m / self.tu_s

    @property
    def earth_rot(self) -> np.ndarray:
        """Earth's position in the rotating frame (nondimensional)."""
        return np.array([-self.mu, 0.0, 0.0])

    @property
    def gm_earth(self) -> float:
        """Earth's GM [m^3/s^2], consistent with mu, DU and TU (about 3.986e14)."""
        return (1.0 - self.mu) * self.du_m**3 / self.tu_s**2

    @property
    def gm_moon(self) -> float:
        """Moon's GM [m^3/s^2], consistent with mu, DU and TU (about 4.90e12)."""
        return self.mu * self.du_m**3 / self.tu_s**2

    def tau(self, t_s):
        return np.asarray(t_s) / self.tu_s


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotating_to_emi(tau: float, x_rot: np.ndarray, sys: EarthMoonSystem):
    """Rotating nondimensional state [x, y, z, vx, vy, vz] -> EMI (r [m], v [m/s])."""
    rho = np.asarray(x_rot[:3]) - sys.earth_rot         # Earth -> object, rotating axes
    rho_dot = np.asarray(x_rot[3:6])
    R = rot_z(tau)
    r = R @ rho * sys.du_m
    v = R @ (rho_dot + np.cross(Z_HAT, rho)) * sys.vu_ms   # add frame rotation (omega = 1)
    return r, v


def emi_to_rotating(tau: float, r: np.ndarray, v: np.ndarray, sys: EarthMoonSystem) -> np.ndarray:
    """EMI (r [m], v [m/s]) -> rotating nondimensional state [x, y, z, vx, vy, vz]."""
    Rt = rot_z(tau).T
    rho = Rt @ (np.asarray(r) / sys.du_m)
    rho_dot = Rt @ (np.asarray(v) / sys.vu_ms) - np.cross(Z_HAT, rho)
    return np.concatenate([rho + sys.earth_rot, rho_dot])


def radec_to_los(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Right ascension / declination [rad] -> unit line-of-sight vectors."""
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def trajectory_to_rotating(traj, sys: EarthMoonSystem) -> np.ndarray:
    """Trajectory (EMI, SI) -> (M, 6) rotating nondimensional states, e.g. for CR3BP plots."""
    return np.array([emi_to_rotating(sys.tau(t), r, v, sys) for t, r, v in zip(traj.t, traj.r, traj.v)])


# --- Real calendar time: orienting EMI in the sky -------------------------------------------
#
# The CR3BP needs no calendar date. A ground observer does: Earth's rotation, and where the
# Moon (and so the whole EMI frame) sits in the sky, depend on the real date and time.
# With system.epoch_utc set, EMI is tied to GCRF (Earth's mean equator and equinox of J2000)
# from the Moon's real position at that epoch:
#     x  = direction to the Moon at the epoch, projected into the Moon's orbit plane
#     z  = normal of the Moon's orbit plane (mean elements)
#     y  = z cross x
# The Moon comes from a low-precision analytic series (Meeus, Astronomical Algorithms ch. 47,
# main terms; about 0.3 deg accurate), so no ephemeris files are needed. This only decides how
# the idealized CR3BP is placed in the real sky; it does not change the dynamics.

OBLIQUITY_J2000_DEG = 23.4392911


def parse_epoch(value):
    """'2026-11-20T08:00:00Z' (or a datetime) -> timezone-aware UTC datetime."""
    from datetime import datetime, timezone

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _julian_centuries(dt) -> float:
    """Julian centuries (TT, to within about a minute) since J2000 for a UTC datetime."""
    from datetime import datetime, timezone

    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (dt - j2000).total_seconds() / 86400.0 + 69.184 / 86400.0
    return days / 36525.0


def moon_ecliptic(dt):
    """Approximate geocentric ecliptic longitude and latitude of the Moon [deg], mean ecliptic
    and equinox of J2000, plus the longitude of its mean ascending node [deg]."""
    T = _julian_centuries(dt)
    d = np.radians
    Lp = 218.3164477 + 481267.88123421 * T
    D = d(297.8501921 + 445267.1114034 * T)
    M = d(357.5291092 + 35999.0502909 * T)
    Mp = d(134.9633964 + 477198.8675055 * T)
    F = d(93.2720950 + 483202.0175233 * T)
    lon = (Lp + 6.289 * np.sin(Mp) + 1.274 * np.sin(2 * D - Mp) + 0.658 * np.sin(2 * D)
           + 0.214 * np.sin(2 * Mp) - 0.186 * np.sin(M) - 0.114 * np.sin(2 * F))
    lat = (5.128 * np.sin(F) + 0.281 * np.sin(Mp + F) + 0.278 * np.sin(Mp - F)
           + 0.173 * np.sin(2 * D - F))
    node = 125.0445479 - 1934.1362891 * T
    precession = 1.3969713 * T          # equinox of date -> J2000
    return (lon - precession) % 360.0, lat, (node - precession) % 360.0


def emi_axes_in_gcrf(epoch) -> np.ndarray:
    """3x3 matrix whose columns are the EMI x, y, z axes written in GCRF, so that
    r_gcrf = R @ r_emi for the given epoch (see the note above)."""
    lon, lat, node = (np.radians(a) for a in moon_ecliptic(parse_epoch(epoch)))
    inc = np.radians(5.145)
    moon = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    normal = np.array([np.sin(inc) * np.sin(node), -np.sin(inc) * np.cos(node), np.cos(inc)])
    x = moon - (moon @ normal) * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    eps = np.radians(OBLIQUITY_J2000_DEG)
    ecl_to_eq = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(eps), -np.sin(eps)], [0.0, np.sin(eps), np.cos(eps)]])
    return ecl_to_eq @ np.column_stack((x, y, normal))

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

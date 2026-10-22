"""
Data sources: turn files into Observations (and the truth file into a Trajectory).

A new data source (a real ground station, real telescope data, a different
generator) only needs a new function that returns an Observations object.
"""

from __future__ import annotations

import csv
import warnings

import numpy as np

from .contracts import Observations, Trajectory
from .dynamics import propagate_cr3bp
from .frames import EarthMoonSystem, radec_to_los, rotating_to_emi

OBSERVER_COLUMNS = ("obs_x_km", "obs_y_km", "obs_z_km")


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_synthetic_obs(path: str, sys: EarthMoonSystem) -> Observations:
    """Read synthetic_obs.csv from obs_generator.py.

    If the file has observer columns (obs_x_km, obs_y_km, obs_z_km in EMI), they
    are used. Otherwise the observer is assumed to be at Earth's center, which is
    what obs_generator.py v1 does.
    """
    rows = _read_csv(path)
    t = np.array([float(r["time_days"]) for r in rows]) * 86400.0
    ra = np.radians([float(r["ra_deg"]) for r in rows])
    dec = np.radians([float(r["dec_deg"]) for r in rows])
    sigma = np.radians([float(r["sigma_arcsec"]) for r in rows]) / 3600.0

    if all(c in rows[0] for c in OBSERVER_COLUMNS):
        observer = np.array([[float(r[c]) for c in OBSERVER_COLUMNS] for r in rows]) * 1e3
    else:
        warnings.warn("No observer columns in observation file: assuming observer at Earth's center.")
        observer = np.zeros((len(rows), 3))

    return Observations(t=t, los=radec_to_los(ra, dec), observer=observer, sigma=sigma, ra=ra, dec=dec)


def load_synthetic_truth(path: str, sys: EarthMoonSystem):
    """Read synthetic_truth.csv. Returns (Trajectory in EMI/SI, raw rotating states, taus)."""
    rows = _read_csv(path)
    t = np.array([float(r["time_days"]) for r in rows]) * 86400.0
    x_rot = np.array([[float(r[k]) for k in ("x", "y", "z", "vx", "vy", "vz")] for r in rows])
    taus = sys.tau(t)
    rv = [rotating_to_emi(tau, x, sys) for tau, x in zip(taus, x_rot)]
    traj = Trajectory(t=t, r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv]), label="Truth")
    return traj, x_rot, taus


def dense_truth(truth_rot: np.ndarray, truth_taus: np.ndarray, t_s: np.ndarray, sys: EarthMoonSystem) -> Trajectory:
    """Truth at arbitrary times, by re-propagating the first truth state with the
    same CR3BP model the generator used (for smooth error plots)."""
    taus = sys.tau(t_s)
    x = propagate_cr3bp(truth_rot[0], truth_taus[0], taus, sys.mu)
    rv = [rotating_to_emi(tau, xi, sys) for tau, xi in zip(taus, x)]
    return Trajectory(t=np.asarray(t_s), r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv]), label="Truth")

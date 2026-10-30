"""Other truth orbits (od_stack/orbit_catalog.py). No Orekit needed."""

import csv
import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from od_stack.dynamics import TOL, cr3bp_eom
from od_stack.frames import EarthMoonSystem
from od_stack.orbit_catalog import ORBITS, correct_periodic, describe, write_orbit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("name", ["NRHO_9_2", "DRO", "L1_lyapunov", "L2_halo_large"])
def test_corrected_orbits_close_after_one_period(name):
    seed, planar, _ = ORBITS[name]
    x0, T = correct_periodic(seed, SYS.mu, planar)
    xT = solve_ivp(cr3bp_eom, [0, T], x0, args=(SYS.mu,), **TOL).y[:, -1]
    assert np.abs(xT - x0).max() < 1e-7
    assert np.abs(x0 - seed).max() < 1e-3                  # the mu change only nudges the seed


def test_nrho_matches_the_published_9_2_orbit():
    seed, planar, _ = ORBITS["NRHO_9_2"]
    x0, T = correct_periodic(seed, SYS.mu, planar)
    d = describe(x0, T, SYS.mu, SYS)
    assert d["period_days"] == pytest.approx(6.56, abs=0.02)   # 9:2 resonance with the lunar synodic month
    assert 3_000 < d["perilune_km"] < 3_500 and 65_000 < d["apolune_km"] < 75_000


def test_regenerated_truth_orbit_matches_the_generator(tmp_path):
    """L1_halo_truth through write_orbit reproduces data/synthetic_truth.csv and synthetic_obs.csv."""
    paths = write_orbit(str(tmp_path), "L1_halo_truth", np.array(ORBITS["L1_halo_truth"][0]), SYS)
    def rows(p):
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    mine, his = rows(paths["truth"]), rows("data/synthetic_truth.csv")
    for a, b in zip(mine, his):
        assert all(abs(float(a[k]) - float(b[k])) < 2e-10 for k in ("x", "y", "z", "vx", "vy", "vz"))
        assert abs(float(a["ra_true_deg"]) - float(b["ra_true_deg"])) < 1e-8
        assert abs(float(a["dec_true_deg"]) - float(b["dec_true_deg"])) < 1e-8
    for a, b in zip(rows(paths["obs"]), rows("data/synthetic_obs.csv")):
        assert abs(float(a["ra_deg"]) - float(b["ra_deg"])) < 1e-8 and abs(float(a["dec_deg"]) - float(b["dec_deg"])) < 1e-8

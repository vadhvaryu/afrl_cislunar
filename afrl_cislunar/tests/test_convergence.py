"""Tests for the convergence study (Part A): stand-in extensions, features, parallel runner.
No Orekit needed. Run from the folder that holds config.yaml and data/:   python -m pytest -q"""

import copy
import os
import warnings

import numpy as np
import pytest

from od_stack.contracts import Observations
from od_stack.frames import EarthMoonSystem
from od_stack.iod import TruthPlusError
from od_stack.run import load_config
from od_stack.sources import load_synthetic_obs
from od_stack.sweep import Sweeper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH = "data/synthetic_truth.csv"
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _obs():
    return load_synthetic_obs("data/synthetic_obs.csv", SYS)


def _no_orekit_cfg():
    cfg = load_config("config.yaml")
    cfg["estimation"] = {"method": "none"}
    cfg["propagator"]["method"] = "cr3bp"
    cfg["iod"].update(method="truth_plus_error")
    cfg["sweeps"]["check_points"] = 20
    return cfg


# ----------------------------------------------------------------------------- A1: stand-in extensions

def test_stand_in_default_directions_unchanged():
    """Without overrides the stand-in draws exactly what it did before (seed -> same directions)."""
    x = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=1000, velocity_error_m_s=10, seed=7).estimate(_obs())
    rng = np.random.default_rng(7)
    u_r, u_v = rng.normal(size=3), rng.normal(size=3)
    truth = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=0, velocity_error_m_s=0).estimate(_obs())
    assert np.allclose((x.r - truth.r) / 1e6, u_r / np.linalg.norm(u_r), atol=1e-12)
    assert np.allclose((x.v - truth.v) / 10, u_v / np.linalg.norm(u_v), atol=1e-12)


def test_stand_in_direction_overrides():
    truth = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=0, velocity_error_m_s=0).estimate(_obs())
    x = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=2000, velocity_error_m_s=5, seed=3,
                       position_direction=[0, 0, 4.0], velocity_direction=[1.0, 1.0, 0]).estimate(_obs())
    assert np.allclose(x.r - truth.r, [0, 0, 2e6], atol=1e-6)
    assert np.allclose(x.v - truth.v, 5 * np.array([1, 1, 0]) / np.sqrt(2), atol=1e-12)
    # Overriding only the position keeps the velocity on its usual (second) random draw.
    y = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=2000, velocity_error_m_s=5, seed=3,
                       position_direction=[0, 0, 1]).estimate(_obs())
    z = TruthPlusError(SYS, truth_path=TRUTH, position_error_km=2000, velocity_error_m_s=5, seed=3).estimate(_obs())
    assert np.allclose(y.v, z.v)
    with pytest.raises(ValueError):
        TruthPlusError(SYS, truth_path=TRUTH, position_direction=[0, 0, 0])


def test_sweeper_obs_seed_default_and_override():
    sw = Sweeper(copy.deepcopy(_no_orekit_cfg()))
    ov = {"position_error_km": 100.0, "velocity_error_m_s": 1.0}
    a = sw.one(10.0, 4, {**ov, "seed": 4})
    b = sw.one(10.0, 4, {**ov, "seed": 4}, obs_seed=10_004)
    c = sw.one(10.0, 4, {**ov, "seed": 4}, obs_seed=10_000)
    assert a["obs_seed"] == b["obs_seed"] == 10_004 and c["obs_seed"] == 10_000
    assert a["position_error_km"] == b["position_error_km"]          # estimation none: error = IOD error
    seen = {}
    sw.one(10.0, 4, {**ov, "seed": 4}, obs_seed=10_000,
           features=lambda obs, x_iod, x0: seen.update(ra=obs.ra.copy()) or {"extra": 1.0})
    sw2 = sw.one(10.0, 9, {**ov, "seed": 9}, obs_seed=10_000, features=lambda o, xi, x0: {"extra": 2.0})
    assert sw2["extra"] == 2.0
    # Same obs_seed -> same noise realization regardless of the direction seed.
    seen2 = {}
    sw.one(10.0, 9, {**ov, "seed": 9}, obs_seed=10_000,
           features=lambda obs, x_iod, x0: seen2.update(ra=obs.ra.copy()) or {})
    assert np.array_equal(seen["ra"], seen2["ra"])

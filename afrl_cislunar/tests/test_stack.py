"""Run from the folder that holds config.yaml and data/:   python -m pytest -q"""

import copy
import csv
import os
import warnings

import numpy as np
import pytest

from od_stack.contracts import Observations
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi
from od_stack.run import load_config, run
from od_stack.sources import load_synthetic_truth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _truth_rows():
    with open("data/synthetic_truth.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_frame_round_trip():
    x = np.array([0.9023581414, 0.0936861644, 0.1421356838, 0.1218991572, 0.1379776334, -0.2067777368])
    r, v = rotating_to_emi(0.46, x, SYS)
    assert np.allclose(emi_to_rotating(0.46, r, v, SYS), x, atol=1e-13)


def test_dynamics_match_generator():
    """Our CR3BP model reproduces every truth row from the first one -> same model as cr3bp.py."""
    truth, x_rot, taus = load_synthetic_truth("data/synthetic_truth.csv", SYS)
    x = propagate_cr3bp(x_rot[0], taus[0], taus, SYS.mu)
    assert np.abs(x - x_rot).max() < 1e-8


def test_frame_matches_generator_angles():
    """RA/Dec computed from our EMI positions equal the generator's noise-free angles."""
    truth, _, _ = load_synthetic_truth("data/synthetic_truth.csv", SYS)
    rows = _truth_rows()
    ra = np.degrees(np.arctan2(truth.r[:, 1], truth.r[:, 0])) % 360
    dec = np.degrees(np.arcsin(truth.r[:, 2] / np.linalg.norm(truth.r, axis=1)))
    assert np.allclose(ra, [float(r["ra_true_deg"]) for r in rows], atol=1e-6)
    assert np.allclose(dec, [float(r["dec_true_deg"]) for r in rows], atol=1e-6)
    assert np.allclose(np.linalg.norm(truth.r, axis=1) / 1e3, [float(r["range_km"]) for r in rows], atol=1e-2)


def _cfg(**iod):
    cfg = load_config("config.yaml")
    cfg["iod"].update(method="truth_plus_error", **iod)
    cfg["estimation"] = {"method": "none"}
    cfg["propagator"]["method"] = "cr3bp"
    return cfg


def test_zero_error_round_trip():
    """Truth in -> CR3BP propagator -> results must give ~zero error everywhere."""
    s = run(_cfg(position_error_km=0, velocity_error_m_s=0), make_files=False)
    assert s["max_position_error_km"] < 1e-3


def test_orekit_propagator_matches_cr3bp():
    """Orekit with Earth + circular Moon reproduces the CR3BP truth (same physics)."""
    pytest.importorskip("orekit_jpype")
    cfg = _cfg(position_error_km=0, velocity_error_m_s=0)
    cfg["propagator"]["method"] = "orekit_cr3bp"
    s = run(cfg, make_files=False)
    assert s["max_position_error_km"] < 0.01      # 10 m over 8 days


def test_stand_in_injects_requested_error():
    s = run(_cfg(position_error_km=1000, velocity_error_m_s=10), make_files=False)
    assert s["iod_position_error_km"] == pytest.approx(1000, rel=1e-6)
    assert s["iod_velocity_error_m_s"] == pytest.approx(10, rel=1e-6)


def test_gooding_wiring_on_two_body_orbit():
    """Gooding recovers a pure two-body orbit seen from Earth's surface -> plumbing is right."""
    pytest.importorskip("orekit_jpype")
    from od_stack.iod import OrekitGooding
    from od_stack.orekit_setup import start_orekit

    start_orekit()
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.time import AbsoluteDate

    frame, t0 = FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH
    orbit = KeplerianOrbit(330e6, 0.05, 0.3, 0.3, 0.4, 0.0, PositionAngleType.TRUE, frame, t0, SYS.gm_earth)
    t = np.array([0.0, 2.0, 4.0]) * 86400
    prop = KeplerianPropagator(orbit)
    r = np.array([[p.getX(), p.getY(), p.getZ()]
                  for p in (prop.propagate(t0.shiftedBy(float(ti))).getPosition() for ti in t)])
    observer = np.column_stack([6378e3 * np.cos(7.29e-5 * t), 6378e3 * np.sin(7.29e-5 * t), 0 * t])
    los = (r - observer) / np.linalg.norm(r - observer, axis=1)[:, None]
    obs = Observations(t=t, los=los, observer=observer, sigma=np.zeros(3), ra=np.zeros(3), dec=np.zeros(3))

    x0 = OrekitGooding(SYS, obs_indices=[0, 1, 2], range_guess_km=[300000, 300000]).estimate(obs)
    k = int(np.argmin(abs(t - x0.t)))
    assert np.linalg.norm(x0.r - r[k]) < 1.0   # meters


def _orekit_data_available():
    path = load_config(os.path.join(ROOT, "config.yaml")).get("orekit", {}).get("data_path") or os.environ.get("OREKIT_DATA")
    return bool(path) and os.path.exists(os.path.expanduser(path))


@pytest.mark.skipif(not _orekit_data_available(), reason="set orekit.data_path or OREKIT_DATA to run")
def test_batch_least_squares_recovers_orbit():
    """BLS pulls a 1,000 km / 10 m/s starting error in to well under 100 km,
    and the true error lies inside 3 sigma of the covariance it reports."""
    pytest.importorskip("orekit_jpype")
    cfg = _cfg(position_error_km=1000, velocity_error_m_s=10)
    cfg["estimation"] = {"method": "orekit_bls"}
    s = run(cfg, make_files=False)
    assert s["position_error_km"] < 100
    assert s["position_error_km"] < 3 * s["sigma_position_km"]
    assert s["info"]["rms_normalized_residual"] < 3

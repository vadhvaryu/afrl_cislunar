"""Tests for the results layer (results.py, uncertainty.py, sweep.py). No Orekit needed.
Run from the folder that holds config.yaml and data/:   python -m pytest -q"""

import copy
import csv
import os
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from od_stack.contracts import StateEstimate
from od_stack.frames import EarthMoonSystem
from od_stack.propagators import CR3BPPropagator
from od_stack.results import ARCSEC, report, residuals
from od_stack.run import load_config
from od_stack.sources import dense_truth, load_synthetic_obs, load_synthetic_truth
from od_stack.sweep import Sweeper, aggregate, resample_observations
from od_stack.uncertainty import confidence_ellipsoid, nees, propagate_covariance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _data():
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    _, x_rot, taus = load_synthetic_truth("data/synthetic_truth.csv", SYS)
    return obs, x_rot, taus


def _cov():
    """A plausible BLS-like covariance: ~15 km and ~0.1 m/s per axis, some correlation."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6)) * np.r_[15e3, 15e3, 15e3, 0.1, 0.1, 0.1][:, None]
    return A @ A.T / 6


def test_residuals_of_truth_equal_the_injected_noise():
    """Truth through residuals() gives back exactly what the generator added to the angles."""
    obs, x_rot, taus = _data()
    res = residuals(obs, dense_truth(x_rot, taus, obs.t, SYS))
    with open("data/synthetic_obs.csv", newline="") as f, open("data/synthetic_truth.csv", newline="") as g:
        o, t = list(csv.DictReader(f)), list(csv.DictReader(g))
    dra = np.array([(float(a["ra_deg"]) - float(b["ra_true_deg"]) + 180) % 360 - 180 for a, b in zip(o, t)]) * 3600
    ddec = np.array([float(a["dec_deg"]) - float(b["dec_true_deg"]) for a, b in zip(o, t)]) * 3600
    assert np.allclose(res["ra_arcsec"], dra, atol=1e-3)
    assert np.allclose(res["dec_arcsec"], ddec, atol=1e-3)


def test_resampled_observations_have_requested_noise():
    obs, x_rot, taus = _data()
    truth_at_obs = dense_truth(x_rot, taus, obs.t, SYS)
    errs = []
    for seed in range(200):
        o = resample_observations("data/synthetic_truth.csv", obs, 10.0, seed)
        r = residuals(o, truth_at_obs)
        errs.append(np.r_[r["ra_arcsec"], r["dec_arcsec"]])
        assert np.allclose(o.sigma / ARCSEC, 10.0)
    assert np.std(errs) == pytest.approx(10.0, rel=0.05)


def test_covariance_propagation_is_identity_at_epoch_and_stays_psd():
    obs, x_rot, taus = _data()
    tr = dense_truth(x_rot, taus, np.array([2 * 86400.0]), SYS)
    x0 = StateEstimate(t=2 * 86400.0, r=tr.r[0], v=tr.v[0], cov=_cov(), method="test")
    t = np.array([0.0, 2 * 86400.0, 8 * 86400.0])
    P = propagate_covariance(x0, t, CR3BPPropagator(SYS))
    assert np.allclose(P[1], x0.cov, rtol=1e-6, atol=1e-9)
    for Pi in P:
        assert np.allclose(Pi, Pi.T)
        assert np.linalg.eigvalsh(Pi).min() > -1e-6 * np.abs(Pi).max()


def test_confidence_ellipsoid_and_nees_calibration():
    a, _ = confidence_ellipsoid(np.eye(3), 0.95)
    assert np.allclose(a, np.sqrt(chi2.ppf(0.95, 3)))
    P = _cov()
    L = np.linalg.cholesky(P)
    e = (L @ np.random.default_rng(0).normal(size=(6, 4000))).T
    n = np.array([nees(x, P) for x in e])
    assert n.mean() == pytest.approx(6, rel=0.05)                       # chi-square(6) mean
    assert np.mean(n <= chi2.ppf(0.95, 6)) == pytest.approx(0.95, abs=0.015)


def test_report_without_truth_writes_files(tmp_path):
    """Real data has no truth file: the results stage must still give residuals and uncertainty."""
    obs, x_rot, taus = _data()
    tr = dense_truth(x_rot, taus, np.array([obs.t[2]]), SYS)
    x0 = StateEstimate(t=obs.t[2], r=tr.r[0], v=tr.v[0], cov=_cov(), method="test")
    prop = CR3BPPropagator(SYS)
    t = np.linspace(obs.t.min(), obs.t.max(), 50)
    s = report(obs, x0, x0, prop.propagate(x0, t), prop.propagate(x0, obs.t), prop, SYS, out_dir=str(tmp_path))
    assert "position_error_km" not in s                   # nothing graded without truth
    assert s["residuals"]["rms_normalized"] > 0
    assert len(s["uncertainty"]["confidence_region_at_epoch"]["position_semi_axes_km"]) == 3
    for f in ("summary.json", "trajectory_comparison.csv", "residuals.csv", "results.png", "residuals.png",
              "uncertainty.png"):
        assert (tmp_path / f).exists(), f


def test_sweeper_runs_without_orekit(tmp_path):
    cfg = load_config("config.yaml")
    cfg["estimation"] = {"method": "none"}
    cfg["propagator"]["method"] = "cr3bp"
    cfg["iod"].update(method="truth_plus_error", position_error_km=0, velocity_error_m_s=0)
    cfg["sweeps"]["check_points"] = 20
    sw = Sweeper(copy.deepcopy(cfg))
    rows = [sw.one(10.0, s, {"position_error_km": 50.0, "velocity_error_m_s": 0.5, "seed": s}) for s in range(3)]
    assert all(r["converged"] for r in rows)                 # 50 km < 1,000 km threshold
    assert rows[0]["position_error_km"] == pytest.approx(50.0, rel=1e-6)
    agg = aggregate(rows, "noise_arcsec")
    assert agg[0]["converged"] == 3 and agg[0]["runs"] == 3
    assert sw.threshold_km(100.0) == pytest.approx(10 * sw.threshold_km(10.0))

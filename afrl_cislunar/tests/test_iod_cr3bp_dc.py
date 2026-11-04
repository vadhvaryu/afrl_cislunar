"""Tests for the CR3BP differential-corrections IOD (Jo, Baker-McEvilly & Canales 2025).

Truth comes only from the TRUTH_STATE constant (obs_generator.py's initial rotating state),
propagated with propagate_cr3bp; no test reads data/synthetic_truth.csv.
The full multi-start takes about a minute per call on the 8-day arc."""

import json
import os
import warnings

import numpy as np
import pytest

from od_stack.cr3bp_tools import make_synthetic_obs, obs_to_rotating
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.iod import IODFailure
from od_stack.iod_cr3bp_dc import CR3BPDifferentialCorrection, solve_dc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
DAY = 86400.0
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py
GEN_TIMES = np.linspace(0, 8 * DAY, 5)                                  # obs_generator.py: 5 obs over 8 days
DENSE_1D = 3 * DAY + np.linspace(0, 1 * DAY, 5)                         # B3 'dense_1d'
DENSE_6H = 4 * DAY + np.linspace(0, 6 * 3600.0, 5)                      # B3 'dense_6h'
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _truth_rot(t_s):
    return propagate_cr3bp(TRUTH_STATE, 0.0, np.atleast_1d(SYS.tau(t_s)), SYS.mu)


def _truth_emi(t_s):
    """True EMI state (r [m], v [m/s]) at t_s, from TRUTH_STATE (no file reading)."""
    return rotating_to_emi(SYS.tau(t_s), _truth_rot(t_s)[0], SYS)


def _clean_obs(times=GEN_TIMES):
    # 1e-3 arcsec noise: practically noise-free, but sigma stays nonzero for the RMS
    return make_synthetic_obs(TRUTH_STATE, times, 1e-3, 0, SYS)


def _err(est):
    r, v = _truth_emi(est.t)
    return np.linalg.norm(est.r - r) / 1e3, np.linalg.norm(est.v - v)


def test_eq22_newton_converges_quadratically_near_truth():
    """The paper's Newton iteration with the Eq. 22 Jacobian, from true ranges and a velocity
    0.02 DU/TU (20 m/s) off, reaches ||F|| < 1e-12 in a few iterations, at the truth."""
    obs = _clean_obs()
    tau, R, u = obs_to_rotating(obs, SYS)
    idx = [0, 2, 4]
    X = _truth_rot(obs.t)
    chi0 = np.r_[np.linalg.norm(X[idx, :3] - R[idx], axis=1), X[2, 3:] + [0.02, 0, 0]]
    res = solve_dc(tau[idx], R[idx], u[idx], chi0=chi0, mu=SYS.mu)
    assert res["converged"] and res["reason"] == "tol" and res["iterations"] <= 6
    assert np.linalg.norm(res["x_j"][:3] - X[2, :3]) * SYS.du_km < 0.1


@pytest.mark.parametrize("offset_km", [
    20_000.0,
    pytest.param(-20_000.0, marks=pytest.mark.xfail(
        strict=False, reason="8-day arc: no start converges 20,000 km short of the true range (see docstring)")),
])
def test_noise_free_single_start_near_truth(offset_km):
    obs = _clean_obs()
    a0 = float(np.linalg.norm(_truth_emi(obs.t[2])[0]) / 1e3) + offset_km     # observer at geocenter
    est = CR3BPDifferentialCorrection(SYS, alpha0_km=[a0], verify=False).estimate(obs)
    dr_km, dv = _err(est)
    assert est.t == obs.t[2] and est.method == "cr3bp_dc"
    assert dr_km < 1.0, dr_km
    assert dv < 0.01, dv
    assert est.info["solutions"][0]["F_norm_du"] < 1e-9


def test_noise_free_multistart_picks_truth():
    obs = _clean_obs()
    est = CR3BPDifferentialCorrection(SYS).estimate(obs)
    dr_km, _ = _err(est)
    assert dr_km < 1.0, dr_km
    sols = est.info["solutions"]
    assert sols[0]["rms_sigma"] == min(s["rms_sigma"] for s in sols if s["rms_sigma"] is not None)
    assert sols[0]["rms_sigma"] < 3.0
    json.dumps(est.info, allow_nan=False)            # strict JSON
    for v in est.info["verification"]:               # other triples agree on noise-free data
        assert v["converged"] and v["position_diff_km"] < 1.0, v


@pytest.mark.parametrize("times, tol_km", [(DENSE_1D, 1.0), (DENSE_6H, 20.0)], ids=["dense_1d", "dense_6h"])
def test_noise_free_dense_arcs(times, tol_km):
    """Short arcs: the paper's own seed converges; an alternative solution exists and is
    rejected by its residuals. 6 h: 1e-3 arcsec already moves the exact fit by ~5 km."""
    est = CR3BPDifferentialCorrection(SYS).estimate(_clean_obs(times))
    assert _err(est)[0] < tol_km
    assert any(s.startswith("paper/") for s in est.info["solutions"][0]["starts"])


def test_runs_without_truth(no_truth):
    from od_stack.sources import load_synthetic_obs
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    est = CR3BPDifferentialCorrection(SYS, truth_path="data/synthetic_truth.csv", orekit_data=None,
                                      alpha0_km=[380_000.0, 400_000.0]).estimate(obs)
    assert est.method == "cr3bp_dc" and np.all(np.isfinite(est.r)) and np.all(np.isfinite(est.v))


def test_no_solution_raises_iod_failure():
    obs = _clean_obs()
    with pytest.raises(IODFailure):     # one start at 1 km: below the admissible range, no fallback
        CR3BPDifferentialCorrection(SYS, alpha0_km=[1.0], trust_region=False).estimate(obs)


def _orekit_data_available():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("OREKIT_DATA"))


def test_noisy_synthetic_obs_lands_within_bls_basin():
    from od_stack.sources import load_synthetic_obs
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    est = CR3BPDifferentialCorrection(SYS).estimate(obs)
    dr_km, _ = _err(est)
    assert dr_km < 10_000.0, dr_km
    if not _orekit_data_available():
        pytest.skip("Orekit or OREKIT_DATA not available: BLS part skipped")
    from od_stack.estimation import OrekitBatchLS
    fit = OrekitBatchLS(SYS).refine(obs, est)
    assert fit.info["rms_normalized_residual"] < 3.0

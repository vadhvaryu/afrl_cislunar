"""Tests for the grid-of-hypotheses IODs (range_grid, range_grid_fd).

Every method call runs under the no_truth fixture; truth comes from TRUTH_STATE via propagate_cr3bp.
Orekit tests are skipped without orekit-jpype and OREKIT_DATA; multiple_shooting tests need neither.

Noise-free means 0.001" noise. Two refiner floors at that level (not method errors):
  * Orekit's AngularRaDec includes light-time and the generator does not, so Orekit fits sit at
    ~175 sigma (0.175") and ~0.8 km from truth: those tests use gate_rms=1000 (1").
  * multiple_shooting accepts a 1e-8 DU (~4 m) continuity defect, ~2 sigma at 0.001", so its fits
    sit at ~2.4 sigma: those tests use gate_rms=10.
"""

import json
import os
import warnings

import numpy as np
import pytest

from od_stack.cr3bp_tools import make_synthetic_obs
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi
from od_stack.iod import IODFailure
from od_stack.iod_range_grid import RangeGridFD, RangeRateGrid, propagate_batch, residual_rms_batch
from od_stack.sources import load_synthetic_obs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py
GEN_TIMES = np.linspace(0, 8 * 86400.0, 5)
warnings.filterwarnings("ignore", message="No observer columns")


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    p = os.environ.get("OREKIT_DATA")
    return bool(p) and os.path.exists(os.path.expanduser(p))


needs_orekit = pytest.mark.skipif(not _orekit_ready(), reason="needs orekit-jpype and OREKIT_DATA")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _truth_at(t):
    tau = SYS.tau(t)
    return rotating_to_emi(tau, propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau]), SYS.mu)[0], SYS)


def _noise_free():
    return make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 1e-3, 1, SYS)


def _check_result(x, obs):
    assert x.t == obs.t[len(obs) // 2]
    assert x.info["n_distinct_solutions"] >= 1 and x.info["chosen_rms"] <= x.info["gate_rms"]
    json.dumps(x.info)                                          # the stack writes info to summary.json


# ----------------------------------------------------------------------------- prefilter (fast, no Orekit)

def test_vectorized_rk4_matches_dop853():
    tau0 = SYS.tau(GEN_TIMES[2])
    x0 = propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau0]), SYS.mu)[0]
    X0 = np.array([x0, x0 + [0.02, 0, 0, 0, 0.05, 0]])
    taus = SYS.tau(GEN_TIMES)
    ref = np.stack([propagate_cr3bp(x, tau0, taus, SYS.mu) for x in X0], axis=1)
    assert np.abs(propagate_batch(X0, tau0, taus, SYS.mu)[..., :3] - ref[..., :3]).max() * SYS.du_km < 0.01  # km


def test_prefilter_scores_truth_at_noise_level_and_reports_truth_cell():
    obs = make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 10.0, 1, SYS)
    r, v = _truth_at(GEN_TIMES[2])
    assert residual_rms_batch(SYS, obs, GEN_TIMES[2], r[None], v[None])[0] < 2      # the truth fits the data
    m = RangeRateGrid(SYS, refine_with="multiple_shooting")
    pf = m.prefilter(obs)
    assert len(pf["score"]) == 41 * 21 + 41 and pf["runtime_s"] < 10
    u = obs.los[2]
    i = int(np.argmin(abs(m.rho_m - np.linalg.norm(r)))) * 21 + int(np.argmin(abs(m.rho_dot_m_s - v @ u)))
    rank = int(np.flatnonzero(pf["order"] == i)[0])
    top5 = m.select(pf)
    print(f"truth cell: rho {pf['rho_m'][i] / 1e3:.0f} km, rho_dot {pf['rho_dot_m_s'][i] / 1e3:.2f} km/s, "
          f"prefilter rank {rank}, among the 5 refined: {i in top5}; its velocity error "
          f"{np.linalg.norm(pf['v'][i] - v):.0f} m/s (transverse part from the Lagrange rho_hat')")
    assert len(top5) == 5
    assert min(np.linalg.norm(pf["r"][a] - pf["r"][b]) for a in top5 for b in top5 if a != b) > 20_000e3


def test_input_checks():
    obs = make_synthetic_obs(TRUTH_STATE, GEN_TIMES[:2], 10.0, 1, SYS)
    with pytest.raises(IODFailure):
        RangeRateGrid(SYS, refine_with="multiple_shooting").estimate(obs)
    with pytest.raises(ValueError):
        RangeRateGrid(SYS, refine_with="nonsense")


# ----------------------------------------------------------------------------- multiple_shooting (no Orekit)

def test_range_grid_noise_free_multiple_shooting(no_truth):
    obs = _noise_free()
    x = RangeRateGrid(SYS, refine_with="multiple_shooting", gate_rms=10, truth_path="data/synthetic_truth.csv",
                      orekit_data=None).estimate(obs)
    r, v = _truth_at(x.t)
    _check_result(x, obs)
    assert np.linalg.norm(x.r - r) < 1e3 and np.linalg.norm(x.v - v) < 0.1
    print(f"range_grid/ms noise-free: {np.linalg.norm(x.r - r):.1f} m, prefilter {x.info['prefilter_runtime_s']} s, "
          f"total {x.info['runtime_s']} s, passing {x.info['n_passed_gate']}/5")


def test_range_grid_fd_noise_free_multiple_shooting(no_truth):
    obs = _noise_free()
    x = RangeGridFD(SYS, refine_with="multiple_shooting", gate_rms=10, prefilter_top=2).estimate(obs)
    r, _ = _truth_at(x.t)
    _check_result(x, obs)
    assert np.linalg.norm(x.r - r) < 1e3


def test_gate_rejects_everything_raises(no_truth):
    obs = make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 10.0, 1, SYS)
    with pytest.raises(IODFailure, match="residual gate"):
        RangeRateGrid(SYS, refine_with="multiple_shooting", top_k=1, gate_rms=1e-6).estimate(obs)


# ----------------------------------------------------------------------------- Orekit BLS

@needs_orekit
def test_range_grid_bls_on_synthetic_obs_csv(no_truth):
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    x = RangeRateGrid(SYS, truth_path="data/synthetic_truth.csv").estimate(obs)
    r, v = _truth_at(x.t)
    _check_result(x, obs)
    err_km = np.linalg.norm(x.r - r) / 1e3
    assert err_km < 10_000
    assert err_km < 50 and x.info["chosen_rms"] < 1        # it lands on the BLS solution (measured 15 km, 0.495 sigma)
    print(f"range_grid/bls 10\": {err_km:.1f} km, {np.linalg.norm(x.v - v):.3f} m/s, distinct "
          f"{x.info['n_distinct_solutions']}, prefilter {x.info['prefilter_runtime_s']} s, total {x.info['runtime_s']} s")


@needs_orekit
def test_range_grid_bls_noise_free(no_truth):
    obs = _noise_free()
    x = RangeRateGrid(SYS, gate_rms=1000).estimate(obs)
    r, _ = _truth_at(x.t)
    _check_result(x, obs)
    assert np.linalg.norm(x.r - r) < 1e3                   # measured 0.77 km (light-time floor)


@needs_orekit
def test_range_grid_fd_bls_noise_free(no_truth):
    obs = _noise_free()
    x = RangeGridFD(SYS, gate_rms=1000, prefilter_top=3).estimate(obs)
    r, _ = _truth_at(x.t)
    _check_result(x, obs)
    assert np.linalg.norm(x.r - r) < 1e3

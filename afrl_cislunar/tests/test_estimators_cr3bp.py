"""Estimators without Orekit BLS (convergence study A7): noise-free recovery from 1,000 km."""

import os
import warnings

import numpy as np
import pytest

from od_stack.contracts import StateEstimate
from od_stack.cr3bp_tools import make_synthetic_obs
from od_stack.estimation import ESTIMATORS
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.dynamics import propagate_cr3bp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])
T = np.linspace(0, 8 * 86400.0, 5)
warnings.filterwarnings("ignore", message="No observer columns")


def _case(d_km=1000.0, w=10.0, noise=1e-3, seed=0):
    """Near noise-free observations (sigma 0.001") and a starting guess d km / w m/s off at day 4."""
    obs = make_synthetic_obs(TRUTH_STATE, T, noise, seed, SYS)
    tau = SYS.tau(T[2])
    r, v = rotating_to_emi(tau, propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau]), SYS.mu)[0], SYS)
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=3), rng.normal(size=3)
    x0 = StateEstimate(t=T[2], r=r + d_km * 1e3 * a / np.linalg.norm(a), v=v + w * b / np.linalg.norm(b),
                       method="test", info={"position_error_km_set": d_km, "velocity_error_m_s_set": w})
    return obs, x0, r, v


def test_multiple_shooting_recovers_the_truth():
    obs, x0, r, v = _case()
    fit = ESTIMATORS["multiple_shooting"](SYS).refine(obs, x0)
    assert np.linalg.norm(fit.r - r) / 1e3 < 0.1
    assert np.linalg.norm(fit.v - v) < 1e-3                           # m/s
    assert fit.cov is not None and np.all(np.linalg.eigvalsh(fit.cov) > 0)
    assert fit.info["rms_normalized_residual"] < 3
    e = np.r_[fit.r - r, fit.v - v]
    assert e @ np.linalg.solve(fit.cov, e) < 30                        # within ~5 sigma in 6-D


def test_ukf_recovers_the_truth_to_its_linearization_floor():
    """A single UKF + RTS pass from 1,000 km cannot reach metres: the unscented update with a
    1,000 km prior leaves a ~6 km nonlinearity floor (measured: 5.8-5.9 km at 0.001", NEES 10.7),
    and below ~3" it is overconfident (NEES 13-24). At 10" it is consistent. We test that
    behaviour rather than iterate the filter, which would turn it into a batch method."""
    obs, x0, r, v = _case()
    fit = ESTIMATORS["ukf"](SYS).refine(obs, x0)
    assert np.linalg.norm(fit.r - r) / 1e3 < 10
    assert fit.cov is not None and np.all(np.linalg.eigvalsh(fit.cov) > 0)
    nees = []
    for seed in range(5):
        obs, x0, r, v = _case(noise=10.0, seed=seed)
        f = ESTIMATORS["ukf"](SYS).refine(obs, x0)
        e = np.r_[f.r - r, f.v - v]
        nees.append(e @ np.linalg.solve(f.cov, e))
    assert np.mean(nees) < 12.6                                          # chi2(6) 95% bound on a mean of 5


def test_multiple_shooting_covariance_matches_linear_theory_at_10_arcsec():
    """With 10" noise its 1-sigma at day 4 is the ~26 km Orekit BLS reports on the same geometry."""
    obs, x0, r, v = _case(noise=10.0, seed=1)
    fit = ESTIMATORS["multiple_shooting"](SYS).refine(obs, x0)
    assert 20 < np.sqrt(np.trace(fit.cov[:3, :3])) / 1e3 < 32


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    p = os.environ.get("OREKIT_DATA")
    return bool(p) and os.path.exists(os.path.expanduser(p))


@pytest.mark.skipif(not _orekit_ready(), reason="needs orekit-jpype and OREKIT_DATA")
def test_multiple_shooting_then_bls_recovers_the_truth():
    """Within a few km, not metres: Orekit's AngularRaDec includes light-time (~0.5-0.8" here),
    which the generator's geometric angles do not."""
    obs, x0, r, v = _case()
    fit = ESTIMATORS["multiple_shooting_then_bls"](SYS).refine(obs, x0)
    assert np.linalg.norm(fit.r - r) / 1e3 < 5
    assert fit.info["ms_iterations"] > 0 and fit.info["bls_iterations"] > 0


def test_propagation_stops_at_an_impact_instead_of_crawling():
    """A state falling straight into the Moon raises quickly instead of integrating through the singularity."""
    import time
    from od_stack.estimators_cr3bp import _propagate_many
    x0 = np.array([1 - SYS.mu + 5000 / SYS.du_km, 0, 0, -1.0, 0, 0])
    t = time.time()
    with pytest.raises(RuntimeError, match="hits"):
        _propagate_many(x0, 0.0, np.array([0.5]), SYS.mu, stm=True)
    assert time.time() - t < 5

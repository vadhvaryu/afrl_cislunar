"""Catalog-search IOD on the L1 northern halo family (od_stack/halo_family.py, od_stack/iod_halo_catalog.py).

Grading uses truth made here from TRUTH_STATE (obs_generator.py) by propagate_cr3bp; the IOD itself
runs under the no_truth fixture and is never given the truth. Run with -s to see the off-family table."""

import inspect
import json
import os
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from od_stack import halo_family, iod_halo_catalog
from od_stack.contracts import StateEstimate
from od_stack.cr3bp_tools import make_synthetic_obs
from od_stack.dynamics import TOL, cr3bp_eom, propagate_cr3bp
from od_stack.estimation import ESTIMATORS
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.halo_family import load_family
from od_stack.iod import IODFailure
from od_stack.iod_halo_catalog import HaloCatalogIOD
from od_stack.orbit_catalog import ORBITS, correct_periodic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
DAY = 86400.0
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py (grading only)
GEN_TIMES = np.linspace(0, 8 * DAY, 5)                                  # obs_generator.py
ARCS = {"generator": GEN_TIMES,
        "dense_1d": 3 * DAY + np.linspace(0, DAY, 5),                   # B3: 5 obs over 1 day from day 3
        "dense_6h": 4 * DAY + np.linspace(0, 6 * 3600.0, 5)}            # B3: 5 obs over 6 h from day 4
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def fam():
    return load_family(SYS)


@pytest.fixture(scope="module")
def iod(fam):
    return HaloCatalogIOD(SYS, family=fam)


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    p = os.environ.get("OREKIT_DATA")
    return bool(p) and os.path.exists(os.path.expanduser(p))


def _truth_at(x0_rot, t):
    tau = float(SYS.tau(t))
    return rotating_to_emi(tau, propagate_cr3bp(np.asarray(x0_rot, float), 0.0, np.array([tau]), SYS.mu)[0], SYS)


def _errors(est: StateEstimate, x0_rot):
    r, v = _truth_at(x0_rot, est.t)
    return np.linalg.norm(est.r - r) / 1e3, np.linalg.norm(est.v - v)


# ----------------------------------------------------------------------------- Part 1: the family

def test_family_is_not_seeded_from_the_truth():
    """The answer key must not appear in the generator or the method (it may appear in tests only)."""
    for mod in (halo_family, iod_halo_catalog):
        src = inspect.getsource(mod)
        assert "0.872763" not in src and "0.190505" not in src and "L1_halo_truth" not in src
        assert "synthetic_truth" not in src and "load_synthetic_truth" not in src


def test_every_member_closes_after_one_period(fam):
    assert 90 <= fam.n_members <= 150 and fam.n_phases == 1000
    assert fam.x0[0, 2] <= 0.02 and fam.x0[-1, 2] >= 0.22
    assert np.all(np.diff(fam.x0[:, 2]) > 0)                           # ordered by z0
    for x0, T in zip(fam.x0, fam.period):
        xT = solve_ivp(cr3bp_eom, [0, T], x0, args=(SYS.mu,), **TOL).y[:, -1]
        assert np.abs(xT - x0).max() < 1e-7
    assert np.allclose(fam.states[:, 0], fam.x0)                       # phase 0 is the initial state
    assert np.all(np.isfinite(fam.stability)) and np.all(fam.stability >= 1 - 1e-9)
    assert np.all(fam.perilune_km > 1_000)


def test_family_passes_through_the_generator_orbit(fam):
    """Checks the generator (our continuation reaches the truth's family); not used by the IOD."""
    m = int(np.argmin(np.abs(fam.x0[:, 2] - TRUTH_STATE[2])))
    assert abs(fam.x0[m, 0] - TRUTH_STATE[0]) < 1e-3
    assert fam.period[m] == pytest.approx(2.22, abs=0.02)


@pytest.mark.parametrize("m", [10, 45, 57, 75])
def test_interpolation_between_members_is_close_to_the_family(fam, m):
    """Halfway between members, the interpolated orbit is within 50 km of the corrected member there."""
    s = m + 0.5
    x, T = correct_periodic(fam.initial_state_at(s), SYS.mu, planar=False)
    ph = np.linspace(0, 1, 40, endpoint=False)
    exact = propagate_cr3bp(x, 0.0, ph * T, SYS.mu)
    assert np.abs(exact[:, :3] - fam.state(s, ph)[:, :3]).max() * SYS.du_km < 50
    assert abs(T - fam.period_at(s)) < 2e-3


# ----------------------------------------------------------------------------- Part 2: the IOD

@pytest.mark.parametrize("m, p0, arc", [(25, 0, "generator"), (90, 400, "generator"), (40, 700, "dense_6h")])
def test_noise_free_member_is_recovered(no_truth, iod, fam, m, p0, arc):
    """A member that is not the truth (z0 far from 0.1905), started at a stored phase."""
    assert abs(fam.x0[m, 2] - TRUTH_STATE[2]) > 0.02
    x0 = fam.states[m, p0]
    obs = make_synthetic_obs(x0, ARCS[arc], 1e-3, 0, SYS)
    est = iod.estimate(obs)
    dr, dv = _errors(est, x0)
    assert dr < 100 and dv < 1.0
    assert est.info["member"] == pytest.approx(m, abs=0.05)


def _check_contract(est, obs):
    assert est.method == "halo_catalog"
    assert est.t == obs.t[len(obs) // 2]
    assert est.r.shape == (3,) and est.v.shape == (3,)
    json.dumps(est.info)
    assert est.info["prior"] == "L1 northern halo family"
    assert len(est.info["candidates"]) >= 1


@pytest.fixture(scope="module")
def current():
    from od_stack.sources import load_synthetic_obs
    os.chdir(ROOT)
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    est = HaloCatalogIOD(SYS, orekit_data=None, truth_path="data/synthetic_truth.csv").estimate(obs)
    return obs, est


def test_runs_under_no_truth_and_ignores_truth_path(no_truth):
    from od_stack.sources import load_synthetic_obs
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    est = HaloCatalogIOD(SYS, orekit_data=None, truth_path="data/synthetic_truth.csv").estimate(obs)
    _check_contract(est, obs)


def test_current_data_within_bls_basin(current):
    obs, est = current
    _check_contract(est, obs)
    dr, dv = _errors(est, TRUTH_STATE)
    assert dr < 10_000 and dv < 100
    assert est.info["rms_normalized"] < 3


def test_multiple_shooting_converges_from_it(current):
    obs, est = current
    fit = ESTIMATORS["multiple_shooting"](SYS).refine(obs, est)
    assert fit.info["rms_normalized_residual"] < 3
    assert _errors(fit, TRUTH_STATE)[0] < 200


@pytest.mark.skipif(not _orekit_ready(), reason="needs orekit-jpype and OREKIT_DATA")
def test_bls_converges_from_it(current):
    from od_stack.estimation import OrekitBatchLS
    obs, est = current
    fit = OrekitBatchLS(SYS).refine(obs, est)
    assert fit.info["rms_normalized_residual"] < 3
    assert _errors(fit, TRUTH_STATE)[0] < 200


@pytest.mark.parametrize("arc", ["dense_1d", "dense_6h"])
def test_other_arcs(no_truth, iod, arc):
    obs = make_synthetic_obs(TRUTH_STATE, ARCS[arc], 10.0, 1, SYS)
    est = iod.estimate(obs)
    _check_contract(est, obs)
    assert _errors(est, TRUTH_STATE)[0] < 10_000


def test_far_from_the_family_raises(no_truth, iod):
    """A DRO is nothing like an L1 halo: the best halo fit is degrees off, so it refuses."""
    x0, _ = correct_periodic(ORBITS["DRO"][0], SYS.mu, planar=True)
    with pytest.raises(IODFailure):
        iod.estimate(make_synthetic_obs(x0, GEN_TIMES, 10.0, 1, SYS))


def test_off_family_robustness(no_truth, iod):
    """Truth perturbed off the family (random directions, fixed seed): report the catalog guess's
    error and whether BLS / multiple shooting converge from it. Loose asserts."""
    rng = np.random.default_rng(42)
    bls = None
    if _orekit_ready():
        from od_stack.estimation import OrekitBatchLS
        bls = OrekitBatchLS(SYS)
    ms = ESTIMATORS["multiple_shooting"](SYS)
    rows = []
    for dr_km, dv_ms in ((100.0, 1.0), (1000.0, 10.0)):
        for trial in range(3):
            u, w = rng.normal(size=3), rng.normal(size=3)
            x0 = TRUTH_STATE.copy()
            x0[:3] += u / np.linalg.norm(u) * dr_km / SYS.du_km
            x0[3:] += w / np.linalg.norm(w) * dv_ms / SYS.vu_ms
            for noise in (1e-3, 10.0):
                obs = make_synthetic_obs(x0, GEN_TIMES, noise, trial + 1, SYS)
                est = iod.estimate(obs)
                e_r, e_v = _errors(est, x0)
                assert e_r < 10_000 and e_v < 100                     # inside the measured BLS basin
                row = [dr_km, dv_ms, trial, noise, e_r, e_v, est.info["rms_arcsec"]]
                for name, estimator in (("bls", bls), ("ms", ms)):
                    if estimator is None:
                        row.append(None)
                        continue
                    try:
                        fit = estimator.refine(obs, est)
                        row.append(_errors(fit, x0)[0])
                    except Exception:
                        row.append(np.inf)
                rows.append(row)
    print("\n  perturb      trial noise\"   IOD err km  m/s   RMS\"    BLS err km   MS err km")
    for r in rows:
        f = lambda x: "   skipped" if x is None else ("    FAILED" if not np.isfinite(x) else f"{x:10.1f}")
        print(f"  {r[0]:5.0f}km/{r[1]:3.0f}m/s {r[2]:4d} {r[3]:6.3g} {r[4]:9.0f} {r[5]:6.1f} {r[6]:7.1f} {f(r[7])}  {f(r[8])}")
    # BLS converges from every catalog guess (when Orekit is available); multiple shooting from
    # every 10" case (its continuity gate is tight at 0.001" noise, so those are not asserted).
    if bls is not None:
        assert all(r[7] < 200 for r in rows)
    assert all(r[8] < 200 for r in rows if r[3] == 10.0)

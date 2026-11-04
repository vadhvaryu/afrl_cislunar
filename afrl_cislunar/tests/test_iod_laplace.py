"""Tests for od_stack/iod_laplace.py: perturbed_laplace, laplace_moon, gooding_moon.

Truth comes from TRUTH_STATE (obs_generator.py) propagated with dynamics.propagate_cr3bp, never
from data/synthetic_truth.csv. Methods run under the no_truth fixture where they see real data.
Run with -s to see the report tables (test_report_*).
"""

import json
import os
import time
import warnings

import numpy as np
import pytest

from od_stack.contracts import Observations, StateEstimate
from od_stack.cr3bp_tools import make_synthetic_obs
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.iod import IODFailure
from od_stack.iod_laplace import (GoodingMoon, LaplaceMoon, PerturbedLaplace, earth_moon_accel,
                                  fit_residuals, moon_state)
from od_stack.sources import load_synthetic_obs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py
DAY = 86400.0
ARCS = {
    "dense_6h": 4 * DAY + np.linspace(-3, 3, 5) * 3600,     # 5 obs over 6 h centred on day 4
    "dense_1d": 3 * DAY + np.linspace(0, 1, 5) * DAY,       # 5 obs over 1 day from day 3
    "spacing_2d": np.linspace(0, 8, 5) * DAY,               # obs_generator.py: days 0, 2, 4, 6, 8
}
warnings.filterwarnings("ignore", message="No observer columns")


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    path = os.environ.get("OREKIT_DATA")
    return bool(path) and os.path.exists(os.path.expanduser(path))


needs_orekit = pytest.mark.skipif(not _orekit_ready(), reason="needs orekit-jpype and OREKIT_DATA")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def truth_emi(t_s):
    """True EMI state at t_s from TRUTH_STATE (no file reading)."""
    tau = SYS.tau(t_s)
    x = propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau]), SYS.mu)[0]
    return rotating_to_emi(tau, x, SYS)


def errors(x: StateEstimate):
    r, v = truth_emi(x.t)
    return np.linalg.norm(x.r - r) / 1e3, np.linalg.norm(x.v - v)


# ----------------------------------------------------------------------------- building blocks

def test_earth_moon_accel_is_the_cr3bp_in_emi():
    """a(r) with the indirect term equals the CR3BP acceleration mapped to EMI (central differences
    of the true EMI trajectory), so g(rho) uses the same dynamics as the data."""
    t, h = 4 * DAY, 60.0
    r_m, h_m = truth_emi(t - h)[0], truth_emi(t + h)[0]
    r0 = truth_emi(t)[0]
    a_fd = (h_m - 2 * r0 + r_m) / h**2
    a = earth_moon_accel(r0, moon_state(t, SYS)[0], SYS)
    assert np.linalg.norm(a - a_fd) / np.linalg.norm(a) < 1e-6
    # Moon on the CR3BP circle: matches the rotating frame's (1 - mu, 0, 0)
    r, v, acc = moon_state(t, SYS)
    rr, vv = rotating_to_emi(SYS.tau(t), np.array([1 - SYS.mu, 0, 0, 0, 0, 0]), SYS)
    assert np.allclose(r, rr, atol=1e-3) and np.allclose(v, vv, atol=1e-9)
    assert np.allclose(acc, -r * (SYS.gm_earth + SYS.gm_moon) / SYS.du_m**3, rtol=1e-12)


def test_g_vanishes_at_the_true_range_with_exact_derivatives():
    """With the exact line of sight and its derivatives (from the true state), the true range is
    a root of g and step 4 returns the true velocity: the algebra of the method is right."""
    t = 4 * DAY
    r, v = truth_emi(t)
    a = earth_moon_accel(r, moon_state(t, SYS)[0], SYS)
    rho = np.linalg.norm(r)
    u = r / rho
    rho_d = u @ v
    du = (v - rho_d * u) / rho
    rho_dd = u @ a + rho * (du @ du)
    d2u = (a - rho_dd * u - 2 * rho_d * du) / rho
    z = np.zeros(3)
    geo = ([0, 1, 2], t, u, du, d2u, z, z, z, moon_state(t, SYS)[0])
    n = np.cross(u, du)
    g = rho * (n @ d2u) - n @ a
    assert abs(g) < 1e-9 * abs(n @ a) + 1e-30
    r_est, v_est = PerturbedLaplace(SYS).state_for_range(rho, geo)
    assert np.allclose(r_est, r, atol=1e-3) and np.allclose(v_est, v, atol=1e-9)


# ----------------------------------------------------------------------------- perturbed_laplace

@pytest.mark.parametrize("indices", [(0, 2, 4), "all"])
def test_perturbed_laplace_dense_6h_noise_free(indices, no_truth):
    """Required: on 5 noise-free obs over 6 h, one root is within 1% of the true range at the
    middle time, and the chosen solution is within 10,000 km."""
    obs = make_synthetic_obs(TRUTH_STATE, ARCS["dense_6h"], 1e-3, 0, SYS)
    x = PerturbedLaplace(SYS, obs_indices=indices).estimate(obs)
    assert x.t == obs.t[2] and x.method == "perturbed_laplace"
    true_range = np.linalg.norm(truth_emi(x.t)[0]) / 1e3
    roots = [k["rho_km"] for k in x.info["roots"]]
    assert min(abs(np.array(roots) - true_range)) < 0.01 * true_range
    dr, _ = errors(x)
    assert dr < 10_000
    if indices == "all":                     # the degree-4 fit is much better
        assert dr < 100
    json.dumps(x.info, allow_nan=False)


def test_perturbed_laplace_dense_1d_all_obs_noise_free(no_truth):
    obs = make_synthetic_obs(TRUTH_STATE, ARCS["dense_1d"], 0.0, 0, SYS)
    dr, dv = errors(PerturbedLaplace(SYS, obs_indices="all").estimate(obs))
    assert dr < 500 and dv < 5


def test_perturbed_laplace_fails_cleanly_on_two_day_spacing(no_truth):
    """On the 2-day spacing the Lagrange fit's rho_hat'' is wrong in sign, so g has no root."""
    obs = make_synthetic_obs(TRUTH_STATE, ARCS["spacing_2d"], 0.0, 0, SYS)
    with pytest.raises(IODFailure):
        PerturbedLaplace(SYS).estimate(obs)


def test_bad_indices_raise_iod_failure():
    obs = make_synthetic_obs(TRUTH_STATE, ARCS["dense_6h"][:3], 0.0, 0, SYS)
    with pytest.raises(IODFailure):
        PerturbedLaplace(SYS).estimate(obs)                     # (0, 2, 4) on 3 observations
    PerturbedLaplace(SYS, obs_indices=(0, 1, 2)).estimate(obs)


# ----------------------------------------------------------------------------- Moon-centered Orekit

def _lunar_two_body_obs(a_km, e, dt_s):
    """Three observations from Earth's center of a pure Keplerian orbit about the CR3BP Moon."""
    from od_stack.orekit_setup import start_orekit
    start_orekit()
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.time import AbsoluteDate

    frame, t0 = FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH
    prop = KeplerianPropagator(KeplerianOrbit(a_km * 1e3, e, 0.5, 0.3, 0.4, 0.0, PositionAngleType.TRUE,
                                              frame, t0, SYS.gm_moon))
    t = DAY + np.arange(3) * dt_s
    pvs = [prop.propagate(t0.shiftedBy(float(ti))).getPVCoordinates(frame) for ti in t]
    r = np.array([[p.getPosition().getX(), p.getPosition().getY(), p.getPosition().getZ()] for p in pvs])
    v = np.array([[p.getVelocity().getX(), p.getVelocity().getY(), p.getVelocity().getZ()] for p in pvs])
    r += np.array([moon_state(ti, SYS)[0] for ti in t])
    v += np.array([moon_state(ti, SYS)[1] for ti in t])
    los = r / np.linalg.norm(r, axis=1)[:, None]
    obs = Observations(t=t, los=los, observer=np.zeros((3, 3)), sigma=np.zeros(3),
                       ra=np.arctan2(los[:, 1], los[:, 0]), dec=np.arcsin(los[:, 2]))
    return obs, r, v


@needs_orekit
def test_gooding_moon_wiring_on_lunar_two_body_orbit():
    """Gooding about the Moon recovers a pure two-body lunar orbit (given range guesses near the
    truth: with three lines of sight there is a second, mirror solution) -> plumbing is right."""
    obs, r, v = _lunar_two_body_obs(60_000, 0.3, 3600.0)
    rho = np.linalg.norm(r, axis=1) / 1e3
    x = GoodingMoon(SYS, obs_indices=[0, 1, 2], range_guesses_km=[(rho[0] * 1.01, rho[2] * 0.99)]).estimate(obs)
    assert x.t == obs.t[1]
    assert np.linalg.norm(x.r - r[1]) < 1.0 and np.linalg.norm(x.v - v[1]) < 1e-3


@needs_orekit
def test_laplace_moon_wiring_and_unit_rescaling():
    """Orekit's Laplace about the Moon: (a) in plain SI units it always returns null (its
    |det| < 1e-14 threshold), (b) in CR3BP units its own root recovers an orbit farther from the
    Moon than Earth is, (c) for a close lunar orbit its own (largest) root is spurious and the
    correct root is among the replica polynomial's roots."""
    obs, r, v = _lunar_two_body_obs(600_000, 0.1, 3600.0)
    with pytest.raises(IODFailure):
        LaplaceMoon(SYS, obs_indices=[0, 1, 2], time_unit_s=1.0, length_unit_m=1.0, all_roots=False).estimate(obs)
    x = LaplaceMoon(SYS, obs_indices=[0, 1, 2], all_roots=False).estimate(obs)
    assert x.info["chosen_attempt"] == "orekit"
    assert np.linalg.norm(x.r - r[1]) < 10e3 and np.linalg.norm(x.v - v[1]) < 0.1   # 3-point Lagrange error

    obs, r, v = _lunar_two_body_obs(60_000, 0.3, 3600.0)
    x = LaplaceMoon(SYS, obs_indices=[0, 1, 2]).estimate(obs)
    true_moon_range = np.linalg.norm(r[1] - moon_state(obs.t[1], SYS)[0]) / 1e3
    ranges = {a["attempt"]: a["range_from_moon_km"] for a in x.info["attempts"] if "range_from_moon_km" in a}
    assert abs(ranges["orekit"] - true_moon_range) > 100_000                # Orekit's pick: spurious
    assert min(abs(np.array(list(ranges.values())) - true_moon_range)) < 0.01 * true_moon_range


# ----------------------------------------------------------------------------- honesty and reports

def _methods():
    out = [PerturbedLaplace(SYS), PerturbedLaplace(SYS, obs_indices="all")]
    if _orekit_ready():
        out += [LaplaceMoon(SYS), GoodingMoon(SYS)]
    return out


def test_all_methods_run_without_truth(no_truth):
    """Every method runs on data/synthetic_obs.csv under no_truth, or raises IODFailure cleanly."""
    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    for m in _methods():
        try:
            x = m.estimate(obs)
        except IODFailure:
            continue
        assert x.method == m.name and x.r.shape == (3,) and x.v.shape == (3,)
        json.dumps(x.info, allow_nan=False)


def _report(rows):
    print()
    for row in rows:
        print("   " + row)


def test_report_all_arcs(no_truth):
    """Report only (run with -s): position error per method and arc, noise-free and 10" (seed 1);
    the 10" 2-day case is data/synthetic_obs.csv itself."""
    rows = []
    for arc, times in ARCS.items():
        for noise in (0.0, 10.0):
            obs = (load_synthetic_obs("data/synthetic_obs.csv", SYS) if arc == "spacing_2d" and noise
                   else make_synthetic_obs(TRUTH_STATE, times, noise, 1, SYS))
            for m in _methods():
                label = m.name + ("[all]" if getattr(m, "obs_indices", None) == "all" else "")
                t0 = time.time()
                try:
                    x = m.estimate(obs)
                except IODFailure as e:
                    rows.append(f"{arc:10s} {noise:4.0f}\" {label:24s} FAIL ({str(e)[:70]}) {time.time() - t0:.1f}s")
                    continue
                dr, dv = errors(x)
                extra = ""
                if "roots" in x.info:
                    true_range = np.linalg.norm(truth_emi(x.t)[0]) / 1e3
                    extra = f" roots_km={[round(k['rho_km']) for k in x.info['roots']]} true={true_range:.0f}"
                rows.append(f"{arc:10s} {noise:4.0f}\" {label:24s} {dr:10.1f} km {dv:8.2f} m/s "
                            f"rms={x.info['rms_arcsec']:.3g}\" {time.time() - t0:.1f}s{extra}")
    _report(rows)


@needs_orekit
def test_bls_from_perturbed_laplace_if_within_reach(no_truth):
    """Required 5: if perturbed_laplace lands within 10,000 km on data/synthetic_obs.csv, BLS
    from there converges (RMS < 3). It does not (no root on the 2-day spacing): skip then."""
    from od_stack.estimation import OrekitBatchLS

    obs = load_synthetic_obs("data/synthetic_obs.csv", SYS)
    try:
        x = PerturbedLaplace(SYS).estimate(obs)
    except IODFailure as e:
        pytest.skip(f"perturbed_laplace has no solution on synthetic_obs.csv: {e}")
    if errors(x)[0] > 10_000:
        pytest.skip("perturbed_laplace solution is beyond BLS reach")
    fit = OrekitBatchLS(SYS).refine(obs, x)
    assert fit.info["rms_normalized_residual"] < 3


def test_fit_residuals_zero_for_the_truth():
    obs = make_synthetic_obs(TRUTH_STATE, ARCS["spacing_2d"], 0.0, 0, SYS)
    r, v = truth_emi(4 * DAY)
    assert fit_residuals(obs, 4 * DAY, r, v, SYS)["rms_arcsec"] < 1e-4
    assert fit_residuals(obs, 4 * DAY, np.array([1e6, 0, 0]), v, SYS)["rms_arcsec"] == np.inf   # inside Earth

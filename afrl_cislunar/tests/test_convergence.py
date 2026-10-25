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


# ----------------------------------------------------------------------------- A4: features

from od_stack.contracts import StateEstimate                                    # noqa: E402
from od_stack.convergence_features import (FeatureCalculator, jacobi_constant, l1_x,  # noqa: E402
                                           outcome_category)
from od_stack.frames import rotating_to_emi, trajectory_to_rotating            # noqa: E402
from od_stack.propagators import CR3BPPropagator                               # noqa: E402

_FC = {}


def _fc():
    if "fc" not in _FC:
        _FC["fc"] = FeatureCalculator(SYS, TRUTH)
    return _FC["fc"]


def _guess(d_km=0.0, w=0.0, seed=0, **dirs):
    return TruthPlusError(SYS, truth_path=TRUTH, position_error_km=d_km, velocity_error_m_s=w, seed=seed,
                          **dirs).estimate(_obs())


def test_l1_and_jacobi_conventions():
    import cr3bp                                    # top-level module in ROOT
    x = l1_x(SYS.mu)
    assert x == pytest.approx(0.83691, abs=1e-5)    # Earth-Moon L1, mu = 0.01215
    s = np.array([0.9023581414, 0.0936861644, 0.1421356838, 0.1218991572, 0.1379776334, -0.2067777368])
    e = cr3bp.jacobi(s[None, :3], s[None, 3:], SYS.mu)[0]
    assert jacobi_constant(s, SYS.mu) == pytest.approx(-2 * e, rel=1e-14)   # cr3bp.jacobi is the energy


def test_features_of_the_truth_are_zero():
    f = _fc()(_obs(), _guess(), None)
    for k in ("peri_delta_8d_km", "peri_delta_12d_km", "dr_along_los_km", "dr_across_los_km", "dr_R_km",
              "dv_C_m_s", "dr_moon_radial_km", "dv_along_v_m_s"):
        assert abs(f[k]) < 1e-3, k              # < 1 m: truth CSV rounding, periselene refinement
    assert abs(f["delta_jacobi"]) < 1e-8                     # real cases: 0.1 to 1
    assert np.isnan(f["angle_dr_los_deg"]) and np.isnan(f["angle_dv_v_deg"])
    assert not (f["hits_moon"] or f["hits_earth"] or f["leaves_l1_region"])
    assert f["start_residual_rms_sigma"] < 3                     # just the 10" noise
    assert f["final_state_kind"] == "none"


def test_periselene_matches_cr3bp_propagator_on_hourly_grid():
    """The refined periselene is at or just below the hourly minimum from CR3BPPropagator."""
    fc = _fc()
    x = _guess(20000, 200, seed=5)
    f = fc(_obs(), x, None)
    t = np.arange(0, 12 * 86400 + 1, 3600.0)
    rot = trajectory_to_rotating(CR3BPPropagator(SYS).propagate(x, t), SYS)
    d = np.linalg.norm(rot[:, :3] - [1 - SYS.mu, 0, 0], axis=1) * SYS.du_km
    assert f["peri_guess_12d_km"] <= d.min() + 1e-6
    assert f["peri_guess_12d_km"] > d.min() - 50                # hourly samples are close already
    assert abs(f["peri_guess_12d_day"] - t[np.argmin(d)] / 86400) < 1 / 24
    truth = fc.truth_periselene["8d"]
    assert 15_000 < truth["km"] < 25_000 and 0 < truth["day"] < 8


def test_line_of_sight_and_ric_split():
    obs = _obs()
    f = _fc()(obs, _guess(5000, 0, position_direction=obs.los[2]), None)
    # Tolerances of 1 m: the stand-in adds its error to the CSV truth row (10 decimals), the
    # features compare against the re-propagated truth; they differ by about 1 cm.
    assert f["dr_along_los_km"] == pytest.approx(5000, abs=1e-3)
    assert f["dr_across_los_km"] == pytest.approx(0, abs=1e-3)
    tr = _guess()
    f = _fc()(obs, _guess(3000, 7, position_direction=tr.r, velocity_direction=tr.v), None)
    assert f["dr_R_km"] == pytest.approx(3000, abs=1e-3)
    assert f["dv_along_v_m_s"] == pytest.approx(7, abs=1e-6) and f["angle_dv_v_deg"] < 1e-3
    g = _fc()(obs, _guess(4000, 9, seed=11), None)
    assert np.hypot(np.hypot(g["dr_R_km"], g["dr_I_km"]), g["dr_C_km"]) == pytest.approx(4000, abs=1e-3)
    assert np.hypot(g["dr_along_los_km"], g["dr_across_los_km"]) == pytest.approx(4000, abs=1e-3)


def test_impact_and_l1_region_flags():
    fc, obs = _fc(), _obs()
    tau = SYS.tau(4 * 86400.0)
    # 5,000 km from the Moon, falling straight at it -> hits the Moon
    r, v = rotating_to_emi(tau, np.array([1 - SYS.mu + 5000 / SYS.du_km, 0, 0, -1.0, 0, 0]), SYS)
    f = fc(obs, StateEstimate(t=4 * 86400.0, r=r, v=v), None)
    assert f["hits_moon"] and f["peri_guess_12d_km"] < 1800
    # Geostationary-like orbit: far from L1 the whole time, never near the Moon
    r, v = rotating_to_emi(tau, np.array([-SYS.mu + 42164 / SYS.du_km, 0, 0, 0, 3.07 / (SYS.vu_ms / 1e3), 0]), SYS)
    f = fc(obs, StateEstimate(t=4 * 86400.0, r=r, v=v), None)
    assert f["leaves_l1_region"] and not f["hits_moon"] and not f["hits_earth"]


def test_final_state_and_outcome_categories():
    x = _guess(1000, 10)
    last = StateEstimate(t=x.t, r=x.r, v=x.v, info={"state_kind": "last_iterate"})
    f = _fc()(_obs(), x, last)
    assert f["final_state_kind"] == "last_iterate" and f["peri_final_8d_km"] == pytest.approx(f["peri_guess_8d_km"])
    assert outcome_category({"converged": True, "bls_finished": True}) == "converged"
    assert outcome_category({"converged": False, "bls_finished": True}) == "wrong_orbit"
    assert outcome_category({"bls_finished": False, "error": "maximal count (50) exceeded"}) == "iteration_limit"
    assert outcome_category({"bls_finished": False, "error": "something else"}) == "other_error"


# ----------------------------------------------------------------------------- A0: parallel runner

from od_stack import parallel                                                   # noqa: E402


def _dummy_init(tag):
    parallel.WORKER["tag"] = tag


def _dummy_work(job):
    if job["i"] == 3:
        raise ValueError("boom")
    row = {**job, "tag": parallel.WORKER["tag"], "sq": job["i"] ** 2}
    if job["i"] == 5:
        row["late_column"] = "x"                  # forces a header rewrite
    return row, [{**job, "step": s} for s in range(2)]


def test_parallel_runner_streams_resumes_and_survives_errors(tmp_path):
    path, extra = str(tmp_path / "runs.csv"), str(tmp_path / "steps.csv")
    jobs = [{"i": i, "mode": "m"} for i in range(8)]
    kw = dict(key=("mode", "i"), work=_dummy_work, init=_dummy_init, init_args=("w",), workers=2,
              extra_csv_path=extra)
    rows = parallel.run_jobs(jobs[:6], path, **kw)
    assert sorted(int(r["i"]) for r in rows) == list(range(6))
    bad = [r for r in rows if r["i"] == "3"][0]
    assert bad["outcome"] == "other_error" and "boom" in bad["error"]
    good = [r for r in rows if r["i"] == "4"][0]
    assert good["tag"] == "w" and good["sq"] == "16" and "late_column" in good
    assert len(parallel.read_rows(extra)) == 2 * 5             # the failed job wrote no extras

    # An orphan extra row (its job never finished) is dropped on resume, not duplicated.
    parallel.CsvAppender(extra).append([{"i": 7, "mode": "m", "step": 0}])
    rows = parallel.run_jobs(jobs, path, **kw)                  # resume: only jobs 6 and 7 run
    assert sorted(int(r["i"]) for r in rows) == list(range(8))
    steps = parallel.read_rows(extra)
    assert len(steps) == 2 * 7 and sum(r["i"] == "7" for r in steps) == 2
    assert parallel.job_key({"i": 7.0, "mode": "m"}, ("mode", "i")) == parallel.job_key({"i": "7", "mode": "m"}, ("mode", "i"))


# ----------------------------------------------------------------------------- A2/A3 driver

from od_stack import convergence as conv                                       # noqa: E402


def test_grid_sizes_and_directions_match_the_brief():
    assert len(conv.GRID_KM) == 36 and len(conv.GRID_M_S) == 15
    assert len(conv.grid_jobs()) == (36 * 2 + 15) * 50
    u_r, u_v = conv.directions(2)
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(u_r, a / np.linalg.norm(a)) and np.allclose(u_v, b / np.linalg.norm(b))
    assert conv.errors_for("combined", 20_000) == (20_000.0, 200.0)
    assert conv.errors_for("position_only", 20_000) == (20_000.0, 0.0)
    assert conv.errors_for("velocity_only", 150) == (0.0, 150.0)


def test_grid_and_threshold_jobs_without_orekit():
    """With estimation 'none' the final error equals the IOD error, so 'converged' means
    d < 1,000 km exactly, and the bisection must bracket 1,000 km to within 250 km."""
    conv.init_worker(_no_orekit_cfg())
    row = conv.grid_job({"mode": "combined", "size": 500.0, "direction": 4, "obs_seed": 10_000})
    assert row["outcome"] == "converged" and row["position_error_km"] == pytest.approx(500, abs=1e-3)
    assert row["pos_dir_x"] == pytest.approx(conv.directions(4)[0][0])
    assert "peri_guess_8d_km" in row and row["final_state_kind"] == "solution"
    row = conv.grid_job({"mode": "velocity_only", "size": 50.0, "direction": 4, "obs_seed": 10_000})
    assert row["position_error_km_set"] == 0 and row["outcome"] == "converged"
    summary, runs = conv.threshold_job({"mode": "combined", "direction": 1_000, "obs_seed": 10_000})
    assert 750 <= summary["threshold"] < 1_000 <= summary["smallest_failed"]
    assert summary["smallest_failed"] - summary["threshold"] <= 250
    assert summary["n_runs"] == len(runs) and summary["monotonic_in_evaluations"]
    assert [r["bisection_step"] for r in runs] == list(range(len(runs)))


# ----------------------------------------------------------------------------- A5: statistics

from od_stack import convergence_plots as cplots                               # noqa: E402


def test_auc_and_logistic_regression_recover_a_known_effect():
    rng = np.random.default_rng(0)
    n = 600
    x_real, x_noise = rng.normal(size=n), rng.normal(size=n)
    failed = rng.random(n) < 1 / (1 + np.exp(-(2.0 * x_real - 0.5)))
    a = cplots.auc(x_real, failed)
    assert 0.8 < a["auc"] < 0.95 and a["p"] < 1e-10            # failures have larger x_real
    assert abs(cplots.auc(x_noise, failed)["auc"] - 0.5) < 0.07
    assert cplots.auc(-x_real, failed)["auc"] == pytest.approx(1 - a["auc"])
    rows = [{"a": u, "b": v, "failed": bool(f)} for u, v, f in zip(x_real, x_noise, failed)]
    lg = cplots.logistic(rows, ["a", "b"], n_boot=50)
    assert lg["coefficients"]["a"]["excludes_zero"] and lg["coefficients"]["a"]["beta"] > 1.2
    assert not lg["coefficients"]["b"]["excludes_zero"]


def test_non_monotonic_rates_and_thresholds():
    def r(mode, k, d, ok):
        return {"mode": mode, "direction": k, "size": d, "failed": not ok, "outcome": "converged" if ok else "wrong_orbit"}
    rows = ([r("combined", 0, d, d <= 20) for d in (10, 20, 30, 40)]               # clean cliff
            + [r("combined", 1, d, d in (10, 30)) for d in (10, 20, 30, 40)])       # fails at 20, ok at 30
    nm = cplots.non_monotonic(rows)
    assert nm["combined"] == {"count": 1, "of": 2, "directions": [1]}
    assert cplots.largest_converged(rows, "combined") == {0: 20, 1: 30}
    b = cplots.rate_by(rows, "size")
    assert [v["converged"] for v in b] == [2, 1, 1, 0] and all(v["lo"] <= v["rate"] <= v["hi"] for v in b)
    thr = [{"mode": "combined", "threshold": t, "censored": c, "monotonic_in_evaluations": True}
           for t, c in ((1000.0, ""), (2000.0, ""), (100_000.0, ">= 100,000"), (np.nan, "< 500"))]
    s = cplots.threshold_stats(thr)["combined"]
    assert s["at_least_100000"] == 1 and s["below_500"] == 1 and s["min_km"] == 0.0 and s["max_km"] == 100_000


def test_residual_gate():
    g = conv.gated_outcome
    assert g({"bls_finished": True, "converged": True, "rms_normalized_residual": 0.6}) == "converged"
    assert g({"bls_finished": True, "converged": False, "rms_normalized_residual": 1638.1}) == "wrong_orbit_rejected"
    assert g({"bls_finished": True, "converged": False, "rms_normalized_residual": 1.2}) == "wrong_orbit"
    assert g({"bls_finished": True, "converged": True, "rms_normalized_residual": 3.5}) == "good_fit_rejected"
    assert g({"bls_finished": True, "converged": True, "rms_normalized_residual": float("nan")}) == "good_fit_rejected"
    assert g({"bls_finished": True, "converged": True, "rms_normalized_residual": None}) == "converged"  # no BLS
    assert g({"bls_finished": False, "error": "maximal count (50) exceeded"}) == "iteration_limit"


# ----------------------------------------------------------------------------- robust estimator (needs Orekit data)

def _orekit_ready():
    path = load_config(os.path.join(ROOT, "config.yaml")).get("orekit", {}).get("data_path") or os.environ.get("OREKIT_DATA")
    if not (path and os.path.exists(os.path.expanduser(path))):
        return False
    try:
        import orekit_jpype  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _orekit_ready(), reason="needs orekit-jpype and OREKIT_DATA")
def test_residual_gate_option_and_robust_estimator():
    from od_stack.estimation import EstimationFailure, OrekitBatchLS, OrekitBatchLSRobust, subset
    from od_stack.sweep import resample_observations

    obs = resample_observations(TRUTH, _obs(), 10.0, 10_000)
    assert len(subset(obs, [1, 2, 3])) == 3 and subset(obs, [1, 2, 3]).t[0] == obs.t[1]
    easy = _guess(1000, 10)
    with pytest.raises(EstimationFailure) as e:                    # an impossible gate rejects a good fit
        OrekitBatchLS(SYS, gate_rms=0.01).refine(obs, easy)
    assert "residual gate" in str(e.value) and e.value.last_state.info["state_kind"] == "rejected_fit"
    # Direction 2 at 30,000 km: plain BLS hits the iteration limit (convergence study pilot);
    # the robust estimator's short-arc continuation recovers the same solution plain BLS finds from 1,000 km.
    u_r, u_v = conv.directions(2)
    hard = _guess(30_000, 300, position_direction=u_r, velocity_direction=u_v)
    ref = OrekitBatchLS(SYS).refine(obs, easy)
    fit = OrekitBatchLSRobust(SYS).refine(obs, hard)
    assert fit.info["robust_step"] == "continuation" and fit.info["rms_normalized_residual"] < 3
    assert np.linalg.norm(fit.r - ref.r) < 1e3 and fit.cov is not None

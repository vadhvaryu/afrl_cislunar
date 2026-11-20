"""Impulsive burns: truth generation, burn-aware propagation, and batch least squares with burn parameters."""
import numpy as np
import pytest

from od_stack.config import ConfigError, prepare
from od_stack.dynamics import propagate_cr3bp, propagate_with_burns
from od_stack.estimation import cluster_impulses
from od_stack.frames import EarthMoonSystem
from od_stack.run import run
from od_stack.synthetic import generate

DV = [3.0, -2.0, 3.0]
TIMES = [round(0.5 * i, 1) for i in range(17)]


def scenario(tmp_path, iod, estimator, burn=True, backend="scipy", noise=10.0):
    sim = {"observations": {"times_days": TIMES, "noise": {"sigma_arcsec": noise}}}
    if burn:
        sim["truth"] = {"maneuvers": [{"time_days": 3.1, "delta_v_m_s": DV}]}
    return {"schema_version": 1, "simulation": sim,
            "determination": {"iod": {**iod, "epoch_index": 8, "fallback": []}, "estimator": estimator},
            "prediction": {"backend": backend, "end_days": 8, "points": 100},
            "output": {"directory": str(tmp_path / "out"), "plots": False}}


def test_propagation_with_burns_has_exact_partials():
    mu, x0 = 0.0121505, np.array([0.8727638863136087, 0, 0.190505, 0, 0.23598602025575388, 0])
    taus = np.array([-1.0, -0.7, -0.3, 0.0, 0.4, 0.95, 1.3, 2.0])          # no observation exactly at a burn
    p = np.concatenate([x0, [0.002, -0.001, 0.0015], [-0.001, 0.002, 0.0], [0.9, -0.6]])
    F = lambda q: propagate_with_burns(q[:6], 0.0, taus, mu, [(q[12], q[6:9]), (q[13], q[9:12])])
    _, P = propagate_with_burns(p[:6], 0.0, taus, mu, [(p[12], p[6:9]), (p[13], p[9:12])], partials=True)
    fd = np.stack([(F(p + 1e-6 * e) - F(p - 1e-6 * e)) / 2e-6 for e in np.eye(14)], axis=-1)
    assert np.all(np.abs(fd - P).max(axis=(0, 1)) <= 1e-5 * np.abs(P).max(axis=(0, 1)))
    assert np.abs(propagate_with_burns(x0, 0.0, taus, mu) - propagate_cr3bp(x0, 0.0, taus, mu)).max() < 1e-8


def test_truth_velocity_jumps_by_the_burn(tmp_path):
    cfg = prepare(scenario(tmp_path, {"method": "range_search"}, {"method": "none"}))
    obs, truth = generate(cfg["simulation"], EarthMoonSystem())
    t = 3.1 * 86400
    before, after = truth.at(np.array([t - 1e-3])).v[0], truth.at(np.array([t + 1e-3])).v[0]
    np.testing.assert_allclose(after - before, DV, atol=1e-3)
    assert not prepare(scenario(tmp_path, {"method": "range_search"}, {"method": "none"}, burn=False))[
        "simulation"]["truth"]["maneuvers"]


def test_impulses_spread_over_mesh_points_become_one_burn():
    imp = [{"t_s": 3.2 * 86400, "delta_v_m_s": [1.0, 0, 0]}, {"t_s": 3.4 * 86400, "delta_v_m_s": [1.0, 0, 0]},
           {"t_s": 1.5 * 86400, "delta_v_m_s": [0.3, 0, 0]}, {"t_s": 6.0 * 86400, "delta_v_m_s": [0.0, 0.4, 0]}]
    burns = cluster_impulses(imp, min_burn_m_s=0.5)
    assert len(burns) == 1 and burns[0][0] == pytest.approx(3.3 * 86400) and np.allclose(burns[0][1], [2, 0, 0])


def test_least_squares_with_burns_recovers_a_maneuvering_orbit(tmp_path):
    s = run(scenario(tmp_path, {"method": "collocation", "options": {"mode": "impulsive"}}, {"method": "cr3bp_bls"}),
            make_files=False)
    (b,) = s["info"]["maneuvers"]
    assert s["info"]["rms_normalized_residual"] < 3 and s["max_position_error_km"] < 200
    assert np.linalg.norm(np.array(b["delta_v_m_s"]) - DV) < 3 * b["sigma_delta_v_m_s"]
    assert abs(b["t_s"] / 86400 - 3.1) * 86400 < 4 * b["sigma_t_s"]


def test_without_least_squares_the_prediction_follows_collocations_burns(tmp_path):
    s = run(scenario(tmp_path, {"method": "collocation", "options": {"mode": "impulsive"}}, {"method": "none"}),
            make_files=False)
    # ~1,100 km if its burns are dropped; ~500 km with them (collocation uses as little Δv as the noise allows,
    # 3.2 of the true 4.7 m/s here), which is what least squares with burn parameters then fixes
    assert s["info"]["maneuvers"] and s["max_position_error_km"] < 700


def test_a_coasting_fit_cannot_match_a_burn(tmp_path):
    from od_stack.estimation import EstimationFailure
    with pytest.raises(EstimationFailure, match="residual check failed"):
        run(scenario(tmp_path, {"method": "range_search"}, {"method": "cr3bp_bls", "options": {"maneuvers": "none"}}),
            make_files=False)


def test_cr3bp_bls_also_fits_coasting_orbits(tmp_path):
    s = run(scenario(tmp_path, {"method": "range_search"}, {"method": "cr3bp_bls"}, burn=False), make_files=False)
    assert s["info"]["maneuvers"] == [] and s["position_error_km"] < 3 * s["sigma_position_km"] < 100


@pytest.mark.parametrize("mode,estimator,backend,message", [
    ("impulsive", "orekit_bls", "orekit", "use estimation method cr3bp_bls"),
    ("hybrid", "cr3bp_bls", "scipy", "cannot fit"),
    ("impulsive", "cr3bp_bls", "orekit", "set prediction backend scipy"),
    ("impulsive", "none", "orekit", "set prediction backend scipy"),
])
def test_burn_configurations_that_cannot_work_are_rejected(tmp_path, mode, estimator, backend, message):
    c = scenario(tmp_path, {"method": "collocation", "options": {"mode": mode}}, {"method": estimator}, backend=backend)
    with pytest.raises(ConfigError, match=message):
        prepare(c)


@pytest.mark.parametrize("bad", [[{"time_days": 3}], [{"time_days": 3, "delta_v_m_s": [1, 2]}], "3.1", [1]])
def test_bad_maneuvers_rejected(tmp_path, bad):
    c = scenario(tmp_path, {"method": "range_search"}, {"method": "none"})
    c["simulation"]["truth"]["maneuvers"] = bad
    with pytest.raises(ConfigError, match="maneuvers"):
        prepare(c)

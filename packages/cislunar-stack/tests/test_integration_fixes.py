"""The residual check, IOD fallback, configuration checks and run provenance."""
import numpy as np
import pytest

from od_stack.config import ConfigError, prepare
from od_stack.contracts import StateEstimate
from od_stack.estimation import EstimationFailure
from od_stack.frames import EarthMoonSystem
from od_stack.run import run
from od_stack.synthetic import generate


def scenario(tmp_path, **determination):
    return {"schema_version": 1,
            "determination": {"estimator": {"method": "none"}, **determination},
            "prediction": {"backend": "scipy", "end_days": 8, "points": 30},
            "output": {"directory": str(tmp_path / "results"), "plots": False}}


def _orekit():
    pytest.importorskip("orekit_jpype")
    import os
    if not os.environ.get("OREKIT_DATA"):
        pytest.skip("needs OREKIT_DATA")


def test_residual_check_rejects_a_fit_and_keeps_it_for_diagnosis(tmp_path):
    _orekit()
    from od_stack.estimation import OrekitBatchLS

    cfg = prepare(scenario(tmp_path))
    sys_ = EarthMoonSystem()
    obs, truth = generate(cfg["simulation"], sys_)
    t = float(obs.t[2])
    tr = truth.at(np.array([t]))
    start = StateEstimate(t=t, r=tr.r[0], v=tr.v[0], method="truth")
    fit = OrekitBatchLS(sys_).refine(obs, start)                      # default 3 sigma: a correct fit passes
    assert fit.info["rms_normalized_residual"] < 3 and fit.info["residual_gate_sigma"] == 3
    with pytest.raises(EstimationFailure, match="residual check") as e:
        OrekitBatchLS(sys_, residual_gate_sigma=1e-6).refine(obs, start)
    assert e.value.state is not None and np.allclose(e.value.state.r, fit.r)
    assert OrekitBatchLS(sys_, residual_gate_sigma=None).refine(obs, start).info["residual_gate_sigma"] is None


def test_fallback_runs_the_next_method_when_the_first_iod_fails(tmp_path):
    _orekit()
    c = scenario(tmp_path, iod={"method": "gauss", "fallback": ["collocation"]})   # Gauss fails on a halo
    c["determination"]["estimator"] = {"method": "orekit_bls"}
    s = run(c, make_files=False)
    assert s["iod"] == "collocation" and s["position_error_km"] < 100
    assert [a["iod"] for a in s["attempts"]] == ["gauss", "collocation"]
    assert s["attempts"][0]["failed_at"] == "iod" and s["attempts"][1]["ok"]


def test_fallback_after_a_rejected_fit_and_a_clear_error_when_all_fail(tmp_path):
    _orekit()
    c = scenario(tmp_path, iod={"method": "collocation", "fallback": ["gauss"]})
    c["determination"]["estimator"] = {"method": "orekit_bls", "options": {"residual_gate_sigma": 1e-6}}
    with pytest.raises(EstimationFailure, match="no IOD method gave an acceptable orbit") as e:
        run(c, make_files=False)
    assert "collocation (estimation): orekit_bls: residual check failed" in str(e.value)
    assert "gauss (iod)" in str(e.value)


def test_product_configs_use_range_search_then_collocation_by_default(tmp_path):
    cfg = prepare(scenario(tmp_path))
    assert cfg["iod"]["method"] == "range_search" and cfg["iod"]["fallback"] == ["collocation"]
    co = prepare(scenario(tmp_path, iod={"method": "collocation"}))
    assert co["iod"]["fallback"] == []                    # no point falling back to itself
    assert prepare(scenario(tmp_path, iod={"method": "collocation", "fallback": []}))["iod"]["fallback"] == []


@pytest.mark.parametrize("iod", [{"fallback": ["nope"]}, {"fallback": ["truth_plus_error"]},
                                 {"fallback": ["gauss", "gauss"]}, {"fallback": "range_search"}])
def test_bad_fallback_rejected(tmp_path, iod):
    with pytest.raises(ConfigError):
        prepare(scenario(tmp_path, iod={"method": "collocation", **iod}))


@pytest.mark.parametrize("gate", [0, -1, "3", float("nan")])
def test_bad_residual_gate_rejected(tmp_path, gate):
    c = scenario(tmp_path)
    c["determination"]["estimator"] = {"method": "orekit_bls", "options": {"residual_gate_sigma": gate}}
    with pytest.raises(ConfigError):
        prepare(c)


@pytest.mark.parametrize("mode", ["impulsive", "hybrid"])
def test_burn_modes_rejected_with_batch_least_squares(tmp_path, mode):
    c = scenario(tmp_path, iod={"method": "collocation", "options": {"mode": mode}})
    prepare(c)                                            # fine without least squares
    c["determination"]["estimator"] = {"method": "orekit_bls"}
    with pytest.raises(ConfigError, match="assumes the spacecraft coasts"):
        prepare(c)


def test_mesh_too_coarse_for_the_arc_is_raised_and_noted(tmp_path):
    from od_stack.run import size_collocation_mesh
    c = scenario(tmp_path, iod={"method": "collocation", "options": {"nodes": 40}})   # 8-day arc needs 80
    s = run(c, make_files=False)
    assert s["position_error_km"] < 100 and "80 mesh nodes instead of 40" in s["notes"][0]
    # collocation only as a fallback, on a 10-day arc: the fallback gets 100 nodes, not the 80 default
    cfg = prepare({**scenario(tmp_path, iod={"method": "range_search", "fallback": ["collocation"]}),
                   "simulation": {"observations": {"times_days": [0, 2.5, 5, 7.5, 10]}}})
    obs, _ = generate(cfg["simulation"], EarthMoonSystem())
    assert "100 mesh nodes" in size_collocation_mesh(cfg, obs) and cfg["iod"]["collocation"]["nodes"] == 100
    assert size_collocation_mesh(cfg, obs) is None                                  # already enough


def test_mesh_rule_follows_the_arc_length():
    from od_stack.contracts import Observations
    from od_stack.run import collocation_nodes_needed

    def arc(days):
        t = np.linspace(0, days * 86400, 5)
        return Observations(t=t, los=np.tile([1.0, 0, 0], (5, 1)), observer=np.zeros((5, 3)), sigma=np.ones(5),
                            ra=np.zeros(5), dec=np.zeros(5))
    assert collocation_nodes_needed(arc(8)) == 80
    assert collocation_nodes_needed(arc(1)) == 20 and collocation_nodes_needed(arc(0.25)) == 20


def test_any_error_in_a_method_moves_on_to_the_fallback(tmp_path, monkeypatch):
    """Errors that are not IODFailure (an IndexError, a Java exception) must not skip the fallback."""
    import od_stack.iod as iods

    class Broken:
        name, config_block = "gauss", "orekit"
        def __init__(self, *a, **k): pass
        def estimate(self, obs): raise IndexError("list index out of range")
    monkeypatch.setitem(iods.IOD_METHODS, "gauss", Broken)
    s = run(scenario(tmp_path, iod={"method": "gauss", "fallback": ["collocation"]}), make_files=False)
    assert s["iod"] == "collocation" and s["attempts"][0]["error"] == "IndexError: list index out of range"


def test_residual_check_is_on_for_the_product_and_off_for_legacy_configs(tmp_path):
    from od_stack.config import DEFAULTS
    assert prepare(scenario(tmp_path))["estimation"]["residual_gate_sigma"] == 3
    assert DEFAULTS["estimation"]["residual_gate_sigma"] is None
    assert prepare({"iod": {"method": "truth_plus_error"}})["estimation"]["residual_gate_sigma"] is None


def test_legacy_fallback_may_not_repeat_the_primary():
    with pytest.raises(ConfigError, match="repeats iod.method"):
        prepare({"iod": {"method": "range_search", "fallback": ["range_search"]}})


def test_stm_matches_finite_differences():
    from od_stack.dynamics import propagate_cr3bp, propagate_cr3bp_stm
    mu, x0, taus = 0.0121505, np.array([1.17539, 0, 0, 0, -0.494244, 0.01]), np.array([-1.0, 0.3, 0.5, 2.0])
    states, stm = propagate_cr3bp_stm(x0, 0.3, taus, mu)
    assert np.abs(states - propagate_cr3bp(x0, 0.3, taus, mu)).max() < 1e-9
    h = 1e-6
    fd = np.stack([(propagate_cr3bp(x0 + h * e, 0.3, taus, mu) - propagate_cr3bp(x0 - h * e, 0.3, taus, mu)) / (2 * h)
                   for e in np.eye(6)], axis=-1)
    assert np.abs(fd - stm).max() < 1e-6 * np.abs(stm).max()


@pytest.mark.parametrize("orbit", [[0.9873801492146178, 0, 0.008439917431935787, 0, 1.66728339697783, 0],   # NRHO
                                   [1.17539, 0, 0, 0, -0.49424426634269614, 0]])                         # DRO
def test_range_search_polish_finds_orbits_collocation_misses(tmp_path, orbit):
    """Collocation's built-in starts fail on these; the polished range search lands within 100 km."""
    from od_stack.iod import RangeSearch
    cfg = prepare({**scenario(tmp_path), "simulation": {"truth": {"initial_state": {"values": orbit}}}})
    sys_ = EarthMoonSystem()
    obs, truth = generate(cfg["simulation"], sys_)
    x = RangeSearch(sys_).estimate(obs)                     # default grid and polishing
    assert x.info["polished"] and np.linalg.norm(x.r - truth.at(np.array([x.t])).r[0]) < 100e3
    rough = RangeSearch(sys_, refine=False)
    assert np.linalg.norm(rough.estimate(obs).r - truth.at(np.array([x.t])).r[0]) > 1000e3


def test_short_arc_reports_a_covariance_and_a_warning(tmp_path):
    _orekit()
    c = scenario(tmp_path, iod={"method": "range_search"})
    c["simulation"] = {"observations": {"times_days": [3, 3.25, 3.5, 3.75, 4]}}
    c["determination"]["estimator"] = {"method": "orekit_bls"}
    s = run(c, make_files=False)
    assert s["covariance_available"] and any("only 1 days" in w for w in s["warnings"])
    # On this draw two orbits ~11,000 km apart fit the angles and the true one is the runner-up:
    # the chosen fit is far outside its own 3 sigma, so the alternative must be reported.
    assert s["position_error_km"] > 3 * s["sigma_position_km"]
    alternatives = s["info"]["alternatives"]
    assert any("fit the angles about as well" in w for w in s["warnings"]) and alternatives
    cfg = prepare(c)
    _, truth = generate(cfg["simulation"], EarthMoonSystem())
    true_r = truth.at(np.array([s["epoch_days"] * 86400])).r[0]
    assert min(np.linalg.norm(np.array(a["position_m"]) - true_r) for a in alternatives) < 3e3 * s["sigma_position_km"]


def test_summary_records_attempts_and_provenance(tmp_path):
    s = run(scenario(tmp_path), make_files=False)
    assert s["attempts"] == [{"iod": "range_search", "ok": True, "rms_normalized_residual": None}]
    assert s["provenance"]["packages"]["cislunar-stack"] and s["provenance"]["machine"]

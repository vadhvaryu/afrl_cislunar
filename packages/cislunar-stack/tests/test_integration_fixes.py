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


def test_product_configs_fall_back_to_range_search_by_default(tmp_path):
    cfg = prepare(scenario(tmp_path))
    assert cfg["iod"]["method"] == "collocation" and cfg["iod"]["fallback"] == ["range_search"]
    rs = prepare(scenario(tmp_path, iod={"method": "range_search"}))
    assert rs["iod"]["fallback"] == []                    # no point falling back to itself
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


def test_mesh_too_coarse_for_the_arc_fails_before_computation(tmp_path, monkeypatch):
    import od_stack.run as runner
    monkeypatch.setattr(runner, "build_stages", lambda *a: pytest.fail("must fail before building stages"))
    c = scenario(tmp_path, iod={"method": "collocation", "options": {"nodes": 40}})   # 8-day arc needs 80
    with pytest.raises(ConfigError, match="needs at least 80 mesh nodes"):
        run(c, make_files=False)


def test_mesh_rule_follows_the_arc_length():
    from od_stack.contracts import Observations
    from od_stack.run import collocation_nodes_needed

    def arc(days):
        t = np.linspace(0, days * 86400, 5)
        return Observations(t=t, los=np.tile([1.0, 0, 0], (5, 1)), observer=np.zeros((5, 3)), sigma=np.ones(5),
                            ra=np.zeros(5), dec=np.zeros(5))
    assert collocation_nodes_needed(arc(8)) == 80
    assert collocation_nodes_needed(arc(1)) == 20 and collocation_nodes_needed(arc(0.25)) == 20


def test_summary_records_attempts_and_provenance(tmp_path):
    s = run(scenario(tmp_path), make_files=False)
    assert s["attempts"] == [{"iod": "collocation", "ok": True, "rms_normalized_residual": None}]
    assert s["provenance"]["packages"]["cislunar-stack"] and s["provenance"]["machine"]

"""The optional adapter obeys the existing stack's actual contracts and frames."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import collocation_iod
from collocation_iod import CR3BPSystem, SolverOptions
from collocation_iod.adapters import CollocationIOD, observations_from_emi, register_with_od_stack, rotating_state_to_emi


def test_observation_conversion_preserves_absolute_phase_and_observer_origin():
    sys = CR3BPSystem()
    t = np.array([0.8, 1.0, 1.2]) * sys.tu_s
    los = np.tile([1., 0., 0.], (3, 1))
    observer = np.tile([2e7, -3e7, 4e6], (3, 1))
    arc = observations_from_emi(t, los, observer, 2e-5, system=sys)
    np.testing.assert_allclose(arc.t, t / sys.tu_s)
    np.testing.assert_allclose(arc.los[:, 0], np.cos(arc.t))
    np.testing.assert_allclose(arc.los[:, 1], -np.sin(arc.t))
    for i, tau in enumerate(arc.t):
        rotation = np.array([[np.cos(tau), np.sin(tau), 0], [-np.sin(tau), np.cos(tau), 0], [0, 0, 1]])
        expected = rotation @ observer[i]/sys.du_m - np.array([sys.mu, 0, 0])
        np.testing.assert_allclose(arc.observer[i], expected)


def test_radec_covariance_preserves_cos_declination():
    sys = CR3BPSystem()
    sigma = 3e-5
    dec = np.array([0.0, np.pi/6, np.pi/3])
    los = np.column_stack([np.cos(dec), np.zeros(3), np.sin(dec)])
    arc = observations_from_emi(np.arange(3)*sys.tu_s, los, np.zeros((3, 3)), sigma, system=sys)
    np.testing.assert_allclose(arc.covariance[:, 0, 0], sigma**2*np.cos(dec)**2, rtol=1e-12)
    np.testing.assert_allclose(arc.covariance[:, 1, 1], sigma**2, rtol=1e-12)
    np.testing.assert_allclose(arc.covariance[:, 0, 1], 0, atol=1e-20)


def test_state_conversion_matches_existing_stack_at_nonzero_epoch():
    frames = pytest.importorskip("od_stack.frames")
    sys = CR3BPSystem()
    old_sys = frames.EarthMoonSystem(mu=sys.mu, du_km=sys.du_m/1e3, tu_s=sys.tu_s)
    state = np.array([0.9, -0.2, 0.1, 0.04, 0.2, -0.06])
    t = 1.3 * sys.tu_s
    r, v = rotating_state_to_emi(state, t, system=sys)
    old_r, old_v = frames.rotating_to_emi(t/sys.tu_s, state, old_sys)
    np.testing.assert_allclose(r, old_r, atol=1e-7, rtol=1e-14)
    np.testing.assert_allclose(v, old_v, atol=1e-10, rtol=1e-14)


def test_adapter_returns_actual_stateestimate_without_reading_truth(monkeypatch):
    contracts = pytest.importorskip("od_stack.contracts")
    frames = pytest.importorskip("od_stack.frames")
    sys = frames.EarthMoonSystem()
    obs = contracts.Observations(t=np.array([0.4, 0.6, 0.8])*sys.tu_s,
        los=np.tile([1., 0., 0.], (3, 1)), observer=np.zeros((3, 3)),
        sigma=np.zeros(3), ra=np.zeros(3), dec=np.zeros(3))
    rotating = np.array([0.9, 0.1, 0.2, -0.01, 0.3, 0.03])
    calls = []

    def fake_solve(arc, *, system, options):
        calls.append(arc)
        return SimpleNamespace(state_at=lambda tau: rotating, diagnostics={"success": True})

    monkeypatch.setattr(collocation_iod, "solve", fake_solve)
    adapter = CollocationIOD(sys, epoch_index=1, options=SolverOptions(),
        truth_path="this-file-does-not-exist.csv", orekit_data="this-folder-does-not-exist")
    result = adapter.estimate(obs)
    assert isinstance(result, contracts.StateEstimate)
    assert result.t == obs.t[1]
    assert result.method == "collocation"
    assert result.cov is None
    np.testing.assert_allclose(calls[0].t, [0.4, 0.6, 0.8])
    r, v = frames.rotating_to_emi(0.6, rotating, sys)
    np.testing.assert_allclose(result.r, r)
    np.testing.assert_allclose(result.v, v)
    json.dumps(result.info)  # Summary writer must not need NumPy serialization.


def test_registration_is_accepted_by_existing_config(monkeypatch):
    iod = pytest.importorskip("od_stack.iod")
    config = pytest.importorskip("od_stack.config")
    monkeypatch.setattr(iod, "IOD_METHODS", dict(iod.IOD_METHODS))
    register_with_od_stack()
    cfg = config.prepare({"iod": {"method": "collocation", "collocation": {"nodes": 20}}})
    assert cfg["iod"]["method"] == "collocation"
    assert iod.IOD_METHODS["collocation"] is CollocationIOD


def test_adapter_wraps_failure_in_existing_iodfailure(monkeypatch):
    iod = pytest.importorskip("od_stack.iod")
    frames = pytest.importorskip("od_stack.frames")
    def failing_solve(*args, **kwargs):
        raise RuntimeError("optimizer is infeasible")
    monkeypatch.setattr(collocation_iod, "solve", failing_solve)
    obs = SimpleNamespace(t=np.arange(3.), los=np.tile([1., 0., 0.], (3, 1)),
                          observer=np.zeros((3, 3)), sigma=np.zeros(3))
    with pytest.raises(iod.IODFailure, match="infeasible"):
        CollocationIOD(frames.EarthMoonSystem()).estimate(obs)

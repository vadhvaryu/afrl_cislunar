"""Solve against independently integrated truth; truth is never an optimizer input."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import chi2

from collocation_iod import CollocationFailure, CR3BPSystem, ObservationArc, SolverOptions, solve
from collocation_iod.models import tangent_basis
from test_numerics import independent_rhs


def make_arc(*, rates=False, noise=False):
    # A nonzero epoch checks that collocation does not assume the arc begins at 0.
    t = np.linspace(0.3, 1.5, 9)
    x0 = np.array([0.872763, 0, 0.190505, 0, 0.235988, 0])
    sol = solve_ivp(independent_rhs, (t[0], t[-1]), x0, dense_output=True,
                    method="DOP853", rtol=2e-12, atol=2e-13)
    state = sol.sol(t).T
    # A synthetic moving observer gives parallax, without needing a ground model.
    observer = np.column_stack([0.1*np.cos(t), -0.3+0.1*np.sin(t), 0.04*np.sin(2*t)])
    observer_v = np.column_stack([-0.1*np.sin(t), 0.1*np.cos(t), 0.08*np.cos(2*t)])
    rel = state[:, :3] - observer
    distance = np.linalg.norm(rel, axis=1)
    los = rel / distance[:, None]
    rel_v = state[:, 3:] - observer_v
    los_rate = (rel_v - los*np.sum(los*rel_v, axis=1)[:, None]) / distance[:, None]
    sigma = np.full(len(t), np.deg2rad(10/3600) if noise else 0.0)
    if noise:
        rng = np.random.default_rng(71)
        perturbed = los + rng.normal(0, sigma[:, None], los.shape)
        los = perturbed/np.linalg.norm(perturbed, axis=1)[:, None]
    extras = dict(los_rate=los_rate, observer_velocity=observer_v) if rates else {}
    return ObservationArc(t=t, los=los, observer=observer, sigma=sigma, **extras), sol


def test_noise_free_continuous_recovery_without_initial_truth():
    arc, truth = make_arc()
    result = solve(arc, options=SolverOptions(nodes=35, measurement_mode="exact", tolerance=1e-9))
    predicted = np.array([result.state_at(t) for t in arc.t])
    true = truth.sol(arc.t).T
    assert np.max(np.linalg.norm(predicted[:, :3] - true[:, :3], axis=1)) < 3e-3
    assert np.max(np.linalg.norm(predicted[:, 3:] - true[:, 3:], axis=1)) < 1e-2
    assert np.isfinite(result.states).all()
    assert np.max(np.abs(result.controls)) < 0.02


def test_angle_rates_are_supported_and_constrain_velocity():
    arc, truth = make_arc(rates=True)
    result = solve(arc, options=SolverOptions(nodes=35, measurement_mode="exact", tolerance=1e-9))
    t = arc.t[len(arc.t)//2]
    np.testing.assert_allclose(result.state_at(t), truth.sol(t), atol=3e-3, rtol=0)


def test_bounded_measurements_accept_noisy_observations():
    arc, truth = make_arc(noise=True)
    result = solve(arc, options=SolverOptions(nodes=35, measurement_mode="bounded", confidence=0.99))
    x = result.state_at(arc.t[4])
    assert np.linalg.norm(x[:3] - truth.sol(arc.t[4])[:3]) < 0.01
    assert np.isfinite(result.controls).all()


@pytest.mark.parametrize("bad_times", [[0, 1, 1], [0, 2, 1], [0, 1, np.nan]])
def test_invalid_times_rejected(bad_times):
    with pytest.raises(ValueError):
        arc = ObservationArc(t=np.array(bad_times), los=np.tile([1., 0., 0.], (3, 1)),
                             observer=np.zeros((3, 3)), sigma=np.zeros(3))
        solve(arc)


def test_zero_los_rejected():
    with pytest.raises(ValueError):
        arc = ObservationArc(t=np.arange(3.), los=np.zeros((3, 3)),
                             observer=np.zeros((3, 3)), sigma=np.zeros(3))
        solve(arc)


def test_rates_require_observer_velocity():
    arc, _ = make_arc(rates=True)
    with pytest.raises(ValueError):
        bad = ObservationArc(t=arc.t, los=arc.los, observer=arc.observer, sigma=arc.sigma,
                             los_rate=arc.los_rate)
        solve(bad)


def test_unresolved_mesh_is_rejected_despite_optimizer_convergence():
    arc, _ = make_arc()
    with pytest.raises(CollocationFailure, match="mesh") as exc:
        solve(arc, options=SolverOptions(nodes=35, initial_guess="los", measurement_mode="exact",
                                        mesh_tolerance=1e-14))
    diagnostics = exc.value.diagnostics
    assert diagnostics["status"] == "Solve_Succeeded"
    assert diagnostics["success"] is False
    assert diagnostics["mesh_check_passed"] is False
    assert diagnostics["max_scaled_mesh_error"] > diagnostics["mesh_tolerance"]


def test_correlated_angle_covariance_obeys_independent_mahalanobis_bound():
    nominal, _ = make_arc(noise=True)
    covariance = nominal.sigma[:, None, None]**2 * np.array([[1.0, 0.6], [0.6, 2.25]])
    arc = ObservationArc(t=nominal.t, los=nominal.los, observer=nominal.observer,
                         sigma=nominal.sigma, covariance=covariance)
    confidence = 0.99
    result = solve(arc, options=SolverOptions(nodes=35, measurement_mode="bounded", confidence=confidence))
    relative = result.state_at(arc.t)[:, :3]-arc.observer
    direction = relative / np.linalg.norm(relative, axis=1)[:, None]
    residual = np.einsum("nij,nj->ni", tangent_basis(arc.los), direction-arc.los)
    mahal = np.einsum("ni,nij,nj->n", residual, np.linalg.inv(covariance), residual)
    assert np.max(mahal) <= chi2.ppf(confidence, 2)*(1+1e-5)

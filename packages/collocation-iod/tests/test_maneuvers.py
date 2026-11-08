"""Maneuver reconstruction checked against independent explicit integration.

These are reproducible numerical experiments, not reproductions of the paper's
tables (the paper does not provide their raw observations or control histories).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import chi2

from collocation_iod import CollocationFailure, ObservationArc, SolverOptions, solve
from collocation_iod.models import tangent_basis
from test_numerics import independent_rhs


def maneuver_arc(kind, *, bounded=False):
    t = np.linspace(0.3, 1.5, 13)
    x0 = np.array([0.872763, 0, 0.190505, 0, 0.235988, 0])
    control = np.array([0.02, -0.01, 0.015])
    jump = np.array([0.0, 0.02, 0.0])
    burn_time = t[6]
    if kind == "continuous":
        sol = solve_ivp(lambda t, x: independent_rhs(t, x, control), (t[0], t[-1]), x0,
                        dense_output=True, method="DOP853", rtol=2e-12, atol=2e-13)
        truth = lambda times: sol.sol(times).T
    else:
        first = solve_ivp(independent_rhs, (t[0], burn_time), x0,
                          dense_output=True, method="DOP853", rtol=2e-12, atol=2e-13)
        after = first.y[:, -1].copy()
        after[3:] += jump
        second = solve_ivp(independent_rhs, (burn_time, t[-1]), after,
                           dense_output=True, method="DOP853", rtol=2e-12, atol=2e-13)

        def truth(times):
            return np.array([first.sol(ti) if ti < burn_time else second.sol(ti) for ti in np.atleast_1d(times)])

    state = truth(t)
    # A prescribed observer trajectory with substantial parallax makes range
    # observable, isolating the maneuver reconstruction from geometric ambiguity.
    observer = np.column_stack([0.3*np.cos(5*t), -0.3+0.3*np.sin(5*t), 0.2*np.sin(3*t)])
    observer_v = np.column_stack([-1.5*np.sin(5*t), 1.5*np.cos(5*t), 0.6*np.cos(3*t)])
    rel = state[:, :3] - observer
    distance = np.linalg.norm(rel, axis=1)
    los = rel/distance[:, None]
    relative_velocity = state[:, 3:] - observer_v
    rate = (relative_velocity - los*np.sum(los*relative_velocity, axis=1)[:, None])/distance[:, None]
    covariance = None
    sigma = 0.0
    if bounded:
        # Exact nominal measurements with nonzero error bars isolate whether the
        # full four-dimensional uncertainty constraint is implemented correctly.
        sigma = 2e-5
        covariance = np.broadcast_to(np.diag([sigma**2, sigma**2, 1e-8, 1e-8]), (len(t), 4, 4)).copy()
    arc = ObservationArc(t=t, los=los, observer=observer, sigma=sigma, los_rate=rate,
                         observer_velocity=observer_v, covariance=covariance)
    return arc, truth, control, jump, burn_time


def test_continuous_control_recovery_from_independent_forced_dynamics():
    arc, truth, control, _, _ = maneuver_arc("continuous")
    coarse = solve(arc, options=SolverOptions(nodes=49, measurement_mode="exact", tolerance=1e-9))
    result = solve(arc, options=SolverOptions(nodes=193, measurement_mode="exact", tolerance=1e-9))
    times = np.linspace(arc.t[0], arc.t[-1], 101)
    fitted = result.state_at(times)
    error = np.linalg.norm(fitted[:, :3] - truth(times)[:, :3], axis=1)
    assert error.max() < 0.005
    # Endpoints are free in the paper's objective, so compare interior controls.
    reconstructed = result.control_at(times[20:-20])
    assert np.sqrt(np.mean(np.sum((reconstructed-control)**2, axis=1))) < 0.015
    assert result.diagnostics["max_interior_dynamics_residual"] < 0.002
    # A tiny NLP equality residual does not prove integration accuracy. The
    # independent vector-field residual must decrease as the mesh is refined.
    assert result.diagnostics["max_interior_dynamics_residual"] < coarse.diagnostics["max_interior_dynamics_residual"]/20
    propagated = solve_ivp(lambda t, x: independent_rhs(t, x, result.control_at(t)),
        (arc.t[0], arc.t[-1]), result.states[0], t_eval=arc.t, method="DOP853", rtol=1e-11, atol=1e-12)
    assert propagated.success
    assert np.max(np.linalg.norm(propagated.y.T-result.state_at(arc.t), axis=1)) < 5e-6


def test_impulsive_control_recovers_physical_jump_sign_and_timing():
    arc, truth, _, jump, burn = maneuver_arc("impulsive")
    result = solve(arc, options=SolverOptions(mode="impulsive", nodes=49,
                  measurement_mode="exact", tolerance=1e-9))
    # Literal Eq29 increments have the opposite sign to the physical velocity
    # jump. An independently integrated +y kick should be recovered as +y here.
    magnitudes = np.linalg.norm(result.impulses, axis=1)
    peak = np.argmax(magnitudes)
    assert abs(result.t[peak+1] - burn) <= 2*np.diff(result.t).max()
    assert result.impulses[peak, 1] > 0.01
    assert np.linalg.norm(result.impulses.sum(axis=0)-jump) < 0.01
    assert np.linalg.norm(result.state_at(burn)-truth([burn])[0]) < 0.01
    np.testing.assert_allclose(result.state_at(result.t[peak+1], side="right")[3:] -
                               result.state_at(result.t[peak+1], side="left")[3:],
                               result.impulses[peak], atol=2e-14)
    # The left limit must be the actual interval polynomial endpoint, not merely
    # a synthetic pair of node states whose difference has the desired sign.
    knot = result.t[peak+1]
    np.testing.assert_allclose(result.state_at(np.nextafter(knot, -np.inf)),
                               result.state_at(knot, side="left"), atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.state_at(knot, side="right"), result.states[peak+1], atol=1e-14)
    np.testing.assert_allclose(result.state_at(np.nextafter(knot, np.inf)),
                               result.state_at(knot, side="right"), atol=1e-7, rtol=0)
    for outside in (result.t[0]-1e-6, result.t[-1]+1e-6, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            result.state_at(outside)
        with pytest.raises(ValueError):
            result.control_at(outside)


def test_hybrid_reconstruction_tracks_independent_impulsive_truth():
    arc, truth, _, _, _ = maneuver_arc("impulsive")
    result = solve(arc, options=SolverOptions(mode="hybrid", nodes=49,
                  hybrid_weight=0.01, measurement_mode="exact", tolerance=1e-9))
    times = np.linspace(arc.t[0], arc.t[-1], 97)
    assert np.max(np.linalg.norm(result.state_at(times)[:, :3]-truth(times)[:, :3], axis=1)) < 0.01
    assert result.diagnostics["continuous_cost"] >= 0
    assert result.diagnostics["impulsive_l1_cost"] >= 0
    assert result.diagnostics["max_collocation_defect"] < 1e-7


def test_bounded_angles_and_rates_obey_independent_mahalanobis_check():
    arc, truth, _, _, _ = maneuver_arc("continuous", bounded=True)
    confidence = 0.99
    result = solve(arc, options=SolverOptions(nodes=49, measurement_mode="bounded", confidence=confidence))
    x = result.state_at(arc.t)
    rel = x[:, :3]-arc.observer
    distance = np.linalg.norm(rel, axis=1)
    direction = rel/distance[:, None]
    velocity = x[:, 3:]-arc.observer_velocity
    rate = (velocity-direction*np.sum(direction*velocity, axis=1)[:, None])/distance[:, None]
    tangent = tangent_basis(arc.los)
    residual = np.concatenate([np.einsum("nij,nj->ni", tangent, direction-arc.los),
                               np.einsum("nij,nj->ni", tangent, rate-arc.los_rate)], axis=1)
    mahal = np.einsum("ni,nij,nj->n", residual, np.linalg.inv(arc.covariance), residual)
    assert np.max(mahal) <= chi2.ppf(confidence, 4)*(1+1e-5)
    assert np.max(np.linalg.norm(x[:, :3]-truth(arc.t)[:, :3], axis=1)) < 0.01


def test_iteration_limit_is_an_explicit_failure():
    arc, _, _, _, _ = maneuver_arc("continuous")
    with pytest.raises(CollocationFailure) as exc:
        solve(arc, options=SolverOptions(nodes=20, max_iterations=1, measurement_mode="exact"))
    assert not exc.value.diagnostics["success"]

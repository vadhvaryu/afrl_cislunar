"""Independent checks of the physical and transcription building blocks."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from collocation_iod.models import cr3bp_rhs, tangent_basis
from collocation_iod.transcription import control_polynomial, hermite_simpson, state_polynomial


MU = 0.0121505


def independent_rhs(t, state, control=None):
    """Vector mechanics form, independent of the implementation's component equations."""
    p, v = state[:3], state[3:]
    omega = np.array([0.0, 0.0, 1.0])
    earth = p + np.array([MU, 0.0, 0.0])
    moon = p - np.array([1.0 - MU, 0.0, 0.0])
    a = -(1.0 - MU) * earth / np.linalg.norm(earth)**3 - MU * moon / np.linalg.norm(moon)**3
    a -= 2 * np.cross(omega, v) + np.cross(omega, np.cross(omega, p))
    if control is not None:
        a += control
    return np.concatenate([v, a])


@pytest.mark.parametrize("state", [
    [0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0],
    [1.15, -0.2, 0.04, 0.04, -0.13, 0.08],
    [0.4, 0.7, -0.2, -0.2, 0.1, 0.05],
])
def test_dynamics_against_vector_mechanics(state):
    state = np.asarray(state)
    u = np.array([0.03, -0.04, 0.05])
    np.testing.assert_allclose(cr3bp_rhs(state, MU, u), independent_rhs(0.0, state, u), atol=2e-14)


def test_dynamics_agree_with_existing_repository():
    existing = pytest.importorskip("od_stack.reference_cr3bp")
    x = np.array([0.9, -0.12, 0.18, 0.04, 0.2, -0.03])
    np.testing.assert_allclose(cr3bp_rhs(x, MU), existing.nondim_cr3bp(0.0, x, MU), atol=1e-14)


def test_state_interpolation_integrates_quadratic_derivative_exactly():
    # The antiderivative is cubic. Simpson's rule and the continuous extension
    # must recover it everywhere, not only at collocation points.
    a = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.4])
    b = np.array([0.5, 0.1, -0.2, 0.3, 0.4, 0.1])
    c = np.array([-0.2, 0.3, 0.1, 0.2, -0.1, 0.3])
    x0 = np.arange(6) / 3.0
    h = 0.8
    deriv = lambda t: a + b * t + c * t**2
    for t in np.linspace(0, h, 11):
        expected = x0 + a*t + b*t**2/2 + c*t**3/3
        actual = state_polynomial(x0, deriv(0), deriv(h/2), deriv(h), h, t)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_control_interpolation_preserves_quadratic():
    h = 0.7
    func = lambda t: np.array([t*t + 0.2, 2*t - 0.5, 1 - t + 3*t*t])
    for t in np.linspace(0, h, 11):
        np.testing.assert_allclose(control_polynomial(func(0), func(h/2), func(h), h, t), func(t), atol=2e-14)


def test_transcription_defect_has_fifth_order_local_accuracy():
    x0 = np.array([0.872763, 0, 0.190505, 0, 0.235988, 0])
    errors = []
    for h in (0.1, 0.05, 0.025):
        truth = solve_ivp(independent_rhs, (0, h), x0, method="DOP853", rtol=2e-13, atol=2e-14)
        defect, *_ = hermite_simpson(x0, truth.y[:, -1], np.zeros(3), np.zeros(3), np.zeros(3), h, MU)
        errors.append(np.linalg.norm(defect))
    # Halving the interval should reduce the local defect by about 2**5.
    assert errors[0] / errors[1] > 20
    assert errors[1] / errors[2] > 20


def test_tangent_basis_handles_poles_and_ra_wrap():
    los = np.array([[0, 0, 1], [0, 0, -1], [1, 1e-13, 0], [1, -1e-13, 0]], dtype=float)
    basis = tangent_basis(los)
    assert basis.shape == (4, 2, 3)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", basis, los), 0, atol=1e-13)
    np.testing.assert_allclose(basis @ basis.transpose(0, 2, 1), np.broadcast_to(np.eye(2), (4, 2, 2)), atol=1e-13)

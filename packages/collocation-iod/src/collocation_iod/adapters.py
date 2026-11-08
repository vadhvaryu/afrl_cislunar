"""Explicit unit/frame conversion and optional integration with ``od_stack``.

The numerical solver has no dependency on the simulation repository. This
adapter is the only code that imports its contracts or registers an IOD method.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rotation(tau: np.ndarray) -> np.ndarray:
    """Matrices mapping rotating axes to EMI at absolute nondimensional time."""
    tau = np.atleast_1d(tau)
    c, s = np.cos(tau), np.sin(tau)
    matrix = np.zeros((len(tau), 3, 3))
    matrix[:, 0, 0], matrix[:, 0, 1] = c, -s
    matrix[:, 1, 0], matrix[:, 1, 1] = s, c
    matrix[:, 2, 2] = 1.0
    return matrix


def observations_from_emi(t_s, los, observer_m, sigma_rad, *, system):
    """Map the stack's angular observations to a rotating nondimensional arc.

    Time is relative to the original EMI frame origin, even for observation
    subsets. ``sigma_rad`` denotes equal independent RA/Dec standard deviations,
    as in ``obs_generator.py``. Their covariance is projected through the angular
    Jacobian into the solver's local tangent basis; RA's cos(dec) factor is kept.
    ``observer_m`` is Earth-centered. Core observers are barycentric.
    """
    from . import ObservationArc
    from .models import tangent_basis

    t_s, los = np.asarray(t_s, dtype=float), np.asarray(los, dtype=float)
    observer_m = np.asarray(observer_m, dtype=float)
    if t_s.ndim != 1 or los.shape != (len(t_s), 3) or observer_m.shape != los.shape:
        raise ValueError("Expected t_s (N,), los (N,3), and observer_m (N,3)")
    sigma = np.broadcast_to(np.asarray(sigma_rad, dtype=float), (len(t_s),)).copy()
    norms = np.linalg.norm(los, axis=1)
    if not np.isfinite(np.r_[t_s, los.ravel(), observer_m.ravel(), sigma]).all():
        raise ValueError("Observations must be finite")
    if np.any(norms <= 0) or np.any(sigma < 0):
        raise ValueError("LOS must be nonzero and standard deviations nonnegative")
    if np.any(sigma == 0) and not np.all(sigma == 0):
        raise ValueError("Mixed exact and uncertain observations are not supported; use all positive or all zero sigma")
    los = los / norms[:, None]
    tau = t_s / system.tu_s
    inverse = _rotation(tau).transpose(0, 2, 1)
    los_rot = np.einsum("nij,nj->ni", inverse, los)
    observer_rot = np.einsum("nij,nj->ni", inverse, observer_m / system.du_m)
    observer_rot[:, 0] -= system.mu

    ra = np.arctan2(los[:, 1], los[:, 0])
    dec = np.arctan2(los[:, 2], np.hypot(los[:, 0], los[:, 1]))
    jac = np.stack((
        np.column_stack((-np.cos(dec) * np.sin(ra), np.cos(dec) * np.cos(ra), np.zeros_like(ra))),
        np.column_stack((-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec))),
    ), axis=-1)
    jac_rot = inverse @ jac
    projected = tangent_basis(los_rot) @ jac_rot
    covariance = (projected @ projected.transpose(0, 2, 1)) * sigma[:, None, None] ** 2
    if np.all(sigma == 0):
        covariance = None  # The core selects exact constraints for a noise-free arc.
    return ObservationArc(t=tau, los=los_rot, observer=observer_rot, sigma=sigma, covariance=covariance)


def rotating_state_to_emi(state, t_s: float, *, system):
    """Convert one rotating barycentric state to Earth-centered SI position/velocity."""
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all() or not np.isfinite(t_s):
        raise ValueError("Expected a finite rotating six-state and time")
    rho = state[:3] + np.array([system.mu, 0.0, 0.0])
    matrix = _rotation(np.array([t_s / system.tu_s]))[0]
    r = matrix @ rho * system.du_m
    v = matrix @ (state[3:] + np.cross([0.0, 0.0, 1.0], rho)) * system.du_m / system.tu_s
    return r, v


class CollocationIOD:
    """Drop-in IOD stage using the stack's ``Observations``/``StateEstimate``.

    Keyword settings are explicit ``SolverOptions`` fields. ``last_result``
    retains the reconstructed trajectory and controls; the downstream stack's
    single-state contract cannot itself propagate reconstructed maneuvers.
    """

    name = "collocation"
    config_block = "collocation"

    def __init__(self, sys: Any, epoch_index: int = 2, *, options=None,
                 orekit_data=None, truth_path=None, **settings):
        from . import CR3BPSystem, SolverOptions

        if isinstance(epoch_index, bool) or int(epoch_index) != epoch_index:
            raise ValueError("epoch_index must be an integer")
        if options is not None and settings:
            raise ValueError("Pass options or individual solver settings, not both")
        self.system = CR3BPSystem(mu=sys.mu, du_m=sys.du_m, tu_s=sys.tu_s)
        self.epoch_index = int(epoch_index)
        self.options = options if options is not None else SolverOptions(**settings)
        self.last_result = None
        # orekit_data/truth_path are accepted only for build_stages compatibility.
        # Neither is read: this method estimates exclusively from observations.

    def estimate(self, obs):
        from od_stack.contracts import StateEstimate
        from od_stack.iod import IODFailure

        from . import solve

        self.last_result = None
        if not 0 <= self.epoch_index < len(obs.t):
            raise IODFailure(f"collocation: epoch_index {self.epoch_index} is outside the observation arc")
        try:
            arc = observations_from_emi(obs.t, obs.los, obs.observer, obs.sigma, system=self.system)
            result = solve(arc, system=self.system, options=self.options)
            t = float(obs.t[self.epoch_index])
            r, v = rotating_state_to_emi(result.state_at(t / self.system.tu_s), t, system=self.system)
        except (ValueError, RuntimeError) as exc:
            raise IODFailure(f"collocation: {exc}") from exc
        self.last_result = result
        return StateEstimate(
            t=t, r=r, v=v, cov=None, method=self.name,
            info={"collocation": result.diagnostics.copy(), "obs_indices": list(range(len(obs.t))),
                  "maneuver_model": self.options.mode,
                  "control_history_available_on_adapter": True},
        )


def register_with_od_stack():
    """Register the adapter explicitly; importing this package has no side effects."""
    from od_stack.iod import IOD_METHODS

    IOD_METHODS[CollocationIOD.name] = CollocationIOD
    return CollocationIOD


def stack_main(argv=None):
    """Compatibility entry point; YAML alone selects the IOD method."""
    import sys

    register_with_od_stack()
    from od_stack.run import main

    args = list(sys.argv[1:] if argv is None else argv)
    return main(args)

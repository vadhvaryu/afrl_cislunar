"""Public data contracts. Core quantities use barycentric rotating CR3BP units."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CR3BPSystem:
    mu: float = 0.0121505
    du_m: float = 384338000.0
    tu_s: float = 375099.0

    def __post_init__(self):
        if not np.isfinite([self.mu, self.du_m, self.tu_s]).all() or not (
            0 < self.mu < 0.5 and self.du_m > 0 and self.tu_s > 0
        ):
            raise ValueError("Require 0 < mu < 0.5 and positive finite DU and TU")


def tangent_basis(los):
    """Rows are east and north unit tangent vectors; supports (...,3) LOS."""
    los = np.asarray(los, dtype=float)
    east = np.stack((-los[..., 1], los[..., 0], np.zeros(los.shape[:-1])), axis=-1)
    length = np.linalg.norm(east, axis=-1, keepdims=True)
    east = np.where(length > 1e-12, east / np.maximum(length, 1e-12), [0., 1., 0.])
    return np.stack((east, np.cross(los, east)), axis=-2)


def cr3bp_rhs(state, mu, control=None):
    """Eq. (1), NumPy evaluation with arbitrary leading dimensions."""
    state = np.asarray(state)
    r, v = state[..., :3], state[..., 3:]
    d1 = r + np.array([mu, 0, 0])
    d2 = r - np.array([1 - mu, 0, 0])
    n1 = np.linalg.norm(d1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(d2, axis=-1, keepdims=True)
    if np.any(n1 <= 1e-12) or np.any(n2 <= 1e-12):
        raise ValueError("CR3BP state is at a primary's singularity")
    a = -(1 - mu) * d1 / n1**3 - mu * d2 / n2**3
    a = a + np.stack((r[..., 0] + 2*v[..., 1], r[..., 1] - 2*v[..., 0],
                      np.zeros(r.shape[:-1])), axis=-1)
    if control is not None:
        a = a + np.asarray(control)
    return np.concatenate((v, a), axis=-1)


@dataclass(frozen=True)
class ObservationArc:
    t: np.ndarray
    los: np.ndarray
    observer: np.ndarray
    sigma: np.ndarray | float = 0.0
    los_rate: np.ndarray | None = None
    observer_velocity: np.ndarray | None = None
    covariance: np.ndarray | None = None

    def __post_init__(self):
        for key in ("t", "los", "observer", "sigma", "los_rate", "observer_velocity", "covariance"):
            value = getattr(self, key)
            if value is not None:
                arr = np.array(value, dtype=float, copy=True)
                if not np.isfinite(arr).all():
                    raise ValueError(f"{key} must be finite")
                object.__setattr__(self, key, arr)
        n = self.t.size
        if self.t.ndim != 1 or n < 3 or np.any(np.diff(self.t) <= 0):
            raise ValueError("At least three strictly increasing observation times are required")
        if self.los.shape != (n, 3) or self.observer.shape != (n, 3):
            raise ValueError("los and observer must have shape (N,3)")
        if not np.allclose(np.linalg.norm(self.los, axis=1), 1, atol=1e-8, rtol=0):
            raise ValueError("los must contain unit vectors")
        sigma = np.broadcast_to(self.sigma, (n,)).copy()
        if np.any(sigma < 0):
            raise ValueError("sigma must be nonnegative")
        object.__setattr__(self, "sigma", sigma)
        if (self.los_rate is None) != (self.observer_velocity is None):
            raise ValueError("los_rate and observer_velocity must be supplied together")
        if self.los_rate is not None:
            if self.los_rate.shape != (n, 3) or self.observer_velocity.shape != (n, 3):
                raise ValueError("los_rate and observer_velocity must have shape (N,3)")
            if np.max(np.abs(np.sum(self.los * self.los_rate, axis=1))) > 1e-8:
                raise ValueError("los_rate must be perpendicular to los")
        if self.covariance is not None:
            q = 2 if self.los_rate is None else 4
            if self.covariance.shape != (n, q, q):
                raise ValueError(f"covariance must have shape (N,{q},{q})")
            if not np.allclose(self.covariance, self.covariance.transpose(0,2,1), rtol=1e-10, atol=1e-24):
                raise ValueError("covariance must be symmetric")
            try:
                np.linalg.cholesky(self.covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("covariance must be positive definite") from exc


@dataclass(frozen=True)
class SolverOptions:
    mode: str = "continuous"
    nodes: int = 80
    measurement_mode: str = "bounded"
    confidence: float = 0.95
    initial_guess: str = "auto"
    max_iterations: int = 1000
    tolerance: float = 1e-8
    feasibility_tolerance: float = 1e-6
    mesh_tolerance: float = 1e-3
    range_bounds: tuple[float, float] = (1e-4, 5.0)
    control_bound: float = 10.0
    impulse_bound: float = 2.0
    hybrid_weight: float = 0.5
    print_level: int = 0

    def __post_init__(self):
        if self.mode not in ("continuous", "impulsive", "hybrid"):
            raise ValueError("mode must be continuous, impulsive, or hybrid")
        if self.measurement_mode not in ("exact", "bounded"):
            raise ValueError("measurement_mode must be exact or bounded")
        if self.initial_guess not in ("auto", "los", "L1", "L2", "L4"):
            raise ValueError("initial_guess must be auto, los, L1, L2, or L4")
        if not isinstance(self.nodes, int) or self.nodes < 3:
            raise ValueError("nodes must be an integer >= 3")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not 0 < self.confidence < 1 or not 0 < self.hybrid_weight < 1:
            raise ValueError("confidence and hybrid_weight must lie strictly between 0 and 1")
        vals = [self.tolerance, self.feasibility_tolerance, self.mesh_tolerance, self.control_bound, self.impulse_bound]
        if not np.isfinite(vals).all() or min(vals) <= 0:
            raise ValueError("tolerances and control/impulse bounds must be positive and finite")
        if len(self.range_bounds) != 2 or not np.isfinite(self.range_bounds).all() or not (
            0 < self.range_bounds[0] < self.range_bounds[1]
        ):
            raise ValueError("range_bounds must be positive increasing finite distances in DU")
        if not isinstance(self.print_level, int) or not 0 <= self.print_level <= 12:
            raise ValueError("print_level must be an integer between 0 and 12")


class CollocationFailure(RuntimeError):
    """NLP failure, with diagnostics retained for callers."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

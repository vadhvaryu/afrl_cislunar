"""
Data contracts: what each stage of the stack passes to the next.

    data source --Observations--> IOD --StateEstimate--> propagator --Trajectory--> results

Every module talks only through these three objects, so any IOD method, data
source, or propagator can be swapped without touching the others.

Conventions (one set, used everywhere):
    Frame : "EMI" = Earth-centered, non-rotating. x points at the Moon at t0,
            z is normal to the Earth-Moon orbital plane. This is the frame the
            observation generator measures RA/Dec in. It is NOT Earth's equator.
    Units : SI. meters, meters/second, seconds, radians.
    Time  : seconds since t0 (the first observation in the generator's data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FRAME = "EMI"


@dataclass
class Observations:
    """Output of a data source; input to IOD."""
    t: np.ndarray          # (N,)   seconds since t0
    los: np.ndarray        # (N, 3) unit line-of-sight vectors, observer -> object, EMI
    observer: np.ndarray   # (N, 3) observer position, meters, EMI (all zeros = geocenter)
    sigma: np.ndarray      # (N,)   1-sigma angle noise, radians
    ra: np.ndarray         # (N,)   radians (kept for plots/debugging)
    dec: np.ndarray        # (N,)   radians

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class StateEstimate:
    """Output of IOD (or batch least squares); input to a propagator."""
    t: float               # epoch of the state, seconds since t0
    r: np.ndarray          # (3,) meters, EMI
    v: np.ndarray          # (3,) meters/second, EMI
    cov: np.ndarray | None = None   # (6, 6) covariance, if the method provides one
    method: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Output of a propagator (or the truth file); input to results."""
    t: np.ndarray          # (M,)   seconds since t0
    r: np.ndarray          # (M, 3) meters, EMI
    v: np.ndarray          # (M, 3) meters/second, EMI
    label: str = ""

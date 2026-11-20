"""Synthetic optical observations, without filesystem or plotting side effects."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import Observations, Trajectory
from .dynamics import cr3bp_eom
from .frames import rotating_to_emi, radec_to_los


class SyntheticTruth:
    def __init__(self, settings, system):
        self.settings, self.system = settings, system

    def at(self, times):
        t = np.asarray(times, dtype=float)
        tau = self.system.tau(t)
        epoch = self.settings["epoch_days"] * 86400 / self.system.tu_s
        initial = np.asarray(self.settings["initial_state"]["values"], dtype=float)
        states = np.empty((len(t), 6))
        for mask, end in ((tau >= epoch, tau.max()), (tau < epoch, tau.min())):
            if not mask.any():
                continue
            if end == epoch:
                states[mask] = initial
            else:
                solution = solve_ivp(cr3bp_eom, (epoch, end), initial, args=(self.system.mu,),
                                     dense_output=True, **self.settings["integrator"])
                if not solution.success:
                    raise RuntimeError(f"Truth propagation failed: {solution.message}")
                states[mask] = solution.sol(tau[mask]).T
        rv = [rotating_to_emi(ti, x, self.system) for ti, x in zip(tau, states)]
        return Trajectory(t=t, r=np.array([r for r, v in rv]), v=np.array([v for r, v in rv]), label="Truth")


SPEED_OF_LIGHT_M_S = 299_792_458.0


def generate(settings, system):
    """Return observations and a truth evaluator; truth never enters the estimator.
    With observations.light_time (the default), each line of sight points to where the object was when the
    light left it, as a telescope sees it and as Orekit's angle model assumes; about 0.6-0.8 arcsec for a
    halo orbit. Without it, the angles disagree with Orekit's by that much, which dominates below ~1 arcsec."""
    truth = SyntheticTruth(settings["truth"], system)
    t = np.asarray(settings["observations"]["times_days"], dtype=float) * 86400
    position = np.asarray(settings["observer"]["position_km"]) * 1000
    observers = np.broadcast_to(position, (len(t), 3)).copy()
    emitted = t
    for _ in range(3 if settings["observations"].get("light_time", True) else 0):   # converges in 2
        emitted = t - np.linalg.norm(truth.at(emitted).r - observers, axis=1) / SPEED_OF_LIGHT_M_S
    delta = truth.at(emitted).r - observers
    ranges = np.linalg.norm(delta, axis=1)
    if np.any(ranges <= 0):
        raise ValueError("Observer and target cannot coincide")
    ra = np.arctan2(delta[:, 1], delta[:, 0])
    dec = np.arcsin(np.clip(delta[:, 2] / ranges, -1, 1))
    sigma = np.deg2rad(settings["observations"]["noise"]["sigma_arcsec"] / 3600)
    rng = np.random.default_rng(settings["seed"])
    ra = (ra + rng.normal(0, sigma, len(t))) % (2 * np.pi)
    dec = dec + rng.normal(0, sigma, len(t))
    if np.any(np.abs(dec) > np.pi / 2):
        raise ValueError("Noise generated declination outside [-90, 90] degrees; reduce noise or change geometry")
    return Observations(t=t, los=radec_to_los(ra, dec), observer=observers,
                        sigma=np.full(len(t), sigma), ra=ra, dec=dec), truth

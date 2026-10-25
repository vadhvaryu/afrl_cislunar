"""
Other cislunar truth orbits, so results can be checked beyond the one L1 halo in data/.

    python -m od_stack.orbit_catalog            # writes results/orbits/<name>/synthetic_obs.csv and _truth.csv

Each orbit is made exactly periodic in OUR CR3BP (mu = 0.0121505) by differential correction
of a perpendicular x-z-plane crossing, then observed with the same recipe as obs_generator.py
(cr3bp_tools.make_synthetic_obs: Earth-center observer, 5 RA/Dec pairs over 8 days, 10" noise,
seed 1). The files have the same columns as data/synthetic_obs.csv and data/synthetic_truth.csv,
so the whole stack runs on them by pointing data.observations / data.truth at them.

Seeds: except for the current truth halo (obs_generator.py's TRUTH_STATE), the starting states
are members of the periodic-orbit catalogue in https://github.com/Bishwaswarup/cislunar-custody
(data/catalogue.npz and data/named_orbits.json, MIT License, Copyright (c) 2026 Bishwaswarup
Nayak). That catalogue uses mu = 0.012150584269542242, so every seed is re-corrected here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys as _sys

import numpy as np
from scipy.integrate import solve_ivp

from .cr3bp_tools import _eom_stm, _line_of_sight_inertial, _ra_dec_deg, make_synthetic_obs
from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem

DAY = 86400.0
OBS_TIMES = np.linspace(0, 8 * DAY, 5)          # obs_generator.py: N_OBS = 5 over SPAN_DAYS = 8
NOISE_ARCSEC, SEED = 10.0, 1                    # obs_generator.py: NOISE_ARCSEC, SEED
OUT_DIR = "results/orbits"

# name: (seed state at a perpendicular y = 0 crossing, planar?, description)
ORBITS = {
    "L1_halo_truth": ([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0], False,
                      "The current truth orbit (obs_generator.py), L1 northern halo"),
    "L1_halo_small": ([0.82476762, 0.0, 0.06773, 0.0, 0.17970, 0.0], False,
                      "Small L1 northern halo (cislunar-custody L1_halo_north[14]), very unstable"),
    "L1_lyapunov": ([0.81581, 0.0, 0.0, 0.0, 0.20897, 0.0], True,
                    "Planar L1 Lyapunov orbit (cislunar-custody L1_lyapunov[19])"),
    "L2_halo_large": ([1.02745, 0.0, 0.07349, 0.0, 0.44713, 0.0], False,
                      "Large L2 southern-family halo (cislunar-custody L2_halo_south[23])"),
    "NRHO_9_2": ([0.9873800654424657, 0.0, 0.008439917431935787, 0.0, 1.6672892257591334, 0.0], False,
                 "9:2 near-rectilinear halo orbit, Gateway's orbit (cislunar-custody named_orbits.json)"),
    "DRO": ([1.17539, 0.0, 0.0, 0.0, -0.49425, 0.0], True,
            "Distant retrograde orbit (cislunar-custody DRO[46]), stable"),
}


# ----------------------------------------------------------------------------- periodic orbits

def _half_period(x0, mu, stm=True, t_max=20.0):
    """Propagate to the next y = 0 crossing (half a period for a symmetric orbit)."""
    direction = -np.sign(x0[4]) if x0[4] != 0 else 0

    def cross(t, s, mu):
        return s[1]

    cross.terminal, cross.direction = True, direction
    y0 = np.concatenate([x0, np.eye(6).ravel()]) if stm else np.asarray(x0, float)
    sol = solve_ivp(_eom_stm if stm else cr3bp_eom, [0, t_max], y0, args=(mu,), events=cross,
                    first_step=1e-6, **TOL)
    if not len(sol.t_events[0]):
        raise RuntimeError("no y = 0 crossing found")
    y = sol.y_events[0][0]
    return sol.t_events[0][0], y[:6], (y[6:].reshape(6, 6) if stm else None)


def correct_periodic(x0, mu: float, planar: bool, tol: float = 1e-12, max_iter: int = 30):
    """Differential correction of a symmetric periodic orbit: start on the x-z plane with velocity
    along y (state [x, 0, z, 0, vy, 0]) and require the next y = 0 crossing to be perpendicular
    (vx = vz = 0). z0 is held fixed; x0 and vy0 are adjusted (planar: x0 fixed, vy0 adjusted).
    Returns (corrected state, period)."""
    x = np.array(x0, float)
    for _ in range(max_iter):
        t_half, xf, phi = _half_period(x, mu)
        f = np.array(cr3bp_eom(0, xf, mu))
        if planar:
            F = np.array([xf[3]])
            DF = np.array([[phi[3, 4] - f[3] / f[1] * phi[1, 4]]])
            free = [4]
        else:
            F = np.array([xf[3], xf[5]])
            free = [0, 4]
            DF = np.array([[phi[j, a] - f[j] / f[1] * phi[1, a] for a in free] for j in (3, 5)])
        if np.abs(F).max() < tol:
            return x, 2 * t_half
        x[free] -= np.linalg.solve(DF, F)
    raise RuntimeError(f"differential correction did not converge (residual {np.abs(F).max():.2e})")


def stability_index(x0, period, mu):
    """(|lambda_max| + 1/|lambda_max|) / 2 of the monodromy matrix (1 = neutrally stable)."""
    sol = solve_ivp(_eom_stm, [0, period], np.concatenate([x0, np.eye(6).ravel()]), args=(mu,), **TOL)
    lam = np.abs(np.linalg.eigvals(sol.y[6:, -1].reshape(6, 6))).max()
    return 0.5 * (lam + 1 / lam)


def describe(x0, period, mu, sys: EarthMoonSystem) -> dict:
    sol = solve_ivp(cr3bp_eom, [0, period], x0, args=(mu,), dense_output=True, **TOL)
    tt = np.linspace(0, period, 4000)
    p = sol.sol(tt)[:3].T
    d_moon = np.linalg.norm(p - [1 - mu, 0, 0], axis=1) * sys.du_km
    d_earth = np.linalg.norm(p - [-mu, 0, 0], axis=1) * sys.du_km
    return {"period_days": period * sys.tu_s / DAY, "perilune_km": float(d_moon.min()),
            "apolune_km": float(d_moon.max()), "earth_range_km": [float(d_earth.min()), float(d_earth.max())],
            "stability_index": float(stability_index(x0, period, mu))}


# ----------------------------------------------------------------------------- files

def write_orbit(out_dir: str, name: str, x0, sys: EarthMoonSystem) -> dict:
    """synthetic_obs.csv / synthetic_truth.csv in obs_generator.py's format (truth with 12 decimals)."""
    os.makedirs(out_dir, exist_ok=True)
    obs = make_synthetic_obs(x0, OBS_TIMES, NOISE_ARCSEC, SEED, sys)
    taus = sys.tau(OBS_TIMES)
    sol = solve_ivp(cr3bp_eom, [0, taus[-1]], np.asarray(x0, float), args=(sys.mu,), dense_output=True, **TOL)
    Y = sol.sol(taus).T
    l = _line_of_sight_inertial(Y[:, :3], taus, sys.earth_rot)
    ra_true, dec_true = _ra_dec_deg(l)
    rng_km = np.linalg.norm(l, axis=1) * sys.du_km
    days = OBS_TIMES / DAY
    with open(os.path.join(out_dir, "synthetic_obs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_id", "time_days", "ra_deg", "dec_deg", "sigma_arcsec"])
        for i in range(len(days)):
            w.writerow([i, f"{days[i]:.6f}", f"{np.degrees(obs.ra[i]) % 360:.10f}", f"{np.degrees(obs.dec[i]):.10f}",
                        NOISE_ARCSEC])
    with open(os.path.join(out_dir, "synthetic_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_id", "time_days", "x", "y", "z", "vx", "vy", "vz", "range_km", "ra_true_deg", "dec_true_deg"])
        for i in range(len(days)):
            w.writerow([i, f"{days[i]:.6f}", *[f"{v:.12f}" for v in Y[i]], f"{rng_km[i]:.3f}",
                        f"{ra_true[i]:.10f}", f"{dec_true[i]:.10f}"])
    return {"obs": os.path.join(out_dir, "synthetic_obs.csv"), "truth": os.path.join(out_dir, "synthetic_truth.csv")}


def build(out: str = OUT_DIR, sys: EarthMoonSystem | None = None, names=None) -> dict:
    sys = sys or EarthMoonSystem()
    catalog = {}
    for name in names or ORBITS:
        seed, planar, desc = ORBITS[name]
        if name == "L1_halo_truth":
            x0 = np.array(seed, float)                    # keep the generator's state exactly
            period = correct_periodic(seed, sys.mu, planar)[1]
        else:
            x0, period = correct_periodic(seed, sys.mu, planar)
        info = {"description": desc, "planar": planar, "state0": [float(v) for v in x0], "period_TU": float(period),
                **describe(x0, period, sys.mu, sys), **write_orbit(os.path.join(out, name), name, x0, sys)}
        catalog[name] = info
    with open(os.path.join(out, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
    return catalog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate other cislunar truth orbits and their observations")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    cat = build(a.out)
    print(f"{'orbit':<15} {'period (d)':>10} {'perilune (km)':>14} {'apolune (km)':>13} {'Earth range (km)':>22} {'stability':>10}")
    for n, c in cat.items():
        print(f"{n:<15} {c['period_days']:10.2f} {c['perilune_km']:14,.0f} {c['apolune_km']:13,.0f} "
              f"{c['earth_range_km'][0]:10,.0f}-{c['earth_range_km'][1]:<10,.0f} {c['stability_index']:10.1f}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

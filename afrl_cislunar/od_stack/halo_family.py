"""
The L1 northern halo family in OUR CR3BP (mu = 0.0121505), sampled for catalog-search IOD.

    fam = load_family(sys)          # results/halo_catalog/L1N.npz, generated on first use (~1 min)
    fam.state(s, phase)             # rotating state on (interpolated) member s at normalized phase

How the family is made (no knowledge of the truth orbit is used):
    1. Seed: the small L1 northern halo [0.82476762, 0, 0.06773, 0, 0.17970, 0] from the
       cislunar-custody catalogue (L1_halo_north[14], MIT License, credited in orbit_catalog.py),
       re-corrected in our mu by orbit_catalog.correct_periodic.
    2. Natural-parameter continuation in z0 (the z of the perpendicular x-z-plane crossing with
       vy > 0), down to z0_min and up to z0_max, with a secant predictor. z0 is monotonic along
       this stretch of the family; x0 and the period are not (x0 runs from ~0.823 to ~0.933 and
       turns back near z0 = 0.25, and most of that change happens for z0 in 0.17 - 0.22).
    3. Members are then re-placed evenly in arc length of (x0, z0, vy0, 0.1 T), so the fast part
       of the family near z0 ~ 0.19 gets more members than a uniform z0 grid would give it, and
       each is corrected again from an interpolated guess.

Stored per member (index m = 0..M-1, ordered by z0): the corrected initial state [x0, 0, z0, 0,
vy0, 0], period T (TU), Jacobi constant, perilune (km), stability index, and the rotating state
at N evenly spaced phases phi = k/N (k = 0..N-1; phase 0 is the initial state, time = phi T).

Interpolation (`state`): within a member, cubic Hermite in phase using the stored states and
their CR3BP derivatives (error ~1e-9 DU with N = 1000). Between members, linear in the
continuous member index s at the same normalized phase, with the period interpolated the same
way. An interpolated "member" is not exactly periodic; it is a smooth stand-in for the family
between the stored members (measured midpoint error: see tests/test_iod_halo_catalog.py).
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import TOL, cr3bp_eom
from .frames import EarthMoonSystem
from .orbit_catalog import correct_periodic, stability_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "results", "halo_catalog", "L1N.npz")

# cislunar-custody L1_halo_north[14] (MIT); NOT the truth orbit.
SEED_SMALL_HALO = (0.82476762, 0.0, 0.06773, 0.0, 0.17970, 0.0)
DEFAULTS = dict(z0_min=0.01, z0_max=0.26, n_members=100, n_phases=1000, period_weight=0.1)
FORMAT_VERSION = 1


def cr3bp_accel(X: np.ndarray, mu: float) -> np.ndarray:
    """Vectorized CR3BP state derivative for states X (..., 6)."""
    x, y, z, vx, vy, vz = np.moveaxis(X, -1, 0)
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    a1, a2 = (1 - mu) / r1**3, mu / r2**3
    ax = 2 * vy + x - a1 * (x + mu) - a2 * (x - 1 + mu)
    ay = -2 * vx + y - a1 * y - a2 * y
    az = -a1 * z - a2 * z
    return np.stack([vx, vy, vz, ax, ay, az], axis=-1)


def jacobi_constant(X, mu: float):
    X = np.asarray(X, float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - (X[..., 3:6] ** 2).sum(-1)


@dataclass
class HaloFamily:
    mu: float
    x0: np.ndarray            # (M, 6) initial states [x0, 0, z0, 0, vy0, 0]
    period: np.ndarray        # (M,) TU
    jacobi: np.ndarray        # (M,)
    perilune_km: np.ndarray   # (M,)
    stability: np.ndarray     # (M,)
    states: np.ndarray        # (M, N, 6) rotating states at phases k/N
    gen_seconds: float = 0.0

    def __post_init__(self):
        self.derivs = cr3bp_accel(self.states, self.mu)     # (M, N, 6), d/dtau

    @property
    def n_members(self) -> int:
        return len(self.period)

    @property
    def n_phases(self) -> int:
        return self.states.shape[1]

    def member_state(self, m, phase) -> np.ndarray:
        """Cubic Hermite interpolation in phase on integer member(s) m. m and phase broadcast;
        returns (..., 6)."""
        m = np.asarray(m, int)
        N = self.n_phases
        u = np.mod(np.asarray(phase, float), 1.0) * N
        i = np.floor(u).astype(int) % N
        f = (u - np.floor(u))[..., None]
        j = (i + 1) % N
        h = (self.period[m] / N)[..., None]                 # dtau per sample
        p0, p1 = self.states[m, i], self.states[m, j]
        d0, d1 = self.derivs[m, i] * h, self.derivs[m, j] * h
        f2, f3 = f * f, f * f * f
        return ((2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * d0
                + (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * d1)

    def _split(self, s):
        s = np.clip(np.asarray(s, float), 0.0, self.n_members - 1)
        m0 = np.minimum(np.floor(s).astype(int), self.n_members - 2)
        return m0, s - m0

    def period_at(self, s):
        m0, w = self._split(s)
        return (1 - w) * self.period[m0] + w * self.period[m0 + 1]

    def state(self, s, phase) -> np.ndarray:
        """Rotating state on the family at continuous member index s (0..M-1) and normalized
        phase (cycles). s and phase broadcast; returns (..., 6)."""
        m0, w = self._split(s)
        m0, w, phase = np.broadcast_arrays(m0, w, np.asarray(phase, float))
        w = w[..., None]
        return (1 - w) * self.member_state(m0, phase) + w * self.member_state(m0 + 1, phase)

    def initial_state_at(self, s) -> np.ndarray:
        m0, w = self._split(s)
        w = np.asarray(w)[..., None]
        return (1 - w) * self.x0[m0] + w * self.x0[m0 + 1]


# ----------------------------------------------------------------------------- generation

def _continue(x_start, z_targets, mu):
    """Correct a member at each z in z_targets (monotonic, moving away from x_start's z0),
    with a secant predictor. Returns lists of states and periods."""
    xs, Ts, prev, x = [], [], None, np.array(x_start, float)
    for z in z_targets:
        g = x.copy()
        if prev is not None:
            g = x + (x - prev) * (z - x[2]) / (x[2] - prev[2])
        g[2] = z
        xn, T = correct_periodic(g, mu, planar=False)
        prev, x = x, xn
        xs.append(xn)
        Ts.append(T)
    return xs, Ts


def generate(sys: EarthMoonSystem | None = None, z0_min=DEFAULTS["z0_min"], z0_max=DEFAULTS["z0_max"],
             n_members=DEFAULTS["n_members"], n_phases=DEFAULTS["n_phases"],
             period_weight=DEFAULTS["period_weight"], verbose=False) -> HaloFamily:
    sys = sys or EarthMoonSystem()
    mu = sys.mu
    t_start = time.time()
    seed, T_seed = correct_periodic(np.array(SEED_SMALL_HALO), mu, planar=False)

    # Pass 1: coarse continuation in z0 both ways from the seed.
    dz = 0.005
    up = np.arange(seed[2] + dz, z0_max + dz / 2, dz)
    up = np.append(up[up < z0_max - 1e-9], z0_max)
    down = np.arange(seed[2] - dz, z0_min - dz / 2, -dz)
    down = np.append(down[down > z0_min + 1e-9], z0_min)
    xu, Tu = _continue(seed, up, mu)
    xd, Td = _continue(seed, down, mu)
    X1 = np.array(xd[::-1] + [seed] + xu)
    T1 = np.array(Td[::-1] + [T_seed] + Tu)

    # Pass 2: members evenly spaced in arc length of (x0, z0, vy0, w T), corrected again.
    feat = np.column_stack([X1[:, 0], X1[:, 2], X1[:, 4], period_weight * T1])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(feat, axis=0), axis=1))])
    targets = np.linspace(0, arc[-1], n_members)
    z_t = np.interp(targets, arc, X1[:, 2])
    x0s, Ts = [], []
    for z in z_t:
        g = np.array([np.interp(z, X1[:, 2], X1[:, k]) for k in range(6)])
        g[2] = z
        x, T = correct_periodic(g, mu, planar=False)
        x0s.append(x)
        Ts.append(T)
    x0s, Ts = np.array(x0s), np.array(Ts)

    # Phase samples, perilune, Jacobi, stability.
    states = np.empty((n_members, n_phases, 6))
    peri, stab = np.empty(n_members), np.empty(n_members)
    for m, (x, T) in enumerate(zip(x0s, Ts)):
        tt = np.arange(n_phases) / n_phases * T
        sol = solve_ivp(cr3bp_eom, [0, T], x, args=(mu,), t_eval=tt, **TOL)
        states[m] = sol.y.T
        peri[m] = np.linalg.norm(states[m, :, :3] - [1 - mu, 0, 0], axis=1).min() * sys.du_km
        stab[m] = stability_index(x, T, mu)
        if verbose:
            print(f"member {m:3d}: z0 {x[2]:.5f} x0 {x[0]:.5f} T {T:.4f} stab {stab[m]:.1f}")
    return HaloFamily(mu=mu, x0=x0s, period=Ts, jacobi=jacobi_constant(x0s, mu), perilune_km=peri,
                      stability=stab, states=states, gen_seconds=time.time() - t_start)


def _config(sys, **kw):
    cfg = {**DEFAULTS, **kw}
    return np.array([FORMAT_VERSION, sys.mu, sys.du_km, sys.tu_s, cfg["z0_min"], cfg["z0_max"],
                     cfg["n_members"], cfg["n_phases"], cfg["period_weight"], *SEED_SMALL_HALO], float)


def save_family(fam: HaloFamily, path: str, config: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    np.savez(tmp, config=config, mu=fam.mu, x0=fam.x0, period=fam.period, jacobi=fam.jacobi,
             perilune_km=fam.perilune_km, stability=fam.stability, states=fam.states, gen_seconds=fam.gen_seconds)
    os.replace(tmp, path)                     # atomic: parallel workers never see a partial file


def load_family(sys: EarthMoonSystem | None = None, path: str = CACHE_PATH, regenerate: bool = False,
                **kw) -> HaloFamily:
    """The cached family if it matches sys and the generation settings, else generate and save it."""
    sys = sys or EarthMoonSystem()
    config = _config(sys, **kw)
    if not regenerate and os.path.exists(path):
        try:
            d = np.load(path)
            if d["config"].shape == config.shape and np.allclose(d["config"], config, rtol=0, atol=0):
                return HaloFamily(mu=float(d["mu"]), x0=d["x0"], period=d["period"], jacobi=d["jacobi"],
                                  perilune_km=d["perilune_km"], stability=d["stability"], states=d["states"],
                                  gen_seconds=float(d["gen_seconds"]))
        except Exception:
            pass                              # unreadable or stale: regenerate below
    fam = generate(sys, **kw)
    save_family(fam, path, config)
    return fam


if __name__ == "__main__":
    f = load_family(regenerate=True)
    print(f"{f.n_members} members in {f.gen_seconds:.1f} s; z0 {f.x0[0, 2]:.3f}-{f.x0[-1, 2]:.3f}, "
          f"x0 {f.x0[:, 0].min():.4f}-{f.x0[:, 0].max():.4f}, T {f.period.min():.3f}-{f.period.max():.3f} TU")

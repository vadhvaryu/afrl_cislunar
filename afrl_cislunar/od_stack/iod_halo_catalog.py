"""
Catalog-search IOD: fit the observations with a member of the L1 northern halo family.

STRONG PRIOR: this method assumes the object is on, or close to, an L1 NORTHERN halo orbit
(z0 from 0.01 to 0.26 DU at the perpendicular x-z-plane crossing with vy > 0). It can only
return states on that family. For an object on another kind of orbit it returns the
best-fitting halo state anyway (check info["rms_normalized"]) or raises IODFailure if even
that fits worse than max_rms_arcsec. It is meant to get batch least squares within its convergence
basin (measured: ~10,000 km / 100 m/s), not to be an answer by itself.

How it works (no per-candidate propagation; never reads the truth):
    1. Family: od_stack.halo_family.load_family (seeded from the cislunar-custody small halo,
       continued in z0; cached in results/halo_catalog/L1N.npz).
    2. A periodic orbit repeats in the rotating frame, so candidate (member s, phase psi at the
       epoch) has rotating state orbit_s(psi + (tau - tau_e) / T_s) at time t. Rotate to EMI
       (R(tau) (p - Earth) DU), subtract the observer, compute RA/Dec, and score by the RMS of
       the RA/Dec residuals in sigma (RA wrapped, not scaled by cos Dec: same convention as
       results.residuals). Grid: every member x every stored phase (100 x 1,000) at once.
    3. The best local minima of that grid (distinct (member, phase) cells) are refined with
       scipy least_squares over continuous (s, psi), interpolating between neighbouring members.
    4. The best refined candidate is returned at the epoch = the middle observation's time
       (index len(obs) // 2), converted to EMI SI with frames.rotating_to_emi.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares

from .contracts import Observations, StateEstimate
from .frames import EarthMoonSystem, rotating_to_emi
from .halo_family import CACHE_PATH, HaloFamily, load_family
from .iod import IODFailure


ARCSEC = np.pi / (180 * 3600)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class HaloCatalogIOD:
    """Catalog search over the L1 northern halo family (see the module docstring for the prior).

    n_refine     number of distinct grid minima refined with least_squares
    top_k        number of distinct refined candidates reported in info["candidates"]
    max_rms_arcsec  raise IODFailure if the best candidate's RMS angle residual is above this
                 (default 2 deg: at lunar distance that is ~13,000 km across the line of sight,
                 i.e. outside the measured BLS basin, so nothing on the family "fits"). An
                 absolute angle, not sigma, so the gate does not depend on the noise level.
    family       a HaloFamily to use instead of the cached one (tests)
    """

    name = "halo_catalog"

    def __init__(self, sys: EarthMoonSystem, n_refine: int = 8, top_k: int = 5, max_rms_arcsec: float = 7200.0,
                 catalog_path: str | None = None, family: HaloFamily | None = None, **_):
        # **_ swallows orekit_data= and truth_path=; truth_path is deliberately ignored.
        self.sys = sys
        self.n_refine, self.top_k, self.max_rms_arcsec = int(n_refine), int(top_k), float(max_rms_arcsec)
        self.family = family if family is not None else load_family(sys, path=catalog_path or CACHE_PATH)

    # ------------------------------------------------------------------ residuals

    def _setup(self, obs: Observations):
        sys = self.sys
        k = len(obs) // 2
        tau = sys.tau(obs.t)
        c, s = np.cos(tau), np.sin(tau)
        sig = np.where(obs.sigma > 0, obs.sigma, np.nan)
        return dict(k=k, tau=tau, dtau=tau - tau[k], c=c, s=s, obs=obs, sig=sig,
                    ra=np.asarray(obs.ra, float), dec=np.asarray(obs.dec, float))

    def _norm_residuals(self, P, g):
        """Rotating positions P (..., N, 3) at the observation times -> residuals in sigma (..., 2N)."""
        sys = self.sys
        rho = (P - sys.earth_rot) * sys.du_m
        d = np.stack([g["c"] * rho[..., 0] - g["s"] * rho[..., 1],
                      g["s"] * rho[..., 0] + g["c"] * rho[..., 1], rho[..., 2]], axis=-1) - g["obs"].observer
        ra = np.arctan2(d[..., 1], d[..., 0])
        dec = np.arcsin(np.clip(d[..., 2] / np.linalg.norm(d, axis=-1), -1, 1))
        return np.concatenate([_wrap(g["ra"] - ra) / g["sig"], (g["dec"] - dec) / g["sig"]], axis=-1)

    def _grid(self, g):
        """RMS (sigma) for every stored member x stored epoch phase: (M, P)."""
        fam = self.family
        M, P = fam.n_members, fam.n_phases
        m = np.arange(M)[:, None, None]
        phase = (np.arange(P)[None, :, None] / P) + g["dtau"][None, None, :] / fam.period[:, None, None]
        pos = fam.member_state(np.broadcast_to(m, phase.shape), phase)[..., :3]   # (M, P, N, 3)
        r = self._norm_residuals(pos, g)
        return np.sqrt(np.nanmean(r**2, axis=-1))

    def _resid(self, p, g):
        s, psi = p
        fam = self.family
        T = fam.period_at(s)
        X = fam.state(s, psi + g["dtau"] / T)
        return self._norm_residuals(X[:, :3], g)

    # ------------------------------------------------------------------ search

    def _seeds(self, rms):
        """Distinct grid cells: local minima of the RMS grid (periodic in phase), best first."""
        local = minimum_filter(rms, size=(5, 21), mode=("nearest", "wrap"))
        mi, pi = np.nonzero(rms <= local)
        order = np.argsort(rms[mi, pi])
        out = []
        P = rms.shape[1]
        for j in order:
            m, p = int(mi[j]), int(pi[j])
            if all(abs(m - a) > 2 or min(abs(p - b), P - abs(p - b)) > 10 for a, b, _ in out):
                out.append((m, p, float(rms[m, p])))
            if len(out) >= self.n_refine:
                break
        return out

    def estimate(self, obs: Observations) -> StateEstimate:
        t_start = time.time()
        if len(obs) < 2:
            raise IODFailure(f"{self.name}: needs at least 2 observations")
        fam, sys = self.family, self.sys
        g = self._setup(obs)
        rms_grid = self._grid(g)
        if not np.isfinite(rms_grid).any():
            raise IODFailure(f"{self.name}: no finite residuals (check sigma)")
        seeds = self._seeds(np.where(np.isfinite(rms_grid), rms_grid, np.inf))

        refined = []
        for m, p, rms0 in seeds:
            x0 = np.array([float(m), p / fam.n_phases])
            try:
                sol = least_squares(self._resid, x0, args=(g,), bounds=([0.0, -np.inf], [fam.n_members - 1.0, np.inf]),
                                    x_scale=[1.0, 1e-3], diff_step=1e-7, xtol=1e-12, ftol=1e-12, max_nfev=200)
                s, psi = float(sol.x[0]), float(sol.x[1] % 1.0)
                rms = float(np.sqrt(np.nanmean(self._resid((s, psi), g) ** 2)))
            except (ValueError, FloatingPointError):
                s, psi, rms = float(m), p / fam.n_phases, rms0
            refined.append(dict(member=s, phase=psi, rms_normalized=rms, grid_member=m,
                                grid_phase=p / fam.n_phases, grid_rms=rms0))

        refined.sort(key=lambda c: c["rms_normalized"])
        distinct = []
        for c in refined:
            if all(abs(c["member"] - d["member"]) > 0.5 or
                   abs(_wrap(2 * np.pi * (c["phase"] - d["phase"]))) > 2 * np.pi * 2e-3 for d in distinct):
                distinct.append(c)
        best = distinct[0]
        for c in distinct:
            r_sig = self._resid((c["member"], c["phase"]), g)
            c["rms_arcsec"] = float(np.sqrt(np.nanmean((r_sig * np.tile(g["sig"], 2)) ** 2)) / ARCSEC)
        if not best["rms_arcsec"] <= self.max_rms_arcsec:
            raise IODFailure(f"{self.name}: best L1 northern halo fit has RMS {best['rms_arcsec']:.3g} arcsec "
                             f"({best['rms_normalized']:.3g} sigma) > max_rms_arcsec {self.max_rms_arcsec:g}")

        for c in distinct:
            x_init = fam.initial_state_at(c["member"])
            c.update(z0=float(x_init[2]), x0=float(x_init[0]), period_TU=float(fam.period_at(c["member"])))
        k = g["k"]
        x_rot = fam.state(best["member"], best["phase"])
        r, v = rotating_to_emi(float(g["tau"][k]), x_rot, sys)
        info = {"obs_indices": [int(k)], "epoch_index": int(k), "prior": "L1 northern halo family",
                "member": best["member"], "phase": best["phase"], "z0": best["z0"], "x0": best["x0"],
                "period_TU": best["period_TU"], "rms_normalized": best["rms_normalized"],
                "rms_arcsec": best["rms_arcsec"],
                "state_rotating": [float(a) for a in x_rot],
                "candidates": distinct[:self.top_k], "n_members": int(fam.n_members),
                "n_phases": int(fam.n_phases), "runtime_s": time.time() - t_start}
        return StateEstimate(t=float(obs.t[k]), r=r, v=v, method=self.name, info=info)

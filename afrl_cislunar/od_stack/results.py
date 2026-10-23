"""
Results stage: grade, describe and plot whatever the earlier stages produced.

Inputs come only through the contracts (contracts.py):
    Observations   from the data source       (always)
    StateEstimate  from IOD and estimation    (always; covariance optional)
    Trajectory     from the propagator        (always)
    Trajectory     truth                      (optional: synthetic data only)
plus the propagator object itself, reused to carry the covariance forward in time.

Two kinds of output, so the same code works on synthetic and real data:
    needs no truth   post-fit residuals, covariance over time, confidence regions
    needs truth      error vs truth, NEES / chi-square consistency, error-inside-sigma checks

Files written to output.dir (see `report`):
    summary.json               every number below, including the 6x6 covariance
    trajectory_comparison.csv  estimate (+ truth, error, 1-sigma) at dense times
    residuals.csv              observed - computed RA/Dec at each observation
    results.png                trajectory (rotating frame) + error vs truth
    residuals.png              post-fit residuals with 1/3-sigma bands
    uncertainty.png            error vs covariance envelope, NEES, epoch confidence ellipses
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # write files only; works headless and in Docker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from .contracts import Observations, StateEstimate, Trajectory
from .frames import EarthMoonSystem, rot_z, trajectory_to_rotating
from .uncertainty import (confidence_ellipsoid, confidence_summary, consistency, nees_series,
                          propagate_covariance, sigma_position_m)

DAY = 86400.0
ARCSEC = np.pi / (180 * 3600)


# ----------------------------------------------------------------------------- numbers

def position_error_km(est: Trajectory, truth: Trajectory) -> np.ndarray:
    if not np.allclose(est.t, truth.t):
        raise ValueError("Trajectories must be sampled at the same times")
    return np.linalg.norm(est.r - truth.r, axis=1) / 1e3


def residuals(obs: Observations, est_at_obs: Trajectory) -> dict:
    """Observed minus computed angles at each observation time (no truth needed).

    RA residuals are plain differences in RA (not multiplied by cos(Dec)), because that
    is what the generator adds noise to and what Orekit's AngularRaDec fits.
    """
    if not np.allclose(obs.t, est_at_obs.t):
        raise ValueError("est_at_obs must be sampled at the observation times")
    d = est_at_obs.r - obs.observer
    ra_c = np.arctan2(d[:, 1], d[:, 0])
    dec_c = np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1))
    dra = (obs.ra - ra_c + np.pi) % (2 * np.pi) - np.pi
    ddec = obs.dec - dec_c
    sig = np.where(obs.sigma > 0, obs.sigma, np.nan)
    norm = np.concatenate([dra / sig, ddec / sig])
    return {
        "t": obs.t,
        "ra_arcsec": dra / ARCSEC,
        "dec_arcsec": ddec / ARCSEC,
        "sigma_arcsec": obs.sigma / ARCSEC,
        "rms_normalized": float(np.sqrt(np.nanmean(norm**2))),
        "max_abs_normalized": float(np.nanmax(np.abs(norm))),
    }


def summarize(x_iod: StateEstimate, x0: StateEstimate, truth_at_x0: Trajectory, est: Trajectory,
              truth: Trajectory, est_at_obs: Trajectory, truth_at_obs: Trajectory) -> dict:
    """Errors vs truth. x_iod: IOD output. x0: state after estimation (same epoch)."""
    err = position_error_km(est, truth)
    err_obs = position_error_km(est_at_obs, truth_at_obs)
    s = {
        "method": x0.method,
        "epoch_days": x0.t / DAY,
        "iod_position_error_km": float(np.linalg.norm(x_iod.r - truth_at_x0.r[0]) / 1e3),
        "iod_velocity_error_m_s": float(np.linalg.norm(x_iod.v - truth_at_x0.v[0])),
        "position_error_km": float(np.linalg.norm(x0.r - truth_at_x0.r[0]) / 1e3),
        "velocity_error_m_s": float(np.linalg.norm(x0.v - truth_at_x0.v[0])),
        "error_at_obs_times_km": [round(float(e), 3) for e in err_obs],
        "max_position_error_km": float(err.max()),
        "rms_position_error_km": float(np.sqrt(np.mean(err**2))),
    }
    if x0.cov is not None:
        # 1-sigma sizes from the covariance (square root of the trace of each 3x3 block)
        s["sigma_position_km"] = float(np.sqrt(np.trace(x0.cov[:3, :3])) / 1e3)
        s["sigma_velocity_m_s"] = float(np.sqrt(np.trace(x0.cov[3:, 3:])))
        s["covariance_6x6_SI"] = x0.cov.tolist()
    s["info"] = x0.info
    return s


def report(obs: Observations, x_iod: StateEstimate, x0: StateEstimate, est: Trajectory,
           est_at_obs: Trajectory, propagator, sys: EarthMoonSystem, truth: Trajectory | None = None,
           truth_at_obs: Trajectory | None = None, truth_at_x0: Trajectory | None = None,
           out_dir: str | None = None, propagate_cov: bool = True, confidence_level: float = 0.95) -> dict:
    """The whole results stage in one call. Returns the summary dict; writes files if out_dir."""
    if truth is not None:
        s = summarize(x_iod, x0, truth_at_x0, est, truth, est_at_obs, truth_at_obs)
    else:
        s = {"method": x0.method, "epoch_days": x0.t / DAY, "info": x0.info}
        if x0.cov is not None:
            s["sigma_position_km"] = float(np.sqrt(np.trace(x0.cov[:3, :3])) / 1e3)
            s["sigma_velocity_m_s"] = float(np.sqrt(np.trace(x0.cov[3:, 3:])))
            s["covariance_6x6_SI"] = x0.cov.tolist()

    res = residuals(obs, est_at_obs)
    s["residuals"] = {"rms_normalized": res["rms_normalized"], "max_abs_normalized": res["max_abs_normalized"],
                      "ra_arcsec": [round(float(x), 3) for x in res["ra_arcsec"]],
                      "dec_arcsec": [round(float(x), 3) for x in res["dec_arcsec"]]}

    P_t = None
    if x0.cov is not None:
        u = {"confidence_region_at_epoch": confidence_summary(x0.cov, confidence_level)}
        if propagate_cov:
            P_t = propagate_covariance(x0, est.t, propagator)
            sig_t = sigma_position_m(P_t) / 1e3
            u["sigma_position_km_over_span"] = {"min": float(sig_t.min()), "max": float(sig_t.max()),
                                               "at_last_obs": float(np.interp(obs.t.max(), est.t, sig_t))}
        if truth is not None:
            u["consistency_at_epoch"] = consistency(x0, truth_at_x0.r[0], truth_at_x0.v[0])
            if P_t is not None:
                ns = nees_series(est, truth, P_t)
                u["fraction_of_span_inside_95pct_ellipsoid"] = float(np.mean(ns <= chi2.ppf(0.95, 3)))
                u["fraction_of_span_error_below_3sigma"] = float(
                    np.mean(position_error_km(est, truth) <= 3 * sigma_position_m(P_t) / 1e3))
        s["uncertainty"] = u

    if out_dir:
        write_outputs(out_dir, s, est, truth, P_t)
        write_residuals(out_dir, res)
        s["figures"] = [plot(out_dir, obs, x0, est, truth, sys),
                        plot_residuals(out_dir, res, x0)]
        if x0.cov is not None:
            s["figures"].append(plot_uncertainty(out_dir, obs, x0, est, truth, truth_at_x0, P_t, sys))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:   # again, now with figure paths
            json.dump(s, f, indent=2)
    return s


# ----------------------------------------------------------------------------- files

def write_outputs(out_dir: str, summary: dict, est: Trajectory, truth: Trajectory | None = None,
                  P_t: np.ndarray | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    head = ["time_days", "est_x_km", "est_y_km", "est_z_km"]
    if truth is not None:
        head += ["truth_x_km", "truth_y_km", "truth_z_km", "position_error_km"]
        err = position_error_km(est, truth)
    if P_t is not None:
        head += ["sigma_position_km"]
        sig = sigma_position_m(P_t) / 1e3
    with open(os.path.join(out_dir, "trajectory_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i in range(len(est.t)):
            row = [f"{est.t[i] / DAY:.6f}", *[f"{v:.3f}" for v in est.r[i] / 1e3]]
            if truth is not None:
                row += [*[f"{v:.3f}" for v in truth.r[i] / 1e3], f"{err[i]:.3f}"]
            if P_t is not None:
                row += [f"{sig[i]:.3f}"]
            w.writerow(row)


def write_residuals(out_dir: str, res: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "residuals.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_id", "time_days", "ra_residual_arcsec", "dec_residual_arcsec", "sigma_arcsec",
                    "ra_residual_sigma", "dec_residual_sigma"])
        for i, t in enumerate(res["t"]):
            s = res["sigma_arcsec"][i]
            w.writerow([i, f"{t / DAY:.6f}", f"{res['ra_arcsec'][i]:.4f}", f"{res['dec_arcsec'][i]:.4f}",
                        f"{s:.4f}", f"{res['ra_arcsec'][i] / s:.3f}" if s else "",
                        f"{res['dec_arcsec'][i] / s:.3f}" if s else ""])


# ----------------------------------------------------------------------------- figures

def _obs_lines(ax, obs, used):
    for k in range(len(obs.t)):
        ax.axvline(obs.t[k] / DAY, color="tab:red" if k in used else "0.8", lw=0.8, ls="--", zorder=0)


def plot(out_dir: str, obs: Observations, x0: StateEstimate, est: Trajectory, truth: Trajectory | None,
         sys: EarthMoonSystem) -> str:
    """Trajectory in the rotating frame (top and side view) + error vs truth."""
    est_rot = trajectory_to_rotating(est, sys)
    ref_rot = trajectory_to_rotating(truth, sys) if truth is not None else est_rot
    ref = truth if truth is not None else est
    obs_pts = np.array([np.interp(obs.t, ref.t, ref_rot[:, k]) for k in range(3)]).T
    used = x0.info.get("obs_indices", [])

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    for ax, (a, b, la, lb) in zip(axs[:2], [(0, 1, "x", "y"), (0, 2, "x", "z")]):
        if truth is not None:
            ax.plot(ref_rot[:, a], ref_rot[:, b], "k", lw=1.6, label="Truth")
        ax.plot(est_rot[:, a], est_rot[:, b], color="tab:orange", lw=1.4, label=est.label)
        ax.plot(obs_pts[:, a], obs_pts[:, b], "s", mfc="none", mec="0.4", ms=6, label="Observation times")
        if used:
            ax.plot(obs_pts[used, a], obs_pts[used, b], "s", color="tab:red", ms=6, label="Used by IOD")
        ax.plot(1 - sys.mu, 0, "o", color="gray", ms=7, label="Moon")
        # Zoom on the reference orbit and the Moon; a badly wrong estimate may leave the view.
        pts = np.vstack([ref_rot[:, [a, b]], [[1 - sys.mu, 0.0]]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c, half = (lo + hi) / 2, max(hi - lo) * 0.75
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} (DU, rotating frame; Earth is at x = {-sys.mu:.3f})")
        ax.set_ylabel(f"{lb} (DU, rotating frame)")
        ax.grid(alpha=0.3)
    axs[0].set_title("Trajectory, top view")
    axs[1].set_title("Trajectory, side view")
    handles, labels = axs[0].get_legend_handles_labels()

    ax = axs[2]
    if truth is not None:
        ax.semilogy(est.t / DAY, np.maximum(position_error_km(est, truth), 1e-3), color="tab:orange", lw=1.6)
        ax.set_ylabel("Position error (km)")
        ax.set_title("Estimate vs truth")
    else:
        ax.text(0.5, 0.5, "No truth file:\nsee residuals.png\nand uncertainty.png", ha="center", va="center",
                transform=ax.transAxes, color="0.4")
        ax.set_title("Estimate vs truth (n/a)")
    _obs_lines(ax, obs, used)
    ax.set_xlabel("Days since first observation")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"{x0.method.replace(' + ', ' → ')} → {est.label.split(' + ')[-1]}", fontsize=13)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residuals(out_dir: str, res: dict, x0: StateEstimate) -> str:
    """Post-fit O-C residuals per angle, with the 1- and 3-sigma noise bands."""
    t = res["t"] / DAY
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110, sharey=True)
    for ax, key, name in zip(axs, ("ra_arcsec", "dec_arcsec"), ("Right ascension", "Declination")):
        s = res["sigma_arcsec"]
        ax.fill_between(t, -3 * s, 3 * s, color="tab:blue", alpha=0.08, step="mid", label="±3σ noise")
        ax.fill_between(t, -s, s, color="tab:blue", alpha=0.18, step="mid", label="±1σ noise")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.plot(t, res[key], "o", color="tab:orange", ms=7, label="Residual (observed − computed)")
        ax.set_title(f"{name} residuals")
        ax.set_xlabel("Days since first observation")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("arcsec")
    lim = max(3.5 * np.nanmax(res["sigma_arcsec"]), 1.2 * np.nanmax(np.abs(np.r_[res["ra_arcsec"], res["dec_arcsec"]])))
    axs[0].set_ylim(-lim, lim)
    handles, labels = axs[0].get_legend_handles_labels()
    n = 2 * len(t)
    note = (f"expected ≈ {np.sqrt(max(n - 6, 0) / n):.2f}σ for {n} angles and 6 fitted parameters"
            if x0.cov is not None else "≈1σ means the model matches the noise")
    fig.suptitle(f"Post-fit residuals: {x0.method.replace(' + ', ' → ')}   "
                 f"(RMS = {res['rms_normalized']:.2f}σ; {note})", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out_dir, "residuals.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _ellipse(ax, P2, level, **kw):
    a, V = confidence_ellipsoid(P2, level)
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    ax.add_patch(Ellipse((0, 0), 2 * a[0], 2 * a[1], angle=ang, fill=False, **kw))
    return a[0]


def plot_uncertainty(out_dir: str, obs: Observations, x0: StateEstimate, est: Trajectory,
                     truth: Trajectory | None, truth_at_x0: Trajectory | None, P_t: np.ndarray | None,
                     sys: EarthMoonSystem) -> str:
    """(a) error vs covariance envelope over time, (b) NEES over time (or per-axis sigma without truth),
    (c, d) confidence ellipses at the epoch, in rotating-frame axes, centered on the estimate."""
    used = x0.info.get("obs_indices", [])
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=110)
    d = est.t / DAY

    ax = axs[0, 0]
    lows = []
    if P_t is not None:
        sig = sigma_position_m(P_t) / 1e3
        ax.fill_between(d, sig, 3 * sig, color="tab:blue", alpha=0.12, label="1σ to 3σ band")
        ax.plot(d, sig, color="tab:blue", lw=1.2, label="1σ (√trace of position covariance)")
        ax.plot(d, 3 * sig, color="tab:blue", lw=0.8, ls="--", label="3σ")
        lows.append(sig.min())
    if truth is not None:
        err = np.maximum(position_error_km(est, truth), 1e-3)
        ax.plot(d, err, color="tab:orange", lw=1.6, label="True error")
        lows.append(np.percentile(err, 2))
    ax.set_yscale("log")
    if lows:
        ax.set_ylim(bottom=0.3 * min(lows))
    _obs_lines(ax, obs, used)
    ax.set_xlabel("Days since first observation")
    ax.set_ylabel("Position (km)")
    ax.set_title("Error vs reported uncertainty over time")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    ax = axs[0, 1]
    if truth is not None and P_t is not None:
        ns = nees_series(est, truth, P_t)
        ax.semilogy(d, np.maximum(ns, 1e-4), color="tab:purple", lw=1.4, label="Position NEES")
        for lvl, ls in ((0.95, "--"), (0.9973, ":")):
            ax.axhline(chi2.ppf(lvl, 3), color="0.3", ls=ls, lw=1, label=f"χ²(3) {lvl:.2%} bound")
        ax.axhline(3, color="0.6", lw=0.8, label="Expected value (3)")
        ax.set_ylabel("eᵀ P⁻¹ e")
        ax.set_title("Is the covariance honest? (below the bounds = yes)")
    elif P_t is not None:
        R = np.array([rot_z(sys.tau(t)).T for t in est.t])
        Prot = np.einsum("mij,mjk,mlk->mil", R, P_t[:, :3, :3], R)
        for k, n in enumerate("xyz"):
            ax.semilogy(d, np.sqrt(Prot[:, k, k]) / 1e3, lw=1.4, label=f"1σ {n} (rotating axes)")
        ax.set_ylabel("km")
        ax.set_title("Uncertainty per axis")
    _obs_lines(ax, obs, used)
    ax.set_xlabel("Days since first observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    Rt = rot_z(sys.tau(x0.t)).T
    Pr = Rt @ x0.cov[:3, :3] @ Rt.T / 1e6              # km^2, rotating-frame axes
    off = Rt @ (truth_at_x0.r[0] - x0.r) / 1e3 if truth_at_x0 is not None else None
    for ax, (a, b) in zip(axs[1], [(0, 1), (0, 2)]):
        P2 = Pr[np.ix_([a, b], [a, b])]
        big = 0
        for lvl, alpha in ((0.6827, 0.9), (0.95, 0.6), (0.9973, 0.35)):
            big = max(big, _ellipse(ax, P2, lvl, color="tab:blue", alpha=alpha, lw=1.5, label=f"{lvl:.1%} region"))
        ax.plot(0, 0, "+", color="tab:orange", ms=12, mew=2, label="Estimate")
        if off is not None:
            ax.plot(off[a], off[b], "*", color="k", ms=12, label="Truth")
            big = max(big, abs(off[a]), abs(off[b]))
        ax.set_xlim(-1.15 * big, 1.15 * big)
        ax.set_ylim(-1.15 * big, 1.15 * big)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Δ{'xyz'[a]} (km, rotating axes)")
        ax.set_ylabel(f"Δ{'xyz'[b]} (km, rotating axes)")
        ax.set_title(f"Confidence regions at epoch (day {x0.t / DAY:.1f}), {'xyz'[a]}–{'xyz'[b]}")
        ax.grid(alpha=0.3)
    axs[1, 0].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Uncertainty: {x0.method.replace(' + ', ' → ')}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "uncertainty.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

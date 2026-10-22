"""Results: grade an estimated trajectory against the truth, write files, make plots."""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # write files only; works headless and in Docker
import matplotlib.pyplot as plt
import numpy as np

from .contracts import Observations, StateEstimate, Trajectory
from .frames import EarthMoonSystem, trajectory_to_rotating

DAY = 86400.0


def position_error_km(est: Trajectory, truth: Trajectory) -> np.ndarray:
    if not np.allclose(est.t, truth.t):
        raise ValueError("Trajectories must be sampled at the same times")
    return np.linalg.norm(est.r - truth.r, axis=1) / 1e3


def summarize(x_iod: StateEstimate, x0: StateEstimate, truth_at_x0: Trajectory, est: Trajectory,
              truth: Trajectory, est_at_obs: Trajectory, truth_at_obs: Trajectory) -> dict:
    """x_iod: IOD output. x0: state after estimation (same epoch). Both graded against truth."""
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


def write_outputs(out_dir: str, summary: dict, est: Trajectory, truth: Trajectory) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    err = position_error_km(est, truth)
    with open(os.path.join(out_dir, "trajectory_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_days", "est_x_km", "est_y_km", "est_z_km", "truth_x_km", "truth_y_km",
                    "truth_z_km", "position_error_km"])
        for i in range(len(est.t)):
            w.writerow([f"{est.t[i] / DAY:.6f}", *[f"{v:.3f}" for v in est.r[i] / 1e3],
                        *[f"{v:.3f}" for v in truth.r[i] / 1e3], f"{err[i]:.3f}"])


def plot(out_dir: str, obs: Observations, x0: StateEstimate, est: Trajectory, truth: Trajectory,
         sys: EarthMoonSystem) -> str:
    est_rot = trajectory_to_rotating(est, sys)
    tru_rot = trajectory_to_rotating(truth, sys)
    obs_tru = np.array([np.interp(obs.t, truth.t, tru_rot[:, k]) for k in range(3)]).T
    used = x0.info.get("obs_indices", [])
    err = position_error_km(est, truth)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), dpi=110)
    for ax, (a, b, la, lb) in zip(axs[:2], [(0, 1, "x", "y"), (0, 2, "x", "z")]):
        ax.plot(tru_rot[:, a], tru_rot[:, b], "k", lw=1.6, label="Truth")
        ax.plot(est_rot[:, a], est_rot[:, b], color="tab:orange", lw=1.4, label=est.label)
        ax.plot(obs_tru[:, a], obs_tru[:, b], "s", mfc="none", mec="0.4", ms=6, label="Observation times")
        if used:
            ax.plot(obs_tru[used, a], obs_tru[used, b], "s", color="tab:red", ms=6, label="Used by IOD")
        ax.plot(1 - sys.mu, 0, "o", color="gray", ms=7, label="Moon")
        # Zoom on the truth orbit and the Moon; a badly wrong estimate may leave the view.
        pts = np.vstack([tru_rot[:, [a, b]], [[1 - sys.mu, 0.0]]])
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
    ax.semilogy(est.t / DAY, np.maximum(err, 1e-3), color="tab:orange", lw=1.6)
    for k in range(len(obs.t)):
        ax.axvline(obs.t[k] / DAY, color="tab:red" if k in used else "0.8", lw=0.8, ls="--")
    ax.set_xlabel("Days since first observation")
    ax.set_ylabel("Position error (km)")
    ax.set_title("Estimate vs truth")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"{x0.method.replace(' + ', ' → ')} → {est.label.split(' + ')[-1]}", fontsize=13)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

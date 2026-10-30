"""
Convergence study (Part A): at what starting error does batch least squares stop
converging, and why?

    python -m od_stack.convergence pilot              # directions 0-2 on the grid + 2 bisections
    python -m od_stack.convergence grid               # A2: fine grid + noise robustness check
    python -m od_stack.convergence thresholds         # A3: bisected threshold per direction
    python -m od_stack.convergence grid --workers 3 --dirs 0 1 2
    python -m od_stack.convergence grid --estimation orekit_bls_robust    # same grid, other estimator

Every case uses the configured stack (config.yaml: truth_plus_error -> orekit_bls ->
orekit_cr3bp) with the stand-in IOD's error set explicitly:

    direction id k   default_rng(k): first normal(size=3) draw = position error direction,
                     second = velocity error direction (exactly what the stand-in does with
                     seed=k, so ids 0-19 are the old sweep's directions)
    modes            combined       position d km, velocity 0.01*d m/s
                     position_only  position d km, velocity 0
                     velocity_only  position 0,    velocity w m/s
    noise            10", one fixed realization obs_seed = 10_000 for every case (so only the
                     IOD error changes), except the robustness check (obs seeds 10_000-10_004)
    outcome          gated: a finished fit with residual RMS > 3 sigma counts as failed
                     (outcome_ungated keeps the plain category)

Outputs in results/convergence/<estimator>/ (resumable; rerun the same command to continue):
    grid.csv               one row per run (A2), with every variable from convergence_features
    thresholds.csv         one row per (mode, direction): threshold, smallest failure, evaluations
    thresholds_runs.csv    every run made inside the bisections, with all variables
    config_used.yaml       the exact configuration the runs used
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys as _sys

import numpy as np
import yaml

from . import parallel
from .convergence_features import outcome_category
from .run import load_config

OUT_DIR = "results/convergence"      # + /<estimator>, e.g. results/convergence/orekit_bls
GATE_RMS = 3.0                        # a finished fit with residual RMS above this (in sigma) counts as failed
NOISE_ARCSEC = 10.0
OBS_SEED = 10_000
VEL_PER_KM = 0.01                     # combined mode: m/s of velocity error per km of position error

GRID_KM = ([1_000, 5_000, 7_500] + list(range(10_000, 15_001, 500)) + list(range(16_000, 30_001, 1_000))
           + list(range(40_000, 100_001, 10_000)))                                       # 36 sizes
GRID_M_S = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1_000]     # 15 sizes
GRID_DIRS = range(0, 50)
ROBUST_DIRS, ROBUST_SEEDS, ROBUST_KM = range(0, 20), range(10_000, 10_005), (15_000, 30_000)
THRESH_DIRS = range(1_000, 1_200)
THRESH_LO, THRESH_MID, THRESH_HI, THRESH_BRACKET = 500.0, 5_000.0, 100_000.0, 250.0

GRID_KEY = ("mode", "size", "direction", "obs_seed")
THRESH_KEY = ("mode", "direction", "obs_seed")


def directions(k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(k))
    a, b = rng.normal(size=3), rng.normal(size=3)
    return a / np.linalg.norm(a), b / np.linalg.norm(b)


def errors_for(mode: str, size: float) -> tuple[float, float]:
    """(position error km, velocity error m/s) for a mode and size (km, or m/s for velocity_only)."""
    if mode == "combined":
        return float(size), VEL_PER_KM * float(size)
    if mode == "position_only":
        return float(size), 0.0
    if mode == "velocity_only":
        return 0.0, float(size)
    raise ValueError(f"unknown mode {mode!r}")


# ----------------------------------------------------------------------------- worker side

def init_worker(cfg: dict) -> None:
    parallel.init_sweeper(cfg)
    from .convergence_features import FeatureCalculator

    sw = parallel.WORKER["sweeper"]
    parallel.WORKER["features"] = FeatureCalculator(sw.sys, sw.truth_path)


def run_case(mode: str, size: float, k: int, obs_seed: int, sw=None, fc=None,
             noise_arcsec: float = NOISE_ARCSEC) -> dict:
    sw = sw or parallel.WORKER["sweeper"]
    fc = fc or parallel.WORKER["features"]
    d_km, w = errors_for(mode, size)
    u_r, u_v = directions(k)
    row = {"mode": mode, "size": float(size), "size_unit": "m/s" if mode == "velocity_only" else "km",
           "direction": int(k), "obs_seed": int(obs_seed), "position_error_km_set": d_km,
           "velocity_error_m_s_set": w,
           **{f"pos_dir_{c}": float(x) for c, x in zip("xyz", u_r)},
           **{f"vel_dir_{c}": float(x) for c, x in zip("xyz", u_v)}}
    ov = {"method": "truth_plus_error", "position_error_km": d_km, "velocity_error_m_s": w, "seed": int(k),
          "position_direction": u_r, "velocity_direction": u_v}
    out = sw.one(float(noise_arcsec), int(k), ov, obs_seed=int(obs_seed), features=fc)
    out.pop("seed", None)                         # = direction
    row.update(out)
    row["estimator"] = sw.estimator.name
    row["outcome_ungated"] = outcome_category(row)
    row["outcome"] = gated_outcome(row)
    return row


def gated_outcome(row: dict) -> str:
    """Outcome with the residual gate: a finished fit whose residual RMS is above GATE_RMS
    sigma is a failure ('wrong_orbit_rejected'), or a false rejection if it was in fact
    close to the truth ('good_fit_rejected'). Other outcomes are unchanged."""
    ungated = outcome_category(row)
    rms = row.get("rms_normalized_residual")
    if row.get("bls_finished") and rms is not None and not rms <= GATE_RMS:
        return "good_fit_rejected" if ungated == "converged" else "wrong_orbit_rejected"
    return ungated


def grid_job(job: dict) -> dict:
    return run_case(job["mode"], job["size"], job["direction"], job["obs_seed"])


def threshold_job(job: dict):
    """Bisect the largest converging size for one direction. Assumes one crossing; the
    evaluated sizes are all stored so non-monotonic directions can be spotted."""
    mode, k, seed = job["mode"], int(job["direction"]), int(job["obs_seed"])
    runs = []

    def ok(d):
        r = run_case(mode, d, k, seed)
        runs.append(r)
        return r["outcome"] == "converged"

    note = ""
    if ok(THRESH_MID):
        if ok(THRESH_HI):
            lo, hi, note = THRESH_HI, float("nan"), ">= 100,000"
        else:
            lo, hi = THRESH_MID, THRESH_HI
    elif ok(THRESH_LO):
        lo, hi = THRESH_LO, THRESH_MID
    else:
        lo, hi, note = float("nan"), THRESH_LO, "< 500"
    while not note and hi - lo > THRESH_BRACKET:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    evaluated = [(r["size"], r["outcome"]) for r in runs]
    conv = sorted(d for d, o in evaluated if o == "converged")
    fail = sorted(d for d, o in evaluated if o != "converged")
    summary = {"mode": mode, "direction": k, "obs_seed": seed,
               **{key: runs[0][key] for key in runs[0] if key.startswith(("pos_dir_", "vel_dir_"))},
               "threshold": lo, "smallest_failed": hi, "censored": note,
               "monotonic_in_evaluations": not (conv and fail and min(fail) < max(conv)),
               "n_runs": len(runs), "evaluated": json.dumps(evaluated),
               "seconds": float(sum(r.get("seconds", 0) or 0 for r in runs))}
    for i, r in enumerate(runs):
        r["bisection_step"] = i
    return summary, runs


# ----------------------------------------------------------------------------- job lists

def grid_jobs(dirs=GRID_DIRS) -> list[dict]:
    jobs = []
    for mode, sizes in (("combined", GRID_KM), ("position_only", GRID_KM), ("velocity_only", GRID_M_S)):
        jobs += [{"mode": mode, "size": float(s), "direction": k, "obs_seed": OBS_SEED} for s in sizes for k in dirs]
    return jobs


def robustness_jobs() -> list[dict]:
    return [{"mode": "combined", "size": float(d), "direction": k, "obs_seed": s}
            for d in ROBUST_KM for k in ROBUST_DIRS for s in ROBUST_SEEDS]


def threshold_jobs(dirs=THRESH_DIRS, modes=("combined", "position_only")) -> list[dict]:
    return [{"mode": m, "direction": k, "obs_seed": OBS_SEED} for m in modes for k in dirs]


def pilot_jobs() -> tuple[list[dict], list[dict]]:
    grid = [{"mode": "combined", "size": float(s), "direction": k, "obs_seed": OBS_SEED}
            for s in GRID_KM for k in (0, 1, 2)]
    # Old-sweep recipe (noise seed 10_000 + direction) for the reproduction check
    grid += [{"mode": "combined", "size": float(s), "direction": k, "obs_seed": 10_000 + k}
             for s in (10_000, 15_000, 20_000, 25_000, 30_000, 100_000) for k in (1, 2)]
    return grid, threshold_jobs(dirs=(1_000, 1_001), modes=("combined",))


# ----------------------------------------------------------------------------- driver

def _paths(out_dir):
    return {n: os.path.join(out_dir, f) for n, f in
            (("grid", "grid.csv"), ("thr", "thresholds.csv"), ("thr_runs", "thresholds_runs.csv"))}


def run_grid(cfg, jobs, out_dir, workers=None):
    return parallel.run_jobs(jobs, _paths(out_dir)["grid"], GRID_KEY, grid_job, init=init_worker,
                             init_args=(cfg,), workers=workers, label="grid")


def run_thresholds(cfg, jobs, out_dir, workers=None):
    p = _paths(out_dir)
    return parallel.run_jobs(jobs, p["thr"], THRESH_KEY, threshold_job, init=init_worker, init_args=(cfg,),
                             workers=workers, extra_csv_path=p["thr_runs"], label="thresholds")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convergence study: where and why BLS stops converging")
    ap.add_argument("which", choices=["pilot", "grid", "robustness", "thresholds", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", help=f"default: {OUT_DIR}/<estimator>")
    ap.add_argument("--estimation", help="override estimation.method (default: config.yaml)")
    ap.add_argument("--workers", type=int, help="default: cpu_count - 1")
    ap.add_argument("--dirs", type=int, nargs="+", help="override the direction ids")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.estimation:
        cfg["estimation"] = {**(cfg.get("estimation") or {}), "method": a.estimation}
    a.out = a.out or os.path.join(OUT_DIR, cfg["estimation"]["method"])
    if cfg["iod"]["method"] != "truth_plus_error":
        raise SystemExit("The convergence study sets the IOD error, so iod.method must be truth_plus_error")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump({**copy.deepcopy(cfg), "convergence": {
            "noise_arcsec": NOISE_ARCSEC, "obs_seed": OBS_SEED, "velocity_m_s_per_km": VEL_PER_KM,
            "gate_rms_sigma": GATE_RMS,
            "grid_km": GRID_KM, "grid_m_s": GRID_M_S}}, f, sort_keys=False)

    if a.which == "pilot":
        g, t = pilot_jobs()
        run_grid(cfg, g, a.out, a.workers)
        run_thresholds(cfg, t, a.out, a.workers)
    if a.which in ("grid", "all"):
        run_grid(cfg, grid_jobs(a.dirs if a.dirs else GRID_DIRS), a.out, a.workers)
    if a.which in ("grid", "robustness", "all") and not a.dirs:
        run_grid(cfg, robustness_jobs(), a.out, a.workers)
    if a.which in ("thresholds", "all"):
        run_thresholds(cfg, threshold_jobs(a.dirs if a.dirs else THRESH_DIRS), a.out, a.workers)
    return 0


if __name__ == "__main__":
    _sys.exit(main())

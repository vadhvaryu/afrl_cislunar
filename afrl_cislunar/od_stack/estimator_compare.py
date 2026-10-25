"""
Compare estimator variants on the same cases: which settings or methods converge more often,
and is their covariance honest?

    python -m od_stack.estimator_compare settings        # Orekit BLS scale/threshold test
    python -m od_stack.estimator_compare settings --noise-only --best scale_1km_thr_1e-3
    python -m od_stack.estimator_compare a7              # pipelines with and without batch least squares

A "variant" is a label plus the estimation block of config.yaml (method + options). Every
variant runs the same cases through the same stand-in IOD, noise and grading as the
convergence study (convergence.run_case), so the rows are directly comparable:

    grid cases    combined mode, chosen directions x starting errors, one fixed 10" noise draw
    noise cases   starting error 1,000 km / 10 m/s, noise levels x seeds (noise seed 10,000 + seed),
                  same recipe as sweep.py's noise sweep; reports NEES and the share inside 95%

Outputs (resumable) in results/estimator_compare/<study>/: runs.csv, summary.json, summary.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys as _sys

import numpy as np

from . import parallel
from .convergence import GRID_KEY, directions, gated_outcome, run_case  # noqa: F401
from .run import load_config
from .sweep import _wilson

OUT_DIR = "results/estimator_compare"
KEY = ("variant", "kind", "mode", "size", "direction", "obs_seed", "noise_arcsec")

# Orekit BLS settings test: position scale x convergence threshold (velocity scale follows the position scale)
SETTINGS = {f"scale_{lab}_thr_{t:g}": {"method": "orekit_bls", "position_scale_m": sc, "convergence_threshold": t}
            for lab, sc in (("1m", 1.0), ("1km", 1e3), ("100km", 1e5)) for t in (1e-3, 1e-5)}
SETTINGS_DIRS, SETTINGS_KM = range(0, 10), (15_000.0, 30_000.0)
NOISE_LEVELS, NOISE_SEEDS = (0.3, 1.0), range(0, 20)

# A7: pipelines with and without Orekit batch least squares
PIPELINES = {
    "P0_none": {"method": "none"},
    "P1_orekit_bls": {"method": "orekit_bls"},
    "P2_orekit_bls_robust": {"method": "orekit_bls_robust"},
    "P3_multiple_shooting": {"method": "multiple_shooting"},
    "P4_multiple_shooting_then_bls": {"method": "multiple_shooting_then_bls"},
    "P5_ukf": {"method": "ukf"},
}
A7_DIRS, A7_KM = range(0, 50), (1_000.0, 10_000.0, 15_000.0, 20_000.0, 30_000.0, 50_000.0, 100_000.0)
A7_NOISE, A7_SEEDS = (1.0, 10.0, 100.0), range(0, 20)


# ----------------------------------------------------------------------------- worker side

def init_worker(cfg: dict, variants: dict) -> None:
    """One Sweeper per variant (they share the JVM), plus the feature calculator."""
    import warnings

    warnings.filterwarnings("ignore", message="No observer columns")
    from .convergence_features import FeatureCalculator
    from .sweep import Sweeper

    parallel.WORKER["variants"], parallel.WORKER["features_by_truth"] = {}, {}
    for label, est in variants.items():
        est = dict(est)
        data = est.pop("_data", None)                 # optional: a different truth orbit / observation file
        c = copy.deepcopy(cfg)
        c["estimation"] = {**{k: v for k, v in (cfg.get("estimation") or {}).items() if k != "method"}, **est}
        if data:
            c["data"] = {**c["data"], **data}
        sw = parallel.WORKER["variants"][label] = Sweeper(c)
        if sw.truth_path not in parallel.WORKER["features_by_truth"]:
            parallel.WORKER["features_by_truth"][sw.truth_path] = FeatureCalculator(sw.sys, sw.truth_path)


def do_job(job: dict) -> dict:
    sw = parallel.WORKER["variants"][job["variant"]]
    row = run_case(job["mode"], job["size"], job["direction"], job["obs_seed"], sw=sw,
                   fc=parallel.WORKER["features_by_truth"][sw.truth_path], noise_arcsec=job["noise_arcsec"])
    row.update(variant=job["variant"], kind=job["kind"])
    return row


# ----------------------------------------------------------------------------- job lists

def grid_jobs(variants, dirs, sizes_km, obs_seed=10_000, noise=10.0):
    return [{"variant": v, "kind": "grid", "mode": "combined", "size": float(d), "direction": int(k),
             "obs_seed": int(obs_seed), "noise_arcsec": float(noise)}
            for v in variants for d in sizes_km for k in dirs]


def noise_jobs(variants, levels, seeds, size_km=1_000.0, direction=None):
    """Noise sweep cases: direction id = seed (as in sweep.py), noise seed 10,000 + seed."""
    return [{"variant": v, "kind": "noise", "mode": "combined", "size": float(size_km),
             "direction": int(s if direction is None else direction), "obs_seed": 10_000 + int(s),
             "noise_arcsec": float(n)} for v in variants for n in levels for s in seeds]


# ----------------------------------------------------------------------------- summary

def _f(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return np.nan


def summarize(rows: list[dict]) -> dict:
    out = {"grid": {}, "noise": {}}
    for r in rows:
        r["outcome"] = gated_outcome({**r, "bls_finished": r.get("bls_finished") in (True, "True"),
                                      "converged": r.get("converged") in (True, "True"),
                                      "rms_normalized_residual": _f(r, "rms_normalized_residual")
                                      if r.get("rms_normalized_residual") not in ("", None) else None})
    for kind, by in (("grid", "size"), ("noise", "noise_arcsec")):
        for v in sorted({r["variant"] for r in rows if r["kind"] == kind}):
            for x in sorted({_f(r, by) for r in rows if r["kind"] == kind and r["variant"] == v}):
                g = [r for r in rows if r["kind"] == kind and r["variant"] == v and _f(r, by) == x]
                conv = [r for r in g if r["outcome"] == "converged"]
                p, lo, hi = _wilson(len(conv), len(g))
                cov = [r for r in conv if np.isfinite(_f(r, "nees_6d"))]
                out[kind].setdefault(v, []).append({
                    by: x, "runs": len(g), "converged": len(conv), "rate": p, "lo": lo, "hi": hi,
                    "outcomes": {o: sum(r["outcome"] == o for r in g) for o in sorted({r["outcome"] for r in g})},
                    "median_iterations_converged": float(np.median([_f(r, "iterations") for r in conv])) if conv else np.nan,
                    "median_seconds": float(np.median([_f(r, "seconds") for r in g])),
                    "median_error_km": float(np.median([_f(r, "position_error_km") for r in conv])) if conv else np.nan,
                    "median_max_error_km": float(np.nanmedian([_f(r, "max_position_error_km") for r in conv])) if conv else np.nan,
                    "mean_seconds": float(np.mean([_f(r, "seconds") for r in g])),
                    "mean_nees_6d": float(np.mean([_f(r, "nees_6d") for r in cov])) if cov else np.nan,
                    "inside_95pct": f"{sum(r.get('inside_95pct_6d') in (True, 'True') for r in cov)}/{len(cov)}",
                    "inside_95pct_fraction": (sum(r.get('inside_95pct_6d') in (True, 'True') for r in cov) / len(cov)
                                              if cov else np.nan)})
    return out


def markdown(summ: dict) -> str:
    lines = []
    if summ["grid"]:
        lines += ["## Convergence (combined mode, fixed 10\" noise draw; gated at 3 sigma)\n",
                  "| Variant | d (km) | Converged | 95% interval | Outcomes | Median iterations (conv.) | Median s/run |",
                  "|---|---|---|---|---|---|---|"]
        for v, rows in summ["grid"].items():
            for a in rows:
                oc = ", ".join(f"{k} {n}" for k, n in a["outcomes"].items())
                lines.append(f"| {v} | {a['size']:,.0f} | {a['converged']}/{a['runs']} | {a['lo']:.2f}–{a['hi']:.2f} | "
                             f"{oc} | {a['median_iterations_converged']:.0f} | {a['median_seconds']:.1f} |")
    if summ["noise"]:
        lines += ["", "## Noise (start 1,000 km / 10 m/s; NEES expect 6, inside 95% expect ~95%)\n",
                  "| Variant | Noise (″) | Converged | Median epoch error (km) | Median max error (km) | Mean NEES | "
                  "Inside 95% | Mean s/run |", "|---|---|---|---|---|---|---|---|"]
        for v, rows in summ["noise"].items():
            for a in rows:
                lines.append(f"| {v} | {a['noise_arcsec']:g} | {a['converged']}/{a['runs']} | {a['median_error_km']:.2f} | "
                             f"{a['median_max_error_km']:.1f} | {a['mean_nees_6d']:.1f} | {a['inside_95pct']} | "
                             f"{a['mean_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def best_variant(summ: dict) -> str:
    """Most converged grid cases; ties -> fewer median iterations."""
    score = {}
    for v, rows in summ["grid"].items():
        score[v] = (sum(a["converged"] for a in rows), -np.nanmean([a["median_iterations_converged"] for a in rows]))
    return max(score, key=score.get)


# ----------------------------------------------------------------------------- figures (A7)

def plot_a7(summ: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    colors = dict(zip(PIPELINES, ["#808F9C", "#004B8D", "#56B4E9", "#D55E00", "#CC79A7", "#E69F00"]))
    paths = []

    def fig(name, title):
        f, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        paths.append(os.path.join(out_dir, name))
        return f, ax

    def save(f, ax):
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(paths[-1], bbox_inches="tight")
        plt.close(f)

    if summ["grid"]:
        f, ax = fig("a7_convergence.png", "Convergence vs starting error (50 directions, 10″, 3σ residual gate)")
        for v, rows in summ["grid"].items():
            x = [a["size"] for a in rows]
            p = np.array([a["rate"] for a in rows])
            ax.errorbar(x, p, yerr=[p - [a["lo"] for a in rows], [a["hi"] for a in rows] - p], fmt="o-", ms=4,
                        capsize=3, color=colors.get(v), label=v)
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Starting position error (km), velocity 0.01 m/s per km"); ax.set_ylabel("Fraction converged (95% interval)")
        save(f, ax)
        f, ax = fig("a7_runtime.png", "Runtime per case vs starting error (one worker)")
        for v, rows in summ["grid"].items():
            ax.plot([a["size"] for a in rows], [a["mean_seconds"] for a in rows], "o-", ms=4, color=colors.get(v), label=v)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Starting position error (km)"); ax.set_ylabel("Mean seconds per case")
        save(f, ax)
    if summ["noise"]:
        for key, name, ylab, logy, ref in (
                ("median_error_km", "a7_epoch_error.png", "Median position error at epoch (km)", True, None),
                ("median_max_error_km", "a7_max_error.png", "Median max position error over 8 days (km)", True, None),
                ("mean_nees_6d", "a7_nees.png", "Mean NEES at epoch (expect 6)", True, 6.0),
                ("inside_95pct_fraction", "a7_inside95.png", "Share of truths inside the 95% region", False, 0.95)):
            f, ax = fig(name, f"{ylab} vs noise (start 1,000 km / 10 m/s, 20 seeds)")
            for v, rows in summ["noise"].items():
                ax.plot([a["noise_arcsec"] for a in rows], [a[key] for a in rows], "o-", ms=4, color=colors.get(v), label=v)
            if ref is not None:
                ax.axhline(ref, color="0.4", ls="--", lw=0.8)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Angle noise, 1σ (arcsec)"); ax.set_ylabel(ylab)
            save(f, ax)
    return paths


# ----------------------------------------------------------------------------- driver

def run_study(cfg, variants, jobs, out_dir, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = parallel.run_jobs(jobs, os.path.join(out_dir, "runs.csv"), KEY, do_job, init=init_worker,
                             init_args=(cfg, variants), workers=workers, label=os.path.basename(out_dir))
    summ = summarize(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(markdown(summ))
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare estimator variants on the same cases")
    ap.add_argument("study", choices=["settings", "a7"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--best", help="variant for the low-noise check (default: best on the grid)")
    ap.add_argument("--noise-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg["iod"]["method"] = "truth_plus_error"
    out = os.path.join(OUT_DIR, a.study)

    if a.study == "settings":
        summ = {"grid": {}}
        if not a.noise_only:
            summ = run_study(cfg, SETTINGS, grid_jobs(SETTINGS, SETTINGS_DIRS, SETTINGS_KM), out, a.workers)
        best = a.best or best_variant(summ)
        default = "scale_1m_thr_0.001"
        pick = {k: SETTINGS[k] for k in dict.fromkeys([default, best])}
        print(f"Low-noise check with: {', '.join(pick)}")
        jobs = grid_jobs(SETTINGS, SETTINGS_DIRS, SETTINGS_KM) if not a.noise_only else []
        summ = run_study(cfg, {**SETTINGS}, jobs + noise_jobs(pick, NOISE_LEVELS, NOISE_SEEDS), out, a.workers)
        print(markdown(summ))
        print(f"Best grid variant: {best_variant(summ) if summ['grid'] else best}")
    if a.study == "a7":
        jobs = grid_jobs(PIPELINES, A7_DIRS, A7_KM) + noise_jobs(PIPELINES, A7_NOISE, A7_SEEDS)
        summ = run_study(cfg, PIPELINES, jobs, out, a.workers)
        print(markdown(summ))
        print("Figures:", *plot_a7(summ, out), sep="\n  ")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

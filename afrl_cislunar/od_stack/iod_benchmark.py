"""
Cross-check harness for the IOD methods (Part B, B3): which methods work, where, and how fast.

    python -m od_stack.iod_benchmark                              # default (trimmed) set, results/iod_benchmark/
    python -m od_stack.iod_benchmark --methods cr3bp_dc range_grid --noise 1 10 --seeds 5 --arcs current

Scenario: the L1 halo truth (obs_generator.py's TRUTH_STATE), observed from Earth's center with the
same geometry and noise model as obs_generator.py (cr3bp_tools.make_synthetic_obs), on three arcs:
    current    5 observations over days 0-8 (the data/synthetic_obs.csv geometry)
    dense_1d   5 observations over 1 day starting at day 3
    dense_6h   5 observations over 6 hours starting at day 4
Noise seed per run: 20,000 + seed. The truth is only used for grading, never passed to a method.

Per run: IOD success, position/velocity error at the IOD epoch, number of distinct solutions the
method reports, residual RMS of the chosen state on all observations and on the observations the
method did not use (3-observation methods), IOD runtime; then the gated robust estimator
(orekit_bls_robust: 3-sigma residual gate, short-arc continuation, line-of-sight restarts) from the
IOD state: outcome, iterations, final error, total runtime.

"Converged" means the estimator, started from the IOD, reaches the SAME solution it reaches when
started from the truth (within 10 km) and passes the 3-sigma gate. On short arcs the best fit to the
angles can itself be thousands of km from the truth (range is weakly observable from Earth's center:
on the 6-hour arc at 10" every start lands ~8,500 km away), so "within X km of the truth" would grade
the data, not the IOD. The error of that solution against the truth is reported separately
(bls_error_km, reference_error_km).

A method WORKS on a scenario (arc, noise) if >= 90% of seeds land within 10,000 km AND the
estimator then converges in that sense.

Outputs: runs.csv (resumable), summary.json, SUMMARY_tables.md, success_heatmap.png,
iod_error_boxplot.png, runtime.png, bls_after_iod.png, cross_check.csv (pairwise distances between
methods' IOD states on the same case).
"""

from __future__ import annotations

import argparse
import json
import os
import sys as _sys
import time

import numpy as np

from . import parallel

OUT_DIR = "results/iod_benchmark"
DAY = 86400.0
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py
ARCS = {"current": (0.0, 8 * DAY), "dense_1d": (3 * DAY, 4 * DAY), "dense_6h": (4 * DAY, 4 * DAY + 6 * 3600.0)}
METHODS = ["cr3bp_dc", "range_grid", "range_grid_fd", "halo_catalog", "perturbed_laplace", "laplace_moon",
           "gooding_moon", "gauss", "laplace", "gooding"]
THREE_OBS = {"cr3bp_dc", "laplace_moon", "gooding_moon", "gauss", "laplace", "gooding"}
TRIPLES = [(0, 2, 4), (0, 1, 2), (1, 2, 3), (2, 3, 4)]
SLOW_TRIPLE_ONLY_DEFAULT = {"cr3bp_dc"}            # ~60 s per call: default triple only (trimmed set)
KEY = ("method", "arc", "triple", "noise_arcsec", "seed")
WITHIN_KM = 10_000.0


# ----------------------------------------------------------------------------- worker side

def init_worker(orekit_data):
    import warnings

    warnings.filterwarnings("ignore")
    from .estimation import ESTIMATORS
    from .frames import EarthMoonSystem

    sys = EarthMoonSystem()
    parallel.WORKER.update(sys=sys, methods={}, orekit_data=orekit_data,
                           estimator=ESTIMATORS["orekit_bls_robust"](sys, orekit_data=orekit_data))


def _method(name, triple):
    from .iod import IOD_METHODS

    key = (name, triple)
    cache = parallel.WORKER["methods"]
    if key not in cache:
        kw = {"orekit_data": parallel.WORKER["orekit_data"]}
        if name in THREE_OBS and triple:
            kw["obs_indices"] = list(triple)
        cache[key] = IOD_METHODS[name](parallel.WORKER["sys"], **kw)
    return cache[key]


def _truth(sys, t):
    from .dynamics import propagate_cr3bp
    from .frames import rotating_to_emi

    tau = float(sys.tau(t))
    return rotating_to_emi(tau, propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau]), sys.mu)[0], sys)


def _rms(obs, x, idx, sys):
    """RA/Dec residual RMS (sigma) of state x at observations idx (CR3BP propagation)."""
    from .contracts import Observations, Trajectory
    from .dynamics import propagate_cr3bp
    from .frames import emi_to_rotating, rotating_to_emi
    from .results import residuals

    idx = list(idx)
    if not idx:
        return float("nan")
    tau0 = float(sys.tau(x.t))
    X = propagate_cr3bp(emi_to_rotating(tau0, x.r, x.v, sys), tau0, sys.tau(obs.t[idx]), sys.mu)
    rv = [rotating_to_emi(float(sys.tau(t)), xi, sys) for t, xi in zip(obs.t[idx], X)]
    sub = Observations(t=obs.t[idx], los=obs.los[idx], observer=obs.observer[idx], sigma=obs.sigma[idx],
                       ra=obs.ra[idx], dec=obs.dec[idx])
    return residuals(sub, Trajectory(t=sub.t, r=np.array([a for a, _ in rv]), v=np.array([b for _, b in rv])))["rms_normalized"]


def _reference(obs, job, t_epoch):
    """The estimator's fit started from the truth at the IOD epoch (cached per worker and case)."""
    from .contracts import StateEstimate
    from .estimation import EstimationFailure

    key = (job["arc"], float(job["noise_arcsec"]), int(job["seed"]), round(float(t_epoch), 3))
    cache = parallel.WORKER.setdefault("reference", {})
    if key not in cache:
        r, v = _truth(parallel.WORKER["sys"], t_epoch)
        try:
            cache[key] = parallel.WORKER["estimator"].refine(obs, StateEstimate(t=t_epoch, r=r, v=v, method="truth"))
        except EstimationFailure:
            cache[key] = None
    return cache[key]


def _n_solutions(info):
    for k in ("n_distinct_solutions", "n_fitting_solutions", "distinct_solutions", "solutions", "roots",
              "candidates", "top_candidates"):
        v = info.get(k)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (list, tuple)):
            return len(v)
    return None


def do_job(job):
    from .cr3bp_tools import make_synthetic_obs
    from .estimation import EstimationFailure
    from .iod import IODFailure

    sys = parallel.WORKER["sys"]
    row = dict(job)
    a, b = ARCS[job["arc"]]
    obs = make_synthetic_obs(TRUTH_STATE, np.linspace(a, b, 5), float(job["noise_arcsec"]), 20_000 + int(job["seed"]), sys)
    triple = tuple(int(c) for c in job["triple"].split("-")) if job["triple"] else None
    t0 = time.time()
    try:
        x = _method(job["method"], triple).estimate(obs)
    except IODFailure as e:
        row.update(iod_ok=False, outcome="iod_failed", iod_seconds=time.time() - t0, error=str(e)[:160])
        return row
    except Exception as e:  # a method bug must not stop the benchmark; record it
        row.update(iod_ok=False, outcome="iod_error", iod_seconds=time.time() - t0, error=f"{type(e).__name__}: {e}"[:160])
        return row
    row["iod_seconds"] = time.time() - t0
    rt, vt = _truth(sys, x.t)
    used = list(triple) if triple else list(range(len(obs)))
    row.update(iod_ok=True, iod_epoch_days=x.t / DAY,
               iod_position_error_km=float(np.linalg.norm(x.r - rt) / 1e3),
               iod_velocity_error_m_s=float(np.linalg.norm(x.v - vt)),
               iod_within_10000km=bool(np.linalg.norm(x.r - rt) < WITHIN_KM * 1e3),
               n_solutions=_n_solutions(x.info),
               iod_rms_all=_rms(obs, x, range(len(obs)), sys),
               iod_rms_held_out=_rms(obs, x, [i for i in range(len(obs)) if i not in used], sys),
               **{f"iod_r{c}_m": float(v) for c, v in zip("xyz", x.r)}, **{f"iod_v{c}_m_s": float(v) for c, v in zip("xyz", x.v)})
    t1 = time.time()
    ref = _reference(obs, job, x.t)
    try:
        fit = parallel.WORKER["estimator"].refine(obs, x)
        rt2, _ = _truth(sys, fit.t)
        err = float(np.linalg.norm(fit.r - rt2) / 1e3)
        if ref is not None:
            ok = float(np.linalg.norm(fit.r - ref.r) / 1e3) < 10.0
            row["reference_error_km"] = float(np.linalg.norm(ref.r - rt2) / 1e3)
        else:                                   # no reference fit: fall back to distance from the truth
            ok = err < 1000.0 * max(1.0, float(job["noise_arcsec"]) / 10.0)
        row.update(bls_ok=ok, outcome="converged" if ok else "wrong_orbit", bls_error_km=err,
                   bls_rms=fit.info.get("rms_normalized_residual"), bls_iterations=fit.info.get("iterations"),
                   bls_step=fit.info.get("robust_step"))
    except EstimationFailure as e:
        row.update(bls_ok=False, outcome="bls_rejected" if "residual gate" in str(e) else "bls_failed",
                   error=str(e).split(": ", 1)[-1][:160])
    row.update(bls_seconds=time.time() - t1, total_seconds=time.time() - t0)
    return row


# ----------------------------------------------------------------------------- jobs and summary

def jobs(methods, arcs, noises, seeds, all_triples=True):
    out = []
    for m in methods:
        if m in THREE_OBS:
            trip = [TRIPLES[0]] if (m in SLOW_TRIPLE_ONLY_DEFAULT or not all_triples) else TRIPLES
        else:
            trip = [None]
        for arc in arcs:
            for t in trip:
                for n in noises:
                    for s in seeds:
                        out.append({"method": m, "arc": arc, "triple": "-".join(map(str, t)) if t else "",
                                    "noise_arcsec": float(n), "seed": int(s)})
    return out


def _b(v):
    return v in (True, "True")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["arc"], r.get("triple", ""), _f(r["noise_arcsec"])), []).append(r)
    out = []
    for (m, arc, tri, n), g in sorted(groups.items()):
        within = sum(_b(r.get("iod_within_10000km")) for r in g)
        conv = sum(r.get("outcome") == "converged" for r in g)
        e = np.array([_f(r.get("iod_position_error_km")) for r in g if _b(r.get("iod_ok"))])
        out.append({"method": m, "arc": arc, "triple": tri, "noise_arcsec": n, "runs": len(g),
                    "iod_ok": sum(_b(r.get("iod_ok")) for r in g), "within_10000km": within, "bls_converged": conv,
                    "works": within >= 0.9 * len(g) and conv >= 0.9 * len(g),
                    "median_iod_error_km": float(np.median(e)) if len(e) else np.nan,
                    "median_iod_seconds": float(np.nanmedian([_f(r.get("iod_seconds")) for r in g])),
                    "median_total_seconds": float(np.nanmedian([_f(r.get("total_seconds")) for r in g]))})
    return out


def cross_check(rows):
    """Pairwise distances between different methods' IOD states (same arc, noise, seed; default triple)."""
    by = {}
    for r in rows:
        if _b(r.get("iod_ok")) and r.get("triple", "") in ("", "0-2-4"):
            by.setdefault((r["arc"], _f(r["noise_arcsec"]), int(_f(r["seed"]))), {})[r["method"]] = (
                np.array([_f(r[f"iod_r{c}_m"]) for c in "xyz"]), _f(r.get("iod_epoch_days")))
    out = []
    for (arc, n, s), d in sorted(by.items()):
        names = sorted(d)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if abs(d[a][1] - d[b][1]) < 1e-9:          # same epoch only
                    out.append({"arc": arc, "noise_arcsec": n, "seed": s, "method_a": a, "method_b": b,
                                "distance_km": float(np.linalg.norm(d[a][0] - d[b][0]) / 1e3)})
    return out


def tables_markdown(summ):
    lines = ["| Method | Arc | Triple | Noise (″) | IOD ok | Within 10,000 km | BLS converged | Works | "
             "Median IOD error (km) | Median IOD s | Median total s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in summ:
        lines.append(f"| {a['method']} | {a['arc']} | {a['triple'] or 'all'} | {a['noise_arcsec']:g} | {a['iod_ok']}/{a['runs']} | "
                     f"{a['within_10000km']}/{a['runs']} | {a['bls_converged']}/{a['runs']} | {'yes' if a['works'] else 'no'} | "
                     f"{a['median_iod_error_km']:,.1f} | {a['median_iod_seconds']:.2f} | {a['median_total_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures

def plots(rows, summ, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    paths = []
    methods = list(dict.fromkeys(a["method"] for a in summ))
    noises = sorted({a["noise_arcsec"] for a in summ})
    arcs = [a for a in ARCS if any(s["arc"] == a for s in summ)]

    def best(m, arc, n, key):
        g = [a for a in summ if a["method"] == m and a["arc"] == arc and a["noise_arcsec"] == n]
        return max((a[key] / a["runs"] for a in g), default=np.nan)

    for key, name, title in (("within_10000km", "success_heatmap.png", "IOD lands within 10,000 km (best triple)"),
                             ("bls_converged", "bls_after_iod.png", "Gated robust BLS converges from the IOD (best triple)")):
        fig, axs = plt.subplots(1, len(arcs), figsize=(4.2 * len(arcs), 0.45 * len(methods) + 1.8), dpi=110, squeeze=False)
        for ax, arc in zip(axs[0], arcs):
            M = np.array([[best(m, arc, n, key) for n in noises] for m in methods])
            im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            for i in range(len(methods)):
                for j in range(len(noises)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8,
                                color="white" if M[i, j] > 0.6 else "black")
            ax.set_xticks(range(len(noises)))
            ax.set_xticklabels([f'{n:g}"' for n in noises])
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods if ax is axs[0][0] else [])
            ax.set_title(arc)
        fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label="Fraction of seeds")
        fig.suptitle(title)
        paths.append(os.path.join(out_dir, name))
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)

    fig, axs = plt.subplots(1, len(arcs), figsize=(4.6 * len(arcs), 4.4), dpi=110, sharey=True, squeeze=False)
    for ax, arc in zip(axs[0], arcs):
        data = [[max(_f(r.get("iod_position_error_km")), 1e-3) for r in rows if r["method"] == m and r["arc"] == arc
                 and _b(r.get("iod_ok")) and r.get("triple", "") in ("", "0-2-4") and _f(r["noise_arcsec"]) == 10.0]
                for m in methods]
        ax.boxplot([d if d else [np.nan] for d in data], vert=False)
        ax.set_yticks(range(1, len(methods) + 1))
        ax.set_yticklabels(methods)
        ax.set_xscale("log")
        ax.axvline(WITHIN_KM, color="#D55E00", ls="--", lw=1)
        ax.set_xlabel("IOD position error (km), 10″, default triple")
        ax.set_title(arc)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("IOD error by method (dashed: 10,000 km, the BLS basin)")
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "iod_error_boxplot.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(methods) + 1.5), dpi=110)
    med = [np.nanmedian([_f(r.get("iod_seconds")) for r in rows if r["method"] == m] or [np.nan]) for m in methods]
    ax.barh(methods, med, color="#004B8D")
    ax.set_xscale("log")
    ax.set_xlabel("Median IOD runtime per call (s), one worker")
    ax.set_title("IOD runtime")
    ax.grid(alpha=0.3, which="both", axis="x")
    fig.tight_layout()
    paths.append(os.path.join(out_dir, "runtime.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IOD method benchmark (Part B, B3)")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--arcs", nargs="+", default=list(ARCS), choices=list(ARCS))
    ap.add_argument("--noise", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--default-triple-only", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    js = jobs(a.methods, a.arcs, a.noise, range(a.seeds), all_triples=not a.default_triple_only)
    rows = parallel.run_jobs(js, os.path.join(a.out, "runs.csv"), KEY, do_job, init=init_worker,
                             init_args=(os.environ.get("OREKIT_DATA"),), workers=a.workers, label="iod_benchmark")
    summ = summarize(rows)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    md = tables_markdown(summ)
    with open(os.path.join(a.out, "SUMMARY_tables.md"), "w") as f:
        f.write(md)
    cc = cross_check(rows)
    if cc:
        import csv

        with open(os.path.join(a.out, "cross_check.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cc[0]))
            w.writeheader()
            w.writerows(cc)
    print(md)
    print("Figures:", *plots(rows, summ, a.out), sep="\n  ")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

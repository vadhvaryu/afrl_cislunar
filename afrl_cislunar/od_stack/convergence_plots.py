"""
Plots and statistics for the convergence study (reads results/convergence/*.csv).

    python -m od_stack.convergence_plots                  # all figures + analysis.json + SUMMARY_tables.md

Figures (results/convergence/):
    periselene_by_error.png        one panel per starting error: final error vs periselene of the guess
    convergence_vs_periselene.png  convergence rate vs periselene of the guess (d >= 10,000 km pooled)
    convergence_curve.png          convergence rate vs starting error, three modes
    convergence_vs_features.png    convergence rate per bin of each candidate variable
    threshold_hist.png             distribution of per-direction thresholds (A3)
    threshold_vs_direction.png     threshold vs error direction (line of sight, Moon-radial)
    direction_map.png              error direction in RIC (azimuth/elevation) colored by largest converged d

Statistics (numpy/scipy only):
    AUC     area under the ROC curve for predicting failure from one variable,
            from scipy.stats.mannwhitneyu (0.5 = no information; > 0.5: failures have larger values)
    logit   logistic regression of failure on standardized variables (scipy.optimize.minimize),
            95% intervals from 200 bootstrap resamples
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys as _sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

from .sweep import _wilson

OUT_DIR = "results/convergence"
OBS_SEED = 10_000
COLORS = {"converged": "#004B8D", "iteration_limit": "#E69F00", "wrong_orbit": "#D55E00", "other_error": "#808F9C"}
LABELS = {"converged": "Converged", "iteration_limit": "Iteration limit", "wrong_orbit": "Wrong orbit",
          "other_error": "Other error"}
MODE_LABELS = {"combined": "Combined (d km, 0.01·d m/s)", "position_only": "Position only",
               "velocity_only": "Velocity only"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13})

# Candidate explanations for failure, as (column, label). "_norm" columns are divided by the
# position error size, so they describe the error DIRECTION independent of its size.
FEATURES = [
    ("size", "Starting position error d (km)"),
    ("peri_guess_8d_km", "Periselene of guess, days 0–8 (km)"),
    ("peri_guess_12d_km", "Periselene of guess, days 0–12 (km)"),
    ("abs_peri_delta_8d_km", "|Periselene guess − truth|, 0–8 d (km)"),
    ("angle_dr_los_deg", "Angle between Δr and line of sight (deg)"),
    ("abs_los_frac", "|Δr · LOS| / |Δr| (range fraction)"),
    ("dr_moon_radial_norm", "Δr · Moon-radial / |Δr|"),
    ("dr_R_norm", "Δr radial (Earth) / |Δr|"),
    ("dr_I_norm", "Δr in-track / |Δr|"),
    ("dr_C_norm", "Δr cross-track / |Δr|"),
    ("dv_along_v_m_s", "Δv along v (m/s)"),
    ("abs_delta_jacobi", "|ΔC| Jacobi (nondim)"),
    ("log_start_residual", "log10 starting residual RMS (σ)"),
]
LOGIT_FEATURES = ["log_size", "abs_los_frac", "dr_moon_radial_norm", "dr_C_norm", "dv_along_v_norm",
                  "abs_delta_jacobi", "peri_guess_8d_km"]


# ----------------------------------------------------------------------------- loading

def _num(v):
    if v in ("", None):
        return np.nan
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        d = r.get("position_error_km_set", np.nan)
        dnz = d if isinstance(d, float) and d > 0 else np.nan
        r["abs_peri_delta_8d_km"] = abs(r.get("peri_delta_8d_km", np.nan))
        r["abs_los_frac"] = abs(r.get("dr_along_los_km", np.nan)) / dnz
        r["dr_moon_radial_norm"] = r.get("dr_moon_radial_km", np.nan) / dnz
        for c in "RIC":
            r[f"dr_{c}_norm"] = r.get(f"dr_{c}_km", np.nan) / dnz
        w = r.get("velocity_error_m_s_set", np.nan)
        r["dv_along_v_norm"] = r.get("dv_along_v_m_s", np.nan) / w if isinstance(w, float) and w > 0 else np.nan
        r["abs_delta_jacobi"] = abs(r.get("delta_jacobi", np.nan))
        s = r.get("start_residual_rms_sigma", np.nan)
        r["log_start_residual"] = np.log10(s) if isinstance(s, float) and s > 0 else np.nan
        r["log_size"] = np.log10(r["size"]) if r.get("size", 0) > 0 else np.nan
        r["failed"] = r.get("outcome") != "converged"
        r["final_error_shown_km"] = (r.get("position_error_km") if r.get("bls_finished") is True
                                     else r.get("final_state_error_km", np.nan))
    return rows


def col(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], dtype=float)


def sel(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


# ----------------------------------------------------------------------------- statistics

def rate_by(rows, key):
    out = []
    for v in sorted({r[key] for r in rows}):
        g = [r for r in rows if r[key] == v]
        k = sum(not r["failed"] for r in g)
        p, lo, hi = _wilson(k, len(g))
        out.append({"value": v, "n": len(g), "converged": k, "rate": p, "lo": lo, "hi": hi})
    return out


def binned_rate(x, failed, nbins=8):
    ok = np.isfinite(x)
    x, failed = x[ok], failed[ok]
    if len(x) < 2 * nbins:
        return []
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & ((x < b) | (b == edges[-1]))
        if not m.any():
            continue
        k = int((~failed[m]).sum())
        p, lo, hi = _wilson(k, int(m.sum()))
        out.append({"lo_edge": float(a), "hi_edge": float(b), "center": float(np.median(x[m])), "n": int(m.sum()),
                    "rate": p, "lo": lo, "hi": hi})
    return out


def auc(x, failed):
    """AUC for predicting failure from x (Mann-Whitney U / (n_fail * n_ok)), with its p-value."""
    ok = np.isfinite(x)
    a, b = x[ok & failed], x[ok & ~failed]
    if len(a) < 3 or len(b) < 3:
        return {"auc": np.nan, "p": np.nan, "n_failed": int(len(a)), "n_converged": int(len(b))}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"auc": float(u / (len(a) * len(b))), "p": float(p), "n_failed": int(len(a)), "n_converged": int(len(b))}


def _logit_fit(X, y, l2=1e-3):
    def nll(beta):
        z = X @ beta
        return float(np.sum(np.logaddexp(0, z) - y * z) + l2 * beta[1:] @ beta[1:])

    def grad(beta):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (p - y)
        g[1:] += 2 * l2 * beta[1:]
        return g

    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x


def logistic(rows, features, n_boot=200, seed=0):
    """Failure ~ standardized features. Small L2 penalty keeps it finite if the data separate."""
    X = np.column_stack([col(rows, f) for f in features])
    y = np.array([r["failed"] for r in rows], float)
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 20 or y.min() == y.max():
        return {"n": int(len(y)), "note": "not enough runs or no failures"}
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    beta = _logit_fit(Z, y)
    rng = np.random.default_rng(seed)
    boots = np.array([_logit_fit(Z[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"n": int(len(y)), "n_failed": int(y.sum()), "intercept": float(beta[0]),
            "coefficients": {f: {"beta": float(b), "lo": float(a), "hi": float(c),
                                 "excludes_zero": bool(a > 0 or c < 0)}
                             for f, b, a, c in zip(features, beta[1:], lo[1:], hi[1:])}}


def non_monotonic(rows):
    """Directions that fail at some size and converge at a larger one (per mode)."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        bad = []
        for k in sorted({r["direction"] for r in rows if r["mode"] == mode}):
            g = sorted(sel(rows, mode=mode, direction=k), key=lambda r: r["size"])
            fails = [r["size"] for r in g if r["failed"]]
            if fails and any(not r["failed"] and r["size"] > fails[0] for r in g):
                bad.append(int(k))
        out[mode] = {"count": len(bad), "of": len({r["direction"] for r in rows if r["mode"] == mode}),
                     "directions": bad}
    return out


def largest_converged(rows, mode):
    out = {}
    for k in sorted({r["direction"] for r in rows if r["mode"] == mode}):
        g = sel(rows, mode=mode, direction=k)
        conv = [r["size"] for r in g if not r["failed"]]
        out[int(k)] = max(conv) if conv else 0.0
    return out


def robustness(all_grid):
    rows = [r for r in all_grid if r["mode"] == "combined" and r["size"] in (15_000.0, 30_000.0)
            and r["direction"] < 20]
    cases, varying = {}, []
    for r in rows:
        cases.setdefault((r["size"], r["direction"]), []).append((int(r["obs_seed"]), r["outcome"]))
    for (d, k), v in sorted(cases.items()):
        if len(v) >= 5 and len({o for _, o in v}) > 1:
            varying.append({"size_km": d, "direction": int(k), "outcomes": dict(sorted(v))})
    complete = sum(len(v) >= 5 for v in cases.values())
    return {"cases_with_5_seeds": complete, "outcome_depends_on_noise": len(varying), "varying": varying}


def threshold_stats(thr):
    out = {}
    for mode in sorted({r["mode"] for r in thr}):
        g = sel(thr, mode=mode)
        t = np.array([r["threshold"] if r.get("censored") != "< 500" else 0.0 for r in g], float)
        cens_hi = sum(r.get("censored") == ">= 100,000" for r in g)
        out[mode] = {"directions": len(g), "median_km": float(np.median(t)), "p10_km": float(np.percentile(t, 10)),
                     "p90_km": float(np.percentile(t, 90)), "min_km": float(t.min()), "max_km": float(t.max()),
                     "at_least_100000": int(cens_hi), "below_500": int(sum(r.get("censored") == "< 500" for r in g)),
                     "non_monotonic_in_bisection": int(sum(r.get("monotonic_in_evaluations") is False for r in g))}
    return out


# ----------------------------------------------------------------------------- figures

def _legend_outcomes(ax, present, **kw):
    for o in COLORS:
        if o in present:
            ax.scatter([], [], s=18, color=COLORS[o], label=LABELS[o])
    ax.legend(**kw)


def fig_periselene_by_error(path, rows, truth_peri):
    sizes = sorted({r["size"] for r in rows if r["size"] >= 10_000})
    n = len(sizes)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow + 0.8), sharex=True, sharey=True,
                            dpi=110, squeeze=False)
    present = set()
    for ax, d in zip(axs.flat, sizes):
        g = [r for r in rows if r["size"] == d]
        for r in g:
            y = r["final_error_shown_km"]
            if not np.isfinite(y):
                continue
            present.add(r["outcome"])
            ax.scatter(r["peri_guess_8d_km"], max(y, 1.0), s=12, color=COLORS[r["outcome"]], alpha=0.85,
                       marker="o" if r["outcome"] == "converged" else "^")
        ax.axvline(truth_peri, color="0.3", ls="--", lw=0.8)
        k = sum(not r["failed"] for r in g)
        ax.set_title(f"d = {d:,.0f} km  ({k}/{len(g)})", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Periselene of guess (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Final error (km)")
    _legend_outcomes(axs.flat[0], present, fontsize=8, loc="lower right")
    fig.suptitle("Final error vs periselene of the starting guess, per starting error (combined mode; "
                 f"dashed = truth periselene {truth_peri:,.0f} km; failures plotted at BLS's last iterate)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rate_panel(ax, b, color="#004B8D", label=None, x="center"):
    if not b:
        return
    xs = np.array([v[x] for v in b], float)
    p = np.array([v["rate"] for v in b])
    ax.errorbar(xs, p, yerr=[p - [v["lo"] for v in b], [v["hi"] for v in b] - p], fmt="o-", color=color,
                capsize=3, label=label, ms=4)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)


def fig_vs_periselene(path, rows, truth_peri, a):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
    b = binned_rate(col(rows, "peri_guess_8d_km"), np.array([r["failed"] for r in rows]), nbins=10)
    _rate_panel(ax, b)
    for v in b:
        ax.annotate(f"n={v['n']}", (v["center"], v["rate"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.axvline(truth_peri, color="0.3", ls="--", lw=0.8, label=f"Truth periselene ({truth_peri:,.0f} km)")
    ax.set_xlabel("Periselene of starting guess, days 0–8 (km); bins of equal count")
    ax.set_ylabel("Fraction converged (95% interval)")
    ax.set_title(f"Convergence vs periselene of the guess (combined, d ≥ 10,000 km; AUC = {a['auc']:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_curve(path, grid):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    ax = axs[0]
    for mode, c in (("combined", "#004B8D"), ("position_only", "#D55E00")):
        b = rate_by(sel(grid, mode=mode), "size")
        _rate_panel(ax, b, color=c, label=MODE_LABELS[mode], x="value")
    ax.set_xscale("log")
    ax.set_xlabel("Starting position error d (km)")
    ax.set_ylabel("Fraction converged (95% interval)")
    ax.set_title("Position error")
    ax.legend(fontsize=9, loc="lower left")
    ax = axs[1]
    b = rate_by(sel(grid, mode="velocity_only"), "size")
    _rate_panel(ax, b, color="#E69F00", label=MODE_LABELS["velocity_only"], x="value")
    ax.set_xscale("log")
    ax.set_xlabel("Starting velocity error w (m/s), position exact")
    ax.set_title("Velocity error")
    ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("BLS convergence vs starting error (50 directions per size, one 10″ noise draw)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_features(path, rows, aucs):
    n = len(FEATURES)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), dpi=110, squeeze=False, sharey=True)
    failed = np.array([r["failed"] for r in rows])
    for ax, (k, label) in zip(axs.flat, FEATURES):
        _rate_panel(ax, binned_rate(col(rows, k), failed))
        ax.set_xlabel(label)
        ax.set_title(f"AUC {aucs[k]['auc']:.2f}", fontsize=10)
    for ax in axs[:, 0]:
        ax.set_ylabel("Fraction converged")
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle("Convergence rate per bin of each candidate variable (combined, d ≥ 10,000 km; 95% intervals)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_threshold_hist(path, thr, stats):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=110)
    bins = np.linspace(0, 100_000, 41)
    for mode, c in (("combined", "#004B8D"), ("position_only", "#D55E00")):
        g = sel(thr, mode=mode)
        if not g:
            continue
        t = np.array([r["threshold"] if np.isfinite(r["threshold"]) else 0.0 for r in g])
        s = stats[mode]
        ax.hist(t, bins=bins, histtype="step", lw=1.8, color=c,
                label=f"{MODE_LABELS[mode]}: median {s['median_km']:,.0f}, 10th pct {s['p10_km']:,.0f} km")
        ax.axvline(s["median_km"], color=c, ls="-", lw=0.9)
        ax.axvline(s["p10_km"], color=c, ls=":", lw=1.2)
    ax.set_xlabel("Largest converged starting error (km), bisected to 250 km (100,000 = never failed)")
    ax.set_ylabel("Directions")
    ax.set_title("Convergence threshold per error direction (solid: median, dotted: 10th percentile)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_threshold_vs_direction(path, thr, first_runs):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110, sharey=True)
    for mode, c, m in (("combined", "#004B8D", "o"), ("position_only", "#D55E00", "^")):
        g = sel(thr, mode=mode)
        x1, x2, y = [], [], []
        for r in g:
            f = first_runs.get((mode, r["direction"]))
            if f is None:
                continue
            x1.append(f["angle_dr_los_deg"])
            x2.append(f["dr_moon_radial_norm"])
            y.append(r["threshold"] if np.isfinite(r["threshold"]) else 0.0)
        axs[0].scatter(x1, y, s=14, color=c, marker=m, alpha=0.7, label=MODE_LABELS[mode])
        axs[1].scatter(x2, y, s=14, color=c, marker=m, alpha=0.7, label=MODE_LABELS[mode])
    axs[0].set_xlabel("Angle between position error and line of sight at day 4 (deg)")
    axs[1].set_xlabel("Moon-radial component of position error direction (−1 = toward Moon)")
    axs[0].set_ylabel("Threshold (km)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Per-direction convergence threshold vs error direction")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_direction_map(path, grid):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=110)
    for ax, mode in zip(axs, ("combined", "position_only")):
        g = sel(grid, mode=mode)
        best = largest_converged(g, mode)
        ref = {}
        for r in g:
            if r["direction"] not in ref and r["size"] > 0:
                ref[r["direction"]] = r
        k = sorted(ref)
        az = [np.degrees(np.arctan2(ref[i]["dr_I_norm"], ref[i]["dr_R_norm"])) for i in k]
        el = [np.degrees(np.arcsin(np.clip(ref[i]["dr_C_norm"], -1, 1))) for i in k]
        sc = ax.scatter(az, el, c=[best[i] for i in k], cmap="viridis", s=40, vmin=0, vmax=100_000,
                        edgecolor="k", lw=0.3)
        for i, x, y in zip(k, az, el):
            ax.annotate(str(i), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Azimuth from radial toward in-track (deg)")
        ax.set_ylabel("Elevation toward cross-track (deg)")
        ax.set_title(MODE_LABELS[mode])
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Largest converged d on the grid (km)")
    fig.suptitle("Position-error direction (Earth-centered RIC at day 4) and how far it converges")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver

def _md_rate_table(b, unit):
    lines = [f"| Size ({unit}) | Converged | 95% interval |", "|---|---|---|"]
    for v in b:
        lines.append(f"| {v['value']:,.0f} | {v['converged']}/{v['n']} | {v['lo']:.2f}–{v['hi']:.2f} |")
    return "\n".join(lines)


def analyze(out_dir: str = OUT_DIR) -> dict:
    all_grid = load(os.path.join(out_dir, "grid.csv"))
    thr = load(os.path.join(out_dir, "thresholds.csv"))
    thr_runs = load(os.path.join(out_dir, "thresholds_runs.csv"))
    grid = [r for r in all_grid if r["obs_seed"] == OBS_SEED]
    from .convergence_features import FeatureCalculator
    from .frames import EarthMoonSystem

    truth_peri = FeatureCalculator(EarthMoonSystem(), "data/synthetic_truth.csv").truth_periselene
    tp8 = truth_peri["8d"]["km"]
    res = {"truth_periselene": truth_peri, "runs": {"grid": len(grid), "all_grid_rows": len(all_grid),
                                                     "threshold_directions": len(thr),
                                                     "threshold_runs": len(thr_runs)}}

    res["curve"] = {m: rate_by(sel(grid, mode=m), "size") for m in ("combined", "position_only", "velocity_only")}
    res["outcomes"] = {m: {o: sum(r["outcome"] == o for r in sel(grid, mode=m)) for o in COLORS}
                       for m in ("combined", "position_only", "velocity_only")}
    res["non_monotonic"] = non_monotonic(grid)
    res["robustness"] = robustness(all_grid)

    big = [r for r in sel(grid, mode="combined") if r["size"] >= 10_000]
    big_pos = [r for r in sel(grid, mode="position_only") if r["size"] >= 10_000]
    pooled = big + [r for r in thr_runs if r["mode"] == "combined" and r["size"] >= 10_000]
    failed = np.array([r["failed"] for r in big])
    res["auc_combined_d_ge_10000"] = {k: auc(col(big, k), failed) for k, _ in FEATURES}
    if big_pos:
        fp = np.array([r["failed"] for r in big_pos])
        res["auc_position_only_d_ge_10000"] = {k: auc(col(big_pos, k), fp) for k, _ in FEATURES}
    res["logit_combined_d_ge_10000"] = logistic(pooled, LOGIT_FEATURES)
    if thr:
        res["thresholds"] = threshold_stats(thr)

    figs = {}
    if big:
        figs["periselene_by_error"] = os.path.join(out_dir, "periselene_by_error.png")
        fig_periselene_by_error(figs["periselene_by_error"], sel(grid, mode="combined"), tp8)
        figs["convergence_vs_periselene"] = os.path.join(out_dir, "convergence_vs_periselene.png")
        fig_vs_periselene(figs["convergence_vs_periselene"], big, tp8, res["auc_combined_d_ge_10000"]["peri_guess_8d_km"])
        figs["convergence_vs_features"] = os.path.join(out_dir, "convergence_vs_features.png")
        fig_features(figs["convergence_vs_features"], big, res["auc_combined_d_ge_10000"])
    if grid:
        figs["convergence_curve"] = os.path.join(out_dir, "convergence_curve.png")
        fig_curve(figs["convergence_curve"], grid)
        figs["direction_map"] = os.path.join(out_dir, "direction_map.png")
        fig_direction_map(figs["direction_map"], grid)
    if thr:
        first = {(r["mode"], r["direction"]): r for r in thr_runs if r.get("bisection_step") == 0}
        figs["threshold_hist"] = os.path.join(out_dir, "threshold_hist.png")
        fig_threshold_hist(figs["threshold_hist"], thr, res["thresholds"])
        figs["threshold_vs_direction"] = os.path.join(out_dir, "threshold_vs_direction.png")
        fig_threshold_vs_direction(figs["threshold_vs_direction"], thr, first)
    res["figures"] = figs

    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(out_dir, "SUMMARY_tables.md"), "w") as f:
        f.write(tables_markdown(res))
    return res


def tables_markdown(res) -> str:
    out = ["# Convergence study: tables (generated by od_stack/convergence_plots.py)\n"]
    for m, unit in (("combined", "km"), ("position_only", "km"), ("velocity_only", "m/s")):
        if res["curve"][m]:
            out += [f"## {MODE_LABELS[m]}\n", _md_rate_table(res["curve"][m], unit), ""]
    if "thresholds" in res:
        out += ["## Per-direction thresholds (A3)\n",
                "| Mode | Directions | Median | 10th pct | 90th pct | Min | Max | ≥ 100,000 | < 500 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for m, s in res["thresholds"].items():
            out.append(f"| {MODE_LABELS[m]} | {s['directions']} | {s['median_km']:,.0f} | {s['p10_km']:,.0f} | "
                       f"{s['p90_km']:,.0f} | {s['min_km']:,.0f} | {s['max_km']:,.0f} | {s['at_least_100000']} | "
                       f"{s['below_500']} |")
        out.append("")
    out += ["## AUC for predicting failure (combined, d ≥ 10,000 km)\n", "| Variable | AUC | p | failed / converged |",
            "|---|---|---|---|"]
    label = dict(FEATURES)
    for k, a in sorted(res["auc_combined_d_ge_10000"].items(), key=lambda kv: -abs((kv[1]["auc"] or 0.5) - 0.5)):
        out.append(f"| {label[k]} | {a['auc']:.3f} | {a['p']:.1e} | {a['n_failed']} / {a['n_converged']} |")
    lg = res["logit_combined_d_ge_10000"]
    if "coefficients" in lg:
        out += ["", f"## Logistic regression of failure (n = {lg['n']}, {lg['n_failed']} failed; standardized)\n",
                "| Variable | β | 95% bootstrap interval | Excludes 0 |", "|---|---|---|---|"]
        for k, c in lg["coefficients"].items():
            out.append(f"| {k} | {c['beta']:+.2f} | {c['lo']:+.2f} to {c['hi']:+.2f} | {'yes' if c['excludes_zero'] else 'no'} |")
    out += ["", "## Non-monotonic directions on the grid\n"]
    for m, v in res["non_monotonic"].items():
        out.append(f"- {m}: {v['count']} of {v['of']} ({v['directions']})")
    rb = res["robustness"]
    out += ["", f"## Noise robustness (directions 0–19 × obs seeds 10,000–10,004 at 15,000 and 30,000 km)\n",
            f"{rb['outcome_depends_on_noise']} of {rb['cases_with_5_seeds']} (direction, size) cases change outcome "
            "with the noise draw."]
    for v in rb["varying"]:
        out.append(f"- d = {v['size_km']:,.0f} km, direction {v['direction']}: {v['outcomes']}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plots and statistics for the convergence study")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    res = analyze(a.out)
    print(tables_markdown(res))
    print("Figures:", *res["figures"].values(), sep="\n  ")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

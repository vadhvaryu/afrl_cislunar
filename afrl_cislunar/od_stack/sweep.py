"""
Monte Carlo sweeps: how far can the stack be pushed before it breaks, and is the
covariance it reports honest?

    python -m od_stack.sweep                     # both sweeps, settings from config.yaml (sweeps:)
    python -m od_stack.sweep noise               # angle noise vs accuracy / covariance honesty
    python -m od_stack.sweep iod-error           # how close the IOD must land for BLS to converge
    python -m od_stack.sweep noise --seeds 3     # quick look

Each run: fresh noisy angles (same recipe as obs_generator.py: Gaussian noise in degrees
added to the true RA and Dec), then the configured IOD -> estimator, then grading against
the truth. The configured IOD, estimator and propagator are used as-is, so a new IOD
method can be swept by changing config.yaml only. With the truth_plus_error stand-in,
the IOD error size is what the iod-error sweep varies.

Writes to sweeps.dir:  noise_sweep.csv / .png,  iod_error_sweep.csv / .png,  sweep_summary.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys as _sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .contracts import Observations
from .estimation import ESTIMATORS, EstimationFailure
from .frames import EarthMoonSystem, radec_to_los
from .iod import IOD_METHODS, IODFailure
from .propagators import PROPAGATORS
from .run import load_config
from .sources import _read_csv, dense_truth, load_synthetic_obs, load_synthetic_truth
from .uncertainty import consistency

DAY = 86400.0


def resample_observations(truth_path: str, template: Observations, noise_arcsec: float, seed: int) -> Observations:
    """New noisy angles at the same times and observer as `template`, from the noise-free
    angles stored in the truth file. Same noise model as obs_generator.py."""
    rows = _read_csv(truth_path)
    ra_true = np.array([float(r["ra_true_deg"]) for r in rows])
    dec_true = np.array([float(r["dec_true_deg"]) for r in rows])
    rng = np.random.default_rng(seed)
    s_deg = noise_arcsec / 3600.0
    ra = np.radians((ra_true + rng.normal(0, s_deg, len(rows))) % 360.0)
    dec = np.radians(dec_true + rng.normal(0, s_deg, len(rows)))
    sigma = np.full(len(rows), np.radians(s_deg))
    return Observations(t=template.t.copy(), los=radec_to_los(ra, dec), observer=template.observer.copy(),
                        sigma=sigma, ra=ra, dec=dec)


class Sweeper:
    """Builds the estimator and propagator once (JVM, force models), then runs many cases."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sys = EarthMoonSystem(**cfg.get("system", {}))
        self.orekit_data = cfg.get("orekit", {}).get("data_path")
        self.truth_path = cfg["data"].get("truth")
        if not self.truth_path:
            raise SystemExit("Sweeps grade against the truth, so data.truth must be set in config.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.template = load_synthetic_obs(cfg["data"]["observations"], self.sys)
        _, self.tru_rot, self.tru_tau = load_synthetic_truth(self.truth_path, self.sys)
        ecfg = dict(cfg.get("estimation") or {"method": "none"})
        self.estimator = ESTIMATORS[ecfg.pop("method")](self.sys, orekit_data=self.orekit_data, **ecfg)
        pcfg = dict(cfg["propagator"])
        self.prop = PROPAGATORS[pcfg.pop("method")](self.sys, orekit_data=self.orekit_data, **pcfg)
        scfg = cfg.get("sweeps", {})
        self.ok_km = float(scfg.get("converged_if_error_below_km", 1000))
        self.ref_noise = float(scfg.get("threshold_reference_noise_arcsec", 10))
        n = int(scfg.get("check_points", 100))
        self.t_check = np.linspace(self.template.t.min(), self.template.t.max(), n)
        self.truth_check = dense_truth(self.tru_rot, self.tru_tau, self.t_check, self.sys)

    def threshold_km(self, noise_arcsec: float) -> float:
        """'Converged' means BLS finished AND landed within this distance of the truth.
        The error of a good fit grows in proportion to the noise, so the threshold does too
        above the 10" reference level (1,000 km at 10" is ~60x the typical good-fit error)."""
        return self.ok_km * max(1.0, noise_arcsec / self.ref_noise)

    def one(self, noise_arcsec: float, seed: int, iod_overrides: dict | None = None) -> dict:
        icfg = dict(self.cfg["iod"])
        icfg.update(iod_overrides or {})
        icfg.setdefault("seed", seed)
        method = icfg.pop("method")
        row = {"noise_arcsec": noise_arcsec, "seed": seed, "iod_method": method,
               "iod_position_error_km_requested": icfg.get("position_error_km") if method == "truth_plus_error" else None}
        obs = resample_observations(self.truth_path, self.template, noise_arcsec, seed=10_000 + seed)
        t0 = time.time()
        try:
            iod = IOD_METHODS[method](self.sys, orekit_data=self.orekit_data, truth_path=self.truth_path, **icfg)
            x_iod = iod.estimate(obs)
        except IODFailure as e:
            return {**row, "iod_ok": False, "bls_finished": False, "converged": False, "error": str(e)}
        tx = dense_truth(self.tru_rot, self.tru_tau, np.array([x_iod.t]), self.sys)
        row.update(iod_ok=True, iod_position_error_km=float(np.linalg.norm(x_iod.r - tx.r[0]) / 1e3))
        try:
            x0 = self.estimator.refine(obs, x_iod)
        except EstimationFailure as e:
            return {**row, "bls_finished": False, "converged": False, "seconds": time.time() - t0,
                    "error": str(e).split(": ", 1)[-1][:120]}
        tx = dense_truth(self.tru_rot, self.tru_tau, np.array([x0.t]), self.sys)
        err_km = float(np.linalg.norm(x0.r - tx.r[0]) / 1e3)
        row.update(bls_finished=True, iterations=x0.info.get("iterations"),
                   rms_normalized_residual=x0.info.get("rms_normalized_residual"),
                   position_error_km=err_km, velocity_error_m_s=float(np.linalg.norm(x0.v - tx.v[0])),
                   converged=err_km < self.threshold_km(noise_arcsec))
        try:
            est = self.prop.propagate(x0, self.t_check)
            row["max_position_error_km"] = float(np.linalg.norm(est.r - self.truth_check.r, axis=1).max() / 1e3)
        except Exception as e:                          # propagation of a diverged fit can blow up
            row["max_position_error_km"] = float("nan")
            row["error"] = f"propagation: {str(e).splitlines()[0][:100]}"
        row["covariance_ok"] = x0.cov is not None
        if x0.cov is not None:
            row["sigma_position_km"] = float(np.sqrt(np.trace(x0.cov[:3, :3])) / 1e3)
            c = consistency(x0, tx.r[0], tx.v[0])
            row.update(nees_6d=c["nees_6d"], nees_position_3d=c["nees_position_3d"],
                       inside_95pct_6d=c["inside_95pct_ellipsoid_6d"])
        row["seconds"] = time.time() - t0
        return row


# ----------------------------------------------------------------------------- aggregation

def _wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a success fraction k/n (sensible even at 0/n and n/n)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def aggregate(rows: list[dict], key: str) -> list[dict]:
    out = []
    for v in sorted({r[key] for r in rows}):
        g = [r for r in rows if r[key] == v]
        conv = [r for r in g if r.get("converged")]
        cov = [r for r in conv if r.get("covariance_ok")]
        a = {key: v, "runs": len(g),
             "bls_finished": sum(bool(r.get("bls_finished")) for r in g),
             "converged": len(conv), "covariance_ok": len(cov)}
        a["converged_fraction"], a["converged_lo"], a["converged_hi"] = _wilson(len(conv), len(g))
        if conv:
            e = np.array([r["position_error_km"] for r in conv])
            a.update(median_error_km=float(np.median(e)), p25_error_km=float(np.percentile(e, 25)),
                     p75_error_km=float(np.percentile(e, 75)),
                     median_max_error_km=float(np.nanmedian([r.get("max_position_error_km", np.nan) for r in conv])),
                     median_iterations=float(np.median([r["iterations"] for r in conv if r.get("iterations") is not None]
                                                       or [np.nan])))
        if cov:
            inside = sum(bool(r["inside_95pct_6d"]) for r in cov)
            a["median_sigma_km"] = float(np.median([r["sigma_position_km"] for r in cov]))
            a["inside_95pct"] = inside
            a["inside_95pct_fraction"], a["inside_lo"], a["inside_hi"] = _wilson(inside, len(cov))
            a["mean_nees_6d"] = float(np.mean([r["nees_6d"] for r in cov]))
        out.append(a)
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(x, f="{:,.1f}"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def markdown_table(agg: list[dict], key: str, label: str) -> str:
    lines = [f"| {label} | Converged | Median error (km) | Median 1σ (km) | Inside 95% region | Mean NEES (expect 6) |",
             "|---|---|---|---|---|---|"]
    for a in agg:
        inside = (f"{a['inside_95pct']}/{a['covariance_ok']}" if a.get("covariance_ok")
                  else "covariance fails" if a["converged"] else "—")
        lines.append(f"| {a[key]:,g} | {a['converged']}/{a['runs']} | {_fmt(a.get('median_error_km'))} | "
                     f"{_fmt(a.get('median_sigma_km'))} | {inside} | {_fmt(a.get('mean_nees_6d'), '{:.1f}')} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures

def plot_noise(path: str, rows: list[dict], agg: list[dict], ok_km: float) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), dpi=110)
    x = np.array([a["noise_arcsec"] for a in agg], float)

    ax = axs[0]
    have = [a for a in agg if "median_error_km" in a]
    xs = np.array([a["noise_arcsec"] for a in have], float)
    ax.fill_between(xs, [a["p25_error_km"] for a in have], [a["p75_error_km"] for a in have],
                    color="tab:orange", alpha=0.2, label="True error, middle 50%")
    ax.plot(xs, [a["median_error_km"] for a in have], "o-", color="tab:orange", label="True error, median")
    hs = [a for a in have if "median_sigma_km" in a]
    ax.plot([a["noise_arcsec"] for a in hs], [a["median_sigma_km"] for a in hs], "s--", color="tab:blue",
            label="Reported 1σ, median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Angle noise, 1σ (arcsec)"); ax.set_ylabel("Position at epoch (km)")
    ax.set_title("Accuracy vs noise (converged runs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    p = np.array([a["converged_fraction"] for a in agg])
    ax.errorbar(x, p, yerr=[p - [a["converged_lo"] for a in agg], [a["converged_hi"] for a in agg] - p],
                fmt="o-", color="tab:green", capsize=3, label=f"Converged (error < {ok_km:,.0f} km × noise/10″)")
    hc = [a for a in agg if a.get("covariance_ok")]
    if hc:
        q = np.array([a["inside_95pct_fraction"] for a in hc])
        ax.errorbar([a["noise_arcsec"] for a in hc], q,
                    yerr=[q - [a["inside_lo"] for a in hc], [a["inside_hi"] for a in hc] - q],
                    fmt="s-", color="tab:purple", capsize=3, label="Truth inside 95% region")
    for a in agg:
        if a["converged"] and a["covariance_ok"] < a["converged"]:
            ax.annotate(f"cov. failed\n{a['converged'] - a['covariance_ok']}/{a['converged']}",
                        (a["noise_arcsec"], 0.05), ha="center", fontsize=7, color="tab:red")
    ax.axhline(0.95, color="0.4", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Angle noise, 1σ (arcsec)"); ax.set_ylabel("Fraction of runs (95% interval)")
    ax.set_title("Where it breaks")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")

    ax = axs[2]
    pts = [(r["noise_arcsec"], r["nees_6d"]) for r in rows if r.get("converged") and r.get("covariance_ok")]
    if pts:
        jit = np.exp(np.random.default_rng(0).uniform(-0.08, 0.08, len(pts)))
        ax.scatter(np.array([p_[0] for p_ in pts]) * jit, [max(p_[1], 1e-3) for p_ in pts], s=12,
                   color="tab:purple", alpha=0.6, label="One run")
    for lvl, ls in ((0.95, "--"), (0.9973, ":")):
        ax.axhline(chi2.ppf(lvl, 6), color="0.3", ls=ls, lw=1, label=f"χ²(6) {lvl:.2%} bound")
    ax.axhline(6, color="0.6", lw=0.8, label="Expected value (6)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Angle noise, 1σ (arcsec)"); ax.set_ylabel("NEES = eᵀ P⁻¹ e at epoch (6 states)")
    ax.set_title("Covariance honesty per run")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Noise sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_iod(path: str, rows: list[dict], agg: list[dict], ok_km: float) -> None:
    key = "iod_position_error_km_requested"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110)
    x = np.array([a[key] for a in agg], float)
    p = np.array([a["converged_fraction"] for a in agg])
    ax = axs[0]
    ax.errorbar(x, p, yerr=[p - [a["converged_lo"] for a in agg], [a["converged_hi"] for a in agg] - p],
                fmt="o-", color="tab:green", capsize=3)
    for a in agg:
        ax.annotate(f"{a['converged']}/{a['runs']}", (a[key], a["converged_fraction"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("IOD position error (km)"); ax.set_ylabel("Fraction converged (95% interval)")
    ax.set_title(f"How close the IOD must land (converged = final error < {ok_km:,.0f} km)")
    ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    for r in rows:
        if r.get("bls_finished"):
            ax.scatter(r[key] * np.exp(np.random.default_rng(r["seed"]).uniform(-0.08, 0.08)),
                       max(r["position_error_km"], 1e-2), s=14,
                       color="tab:green" if r["converged"] else "tab:red", alpha=0.7)
    ax.scatter([], [], color="tab:green", s=14, label="Converged")
    ax.scatter([], [], color="tab:red", s=14, label="Finished, wrong answer")
    nf = sum(1 for r in rows if not r.get("bls_finished"))
    if nf:
        ax.scatter([], [], marker="x", color="k", label=f"BLS gave up ({nf} runs, not shown)")
    lim = [min(x) / 2, max(x) * 2]
    ax.plot(lim, lim, color="0.5", ls=":", lw=1, label="No improvement")
    ax.axhline(ok_km, color="0.4", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("IOD position error (km)"); ax.set_ylabel("Error after BLS (km)")
    ax.set_title("Starting error vs final error, per run")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("IOD-error (convergence basin) sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver

def _progress(i, n, row):
    tag = "ok" if row.get("converged") else ("wrong answer" if row.get("bls_finished") else "FAILED")
    print(f"\r  run {i}/{n}  last: {tag:<13}", end="", flush=True)


def noise_sweep(sw: Sweeper, levels, seeds: int, out_dir: str) -> dict:
    scfg = sw.cfg.get("sweeps", {})
    e_km = float(scfg.get("noise_sweep_iod_error_km", 1000))
    ov = {}
    if sw.cfg["iod"]["method"] == "truth_plus_error":
        ov = {"position_error_km": e_km, "velocity_error_m_s": e_km * float(scfg.get("velocity_per_position_error", 0.01))}
    rows, n, i = [], len(levels) * seeds, 0
    print(f"Noise sweep: {len(levels)} levels x {seeds} seeds")
    for lv in levels:
        for s in range(seeds):
            i += 1
            rows.append(sw.one(float(lv), s, {**ov, "seed": s}))
            _progress(i, n, rows[-1])
    print()
    agg = aggregate(rows, "noise_arcsec")
    write_csv(os.path.join(out_dir, "noise_sweep.csv"), rows)
    plot_noise(os.path.join(out_dir, "noise_sweep.png"), rows, agg, sw.ok_km)
    table = markdown_table(agg, "noise_arcsec", "Noise (″)")
    print(table)
    return {"settings": {"iod_error_km": e_km, "seeds": seeds}, "by_level": agg, "table_markdown": table}


def iod_error_sweep(sw: Sweeper, errors_km, seeds: int, out_dir: str) -> dict:
    if sw.cfg["iod"]["method"] != "truth_plus_error":
        raise SystemExit("The IOD-error sweep needs iod.method: truth_plus_error (it sets the IOD error size)")
    scfg = sw.cfg.get("sweeps", {})
    noise = float(scfg.get("iod_sweep_noise_arcsec", 10))
    k = float(scfg.get("velocity_per_position_error", 0.01))
    rows, n, i = [], len(errors_km) * seeds, 0
    print(f"IOD-error sweep: {len(errors_km)} sizes x {seeds} seeds at {noise:g}″ noise")
    for e in errors_km:
        for s in range(seeds):
            i += 1
            rows.append(sw.one(noise, s, {"position_error_km": float(e), "velocity_error_m_s": float(e) * k, "seed": s}))
            _progress(i, n, rows[-1])
    print()
    agg = aggregate(rows, "iod_position_error_km_requested")
    write_csv(os.path.join(out_dir, "iod_error_sweep.csv"), rows)
    plot_iod(os.path.join(out_dir, "iod_error_sweep.png"), rows, agg, sw.ok_km)
    table = markdown_table(agg, "iod_position_error_km_requested", "IOD error (km)")
    print(table)
    return {"settings": {"noise_arcsec": noise, "velocity_m_s_per_km": k, "seeds": seeds}, "by_size": agg,
            "table_markdown": table}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Monte Carlo sweeps for the OD stack")
    p.add_argument("which", nargs="?", choices=["noise", "iod-error", "all"], default="all")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seeds", type=int, help="runs per setting (overrides sweeps.seeds)")
    p.add_argument("--levels", type=float, nargs="+", help="noise levels in arcsec (overrides config)")
    p.add_argument("--errors", type=float, nargs="+", help="IOD position errors in km (overrides config)")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    scfg = cfg.setdefault("sweeps", {})
    seeds = a.seeds or int(scfg.get("seeds", 20))
    out_dir = scfg.get("dir", "results/sweeps")
    os.makedirs(out_dir, exist_ok=True)
    sw = Sweeper(copy.deepcopy(cfg))

    summary_path = os.path.join(out_dir, "sweep_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary["stack"] = {"iod": cfg["iod"]["method"], "estimator": sw.estimator.name, "propagator": sw.prop.name,
                        "converged_if_error_below_km": sw.ok_km,
                        "threshold_reference_noise_arcsec": sw.ref_noise}
    t0 = time.time()
    if a.which in ("noise", "all"):
        summary["noise_sweep"] = noise_sweep(sw, a.levels or scfg.get("noise_arcsec", [1, 10, 100]), seeds, out_dir)
    if a.which in ("iod-error", "all"):
        summary["iod_error_sweep"] = iod_error_sweep(
            sw, a.errors or scfg.get("iod_position_error_km", [100, 1000, 10000]), seeds, out_dir)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Done in {time.time() - t0:,.0f} s. Files in {out_dir}/")
    return 0


if __name__ == "__main__":
    _sys.exit(main())

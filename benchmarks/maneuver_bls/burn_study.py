"""Batch least squares with burn parameters: does fitting collocation's burns recover a maneuvering orbit?

The baseline halo with one impulsive burn at day 3.1, observed every half day for 8 days from Earth's
center (17 observations). For each burn size, noise level and noise draw, three ways to determine it:

  coasting       range search -> cr3bp_bls without burns (what the stack did before)
  collocation    collocation (mode impulsive) alone, no least squares
  burn_bls       collocation (mode impulsive) -> cr3bp_bls fitting each burn's velocity change and time

    python benchmarks/maneuver_bls/burn_study.py --out outputs/benchmarks/maneuver_bls --workers 3
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

DIRECTION = np.array([3.0, -2.0, 3.0]) / np.linalg.norm([3.0, -2.0, 3.0])
BURN_DAY, TIMES = 3.1, [round(0.5 * i, 1) for i in range(17)]
SIZES_M_S, NOISES, SEEDS = (1.0, 5.0, 20.0), (1.0, 10.0), range(10)
METHODS = {
    "coasting": ({"method": "range_search"}, {"method": "cr3bp_bls", "options": {"maneuvers": "none"}}),
    "collocation": ({"method": "collocation", "options": {"mode": "impulsive"}}, {"method": "none"}),
    "burn_bls": ({"method": "collocation", "options": {"mode": "impulsive"}}, {"method": "cr3bp_bls"}),
}
FIELDS = ["size_m_s", "noise_arcsec", "seed", "method", "ok", "error", "seconds", "rms_sigma", "epoch_error_km",
          "max_error_km", "sigma_position_km", "burns_found", "dv_error_m_s", "sigma_dv_m_s", "t_error_h", "sigma_t_h"]


def scenario(size, noise, seed, method, out):
    iod, est = METHODS[method]
    return {"schema_version": 1, "run": {"seed": int(seed)},
            "simulation": {"truth": {"maneuvers": [{"time_days": BURN_DAY, "delta_v_m_s": (size * DIRECTION).tolist()}]},
                           "observations": {"times_days": TIMES, "noise": {"sigma_arcsec": float(noise)}}},
            "determination": {"iod": {**iod, "epoch_index": 8, "fallback": []}, "estimator": est},
            "prediction": {"backend": "scipy", "start_days": 0, "end_days": 8, "points": 200},
            "output": {"directory": out, "plots": False}}


def run_case(job):
    from od_stack.run import run

    row = {k: job[k] for k in ("size_m_s", "noise_arcsec", "seed", "method")}
    t0 = time.perf_counter()
    try:
        s = run(scenario(job["size_m_s"], job["noise_arcsec"], job["seed"], job["method"], job["out"]), make_files=False)
    except Exception as e:                                   # a failed fit is a result, not a crash
        row.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0]}"[:200], seconds=time.perf_counter() - t0)
        return row
    row.update(ok=True, seconds=time.perf_counter() - t0, rms_sigma=s["info"].get("rms_normalized_residual"),
               epoch_error_km=s["position_error_km"], max_error_km=s["max_position_error_km"],
               sigma_position_km=s.get("sigma_position_km"))
    burns = s["info"].get("maneuvers") or []
    if job["method"] == "collocation" and burns:            # its raw mesh impulses: group them into burns first
        from od_stack.estimation import cluster_impulses
        burns = [{"t_s": t, "delta_v_m_s": dv.tolist()}
                 for t, dv in cluster_impulses(burns, min_burn_m_s=1e-9, keep_fraction=0.0)]
    row["burns_found"] = len(burns)
    if burns:
        true_dv = job["size_m_s"] * DIRECTION
        b = min(burns, key=lambda m: abs(m["t_s"] / 86400 - BURN_DAY))     # the one nearest the true burn
        row.update(dv_error_m_s=float(np.linalg.norm(np.array(b["delta_v_m_s"]) - true_dv)),
                   sigma_dv_m_s=b.get("sigma_delta_v_m_s"), t_error_h=(b["t_s"] / 86400 - BURN_DAY) * 24,
                   sigma_t_h=(b.get("sigma_t_s") or np.nan) / 3600)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/benchmarks/maneuver_bls")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), help="re-run only these; other rows are kept")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    methods = a.methods or list(METHODS)
    jobs = [{"size_m_s": z, "noise_arcsec": n, "seed": s, "method": m, "out": str(out / "_scratch")}
            for z in SIZES_M_S for n in NOISES for s in SEEDS for m in methods]
    kept = []
    if a.methods and (out / "runs.csv").exists():
        with open(out / "runs.csv", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["method"] not in methods]
    rows = list(kept)
    with mp.get_context("spawn").Pool(a.workers) as pool, open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)
        for i, row in enumerate(pool.imap_unordered(run_case, jobs), 1):
            rows.append(row)
            w.writerow({k: row.get(k) for k in FIELDS})
            f.flush()
            print(f"\r  {i}/{len(jobs)}", end="", flush=True)
    print()
    print(summary(rows))
    (out / "summary.md").write_text(summary(rows), encoding="utf-8")


def summary(rows):
    f = lambda v: np.nan if v in (None, "") else float(v)
    lines = ["| Burn (m/s) | Noise (″) | Method | Fit passed 3σ | Median error at epoch (km) | Median worst error over 8 days (km) "
             "| Median burn Δv error (m/s) | Median burn time error (h) | Δv error / σ (median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for z in SIZES_M_S:
        for n in NOISES:
            for m in METHODS:
                g = [r for r in rows if f(r["size_m_s"]) == z and f(r["noise_arcsec"]) == n and r["method"] == m]
                ok = [r for r in g if r["ok"] in (True, "True")]
                med = lambda key: np.nanmedian([abs(f(r.get(key))) for r in ok]) if ok else np.nan
                ratio = [f(r.get("dv_error_m_s")) / f(r.get("sigma_dv_m_s")) for r in ok
                         if f(r.get("sigma_dv_m_s")) > 0]
                fmt = lambda v, p=1: "—" if not np.isfinite(v) else f"{v:,.{p}f}"
                lines.append(f"| {z:g} | {n:g} | {m} | {len(ok)}/{len(g)} | {fmt(med('epoch_error_km'))} | "
                             f"{fmt(med('max_error_km'))} | {fmt(med('dv_error_m_s'), 2)} | {fmt(med('t_error_h'))} | "
                             f"{fmt(np.nanmedian(ratio) if ratio else np.nan, 2)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

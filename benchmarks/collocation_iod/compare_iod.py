"""Collocation IOD against the stack's other IOD methods, on identical synthetic cases.

Every method sees the same observations: the plug-and-play halo scenario, regenerated for each
(arc, noise level, seed) by the stack's own synthetic generator. Each case runs the product
pipeline's stages: IOD, then Orekit batch least squares (BLS) from that IOD. A BLS fit started
from the true state, on the same observations, is the reference: "converged" means BLS from
the IOD reached that same fit (within 10 km) with residuals at most 3 sigma.

    python benchmarks/collocation_iod/compare_iod.py --out outputs/benchmarks/collocation_iod
    python benchmarks/collocation_iod/compare_iod.py --quick      # 1 seed, 10" only (smoke test)

Needs both packages installed (pip install -e "./packages/cislunar-stack[orekit]"
-e "./packages/collocation-iod[stack]") and OREKIT_DATA. Writes runs.csv (one row per method
and case; resumable), summary.md and figures to --out.
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "scenarios" / "plug_and_play" / "config.yaml"

# Observation schedules (days since the frame origin) and the epoch observation index.
ARCS = {
    "8 days, 5 obs": ([0, 2, 4, 6, 8], 2),            # the bundled scenario
    "8 days, 9 obs": ([0, 1, 2, 3, 4, 5, 6, 7, 8], 4),
    "1 day, 5 obs": ([3, 3.25, 3.5, 3.75, 4], 2),
    "6 hours, 5 obs": ([4, 4.0625, 4.125, 4.1875, 4.25], 2),
}
NOISE_ARCSEC = (1.0, 10.0, 100.0)
SEEDS = range(10)
METHODS = ("collocation", "range_search", "gooding", "gauss", "laplace")
PRODUCT = "product"                # collocation, falling back to range search
PRODUCT_RS = "product_rs"          # range search, falling back to collocation
PRODUCTS = {PRODUCT: ("collocation", ["range_search"]), PRODUCT_RS: ("range_search", ["collocation"])}
REFERENCE = "truth_start"          # BLS from the true state: the fit every IOD should lead to

# Orbit study (Keira's question, on this harness): periodic states in our CR3BP, corrected from the
# MIT-licensed cislunar-custody seeds (od_stack/orbit_catalog.py on the cr3bp-iod branch).
ORBITS = {
    "L1 halo (baseline)": [0.8727638863136087, 0.0, 0.190505, 0.0, 0.23598602025575388, 0.0],
    "L1 halo, small": [0.8247698373087116, 0.0, 0.06773, 0.0, 0.17970388247374283, 0.0],
    "L2 halo, large": [1.0274662443315938, 0.0, 0.07349, 0.0, 0.4470848771261125, 0.0],
    "NRHO 9:2": [0.9873801492146178, 0.0, 0.008439917431935787, 0.0, 1.66728339697783, 0.0],
    "DRO": [1.17539, 0.0, 0.0, 0.0, -0.49424426634269614, 0.0],
}
ORBIT_ARC, ORBIT_NOISE, ORBIT_SEEDS = "8 days, 5 obs", 10.0, range(20)
CONVERGED_KM, GATE_SIGMA = 10.0, 3.0

FIELDS = ["orbit", "arc", "noise_arcsec", "seed", "method", "iod_used", "attempts", "iod_ok", "iod_error", "iod_s", "iod_position_error_km",
          "iod_velocity_error_m_s", "bls_ok", "bls_error", "bls_s", "bls_iterations", "rms_sigma",
          "final_position_error_km", "r_x", "r_y", "r_z", "collocation_objective", "collocation_guess", "total_s",
          "sigma_position_km", "alternatives", "nearest_alternative_km"]


def config_for(arc: str, noise: float, seed: int, method: str, orbit: str | None = None) -> dict:
    """The bundled scenario with this case's orbit, schedule, noise, seed and IOD method."""
    import yaml

    with open(SCENARIO, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    times, epoch_index = ARCS[arc]
    cfg["run"]["seed"] = int(seed)
    if orbit:
        cfg["simulation"]["truth"]["initial_state"]["values"] = list(ORBITS[orbit])
    cfg["simulation"]["observations"]["times_days"] = list(times)
    cfg["simulation"]["observations"]["noise"]["sigma_arcsec"] = float(noise)
    iod = cfg["determination"]["iod"]
    iod["epoch_index"], iod["options"] = epoch_index, {}          # each method's default options
    if method in PRODUCTS:
        iod["method"], iod["fallback"] = PRODUCTS[method]
    else:
        iod["method"], iod["fallback"] = ("range_search" if method == REFERENCE else method), []
        if method in ("gooding", "gauss", "laplace"):    # first, epoch and last observation: fit at the same epoch
            iod["options"] = {"obs_indices": [0, epoch_index, len(times) - 1]}
    cfg["output"] = {"directory": str(ROOT / "outputs" / "benchmarks" / "_scratch"), "plots": False,
                     "observations": False}
    return cfg


def _sigma_km(x) -> float | None:
    """1-sigma position uncertainty the fit reports (root of the covariance trace), or None if it has none."""
    return None if x.cov is None else float(np.sqrt(np.trace(np.asarray(x.cov)[:3, :3])) / 1e3)


def _alternatives(info: dict, truth_r) -> dict:
    """How many other orbits range search found that fit as well, and the nearest one's distance to the truth."""
    alts = (info or {}).get("alternatives")
    if alts is None:
        return {}
    d = [float(np.linalg.norm(np.array(a["position_m"]) - truth_r) / 1e3) for a in alts]
    return {"alternatives": len(alts), "nearest_alternative_km": min(d) if d else None}


def run_case(job: dict) -> dict:
    """One method on one case. Truth is used only for grading and for the reference fit."""
    from od_stack.config import prepare
    from od_stack.contracts import StateEstimate
    from od_stack.estimation import EstimationFailure
    from od_stack.frames import EarthMoonSystem
    from od_stack.iod import IODFailure
    from od_stack.run import build_stages, size_collocation_mesh, validate_observations
    from od_stack.sources import select_observations
    from od_stack.synthetic import generate

    from od_stack.run import determine

    row = {k: job.get(k) for k in ("orbit", "arc", "noise_arcsec", "seed", "method")}
    cfg = prepare(config_for(job["arc"], job["noise_arcsec"], job["seed"], job["method"], job.get("orbit")))
    sys_ = EarthMoonSystem(**cfg["system"])
    obs, truth = generate(cfg["simulation"], sys_)
    obs = select_observations(obs, cfg)
    validate_observations(obs, cfg)
    size_collocation_mesh(cfg, obs)                  # as run() does
    iod, estimator, _ = build_stages(cfg, sys_)
    t_epoch = float(obs.t[cfg["iod"]["epoch_index"]])
    tru = truth.at(np.array([t_epoch]))

    if job["method"] in PRODUCTS:                    # the whole determination step, fallback included
        t0 = time.perf_counter()
        try:
            x_iod, x0, used, attempts = determine(cfg, sys_, obs, iod, estimator)
        except EstimationFailure as e:
            # "IOD ran" if any method produced a state, even if its fit was then rejected
            ran = "(estimation)" in str(e)
            row.update(iod_ok=ran, bls_ok=False, bls_error=str(e)[:200], total_s=time.perf_counter() - t0)
            return row
        tx, tf = truth.at(np.array([x_iod.t])), truth.at(np.array([x0.t]))
        row.update(iod_ok=True, bls_ok=True, iod_used=used.name, attempts=len(attempts), total_s=time.perf_counter() - t0,
                   iod_position_error_km=float(np.linalg.norm(x_iod.r - tx.r[0]) / 1e3),
                   iod_velocity_error_m_s=float(np.linalg.norm(x_iod.v - tx.v[0])),
                   bls_iterations=x0.info.get("iterations"), rms_sigma=x0.info.get("rms_normalized_residual"),
                   final_position_error_km=float(np.linalg.norm(x0.r - tf.r[0]) / 1e3), sigma_position_km=_sigma_km(x0),
                   r_x=float(x0.r[0]), r_y=float(x0.r[1]), r_z=float(x0.r[2]), **_alternatives(x0.info, tf.r[0]))
        return row

    t0 = time.perf_counter()
    try:
        if job["method"] == REFERENCE:
            x_iod = StateEstimate(t=t_epoch, r=tru.r[0], v=tru.v[0], method=REFERENCE)
        else:
            x_iod = iod.estimate(obs)
        row["iod_ok"] = True
    except (IODFailure, ValueError, RuntimeError) as e:
        row.update(iod_ok=False, iod_error=str(e)[:200], iod_s=time.perf_counter() - t0)
        return row
    row["iod_s"] = time.perf_counter() - t0
    tx = truth.at(np.array([x_iod.t]))
    row["iod_position_error_km"] = float(np.linalg.norm(x_iod.r - tx.r[0]) / 1e3)
    row["iod_velocity_error_m_s"] = float(np.linalg.norm(x_iod.v - tx.v[0]))
    diag = (x_iod.info or {}).get("collocation") or {}
    row["collocation_objective"] = diag.get("objective")
    row["collocation_guess"] = diag.get("initial_guess")      # the start that produced the chosen solution
    row.update(_alternatives(x_iod.info, tx.r[0]))

    t0 = time.perf_counter()
    try:
        estimator.gate = None              # single methods: grade the raw fit; the check is applied in grade()
        x0 = estimator.refine(obs, x_iod)
        row["bls_ok"] = True
    except (EstimationFailure, RuntimeError) as e:
        row.update(bls_ok=False, bls_error=str(e)[:200], bls_s=time.perf_counter() - t0)
        return row
    row["bls_s"] = time.perf_counter() - t0
    tf = truth.at(np.array([x0.t]))
    row.update(bls_iterations=x0.info.get("iterations"), rms_sigma=x0.info.get("rms_normalized_residual"),
               final_position_error_km=float(np.linalg.norm(x0.r - tf.r[0]) / 1e3), sigma_position_km=_sigma_km(x0),
               r_x=float(x0.r[0]), r_y=float(x0.r[1]), r_z=float(x0.r[2]))
    return row


def _safe(job):
    try:
        return run_case(job)
    except Exception as e:                       # one broken case must not stop the benchmark
        return {**{k: job.get(k) for k in ("orbit", "arc", "noise_arcsec", "seed", "method")}, "iod_ok": False,
                "iod_error": f"{type(e).__name__}: {e}"[:200]}


def jobs_for(quick: bool, study: str = "arcs", methods=None) -> list[dict]:
    if study == "orbits":
        ms = methods or ("collocation", "range_search", *PRODUCTS)
        return [{"orbit": o, "arc": ORBIT_ARC, "noise_arcsec": ORBIT_NOISE, "seed": s, "method": m}
                for o in ORBITS for s in (range(1) if quick else ORBIT_SEEDS) for m in (REFERENCE, *ms)]
    arcs, noises, seeds = (ARCS, (10.0,), range(1)) if quick else (ARCS, NOISE_ARCSEC, SEEDS)
    return [{"orbit": None, "arc": a, "noise_arcsec": n, "seed": s, "method": m}
            for a in arcs for n in noises for s in seeds for m in (REFERENCE, *(methods or (*METHODS, *PRODUCTS)))]


def run_all(out: Path, workers: int, quick: bool, study: str = "arcs", methods=None) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.csv"
    key = lambda r: (r.get("orbit") or "", r["arc"], float(r["noise_arcsec"]), int(r["seed"]), r["method"])
    done = []
    if path.exists():
        with open(path, newline="") as f:
            done = list(csv.DictReader(f))
    have = {key(r) for r in done}
    todo = [j for j in jobs_for(quick, study, methods) if key(j) not in have]
    print(f"{len(todo)} runs to do ({len(have)} already in {path}), {workers} workers")
    new = not path.exists()
    with open(path, "a", newline="") as f, mp.get_context("spawn").Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        t0 = time.time()
        for i, row in enumerate(pool.imap_unordered(_safe, todo), 1):
            w.writerow({k: row.get(k) for k in FIELDS})
            f.flush()
            print(f"\r  {i}/{len(todo)} done, {time.time() - t0:,.0f} s", end="", flush=True)
    print()
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- summary

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def grade(rows: list[dict]) -> list[dict]:
    """Mark each run converged if its BLS fit matches the truth-started fit of the same case."""
    case = lambda r: (r.get("orbit") or "", r["arc"], _f(r["noise_arcsec"]), int(r["seed"]))
    ref = {case(r): r for r in rows if r["method"] == REFERENCE}
    for r in rows:
        f = ref.get(case(r))
        r["converged"] = False
        if r.get("bls_ok") == "True" and f is not None and f.get("bls_ok") == "True":
            d = np.linalg.norm([_f(r[c]) - _f(f[c]) for c in ("r_x", "r_y", "r_z")]) / 1e3
            r["distance_to_reference_km"] = d
            r["converged"] = bool(d <= CONVERGED_KM and _f(r["rms_sigma"]) <= GATE_SIGMA)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    groups = ([(o, a) for o in ORBITS for a in ARCS] if any(r.get("orbit") for r in rows) else [("", a) for a in ARCS])
    for orbit, arc in groups:
        for noise in sorted({_f(r["noise_arcsec"]) for r in rows}):
            for m in (*METHODS, *PRODUCTS, REFERENCE):
                g = [r for r in rows if (r.get("orbit") or "") == orbit and r["arc"] == arc
                     and _f(r["noise_arcsec"]) == noise and r["method"] == m]
                if not g:
                    continue
                ok = [r for r in g if r.get("iod_ok") == "True"]
                conv = [r for r in g if r["converged"]]
                med = lambda key, rs: float(np.nanmedian([_f(r[key]) for r in rs])) if rs else np.nan
                used = sorted({r.get("iod_used") for r in g if r.get("iod_used")})
                out.append({"orbit": orbit, "arc": arc, "noise_arcsec": noise, "method": m, "runs": len(g),
                            "iod_ok": len(ok), "fell_back": sum(r.get("iod_used") not in (None, "", PRODUCTS[m][0])
                                                                for r in g if r.get("bls_ok") == "True")
                            if m in PRODUCTS else "", "iod_used": ", ".join(used),
                            "iod_error_km": med("iod_position_error_km", ok),
                            "iod_error_p90_km": float(np.nanpercentile([_f(r["iod_position_error_km"]) for r in ok], 90))
                            if ok else np.nan,
                            "within_10000_km": sum(_f(r["iod_position_error_km"]) <= 10_000 for r in ok),
                            "converged": len(conv), "bls_iterations": med("bls_iterations", conv),
                            "iod_s": med("total_s" if m in PRODUCTS else "iod_s", g),
                            "final_error_km": med("final_position_error_km", conv)})
    return out


def markdown(summary: list[dict]) -> str:
    lines = ["| Orbit | Arc | Noise (″) | Method | IOD ran | Median IOD error (km) | 90th pct (km) | Within 10,000 km "
             "| BLS converged | Used the fallback | Median BLS iterations | Median IOD time (s)* |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        fmt = lambda v, p=0: "—" if not np.isfinite(v) else f"{v:,.{p}f}"
        lines.append(f"| {s['orbit'] or 'baseline'} | {s['arc']} | {s['noise_arcsec']:g} | {s['method']} | {s['iod_ok']}/{s['runs']} | "
                     f"{fmt(s['iod_error_km'], 1)} | {fmt(s['iod_error_p90_km'])} | {s['within_10000_km']} | "
                     f"{s['converged']}/{s['runs']} | {s['fell_back']} | {fmt(s['bls_iterations'])} | {fmt(s['iod_s'], 1)} |")
    return "\n".join(lines) + "\n\n*For the product rows: IOD and least squares together, fallbacks included.\n"


def plot(rows: list[dict], summary: list[dict], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    colors = {"collocation": "#004B8D", "range_search": "#D55E00", "gooding": "#9AA5B1", "gauss": "#CC79A7",
              "laplace": "#E69F00"}
    noises = sorted({s["noise_arcsec"] for s in summary})
    fig, axs = plt.subplots(1, len(noises), figsize=(5.2 * len(noises), 4.6), dpi=130, sharey=True, squeeze=False)
    for ax, noise in zip(axs[0], noises):
        x = np.arange(len(ARCS))
        for j, m in enumerate(("collocation", "range_search")):
            ys, lo, hi = [], [], []
            for arc in ARCS:
                e = [_f(r["iod_position_error_km"]) for r in rows if r["arc"] == arc and r["method"] == m
                     and _f(r["noise_arcsec"]) == noise and r.get("iod_ok") == "True"]
                ys.append(np.median(e) if e else np.nan)
                lo.append(np.percentile(e, 10) if e else np.nan)
                hi.append(np.percentile(e, 90) if e else np.nan)
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.errorbar(x + (j - 0.5) * 0.18, ys, yerr=[ys - lo, hi - ys], fmt="o", capsize=3, color=colors[m],
                        label=m)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(list(ARCS), fontsize=8)
        ax.set_title(f"{noise:g}″ noise")
        ax.grid(alpha=0.3, which="both", axis="y")
        ax.axhline(10_000, color="0.4", ls="--", lw=0.8)
    axs[0][0].set_ylabel("IOD position error at epoch (km); median, 10–90%")
    axs[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("IOD error before batch least squares, 10 noise draws per case (dashed: 10,000 km BLS target)")
    fig.tight_layout()
    paths.append(str(out / "iod_error.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="output folder (default: outputs/benchmarks/collocation_iod[_orbits])")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--study", choices=["arcs", "orbits"], default="arcs",
                    help="arcs: 4 arcs x 3 noise levels x 10 draws (baseline halo); orbits: 5 orbits x 20 draws at 10\"")
    ap.add_argument("--methods", nargs="+", help="subset of methods (the truth-started reference always runs)")
    a = ap.parse_args(argv)
    out = Path(a.out or ROOT / "outputs" / "benchmarks" / ("collocation_iod" + ("_orbits" if a.study == "orbits" else "")))
    rows = grade(run_all(out, a.workers, a.quick, a.study, a.methods))
    summ = summarize(rows)
    text = markdown(summ)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    if a.study == "arcs":
        print("Figures:", *plot(rows, summ, out), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

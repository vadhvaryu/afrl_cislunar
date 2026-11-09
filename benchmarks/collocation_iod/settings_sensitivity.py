"""How much the collocation IOD depends on its own settings: starting guess and mesh size.

Same cases and generator as compare_iod.py (plug-and-play halo, 10" noise), IOD only (no batch
least squares): for each arc and seed, the collocation estimate with each initial guess and
each node count, graded against the truth at the epoch.

    python benchmarks/collocation_iod/settings_sensitivity.py --out outputs/benchmarks/collocation_iod
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_iod import ARCS, config_for  # noqa: E402

VARIANTS = {f"guess={g}": {"initial_guess": g} for g in ("auto", "los", "L1", "L2", "L4")}
VARIANTS.update({f"nodes={n}": {"nodes": n} for n in (40, 160)})
# range_search's distance window (150,000-650,000 km) as collocation's range bound, in DU (384,338 km);
# the default (1e-4, 5) DU allows 38 km to 1.9 million km
VARIANTS["range 150k-650k km"] = {"range_bounds": [150_000 / 384_338, 650_000 / 384_338]}
SEEDS = range(5)


def run(job):
    from od_stack.config import prepare
    from od_stack.frames import EarthMoonSystem
    from od_stack.run import validate_observations
    from od_stack.sources import select_observations
    from od_stack.synthetic import generate
    from collocation_iod.adapters import CollocationIOD

    cfg = config_for(job["arc"], 10.0, job["seed"], "collocation")
    cfg["determination"]["iod"]["options"].update(VARIANTS[job["variant"]])
    scfg = prepare(cfg)
    sys_ = EarthMoonSystem(**scfg["system"])
    obs, truth = generate(scfg["simulation"], sys_)
    obs = select_observations(obs, scfg)
    validate_observations(obs, scfg)
    row = dict(job)
    t0 = time.perf_counter()
    try:
        x = CollocationIOD(sys_, epoch_index=scfg["iod"]["epoch_index"], **scfg["iod"]["collocation"]).estimate(obs)
    except Exception as e:                      # record failures; IODFailure included
        row.update(ok=False, seconds=time.perf_counter() - t0, error=str(e)[:160])
        return row
    tx = truth.at(np.array([x.t]))
    row.update(ok=True, seconds=time.perf_counter() - t0, error_km=float(np.linalg.norm(x.r - tx.r[0]) / 1e3),
               objective=x.info["collocation"].get("objective"), chosen=x.info["collocation"].get("initial_guess"))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"arc": arc, "seed": s, "variant": v} for arc in ARCS for v in VARIANTS for s in SEEDS]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(run, jobs))
    fields = ["arc", "variant", "seed", "ok", "error_km", "seconds", "objective", "chosen", "error"]
    with open(out / "settings_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)
    lines = ["| Arc | Setting | Solved | Median IOD error (km) | Range (km) | Median time (s) |", "|---|---|---|---|---|---|"]
    for arc in ARCS:
        for v in VARIANTS:
            g = [r for r in rows if r["arc"] == arc and r["variant"] == v]
            ok = [r for r in g if r["ok"]]
            e = [r["error_km"] for r in ok]
            lines.append(f"| {arc} | {v} | {len(ok)}/{len(g)} | " + (f"{np.median(e):,.1f} | {min(e):,.0f}–{max(e):,.0f}"
                         if e else "— | —") + f" | {np.median([r['seconds'] for r in g]):.1f} |")
    text = "\n".join(lines) + "\n"
    (out / "settings_sensitivity.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

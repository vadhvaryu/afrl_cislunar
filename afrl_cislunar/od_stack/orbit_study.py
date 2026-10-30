"""
Does the convergence picture change with the orbit? Runs the convergence-study cases on the
other truth orbits from orbit_catalog.py.

    python -m od_stack.orbit_study                         # builds the orbits if needed, then runs
    python -m od_stack.orbit_study --estimators orekit_bls multiple_shooting orekit_bls_robust

Per orbit and estimator: combined-mode starting errors (default 1,000-100,000 km) x direction ids
0-19, one fixed 10" noise draw (obs seed 10,000), 3-sigma residual gate. Same stand-in IOD and
grading as od_stack/convergence.py, applied to each orbit's own truth file.

Outputs in results/orbit_study/: runs.csv (resumable), summary.json/.md, orbits_convergence.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys as _sys

import numpy as np

from . import estimator_compare as ec
from .orbit_catalog import ORBITS, OUT_DIR as ORBIT_DIR, build
from .run import load_config

OUT_DIR = "results/orbit_study"
SIZES_KM = (1_000.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0, 50_000.0, 100_000.0)
DIRS = range(0, 20)
COLORS = {"orekit_bls": "#004B8D", "orekit_bls_robust": "#56B4E9", "multiple_shooting": "#D55E00",
          "multiple_shooting_then_bls": "#CC79A7", "ukf": "#E69F00", "none": "#808F9C"}


def variants(orbits, estimators, orbit_dir=ORBIT_DIR):
    return {f"{o}|{e}": {"method": e, "_data": {"observations": os.path.join(orbit_dir, o, "synthetic_obs.csv"),
                                                 "truth": os.path.join(orbit_dir, o, "synthetic_truth.csv")}}
            for o in orbits for e in estimators}


def plot(summ, catalog, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    orbits = list(dict.fromkeys(v.split("|")[0] for v in summ["grid"]))
    ncol = 3
    nrow = int(np.ceil(len(orbits) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), dpi=110, sharex=True, sharey=True,
                            squeeze=False)
    for ax, o in zip(axs.flat, orbits):
        for v, rows in summ["grid"].items():
            orb, est = v.split("|")
            if orb != o:
                continue
            x = [a["size"] for a in rows]
            p = np.array([a["rate"] for a in rows])
            ax.errorbar(x, p, yerr=[p - [a["lo"] for a in rows], [a["hi"] for a in rows] - p], fmt="o-", ms=4,
                        capsize=3, color=COLORS.get(est), label=est)
        c = catalog.get(o, {})
        ax.set_title(f"{o}: T {c.get('period_days', np.nan):.1f} d, stability {c.get('stability_index', np.nan):,.0f}, "
                     f"perilune {c.get('perilune_km', np.nan):,.0f} km", fontsize=9)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3, which="both")
    for ax in axs.flat[len(orbits):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Starting position error (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Fraction converged")
    axs.flat[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Convergence vs starting error for six cislunar orbits (20 directions, 10″, 3σ gate)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convergence on other cislunar orbits")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--orbits", nargs="+", default=list(ORBITS))
    ap.add_argument("--estimators", nargs="+", default=["orekit_bls", "multiple_shooting"])
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    cat_path = os.path.join(ORBIT_DIR, "catalog.json")
    if not os.path.exists(cat_path) or any(not os.path.isdir(os.path.join(ORBIT_DIR, o)) for o in a.orbits):
        build()
    with open(cat_path) as f:
        catalog = json.load(f)
    cfg = load_config(a.config)
    cfg["iod"]["method"] = "truth_plus_error"
    var = variants(a.orbits, a.estimators)
    summ = ec.run_study(cfg, var, ec.grid_jobs(var, DIRS, SIZES_KM), OUT_DIR, a.workers)
    plot(summ, catalog, os.path.join(OUT_DIR, "orbits_convergence.png"))
    print(ec.markdown(summ))
    print("Figure:", os.path.join(OUT_DIR, "orbits_convergence.png"))
    return 0


if __name__ == "__main__":
    _sys.exit(main())

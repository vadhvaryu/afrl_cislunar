"""
Run the stack:  observations -> IOD -> estimation (BLS) -> propagation -> results.

    python -m od_stack.run                         # uses config.yaml
    python -m od_stack.run --iod gauss             # swap one module from the command line
    python -m od_stack.run --estimation none       # skip Batch Least Squares
    python -m od_stack.run --compare-iod           # run every IOD method and print a table
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys as _sys

import numpy as np
import yaml

from .estimation import ESTIMATORS, EstimationFailure
from .frames import EarthMoonSystem
from .iod import IOD_METHODS, IODFailure
from .propagators import PROPAGATORS
from .results import plot, summarize, write_outputs
from .sources import dense_truth, load_synthetic_obs, load_synthetic_truth

DAY = 86400.0


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, make_files: bool = True) -> dict:
    sys = EarthMoonSystem(**cfg.get("system", {}))
    orekit_data = cfg.get("orekit", {}).get("data_path")

    # 1. Data source -> Observations
    obs = load_synthetic_obs(cfg["data"]["observations"], sys)

    # 2. IOD -> StateEstimate
    icfg = dict(cfg["iod"])
    iod = IOD_METHODS[icfg.pop("method")](sys, orekit_data=orekit_data,
                                          truth_path=cfg["data"].get("truth"), **icfg)
    x_iod = iod.estimate(obs)

    # 3. Estimation (refine with all observations) -> StateEstimate + covariance
    ecfg = dict(cfg.get("estimation") or {"method": "none"})
    estimator = ESTIMATORS[ecfg.pop("method")](sys, orekit_data=orekit_data, **ecfg)
    x0 = estimator.refine(obs, x_iod)

    # 4. Propagator -> Trajectory (dense, over the whole observation span)
    pcfg = dict(cfg["propagator"])
    prop = PROPAGATORS[pcfg.pop("method")](sys, orekit_data=orekit_data, **pcfg)
    n = int(cfg.get("output", {}).get("dense_points", 800))
    t_dense = np.linspace(obs.t.min(), obs.t.max(), n)
    est = prop.propagate(x0, t_dense)

    # 5. Results (only possible when a truth file exists)
    truth_path = cfg["data"].get("truth")
    if not truth_path:
        return {"method": x0.method, "epoch_days": x0.t / DAY, "info": x0.info}
    _, tru_rot, tru_tau = load_synthetic_truth(truth_path, sys)
    truth = dense_truth(tru_rot, tru_tau, t_dense, sys)
    summary = summarize(
        x_iod, x0, dense_truth(tru_rot, tru_tau, np.array([x0.t]), sys), est, truth,
        prop.propagate(x0, obs.t), dense_truth(tru_rot, tru_tau, obs.t, sys))
    summary["estimator"] = estimator.name
    summary["propagator"] = prop.name

    if make_files:
        out_dir = cfg.get("output", {}).get("dir", "results")
        write_outputs(out_dir, summary, est, truth)
        summary["figure"] = plot(out_dir, obs, x0, est, truth, sys)
    return summary


def compare_iod(cfg: dict) -> None:
    print(f"{'IOD method':<16} {'IOD state error':>26} {'max error after estimation':>28}")
    for name in IOD_METHODS:
        c = copy.deepcopy(cfg)
        c["iod"]["method"] = name
        try:
            s = run(c, make_files=False)
            tag = "  (test stand-in)" if name == "truth_plus_error" else ""
            print(f"{name:<16} {s['iod_position_error_km']:>17,.0f} km "
                  f"{s['iod_velocity_error_m_s']:>5,.0f} m/s {s['max_position_error_km']:>18,.1f} km{tag}")
        except (IODFailure, EstimationFailure) as e:
            print(f"{name:<16} FAILED: {str(e).split(': ', 1)[-1]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cislunar IOD -> propagation -> results stack")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--iod", choices=sorted(IOD_METHODS), help="override iod.method")
    p.add_argument("--estimation", choices=sorted(ESTIMATORS), help="override estimation.method")
    p.add_argument("--propagator", choices=sorted(PROPAGATORS), help="override propagator.method")
    p.add_argument("--compare-iod", action="store_true", help="run every IOD method and print a table")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    if a.iod:
        cfg["iod"]["method"] = a.iod
    if a.estimation:
        cfg.setdefault("estimation", {})["method"] = a.estimation
    if a.propagator:
        cfg["propagator"]["method"] = a.propagator

    if a.compare_iod:
        compare_iod(cfg)
        return 0
    try:
        summary = run(cfg)
    except (IODFailure, EstimationFailure) as e:
        print(f"Failed: {e}")
        return 1
    shown = {k: v for k, v in summary.items() if k != "covariance_6x6_SI"}   # full matrix is in summary.json
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    _sys.exit(main())

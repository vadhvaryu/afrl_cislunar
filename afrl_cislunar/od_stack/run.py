"""
Run the full stack, configured by config.yaml:

    1 data -> 2 measurements -> 3 IOD -> 4 orbit -> 5 propagator builder
      -> 6 estimation (BLS) -> 7 propagator -> 8 prediction, Orbit(t) -> 9 results

    python -m od_stack.run                                   # uses config.yaml
    python -m od_stack.run --config other.yaml               # another scenario
    python -m od_stack.run --set data.noise_arcsec=100       # change any setting for one run
    python -m od_stack.run --iod gooding                     # shortcuts for the three method choices
    python -m od_stack.run --compare-iod                     # every IOD method in one table
    python -m od_stack.run --show-config                     # print the full config (with defaults) and stop
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys as _sys

import numpy as np
import yaml

from .config import ConfigError, load_config, prepare, set_value
from .estimation import ESTIMATORS, EstimationFailure
from .frames import EarthMoonSystem
from .iod import IOD_METHODS, IODFailure
from .propagators import PROPAGATORS
from .results import plot, summarize, write_outputs
from .sources import dense_truth, load_observations, load_synthetic_truth

DAY = 86400.0


def build_stages(cfg: dict, sys: EarthMoonSystem):
    """Create the IOD, estimator and propagator objects the config asks for."""
    orekit_data = cfg["orekit"]["data_path"]
    builder = cfg["propagator_builder"]

    icfg = cfg["iod"]
    iod_cls = IOD_METHODS[icfg["method"]]
    iod = iod_cls(sys, orekit_data=orekit_data, truth_path=cfg["data"].get("truth"),
                  epoch_index=icfg["epoch_index"], **icfg.get(iod_cls.config_block, {}))

    ecfg = {k: v for k, v in cfg["estimation"].items() if k != "method"}
    estimator = ESTIMATORS[cfg["estimation"]["method"]](
        sys, orekit_data=orekit_data, builder=builder, orbit_type=cfg["orbit"]["type"],
        station_offset_m=cfg["measurements"]["station_offset_m"], **ecfg)

    prop = PROPAGATORS[cfg["propagator"]["method"]](sys, orekit_data=orekit_data, builder=builder)
    return iod, estimator, prop


def prediction_times(cfg: dict, obs) -> np.ndarray:
    p = cfg["prediction"]
    start = obs.t.min() if p["start_days"] is None else float(p["start_days"]) * DAY
    end = obs.t.max() if p["end_days"] is None else float(p["end_days"]) * DAY
    return np.linspace(start, end, int(p["points"]))


def run(user_cfg: dict, make_files: bool = True) -> dict:
    cfg = prepare(user_cfg)
    sys = EarthMoonSystem(**cfg["system"])

    obs = load_observations(cfg, sys)                         # 1-2  raw data -> measurement list
    iod, estimator, prop = build_stages(cfg, sys)
    x_iod = iod.estimate(obs)                                 # 3    IOD -> rough orbit
    x0 = estimator.refine(obs, x_iod)                         # 4-6  orbit -> builder -> BLS
    t_out = prediction_times(cfg, obs)
    est = prop.propagate(x0, t_out)                           # 7-8  propagator -> Orbit(t)

    truth_path = cfg["data"].get("truth")                     # 9    results
    if not truth_path:
        summary = {"method": x0.method, "epoch_days": x0.t / DAY, "info": x0.info}
        if x0.cov is not None:
            summary["sigma_position_km"] = float(np.sqrt(np.trace(x0.cov[:3, :3])) / 1e3)
        return summary
    _, tru_rot, tru_tau = load_synthetic_truth(truth_path, sys)
    truth = dense_truth(tru_rot, tru_tau, t_out, sys)
    summary = summarize(
        x_iod, x0, dense_truth(tru_rot, tru_tau, np.array([x0.t]), sys), est, truth,
        prop.propagate(x0, obs.t), dense_truth(tru_rot, tru_tau, obs.t, sys))
    summary.update(iod=iod.name, estimator=estimator.name, propagator=prop.name,
                   n_observations=len(obs), prediction_days=[float(t_out[0] / DAY), float(t_out[-1] / DAY)])

    r = cfg["results"]
    if make_files and any(r[k] for k in ("plot", "csv", "summary", "save_config")):
        os.makedirs(r["dir"], exist_ok=True)
        write_outputs(r["dir"], summary, est, truth, write_summary=r["summary"], write_csv=r["csv"])
        if r["plot"]:
            summary["figure"] = plot(r["dir"], obs, x0, est, truth, sys)
        if r["save_config"]:
            with open(os.path.join(r["dir"], "config_used.yaml"), "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
    return summary


def compare_iod(user_cfg: dict) -> None:
    print(f"{'IOD method':<17} {'IOD state error':>24} {'final error at epoch':>22}")
    for name in IOD_METHODS:
        c = copy.deepcopy(user_cfg)
        c.setdefault("iod", {})["method"] = name
        try:
            s = run(c, make_files=False)
            tag = "  (test stand-in)" if name == "truth_plus_error" else ""
            print(f"{name:<17} {s['iod_position_error_km']:>14,.0f} km {s['iod_velocity_error_m_s']:>5,.0f} m/s"
                  f" {s['position_error_km']:>18,.1f} km{tag}")
        except (IODFailure, EstimationFailure) as e:
            print(f"{name:<17} FAILED: {str(e).split(': ', 1)[-1]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cislunar orbit determination stack (see config.yaml)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any setting, e.g. --set data.noise_arcsec=100 (repeatable)")
    p.add_argument("--iod", choices=sorted(IOD_METHODS), help="shortcut for --set iod.method=...")
    p.add_argument("--estimation", choices=sorted(ESTIMATORS), help="shortcut for --set estimation.method=...")
    p.add_argument("--propagator", choices=sorted(PROPAGATORS), help="shortcut for --set propagator.method=...")
    p.add_argument("--compare-iod", action="store_true", help="run every IOD method and print a table")
    p.add_argument("--show-config", action="store_true", help="print the full config (with defaults) and stop")
    a = p.parse_args(argv)

    try:
        cfg = load_config(a.config)
        for s in a.set:
            set_value(cfg, s)
        for key, val in (("iod.method", a.iod), ("estimation.method", a.estimation), ("propagator.method", a.propagator)):
            if val:
                set_value(cfg, f"{key}={val}")
        if a.show_config:
            print(yaml.safe_dump(prepare(cfg), sort_keys=False))
            return 0
        if a.compare_iod:
            prepare(cfg)
            compare_iod(cfg)
            return 0
        summary = run(cfg)
    except ConfigError as e:
        print(e)
        return 2
    except (IODFailure, EstimationFailure) as e:
        print(f"Failed: {e}")
        return 1
    shown = {k: v for k, v in summary.items() if k != "covariance_6x6_SI"}   # full matrix is in summary.json
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    _sys.exit(main())

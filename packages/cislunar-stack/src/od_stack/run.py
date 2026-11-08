"""
Run the full stack, configured by config.yaml:

    1 data -> 2 measurements -> 3 IOD -> 4 orbit -> 5 propagator builder
      -> 6 estimation (BLS) -> 7 propagator -> 8 prediction, Orbit(t) -> 9 results

    python -m od_stack.run                                   # uses config.yaml
    python -m od_stack.run --config other.yaml               # another scenario
    Edit the YAML to change simulation choices; no CLI simulation overrides.
    python -m od_stack.run --show-config                     # print the full config (with defaults) and stop
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys as _sys
from pathlib import Path

import numpy as np
import yaml

from .config import ConfigError, load_config, prepare, set_value
from .estimation import ESTIMATORS, EstimationFailure
from .frames import EarthMoonSystem
from .iod import IOD_METHODS, IODFailure
from .propagators import PROPAGATORS
from .results import plot, plot_estimate, summarize, write_outputs, write_product_outputs
from .sources import dense_truth, load_observations, load_synthetic_truth, select_observations

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
    if end <= start:
        raise ConfigError("Prediction end must be later than start")
    return np.linspace(start, end, int(p["points"]))


def validate_observations(obs, cfg):
    if (len(obs) < 3 or not np.isfinite(obs.t).all() or np.any(np.diff(obs.t) <= 0)
            or not np.isfinite(obs.los).all() or not np.isfinite(obs.observer).all()
            or not np.isfinite(obs.sigma).all() or np.any(obs.sigma < 0)):
        raise ConfigError("Need at least three finite, time-ordered observations with nonnegative uncertainty")
    if not np.allclose(np.linalg.norm(obs.los, axis=1), 1):
        raise ConfigError("Observation lines of sight must be unit vectors")
    if cfg["iod"]["epoch_index"] >= len(obs):
        raise ConfigError("IOD epoch_index lies outside the observation arc")
    if cfg["iod"]["method"] == "range_search" and np.any(obs.sigma <= 0):
        raise ConfigError("range_search requires strictly positive angular uncertainties")
    if cfg["estimation"]["method"] == "orekit_bls":
        if np.any(np.linalg.norm(obs.observer, axis=1) > 1):
            raise ConfigError("orekit_bls currently requires a geocenter observer")
        if np.any(obs.sigma <= 0):
            raise ConfigError("orekit_bls requires strictly positive angular uncertainties")


def run(user_cfg: dict, make_files: bool = True) -> dict:
    cfg = prepare(user_cfg)
    sys = EarthMoonSystem(**cfg["system"])

    truth_at = None
    if cfg["data"]["source"] == "synthetic":
        from .synthetic import generate
        obs, generated_truth = generate(cfg["simulation"], sys)
        obs = select_observations(obs, cfg)
        if cfg["evaluation"]["enabled"]:
            truth_at = generated_truth.at
    else:
        obs = load_observations(cfg, sys)
        truth_path = cfg["data"].get("truth")
        if cfg["evaluation"]["enabled"] and truth_path:
            _, tru_rot, tru_tau = load_synthetic_truth(truth_path, sys)
            truth_at = lambda times: dense_truth(tru_rot, tru_tau, times, sys)
    validate_observations(obs, cfg)
    t_out = prediction_times(cfg, obs)
    output_root = os.environ.get("CISLUNAR_OUTPUT_ROOT")
    out = Path(cfg["results"]["dir"]).resolve()
    if output_root and not out.is_relative_to(Path(output_root).resolve()):
        raise ConfigError(f"output.directory must be inside the mounted output directory: {output_root}")
    if make_files:
        out.mkdir(parents=True, exist_ok=True)
    iod, estimator, prop = build_stages(cfg, sys)
    x_iod = iod.estimate(obs)                                 # 3    IOD -> rough orbit
    x0 = estimator.refine(obs, x_iod)                         # 4-6  orbit -> builder -> BLS
    est = prop.propagate(x0, t_out)                           # 7-8  propagator -> Orbit(t)

    truth = truth_at(t_out) if truth_at else None
    summary = {"method": x0.method, "epoch_days": x0.t / DAY, "info": x0.info,
               "covariance_available": x0.cov is not None, "evaluation_available": truth is not None,
               "run_name": cfg["run"]["name"]}
    if x0.cov is not None:
        summary["sigma_position_km"] = float(np.sqrt(np.trace(x0.cov[:3, :3])) / 1e3)
        summary["covariance_6x6_SI"] = x0.cov.tolist()
    if truth is not None:
        summary.update(summarize(x_iod, x0, truth_at(np.array([x0.t])), est, truth,
                                 prop.propagate(x0, obs.t), truth_at(obs.t)))
    summary.update(iod=iod.name, estimator=estimator.name, propagator=prop.name,
                   n_observations=len(obs), prediction_days=[float(t_out[0] / DAY), float(t_out[-1] / DAY)])

    r = cfg["results"]
    if make_files:
        os.makedirs(r["dir"], exist_ok=True)
        write_product_outputs(r["dir"], x0, est, obs, cfg)
        if truth is not None:
            write_outputs(r["dir"], summary, est, truth, write_summary=False, write_csv=r["csv"])
        if r["plot"]:
            summary["figure"] = (plot(r["dir"], obs, x0, est, truth, sys) if truth is not None
                                 else plot_estimate(r["dir"], est))
        if r["summary"]:
            with open(out / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        if r["save_config"]:
            with open(os.path.join(r["dir"], "config_used.yaml"), "w", encoding="utf-8") as f:
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
    p.add_argument("--show-config", action="store_true", help="print the full config (with defaults) and stop")
    a = p.parse_args(argv)

    try:
        cfg = load_config(a.config)
        if a.show_config:
            print(yaml.safe_dump(prepare(cfg), sort_keys=False))
            return 0
        summary = run(cfg)
    except ConfigError as e:
        print(e)
        return 2
    except (IODFailure, EstimationFailure) as e:
        print(f"Failed: {e}")
        return 1
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"Input/output error: {e}")
        return 2
    shown = {k: v for k, v in summary.items() if k != "covariance_6x6_SI"}   # full matrix is in summary.json
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    _sys.exit(main())

"""Public, versioned YAML schema translated to the existing numerical stages."""
from __future__ import annotations

import copy
import math


SCHEMA = {
    "schema_version": 1,
    "run": {"name": "simulation", "seed": 1},
    "system": {"model": "earth_moon_cr3bp", "mu": 0.0121505,
               "distance_unit_km": 384338.0, "time_unit_s": 375099.0},
    "input": {"source": "synthetic", "observations": None, "truth": None},
    "simulation": {
        "truth": {"dynamics": "cr3bp", "epoch_days": 0.0,
                  "initial_state": {"frame": "barycentric_rotating", "units": "nondimensional",
                                    "values": [0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0]},
                  "integrator": {"method": "DOP853", "rtol": 1e-12, "atol": 1e-12}},
        "observer": {"type": "geocenter", "position_km": [0.0, 0.0, 0.0]},
        "observations": {"type": "angular_radec", "times_days": [0, 2, 4, 6, 8],
                         "noise": {"distribution": "gaussian", "sigma_arcsec": 10.0}},
    },
    "determination": {
        "dynamics": "cr3bp",
        "iod": {"method": "collocation", "epoch_index": 2, "options": {}, "fallback": ["range_search"]},
        "estimator": {"method": "orekit_bls", "options": {}},
        "advanced": {"orbit_type": "cartesian", "station_offset_m": 1.0,
                     "integrator": "dormand_prince_853", "min_step_s": 0.001,
                     "max_step_s": 3600.0, "position_tolerance_m": 0.01, "position_scale_m": 1.0},
    },
    "prediction": {"backend": "orekit", "start_days": None, "end_days": None, "points": 800},
    "evaluation": {"enabled": True},
    "output": {"directory": "outputs/run", "plots": True, "observations": True},
    "runtime": {"orekit_data": None},
}


def strict_merge(defaults, supplied, path=""):
    from .config import ConfigError
    if not isinstance(supplied, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    if any(not isinstance(k, str) for k in supplied):
        raise ConfigError(f"{path or 'config'} keys must be strings")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ConfigError(f"Unknown configuration keys: {', '.join(path + k for k in sorted(unknown))}")
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(defaults[key], dict):
            # Method-specific options are checked against their component below.
            if key == "options":
                if not isinstance(value, dict):
                    raise ConfigError(f"{path}{key} must be a mapping")
                result[key] = copy.deepcopy(value)
            else:
                result[key] = strict_merge(defaults[key], value, path + key + ".")
        else:
            result[key] = value
    return result


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(config):
    """Validate public choices and produce a replayable legacy stage config."""
    from .config import ConfigError, DEFAULTS
    c = strict_merge(SCHEMA, config)
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if not isinstance(c["run"]["name"], str) or not c["run"]["name"].strip():
        raise ConfigError("run.name must be a nonempty string")
    if type(c["run"]["seed"]) is not int or c["run"]["seed"] < 0:
        raise ConfigError("run.seed must be a nonnegative integer")
    system, inp, sim = c["system"], c["input"], c["simulation"]
    if system["model"] != "earth_moon_cr3bp":
        raise ConfigError("system.model currently supports earth_moon_cr3bp")
    if inp["source"] not in ("csv", "synthetic"):
        raise ConfigError("input.source must be csv or synthetic")
    if inp["source"] == "csv" and not inp["observations"]:
        raise ConfigError("input.observations is required for CSV input")
    if inp["source"] == "synthetic" and (inp["observations"] is not None or inp["truth"] is not None):
        raise ConfigError("Synthetic input generates its own observations/truth; remove input file paths")
    for key in ("observations", "truth"):
        if inp[key] is not None and not isinstance(inp[key], str):
            raise ConfigError(f"input.{key} must be a path string or null")
    if not isinstance(c["output"]["directory"], str) or not c["output"]["directory"].strip():
        raise ConfigError("output.directory must be a nonempty path")
    for block, key in (("evaluation", "enabled"), ("output", "plots"), ("output", "observations")):
        if type(c[block][key]) is not bool:
            raise ConfigError(f"{block}.{key} must be a boolean")
    if sim["truth"]["dynamics"] != "cr3bp" or c["determination"]["dynamics"] != "cr3bp":
        raise ConfigError("Only cr3bp truth and determination dynamics are currently implemented")
    state = sim["truth"]["initial_state"]
    if state["frame"] != "barycentric_rotating" or state["units"] != "nondimensional":
        raise ConfigError("Truth initial state must be barycentric_rotating and nondimensional")
    if not isinstance(state["values"], list) or len(state["values"]) != 6 or not all(map(finite_number, state["values"])):
        raise ConfigError("simulation.truth.initial_state.values must contain six finite numbers")
    if not finite_number(sim["truth"]["epoch_days"]):
        raise ConfigError("simulation.truth.epoch_days must be finite")
    integrator = sim["truth"]["integrator"]
    if integrator["method"] not in ("DOP853", "RK45") or not all(
        finite_number(integrator[k]) and integrator[k] > 0 for k in ("rtol", "atol")
    ):
        raise ConfigError("Truth integrator needs DOP853/RK45 and positive finite rtol/atol")
    times = sim["observations"]["times_days"]
    if (not isinstance(times, list) or len(times) < 3 or not all(map(finite_number, times))
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ConfigError("simulation.observations.times_days needs at least three strictly increasing finite times")
    noise = sim["observations"]["noise"]
    if (sim["observations"]["type"] != "angular_radec" or noise["distribution"] != "gaussian"
            or not finite_number(noise["sigma_arcsec"]) or noise["sigma_arcsec"] < 0):
        raise ConfigError("Synthetic observations require angular_radec, gaussian noise, and nonnegative sigma_arcsec")
    observer = sim["observer"]
    if observer["type"] not in ("geocenter", "fixed_emi"):
        raise ConfigError("simulation.observer.type must be geocenter or fixed_emi")
    if (not isinstance(observer["position_km"], list) or len(observer["position_km"]) != 3
            or not all(map(finite_number, observer["position_km"]))):
        raise ConfigError("simulation.observer.position_km must contain three finite numbers")
    if observer["type"] == "geocenter" and any(observer["position_km"]):
        raise ConfigError("geocenter requires position_km: [0, 0, 0]")

    det = c["determination"]
    iod = det["iod"]
    if iod["method"] == "truth_plus_error":
        raise ConfigError("truth_plus_error is a legacy test initializer, not a product IOD method")
    block = "orekit" if iod["method"] in ("gooding", "gauss", "laplace") else iod["method"]
    if block == "collocation":
        try:
            from dataclasses import asdict
            from collocation_iod import SolverOptions
            iod_options = asdict(SolverOptions(**iod["options"]))
        except (ImportError, TypeError, ValueError) as exc:
            raise ConfigError(f"determination.iod.options: {exc}") from exc
    elif block in DEFAULTS["iod"] and isinstance(DEFAULTS["iod"][block], dict):
        iod_options = strict_merge(DEFAULTS["iod"][block], iod["options"], "determination.iod.options.")
    else:
        raise ConfigError(f"Unsupported IOD method: {iod['method']}")
    estimator = det["estimator"]
    est_defaults = {k: v for k, v in DEFAULTS["estimation"].items() if k != "method"}
    if estimator["method"] == "none" and estimator["options"]:
        raise ConfigError("estimator method none accepts no options")
    est_options = strict_merge(est_defaults, estimator["options"], "determination.estimator.options.")
    for key in ("max_iterations", "max_evaluations"):
        if type(est_options[key]) is not int or est_options[key] < 1:
            raise ConfigError(f"determination.estimator.options.{key} must be a positive integer")
    if (not finite_number(est_options["convergence_threshold"]) or est_options["convergence_threshold"] <= 0
            or type(est_options["covariance"]) is not bool):
        raise ConfigError("Estimator needs a positive convergence_threshold and boolean covariance")
    backend = c["prediction"]["backend"]
    if backend not in ("orekit", "scipy"):
        raise ConfigError("prediction.backend must be orekit or scipy")
    advanced = det["advanced"]
    for key in ("min_step_s", "max_step_s", "position_tolerance_m", "position_scale_m", "station_offset_m"):
        if not finite_number(advanced[key]) or advanced[key] <= 0:
            raise ConfigError(f"determination.advanced.{key} must be positive and finite")
    if advanced["min_step_s"] > advanced["max_step_s"]:
        raise ConfigError("Integrator min_step_s must not exceed max_step_s")
    out = {
        "system": {"mu": system["mu"], "du_km": system["distance_unit_km"], "tu_s": system["time_unit_s"]},
        "data": {"source": inp["source"], "observations": inp["observations"], "truth": inp["truth"]},
        "simulation": {**copy.deepcopy(sim), "seed": c["run"]["seed"]},
        "evaluation": c["evaluation"],
        "run": c["run"],
        "iod": {"method": iod["method"], "epoch_index": iod["epoch_index"], block: iod_options,
                "fallback": [f for f in iod["fallback"] if f != iod["method"]] if isinstance(iod["fallback"], list)
                else iod["fallback"]},
        "estimation": {"method": estimator["method"], **est_options},
        "orbit": {"type": advanced["orbit_type"]},
        "measurements": {"station_offset_m": advanced["station_offset_m"]},
        "propagator_builder": {"force_model": det["dynamics"], **{k: v for k, v in advanced.items()
                                if k not in ("orbit_type", "station_offset_m")}},
        "propagator": {"method": "orekit_cr3bp" if backend == "orekit" else "cr3bp"},
        "prediction": {k: v for k, v in c["prediction"].items() if k != "backend"},
        "results": {"dir": c["output"]["directory"], "plot": c["output"]["plots"],
                    "observations": c["output"]["observations"]},
        "orekit": {"data_path": c["runtime"]["orekit_data"]},
    }
    return out

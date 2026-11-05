"""
Load, fill in and check config.yaml.

The config follows the simulation chain in order:

    data -> measurements -> iod -> orbit -> propagator_builder -> estimation -> propagator -> prediction -> results

Anything left out of config.yaml gets the default below. Mistakes (a misspelled
section, an unknown method name, a bad value) are reported all at once, in plain
words, before anything runs.
"""

from __future__ import annotations

import copy

import yaml

DEFAULTS: dict = {
    "system": {"mu": 0.0121505, "du_km": 384338.0, "tu_s": 375099.0},
    "data": {
        "source": "csv",
        "observations": "data/synthetic_obs.csv",
        "truth": "data/synthetic_truth.csv",
        "noise_arcsec": 10.0,
        "noise_seed": 0,
        "observer": "from_file",
        "use_obs": "all",
    },
    "measurements": {"type": "angular_radec", "sigma_arcsec": "from_file", "station_offset_m": 1.0},
    "iod": {
        "method": "range_search",
        "epoch_index": 2,
        "range_search": {"range_km": [150000, 650000, 10000], "range_rate_m_s": [-800, 800, 50]},
        "orekit": {"obs_indices": [0, 2, 4], "range_guess_km": [350000, 350000]},
        "truth_plus_error": {"position_error_km": 1000.0, "velocity_error_m_s": 10.0, "seed": 0},
    },
    "orbit": {"type": "cartesian"},
    "propagator_builder": {
        "force_model": "cr3bp",
        "integrator": "dormand_prince_853",
        "min_step_s": 0.001,
        "max_step_s": 3600.0,
        "position_tolerance_m": 0.01,
        "position_scale_m": 1.0,
    },
    "estimation": {
        "method": "orekit_bls",
        "optimizer": "levenberg_marquardt",
        "max_iterations": 50,
        "max_evaluations": 100,
        "convergence_threshold": 0.001,
        "covariance": True,
    },
    "propagator": {"method": "orekit_cr3bp"},
    "prediction": {"start_days": None, "end_days": None, "points": 800},
    "results": {"dir": "results", "plot": True, "csv": True, "summary": True, "save_config": True},
    "orekit": {"data_path": None},
}

CHOICES = {
    ("data", "source"): ["csv", "resample_truth"],
    ("data", "observer"): ["from_file", "geocenter"],
    ("measurements", "type"): ["angular_radec"],
    ("orbit", "type"): ["cartesian", "equinoctial", "keplerian"],
    ("propagator_builder", "force_model"): ["cr3bp"],
    ("propagator_builder", "integrator"): ["dormand_prince_853", "dormand_prince_54"],
    ("estimation", "optimizer"): ["levenberg_marquardt", "gauss_newton"],
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def set_value(cfg: dict, assignment: str) -> None:
    """Apply a command-line override like 'data.noise_arcsec=100' or 'iod.method=gooding'."""
    if "=" not in assignment:
        raise ConfigError(f"--set needs section.key=value, got '{assignment}'")
    path, raw = assignment.split("=", 1)
    keys = path.strip().split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def prepare(user_cfg: dict) -> dict:
    """Fill in defaults, then check everything. Returns the full config used for the run."""
    from .estimation import ESTIMATORS
    from .iod import IOD_METHODS
    from .propagators import PROPAGATORS

    problems = []
    for k in user_cfg or {}:
        if k not in DEFAULTS:
            problems.append(f"unknown section '{k}' (valid: {', '.join(DEFAULTS)})")
    cfg = _merge(DEFAULTS, {k: v for k, v in (user_cfg or {}).items() if k in DEFAULTS})

    method_lists = {("iod", "method"): IOD_METHODS, ("estimation", "method"): ESTIMATORS,
                    ("propagator", "method"): PROPAGATORS}
    for (sec, key), valid in {**CHOICES, **{k: list(v) for k, v in method_lists.items()}}.items():
        if cfg[sec][key] not in valid:
            problems.append(f"{sec}.{key} = '{cfg[sec][key]}' is not an option (choose from: {', '.join(valid)})")

    d, m, p = cfg["data"], cfg["measurements"], cfg["prediction"]
    if d["source"] == "resample_truth" and not d.get("truth"):
        problems.append("data.source = resample_truth needs data.truth")
    if not (d["use_obs"] == "all" or (isinstance(d["use_obs"], list) and all(isinstance(i, int) for i in d["use_obs"]))):
        problems.append("data.use_obs must be 'all' or a list of observation numbers, e.g. [0, 1, 2, 3, 4]")
    s = m["sigma_arcsec"]
    if not (s == "from_file" or (isinstance(s, (int, float)) and s > 0)):
        problems.append("measurements.sigma_arcsec must be 'from_file' or a positive number")
    if not (isinstance(m["station_offset_m"], (int, float)) and m["station_offset_m"] > 0):
        problems.append("measurements.station_offset_m must be > 0 (exactly 0 makes Orekit return NaN)")
    if not (isinstance(p["points"], int) and p["points"] >= 2):
        problems.append("prediction.points must be a whole number >= 2")
    if p["start_days"] is not None and p["end_days"] is not None and p["end_days"] <= p["start_days"]:
        problems.append("prediction.end_days must be after prediction.start_days")
    for key in ("range_km", "range_rate_m_s"):
        r = cfg["iod"]["range_search"].get(key)
        if not (isinstance(r, list) and len(r) == 3 and r[2] > 0 and r[1] > r[0]):
            problems.append(f"iod.range_search.{key} must be [min, max, step] with max > min and step > 0")
    if cfg["estimation"]["method"] == "orekit_bls" and cfg["orbit"]["type"] == "keplerian":
        pass  # allowed, but see the comment in config.yaml

    if problems:
        raise ConfigError("config.yaml has problems:\n  - " + "\n  - ".join(problems))
    return cfg

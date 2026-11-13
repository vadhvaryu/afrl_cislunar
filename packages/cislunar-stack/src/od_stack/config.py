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
from pathlib import Path

import yaml
from .scenario import SCHEMA, finite_number

DEFAULTS: dict = {
    "system": {"mu": 0.0121505, "du_km": 384338.0, "tu_s": 375099.0},
    "data": {
        "source": "csv",
        "observations": "data/synthetic_obs.csv",
        "truth": None,
        "noise_arcsec": 10.0,
        "noise_seed": 0,
        "observer": "from_file",
        "use_obs": "all",
    },
    "measurements": {"type": "angular_radec", "sigma_arcsec": "from_file", "station_offset_m": 1.0},
    "iod": {
        "method": "range_search",
        "epoch_index": 2,
        # The grid only has to put a start in the right valley; polishing does the rest (iod.RangeSearch).
        "range_search": {"range_km": [150000, 650000, 20000], "range_rate_m_s": [-800, 800, 100],
                         "refine": True, "candidates": 12, "max_candidates": 20},
        "orekit": {"obs_indices": [0, 2, 4], "range_guess_km": [350000, 350000]},
        "truth_plus_error": {"position_error_km": 1000.0, "velocity_error_m_s": 10.0, "seed": 0},
        "fallback": [],             # IOD methods to try, in order, if the first fails or its fit fails the check
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
        "residual_gate_sigma": None,  # reject a fit whose residual RMS exceeds this many sigmas; null = off.
                                      # Off here (legacy configs, tests); the YAML product turns it on at 3.
    },
    "propagator": {"method": "orekit_cr3bp"},
    "prediction": {"start_days": None, "end_days": None, "points": 800},
    "results": {"dir": "results", "plot": True, "csv": True, "summary": True, "save_config": True,
                "observations": True},
    "orekit": {"data_path": None},
    "simulation": {**copy.deepcopy(SCHEMA["simulation"]), "seed": 1},
    "evaluation": {"enabled": True},
    "run": {"name": "simulation", "seed": 1},
}

CHOICES = {
    ("data", "source"): ["csv", "resample_truth", "synthetic"],
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
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError("config.yaml must contain a mapping")
    if "schema_version" in cfg:
        from .scenario import normalize
        cfg = normalize(cfg)
    # Scenario paths belong to the config, not to the shell's working directory.
    base = Path(path).resolve().parent
    for section, key in (("data", "observations"), ("data", "truth"),
                         ("results", "dir"), ("orekit", "data_path")):
        block = cfg.setdefault(section, {})
        if not isinstance(block, dict):
            raise ConfigError(f"{section} must be a mapping")
        value = block.get(key, DEFAULTS[section][key])
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{section}.{key} must be a nonempty path or null")
            candidate = Path(value).expanduser()
            block[key] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())
    return cfg


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

    if not isinstance(user_cfg, dict):
        raise ConfigError("Configuration must be a mapping")
    if "schema_version" in user_cfg:
        from .scenario import normalize
        user_cfg = normalize(user_cfg)
    # Discover the optional installed component without choosing it for the user.
    try:
        from collocation_iod.adapters import register_with_od_stack
    except ModuleNotFoundError as exc:
        if exc.name != "collocation_iod":
            raise
    else:
        register_with_od_stack()

    problems = []
    for k in user_cfg or {}:
        if k not in DEFAULTS:
            problems.append(f"unknown section '{k}' (valid: {', '.join(DEFAULTS)})")
        elif not isinstance(user_cfg[k], dict):
            problems.append(f"{k} must be a mapping")
    if any(k in DEFAULTS and not isinstance(v, dict) for k, v in user_cfg.items()):
        raise ConfigError("config.yaml has problems:\n  - " + "\n  - ".join(problems))
    cfg = _merge(DEFAULTS, {k: v for k, v in (user_cfg or {}).items() if k in DEFAULTS})

    method_lists = {("iod", "method"): IOD_METHODS, ("estimation", "method"): ESTIMATORS,
                    ("propagator", "method"): PROPAGATORS}
    for (sec, key), valid in {**CHOICES, **{k: list(v) for k, v in method_lists.items()}}.items():
        if cfg[sec][key] not in valid:
            problems.append(f"{sec}.{key} = '{cfg[sec][key]}' is not an option (choose from: {', '.join(valid)})")

    d, m, p = cfg["data"], cfg["measurements"], cfg["prediction"]
    system = cfg["system"]
    if not all(finite_number(system[k]) for k in ("mu", "du_km", "tu_s")) or not (
        0 < system["mu"] < 0.5 and system["du_km"] > 0 and system["tu_s"] > 0
    ):
        problems.append("system requires 0 < mu < 0.5 and positive finite distance/time units")
    if type(cfg["iod"]["epoch_index"]) is not int or cfg["iod"]["epoch_index"] < 0:
        problems.append("iod.epoch_index must be a nonnegative integer")
    if d["source"] == "synthetic":
        from .scenario import normalize
        # Also validate generated-input settings in saved/legacy resolved configs.
        sim = copy.deepcopy(cfg["simulation"])
        seed = sim.pop("seed")
        normalize({"schema_version": 1, "simulation": sim, "run": {"seed": seed},
                   "determination": {"iod": {"method": "range_search"}}})
        if cfg["iod"]["method"] == "truth_plus_error":
            problems.append("truth_plus_error is not supported with generated input")
    if d["source"] == "resample_truth" and not d.get("truth"):
        problems.append("data.source = resample_truth needs data.truth")
    if not (d["use_obs"] == "all" or (isinstance(d["use_obs"], list) and len(d["use_obs"]) >= 3
            and all(type(i) is int for i in d["use_obs"]) and all(a < b for a, b in zip(d["use_obs"], d["use_obs"][1:])))):
        problems.append("data.use_obs must be 'all' or a list of observation numbers, e.g. [0, 1, 2, 3, 4]")
    s = m["sigma_arcsec"]
    if not (s == "from_file" or (isinstance(s, (int, float)) and s > 0)):
        problems.append("measurements.sigma_arcsec must be 'from_file' or a positive number")
    if not (isinstance(m["station_offset_m"], (int, float)) and m["station_offset_m"] > 0):
        problems.append("measurements.station_offset_m must be > 0 (exactly 0 makes Orekit return NaN)")
    if not (isinstance(p["points"], int) and p["points"] >= 2):
        problems.append("prediction.points must be a whole number >= 2")
    for key in ("start_days", "end_days"):
        if p[key] is not None and not finite_number(p[key]):
            problems.append(f"prediction.{key} must be finite or null")
    if all(finite_number(p[k]) for k in ("start_days", "end_days")) and p["end_days"] <= p["start_days"]:
        problems.append("prediction.end_days must be after prediction.start_days")
    for key in ("range_km", "range_rate_m_s"):
        r = cfg["iod"]["range_search"].get(key)
        if not (isinstance(r, list) and len(r) == 3 and r[2] > 0 and r[1] > r[0]):
            problems.append(f"iod.range_search.{key} must be [min, max, step] with max > min and step > 0")
    rs = cfg["iod"]["range_search"]
    if type(rs.get("refine", True)) is not bool:
        problems.append("iod.range_search.refine must be true or false")
    for key in ("candidates", "max_candidates"):
        if type(rs.get(key, 4)) is not int or rs.get(key, 4) < 1:
            problems.append(f"iod.range_search.{key} must be a positive integer")
    if cfg["estimation"]["method"] == "orekit_bls" and cfg["orbit"]["type"] == "keplerian":
        pass  # allowed, but see the comment in config.yaml
    gate = cfg["estimation"].get("residual_gate_sigma")
    if gate is not None and not (finite_number(gate) and gate > 0):
        problems.append("estimation.residual_gate_sigma must be a positive number or null (off)")
    fallback = cfg["iod"].get("fallback")
    if not (isinstance(fallback, list) and all(isinstance(f, str) for f in fallback)):
        problems.append("iod.fallback must be a list of IOD method names")
    else:
        for f in fallback:
            if f not in IOD_METHODS or f == "truth_plus_error":
                problems.append(f"iod.fallback: '{f}' is not a product IOD method "
                                f"(choose from: {', '.join(k for k in IOD_METHODS if k != 'truth_plus_error')})")
        if len(set(fallback)) != len(fallback):
            problems.append("iod.fallback lists a method twice")
        if cfg["iod"]["method"] in fallback:
            problems.append("iod.fallback repeats iod.method; list only the other methods to try")
    uses_collocation = cfg["iod"]["method"] == "collocation" or "collocation" in (fallback or [])
    mode = (cfg["iod"].get("collocation") or {}).get("mode", "continuous")
    if uses_collocation and mode in ("impulsive", "hybrid") and cfg["estimation"]["method"] == "orekit_bls":
        problems.append(f"collocation mode '{mode}' estimates burns, but orekit_bls assumes the spacecraft coasts and "
                        "would discard them; use estimation method none to keep collocation's trajectory, "
                        "or mode continuous")

    if problems:
        raise ConfigError("config.yaml has problems:\n  - " + "\n  - ".join(problems))
    return cfg

"""
Reading config.yaml and turning its blocks into stage objects.

Each stage block (iod, estimation, propagator) looks like this:

    iod:
      method: cr3bp_dc          # which method to run
      obs_indices: [0, 2, 4]    # shared settings: passed to whichever method is chosen
      cr3bp_dc:                 # settings only for cr3bp_dc
        alpha0_step_km: 10000
      range_grid:               # settings only for range_grid (ignored unless it is chosen)
        rho_step_km: 7500

The chosen method gets its own block plus the shared settings. Blocks for the other
methods are ignored, so two methods can use the same setting name with different
values without clashing. If one name is set in both places, the shared one wins:
that is how scripts and tests override a single value (cfg["iod"]["seed"] = 3)
without needing to know which method block it lives in.
"""

from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def method_options(block: dict | None, registry: dict, default_method: str | None = None) -> tuple[str, dict]:
    """Split a stage block into (method name, keyword arguments for that method)."""
    block = dict(block or {})
    method = block.pop("method", default_method)
    if method not in registry:
        raise ValueError(f"unknown method {method!r}; choose one of: {', '.join(sorted(registry))}")
    shared = {k: v for k, v in block.items() if k not in registry}   # drop every per-method block
    own = block.get(method) or {}
    return method, {**own, **shared}


def builder_options(cfg: dict) -> dict:
    """Integrator settings for Orekit's NumericalPropagatorBuilder (config.yaml: propagator_builder).
    Used by BOTH the batch least squares fit and the orekit_cr3bp propagator, so they share one model."""
    return dict(cfg.get("propagator_builder") or {})


def build_iod(cfg: dict, sys, method: str | None = None, **overrides):
    from .iod import IOD_METHODS

    block = dict(cfg["iod"])
    if method:
        block["method"] = method
    name, kw = method_options(block, IOD_METHODS)
    kw.update(overrides)
    kw.setdefault("orekit_data", (cfg.get("orekit") or {}).get("data_path"))
    kw.setdefault("truth_path", (cfg.get("data") or {}).get("truth"))
    return IOD_METHODS[name](sys, **kw)


def build_estimator(cfg: dict, sys):
    from .estimation import ESTIMATORS

    name, kw = method_options(cfg.get("estimation"), ESTIMATORS, default_method="none")
    return ESTIMATORS[name](sys, orekit_data=(cfg.get("orekit") or {}).get("data_path"),
                           builder_options=builder_options(cfg), **kw)


def build_propagator(cfg: dict, sys):
    from .propagators import PROPAGATORS

    name, kw = method_options(cfg["propagator"], PROPAGATORS)
    return PROPAGATORS[name](sys, orekit_data=(cfg.get("orekit") or {}).get("data_path"),
                            builder_options=builder_options(cfg), **kw)

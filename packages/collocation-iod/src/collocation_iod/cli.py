"""File/CLI boundary for synthetic-observation CSV input and JSON output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from .adapters import observations_from_emi, rotating_state_to_emi


def read_observations(path, *, system):
    """Read the existing stack's CSV format; the solver never reads files."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"time_days", "ra_deg", "dec_deg", "sigma_arcsec"}
        columns = set(reader.fieldnames or [])
        if not required <= columns:
            raise ValueError(f"Observation CSV is missing columns: {sorted(required - columns)}")
        station = {"obs_x_km", "obs_y_km", "obs_z_km"}
        if station & columns and not station <= columns:
            raise ValueError("Supply all three observer position columns or omit all three")
        rows = list(reader)
    if not rows:
        raise ValueError("Observation CSV is empty")
    t_s = np.array([float(row["time_days"]) * 86400.0 for row in rows])
    ra = np.radians([float(row["ra_deg"]) for row in rows])
    dec = np.radians([float(row["dec_deg"]) for row in rows])
    if np.any(np.abs(dec) > np.pi / 2):
        raise ValueError("Declination must be between -90 and 90 degrees")
    sigma = np.radians([float(row["sigma_arcsec"]) / 3600.0 for row in rows])
    los = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    observer = np.zeros((len(rows), 3))
    if station <= columns:
        observer = np.array([[float(row[c]) * 1000.0 for c in ("obs_x_km", "obs_y_km", "obs_z_km")]
                             for row in rows])
    return observations_from_emi(t_s, los, observer, sigma, system=system), t_s


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv=None):
    from . import CR3BPSystem, SolverOptions, solve

    parser = argparse.ArgumentParser(description="Cislunar IOD with maneuver reconstruction by collocation")
    parser.add_argument("observations", type=Path, help="CSV: time_days, ra_deg, dec_deg, sigma_arcsec")
    parser.add_argument("--output", default="-", help="JSON path, or - for stdout (default)")
    parser.add_argument("--mode", choices=("continuous", "impulsive", "hybrid"), default="continuous")
    parser.add_argument("--nodes", type=int, default=80)
    parser.add_argument("--measurement-mode", choices=("bounded", "exact"), default="bounded")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--initial-guess", choices=("auto", "los", "L1", "L2", "L4"), default="auto")
    parser.add_argument("--mesh-tolerance", type=float, default=1e-3,
                        help="Maximum sampled, scaled interval dynamics residual")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--epoch-index", type=int, default=None, help="Output state at this observation (default: middle)")
    parser.add_argument("--mu", type=float, default=0.0121505)
    parser.add_argument("--du-m", type=float, default=384338000.0)
    parser.add_argument("--tu-s", type=float, default=375099.0)
    args = parser.parse_args(argv)
    try:
        system = CR3BPSystem(mu=args.mu, du_m=args.du_m, tu_s=args.tu_s)
        arc, t_s = read_observations(args.observations, system=system)
        k = len(t_s) // 2 if args.epoch_index is None else args.epoch_index
        if not 0 <= k < len(t_s):
            raise ValueError("epoch-index is outside the observation arc")
        options = SolverOptions(mode=args.mode, nodes=args.nodes, measurement_mode=args.measurement_mode,
                                confidence=args.confidence, initial_guess=args.initial_guess,
                                max_iterations=args.max_iterations, mesh_tolerance=args.mesh_tolerance)
        result = solve(arc, system=system, options=options)
        state = result.state_at(float(arc.t[k]))
        r, v = rotating_state_to_emi(state, float(t_s[k]), system=system)
        payload = {
            "method": "collocation", "mode": args.mode,
            "state": {"frame": "EMI", "t_s": float(t_s[k]), "r_m": r, "v_m_s": v},
            "system": {"mu": system.mu, "du_m": system.du_m, "tu_s": system.tu_s},
            "trajectory": {"frame": "rotating_barycentric", "units": "nondimensional",
                           "t": result.t, "states": result.states, "controls": result.controls,
                           "increments": result.increments, "impulses": result.impulses},
            "diagnostics": result.diagnostics,
        }
        encoded = json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n"
        if args.output == "-":
            print(encoded, end="")
        else:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded, encoding="utf-8")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"collocation-iod: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

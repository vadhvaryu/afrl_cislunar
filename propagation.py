"""
propagation.py -- the PROPAGATION step of the IOD -> trajectory stack.

What it does
------------
Takes ONE starting state (from the IOD, or the true state for testing),
flies it forward with Orekit's Earth-Moon CR3BP model, and writes the
predicted trajectory at the times you ask for.

    starting state  ->  Orekit CR3BP propagator  ->  predicted trajectory
    (x,y,z,vx,vy,vz)                                  (one row per time)

Conventions (agree these with the team)
---------------------------------------
* Frame : Earth-Moon ROTATING frame, origin at the barycenter
          (the same frame as the Orekit tutorial outputs)
* Units : either "si" (m, m/s, seconds) or "nd" (CR3BP nondimensional units)
* Time  : seconds (si) or nondimensional time (nd) measured from the
          starting state's epoch, so t = 0 is the starting state

Run it from the command line
----------------------------
    python propagation.py --state iod_state.csv --times synthetic_truth.csv \
                          --out predicted.csv --units si --orekit-data ./orekit-data

iod_state.csv needs one row with columns:  t, x, y, z, vx, vy, vz
synthetic_truth.csv only needs a "t" column (the times to output at).
"""

import argparse

import numpy as np
import pandas as pd
import orekit_jpype as orekit

STATE_COLUMNS = ["x", "y", "z", "vx", "vy", "vz"]


# ---------------------------------------------------------------------------
# 1. Start Orekit
# ---------------------------------------------------------------------------
def setup_orekit(orekit_data_path=None):
    """Start the Java engine behind Orekit and (optionally) load orekit-data.

    orekit-data is the folder of reference files (planet ephemerides, leap
    seconds). The real Earth-Moon CR3BP needs it to look up the Earth's and
    Moon's masses.
    """
    orekit.initVM()
    if orekit_data_path is not None:
        from orekit_jpype.pyhelpers import setup_orekit_data
        setup_orekit_data(orekit_data_path)


# ---------------------------------------------------------------------------
# 2. Build the Earth-Moon CR3BP system
# ---------------------------------------------------------------------------
def make_earth_moon_system():
    """Return Orekit's standard Earth-Moon CR3BP system (needs orekit-data).

    This is the exact system the Orekit tutorials use, so trajectories from
    this module match Container A's halo / manifold / L4 outputs.
    """
    from org.orekit.bodies import CR3BPFactory
    return CR3BPFactory.getEarthMoonCR3BP()


def describe_system(system):
    """Return the system's scaling constants as a plain dictionary.

    length_m  : 1 nondimensional distance in meters (Earth-Moon distance)
    time_s    : 1 nondimensional time unit in seconds
    speed_mps : 1 nondimensional speed in m/s
    mu        : mass ratio Moon / (Earth + Moon)
    """
    length = system.getDdim()
    speed = system.getVdim()
    return {
        "length_m": length,
        "time_s": length / speed,  # Orekit's getTdim() is one full period (2*pi of these)
        "speed_mps": speed,
        "mu": system.getMassRatio(),
    }


# ---------------------------------------------------------------------------
# 3. Unit conversions (SI <-> nondimensional)
# ---------------------------------------------------------------------------
def to_nondim(state, times, scales):
    """Convert a 6-element state and times from SI to nondimensional units."""
    state = np.asarray(state, dtype=float)
    pos = state[:3] / scales["length_m"]
    vel = state[3:] / scales["speed_mps"]
    return np.concatenate([pos, vel]), np.asarray(times, dtype=float) / scales["time_s"]


def to_si(states, scales):
    """Convert an (N, 6) array of nondimensional states to SI units."""
    states = np.asarray(states, dtype=float).copy()
    states[:, :3] *= scales["length_m"]
    states[:, 3:] *= scales["speed_mps"]
    return states


# ---------------------------------------------------------------------------
# 4. The propagator itself
# ---------------------------------------------------------------------------
def propagate_cr3bp(state0, times, system, units="si", tolerance=1e-12):
    """Propagate one starting state with Orekit's CR3BP model.

    Parameters
    ----------
    state0 : 6 numbers [x, y, z, vx, vy, vz] at t = 0, rotating frame
    times  : times to output at, measured from the starting state
             (seconds if units="si", nondimensional if units="nd")
    system : an Orekit CR3BPSystem (from make_earth_moon_system())
    units  : "si" or "nd" -- used for BOTH the inputs and the outputs

    Returns
    -------
    (N, 6) numpy array of states, one row per requested time, same units.
    """
    from jpype import JObject
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.propagation.numerical.cr3bp import CR3BPForceModel
    from org.orekit.time import AbsoluteDate
    from org.orekit.utils import AbsolutePVCoordinates, PVCoordinates

    # Orekit's CR3BP force model works in NONDIMENSIONAL units internally,
    # so convert SI inputs first.
    scales = describe_system(system)
    if units == "si":
        s0, tau = to_nondim(state0, times, scales)
    elif units == "nd":
        s0, tau = np.asarray(state0, dtype=float), np.asarray(times, dtype=float)
    else:
        raise ValueError('units must be "si" or "nd"')

    # Orekit needs a date for the starting state. In the CR3BP the physics
    # doesn't depend on the date, so any fixed reference works. Orekit's
    # CR3BP convention: one second after this date = one nondimensional
    # time unit.
    epoch = AbsoluteDate.J2000_EPOCH

    frame = system.getRotatingFrame()
    pv = PVCoordinates(Vector3D(*map(float, s0[:3])), Vector3D(*map(float, s0[3:])))
    initial_state = SpacecraftState(AbsolutePVCoordinates(frame, epoch, pv))

    # Step-by-step solver (same family as solve_ivp's "DOP853").
    integrator = DormandPrince853Integrator(1e-10, 1.0, tolerance, tolerance)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(None)               # plain x,y,z,vx,vy,vz, not orbital elements
    propagator.setIgnoreCentralAttraction(True)  # CR3BP gravity comes from the force model below
    propagator.addForceModel(CR3BPForceModel(system))
    propagator.setInitialState(initial_state)

    # Fly forward, stopping at each requested time in order.
    out = np.empty((len(tau), 6))
    for k, t in enumerate(tau):
        state = propagator.propagate(epoch.shiftedBy(float(t)))
        # JObject(..., PVCoordinates) tells Python which Java type to treat it as
        pv_k = JObject(state.getAbsPVA().getPVCoordinates(), PVCoordinates)
        p, v = pv_k.getPosition(), pv_k.getVelocity()
        out[k] = [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()]

    return to_si(out, scales) if units == "si" else out


# ---------------------------------------------------------------------------
# 5. Reading and writing files (the hand-offs to the other modules)
# ---------------------------------------------------------------------------
def read_state_csv(path):
    """Read the starting state (first row) from the IOD's output file."""
    row = pd.read_csv(path, comment="#").iloc[0]
    missing = [c for c in ["t"] + STATE_COLUMNS if c not in row.index]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return float(row["t"]), row[STATE_COLUMNS].to_numpy(dtype=float)


def read_times_csv(path, column="t"):
    """Read the output times from a file (e.g. synthetic_truth.csv)."""
    df = pd.read_csv(path, comment="#")
    if column not in df.columns:
        raise ValueError(f"{path} has no '{column}' column; columns are {list(df.columns)}")
    return df[column].to_numpy(dtype=float)


def write_trajectory_csv(path, times, states, units):
    """Write the predicted trajectory in the same layout as the truth file."""
    unit_note = "m, m/s, s" if units == "si" else "nondimensional CR3BP units"
    with open(path, "w") as f:
        f.write("# frame: Earth-Moon rotating (barycentric)\n")
        f.write(f"# units: {unit_note}\n")
        f.write("# source: propagation.py (Orekit CR3BP)\n")
        pd.DataFrame(np.column_stack([times, states]), columns=["t"] + STATE_COLUMNS) \
          .to_csv(f, index=False, float_format="%.12g")


# ---------------------------------------------------------------------------
# 6. Command-line entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Propagate a state with Orekit's CR3BP.")
    parser.add_argument("--state", required=True, help="CSV with the starting state (t,x,y,z,vx,vy,vz)")
    parser.add_argument("--times", required=True, help="CSV whose 't' column lists the output times")
    parser.add_argument("--out", required=True, help="where to write the predicted trajectory")
    parser.add_argument("--units", default="si", choices=["si", "nd"])
    parser.add_argument("--orekit-data", default="orekit-data", help="path to the orekit-data folder")
    args = parser.parse_args()

    setup_orekit(args.orekit_data)
    system = make_earth_moon_system()

    t0, state0 = read_state_csv(args.state)
    times = read_times_csv(args.times)
    # The propagator measures time from the starting state, so shift.
    states = propagate_cr3bp(state0, times - t0, system, units=args.units)
    write_trajectory_csv(args.out, times, states, args.units)
    print(f"Wrote {len(times)} states to {args.out}")


if __name__ == "__main__":
    main()

# Collocation initial orbit determination

A standalone Python implementation of Heidrich and Holzinger's
[2025 collocation and nonlinear-programming method](https://doi.org/10.1007/s40295-025-00513-7)
for cislunar initial orbit determination with unknown maneuvers. It estimates a
trajectory and continuous, impulsive, or hybrid control history from optical
observations using CR3BP dynamics, Hermite–Simpson transcription, and IPOPT.

The numerical solver reads no configuration files, loads no truth data, and starts
no Java VM. Inputs, physical constants, and settings are explicit Python objects.
A small adapter accepts the existing `od_stack.Observations` and returns the actual
`od_stack.StateEstimate`, so the current simulation runner can select this method
without changing its estimator or propagator implementations.

This is an independent research implementation. See
[the equation mapping and limitations](docs/paper_mapping.md) before interpreting
reconstructed maneuvers or comparing performance with the paper.

## Install and run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./packages/collocation-iod[test]"
.\.venv\Scripts\collocation-iod.exe packages/collocation-iod/examples/synthetic_obs.csv --output estimate.json
```

On Linux/macOS, use `.venv/bin/python` and `.venv/bin/collocation-iod`.
Alternatively, after activating the environment:

```text
python -m pip install -e "./packages/collocation-iod[test]"
collocation-iod packages/collocation-iod/examples/synthetic_obs.csv --mode continuous --nodes 80 --initial-guess auto --measurement-mode bounded --epoch-index 2 --output estimate.json
```

The bundled CSV is a copy of the repository's five synthetic angle observations.
The answer key is not bundled or used by the estimator. The CLI accepts the
existing columns `time_days, ra_deg, dec_deg, sigma_arcsec` and an optional `obs_id`.
It uses the simulation's Earth-centered inertial (EMI) convention; see `--help` for
additional observer columns and numerical options.

The default fits the angles within a chi-square uncertainty bound while minimizing
continuous control energy. A small control objective is not a position-error
metric. A sparse angles-only arc can admit multiple solutions.

## Python interfaces

### Existing simulation stack

Install this package in the environment used by the stack, then construct the
adapter with the stack's Earth–Moon system:

```python
from collocation_iod.adapters import CollocationIOD
from od_stack.frames import EarthMoonSystem

iod = CollocationIOD(EarthMoonSystem(), epoch_index=2, nodes=80)
estimate = iod.estimate(obs)  # obs is od_stack.contracts.Observations

# estimate.t: seconds since the stack's reference epoch
# estimate.r: EMI position in meters
# estimate.v: EMI velocity in meters/second
# estimate.cov: None; the paper does not derive a state covariance
# estimate.info["collocation"]: JSON-serializable solver diagnostics
trajectory_and_controls = iod.last_result
```

The adapter does not use the `truth_path` or `orekit_data` arguments passed by the
existing runner. Frame rotation uses the original absolute time convention, so
observation subsets retain the correct Earth–Moon phase.

Install `./packages/cislunar-stack[orekit]` as well, then from the repository root use the wrapper:

```text
cislunar-stack --config scenarios/plug_and_play/config.yaml
```

For an integration run without Java, set `determination.estimator.method: none`
and `prediction.backend: scipy` in that YAML, then run:

```text
cislunar-stack --config scenarios/plug_and_play/config.yaml
```

The runner itself also needs its existing YAML and plotting dependencies. For
Orekit refinement, install `./packages/collocation-iod[stack]`, provide a supported Java
runtime, and set up the stack's `OREKIT_DATA` as before. The registration wrapper
adds the new IOD class to the existing registry at runtime; it does not rewrite
the repository's existing files or default scenario.

**Maneuver history is not automatically passed to Orekit.** The adapter preserves
the state contract, while `last_result` retains the complete fitted history. The
existing batch least-squares and propagation stages use uncontrolled CR3BP
dynamics. They need a maneuver-aware extension to reproduce a fitted trajectory
with nonzero controls.

### Standalone numerical solver

```python
from collocation_iod import (
    CR3BPSystem, ObservationArc, SolverOptions, solve,
)

arc = ObservationArc(
    t=tau,                  # (P,) nondimensional absolute times, strictly increasing
    los=los_rotating,        # (P, 3) unit directions in the rotating frame
    observer=observer_rot,  # (P, 3) barycentric rotating positions in DU
    sigma=angular_sigma,    # scalar or (P,) tangent angular standard deviation
)
result = solve(
    arc,
    system=CR3BPSystem(),
    options=SolverOptions(mode="continuous", nodes=80, initial_guess="auto"),
)
x = result.state_at(tau[2])  # [x,y,z,vx,vy,vz], rotating and nondimensional
print(result.diagnostics)
```

The default mesh has 80 nodes. `initial_guess="auto"` solves from both a
stationary L2 guess and a LOS-based guess, starting controls at zero, then selects
the lowest-objective result that passes the optimizer, feasibility, and mesh
checks. It tries L4 only if both initial attempts fail. Attempt diagnostics are
retained. Selecting `"los"`, `"L1"`, `"L2"`, or `"L4"`, or supplying explicit
initial states, requests a single solve. This strategy is an implementation
choice, not a guarantee of global optimality or recovery for every orbit.

The core uses nondimensional rotating coordinates; it does not silently interpret
SI arrays as these coordinates. The default units match the paper and existing
synthetic generator: `DU=384338000 m`, `TU=375099 s`, `mu=0.0121505`.

Optional `los_rate` is the LOS derivative with respect to nondimensional time.
When supplying it, also supply `observer_velocity`. For bounded measurements
with rates, provide a full positive-definite `covariance` of shape `(P,4,4)` in
the order `[east LOS, north LOS, east LOS_rate, north LOS_rate]`. Angles-only
covariance has shape `(P,2,2)`. No measured angular rates are fabricated from the
five input observations.

`solve` also accepts an explicit `mesh` and `initial_states`. The first and last
mesh times must equal the first and last observation times. The solver never
reads the repository's truth CSV to build an initial guess or select a solution.

### Choosing the maneuver and measurement models

| Setting | Meaning |
|---|---|
| `mode="continuous"` | Minimize Simpson quadrature of squared acceleration, with independent endpoint and midpoint controls. |
| `mode="impulsive"` | Minimize componentwise L1 velocity increments using nonnegative positive/negative slack variables. |
| `mode="hybrid"` | Include both kinds of control, weighted by `hybrid_weight`. |
| `measurement_mode="exact"` | Enforce the supplied measured directions/rates exactly. |
| `measurement_mode="bounded"` | Constrain each observation's Mahalanobis residual using `confidence`. |

Using exact mode on noisy observations can assign measurement errors to fictitious
control inputs. Bounded mode represents uncertainty but does not guarantee
maneuver identifiability. All-zero angular `sigma` without a covariance selects
exact observations; mixed zero/nonzero fallback sigmas are rejected.

Settings such as range bounds, control bounds, impulse bounds, mesh density,
confidence, and initial guess are explicit modeling choices. The paper does not
provide a universal numerical choice for them. Its minimum-control formulation
is different from unconstrained weighted least squares.

The solver rejects unsuccessful or infeasible NLP results. It also checks a
sampled mesh-resolution indicator, with default `mesh_tolerance=1e-3`: at three
interior points per interval, multiply the discrepancy between the interpolating
polynomial's derivative and the dynamics by the interval width and divide each
state component by `1 + abs(state)`. The largest component must pass the
threshold. This catches some poorly resolved solutions that satisfy the endpoint
constraints. It is a heuristic, not a certified error bound or a replacement for
refining the mesh and independently propagating the result.

### Results and impulse convention

`result.t` and `result.states` hold mesh times and six-component states.
`result.controls` holds interleaved node/midpoint accelerations, with shape
`(2*N-1,3)`. `result.increments` holds the decision variables from the paper's
Equation 29. The physical right-minus-left velocity jumps are
`result.impulses = -result.increments`, at `result.t[1:]`.

`state_at(t, side="right")` returns the stored state at a node;
`side="left"` returns the preceding interval's limit. Away from nodes, evaluation
uses the paper's cubic interpolant. Read the
[impulse interpolation caveat](docs/paper_mapping.md#impulse-sign-timing-and-interpolation-caveat):
the published transcription has an ambiguity around discontinuous endpoint
velocities, and its discrete solution is not an exact physical propagation.

The optional Orekit bridge requires the caller to supply an explicit frame,
reference date, and gravitational parameter. It does not reinterpret a rotating
state as an inertial orbit or start the JVM automatically.

```python
from collocation_iod.orekit_bridge import to_cartesian_orbit

# Start the JVM and construct these Orekit objects in the calling application.
# `estimate` is the SI StateEstimate returned by the adapter above.
orbit = to_cartesian_orbit(
    estimate, frame=emi_frame, reference_date=reference_date, gm=earth_gm,
)
```

`emi_frame` must describe the axes of `estimate.r` and `estimate.v`; supplying a
GCRF label alone does not transform synthetic EMI coordinates into GCRF.

## Containers

The standalone container includes Python, the solver, its numerical dependencies,
and the example observations. No host Python installation or Java runtime is
needed for that image. From the repository root:

```text
docker build -f containers/Dockerfile.solver -t collocation-iod .
docker run --rm collocation-iod
```

For your own CSV, mount a directory containing it and request an output inside
that mount. For example, on Linux/macOS:

```sh
docker run --rm -v "$PWD/my-data:/data" collocation-iod /data/observations.csv --output /data/estimate.json
```

In PowerShell:

```powershell
docker run --rm -v "${PWD}/my-data:/data" collocation-iod /data/observations.csv --output /data/estimate.json
```

The optional Orekit target includes Java and the Python Orekit binding:

```text
docker build -f containers/Dockerfile.solver --target orekit -t collocation-iod:orekit .
```

That target supplies Java-object interoperability. The full-stack recipe bundles
the existing simulation source, its synthetic observations and grading truth,
and the [included IERS leap-second history](data/orekit/README.md). That data is
sufficient for the stack's synthetic J2000/geocenter scenario, whose Moon and
Earth-orientation models are supplied by the stack. Build from the repository root:

```text
docker build -f containers/Dockerfile -t collocation-iod:stack .
python scripts/run_container.py --image collocation-iod:stack --output-dir ./outputs/container
```

No separate Orekit-data download is required for that demonstration. This is not
a full Orekit data distribution: real ground stations or realistic lunar/solar
ephemerides require the corresponding data and model changes. Building requires
network access for OS/Python dependencies and sufficient disk space. Runtime
solver use does not download dependencies. Full-stack reports are written to
`/workspace/outputs`; the launcher bind-mounts the selected host output directory there.
See [the configuration guide](../../docs/configuration.md). The stack compatibility
CLI no longer overrides the IOD selected in YAML.

Both recipes include a numerical smoke run during the build; the full-stack
recipe also exercises Orekit refinement. A failed smoke run stops the build.

To transfer an already built image:

```text
docker save collocation-iod:stack -o collocation-iod-stack.tar
docker load -i collocation-iod-stack.tar
```

Docker was unavailable in the implementation workspace, so these recipes require
a first build and runtime smoke test on a machine with Docker. This is separate
from validation of the Python solver.

## Verification and scientific limits

Validation on 2026-11-06: **44 tests passed** (29 module tests and 15 existing
stack tests), including the Java/Orekit-dependent tests. The Python wheel also
built successfully. The container recipes have not been built in this workspace.

Run package tests from the repository root:

```text
python -m pytest packages/collocation-iod/tests -q
```

The tests are intended to check transcription identities, derivatives and
constraints, independent synthetic recovery, frame/unit conversions, invalid
inputs, and integration contracts. They do not reproduce the authors' numerical
tables: full observation/control datasets and all solver settings are not included
in the paper.

The actual `collocation -> Orekit BLS -> Orekit propagation` chain was also run
on the bundled five-observation synthetic case. At day 4 the IOD error was
16.24 km / 0.089 m/s; after refinement it was 15.12 km / 0.068 m/s, with a maximum
position error of 52.92 km over eight days. These are results for the repository
scenario, not reproductions of the paper's maneuver examples. IPOPT is local and
results can vary with numerical environment and initialization.

With a supported Java runtime and the stack dependencies installed, repeat that
integration check from the repository root:

```text
python packages/collocation-iod/examples/verify_stack.py
```

Use `--java-home PATH` if needed. The check uses the bundled leap-second history,
writes reports to `outputs/stack`, and checks actual Orekit
Cartesian-orbit conversions. Only the grading stage reads synthetic truth.
Add `--run-tests` to run the module and original stack test suites in the same
Java-enabled process (install the `test` extra first).

Before using reconstruction outputs for scientific conclusions:

- Check solver status and constraint feasibility, then repeat with a finer mesh.
- Compare the solution with independently propagated dynamics and any applicable
  maneuvers; finite transcription feasibility alone is insufficient.
- Explore observation geometry, uncertainty, initialization, and model bounds.
- Account for the CR3BP model's omitted solar gravity, eccentric lunar motion,
  and other perturbations, which can otherwise appear as reconstructed control.

The existing geocenter angles-only example is an integration example. The paper's
impulsive examples additionally use moving observers and measured angle rates.
The current stack's Orekit batch estimator itself still supports only the
geocenter observer.

## Implementation and dependencies

| Module | Responsibility |
|---|---|
| [`models.py`](src/collocation_iod/models.py) | Explicit data contracts, input validation, CR3BP dynamics, and solver settings. |
| [`transcription.py`](src/collocation_iod/transcription.py) | Hermite–Simpson defects and state/control interpolation. |
| [`solver.py`](src/collocation_iod/solver.py) | Sparse NLP construction, initialization attempts, result evaluation, and acceptance diagnostics. |
| [`adapters.py`](src/collocation_iod/adapters.py) | Stack registration, EMI/rotating conversions, and existing state contract. |
| [`orekit_bridge.py`](src/collocation_iod/orekit_bridge.py) | Optional explicit conversions to/from Orekit Cartesian orbit objects. |
| [`cli.py`](src/collocation_iod/cli.py) | CSV input and JSON output. |

[NumPy](https://numpy.org/doc/stable/) supplies array operations.
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
supplies chi-square quantiles, interpolation, and Lagrange-point root finding.
[CasADi](https://web.casadi.org/docs/) constructs sparse derivatives and calls
[IPOPT](https://www.coin-or.org/Ipopt/) for constrained nonlinear optimization.
The core container installs the versions in `requirements.lock`; Java and Orekit
are optional interoperability dependencies.

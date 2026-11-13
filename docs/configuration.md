# YAML-controlled simulation and local Docker files

The product command is `cislunar-stack --config PATH`. The compatibility name
`collocation-iod-stack` now has identical selection behavior. Neither command
accepts simulation overrides such as `--iod`, `--set`, or `--propagator`.
`--show-config` prints the resolved configuration without starting a simulation.
The standalone numerical `collocation-iod` research CLI remains independent.

Start with [the complete synthetic configuration](../scenarios/plug_and_play/config.yaml)
or [the CSV configuration](../scenarios/observations/config.yaml). Versioned configs
use `schema_version: 1`; unknown keys and unsupported choices are rejected. The
older stage-oriented YAML schema remains readable for existing scenarios and
resolved run configurations.

## Local files and bind mounts

The launcher uses Python's standard library only; no host numerical packages or
Java installation are needed. Docker must be installed and running with Linux
container support. Build and run the complete default example from the repo root:

```powershell
python scripts/run_container.py --build --output-dir "C:\Users\veerc\cislunar-results"
```

For your own local configuration and observation directory:

```powershell
python scripts/run_container.py --config "C:\My Experiment\config.yaml" --input-dir "D:\Sensor Data" --output-dir "D:\Simulation Results"
```

On Linux/macOS the same launcher accepts native paths:

```sh
python3 scripts/run_container.py --config "$HOME/experiment/config.yaml" --input-dir "$HOME/sensor-data" --output-dir "$HOME/simulation-results"
```

The image defaults to `cislunar-stack:local`; `--image` selects another deployment
version. `--build` rebuilds this repository's image. Neither changes simulation
choices. Add `--dry-run` to inspect the exact Docker arguments without running
Docker or creating directories.

| Local resource | Container location | Access |
|---|---|---|
| File selected by `--config` | `/workspace/config.yaml` | Read-only |
| Directory selected by `--input-dir` | `/workspace/data` | Read-only |
| Directory selected by `--output-dir` | `/workspace/outputs` | Read/write |

If omitted, the input directory is the `data` folder beside the config, when it
exists, and the output directory is `outputs` beside the config. Synthetic mode
does not need an input directory. Explicit input/output directories may be anywhere
on the host, subject to Docker's file-sharing permissions. They must not overlap.
Windows paths and spaces are passed as separate arguments, without shell evaluation.
On POSIX hosts the launcher runs with the caller's UID/GID so local results are
not owned by container root. Plot caches are stored in the container's temporary directory.

Inside **config.yaml**, refer to mounted files using portable paths:

```yaml
schema_version: 1
input:
  source: csv
  observations: data/synthetic_obs.csv
  truth: null
output:
  directory: outputs/experiment_01
```

With `--output-dir "D:\Simulation Results"`, those results appear under
`D:\Simulation Results\experiment_01`. Do not put Windows host paths in YAML
when running the Linux container; the host path is mapped by the launcher.
The container rejects an output directory outside `/workspace/outputs`, avoiding
results that disappear with the container. The launcher does not rewrite YAML.
Native execution resolves relative paths against the config's directory and also
accepts native absolute paths.

## Scientific configuration

| Section | Purpose |
|---|---|
| `run` | Run name and synthetic random seed |
| `system` | Earth-Moon CR3BP mass ratio and distance/time units |
| `input` | `synthetic` generation or `csv` observations; optional truth file |
| `simulation.truth` | Initial state, its epoch, truth model and integration tolerances |
| `simulation.observer` | `geocenter` or a fixed position in the synthetic EMI frame |
| `simulation.observations` | Explicit observation schedule, RA/Dec noise and uncertainty |
| `determination.iod` | Method, epoch observation index and method-specific options |
| `determination.estimator` | `orekit_bls` or `none`, with method-specific options |
| `determination.advanced` | Optional Orekit integration/orbit settings; defaults hidden from routine use |
| `prediction` | Orekit/SciPy backend, start/end times and sample count |
| `evaluation` | Optional truth comparison, independent of output generation |
| `output` | Output directory, plotting and observation export |
| `runtime` | Optional Orekit-data location; image provides its synthetic default |

Synthetic generation is an in-memory stage. It never overwrites the input CSV or
passes truth into the estimator. Editing the initial state, seed, observation
times or noise in YAML changes the generated case without editing Python. Times
are absolute offsets from the simulation frame origin; the initial state's epoch
can differ from the first observation. This preserves the CR3BP rotation phase.

Defaults use Gaussian noise added independently to RA and Dec, with one common
standard deviation in arcseconds. Explicit times permit irregular cadence and
gaps. Truth and determination models are separate settings, but both currently
support CR3BP only. Unsupported dynamics, maneuvers, and observer types fail
validation rather than silently falling back to another model.

For a Java-free native run, edit these fields in YAML:

```yaml
determination:
  estimator:
    method: none
prediction:
  backend: scipy
```

To select range-search IOD, replace the IOD block (do not retain collocation options):

```yaml
determination:
  iod:
    method: range_search
    epoch_index: 2
    options:
      range_km: [150000, 650000, 10000]
      range_rate_m_s: [-800, 800, 50]
```

Two-body `gooding`, `gauss`, and `laplace` are available for appropriate cases;
they are not reliable choices for the baseline halo. `truth_plus_error` is kept
only in the legacy test interface. Collocation options correspond to `SolverOptions`.
Estimator options include `optimizer`, iteration/evaluation limits,
`convergence_threshold`, `covariance`, and `residual_gate_sigma`.

### Residual check and IOD fallback

`determination.estimator.options.residual_gate_sigma` (default 3) rejects a batch least-squares
fit whose residual RMS, in measurement standard deviations, exceeds that value. Correct fits stayed
at or below 1.6 at 1–100″ noise; wrong local solutions were at hundreds
([benchmarks](../benchmarks/collocation_iod/RESULTS.md)). Set it to `null` to turn the check off.
Below about 0.5″ noise it can reject correct fits, because Orekit's angles include light time and
the synthetic generator's do not.

`determination.iod.fallback` (default `[range_search]`) lists IOD methods to try, in order, when the
selected IOD fails or the estimator rejects its fit. The first fit that passes is used. The summary
records every attempt under `attempts`, and `iod` names the method that produced the result. If none
passes, the run stops with an error listing each attempt. Use `fallback: []` to run one method only.

Collocation's `impulsive` and `hybrid` modes estimate burns, which `orekit_bls` cannot represent; that
combination is rejected before computation (use `estimator: method: none` to keep the collocation
trajectory). Collocation also needs about 10 mesh nodes per day of observation arc (minimum 20); a
coarser mesh is rejected once the observation times are known, before any computation.

Every `summary.json` records `provenance`: Python, package versions, machine and OS. Collocation's
optimizer can settle on slightly different solutions on different platforms, so compare runs made
with the same container image.

## Output contract

Successful product runs always export:

- `state.json`: state, frame/units, epoch, covariance availability, and diagnostics.
- `trajectory.csv`: predicted time, position and velocity in EMI/SI.

The versioned configuration also saves `summary.json` and `config_used.yaml` by
default. The latter is a resolved, replayable **stage-oriented** configuration,
including expanded defaults and absolute runtime paths. `observations.csv` and
`results.png` are controlled by the output flags. When truth evaluation is
available, `trajectory_comparison.csv` contains the prediction/truth comparison.
Generated observation exports can be used as CSV input in a subsequent run.

Truth-free CSV runs still write states, trajectories, summaries, configurations,
and optional plots. Reusing an output directory replaces same-named run files;
use a distinct `output.directory` for experiments you want to retain.

## Scope and validation

The existing Orekit batch estimator remains geocenter-only; unsupported observer
combinations are rejected before constructing the estimator. Zero-noise data are
supported by collocation, but Orekit BLS requires positive uncertainty. Fixed EMI
observers are a synthetic geometry, not rotating terrestrial ground stations.
Real epoch/frame transforms, sequential updates, and applying reconstructed
maneuvers downstream remain separate work. The old generator script is retained
as a historical example; the product uses `od_stack.synthetic` instead.

Bind argument construction, paths with spaces/commas, read-only inputs, output
containment, YAML selection, synthetic reproducibility, and truth-free export are
covered by automated tests. Actual Docker build/run verification requires a
Docker-enabled machine; Docker is unavailable in this workspace.

Verification on November 7, 2026: the full suite passed 62 tests, the final
focused YAML/path suite passed 18 tests, and the versioned synthetic example
completed the collocation/Orekit chain with 15.117 km refined epoch error.
The original baseline's error metrics remained unchanged. Original files remain
preserved in the reorganization archive, with an additional pre-YAML source snapshot.

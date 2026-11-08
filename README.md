# AFRL cislunar orbit determination

Fall 2026 project: containerized software for generating and updating cislunar
orbit paths from synthetic sensor data.

The repository contains a working CR3BP batch orbit-determination prototype and
an independently packaged collocation IOD solver. Sequential updating and broader
product validation remain unfinished; see [project status](docs/project_status.md).

## Repository layout

| Directory | Responsibility |
|---|---|
| [packages/cislunar-stack](packages/cislunar-stack/README.md) | Installable observation -> IOD -> estimation -> propagation pipeline |
| [packages/collocation-iod](packages/collocation-iod/README.md) | Standalone numerical solver and optional stack adapter |
| [scenarios](scenarios/halo_geocenter/README.md) | Scenario configuration, observations, and grading truth |
| [simulation](simulation/README.md) | Synthetic observation generation |
| [examples](examples/README.md) | Weber, Artemis, STM, and Orekit tutorial cases |
| [containers](containers/README.md) | Full-product and standalone-solver Docker recipes |
| [benchmarks](benchmarks/README.md) | Validation entry point and future experiment location |
| [archive](archive/README.md) | Preserved scripts, figures, artifacts, and exact pre-migration files |
| `outputs/` | Generated reports; ignored by Git |

## Run with local inputs and outputs

All simulation choices are controlled by [config.yaml](scenarios/plug_and_play/config.yaml).
The full image uses one neutral `cislunar-stack` entry point. It does not force
collocation or override the YAML output directory.

With Python and Docker installed, build and run from the repository root:

```text
python scripts/run_container.py --build --output-dir ./outputs/container
```

Use your own local files (PowerShell example):

```powershell
python scripts/run_container.py --config "C:\My Experiment\config.yaml" --input-dir "D:\Sensor Data" --output-dir "D:\Simulation Results"
```

The configuration and inputs are bind-mounted read-only; results are written
directly to the selected local output directory. In YAML, use `data/file.csv` for
input paths and `outputs/run_name` for output paths. See the
[configuration and mounting guide](docs/configuration.md) for the exact mappings,
CSV example, supported choices, and native execution.

## Native installation

```text
python -m pip install -e "./packages/cislunar-stack[orekit,test,examples]"
python -m pip install -e "./packages/collocation-iod[test]"
cislunar-stack --config scenarios/plug_and_play/config.yaml
```

For native Orekit, configure a supported Java runtime and set `OREKIT_DATA` to the
absolute path of `packages/collocation-iod/data/orekit`. To avoid Java, set
`determination.estimator.method: none` and `prediction.backend: scipy` in YAML.
The container already supplies Java and its synthetic Orekit data.

For verification:

```text
python packages/collocation-iod/examples/verify_stack.py --run-tests
```

Use `--java-home PATH` if needed. Docker build/runtime verification remains pending
because Docker is unavailable in this workspace. [Migration details](docs/repository_layout.md)
and original files are retained; subsequent YAML changes are documented in the
[configuration guide](docs/configuration.md).

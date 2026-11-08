# Repository reorganization

Completed November 6, 2026.

This is the historical layout migration record. The subsequent November 7 YAML-only
interface, truth-independent outputs, and local bind mounts are documented in
[configuration.md](configuration.md); its instructions supersede the old CLI semantics below. This change separates reusable packages, scenario
inputs, synthetic generation, exploratory cases, deployment, and historical
artifacts. Numerical algorithms and existing example calculations are preserved.

## Current layout

```text
packages/
  cislunar-stack/             # installable od_stack package, tests, requirements
    src/od_stack/
  collocation-iod/            # standalone solver, adapter, tests, solver examples
    src/collocation_iod/
scenarios/halo_geocenter/     # config and original synthetic CSV fixtures
simulation/                  # original observation-generator script
examples/
  artemis_comparison/         # original script and OEM fixture
  weber_cr3bp/
  stm_sensitivity/
  orekit_tutorials/           # original tutorial container and text outputs
containers/                  # full product and standalone solver recipes
benchmarks/                  # entry-point documentation; future studies
docs/
archive/                     # historical scripts/docs/figures and backup
outputs/                     # ignored generated results, including preserved runs
```

## Old-to-new paths

| Before | Now |
|---|---|
| `afrl_cislunar/od_stack/` | `packages/cislunar-stack/src/od_stack/` |
| `afrl_cislunar/tests/` | `packages/cislunar-stack/tests/` |
| `afrl_cislunar/cr3bp.py` | `packages/cislunar-stack/src/od_stack/reference_cr3bp.py` |
| `afrl_cislunar/requirements.txt` | `packages/cislunar-stack/requirements.txt` |
| `collocation_iod/` source/package files | `packages/collocation-iod/` |
| `afrl_cislunar/config.yaml` | `scenarios/halo_geocenter/config.yaml` |
| `afrl_cislunar/data/synthetic_*.csv` | `scenarios/halo_geocenter/data/` |
| `afrl_cislunar/obs_generator.py` | `simulation/obs_generator.py` |
| `afrl_cislunar/artemis2_vs_cr3bp.py` and OEM file | `examples/artemis_comparison/` |
| `afrl_cislunar/weber_cr3bp_example.py` | `examples/weber_cr3bp/weber_cr3bp_example.py` |
| `afrl_cislunar/stm_check.py` | `examples/stm_sensitivity/stm_check.py` |
| `docker/orekit-cr3bp/` | `examples/orekit_tutorials/` |
| `collocation_iod/Dockerfile.stack` | `containers/Dockerfile` |
| `collocation_iod/Dockerfile` | `containers/Dockerfile.solver` |
| `collocation_iod/output/` | `outputs/collocation-original/` |
| `collocation_iod/build/`, `dist/` | `archive/build-artifacts/collocation-iod/` |
| `afrl_cislunar/figures/` | `archive/meeting_figures/` |
| Root extensionless prototype scripts | `archive/early_scripts/` |
| Earlier simulation README | `archive/early-docs/afrl_cislunar_README.md` |
| Remaining old simulation-directory metadata/caches/log | `archive/legacy-layout/afrl_cislunar/` |
| Root JVM crash log | `archive/logs/` |

The ordered, machine-readable move list is in
[`archive/reorganization/moves.json`](../archive/reorganization/moves.json).
Unstaged Git status shows removed old paths and untracked new paths. These are
filesystem moves; the preservation audit checks that every destination exists.
No staging or commit was performed.

## Compatibility changes

- Python import names remain `od_stack` and `collocation_iod`; distribution
  directory names do not affect imports. Install both from their new paths.
- `cislunar-stack` is a new console entry point for the existing runner.
  `collocation-iod` and `collocation-iod-stack` remain available.
- Shared original CR3BP/STM routines now live in `od_stack.reference_cr3bp`.
  The generator and examples import this installed module.
- Observation, truth, Orekit-data, and report paths read from YAML resolve
  relative to that YAML file. Explicit CLI path overrides resolve from the shell
  working directory. Programmatic `run(dict)` retains its existing path semantics.
- The generator writes to `outputs/synthetic_generation`, preserving the original
  scenario fixtures. Artemis and STM plots go to their own `outputs/` subfolders.
- Verification/test fixture paths and Docker COPY paths use the new structure.
  Both application Dockerfiles now use the repository root as build context.
- The verification script writes new results to `outputs/stack`, leaving the
  earlier run under `outputs/collocation-original` untouched.

This migration does not introduce new estimation physics, observation updates,
or truth-independent reports. The generator remains a script, and optional
truth evaluation still resides in the existing stack modules. Those functional
changes remain in the [project assessment](project_status.md).

## Preservation

Before edits, 129 original project files were saved byte-for-byte in
`archive/reorganization/original-files.zip`, with SHA-256 hashes. This includes
untracked and ignored project files, but excludes the Git database and virtual
environment. Every original file also has a current destination in the layout;
the ZIP additionally preserves old contents of files edited for compatibility.

```text
python archive/reorganization/verify_preservation.py
```

The audit verifies all archive hashes and all relocated file destinations, and
lists files edited since the snapshot. No original code, fixture, plot, or saved
result was discarded. Original build artifacts and logs remain locally even
where Git ignores them.

## Verification

- Both packages installed successfully in editable mode from their new paths.
- Both distributable wheels built successfully.
- Both wheels were installed into a separate target directory; imports and CLI
  help succeeded with editable-install hooks disabled, confirming that the
  distributions contain the packages rather than relying on the source checkout.
- Full collocation -> Orekit BLS -> Orekit propagation verification passed using
  Java 21, with the exact same baseline position-error metrics as the preserved
  run: 16.235080 km IOD, 15.117042 km refined epoch, 52.924069 km maximum over
  eight days.
- **46 tests passed**: the existing 44 plus two config-path regression tests.
  Expected warnings identify the geocenter assumption; pytest also reported an
  inability to update an old cache directory. No tests failed or skipped.
- Installed CLI ran successfully from `outputs/`, outside the source and scenario
  directories, using an absolute configuration path.
- Synthetic generation, Weber, STM, and Artemis examples all ran headlessly.
  Astropy IERS auto-download was disabled for that check. Interactive-show warnings
  are expected with the headless plotting backend.
- Preservation audit: 129/129 archived originals verified; 129/129 relocated
  files present.
- Docker build/runtime verification remains pending because Docker is unavailable
  in this environment. Updated recipes retain their numerical smoke commands.

The Orekit verification needed execution outside the sandbox after Java failed
to access its JAR filesystem inside it. The wheel build likewise required access
to pip's cache. These are environment execution constraints, not test failures.
The separate wheel-install directory also required an unsandboxed read check
because pip's temporary-directory permissions were retained on installation.

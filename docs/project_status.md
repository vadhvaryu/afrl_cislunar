# Project status and repository cleanup

Assessment date: November 6, 2026.

**Historical assessment:** this document records the state before the subsequent
repository reorganization. Paths and migration steps below describe that earlier
layout. The layout migration and stack packaging have since been completed; see
[the current layout and verification](repository_layout.md). Scientific/product
gaps such as observation updates and truth-independent reports still apply.

Source: page 1 of the supplied **AFRL Student Project Guideline (1).pdf**,
"Build a Containerized Cislunar Orbit Dynamics Software Solution." The document
is used as project requirements, not as instructions to execute actions.

## Overall assessment

The repository has a functioning research prototype for synthetic CR3BP orbit
determination and prediction. It is beyond the initial dynamics-only stage
described by the older README. It is not yet demonstrated as the robust,
portable, easy-to-use generate-and-update product described in the guideline.

The main remaining work is an observation-update workflow, separation of the
product from simulation grading, broader validation, and a verified container
release. Another IOD algorithm is not the principal missing deliverable.

## Requirements and evidence

| Guideline objective | Current evidence | Status and remaining work |
|---|---|---|
| Generate cislunar orbit determinations and paths | `od_stack/iod.py`, `estimation.py`, `propagators.py`, and `run.py` implement the full chain. CR3BP range search and optional collocation provide IOD; Orekit BLS refines the state; Orekit/SciPy propagate it. | Implemented for the synthetic baseline. Establish the supported orbit/geometry envelope rather than generalizing from one case. |
| Improve confidence as information is acquired | BLS uses all supplied observations and can return a 6-by-6 state covariance. | Partial. There is no sequential filter or explicit workflow for loading a previous estimate and updating it with later observations. Re-running a batch fit is possible but has not been packaged or validated as the update feature. |
| Generate and update with minimal effort | YAML configuration, CLI overrides, method registries, plots, CSV, JSON, and saved configuration exist. | Partial. Runtime depends on working directory, Java/Orekit setup, and repository layout. Output handling assumes a truth file. A single documented product entry point and update command/workflow are needed. |
| Implement and test the algorithms | Numerical, frame, solver, maneuver, adapter, and stack tests exist. Saved end-to-end results exist. | Meaningful baseline coverage. Broader scenarios, repeated-noise experiments, invalid-input coverage, update tests, and container tests remain. |
| Containerize for deployment across systems | Core and full-stack Dockerfiles exist in `collocation_iod/`; a separate Orekit tutorial Dockerfile exists. | Recipes exist; the collocation README explicitly says these images have not been built in this workspace. Docker was not discoverable on PATH during this assessment. Cross-system deployment remains unverified. |
| Synthetic data that mimics real systems | Generator produces noisy RA/Dec and separate truth; resampling supports repeatable noise seeds. | Partial realism. Main integrated case uses five geocenter observations over eight days, matched CR3BP truth/estimation physics, and a synthetic frame/time convention. Realistic observer geometry and observation scheduling need coverage. |

The guideline does **not** prescribe a particular Kalman filter, accuracy
threshold, UI, force-model fidelity, maneuver solver, or number of scenarios.
Kalman filtering is an example in the task list. These acceptance criteria need
to be chosen and recorded; they should not be presented as requirements supplied
by the document. Higher-fidelity dynamics and maneuver reconstruction are useful
extensions, not explicitly mandatory deliverables.

## What is already done

- CR3BP dynamics, frame/unit conversions, STM/Jacobian work, and independent
  SciPy/Orekit propagation checks.
- Synthetic optical observations and separate truth data; controlled noise and
  observation-subset settings.
- Shared `Observations`, `StateEstimate`, and `Trajectory` contracts with
  interchangeable IOD, estimator, and propagator implementations.
- CR3BP range-search IOD; two-body Gooding/Gauss/Laplace options for comparisons;
  a clearly identified truth-based test initializer.
- Orekit batch least squares, optimizer choices, state covariance, prediction
  windows, and error plots/reports.
- A separately installable collocation package with continuous, impulsive, and
  hybrid formulations, explicit numerical inputs, CLI, tests, and stack adapter.
- Container recipes, package metadata, and substantial technical documentation.

The collocation package is an IOD component, not the complete simulation-stack
product. The product boundary is the pipeline in `od_stack`, with collocation as
an optional estimator component and synthetic generation/evaluation alongside it.

## Validation evidence and limits

The saved `collocation_iod/output/stack/summary.json` reports the actual
collocation -> Orekit BLS -> Orekit propagation chain for the five-observation
case: 16.235 km initial position error, 15.117 km refined error at day 4,
0.0683 m/s refined velocity error, and 52.924 km maximum position error over
days 0-8. These are inspected existing results, not a fresh integration run in
this assessment. The reported approximately 25.982 km position uncertainty is
the square root of the position covariance trace, not a calibrated spherical
coverage guarantee. Days 0-8 are the fitted observation interval, not an
independent future-prediction test.

The collocation README records a previous 44-test pass (29 package tests and
15 stack tests), including Orekit, and a wheel build. This assessment freshly
ran `python -m pytest collocation_iod/tests -q`: **29 passed in 21.39 seconds**.
The Java-dependent full-stack run was not repeated. Broad robustness,
covariance calibration, and cross-platform execution do not follow from those
test counts alone.

## Concrete product gaps

1. **Observation updates:** choose an initial implementation, such as refitting
   accumulated observations or a sequential EKF/UKF. Persist the state, epoch,
   frame, covariance, configuration, and observation provenance. Demonstrate an
   initial arc followed by later observations and compare predictions before and
   after the update. Do not require every noisy update to reduce the actual error.
2. **Truth-independent output:** `od_stack/run.py` returns early when `data.truth`
   is absent, so the normal file-writing path is skipped. `results.py` requires
   truth for its main reporting functions. Always export the estimate and
   trajectory; make truth-based metrics an optional evaluation layer. Also,
   deleting the YAML truth key restores its default path in `config.py`; explicit
   `truth: null` is currently necessary to disable it.
3. **Observer realism:** observation contracts can carry observer positions, but
   `OrekitBatchLS._measurements` rejects observers more than 1 m from the
   geocenter. Support and validate at least one representative ground or space
   observer through the complete chain. Document cadence, gaps, noise, and
   visibility assumptions.
4. **Frame and time metadata:** EMI is a simulation frame, passed to Orekit under
   a GCRF label with J2000-relative dates. This is internally useful for the
   circular model but is not a real-data frame transformation. Either retain an
   explicitly synthetic supported interface or implement real epochs and frame
   conversions before claiming compatibility with real ephemerides.
5. **Uncertainty and failures:** covariance extraction currently catches errors
   and returns `None`. Surface that status in results. Add propagated uncertainty
   if prediction confidence is a supported output, and test calibration across
   repeated independent noise draws. Expand malformed-input/configuration tests.
6. **Maneuver scope:** reconstructed controls remain on the collocation result;
   downstream BLS and propagation use uncontrolled CR3BP. A maneuver-aware full
   stack requires a control-history contract and matching force/impulse models.
   Otherwise explicitly scope the product demonstration to ballistic motion.
7. **Deployment evidence:** build the full image, run with mounted inputs and
   outputs, run without truth, verify clean-machine installation, and exercise
   each supported host/architecture. Add automated checks and a reproducible
   release/version record. The old tutorial container is separate evidence from
   the full application container.

## Proposed repository layout

Keep one repository with two installable packages. Separate reusable code from
scenario inputs and generated results; a second Git repository is unnecessary.
This is a proposed migration, not the current filesystem layout.

```text
README.md
packages/
  cislunar-stack/
    pyproject.toml
    src/od_stack/               # reusable pipeline, CLI, contracts, dynamics
    tests/
  collocation-iod/
    pyproject.toml
    src/collocation_iod/        # standalone numerical solver + optional adapter
    tests/
simulation/
  generate.py                  # reusable synthetic observation generation
  evaluate.py                  # optional truth comparisons and benchmark metrics
scenarios/
  halo_geocenter/
    config.yaml
    observations.csv
    truth.csv                  # used by generation/evaluation, not estimation
    README.md                  # assumptions and expected baseline
examples/
  weber_cr3bp/
  artemis_comparison/           # ephemeris fixture + provenance
  stm_sensitivity/
  orekit_tutorials/
benchmarks/                     # repeatable noise/geometry/update experiments
containers/
  Dockerfile                   # complete product
  Dockerfile.solver            # optional standalone solver
docs/
  project_status.md
  architecture.md
  validation.md
archive/
  early_scripts/
  meeting_figures/
outputs/                       # ignored generated run artifacts
```

The proposed simulation modules should have callable functions and explicit paths;
the present `obs_generator.py` is a script with generation/output at module level.
The supported user interface should accept a scenario/config path and output
directory from any working directory. Example code must import the installed
packages rather than require the product to import examples.

## Migration sequence and acceptance checks

1. **Establish the baseline and navigation.** Preserve existing scenario/results
   as reference evidence, record dependencies and tests, and add a root README.
   This assessment adds the README and this document; runtime files are not moved.
2. **Extract package and output boundaries first.** Package `od_stack`, separate
   optional truth evaluation, and make input paths resolve relative to the config
   file. Expose a product CLI; keep the numerical collocation core independent.
   Verify an observations-only run exports state, trajectory, and diagnostics.
3. **Move cases and history together with their dependencies.** Move the halo
   config/fixtures into `scenarios`, exploratory scripts into `examples`, and
   dated figures/root extensionless prototype scripts into `archive`. Inventory
   prototype differences before treating them as disposable duplicates. Extract
   shared `cr3bp.py` functionality where examples/generation still need it.
4. **Repair every layout dependency in the same change.** Update Docker COPY
   paths/build contexts, READMEs, example-relative data paths, test fixtures,
   `collocation_iod/tests/conftest.py`, and `examples/verify_stack.py`. The latter
   currently discovers a sibling `afrl_cislunar` folder and inserts it into
   `sys.path`; simply moving directories will break it.
5. **Verify the reorganized product.** Re-run numerical and full-stack tests,
   compare the preserved baseline within tolerances, install built packages in a
   clean environment, and invoke the CLI from outside the source tree. Build and
   run the container against external scenario inputs, including without truth.
6. **Complete the update and robustness deliverables.** Add the observation-update
   demonstration, realistic geometry, repeatable benchmark matrix, and a short
   validation report with explicit supported conditions and failure cases.

During this review, `git status` showed the root `.gitignore` and the entire
`collocation_iod/` directory as untracked. The local working tree therefore has
capabilities that a clone of the tracked repository will not yet contain.
Review and include intended source/fixtures in a release; exclude build outputs,
caches, environments, and raw run artifacts. No staging or commit was performed.

## Suggested completion criteria

These are proposed acceptance checks, not quantitative requirements from AFRL:

- One documented container command estimates and exports a trajectory from a
  supplied synthetic observation file, without access to truth.
- A documented second run incorporates later observations and exports the updated
  estimate, uncertainty status, and prediction.
- A benchmark report covers the baseline, a representative non-geocenter
  observer, varied noise/cadence/arc length, and deliberately difficult inputs.
- Results distinguish fitted-arc accuracy from held-out prediction accuracy,
  estimator success from failure, and covariance availability from calibration.
- A clean checkout/package/container reproduces the supported workflow on the
  declared target systems. Examples and historical figures are optional assets.

Recommended priority: finish the product interface and truth separation, then
the update workflow and container verification, then broaden the realism and
robustness evidence. Keep maneuver research separately scoped until its outputs
are propagated consistently through the complete stack.

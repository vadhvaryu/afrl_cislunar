# Project status and handoff (November 19, 2026)

Where the orbit-determination stack stands at the end of the fall work, what is still open, and where to
start. Details are in the linked write-ups.

## What the stack does

`cislunar-stack --config <scenario.yaml>` (or `scripts/run_container.py` for the pinned container) runs the
whole chain on synthetic or CSV angle data:

1. **Simulation.** A CR3BP truth orbit, optional impulsive burns, and RA/Dec observations from a chosen
   observer, with Gaussian noise and light time.
2. **Initial orbit determination (IOD).**
   - Range search: a grid, then a least-squares polish from 12 starts.
   - Collocation (Heidrich & Holzinger): continuous mode, or impulsive mode, which also finds burns.
   - The two-body Orekit methods (Gooding, Gauss, Laplace), which do not work on these orbits.
   - Fallback methods, run in order if one fails.
3. **Batch least squares.**
   - `orekit_bls`: Orekit, coasting, observer at Earth's center.
   - `cr3bp_bls`: CR3BP, any observer, can fit burns.
   - Both reject fits whose residuals exceed 3σ.
4. **Prediction.** Orekit, or the scipy CR3BP, which applies fitted burns.
5. **Outputs.**
   - The state with its covariance, the trajectory, a plot, and a replayable config.
   - Provenance, meaning the versions, machine and container image.
   - Plain-language `warnings` when a result should not be trusted at face value.

Configuration reference: [configuration.md](configuration.md). Scenarios: `scenarios/`.

## Open pull requests, in merge order

The pull requests are stacked, so merge them in this order:

| PR | Branch | Contents | Write-up |
|---|---|---|---|
| vadhvaryu/afrl_cislunar#9 | `collocation-iod-testing` → `main` | Collocation IOD tested against the previous IOD methods (benchmarks only) | `benchmarks/collocation_iod/RESULTS.md` |
| vadhvaryu/afrl_cislunar#10 | `iod-integration-fixes` → #9 | 3σ residual check, IOD fallback, configuration checks, provenance, pinned container, polished range search, short-arc warnings and alternatives, light time | `benchmarks/collocation_iod/IMPROVEMENTS.md` |
| vadhvaryu/afrl_cislunar#11 | `maneuver-bls` → #10 | Batch least squares with burn parameters (`cr3bp_bls`), burns in the simulation | `benchmarks/maneuver_bls/RESULTS.md` |

After a base merges, retarget the next one to `main`.

Two older items were never brought to `main`:
- The Part B IOD methods (`range_grid`, `cr3bp_dc`, `halo_catalog`, on `cr3bp-iod`) were merged only into
  other feature branches.
- The `docker-image` branch was never opened as a pull request. #10 now pins the product container, but
  the full Orekit data (DE-440 ephemerides) from `docker-image` is still only there.

## Results to know

- **8-day arcs.** The product converges on all 100 test cases across five orbit families:
  - the baseline L1 halo;
  - a small L1 halo;
  - a large L2 halo;
  - the 9:2 NRHO;
  - a DRO.

  The error is set by the noise, about 1.9 km per arcsecond on the baseline (19 km at 10″).
- **Short arcs (1 day, 6 hours) from Earth's center are ambiguous.** The angles often fit two orbits about
  11,000 km apart. The stack now finds the second one and warns about it, whichever IOD method runs
  first. On 6-hour arcs the reported uncertainty is unreliable (errors up to 35σ), and the run says so.
- **Burns.** A 5–20 m/s burn is recovered to within about 0.06 m/s (at 1″) and an hour, with errors of
  about 12 km at 1″ and 130 km at 10″ over 8 days. A 1 m/s burn at 10″ is below detection.

## Decisions still open

1. **Which IOD method the product tries first.**
   - Range search first (the current default in #10) converges more often on short arcs.
   - Collocation first is faster on easy cases and was the original plan.
   - Both reach 100/100 on the orbit study and both flag ambiguous short arcs.
   - The default is one line in `od_stack/scenario.py` plus the scenario configs.
2. **Whether `cr3bp_bls` should replace `orekit_bls` as the default.**
   - It is equivalent on coasting orbits in the CR3BP, needs no Java, and handles burns and any observer.
   - Orekit is the path to real ephemerides and real data.

## Known limits

- **No sequential update yet.** The project guideline asks for estimates that improve as observations
  arrive, and there is no workflow for updating a previous estimate with new observations (re-running
  the batch fit on all data is the only way). [project_status.md](project_status.md) (November 6) records
  this and the other product gaps; they still apply.
- **Everything is tested on synthetic CR3BP data.** No real observations, no real ephemerides, no
  ground stations yet.
- **Coverage of the burn work.** Least squares with burns has been tested with one burn, on one orbit,
  with 17 observations. A burn needs more observations than a coasting orbit.
- **Orekit's batch least squares and prediction still assume coasting.**
- **Collocation's answer depends on the machine and on last-digit input changes** (its optimum is flat):
  about 600 m against a 16 km error. Compare runs made in the same container.
- **Settings were chosen on some of the reported draws.** A few of range search's settings came from
  looking at some of the same noise draws the benchmarks report: the number of starts, the "different
  orbit" distance and the grid. A re-test on fresh draws gave the same results (`IMPROVEMENTS.md`), so
  this is a minor caveat now.

## Suggested next steps

1. Merge #9, #10 and `maneuver-bls`, after the team settles the IOD order.
2. **A second observer.** `cr3bp_bls` accepts any observer position. Simulate a ground station or a second
   spacecraft and measure how much it resolves the short-arc ambiguity.
3. **Burns in Orekit.** Estimate maneuver parameters in Orekit's least squares and apply them in its
   prediction, so burns work with real ephemerides.
4. **Burns that collocation misses.** When a coasting fit fails the 3σ check, try fitting an extra burn.
5. **Sequential updates.** A workflow, or a filter, that updates an estimate as observations arrive.
6. **Real data.** Bring in the full Orekit data from `docker-image` and test on real or realistic
   observations.

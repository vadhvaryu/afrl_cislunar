# Improving the stack from the collocation results

Follow-up to [RESULTS.md](RESULTS.md), on branch `iod-integration-fixes`, November 13, 2026. Each
change below came from a weakness the benchmarks showed. Each one was checked against the same
benchmarks: 4 arcs × 3 noise levels × 10 draws, and 5 orbits × 20 draws at 10″. That process is
also the main caveat: see [How much to trust these numbers](#how-much-to-trust-these-numbers).

**Bottom line.**
- **Orbit cases:** the product now converges on 100 of 100 (it was 82; the DRO went from 2/20 to
  20/20).
- **Short arcs:** it converges more often (1 day: 24/30 instead of 18/30; 6 hours: 22/30 instead of
  19/30).
- **Ambiguous short arcs:** every wrong answer that still passes the 3σ check is now flagged, and the
  true orbit is listed among the alternatives.
- **Uncertainty:** it is reported on short arcs and matches the actual error from 1 day up.
- **Low noise:** the residual check now works at low noise.

Range search, now polished, became the product's first IOD method, with collocation as its
fallback. Accuracy on 8-day arcs is unchanged, because there it was already set by the noise.

## What changed and why

| Weakness found | Change | Evidence |
|---|---|---|
| Range search started BLS 19,000–29,000 km off on the baseline and failed outright on the DRO. Its sideways velocity comes from a 3-point estimate of how fast the line of sight turns, which was 31% off on the baseline: even the true range and range-rate scored 6×10⁷ against 4 at the truth. | `iod.RangeSearch` scores a coarser grid (20,000 km × 100 m/s, 4.7× fewer points). It then polishes the best points from 12 different valleys of the cost: a least-squares fit of all six state components to the angles, in the same CR3BP, with the exact Jacobian from a new state-transition-matrix propagator (`dynamics.propagate_cr3bp_stm`). It keeps polishing, up to 20 starts, until one fits at the 3σ level. | IOD within 10,000 km on the orbit study: 100/100 (was 40/100). Median IOD error 16–77 km (was 1,700–29,000 km). |
| Collocation fails on the small L1 halo, the NRHO and the DRO (0, 0 and 2 of 20). | Product order switched to range search first, collocation as fallback (`scenario.py`, `scenarios/*/config.yaml`). | `product_rs` (the new order) converged on every orbit case without using its fallback, and on short arcs matched or beat collocation-first in every arc and noise group (tables below). |
| On 1-day and 6-hour arcs, wrong fits passed the 3σ check (14 of 60 runs) and nothing flagged them. | Range search records other, clearly different orbits (>1,000 km apart, inside the searched distances) that also fit at the 3σ level, under `info.alternatives`, and `summary.json` gets a plain-language warning. | All 14 wrong-but-passing fits flagged, and in all 14 the true orbit is one of the listed alternatives. No flags on 120 runs on 8-day arcs. |
| Short arcs reported no uncertainty: Orekit's singularity threshold of 1e-12 was too strict for our 1 m position scaling. The normal matrix was actually fine (condition number about 300–20,000). | Threshold 1e-16 (`estimation.py`). New `warnings` for short arcs and for arcs under half a day. | Covariance is now available on all 1-day fits. For fits in the right valley, error/σ has median 0.64 and 95th percentile 2.9 (1 day), and is at most 2.1 on every 8-day orbit case. On 6-hour arcs it is still unreliable (up to 35×), and the warning says so. |
| Below ~0.5″ the 3σ check rejected correct fits. Orekit's angle model includes light time and the generator's angles did not (a 0.6–0.8″ mismatch). | The synthetic generator includes light time (`simulation.observations.light_time`, default on). | At 0.1″ a truth-started fit drops from RMS 1.97σ to 0.28σ, and its error from 0.89 to 0.23 km. At 1″ the 8-day error drops from 2.4 to 1.9 km (5 obs) and from 1.8 to 1.4 km (9 obs), for every method including the truth-started reference. |

### Fixes from an independent code review of the first round of fixes

1. **Mesh check.** It crashed `settings_sensitivity.py`, blocked any arc over 8 days with the default
   80 nodes, and blocked collocation as a fallback. The mesh is now sized automatically (about 10
   nodes per day) and noted in `summary.json` under `notes`; nothing is rejected.
2. **Residual check on old configs.** It was on for legacy stage configs, which broke documented
   setups. It is now on (3σ) for YAML product configs only, and off for legacy configs.
3. **Java errors skipped the fallback.** An Orekit exception, such as a hyperbolic start written as
   equinoctial elements, or any other error inside a method, now counts as that method failing,
   and the next method runs. Only configuration errors stop the run.
4. **Epochs in the benchmark.** The two-body methods in the benchmark were graded at a different
   epoch than the reference on the 9-observation arc. They now use the first, epoch and last
   observations.
5. **Benchmark bookkeeping.**
   - "IOD ran" now counts a product run whose IOD ran even though its fit was rejected.
   - Product timing is reported as IOD plus least squares.
   - Each study writes to its own default output folder.
6. **A fallback repeating the primary method** is rejected in legacy configs.
7. **Container provenance.** It recorded the default image name whatever `-t` said.
   `scripts/run_container.py --build` now passes `IMAGE_TAG`, and the Dockerfile and README say to.

## Results before and after

"Before" is the code on `collocation-iod-testing`: collocation first, range search as fallback,
the grid-only range search, and no light time. Its arc-study product row is derived from the
earlier per-method runs (collocation's fit if it passes 3σ, otherwise range search's).

"Converged" means within 10 km of the fit batch least squares reaches from the true state; on short
arcs that fit is itself hundreds to thousands of km from the truth. "Wrong" means the fit passed the
3σ check but is not that fit.

### Orbit study (8 days, 5 observations, 10″, 20 draws)

| Orbit | Collocation | Range search, before → after | Product, before → after | Median final error (km) |
|---|---|---|---|---|
| L1 halo (baseline) | 20/20 | 20/20 → 20/20 | 20/20 → 20/20 | 16.5 |
| L1 halo, small | 0/20 | 20/20 → 20/20 | 20/20 → 20/20 | 17.3 |
| L2 halo, large | 20/20 | 20/20 → 20/20 | 20/20 → 20/20 | 26.1 |
| NRHO 9:2 | 0/20 | 20/20 → 20/20 | 20/20 → 20/20 | 44.5 |
| DRO | 2/20 | 0/20 → 20/20 | 2/20 → 20/20 | 75.0 |
| **All** | 42/100 | 80/100 → 100/100 | 82/100 → **100/100** | |

Whenever a method converges, its final error equals the truth-started fit's. What is left is set
by the noise and the geometry, not by the method.

### Arc study (baseline halo, 10 draws per noise level)

| Arc | Product before: converged / wrong | Product after: converged / wrong (all flagged) | Truth-started fit, median error |
|---|---|---|---|
| 8 days, 5 obs (1/10/100″) | 30/30, 0 wrong | 30/30, 0 wrong | 1.9 / 19.2 / 194 km |
| 8 days, 9 obs (1/10/100″) | 30/30, 0 wrong | 30/30, 0 wrong | 1.4 / 14.4 / 144 km |
| 1 day, 5 obs (1/10/100″) | 18/30, 11 wrong | 24/30, 6 wrong | 140 / 1,347 / 6,538 km |
| 6 hours, 5 obs (1/10/100″) | 19/30, 11 wrong | 22/30, 8 wrong | 1,780 / 3,912 / 64,248 km |

On short arcs the remaining "wrong" fits are the data's ambiguity, not a method's mistake.
In those runs the angles fit two orbits about 10,000–20,000 km apart within the noise. The truth is
sometimes the runner-up. No method that only has these angles can choose correctly. The stack now
says so and lists the other orbit. In 12 further short-arc runs it chose the right orbit and still
flagged the second, which is correct: the data really do allow both.

Run time per case (IOD plus least squares, 4 runs in parallel on 4 cores):
- Range search first: about 10–12 s on most cases and 28 s on the small L1 halo.
- Collocation first: 2–8 s when collocation succeeds, 17–35 s when it has to fall back.
- The old grid-only range search took 12–17 s on its own.

## How much to trust these numbers

- **Settings were chosen on the reported cases.** These were chosen by looking at some of the same
  noise draws that the tables then report:
  - 12 polished starts, from the 1-day draws, seeds 1–8;
  - the 1,000 km "different orbit" distance;
  - the 20,000 km × 100 m/s grid, from the DRO draws, seeds 0–2.

  They were not tuned to maximize the tables, and the 8-day results do not depend on them. But the
  short-arc detection rate has not been tested on draws held out from that choice.
- **Settings that the numbers rest on:** the 3σ level for "fits as well" (the same as the residual
  check), and the polish limit of 40 evaluations. The limit only saves time: checked on 14 cases,
  the IOD errors were unchanged.
- **Light time.** The before/after orbit study ran without light time, and the arc study and
  ambiguity study ran with it. At 10″ the 0.7″ light-time effect does not change the outcome.
- **What was not tested:** maneuvering targets, observers other than Earth's center, real data, and
  other orbit families or arcs.
- **One machine.** All runs were on one Linux machine; the product container reproduced the native
  result to the last digit there.

## What is still limited, and why

- **Short arcs from Earth's center are fundamentally ambiguous.** Two orbits about 11,000 km apart
  often fit a 1-day arc equally well. Only more information separates them: a longer arc, more
  observations, or a second observer (parallax). BLS still requires an observer at Earth's center;
  supporting ground or space observers is the next step that would actually help here.
- **6-hour arcs: the reported uncertainty is not trustworthy** (error up to 35σ, covariance missing
  in 9 of 30). The run says so.
- **Collocation's answer is not a smooth function of the data.** A last-digit change in the angles,
  which a CSV round trip causes, moved it by 600 m (against an error of about 16 km). The polished
  range search did not move. This is collocation's flat optimum, documented and now tested with a
  tolerance. It is one more reason it is the fallback rather than the first method.
- **Collocation as a fallback uses default options.** With `range_search` first, a YAML config cannot
  set collocation options, because `options` belongs to the first method. Defaults work in every
  benchmark case, and the mesh is sized automatically.
- **Accuracy on 8-day arcs is set by the noise.** Every method that converges ends at the
  truth-started fit (about 1.9 km per arcsecond of noise on the baseline), so no IOD change can
  improve it.

## How to reproduce

As in RESULTS.md, from the repository root with both packages installed and `OREKIT_DATA` set:

```
python benchmarks/collocation_iod/compare_iod.py --study orbits --workers 4                  # ~25 min
python benchmarks/collocation_iod/compare_iod.py --study arcs --methods collocation range_search product product_rs --workers 4   # ~35 min
python -m pytest -q packages/cislunar-stack/tests packages/collocation-iod/tests        # 87 tests, ~2 min
```

`product` is collocation first, and `product_rs` is range search first (the new default). Outputs go
to `outputs/benchmarks/`, which is not committed. The tables above came from those CSVs.

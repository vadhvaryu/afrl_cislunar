# Convergence study: where and why batch least squares stops converging

Owner: Zohar (with Mack). Code: `od_stack/convergence.py` (runs), `od_stack/convergence_features.py` (per-run
variables), `od_stack/convergence_plots.py` (figures, statistics), `od_stack/parallel.py` (resumable runner),
`od_stack/estimator_compare.py` (estimator variants on identical cases). Tests: `tests/test_convergence.py`,
`tests/test_estimators_cr3bp.py`.

```
python -m od_stack.convergence all                                   # A2 grid + robustness + A3 bisections (~2.5 h, 9 workers)
python -m od_stack.convergence all --estimation orekit_bls_robust    # same cases, robust estimator
python -m od_stack.convergence_plots --out results/convergence/orekit_bls
python -m od_stack.estimator_compare settings | a7                    # BLS settings test | estimator comparison
```

## a. Input

| What | Value |
|---|---|
| Truth | `data/synthetic_truth.csv` (L1 northern halo), via the `truth_plus_error` stand-in IOD |
| Observations | 5 RA/Dec pairs over 8 days from Earth's center, 10″; **one fixed noise draw** (obs seed 10,000) for every case, so only the starting error changes |
| Starting error | size d and direction id k (`default_rng(k)`: first draw = position direction, second = velocity direction; ids 0-19 are the old sweep's directions) |
| Modes | `combined` (d km, 0.01·d m/s), `position_only` (d km, 0), `velocity_only` (0, w m/s) |
| A2 grid | 36 position sizes (1,000-100,000 km, every 500 km over 10,000-15,000), 15 velocity sizes (10-1,000 m/s), directions 0-49; robustness: directions 0-19 × obs seeds 10,000-10,004 at 15,000 and 30,000 km |
| A3 | directions 1,000-1,199: bisect the largest converging d to 250 km (combined and position-only) |

## b. Inner working

Each case runs the configured stack (stand-in IOD → Orekit BLS → Orekit CR3BP propagator) and records:
outcome (converged / iteration limit / wrong orbit / other), final error, iterations, residual RMS, runtime,
and the candidate explanations from `convergence_features.py`: periselene of the guess, the truth and the
final state over 8 and 12 days; position error along/across the line of sight; RIC and Moon-radial
components; velocity error along the velocity; Jacobi constant change; impact and L1-region flags; starting
residuals. For failed runs the final state is Orekit's last iterate.

**Converged** = BLS finished, landed within 1,000 km of the truth, **and** its residual RMS is at most 3σ
(the residual gate; `outcome_ungated` keeps the plain category).

Statistics: convergence rate with 95% Wilson intervals; AUC of each variable for predicting failure
(Mann-Whitney); logistic regression of failure on standardized variables with 200 bootstrap resamples.

## c. Output

`results/convergence/<estimator>/`: `grid.csv`, `thresholds.csv`, `thresholds_runs.csv` (one row per run,
all variables), `config_used.yaml`, `analysis.json`, `SUMMARY_tables.md` and the figures
`convergence_curve.png`, `periselene_by_error.png`, `convergence_vs_periselene.png`,
`convergence_vs_features.png`, `non_monotonic_directions.png`, `direction_map.png`, `threshold_hist.png`,
`threshold_vs_direction.png`.

## d. Additional dependencies

None beyond `requirements.txt`. Orekit runs inside the worker processes only (spawned, one JVM each).

---

## Results (Orekit BLS, 4,510 grid runs; A3 bisections pending)

| Starting error | Combined | Position only | Velocity only |
|---|---|---|---|
| ≤ 10,000 km (≤ 100 m/s) | 50/50 at every size | 50/50 | 50/50 |
| 15,000 km | 45/50 | 46/50 | 50/50 at 150 m/s |
| 30,000 km | 43/50 | 46/50 | 45/50 at 300 m/s |
| 50,000 km | 38/50 | 50/50 | 42/50 at 500 m/s |
| 70,000 km | 25/50 | 48/50 | — |
| 100,000 km | 11/50 | 35/50 | 43/50 at 1,000 m/s |

**1. Where does convergence fall off?** Every direction converges up to 10,000 km (combined), 12,000 km
(position only) and 150 m/s (velocity only). The first failure is at 10,500 km. Between 15,000 and 40,000 km
85-90% of directions converge; past 50,000 km the rate drops steeply (22% at 100,000 km).
_A3 threshold percentiles: pending._

**2. Position or velocity?** At large errors, mostly the velocity: at 50,000 km, position alone converges
50/50, adding the matching 500 m/s drops it to 38/50. Below 40,000 km both contribute about equally.

**3. Does periselene explain the failures?** No. On its own it barely predicts failure (AUC 0.58), and with the
other variables in the regression its coefficient is zero (95% interval −0.34 to +0.22). Converged and failed
starts overlap in periselene at every starting error (`periselene_by_error.png`).

**4. What does?** The direction of the error relative to the Moon: errors pointing toward the Moon converge
~62% of the time, ~90% otherwise (strongest regression coefficient, −1.16). The Jacobi constant change
predicts failure about as well as the error size (AUC 0.82 each). The range along the line of sight is not
the culprit: errors along the line of sight fail slightly less often than errors across it (coefficient −0.25).

**5. Recommended IOD accuracy target for David and Mack:** within **10,000 km and 100 m/s** of the truth (every
direction converged there in every mode). _10th-percentile threshold from A3: pending._

**6. Stall or bad minimum?** About half each: 146 iteration-limit and 140 wrong-orbit failures (combined).
Every wrong orbit had residuals far above 3σ (largest good fit: 0.94σ), so the gate caught all 176 wrong
orbits and rejected no good fit. _Rescue rate with restarts (orekit_bls_robust): pending (round 2)._

Other measured facts:
- **Noise draw:** 0 of 40 (direction, size) cases change outcome across 5 noise draws: failure depends on the
  direction, not the noise.
- **Not a single cliff:** 13 of 50 directions fail at one size and converge at a larger one
  (`non_monotonic_directions.png`).
- **Orekit settings don't matter:** position scale 1 m / 1 km / 100 km × threshold 1e-3 / 1e-5 give the same
  8/10 at 15,000 and 30,000 km (directions 0-9).
- **Light time:** Orekit's AngularRaDec includes light time (~1.2 s); the generator does not. For the true
  orbit the two differ by 0.5-0.8″, which makes the reported uncertainty too small below ~3″ (NEES 34.8 at
  0.3″). Fix in the generator (David).

## Estimators without batch least squares (A7)

_Pending (round 2): convergence of none / orekit_bls / orekit_bls_robust / multiple_shooting /
multiple_shooting_then_bls / ukf on 50 directions × 7 sizes, plus NEES vs noise._

Spot checks so far: multiple shooting converges from both 100,000 km cases plain BLS fails, lands 0.77 km from
Orekit BLS's answer, and its covariance matches Orekit's to 6e-5; the UKF works from 1,000 km at 10″ but not
from 15,000 km, and is overconfident below ~3″.

## Other orbits

_Pending: six-orbit study (`od_stack/orbit_study.py`)._

## Open items

- Ground-station observer (David: observer columns; stack: one GroundStation per site, real epoch). Range
  becomes observable through Earth's rotation (~1° of parallax per day vs 10″ noise); rerun A3 after.
- Light time in the generator (above).

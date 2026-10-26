# Convergence study: where and why batch least squares stops converging

Owner: Zohar (with Mack). Code: `od_stack/convergence.py` (runs), `od_stack/convergence_features.py` (per-run
variables), `od_stack/convergence_plots.py` (figures, statistics), `od_stack/parallel.py` (resumable runner),
`od_stack/estimator_compare.py` (estimator variants on identical cases). Tests: `tests/test_convergence.py`,
`tests/test_estimators_cr3bp.py`.

```
python -m od_stack.convergence all                                   # A2 grid + robustness + A3 bisections (~2.5 h, 9 workers)
python -m od_stack.convergence all --estimation orekit_bls_robust    # same cases, robust estimator
python -m od_stack.convergence_plots --out results/convergence/orekit_bls
python -m od_stack.convergence_plots --out results/convergence \
    --compare results/convergence/orekit_bls results/convergence/orekit_bls_robust   # plain vs robust figure
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

Figures shown at the Oct 30 meeting: `figures/2026-10-30/` (convergence_curve, periselene_by_error,
convergence_vs_features, non_monotonic_directions, threshold_hist, orbits_convergence, plain_vs_robust_bls,
a7_convergence, a7_nees, a7_runtime).

## d. Additional dependencies

None beyond `requirements.txt`. Orekit runs inside the worker processes only (spawned, one JVM each).

---

## Results (Orekit BLS: 4,510 grid runs + 2,582 bisection runs over 400 directions)

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

A3 bisection thresholds (200 new directions per mode, bracket ≤ 250 km, 8.3 runs per direction on average):

| Mode | Median | 10th percentile | 90th percentile | Lowest | Never failed (≥ 100,000 km) |
|---|---|---|---|---|---|
| Combined | 73,745 km | 43,557 km | ≥ 100,000 km | 9,453 km | 60 of 200 |
| Position only | ≥ 100,000 km | 76,120 km | ≥ 100,000 km | 10,195 km | 142 of 200 |

**These bisection thresholds are biased high and should not be used as the target.** Bisection tests 5,000 km,
then 100,000 km, then the midpoint; when a direction converges at 52,500 km it searches upward and never sees the
15,000-40,000 km band in which the grid shows 10-15% of directions failing. 13 of the 50 grid directions are
non-monotonic in exactly this way. The lowest bisection threshold (9,453 km) does agree with the grid's first
failure (10,500 km).

**2. Position or velocity?** At large errors, mostly the velocity: at 50,000 km, position alone converges
50/50, adding the matching 500 m/s drops it to 38/50. Below 40,000 km both contribute about equally.

**3. Does periselene explain the failures?** No. On its own it barely predicts failure (AUC 0.58), and with the
other variables in the regression its coefficient is zero (95% interval −0.34 to +0.22). Converged and failed
starts overlap in periselene at every starting error (`periselene_by_error.png`).

**4. What does?** The size of the error first, then its direction relative to the Moon. Errors pointing toward
the Moon converge ~62% of the time, ~90% otherwise. The Jacobi constant change predicts failure about as well as
the error size (AUC 0.82 each), but mostly because it grows with the size. The range along the line of sight is not
the culprit. Logistic regression of failure (standardized coefficients, 95% bootstrap intervals):

| Variable | Grid only, d ≥ 10,000 km (n = 1,650) | Grid + bisection runs (n = 3,106) |
|---|---|---|
| log error size | +0.94 (+0.71 to +1.24) | **+1.11** (+0.95 to +1.30) |
| Moon-radial direction (−1 = toward the Moon) | **−1.16** (−1.44 to −0.92) | −0.39 (−0.53 to −0.26) |
| cross-track fraction | +0.26 (+0.05 to +0.49) | +0.41 (+0.29 to +0.56) |
| Jacobi constant change | +0.80 (+0.54 to +1.19) | +0.23 (+0.13 to +0.33) |
| velocity error along v | +0.23 (+0.08 to +0.40) | +0.16 (+0.06 to +0.26) |
| line-of-sight fraction | −0.25 (−0.41 to −0.09) | −0.02 (−0.10 to +0.09), none |
| periselene of the guess | −0.02 (−0.34 to +0.22), none | +0.12 (−0.03 to +0.25), none |

The bisection runs cluster near each direction's threshold, which shifts the weights; both fits agree that size and
a Moon-ward direction matter and that periselene and the line of sight do not.

**5. Recommended IOD accuracy target for David and Mack:** within **10,000 km and 100 m/s** of the truth. Every
direction converged there in every mode, and it matches the lowest bisection threshold (9,453 km). The brief's
"10th percentile of the A3 threshold" (43,557 km) is not safe because of the bisection bias above. The target also
holds for 5 of the 6 other orbits tested (below), but not for the planar L1 Lyapunov orbit.
**With `orekit_bls_robust` the target can be relaxed to 30,000 km and 300 m/s**: every direction converged up to
there in all three modes on the full robust grid (below). Caveat as in point 6: same 50 directions the robust
estimator was designed on.

**6. Stall or bad minimum?** About half each: 146 iteration-limit and 140 wrong-orbit failures (combined).
Every wrong orbit had residuals far above 3σ (largest good fit: 0.94σ): over the grid and the bisections the
gate caught all 1,016 wrong orbits and rejected no good fit.

**Rescue rate with `orekit_bls_robust`** (round 2: the full A2 grid rerun with the robust estimator, 4,510 runs on
the Mac; `figures/2026-10-30/plain_vs_robust_bls.png`). On the 4,350 grid cases it converges in **266 of the 429
cases plain BLS fails** and fails none of those plain BLS converges; where both converge they reach the same
solution (within 0.1 m). Every direction converges up to:

| Mode | Plain BLS | Robust BLS |
|---|---|---|
| Combined (d km, 0.01·d m/s) | 10,000 km | **40,000 km** (45/50 at 50,000, 18/50 at 100,000) |
| Position only | 12,000 km | **30,000 km** (35/50 at 100,000, same as plain) |
| Velocity only | 150 m/s | **300 m/s** (48–50/50 up to 1,000 m/s) |

- **It removes the stalls.** Plain BLS failed 280 times at the iteration limit; robust BLS never does. All 163 of its
  failures (over all 4,510 runs) end with no fit passing the 3σ gate, so it never returns a wrong orbit. Arc continuation (fit
  the first 3 observations, days 0–4, then all 5) does almost all of the work: 4,053 of its 4,347 converged fits;
  plain BLS 199; the ±10,000/±20,000 km line-of-sight restarts 95.
- **It helps the velocity, not the far position errors.** Past 40,000 km position-only it converges exactly the
  same cases as plain BLS; the gains are in combined and velocity-only errors.
- **Fewer non-monotonic directions:** 8 of 50 (combined) against 13.
- **Cost:** median 5.6 s per fit against 4.9 s, but a mean of 15.5 s against 7.0 s, because a failing case tries
  every restart (up to ~7 minutes). The bisection (A3) was stopped after 7 of 400 directions (~8 hours projected);
  it is not summarized.
- **What predicts its failures** (regression, combined, d ≥ 10,000 km): the size of the error (β +2.54), the
  cross-track part of the error (+1.63) and a Moon-ward direction (−0.88). Periselene of the guess now has a small
  effect (+0.81, interval +0.31 to +1.44), likely because the remaining failures are all at ≥ 50,000 km, where
  large errors also move the periselene; it is still not the main variable.
- Caveat: the continuation was designed after seeing the plain-BLS failures on these same 50 directions, so this is
  not an independent test. The 7 finished bisection directions (new ones) all converge up to at least 45,635 km,
  consistent with the grid, but are too few to be a check.

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

`python -m od_stack.estimator_compare a7` (2,460 runs, run on the Mac). Same truth orbit and 5 observations;
combined mode (velocity error 0.01 m/s per km); 50 directions × 7 starting errors at one 10″ noise draw, plus a
noise sweep (1/10/100″, 20 seeds, start 1,000 km). All pipelines are graded the same way: converged = within
1,000 km of the truth (scaled with noise above 10″) **and** residual RMS ≤ 3σ. Figures:
`figures/2026-10-30/a7_convergence.png`, `a7_nees.png`, `a7_runtime.png`.

Converged out of 50:

| Pipeline | 1k | 10k | 15k | 20k | 30k | 50k | 100k km | Median s/run |
|---|---|---|---|---|---|---|---|---|
| `none` (use the IOD as is) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.2 |
| `orekit_bls` | 50 | 50 | 45 | 45 | 43 | 38 | 11 | 3–12 |
| `orekit_bls_robust` | 50 | 50 | **50** | **50** | **50** | **45** | 18 | 3–9 (95 at 100k) |
| `multiple_shooting` (CR3BP, no Orekit) | 50 | 50 | 47 | 43 | 42 | 40 | **33** | 0.8–1.8 |
| `multiple_shooting_then_bls` | 50 | 50 | 47 | 43 | 42 | 40 | **33** | 2–3 |
| `ukf` (+ RTS smoother) | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1–0.5 |
| robust, else multiple shooting → BLS | 50 | 50 | 50 | 50 | 50 | 49 | 38 | |

The last row is not a run pipeline: it combines the two rows above per direction.

Noise sweep (start 1,000 km; NEES expect 6):

| Pipeline | 1″: median error, NEES, inside 95% | 10″ | 100″ |
|---|---|---|---|
| all BLS pipelines (identical) | 3.0 km, 9.7, 15/20 | 29 km, 6.6, 18/20 | 293 km, 6.6, 19/20 |
| `multiple_shooting` | 2.9 km, **6.5, 19/20** | 29 km, 6.5, 19/20 | 292 km, 6.6, 19/20 |
| `ukf` | 5 of 20 pass the gate (2.0 km, 4.1) | 28 km, 7.8, 16/20 | 279 km, 6.6, 19/20 |
| `none` | 1,000 km by construction; 7,212 km max error over 8 days; ~150σ residual at 10″ | | |

What this shows:

1. **Use an estimator: the IOD alone is not usable.** A 1,000 km start drifts to a median 7,212 km error over the
   8 days, with residuals near 150σ at 10″. (Graded naively, `none` "converged" 22 of 50 times at 1,000 km because its
   error sat on the 1,000 km threshold; it is now graded by its starting residual like every other fit.)
2. **Keep batch least squares and make `orekit_bls_robust` the default.** It converges in every direction up to
   30,000 km (plain BLS: 43–45 of 50 from 15,000–30,000 km), never loses a case plain BLS wins, and costs about the
   same (3–9 s) except from 100,000 km. All BLS pipelines land on the same solution (the default-seed case is
   5.891 km from the truth in every one of them).
3. **Multiple shooting reaches furthest** (33 of 50 from 100,000 km, against 18) and is the fastest, but loses
   3–8 cases from 15,000–30,000 km that the robust BLS gets. All 45 of its failures are iterates that pass
   through the Earth or the Moon (now stopped at impact). The two are complementary: robust BLS, falling back to
   multiple shooting → BLS when the gate rejects everything, would converge 49 of 50 at 50,000 km and 38 of 50 at
   100,000 km. Multiple shooting's advantage is specific to this halo (see Other orbits: it fails on the NRHO).
4. **Multiple shooting's honest uncertainty at 1″ confirms the light-time explanation.** Its observation model
   is the CR3BP's, which, like the generator, has no light time: NEES 6.5 and 19 of 20 truths inside the 95%
   region at 1″, against 9.7 and 15 of 20 for Orekit, whose angles include light time. On real data this
   advantage disappears (both should model light time), so it is not a reason to drop BLS.
5. **The UKF only works from a good start.** From 1,000 km at 10″ it converges every time, but from 10,000 km
   and beyond never: 117 runs end with a sigma point on an impact trajectory, the rest on fits 140 km or more off
   (median 450 km or more; residuals 12σ and up), all caught by the gate. At 1″ it stops short of the least-squares minimum (15 of 20 fits at
   ~7σ, the linearization floor already seen in the tests). Not a replacement for BLS at IOD-sized errors.
6. **Multiple shooting followed by BLS adds nothing over multiple shooting alone on this data** except
   Orekit's model (same answer as plain BLS, same convergence as multiple shooting).

Answer to "should we avoid batch least squares?": no. Every converged pipeline agrees with BLS; the robust
version is the best single estimator up to 50,000 km; multiple shooting is worth keeping as a fallback for very
poor starts on halo-like orbits.

## Other orbits (`od_stack/orbit_catalog.py`, `od_stack/orbit_study.py`)

Six orbits made periodic in our CR3BP (seeds from the MIT-licensed cislunar-custody catalogue, re-corrected
with our μ), observed with the same recipe; 20 directions × 7 starting errors, combined mode, 10″, 3σ gate.
Converged out of 20 (`results/orbit_study/orbits_convergence.png`):

| Orbit (stability index, perilune) | Estimator | 1k | 5k | 10k | 20k | 30k | 50k | 100k km |
|---|---|---|---|---|---|---|---|---|
| L1 halo, current truth (1, 18,749 km) | BLS | 20 | 20 | 20 | 17 | 16 | 15 | 5 |
| | multiple shooting | 20 | 20 | 20 | 19 | 18 | 20 | **18** |
| Small L1 halo (615, 48,009 km) | BLS | 20 | 20 | 20 | 17 | 15 | 4 | 3 |
| | multiple shooting | 20 | 19 | 18 | 19 | 15 | 10 | 3 |
| L1 Lyapunov, planar (933, 45,258 km) | BLS | 20 | **8** | **5** | 10 | 9 | 1 | 2 |
| | multiple shooting | 20 | 16 | 12 | 12 | 10 | 3 | 0 |
| Large L2 halo (37, 32,088 km) | BLS | 20 | 20 | 19 | 17 | 16 | 18 | 18 |
| | multiple shooting | 20 | 20 | 19 | 16 | 16 | 14 | 13 |
| NRHO 9:2 (1.3, 3,249 km) | BLS | 20 | 20 | 19 | 12 | 3 | 1 | 6 |
| | multiple shooting | **14** | 1 | 3 | 1 | 0 | 0 | 0 |
| DRO (1.0, 68,758 km) | BLS | 20 | 20 | 19 | 17 | 13 | 12 | 10 |
| | multiple shooting | 20 | 20 | 19 | 14 | 13 | 12 | 8 |

- **The 10,000 km target holds for BLS on 5 of 6 orbits** (≥ 19/20 at 10,000 km). The exception is the planar
  L1 Lyapunov orbit: BLS fails 12 of 20 times already from 5,000 km, always onto one of two wrong solutions
  3,844 km and 5,568 km from the truth (residuals 46σ and 5.1σ: the second is only just caught by a 3σ gate).
- **Orbit stability does not predict the basin.** The stable DRO falls off past 10,000 km like the others, while
  the unstable large L2 halo (stability 37) converges 18/20 from 100,000 km. (We expected stable orbits to be
  easier; they are not.)
- **Multiple shooting's big win is specific to our halo** (18/20 vs 5/20 from 100,000 km). It helps on the
  Lyapunov orbit at 5,000–10,000 km, is no better on the small L1 halo and the DRO, worse on the large L2 halo
  at large errors, and **fails on the NRHO even from 1,000 km** (14/20). That is a limitation of our
  implementation, not of the idea: 4 equal segments put the fast perilune passage inside one segment, many
  iterates hit the Moon (now stopped at impact), and the continuity tolerance (1e-8 DU) is too strict for that
  sensitivity. Segment boundaries around perilune would be the fix to try.

## Open items

- Ground-station observer (David: observer columns; stack: one GroundStation per site, real epoch). Range
  becomes observable through Earth's rotation (~1° of parallax per day vs 10″ noise); rerun A3 after.
- Light time in the generator (above).

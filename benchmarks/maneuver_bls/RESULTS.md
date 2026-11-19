# Batch least squares with burn parameters

November 19, 2026, branch `maneuver-bls`. The agenda item was "Batch Least Squares with Parameter
Implementation". Collocation's `impulsive` mode can find engine burns. Least squares assumed the
spacecraft coasts, so until now that combination was rejected. This adds an estimator that fits the burns.

**Bottom line.** For burns of 5 and 20 m/s, a coasting fit fails the 3σ residual check every time.

| Approach | Worst error over the 8-day arc |
|---|---|
| Collocation alone | 94–480 km |
| Collocation + least squares with burns | **12–13 km at 1″, 123–134 km at 10″** |

The fitted burns are accurate:
- **Velocity change:** within 0.06 m/s at 1″ and 0.6 m/s at 10″.
- **Time:** within an hour.
- **Uncertainty:** the reported value is honest; median error is 0.7–0.9σ.

A 1 m/s burn is recovered at 1″ but is lost in the noise at 10″: a coasting orbit fits those data within 3σ.

## What was added

- **Burns in the synthetic truth:** `simulation.truth.maneuvers`, each `{time_days, delta_v_m_s}` in EMI
  axes.
- **Propagation through impulsive burns** (`dynamics.propagate_with_burns`). Its exact derivatives with
  respect to the starting state, each burn's velocity change and each burn's time match finite differences
  to 1e-7 or better.
- **Collocation's adapter** passes on the impulses it estimates (`info.impulses`, EMI m/s).
- **A new estimator, `cr3bp_bls`.**
  - It is batch least squares in the CR3BP itself (scipy, no Java), on the angles, with light time.
  - It groups collocation's impulses into burns: neighbors within half a day join one burn, and small ones
    are dropped.
  - It fits the state at the epoch together with each burn's velocity change and, by default, its time.
  - It reports a covariance for the state and each burn, and applies the same 3σ residual check as
    Orekit's.
  - With `maneuvers: none` it is a plain coasting fit. It accepts any observer position.
- **The scipy prediction applies the fitted burns.** With `estimation: none`, it also applies collocation's
  own impulses.
- **Validation:**
  - Orekit's prediction does not apply burns yet, so that combination is rejected.
  - Collocation's `hybrid` mode (continuous thrust) cannot be fitted this way, so it is rejected with
    `cr3bp_bls`.
- **Example:** `scenarios/maneuver/config.yaml`.
- **Tests:** 15 in `packages/cislunar-stack/tests/test_burn_estimation.py`.

## Study (`burn_study.py`)

The setup:
- **Orbit:** the baseline halo, with one impulsive burn at day 3.1 in a fixed direction.
- **Observations:** every half day for 8 days from Earth's center (17).
- **Sweep:** burns of 1, 5 and 20 m/s, at 1″ and 10″ noise, with 10 draws each.
- **Error:** measured against the truth at 200 times over the arc.

The three approaches:
- **coasting:** range search, then `cr3bp_bls` without burns. This is what the stack could do before.
- **collocation:** collocation (`impulsive`) alone. The prediction applies its impulses.
- **burn_bls:** collocation (`impulsive`), then `cr3bp_bls` with burns.

| Burn | Noise | Approach | Passed 3σ | Error at epoch | Worst error over the arc | Burn Δv error | Burn time error | Δv error / σ |
|---|---|---|---|---|---|---|---|---|
| 1 m/s | 1″ | coasting | 0/10 | — | — | — | — | — |
| | | collocation | 10/10 | 3.7 km | 53 km | 0.25 m/s | 1.7 h | — |
| | | **burn_bls** | 10/10 | 3.9 km | **12 km** | **0.05 m/s** | 0.4 h | 0.71 |
| 1 m/s | 10″ | coasting | 10/10 | 22 km | 324 km | — | — | — |
| | | collocation | 10/10 | 23 km | 271 km | 0.86 m/s | 11 h | — |
| | | burn_bls | 10/10 | 22 km | 322 km | 0.99 m/s | 4.2 h | 1.29 |
| 5 m/s | 1″ | coasting | 0/10 | — | — | — | — | — |
| | | collocation | 10/10 | 44 km | 94 km | 0.96 m/s | 1.6 h | — |
| | | **burn_bls** | 10/10 | 4.1 km | **12 km** | **0.06 m/s** | 0.1 h | 0.74 |
| 5 m/s | 10″ | coasting | 0/10 | — | — | — | — | — |
| | | collocation | 10/10 | 45 km | 480 km | 1.85 m/s | 2.0 h | — |
| | | **burn_bls** | 10/10 | 42 km | **123 km** | **0.55 m/s** | 0.7 h | 0.72 |
| 20 m/s | 1″ | coasting | 0/10 | — | — | — | — | — |
| | | collocation | 10/10 | 166 km | 331 km | 2.78 m/s | 1.3 h | — |
| | | **burn_bls** | 10/10 | 5.1 km | **13 km** | **0.06 m/s** | 0.0 h | 0.87 |
| 20 m/s | 10″ | coasting | 0/10 | — | — | — | — | — |
| | | collocation | 10/10 | 148 km | 376 km | 3.98 m/s | 1.7 h | — |
| | | **burn_bls** | 10/10 | 52 km | **134 km** | **0.60 m/s** | 0.2 h | 0.87 |

All values are medians.
- **Collocation alone underestimates burns** because it uses as little velocity change as the noise
  allows (3.2 of a true 4.7 m/s in one 10″ case). Least squares with burn parameters closes that gap.
- **Run time:** about 3 s per run for collocation plus `cr3bp_bls`, and 20 s for the coasting path, most
  of it range search.

## Limits and next steps

- **One burn, one direction, one orbit,** and 17 observations. Each burn adds four unknowns, so with the
  usual 5 observations a burn cannot be fitted. More burns, other orbits and sparser data are untested.
- **The burns must come from collocation.** If collocation misses a burn (as with 1 m/s at 10″), least
  squares does not add one. Testing for an extra burn when the coasting fit's residuals are high would
  be a next step.
- **Only the scipy CR3BP prediction applies burns.** Orekit's batch least squares and prediction still
  assume coasting. Orekit can estimate maneuver parameters (for example thrust and burn start and stop
  times), and that is the path to burns with real ephemerides.
- **`cr3bp_bls` accepts any observer position, but only Earth's center was tested.** That position
  support is also what would help resolve the short-arc ambiguity (`benchmarks/collocation_iod/IMPROVEMENTS.md`).

## How to reproduce

```
python benchmarks/maneuver_bls/burn_study.py --out outputs/benchmarks/maneuver_bls --workers 4      # ~25 min
cislunar-stack --config scenarios/maneuver/config.yaml
```

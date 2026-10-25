# Model mismatch: what the CR3BP costs on real-world-like data

Code: `od_stack/hifi.py`. Tests: `tests/test_hifi.py` (need orekit-data with the DE-440 ephemerides).

```
python -m od_stack.hifi build --orbit L1_halo_truth     # truth + observations -> results/hifi/<orbit>/
python -m od_stack.hifi run --orbit L1_halo_truth --seeds 20 --noise 1 10
```

## a. Input

| What | Value |
|---|---|
| Epoch | 2026-10-26 00:00 UTC (real date; the Moon is 369,489 km away then, 4% closer than the CR3BP's 384,338 km) |
| Frame | real GCRF. The stack's EMI frame is pinned to the real Moon at the epoch: x toward the Moon, z along the Earth–Moon angular momentum (EMI = Mᵀ·GCRF). This answers README open question 2. |
| Truth orbits | the CR3BP orbits from `od_stack/orbit_catalog.py`, mapped into the real Earth–Moon geometry at the epoch (instantaneous Moon distance, rate and dD/dt). That is what a CR3BP-designed orbit does in reality without station-keeping. |
| Observations | 5 RA/Dec pairs over 8 days from Earth's center, Gaussian noise (1″ or 10″), same noise recipe as `obs_generator.py`. The noise-free angles come from Orekit's `AngularRaDec` on the truth, so they include light time, like real data. |
| Starting guess | truth at day 4 plus 1,000 km / 10 m/s in direction id = seed (same directions as the convergence study) |

## b. Inner working

**Truth model** (Orekit numerical propagator, DE440 ephemerides from orekit-data): Earth point mass + J2,
Moon, Sun, Venus, Mars, Jupiter, Saturn, solar radiation pressure on a 1,000 kg spacecraft with 15 m²
(A/m = 0.015 m²/kg) and Cr = 1.3, with Earth and Moon shadows.

**Estimator variants.** All use Orekit's BatchLSEstimator with the same measurement model, so the only
difference between them is the force model:

| Variant | Force model in the estimator |
|---|---|
| `cr3bp_nominal` | the stack's CR3BP (DU 384,338 km, TU 375,099 s) |
| `cr3bp_epoch` | CR3BP scaled to the Moon's actual distance and rate at the epoch |
| `cr3bp_epoch_emp` | `cr3bp_epoch` + 3 estimated constant inertial accelerations (empirical forces) |
| `earth_moon` | Earth + real (DE440) Moon |
| `sun_moon` | Earth + real Moon + Sun |
| `full_no_srp` | everything except radiation pressure (spacecraft unknown) |
| `full_cr_estimated` | everything, reflection coefficient estimated (starts at 1.0; truth 1.3) |
| `full` | exactly the truth model (upper bound) |

**Grading** (GCRF, against the truth): position and velocity error at day 4, max position error over the
8 days (each fit propagated with its own model), post-fit residual RMS and the 3σ gate, reported 1σ, NEES
(expect 6), share of truths inside the 95% region.

## c. Output

`results/hifi/<orbit>/`: `truth_gcrf.csv` (hourly), `obs_gcrf.csv`, `scenario.json` (epoch, Moon geometry,
EMI axes), `runs.csv` (one row per fit), `summary.json`, `summary.md`.

## d. Additional dependencies

orekit-data must include `DE-440-ephemerides`, `Earth-Orientation-Parameters` and `Potential` (all in the
standard orekit-data repository). No new Python packages.

## Results (3 orbits × 8 estimator models × 1″ and 10″ × 20 seeds, start 1,000 km / 10 m/s)

Full tables: `results/hifi/<orbit>/summary.md`. Median error at day 4, and how many of 20 fits pass the 3σ gate:

| Estimator model | L1 halo 10″ | L1 halo 1″ | NRHO 10″ | NRHO 1″ | DRO 10″ | DRO 1″ |
|---|---|---|---|---|---|---|
| `cr3bp_nominal` | 30,331 km, 0 pass | 30,332 km, 0 | 3,103 km, 0 | 3,107 km, 0 | 10,585 km, 0 | 10,589 km, 0 |
| `cr3bp_epoch` | 4,211, 0 | 4,216, 0 | 5,662, 0 | 5,679, 0 | 2,846, 0 | 2,845, 0 |
| `cr3bp_epoch_emp` | 2,245, 0 | 2,245, 0 | **2,926, 20 pass** | **2,619, 20 pass** | 1,240, 12 pass | 901, 18 pass |
| `earth_moon` | 356, 0 | 352, 0 | **651, 17 pass** | 653, 0 | **477, 11 pass** | 470, 0 |
| `sun_moon` | 15.3, 20 | 2.6, 20 | 30.2, 20 | 4.7, 20 | 36.5, 20 | 4.2, 20 |
| `full_no_srp` | 15.3, 20 | 2.6, 20 | 30.2, 20 | 4.7, 20 | 36.5, 20 | 4.1, 20 |
| `full_cr_estimated` | 20.6, 20 | 2.1, 20 | 82.3, 19 | 8.2, 19 | 40.4, 20 | 4.0, 20 |
| `full` (truth model) | 15.6, 20 | 1.6, 20 | 31.9, 19 | 3.2, 19 | 36.4, 20 | 3.6, 20 |

Is the reported uncertainty honest? Mean NEES (expect 6):

| Estimator model | L1 halo 10″ / 1″ | NRHO 10″ / 1″ | DRO 10″ / 1″ |
|---|---|---|---|
| any CR3BP variant, `earth_moon` | 10³–10¹⁰ (meaningless) | 10³–10⁸ | 10²–10⁹ |
| `sun_moon` | 6.7 / **19.9** | 5.9 / **17.4** | 6.7 / **14.3** |
| `full_cr_estimated` | 6.4 / 6.4 | 6.3 / 6.3 | 6.5 / 6.5 |
| `full` | 6.5 / 6.5 | 5.8 / 5.8 | 6.6 / 6.6 |

**What it shows**

1. **The CR3BP is not usable on real-world-like data, even scaled to the real Moon.** Every CR3BP variant is
   off by thousands of km on every orbit while reporting 2–50 km 1σ. The nominal CR3BP is 30,000 km off on the L1
   halo; matching the Moon's actual distance and rate at the epoch helps (4,200 km) but is nowhere near enough.
2. **Earth + real Moon + Sun is the minimum model.** It matches the full model to within a few km at both noise
   levels on all three orbits. Planets and Earth's J2 change nothing over 8 days (`full_no_srp` = `sun_moon`).
3. **Sunlight pressure matters at 1″, not at 10″.** At 1″, leaving it out makes the uncertainty too small (NEES
   14–20, only 2–9 of 20 truths inside the 95% region); at 10″ it doesn't show.
4. **Estimating the reflection coefficient keeps the uncertainty honest on every orbit** (NEES 6.3–6.5), and at
   1″ on the L1 halo it recovers Cr (median 1.29, 10–90% 0.79–2.03; truth 1.3) and beats `sun_moon` (2.1 vs 2.6
   km, max error 6.6 vs 15 km). At 10″, and on the NRHO and DRO even at 1″, Cr is not observable from 5 angle
   pairs (estimates −48 to +32) and the extra freedom costs accuracy (NRHO 82 vs 32 km at 10″). This corrects the
   single-case first look below, which said estimating Cr "backfires": it costs accuracy when the data are weak,
   but it never makes the uncertainty dishonest.
5. **The 3σ residual gate detects a gross model error, but not a moderate one at 10″.** It rejected every CR3BP fit
   on the L1 halo. But `earth_moon` passed it in 17/20 (NRHO) and 11/20 (DRO) runs at 10″ while 480–650 km off;
   at 1″ the same fits all fail it. The gate's power depends on how big the mismatch is compared with the noise.
6. **Empirical accelerations can hide a wrong model from the gate.** CR3BP + 3 estimated constant accelerations
   passes the gate 20/20 on the NRHO while 2,600–2,900 km off (reported 1σ 72–725 km), and 18/20 on the DRO at 1″
   while ~900 km off. The extra parameters absorb the mismatch in the fit, not in the prediction. Use them with a
   physical model, not instead of one.
7. **When the model matches, the full-model pipeline reproduces the CR3BP-world numbers:** on the L1 halo at 10″,
   15.6 km at day 4 and 52 km max over 8 days (the CR3BP baseline on CR3BP data: 15.1 km and 52.9 km).

**Recommendation:** for real data, estimate with Earth + DE440 Moon + Sun as the minimum, add sunlight pressure
with the coefficient estimated once observations are better than a few arcseconds, keep the 3σ residual gate,
and do not rely on the gate alone to catch a model that is only moderately wrong.

### First look (one case: L1 halo, 10″, seed 0), kept for the record

| Estimator model | Error at day 4 | Residual RMS | Note |
|---|---|---|---|
| `cr3bp_nominal` | 30,333 km | 534σ | reports 20 km 1σ; caught by the 3σ gate |
| `cr3bp_epoch` | 4,211 km | 116σ | caught |
| `cr3bp_epoch_emp` | 2,245 km | 60σ | caught; extrapolates badly (26,000 km max error) |
| `earth_moon` | 355 km | 21σ | caught |
| `sun_moon` | 9.8 km | 0.64σ | as good as the full model |
| `full_no_srp` | 9.8 km | 0.64σ | |
| `full_cr_estimated` | 12.9 km | 0.61σ | Cr estimate −3.7 (see point 4: not observable at 10″) |
| `full` | 9.2 km | 0.67σ | upper bound |

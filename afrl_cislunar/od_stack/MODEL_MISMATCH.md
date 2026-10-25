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

## Results

_To be filled in from `results/hifi/*/summary.md`._

### First look (one case: L1 halo, 10″, seed 0)

| Estimator model | Error at day 4 | Residual RMS | Note |
|---|---|---|---|
| `cr3bp_nominal` | 30,333 km | 534σ | reports 20 km 1σ; caught by the 3σ gate |
| `cr3bp_epoch` | 4,211 km | 116σ | caught |
| `cr3bp_epoch_emp` | 2,245 km | 60σ | caught; extrapolates badly (26,000 km max error) |
| `earth_moon` | 355 km | 21σ | caught |
| `sun_moon` | 9.8 km | 0.64σ | as good as the full model |
| `full_no_srp` | 9.8 km | 0.64σ | |
| `full_cr_estimated` | 12.9 km | 0.61σ | Cr estimate −3.7: not observable from 5 angle pairs |
| `full` | 9.2 km | 0.67σ | upper bound |

# od_stack: observations → IOD → Batch Least Squares → propagation → results

A modular pipeline for cislunar orbit determination. Each stage only passes one of
three objects to the next, so any stage can be swapped without touching the others.

```
data source ─Observations─▶ IOD ─StateEstimate─▶ estimation (BLS) ─StateEstimate + covariance─▶ propagator ─Trajectory─▶ results
               (all obs)      (3 obs, rough)       (all obs, refined)                              (any times)
```

Owners (week of 10/12): IOD: David, Mack · Estimation + propagation: Keira, Sanjib · Results/graphs: Zohar

## Setup

Put `od_stack/`, `tests/`, `config.yaml` and `requirements.txt` in the same folder as
`data/` (next to `obs_generator.py`). Then:

```
pip install -r requirements.txt       # orekit-jpype also needs Java (JDK 11+)
export OREKIT_DATA=~/halo-docker/orekit-data   # or set orekit.data_path in config.yaml
python -m pytest -q                   # 8 tests, all should pass
python -m od_stack.run                # writes results/results.png, summary.json, trajectory_comparison.csv
```

On Windows PowerShell use `$env:OREKIT_DATA="C:\path\to\orekit-data"` instead of `export`.
`orekit-data` is needed by the BLS stage (leap seconds for the observer station).

Swap modules from the command line or in `config.yaml`:

```
python -m od_stack.run --estimation none      # skip BLS, propagate the IOD state directly
python -m od_stack.run --propagator cr3bp     # scipy CR3BP instead of Orekit
python -m od_stack.run --iod gooding
python -m od_stack.run --compare-iod          # every IOD method in one table
```

## Conventions (every module uses these)

| | |
|---|---|
| Frame | **EMI**: Earth-centered, non-rotating; x points at the Moon at t0, z is normal to the Earth–Moon orbit plane. Same frame `obs_generator.py` measures RA/Dec in. **Not** Earth's equator. Passed to Orekit labeled as GCRF (consistent, because nothing here depends on Earth's real orientation). |
| Units | SI: m, m/s, s, rad |
| Time | seconds since t0 (first observation); Orekit date = J2000 + t |
| CR3BP ↔ EMI | only through `od_stack/frames.py` |

Data contracts: `od_stack/contracts.py`.

## Modules

| Stage | Name in config | What it is |
|---|---|---|
| Data source | (`data.observations`) | Reads `synthetic_obs.csv`. Uses observer columns `obs_x_km, obs_y_km, obs_z_km` (EMI) if present, otherwise assumes Earth's center. |
| IOD | `truth_plus_error` | **Test stand-in.** True state plus a known error. Reads the answer key, so never use it for real results. Replace with David and Mack's IOD when it works. |
| IOD | `gooding`, `gauss`, `laplace` | Orekit's angles-only methods (two-body). Fail on this data; see findings. |
| Estimation | `orekit_bls` | Orekit `BatchLSEstimator` (Levenberg–Marquardt) on all RA/Dec observations. Estimates the 6 Cartesian state components at the IOD epoch; returns covariance, iterations, RMS. |
| Estimation | `none` | Pass the IOD state straight through. |
| Propagator | `orekit_cr3bp` | Orekit numerical propagator: Earth gravity + a Moon moving on the CR3BP circle. Same physics as the truth (verified to < 10 m over 8 days), and the same model BLS fits with. |
| Propagator | `cr3bp` | The same CR3BP in scipy (independent check). |
| Propagator | `keplerian` | Two-body baseline. |
| Results | — | Error vs truth over time, rotating-frame plots, CSV and JSON. `summary.json` includes the 6×6 covariance. |

### How the BLS stage reproduces the CR3BP inside Orekit (`orekit_models.py`)

- **Forces:** Earth point mass (GM = (1−μ)·DU³/TU²) plus `ThirdBodyAttraction` from a custom
  `CircularMoon` (radius DU, rate 1/TU, GM = μ·DU³/TU²). Orekit's third-body force includes
  the Moon's pull on Earth, which makes this exactly the CR3BP written in an Earth-centered frame.
- **Observer:** Orekit's `AngularRaDec` needs a `GroundStation`. It sits 1 m from Earth's
  center on a non-rotating sphere. Exactly at the center, Orekit returns NaN; 1 m changes
  the angles by ~0.0006″, far below the 10″ noise.
- **Parameters:** Cartesian position/velocity, not Keplerian elements (meaningless out here).

## Adding a module

- **IOD:** class with `name` and `estimate(obs) -> StateEstimate`; add to `IOD_METHODS` in `iod.py`.
- **Estimator:** class with `name` and `refine(obs, x0) -> StateEstimate`; add to `ESTIMATORS` in `estimation.py`.
- **Propagator:** class with `name` and `propagate(x0, t) -> Trajectory`; add to `PROPAGATORS` in `propagators.py`.
- **Data source:** a function that returns `Observations`.

## Findings so far

Default scenario: L1 halo, observer at Earth's center, 5 observations over 8 days, 10″ noise.

**Orekit's built-in IOD fails** (Gooding, Gauss and Laplace), even with noise-free angles.
The plumbing is fine: the same Gooding module recovers a two-body orbit exactly from Earth's
surface (see `tests/`). Causes: no parallax from the geocenter, and the halo orbit is
Moon-dominated, so no two-body orbit fits it.

**Orekit BLS works** when started near the truth:

| | |
|---|---|
| Start 1,000 km / 10 m/s off | 15 km / 0.07 m/s after 6 iterations; max 53 km over the 8 days |
| Reported 1-sigma position | 26 km (the true error is inside it) |

Preliminary sweeps (quick, few seeds; redo properly):

| Starting error (10″ noise, 5 seeds) | Converged | | Noise (start 1,000 km off, 10 seeds) | Median error | Median 1-sigma | Error < 3-sigma |
|---|---|---|---|---|---|---|
| 100 km / 1 m/s | 5/5 | | 1″ | 2 km | 3 km | 10/10 |
| 10,000 km / 100 m/s | 5/5 | | 10″ | 19 km | 26 km | 10/10 |
| 30,000 km / 300 m/s | 3/5 | | 100″ | 190 km | 260 km | 10/10 |
| 100,000 km / 1,000 m/s | 1/5 | | 1,000″ | 1,940 km | covariance fails | — |

So the IOD needs to land within roughly 10,000 km for BLS to be reliable, and the covariance
is trustworthy up to at least 100″ of noise.

## Open questions for the team

1. Add observer position columns to `synthetic_obs.csv` before moving to a ground station.
   `orekit_bls` then needs one `GroundStation` per real observer location.
2. EMI is not a standard frame. Real ephemerides (real Moon, Sun) need a real UTC epoch and
   the EMI → GCRF rotation, defined once in `frames.py`.
3. Why does the covariance become singular at ≥ 1,000″ noise?
4. `time_days` is written with 6 decimals (~0.09 s). Fine for now.

# 4. Results / Data / Graphs

Owner: Zohar. Code: `od_stack/results.py` (single run), `od_stack/uncertainty.py` (covariance math),
`od_stack/sweep.py` (Monte Carlo sweeps). Tests: `tests/test_results.py`.

The last layer of the stack. It never calls Orekit's estimation classes directly: it only reads the
contract objects the earlier layers hand over, so swapping the IOD, estimator or propagator needs no
change here.

## a. Input

From the earlier layers, through `od_stack/contracts.py`:

| From | Object | Fields used |
|---|---|---|
| Data layer | `Observations` | `t`, `ra`, `dec`, `sigma`, `observer` (EMI, SI) |
| IOD | `StateEstimate` (x_iod) | `t`, `r`, `v`, `method`, `info["obs_indices"]` |
| Estimation (BLS) | `StateEstimate` (x0) | `t`, `r`, `v`, `cov` (6×6, SI, optional), `info` (iterations, RMS) |
| Propagator | `Trajectory` (dense) and `Trajectory` at the observation times | `t`, `r`, `v`, `label` |
| Propagator | the propagator object itself | `propagate(x0, t)`, reused to carry the covariance forward |
| Data layer (synthetic only) | truth `Trajectory` at the same times | `r`, `v` |
| `config.yaml` | `output:` and `sweeps:` sections | output folder, dense points, confidence level, sweep grids |

Everything is in the EMI frame, SI units, seconds since the first observation. The truth is
optional: with real data the layer still produces residuals and uncertainty, and simply skips
anything that needs the answer key.

## b. Inner working

**Single run** (`results.report`, called by `python -m od_stack.run`):

1. **Post-fit residuals (no truth needed).** For each observation, the predicted RA/Dec from the
   fitted trajectory is subtracted from the measured RA/Dec (observed − computed), in arcsec and in
   units of the 1σ noise. Their RMS should be about √((n−6)/n) of the noise for n angles and 6
   fitted states (0.63σ for 10 angles). Much larger means the model or the data is wrong.
2. **Errors vs truth (synthetic only).** Position and velocity error at the epoch, at each
   observation time, and over the whole span (max and RMS).
3. **Covariance over time.** The BLS covariance is given at the epoch only. It is carried to every
   output time with 12 symmetric sigma points (x0 ± √6 · columns of √P) pushed through the same
   propagator the stack used. This needs no state-transition-matrix code, so any new propagator
   gets it for free.
4. **Confidence regions (no truth needed).** Semi-axes of the 95% confidence ellipsoid for
   position and velocity (eigenvalues of each 3×3 block, scaled by the χ² quantile).
5. **Is the covariance honest? (synthetic only).** NEES = eᵀP⁻¹e at the epoch (6 states; χ² with 6
   degrees of freedom if the covariance is right) and for position over time (3 degrees of
   freedom), plus the fraction of the span where the true error is inside 3σ.

**Sweeps** (`python -m od_stack.sweep`): many runs with fresh noise (same recipe as
`obs_generator.py`), using whatever IOD / estimator / propagator `config.yaml` selects.

- *Noise sweep:* angle noise 0.3″ to 3,000″, 20 seeds each, IOD error held at 1,000 km / 10 m/s.
- *IOD-error sweep:* starting error 100 km to 100,000 km (velocity error 0.01 m/s per km), 10″ noise.
- A run counts as **converged** when BLS finishes *and* lands within 1,000 km of the truth
  (threshold scaled by noise/10″ above 10″, since a good fit's error grows with the noise).
- Per setting: convergence rate with a 95% Wilson interval, median and interquartile error,
  median reported 1σ, fraction of runs with the truth inside the 95% region, mean NEES (expect 6).

## c. Output

Written to `output.dir` (default `results/`) by a single run:

| File | Contents |
|---|---|
| `summary.json` | all numbers above, the 6×6 covariance, residuals, consistency checks, figure paths |
| `trajectory_comparison.csv` | estimate, truth, error and 1σ position at every dense time |
| `residuals.csv` | RA/Dec residual per observation, in arcsec and in σ |
| `results.png` | rotating-frame trajectory (top and side view) and error vs truth |
| `residuals.png` | residuals with ±1σ / ±3σ noise bands |
| `uncertainty.png` | error vs 1σ/3σ envelope over time, NEES over time, 68/95/99.7% confidence ellipses at the epoch |

Written to `sweeps.dir` (default `results/sweeps/`) by the sweeps: `noise_sweep.csv` / `.png`,
`iod_error_sweep.csv` / `.png` (one row per run), and `sweep_summary.json` (per-setting
statistics plus a paste-ready Markdown table).

Nothing downstream consumes these yet; they are the stack's deliverable (slides, reports, and the
baseline the collocation method will be compared against).

## d. Additional dependencies

- Python: numpy, scipy (`scipy.stats.chi2`), matplotlib (Agg backend, works headless / in Docker),
  pyyaml. No new packages beyond `requirements.txt`.
- Orekit (`orekit-jpype`, Java 11+, `orekit-data`) only through the estimator and propagator it
  is handed. The results layer itself runs without Orekit (`tests/test_results.py` needs none).
- `od_stack/frames.py` for EMI → rotating-frame conversion in the plots.
- Truth file `data/synthetic_truth.csv` for anything graded, and its `ra_true_deg` / `dec_true_deg`
  columns for the sweeps' fresh noise.

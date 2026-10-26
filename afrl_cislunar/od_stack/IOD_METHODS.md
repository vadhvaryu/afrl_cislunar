# CR3BP initial orbit determination: methods and benchmark

Part B (Sanjib, David and Keira's item; Zohar helping). Replaces the `truth_plus_error` test stand-in
with real angles-only IOD methods that work in the Earth–Moon three-body problem.

```
python -m od_stack.run --iod range_grid          # any method below, through the whole stack
python -m od_stack.iod_benchmark                 # B3 benchmark -> results/iod_benchmark/ (~50 min, 3 workers)
```

## a. Input

`Observations` (contracts.py): RA/Dec in the EMI frame, observer positions (Earth's center so far),
times. No method reads the truth file; `tests/test_iod_registry.py` runs every method with every read of
it made to raise. Shared tools: `od_stack/cr3bp_tools.py` (STM, rotating-frame lines of sight, Lagrange
line-of-sight derivatives, an observation generator that reproduces `obs_generator.py` exactly).

## b. Inner working

| Key | File | What it does | Prior it assumes |
|---|---|---|---|
| `cr3bp_dc` | `iod_cr3bp_dc.py` | Jo, Baker-McEvilly & Canales (arXiv 2507.22350): Newton on the ranges at 3 observations + the middle velocity, Eq. 22 Jacobian via the STM, multi-start range 200,000–500,000 km. **Two additions** (the paper's own start converged from 0 of 31 starts on our 8-day arc): a second starting velocity from a curve through all 5 observations, and a bounded least-squares fallback. Picks the solution with the lowest residual on all 5 observations. | range 200,000–500,000 km (stands in for the paper's brightness-based guess) |
| `range_grid` | `iod_range_grid.py` | ~900 (range, range-rate) hypotheses at the middle observation (7,500 km × 0.15 km/s, the spacing of AFRL's CAR-MHF; not an implementation of CAR-MHF), scored by a vectorized CR3BP propagation (0.3 s), best 5 distinct refined by BLS (or multiple shooting), lowest-RMS fit that passes the 3σ gate. Returns a BLS-converged state; flags two passing fits (`info["ambiguous"]`). | range 200,000–500,000 km, range-rate ±1.5 km/s |
| `range_grid_fd` | `iod_range_grid.py` | The range-only baseline: velocity from finite differences of the same range along neighbouring lines of sight; best 8 of 31 refined. | as above |
| `halo_catalog` | `iod_halo_catalog.py`, `halo_family.py` | Searches 100 members × 1,000 phases of the L1 northern halo family (generated in our CR3BP from an independent seed, never the truth) for the best match to the angles, with no propagation per candidate; refines over continuous (member, phase). | **the object is on an L1 northern halo** |
| `perturbed_laplace` | `iod_laplace.py` | Laplace's method with Earth + Moon gravity (our own generalization in the spirit of Liu, Hou & Armellin 2025, which is paywalled): solves the range equation numerically and keeps every root. | none |
| `laplace_moon`, `gooding_moon` | `iod_laplace.py` | Orekit's own `IodLaplace` / `IodGooding` with the **Moon** as the central body. Laplace needed two workarounds: run in CR3BP units (it returns nothing in SI on cislunar arcs: a 1e-14 determinant cutoff), and keep all polynomial roots (it keeps the largest, a false root here). | two-body motion about the Moon |
| `gauss`, `laplace`, `gooding` | `iod.py` | Orekit's two-body methods about the Earth (baselines). | two-body motion about the Earth |

## c. Output

A `StateEstimate` (EMI, SI) at the middle observation, with `info` listing the candidate solutions. The
benchmark writes `results/iod_benchmark/`: `runs.csv`, `summary.json`, `SUMMARY_tables.md`,
`cross_check.csv`, `success_heatmap.png`, `bls_after_iod.png`, `iod_error_boxplot.png`, `runtime.png`.

## d. Additional dependencies

None new. The Orekit-based methods (`range_grid*` with BLS, `*_moon`, `gauss`, `laplace`, `gooding`) need
orekit-jpype and orekit-data, like the rest of the stack.

---

## Benchmark (B3)

L1 halo truth; 5 observations from Earth's center on three arcs (`current` = days 0–8, the
`synthetic_obs.csv` geometry; `dense_1d` = 1 day from day 3; `dense_6h` = 6 hours from day 4); noise 1, 10 and
100″; 10 seeds; the gated robust estimator (`orekit_bls_robust`) run from every IOD. "Converged" = the
estimator reaches the same solution it reaches from the truth (within 10 km) and passes the 3σ gate. A method
**works** on a scenario if ≥ 9 of 10 seeds land within 10,000 km (the BLS basin from Part A) and converge.
Trimmed from TASK.md's full grid (seeds 10 not 20; noise 1/10/100″; all 4 triples only for the fast
3-observation methods, the default triple for `cr3bp_dc`). 2,250 runs, 48 min on 3 workers.

Within 10,000 km / converged, of 10 seeds, and median IOD error (best triple; ✓ = works):

| Method | current 1″ | current 10″ | current 100″ | 1 day 10″ | 6 h 10″ | Median s/call |
|---|---|---|---|---|---|---|
| `halo_catalog` | 10/10, 3 km ✓ | 10/10, 10 km ✓ | 10/10, 96 km ✓ | 10/10, 10 km ✓ | 10/10, 11 km ✓ | 0.3 |
| `range_grid` | 10/10, 3 km ✓ | 10/10, 27 km ✓ | 10/10, 268 km ✓ | 9/9, 1,259 km ✓ | 9/10, 7,779 km ✓ | 2.2 (up to ~25 with BLS) |
| `range_grid_fd` | same as `range_grid` | | | | | 3.1 |
| `cr3bp_dc` | 10/10, 4 km ✓ | 10/10, 37 km ✓ | 10/10, 378 km ✓ | 10/10, 1,403 km ✓ | 4/7, 8,869 km | 3.4 (up to ~70) |
| `gooding_moon` | 8/10 | 10/10, 5,645 km ✓ | 9/10, 4,806 km ✓ | 0/10 | 6/6 | 0.01 |
| `laplace_moon` | 0 | 0 | 0 | 1/1 | 2/3 | 0.01 |
| `perturbed_laplace` | no root | no root | no root | 0/3 | 0/3 | 0.05 |
| `gauss`, `laplace`, `gooding` (Earth) | fail | fail | fail | fail | fail | 0.00 |

What the numbers show:

1. **Three methods work on the current data at every noise level:** `halo_catalog`, `range_grid` and
   `cr3bp_dc`. Their answers agree with each other within 18–31 km (median, 10″). From each, the gated BLS
   converges every time.
2. **On short arcs the data, not the method, set the limit.** BLS started from the truth itself lands a
   median 1,266 km (1 day) and 7,460 km (6 hours) from the truth at 10″: from Earth's center the range is barely
   observable over a short arc. `range_grid` reports two fits that both pass the gate there
   (`info["ambiguous"]`), 6,000 and 12,000 km from the truth. Only `halo_catalog` stays accurate on short arcs
   (10–11 km), because its orbit-family prior supplies the information the angles lack.
3. **`halo_catalog` is the most accurate and the fastest, but only for objects on the L1 northern halo family.**
   On a DRO, the 9:2 NRHO or a large L2 halo it fits at 12,800–18,600″ and refuses (IODFailure); an L1
   Lyapunov orbit gets through with a poor guess (~4,400″), so check `info["rms_arcsec"]`. Its accuracy here
   partly reflects that the truth is on the family.
4. **`cr3bp_dc` (Jo et al.) works on the 8-day and 1-day arcs but needed two additions to do so.** Noise-free
   convergence windows in the starting range: 1 day ≈ [−28,000, +4,000] km and 6 hours ≈ [−22,000, 0] km around
   the true middle range, close to the paper's Table 8 scenario #1 (fixed observer: [−28,265, +1,388] km). On the
   8-day arc the window is lopsided the other way ([0, +90,000] km with gaps), and the paper's own start fails.
5. **Orekit's Laplace and Gooding can be made to run with the Moon as the central body, but a two-body orbit
   about the Moon cannot represent an L1 halo.** `laplace_moon` fails almost everywhere. `gooding_moon` works
   only with observations 0–1–2 (days 0–4), where a lunar two-body arc happens to stay within ~5,600 km of the
   halo, so BLS converges from it; every other triple fails. Earth-centered Gauss, Laplace and Gooding fail
   everywhere, as already known.
6. **`perturbed_laplace` finds no root on the 2-day spacing.** It needs dense, nearly noise-free arcs (6 hours,
   all 5 observations, noise-free: 4.8 km). From Earth's center, Earth's gravity lies along the line of sight and
   drops out of the range equation, leaving only the Moon's small pull to fix the range.

## Recommendation

- **Default `iod.method`: `range_grid`.** It assumes no orbit family, works on every arc at ≤ 10″, returns a
  BLS-converged state, and says when the data allow two answers. Cost: 2–25 s per call.
- **Use `halo_catalog` as a fast first guess and a cross-check** when the object is believed to be on an L1
  halo, and always look at its residual. Extending it to other families (L2 halos, NRHOs, DROs, Lyapunov
  orbits; the catalogue in `orbit_catalog.py` has seeds) is cheap.
- **Keep `cr3bp_dc` as an independent, published method.** It agrees with the others within ~30 km on the
  current data.
- `config.yaml` still defaults to the stand-in (`truth_plus_error`); switching the default is a team decision.

## Open items

- A ground-station observer (David's generator: observer columns) should make range observable on short arcs
  and change point 2 above; rerun the benchmark then.
- `range_grid` and the Orekit fits on noise-free data: Orekit's angles include light time and the generator's
  don't, so a noise-free Orekit fit sits at ~175σ and fails the 3σ gate (the tests use a looser gate there).
  Fix in the generator.
- The Earth-centered `laplace` in `iod.py` likely hits the same 1e-14 determinant cutoff as `laplace_moon` did.

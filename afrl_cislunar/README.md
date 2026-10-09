# afrl_cislunar

NewSpace@Berkeley x AFRL, Fall 2026. Goal is containerized software that estimates and updates spacecraft trajectories in cislunar space from noisy sensor data.

Right now we're on the initial simulation build using the CR3BP (circular restricted three-body problem).

## Files

- `cr3bp.py` - shared CR3BP equations of motion, Jacobi constant, and state transition matrix (STM). Import from here instead of copying them into scripts.
- `weber_cr3bp_example.py` - CR3BP example from orbital-mechanics.space. Shows why tight solver tolerances matter.
- `artemis2_vs_cr3bp.py` - starts the CR3BP from Orion's real state after TLI and compares it to NASA's actual Artemis II trajectory.
- `obs_generator.py` - makes fake telescope observations (RA/Dec angles with noise) of a cislunar orbit for testing IOD. Writes `data/synthetic_obs.csv` (IOD input) and `data/synthetic_truth.csv` (answer key).
- `stm_check.py` - checks the STM against brute-force re-propagation and shows how a small starting error grows along an Artemis-like path. Doesn't need the NASA file.

## Running

Needs numpy, scipy, matplotlib, astropy:

```
pip install -r requirements.txt
python artemis2_vs_cr3bp.py
```

The Artemis script reads NASA's ephemeris file from `data/Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc`, so run it from the repo's main folder. Figures save to `figures/`. Ones we showed at meetings go in dated subfolders (e.g. `figures/2026-10-02/`).

## Notes

- States are `[x, y, z, vx, vy, vz]`, nondimensional, Earth-Moon rotating frame.
- CR3BP drifts from the real trajectory since it ignores the Sun, assumes a circular lunar orbit, and doesn't include course-correction burns.

## TODO

- Lagrange points, known periodic orbits
- higher fidelity models
- orbit determination / filtering
- containerize

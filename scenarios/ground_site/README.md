# Ground-site observer

The `plug_and_play` halo orbit, observed from a telescope on the ground instead of Earth's
center: the AFRL Maui Optical Site (AMOS) on Haleakala (20.7083 N, 156.2571 W, 3058 m).

`system.epoch_utc: 2026-11-19T08:00:00Z` is 22:00 Hawaii time. The five observations
(days 0, 2, 4, 6, 8, all at that clock time) see the object 36 to 80 degrees above the
horizon. Moving the epoch or the observation times can put observations below the horizon;
the run then stops and lists them.

```
python scripts/run_container.py --config scenarios/ground_site/config.yaml --output-dir ./outputs/container
```

Expected (10 arcsec noise, seed 1): collocation IOD about 15 km from the truth, batch least
squares about 14 km with a 1-sigma of about 25 km, similar to the geocenter case because the
noise dominates. See docs/configuration.md, "Observer location".

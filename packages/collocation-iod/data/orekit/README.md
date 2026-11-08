# Data for the synthetic Orekit demonstration

`UTC-TAI.history` is an unmodified IERS Earth Orientation Centre leap-second
history downloaded on 2026-11-05 from
https://hpiers.obspm.fr/iers/bul/bulc/UTC-TAI.history.

SHA-256: `54e702abdc388ae3bf8cfc5f126900a5277829ad90e80f6773df6e714a133642`.

This is sufficient for the existing stack's synthetic J2000/geocenter scenario,
whose code supplies zero Earth-orientation values and an analytic circular Moon.
It is not a full Orekit data distribution. Real ground stations, lunar/solar
ephemerides, and other physical models need the corresponding up-to-date data.
The core collocation solver does not load this file.

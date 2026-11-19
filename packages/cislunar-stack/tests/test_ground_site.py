"""Observer on the real, rotating Earth: latitude + longitude + calendar epoch."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from od_stack.config import ConfigError
from od_stack.frames import emi_axes_in_gcrf, moon_ecliptic, parse_epoch

ROOT = Path(__file__).resolve().parents[3]
SCENARIO = ROOT / "scenarios/ground_site/config.yaml"


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("OREKIT_DATA"))


needs_orekit = pytest.mark.skipif(not _orekit_ready(), reason="needs orekit_jpype and OREKIT_DATA")


def test_moon_series_matches_meeus_example():
    # Meeus, Astronomical Algorithms, example 47.a: 1992 April 12, 0h TD (equinox of date)
    dt = parse_epoch("1992-04-11T23:59:01Z")
    lon, lat, _ = moon_ecliptic(dt)
    T = -0.077221081451
    assert abs(lon + 1.3969713 * T - 133.162655) < 0.3
    assert abs(lat - (-3.229126)) < 0.3


def test_emi_axes_are_a_rotation_with_x_towards_the_moon():
    R = emi_axes_in_gcrf("2026-11-19T08:00:00Z")
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    lon, lat, _ = np.radians(moon_ecliptic(parse_epoch("2026-11-19T08:00:00Z")))
    eps = np.radians(23.4392911)
    moon_ecl = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    moon_eq = np.array([[1, 0, 0], [0, np.cos(eps), -np.sin(eps)], [0, np.sin(eps), np.cos(eps)]]) @ moon_ecl
    assert np.degrees(np.arccos(np.clip(R[:, 0] @ moon_eq, -1, 1))) < 5.2   # x within the orbit tilt of the Moon
    pole = np.array([0, -np.sin(eps), np.cos(eps)])                           # ecliptic pole in GCRF
    assert np.degrees(np.arccos(R[:, 2] @ pole)) == pytest.approx(5.145, abs=1e-6)


def _scenario(tmp_path, **changes):
    c = yaml.safe_load(SCENARIO.read_text())
    c["output"] = {"directory": str(tmp_path / "out"), "plots": False}
    c["prediction"]["points"] = 40
    for path, value in changes.items():
        node = c
        *keys, last = path.split(".")
        for k in keys:
            node = node[k]
        node[last] = value
    return c


@needs_orekit
def test_ground_site_moves_with_earth_rotation():
    from od_stack.orekit_models import configure_reference, observer_positions
    from od_stack.orekit_setup import start_orekit

    start_orekit()
    configure_reference("2026-11-19T08:00:00Z")
    site = {"type": "ground_site", "latitude_deg": 20.7083, "longitude_deg": -156.2571, "altitude_m": 3058.0}
    sidereal_day = 86164.0905
    p = observer_positions(site, np.array([0.0, sidereal_day / 4, sidereal_day]))
    r = np.linalg.norm(p, axis=1)
    np.testing.assert_allclose(r, r[0], atol=1.0)                               # stays on the same circle
    assert 6.37e6 < r[0] < 6.385e6                                              # Earth radius + 3 km
    circle = 6378137.0 * np.cos(np.radians(20.7083))
    assert np.linalg.norm(p[1] - p[0]) == pytest.approx(circle * np.sqrt(2), rel=0.01)   # quarter turn
    assert np.linalg.norm(p[2] - p[0]) < 100.0                                   # back after one sidereal day


@needs_orekit
def test_full_pipeline_from_a_ground_site(tmp_path):
    from od_stack.run import run

    s = run(_scenario(tmp_path))
    assert s["observer"]["type"] == "ground_site"
    assert min(s["observer"]["elevation_deg"]) > 10
    assert s["position_error_km"] < 3 * s["sigma_position_km"] < 150
    assert (tmp_path / "out" / "observations.csv").exists()


@needs_orekit
def test_ground_site_geometry_is_modelled_exactly(tmp_path):
    """Nearly noise-free angles: collocation (same model as the simulator) recovers the state
    to metres, and BLS to within the light-time effect the simulator leaves out (< 1 km)."""
    from od_stack.run import run

    s = run(_scenario(tmp_path, **{"simulation.observations.noise.sigma_arcsec": 0.01}), make_files=False)
    assert s["iod_position_error_km"] < 0.1
    assert s["position_error_km"] < 1.0


@needs_orekit
def test_object_below_horizon_is_reported(tmp_path):
    from od_stack.run import run

    with pytest.raises(ConfigError, match="below the site's minimum elevation"):
        run(_scenario(tmp_path, **{"simulation.observations.times_days": [0, 0.25, 0.5, 2, 4]}), make_files=False)


@needs_orekit
def test_fixed_point_observer_now_works_with_bls(tmp_path):
    from od_stack.run import run

    c = _scenario(tmp_path)
    c["system"]["epoch_utc"] = None
    c["simulation"]["observer"] = {"type": "fixed_emi", "position_km": [3000, 4000, 2000]}
    s = run(c, make_files=False)
    assert s["position_error_km"] < 3 * s["sigma_position_km"]


@needs_orekit
def test_bls_rejects_observations_from_somewhere_else(tmp_path):
    """Observations simulated from Maui, but BLS told the observer is at Earth's center."""
    from od_stack.estimation import EstimationFailure, OrekitBatchLS
    from od_stack.frames import EarthMoonSystem
    from od_stack.orekit_models import configure_reference, observer_positions
    from od_stack.contracts import Observations, StateEstimate
    from od_stack.frames import radec_to_los

    configure_reference("2026-11-19T08:00:00Z")
    site = {"type": "ground_site", "latitude_deg": 20.7083, "longitude_deg": -156.2571, "altitude_m": 3058.0}
    t = np.array([0.0, 86400.0, 172800.0])
    ra, dec = np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.1, 0.2])
    obs = Observations(t=t, los=radec_to_los(ra, dec), observer=observer_positions(site, t),
                       sigma=np.full(3, 5e-5), ra=ra, dec=dec)
    bls = OrekitBatchLS(EarthMoonSystem(), observer={"type": "geocenter", "position_km": [0, 0, 0]})
    x0 = StateEstimate(t=86400.0, r=np.array([3.5e8, 0, 0]), v=np.array([0, 1000.0, 0]), method="test")
    with pytest.raises(EstimationFailure, match="configured observer"):
        bls.refine(obs, x0)

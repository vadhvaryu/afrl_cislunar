"""High-fidelity truth and model-mismatch experiment (od_stack/hifi.py). Needs Orekit + orekit-data (DE440)."""

import os

import numpy as np
import pytest


def _ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    p = os.environ.get("OREKIT_DATA")
    return bool(p) and os.path.isdir(os.path.join(os.path.expanduser(p), "DE-440-ephemerides"))


pytestmark = pytest.mark.skipif(not _ready(), reason="needs orekit-jpype and OREKIT_DATA with DE-440 ephemerides")
_SC = {}


def _scenario():
    if "sc" not in _SC:
        from od_stack.hifi import Scenario
        _SC["sc"] = Scenario("L1_halo_truth")
    return _SC["sc"]


def test_emi_axes_follow_the_real_moon():
    sc = _scenario()
    assert np.allclose(sc.M.T @ sc.M, np.eye(3), atol=1e-12)
    assert 356_000 < sc.D / 1e3 < 407_000                     # perigee to apogee
    r, _ = sc.truth(0.0)
    assert np.allclose(r[0], sc.r0, atol=1e-3)
    rho = sc.M.T @ sc.r0 / sc.D + sc.sys.earth_rot                   # back to CR3BP rotating coordinates
    assert np.allclose(rho, [0.872763, 0.0, 0.190505], atol=1e-12)


def test_observations_include_light_time():
    sc = _scenario()
    ang = sc.noise_free_angles()
    r, _ = sc.truth(sc.t_obs)
    ra_geo = np.arctan2(r[:, 1], r[:, 0]) % (2 * np.pi)
    d = np.degrees((ang[:, 0] - ra_geo + np.pi) % (2 * np.pi) - np.pi) * 3600
    assert np.all(np.abs(d) > 0.3) and np.all(np.abs(d) < 1.5)      # ~0.5-0.8": 1.2 s of motion at ~1 km/s


def test_full_model_recovers_the_truth_and_cr3bp_fails_the_gate():
    from od_stack.hifi import fit
    sc = _scenario()
    t_e = float(sc.t_obs[2])
    (rt,), (vt,) = sc.truth(t_e)
    ra, dec, sig = sc.observations(1e-3, 0)                          # ~noise-free
    f = fit(sc, "full", t_e, rt + 1e6 * np.array([0.6, 0.0, 0.8]), vt + np.array([10.0, 0, 0]), ra, dec, sig)
    assert np.linalg.norm(f["r"] - rt) < 1e3 and f["rms"] < 3
    ra, dec, sig = sc.observations(10.0, 1)
    g = fit(sc, "cr3bp_nominal", t_e, rt + 1e6 * np.array([0.6, 0.0, 0.8]), vt, ra, dec, sig)
    assert g["rms"] > 3                                              # the gate flags the model mismatch

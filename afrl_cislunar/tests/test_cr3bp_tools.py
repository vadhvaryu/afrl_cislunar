"""Tests for the shared CR3BP tools (Part B, B1). No Orekit needed."""

import csv
import os
import warnings

import numpy as np
import pytest

from od_stack.cr3bp_tools import cr3bp_jacobian, los_derivatives, make_synthetic_obs, obs_to_rotating, stm_propagate
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, emi_to_rotating, rotating_to_emi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])   # obs_generator.py
GEN_TIMES = np.linspace(0, 8 * 86400.0, 5)                              # obs_generator.py: 5 obs over 8 days
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _rows(name):
    with open(f"data/{name}", newline="") as f:
        return list(csv.DictReader(f))


def test_jacobian_matches_cr3bp_module():
    import cr3bp
    for s in (TRUTH_STATE, np.array([0.9023581414, 0.0936861644, 0.1421356838, 0.1218991572, 0.1379776334, -0.2067777368])):
        assert np.allclose(cr3bp_jacobian(s, SYS.mu), cr3bp.cr3bp_jacobian(s, SYS.mu), rtol=0, atol=1e-14)


@pytest.mark.parametrize("tau1", [1.2, -0.7])
def test_stm_matches_central_finite_differences(tau1):
    x0, tau0 = np.array([0.9023581414, 0.0936861644, 0.1421356838, 0.1218991572, 0.1379776334, -0.2067777368]), 0.3
    x1, phi = stm_propagate(x0, tau0, tau1, SYS.mu)
    assert np.allclose(x1, propagate_cr3bp(x0, tau0, np.array([tau1]), SYS.mu)[0], atol=1e-12)
    h = 1e-7
    fd = np.column_stack([(propagate_cr3bp(x0 + h * e, tau0, np.array([tau1]), SYS.mu)[0]
                           - propagate_cr3bp(x0 - h * e, tau0, np.array([tau1]), SYS.mu)[0]) / (2 * h) for e in np.eye(6)])
    assert np.abs(fd - phi).max() / np.abs(phi).max() < 1e-6
    assert np.allclose(stm_propagate(x0, tau0, tau0, SYS.mu)[1], np.eye(6))


def test_synthetic_obs_reproduce_the_generator():
    """Zero noise: the noise-free angles in synthetic_truth.csv; seed 1 and 10": synthetic_obs.csv itself."""
    t_rows, o_rows = _rows("synthetic_truth.csv"), _rows("synthetic_obs.csv")
    o = make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 0.0, 1, SYS)
    ra, dec = np.degrees(o.ra) % 360, np.degrees(o.dec)
    assert np.allclose(ra, [float(r["ra_true_deg"]) for r in t_rows], rtol=0, atol=1e-8)
    assert np.allclose(dec, [float(r["dec_true_deg"]) for r in t_rows], rtol=0, atol=1e-8)
    o = make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 10.0, 1, SYS)
    assert np.allclose(np.degrees(o.ra) % 360, [float(r["ra_deg"]) for r in o_rows], rtol=0, atol=1e-8)
    assert np.allclose(np.degrees(o.dec), [float(r["dec_deg"]) for r in o_rows], rtol=0, atol=1e-8)
    assert np.allclose(o.sigma, np.radians(10 / 3600))


def test_obs_to_rotating_frames():
    o = make_synthetic_obs(TRUTH_STATE, GEN_TIMES, 0.0, 0, SYS)
    tau, R_o, u = obs_to_rotating(o, SYS)
    for t in tau:                                   # Earth's center maps to (-mu, 0, 0)
        assert np.allclose(emi_to_rotating(t, np.zeros(3), np.zeros(3), SYS)[:3], [-SYS.mu, 0, 0], atol=1e-15)
    assert np.allclose(R_o, [-SYS.mu, 0, 0], atol=1e-15)
    truth = propagate_cr3bp(TRUTH_STATE, 0.0, tau, SYS.mu)
    d = truth[:, :3] - R_o
    assert np.allclose(u, d / np.linalg.norm(d, axis=1)[:, None], atol=1e-12)
    for t, x, ui, li in zip(tau, truth, u, o.los):  # a direction rotates with R(tau)^T, no offset
        r, _ = rotating_to_emi(t, x, SYS)
        assert np.allclose(li, r / np.linalg.norm(r), atol=1e-12)
        assert np.allclose(emi_to_rotating(t, r, np.zeros(3), SYS)[:3] - R_o[0], np.linalg.norm(r) / SYS.du_m * ui, atol=1e-12)


def test_los_derivatives_on_a_dense_arc():
    """5 noise-free observations over 6 hours: the Lagrange fit's rho_hat' and rho_hat'' agree
    with finite differences of the true line of sight."""
    t_mid = 4 * 86400.0
    times = t_mid + np.linspace(-3, 3, 5) * 3600
    o = make_synthetic_obs(TRUTH_STATE, times, 0.0, 0, SYS)
    u, du, du2 = los_derivatives(o, range(5), t_mid)
    h = 600.0
    tr = make_synthetic_obs(TRUTH_STATE, [t_mid - h, t_mid, t_mid + h], 0.0, 0, SYS).los
    fd1, fd2 = (tr[2] - tr[0]) / (2 * h), (tr[2] - 2 * tr[1] + tr[0]) / h**2
    assert np.allclose(u, tr[1], atol=1e-10)
    assert np.linalg.norm(du - fd1) / np.linalg.norm(fd1) < 1e-4
    assert np.linalg.norm(du2 - fd2) / np.linalg.norm(fd2) < 1e-2
    assert abs(u @ du) < 1e-15 and abs(u @ du2 + du @ du) < 1e-15


def test_truth_guard_blocks_the_answer_key(no_truth):
    from od_stack.iod import TruthPlusError
    with pytest.raises(PermissionError):
        TruthPlusError(SYS, truth_path="data/synthetic_truth.csv")
    with pytest.raises(PermissionError):
        open("data/synthetic_truth.csv")
    open("data/synthetic_obs.csv").close()          # observations stay readable

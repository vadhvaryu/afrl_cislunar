"""Part B IOD methods: registered, and none of them can read the answer key."""

import os
import warnings

import numpy as np
import pytest

from od_stack.cr3bp_tools import make_synthetic_obs
from od_stack.dynamics import propagate_cr3bp
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.iod import IOD_METHODS, PART_B_METHODS, IODFailure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = EarthMoonSystem()
TRUTH_STATE = np.array([0.872763, 0.0, 0.190505, 0.0, 0.235988, 0.0])
warnings.filterwarnings("ignore", message="No observer columns")


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def _orekit_ready():
    try:
        import orekit_jpype  # noqa: F401
    except ImportError:
        return False
    p = os.environ.get("OREKIT_DATA")
    return bool(p) and os.path.exists(os.path.expanduser(p))


def test_all_part_b_methods_are_registered():
    for name in PART_B_METHODS:
        assert name in IOD_METHODS and IOD_METHODS[name].name == name


@pytest.mark.parametrize("name", PART_B_METHODS)
def test_part_b_method_never_reads_the_truth(name, no_truth):
    """Constructed exactly as run.py does (it passes truth_path), then run on noise-free data, with
    every read of the truth file made to raise. A method may fail to find an orbit (IODFailure, e.g.
    the Moon-centered two-body methods on a halo), but it must not touch the answer key."""
    if name in ("laplace_moon", "gooding_moon", "range_grid", "range_grid_fd") and not _orekit_ready():
        pytest.skip("needs Orekit")
    t = np.linspace(0, 8 * 86400.0, 5)
    obs = make_synthetic_obs(TRUTH_STATE, t, 1e-3, 0, SYS)
    kw = {"orekit_data": None, "truth_path": "data/synthetic_truth.csv"}
    if name.startswith("range_grid"):
        kw.update(refine_with="multiple_shooting", gate_rms=10)
    try:
        x = IOD_METHODS[name](SYS, **kw).estimate(obs)
    except IODFailure:
        assert name in ("perturbed_laplace", "laplace_moon", "gooding_moon")   # measured: no solution on this arc
        return
    tau = SYS.tau(x.t)
    r_true, _ = rotating_to_emi(tau, propagate_cr3bp(TRUTH_STATE, 0.0, np.array([tau]), SYS.mu)[0], SYS)
    assert np.linalg.norm(x.r - r_true) < 10_000e3                  # within the BLS basin

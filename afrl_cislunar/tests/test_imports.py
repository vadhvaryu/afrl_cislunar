"""Every od_stack module imports on its own (a fresh interpreter, nothing imported before it), and
the halo-family cache goes somewhere writable when the code folder is read-only (container image)."""

import os
import pkgutil
import subprocess
import sys

import pytest

import od_stack
from od_stack import halo_family

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = sorted(m.name for m in pkgutil.iter_modules(od_stack.__path__))


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_alone(module):
    r = subprocess.run([sys.executable, "-c", f"import od_stack.{module}"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


def test_cache_path_falls_back_to_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("CISLUNAR_CACHE_DIR", raising=False)
    monkeypatch.setattr(halo_family, "_writable", lambda path: False)
    monkeypatch.chdir(tmp_path)
    assert halo_family.default_cache_path() == os.path.join(str(tmp_path), "results", "halo_catalog", "L1N.npz")
    monkeypatch.setenv("CISLUNAR_CACHE_DIR", "/somewhere")
    assert halo_family.default_cache_path() == os.path.join("/somewhere", "halo_catalog", "L1N.npz")

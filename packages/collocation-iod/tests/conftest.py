"""Local source checkout support; installed distributions need no path changes."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
# The existing repository is deliberately not a dependency of the numerical core.
STACK = PROJECT.parent / "cislunar-stack" / "src"
if STACK.is_dir():
    sys.path.insert(0, str(STACK))

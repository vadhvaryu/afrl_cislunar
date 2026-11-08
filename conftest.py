"""Allow both source packages to be tested together before installation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for package in ("cislunar-stack", "collocation-iod"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

"""Collocation IOD inside the Part B IOD benchmark (the cr3bp-iod branch), on the same noise draws.

The Part B methods (range_grid, halo_catalog, cr3bp_dc, ...) live on the cr3bp-iod branch, in the
pre-reorganization layout, so this script runs that branch's own benchmark harness
(od_stack/iod_benchmark.py) with collocation registered as one more method. Nothing on that
branch is modified. Each case: 5 geocenter observations, IOD, then the gated robust batch least
squares estimator; "converged" means it reaches the fit started from the truth (within 10 km).

    PARTB_ROOT=/path/to/cr3bp-iod/afrl_cislunar OREKIT_DATA=... \\
        python benchmarks/collocation_iod/compare_with_part_b.py --out outputs/benchmarks/part_b \\
        --methods collocation range_grid halo_catalog cr3bp_dc --default-triple-only

The environment needs that branch's requirements plus this repository's collocation-iod package
(pip install ./packages/collocation-iod), but not the reorganized cislunar-stack package, whose
od_stack would shadow the branch's.
"""

import os
import sys

ROOT = os.environ.get("PARTB_ROOT")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "od_stack")):
    sys.exit("Set PARTB_ROOT to the afrl_cislunar folder of a cr3bp-iod checkout")
sys.path.insert(0, ROOT)              # also runs in each spawned worker, which re-imports this file

from od_stack import iod_benchmark    # noqa: E402  (the Part B harness)

_partb_init = iod_benchmark.init_worker


def init_worker(orekit_data):
    """The Part B worker setup, plus collocation in the IOD registry."""
    from collocation_iod.adapters import register_with_od_stack

    register_with_od_stack()
    _partb_init(orekit_data)


if __name__ == "__main__":
    iod_benchmark.init_worker = init_worker      # the harness's main() passes init=init_worker by name
    args = sys.argv[1:]
    if "--out" not in args:
        sys.exit("Pass --out (the harness default would overwrite the branch's own results/iod_benchmark)")
    i = args.index("--out") + 1
    args[i] = os.path.abspath(args[i])
    os.chdir(ROOT)                                # the harness reads data/ relative to its own folder
    sys.exit(iod_benchmark.main(args))

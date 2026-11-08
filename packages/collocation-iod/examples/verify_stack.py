"""Exercise the actual Orekit chain on the repository's synthetic scenario.

Run from any directory after installing collocation-iod[stack]. Optional flags
make Java and output locations explicit; nothing modifies the core solver.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--orekit-data", type=Path, default=project / "data" / "orekit")
    repo = project.parent.parent
    parser.add_argument("--output", type=Path, default=repo / "outputs" / "stack")
    parser.add_argument("--run-tests", action="store_true", help="Also run the new module and original stack tests (requires pytest)")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.java_home:
        os.environ["JAVA_HOME"] = str(args.java_home.resolve())
    os.environ["OREKIT_DATA"] = str(args.orekit_data.resolve())
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    stack = project.parent / "cislunar-stack"
    scenario = repo / "scenarios" / "halo_geocenter"
    if not (scenario / "config.yaml").is_file():
        parser.error("This example needs the repository's scenarios/halo_geocenter fixture")
    sys.path.insert(0, str(stack / "src"))
    from collocation_iod.adapters import register_with_od_stack
    from od_stack.config import load_config
    from od_stack.run import run
    from od_stack.contracts import StateEstimate
    from od_stack.frames import EarthMoonSystem
    from collocation_iod.orekit_bridge import to_cartesian_orbit, from_cartesian_orbit

    register_with_od_stack()
    cfg = load_config(scenario / "config.yaml")
    cfg["iod"]["method"] = "collocation"
    cfg["results"]["dir"] = str(output)
    cfg["data"]["observations"] = str(scenario / cfg["data"]["observations"])
    cfg["data"]["truth"] = str(scenario / cfg["data"]["truth"])
    cfg["orekit"]["data_path"] = str(args.orekit_data.resolve())
    summary = run(cfg)
    # Exercise real Java object conversion, independently of the stack's own wrapper.
    import numpy as np
    from org.orekit.frames import FramesFactory
    from org.orekit.time import AbsoluteDate

    probe = StateEstimate(t=1234., r=np.array([3e8,2e7,1e7]), v=np.array([-40.,400.,20.]))
    frame, date = FramesFactory.getGCRF(), AbsoluteDate.J2000_EPOCH
    orbit = to_cartesian_orbit(probe,frame=frame,reference_date=date,gm=EarthMoonSystem().gm_earth)
    t, r, v = from_cartesian_orbit(orbit,frame=frame,reference_date=date)
    assert abs(t-probe.t) < 1e-9
    np.testing.assert_allclose(r,probe.r,rtol=0,atol=1e-8)
    np.testing.assert_allclose(v,probe.v,rtol=0,atol=1e-10)
    # Reference bounds grade this particular bundled simulation, not general IOD.
    assert summary["iod_position_error_km"] < 200, summary
    assert summary["position_error_km"] < 100, summary
    assert summary["max_position_error_km"] < 200, summary
    print(json.dumps({k:v for k,v in summary.items() if k != "covariance_6x6_SI"}, indent=2))
    if args.run_tests:
        import pytest
        return pytest.main([str(project / "tests"), str(stack / "tests"), "-q"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

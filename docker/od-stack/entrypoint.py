#!/usr/bin/env python3
"""cislunar-od: entry point of the cislunar-od container image.

    cislunar-od run [CONFIG] [OPTIONS]   run the stack on a case (default CONFIG: <case>/config.yaml);
                                         OPTIONS go to od_stack.run, e.g. --iod gooding
    cislunar-od init                     copy the example case (config.yaml + data/) into the case folder
    cislunar-od test [PYTEST ARGS]       run the test suite inside the image
    cislunar-od info                     print the image, library and machine versions as JSON
    cislunar-od COMMAND ...              run any other command in the case folder,
                                         e.g. python -m od_stack.run --compare-iod

The case folder is where a run reads its config and data and writes its results: /case (mount a
host folder there: docker run -v "$PWD/case:/case" ...), or the current directory when /case is
not writable (Apptainer without a bind). CASE_DIR overrides both. Every run also writes a
run.json: command, exit code, wall time, image version, library versions and machine.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata

APP = "/app/afrl_cislunar"
PACKAGES = ("numpy", "scipy", "orekit_jpype", "JPype1", "astropy", "matplotlib", "PyYAML")


def case_dir() -> str:
    if os.environ.get("CASE_DIR"):
        return os.environ["CASE_DIR"]
    return "/case" if os.path.isdir("/case") and os.access("/case", os.W_OK) else os.getcwd()


def java_version() -> str:
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=30)
        return (out.stderr or out.stdout).splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return "unavailable"


def info() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "image_version": os.environ.get("CISLUNAR_OD_VERSION", "unknown"),
        "git_commit": os.environ.get("CISLUNAR_OD_COMMIT", "unknown"),
        "orekit_data_commit": os.environ.get("OREKIT_DATA_COMMIT", "unknown"),
        "python": platform.python_version(),
        "java": java_version(),
        "packages": versions,
        "machine": {"cpu_architecture": platform.machine(), "cpu_count": os.cpu_count(),
                    "os": platform.platform(), "hostname": platform.node()},
    }


def init(case: str, quiet: bool = False) -> list[str]:
    """Copy the example config and data into the case folder; never overwrites a file."""
    os.makedirs(case, exist_ok=True)
    made = []
    cfg = os.path.join(case, "config.yaml")
    if not os.path.exists(cfg):
        shutil.copyfile(os.path.join(APP, "config.yaml"), cfg)
        made.append("config.yaml")
    for root, _, files in os.walk(os.path.join(APP, "data")):
        rel = os.path.relpath(root, APP)
        for f in files:
            dst = os.path.join(case, rel, f)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(root, f), dst)
                made.append(os.path.join(rel, f))
    if made and not quiet:
        print(f"cislunar-od: example case copied into {case}: {', '.join(sorted(made))}", file=sys.stderr)
    return made


def results_dir(config: str, options: list[str]) -> str:
    """Where od_stack.run writes this run's results (results.dir, possibly overridden by --set)."""
    import yaml

    rdir = "results"
    try:
        with open(config) as f:
            rdir = ((yaml.safe_load(f) or {}).get("results") or {}).get("dir") or rdir
    except (OSError, yaml.YAMLError):
        pass
    for a, b in zip(options, options[1:] + [""]):
        for s in ([b] if a == "--set" else [a[6:]] if a.startswith("--set=") else []):
            if s.startswith("results.dir="):
                rdir = s.split("=", 1)[1]
    return rdir


def timed(cmd: list[str], cwd: str, log_dir: str, log_name: str) -> int:
    """Run cmd in cwd and record it in log_dir/log_name."""
    start = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        code = subprocess.run(cmd, cwd=cwd).returncode
    except FileNotFoundError:
        print(f"cislunar-od: command not found: {cmd[0]} (see: cislunar-od help)", file=sys.stderr)
        code = 127
    record = {"command": cmd, "case_dir": cwd, "exit_code": code, "started_utc": started,
              "wall_time_s": round(time.time() - start, 3), **info()}
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, log_name), "w") as f:
            json.dump(record, f, indent=2)
    except OSError as e:
        print(f"cislunar-od: could not write the run record: {e}", file=sys.stderr)
    return code


def main(argv: list[str]) -> int:
    cmd, args = (argv[0], argv[1:]) if argv else ("help", [])
    case = case_dir()
    if cmd in ("help", "-h", "--help"):
        print(__doc__.strip())
        return 0
    if cmd == "info":
        print(json.dumps(info(), indent=2))
        return 0
    if cmd == "init":
        made = init(case, quiet=True)
        print(f"{len(made)} file(s) copied into {case}" + (f": {', '.join(sorted(made))}" if made else
                                                           " (the example case is already there)"))
        return 0
    if cmd == "test":
        # the code folder is read-only: caches the tests create go to /tmp
        env = {"CISLUNAR_CACHE_DIR": "/tmp/cislunar-cache", **os.environ}
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
                              cwd=APP, env=env).returncode
    if cmd == "run":
        config = args[0] if args and args[0].endswith((".yaml", ".yml")) else None
        options = args[1:] if config else args
        if config is None:
            config = os.path.join(case, "config.yaml")
            if not os.path.exists(config):
                init(case)
        config = os.path.abspath(os.path.join(case, config))
        if not os.path.exists(config):
            print(f"cislunar-od: config not found: {config}", file=sys.stderr)
            return 2
        here = os.path.dirname(config)                  # relative paths in the config are relative to it
        rdir = os.path.join(here, results_dir(config, options))
        return timed([sys.executable, "-m", "od_stack.run", "--config", config, *options], here, rdir, "run.json")
    # anything else: run it in the case folder, with the example data there if it is missing
    if not os.path.isdir(os.path.join(case, "data")):
        init(case)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return timed([cmd, *args], case, os.path.join(case, "runs"), f"{stamp}.json")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

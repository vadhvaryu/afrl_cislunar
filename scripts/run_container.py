"""Launch a YAML-controlled run with local input/output bind mounts (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def bind(source, target, readonly=False):
    # --mount uses CSV parsing; quote unusual host paths, including commas.
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(
        ["type=bind", f"source={source}", f"target={target}"] + (["readonly"] if readonly else []))
    return stream.getvalue()


def command(config, inputs, outputs, image):
    config, outputs = Path(config).resolve(), Path(outputs).resolve()
    if not config.is_file():
        raise ValueError(f"Configuration does not exist: {config}")
    if outputs == config or (outputs.exists() and not outputs.is_dir()):
        raise ValueError(f"Output must be a directory: {outputs}")
    if config.is_relative_to(outputs):
        raise ValueError("Choose an output directory separate from the configuration file")
    args = ["docker", "run", "--rm", "--mount", bind(config, "/workspace/config.yaml", True)]
    # Preserve host ownership of bind-mounted results on native Linux Docker.
    if os.name == "posix" and hasattr(os, "getuid"):
        args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if inputs is not None:
        inputs = Path(inputs).resolve()
        if not inputs.is_dir():
            raise ValueError(f"Input directory does not exist: {inputs}")
        if outputs.is_relative_to(inputs) or inputs.is_relative_to(outputs):
            raise ValueError("Input and output directories must be separate, non-overlapping directories")
        args += ["--mount", bind(inputs, "/workspace/data", True)]
    args += ["--mount", bind(outputs, "/workspace/outputs"),
             "--env", "CISLUNAR_OUTPUT_ROOT=/workspace/outputs", image,
             "--config", "/workspace/config.yaml"]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scenarios/plug_and_play/config.yaml")
    parser.add_argument("--input-dir", type=Path, help="Host input directory; YAML refers to its files as data/...")
    parser.add_argument("--output-dir", type=Path, help="Host output directory; YAML uses outputs/... inside it")
    parser.add_argument("--image", default="cislunar-stack:local", help="Deployment image, not a simulation choice")
    parser.add_argument("--build", action="store_true", help="Build this repository's image first")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating directories or running Docker")
    args = parser.parse_args(argv)
    config = args.config.resolve()
    inputs = args.input_dir
    if inputs is None and (config.parent / "data").is_dir():
        inputs = config.parent / "data"
    outputs = (args.output_dir or config.parent / "outputs").resolve()
    try:
        run_command = command(config, inputs, outputs, args.image)
        commands = []
        if args.build:
            commands.append(["docker", "build", "-f", str(ROOT / "containers/Dockerfile"),
                             "--build-arg", f"IMAGE_TAG={args.image}",    # recorded in each run's provenance
                             "-t", args.image, str(ROOT)])
        commands.append(run_command)
        if args.dry_run:
            print(json.dumps(commands, indent=2))
            return 0
        if shutil.which("docker") is None:
            parser.error("Docker is unavailable. Install/start Docker, or use the native cislunar-stack CLI.")
        outputs.mkdir(parents=True, exist_ok=True)
        print(f"Local output directory: {outputs}", flush=True)
        for cmd in commands:
            completed = subprocess.run(cmd, check=False)  # argument list, never shell evaluation
            if completed.returncode:
                return completed.returncode
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

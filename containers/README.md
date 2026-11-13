# Full-stack container

The product image runs `cislunar-stack --config /workspace/config.yaml`. All
simulation choices come from YAML, including IOD, estimation, synthetic generation,
prediction, and the output subdirectory. It includes Python, Java, Orekit, both
packages, and synthetic leap-second data. It does not force an IOD method.

Build and run with local bind mounts from the repository root:

```text
python scripts/run_container.py --build --output-dir ./outputs/container
```

For external local files:

```powershell
python scripts/run_container.py --config "C:\My Experiment\config.yaml" --input-dir "D:\Sensor Data" --output-dir "D:\Simulation Results"
```

The config file is mounted at `/workspace/config.yaml` (read-only), local inputs
at `/workspace/data` (read-only), and local outputs at `/workspace/outputs`
(read/write). `output.directory: outputs/example` therefore writes to the host
output directory's `example` subfolder. `--dry-run` prints the commands without
running Docker. The launcher needs only standard-library Python on the host.

For direct Docker usage, build from the repository root:

```text
docker build -f containers/Dockerfile -t cislunar-stack:local .
```

Then mount a config, data directory, and output directory as above. Running the
image without a mounted configuration fails rather than running a hidden default
scenario. The build itself uses `containers/smoke.yaml` to check the full chain.

See [the complete configuration and bind-mount guide](../docs/configuration.md).

**Pinned inputs.** The Python base image and the Java 21 runtime (Eclipse Temurin) are pinned by
digest in the Dockerfile, and every Python package by exact version in `requirements.lock`; bump
the image tag whenever one of them changes. Each run's `summary.json` records the image tag
(`CISLUNAR_IMAGE`), Python, package versions and machine. Only `libgomp1` still comes from
Debian's package archive at build time.

**Networks that re-sign TLS** (common on corporate and government networks): pass the network's CA
bundle as a build secret; it is used for the package install only and is not stored in the image:
`docker build --secret id=pip_cert,src=/path/to/ca.pem -f containers/Dockerfile -t cislunar-stack:local .`

**Validated (November 2026, x86, Docker 29):** the image builds, its smoke run passes, and
`scripts/run_container.py` on `scenarios/plug_and_play/config.yaml` gives the same numbers as a
native run (IOD 18.3 km, final 15.1 km).

## Optional research image

The standalone numerical solver image retains its independent CSV-based CLI:

```text
docker build -f containers/Dockerfile.solver -t collocation-iod .
docker run --rm collocation-iod
```

Its optional `--target orekit` includes Java interoperability. Use the full-stack
image for the YAML-only product. The original Orekit tutorials remain in
`examples/orekit_tutorials`.

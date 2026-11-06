# cislunar-od: the orbit determination stack in one container image

One image holds the whole Python stack in `afrl_cislunar/` and everything it needs: Python 3.13,
a Java 21 runtime for Orekit, pinned Python packages, and pinned orekit-data (Earth orientation,
leap seconds, DE-440 ephemerides). Nobody has to install Java, a virtual environment or
orekit-data to run it.

```
docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 run
```

That runs the stack on the case folder `./case`. If the folder is empty, the example case
(`config.yaml` plus `data/`) is copied in first. Results land in `./case/results/`.

## Build

From the repository root (the build needs `afrl_cislunar/` and this folder):

```
docker build -f docker/od-stack/Dockerfile -t cislunar-od:0.1.0 \
    --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
```

It takes a few minutes the first time and produces a 252 MB download (about 1 GB unpacked).
Any branch builds the same way: the image contains whatever `afrl_cislunar/` holds on the branch
you build from.

The image is built for the machine's own processor (Apple silicon or x86). To publish one image
that runs on both, use buildx and push to a registry the team can reach:

```
docker buildx build --platform linux/amd64,linux/arm64 -f docker/od-stack/Dockerfile \
    -t ghcr.io/<owner>/cislunar-od:0.1.0 --push .
```

## Commands

| Command | What it does |
|---|---|
| `run [CONFIG] [OPTIONS]` | Runs `python -m od_stack.run` on a case. CONFIG defaults to `config.yaml` in the case folder; OPTIONS are passed on (`--iod range_search`, `--set data.noise_seed=3`, ...). Paths in the config are relative to the config's folder. |
| `init` | Copies the example `config.yaml` and `data/` into the case folder. Never overwrites a file. |
| `test` | Runs the test suite inside the image. |
| `info` | Prints the image version, git commit, library versions and machine as JSON. |
| anything else | Runs that command in the case folder, e.g. `python -m od_stack.run --compare-iod`. |

Examples:

```
docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 init                      # start a new study
docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 run --iod gooding
docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 run noisy.yaml            # another config in ./case
docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 python -m od_stack.run --compare-iod
docker run --rm cislunar-od:0.1.0 test
```

## The case folder

A case is a folder with a `config.yaml` and the data it points to. A new study is a new folder,
not a new image. The case folder is the only place the container writes to:

- `results/`: the stack's outputs (`summary.json`, `trajectory_comparison.csv`, `results.png`,
  `config_used.yaml`) and `run.json`.
- `runs/<time>.json`: the same record for commands other than `run`.

`run.json` records the command, exit code, start time, wall time, image version, git commit,
orekit-data commit, Python, Java and library versions, and the CPU architecture and core count.
Results from Apple silicon and x86 can differ in the last digits, and the later runtime study
needs to know which machine produced each number.

The noise is seeded from the config (`data.noise_seed` with `data.source: resample_truth`), so the
same case gives the same numbers on every run.

## What is pinned

| What | Where | How to update |
|---|---|---|
| Python base image (3.13, Debian bookworm) | `PYTHON_IMAGE` in the Dockerfile, by digest | change the digest |
| Java 21 runtime (Eclipse Temurin) | `JAVA_IMAGE` in the Dockerfile, by digest | change the digest |
| Python packages | `requirements.lock` (exact versions) | edit `afrl_cislunar/requirements.txt`, run `docker/od-stack/lock.sh` |
| orekit-data | `OREKIT_DATA_COMMIT` and its `OREKIT_DATA_SHA256` | new commit and checksum from gitlab.orekit.org/orekit/orekit-data |

Bump the image version (`IMAGE_VERSION`, and the tag) whenever one of these changes. Never build
or run from a `latest` tag.

## On NASA and DoD clusters (Apptainer)

HPC centers don't allow Docker; they run Apptainer (formerly Singularity), which runs the same
image unchanged:

```
apptainer build cislunar-od_0.1.0.sif docker-daemon://cislunar-od:0.1.0     # or docker://<registry>/...
apptainer run --bind "$PWD/case:/case" cislunar-od_0.1.0.sif run
```

Apptainer runs the image read-only as your own user. The image is built for that: the code is
read-only, it runs as a non-root user, caches go to `/tmp`, and the only writes go to the case
folder (`/case`, or the current directory when `/case` isn't bound). This was tested with Docker
using `--read-only --user 54321:54321`, which reproduces those conditions. Apptainer itself has not
been tested yet.

## Behind a proxy that re-signs TLS

If `pip install` fails during the build with certificate errors (common on corporate and
government networks), pass the network's CA bundle as a build secret. It is used for that one
step and never stored in the image:

```
docker build --secret id=pip_cert,src=/path/to/ca-bundle.pem -f docker/od-stack/Dockerfile -t cislunar-od:0.1.0 .
```

## Tested

On x86 (linux/amd64), Docker 29:

- The image builds from `main` and from `cr3bp-iod` (which holds the convergence study and the
  CR3BP IOD methods). `test` passes inside the image on both: 15 tests on `main`, 146 on
  `cr3bp-iod`.
- `run` on an empty case folder reproduces the stack's reference result (15.1 km at the epoch,
  52.9 km maximum over 8 days). Two runs in the container and one run outside it give identical
  numbers in every field.
- A read-only image with an arbitrary user ID (the Apptainer conditions) works.

Not tested yet: a build on Apple silicon (linux/arm64) and a real Apptainer run. Both base images
and every Python package are published for arm64, so the arm64 build should work.

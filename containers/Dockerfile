# syntax=docker/dockerfile:1.7
# Complete synthetic simulation stack. Build from REPOSITORY ROOT.
# docker build -f containers/Dockerfile -t cislunar-stack:local .
# Includes the leap-second data required by this synthetic J2000 scenario.

# Base images pinned by digest (amd64 and arm64), so every build starts from the same bytes.
# Override with --build-arg (e.g. a registry mirror), keeping the digest.
ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72
ARG JAVA_IMAGE=eclipse-temurin:21-jre@sha256:0b99b3376fbb50dbf9c0af8535fdd5b8eb0d3fdfc8b1e4967f383b6919f6901a

# Java 21 runtime for Orekit, taken from the pinned Temurin image
FROM ${JAVA_IMAGE} AS java

FROM ${PYTHON_IMAGE}
# The image name each run records in summary.json (provenance). Pass the same name as -t:
#   docker build --build-arg IMAGE_TAG=my-name -t my-name ...   (scripts/run_container.py --build does this)
ARG IMAGE_TAG=cislunar-stack:local
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib OREKIT_DATA=/opt/orekit-data \
    CISLUNAR_OUTPUT_ROOT=/workspace/outputs CISLUNAR_IMAGE=${IMAGE_TAG} \
    JAVA_HOME=/opt/java/openjdk PATH=/opt/java/openjdk/bin:$PATH
COPY --from=java /opt/java/openjdk /opt/java/openjdk
# libgomp1: OpenMP runtime used by CasADi's solvers
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Exact package versions. The optional pip_cert secret is only for networks whose proxy re-signs
# TLS traffic: docker build --secret id=pip_cert,src=/path/to/ca.pem ...
COPY containers/requirements.lock /tmp/requirements.lock
RUN --mount=type=secret,id=pip_cert,required=false \
    if [ -s /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi \
    && python -m pip install --no-cache-dir -r /tmp/requirements.lock
COPY packages/collocation-iod/pyproject.toml packages/collocation-iod/README.md /opt/collocation-iod/
COPY packages/collocation-iod/src /opt/collocation-iod/src
RUN python -m pip install --no-cache-dir --no-deps /opt/collocation-iod
COPY packages/cislunar-stack/pyproject.toml packages/cislunar-stack/README.md /opt/cislunar-stack/
COPY packages/cislunar-stack/src /opt/cislunar-stack/src
RUN python -m pip install --no-cache-dir --no-deps /opt/cislunar-stack
COPY containers/smoke.yaml /opt/smoke.yaml
COPY packages/collocation-iod/data/orekit /opt/orekit-data
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/outputs
RUN CISLUNAR_OUTPUT_ROOT=/tmp/cislunar-smoke cislunar-stack --config /opt/smoke.yaml
ENTRYPOINT ["cislunar-stack"]
CMD ["--config", "/workspace/config.yaml"]

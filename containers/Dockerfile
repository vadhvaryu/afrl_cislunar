# Complete synthetic simulation stack. Build from REPOSITORY ROOT.
# docker build -f containers/Dockerfile -t cislunar-stack .
# Includes the leap-second data required by this synthetic J2000 scenario.
FROM python:3.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib OREKIT_DATA=/opt/orekit-data \
    CISLUNAR_OUTPUT_ROOT=/workspace/outputs
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY packages/collocation-iod/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock \
    "orekit-jpype==13.1.9.0" "matplotlib>=3.9,<4" "pyyaml>=6,<7"
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

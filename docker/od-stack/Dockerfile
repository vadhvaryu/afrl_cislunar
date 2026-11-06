# syntax=docker/dockerfile:1.7
#
# cislunar-od: the Python orbit determination stack (afrl_cislunar/) with everything it needs:
# Python 3.13, a Java 21 runtime for Orekit (orekit-jpype), pinned Python packages and pinned
# orekit-data (Earth orientation, leap seconds, DE-440 ephemerides).
#
# Build from the repository root (the build context must contain afrl_cislunar/):
#   docker build -f docker/od-stack/Dockerfile -t cislunar-od:0.1.0 \
#       --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
# Run (results land in ./case):
#   docker run --rm -v "$PWD/case:/case" cislunar-od:0.1.0 run
# See docker/od-stack/README.md.

# Base images, pinned by digest: the same bytes on every machine (multi-arch: amd64 and arm64).
# Override (e.g. with a registry mirror) with --build-arg; keep the digest.
ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72
ARG JAVA_IMAGE=eclipse-temurin:21-jre@sha256:0b99b3376fbb50dbf9c0af8535fdd5b8eb0d3fdfc8b1e4967f383b6919f6901a

# orekit-data, pinned to one commit of the official repository and checked against its sha256.
ARG OREKIT_DATA_COMMIT=9de5188d8410c0c898cc2ec4b79c4da24b11df07
ARG OREKIT_DATA_SHA256=8fd6727b5e9121a3ada5ad6fbe3d4c8ed68c0b167698ffef72c55bc653825346

# ---------- Java runtime (only the JRE folder is copied into the final image) ----------
FROM ${JAVA_IMAGE} AS java

# ---------- orekit-data ----------
FROM ${PYTHON_IMAGE} AS orekit-data
ARG OREKIT_DATA_COMMIT
ARG OREKIT_DATA_SHA256
ADD --checksum=sha256:${OREKIT_DATA_SHA256} \
    https://gitlab.orekit.org/orekit/orekit-data/-/archive/${OREKIT_DATA_COMMIT}/orekit-data-${OREKIT_DATA_COMMIT}.tar.gz \
    /tmp/orekit-data.tar.gz
RUN mkdir -p /opt/orekit-data \
 && tar -xzf /tmp/orekit-data.tar.gz -C /opt/orekit-data --strip-components=1 \
 && rm -rf /opt/orekit-data/_pyinstaller /opt/orekit-data/update.sh \
 && echo "${OREKIT_DATA_COMMIT}" > /opt/orekit-data/COMMIT

# ---------- final image ----------
FROM ${PYTHON_IMAGE}
ARG IMAGE_VERSION=0.1.0
ARG GIT_COMMIT=unknown
ARG OREKIT_DATA_COMMIT

ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    OREKIT_DATA=/opt/orekit-data \
    PYTHONPATH=/app/afrl_cislunar \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    CISLUNAR_OD_VERSION=${IMAGE_VERSION} \
    CISLUNAR_OD_COMMIT=${GIT_COMMIT} \
    OREKIT_DATA_COMMIT=${OREKIT_DATA_COMMIT}

COPY --from=java /opt/java/openjdk /opt/java/openjdk
COPY --from=orekit-data /opt/orekit-data /opt/orekit-data

# Pinned Python packages. The optional "pip_cert" secret is only for networks whose proxy
# re-signs TLS traffic: docker build --secret id=pip_cert,src=/path/to/ca.pem ...
COPY docker/od-stack/requirements.lock /tmp/requirements.lock
RUN --mount=type=secret,id=pip_cert,required=false \
    if [ -s /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi \
 && pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock

# The code, read-only for the user that runs it. The case folder (/case) is the only place written to.
COPY afrl_cislunar /app/afrl_cislunar
COPY --chmod=0755 docker/od-stack/entrypoint.py /usr/local/bin/cislunar-od
RUN useradd --uid 10001 --create-home --home-dir /home/od --shell /usr/sbin/nologin od \
 && mkdir -p /case && chmod 1777 /case

LABEL org.opencontainers.image.title="cislunar-od" \
      org.opencontainers.image.description="Cislunar orbit determination stack (CR3BP, Orekit batch least squares)" \
      org.opencontainers.image.source="https://github.com/vadhvaryu/afrl_cislunar" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

USER 10001
WORKDIR /case
ENTRYPOINT ["cislunar-od"]
CMD ["help"]

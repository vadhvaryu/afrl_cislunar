#!/bin/sh
# Re-resolve requirements.lock from afrl_cislunar/requirements.txt, inside the same Python base
# image the Dockerfile uses, so the pinned versions are the ones that install in the image.
# Run from the repository root:  docker/od-stack/lock.sh
set -eu
BASE="${BASE:-python:3.13-slim-bookworm}"
OUT=docker/od-stack/requirements.lock
{
  sed -n '1,4p' "$OUT"
  docker run --rm -v "$PWD/afrl_cislunar/requirements.txt:/r.txt:ro" "$BASE" sh -c '
    pip install -q --root-user-action=ignore --disable-pip-version-check -r /r.txt >/dev/null &&
    echo "pip==$(pip --version | cut -d" " -f2)" && pip freeze --all | grep -v "^pip=="'
} > "$OUT.new"
mv "$OUT.new" "$OUT"
echo "updated $OUT"

#!/usr/bin/env bash
# Build the image from a local copy of the model. Usage: build.sh <Qwen3.8-27B-Q4_K_M.gguf> <image:tag> [--push]
# Needs about 60 GB of free disk. Stop anything that loads a model first: this reads and writes 17 GB files.
set -euo pipefail

MODEL_SHA256=7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b
MODEL_BYTES=17106773984
PART_COUNT=4

[ $# -ge 2 ] || { echo "Usage: $0 <Qwen3.8-27B-Q4_K_M.gguf> <image:tag> [--push]" >&2; exit 2; }
model="$1"; image="$2"; mode="${3:---load}"
case "$mode" in --load|--push) ;; *) echo "The third argument is --push or nothing." >&2; exit 2 ;; esac

if pgrep -f 'mk_server|llama-server' >/dev/null; then
    echo "A model server is running. Stop it first: building alongside it can kill a small WSL machine." >&2
    exit 1
fi

here="$(cd "$(dirname "$0")" && pwd)"
context="${OPENJET_BUILD_DIR:-$HOME/.cache/openjet-cloud-build}"
free_gb="$(df --output=avail -BG "$(dirname "$context")" 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)"
[ "${free_gb:-0}" -ge 60 ] || { echo "Need 60 GB free for the build; found ${free_gb:-0} GB." >&2; exit 1; }

[ "$(stat -c %s "$model")" = "$MODEL_BYTES" ] || { echo "$model is not the expected $MODEL_BYTES bytes." >&2; exit 1; }
echo "Checking the model's SHA-256..."
echo "$MODEL_SHA256  $model" | sha256sum --check --quiet

rm -rf "$context"
mkdir -p "$context"
trap 'rm -rf "$context"' EXIT
cp "$here"/{Dockerfile,auth_proxy.py,entrypoint.sh,openjet-agent,llama-server-unavailable,config.yaml,NOTICE} "$context"/
split -b 4G -d -a 2 "$model" "$context/model.part-"
[ "$(find "$context" -name 'model.part-*' | wc -l)" = "$PART_COUNT" ] || { echo "Expected $PART_COUNT parts; update the Dockerfile." >&2; exit 1; }

if [ "$mode" = "--push" ]; then
    # The weights don't compress, so spend as little time on it as gzip allows. gzip is what every host can pull.
    docker buildx build --target full --tag "$image" --provenance=false \
        --output "type=registry,compression=gzip,compression-level=1,force-compression=true" "$context"
else
    docker build --target full --provenance=false --tag "$image" "$context"
fi

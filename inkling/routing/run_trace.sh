#!/usr/bin/env bash
# Record Inkling 2 Mini expert routing for the workload in workload.py.
# Needs llama.cpp built with llama-route-trace.patch (see README.md).
set -euo pipefail

LLAMA=${LLAMA:-/home/louis/bench-engines/llama.cpp-inkling/build/bin/llama-server}
MODEL=${MODEL:-/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf}
OUT=${OUT:-$(dirname "$0")/results/$(date +%F)-q4kxl}
PORT=${PORT:-8099}

mkdir -p "$OUT"
TRACE="$OUT/trace.bin"
: > "$TRACE"
rm -f "$OUT/requests.jsonl"

# the server holds the model in VRAM; the cap keeps a runaway host allocation from taking down WSL
LLAMA_ROUTE_TRACE="$TRACE" systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  "$LLAMA" -m "$MODEL" -ngl 99 -c 16384 -ub 512 -np 1 --port "$PORT" --host 127.0.0.1 \
  --no-warmup > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "server exited"; tail -20 "$OUT/server.log"; exit 1; fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not healthy"; exit 1; }
# the load itself may have run graph evaluations; requests are cut by byte range anyway
python3 "$(dirname "$0")/workload.py" --url "http://127.0.0.1:$PORT" --trace "$TRACE" --out "$OUT/requests.jsonl" "$@"
echo "trace: $TRACE ($(du -h "$TRACE" | cut -f1))"

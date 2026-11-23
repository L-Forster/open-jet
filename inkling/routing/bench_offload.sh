#!/usr/bin/env bash
# llama.cpp decode speed for Inkling 2 Mini with part of the experts on the CPU, simulating smaller GPUs.
# Only expert placement changes; dense weights stay on the GPU. Expert sizes (UD-Q4_K_XL): 573 MB per layer
# for layers 0-19, 1083 MB for 20-23.
set -uo pipefail

LLAMA=${LLAMA:-/home/louis/bench-engines/llama.cpp-inkling/build/bin/llama-server}
MODEL=${MODEL:-/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf}
OUT=${OUT:-$(dirname "$0")/results/$(date +%F)-offload}
PORT=${PORT:-8098}
mkdir -p "$OUT"
touch "$OUT/results.txt"

exps() { echo "blk\\.($1)\\.ffn_.*_exps.*=CPU"; }

declare -a NAMES ARGS
NAMES+=("all-gpu");                     ARGS+=("")
# 12 GB card: ~6.5 GB of experts on the CPU
NAMES+=("12gb-ncmoe11");                ARGS+=("--n-cpu-moe 11")
NAMES+=("12gb-loopaware");              ARGS+=("-ot $(exps '0|1|18|19|20|21|22|23')")
# 8 GB card: ~10.3 GB of experts on the CPU
NAMES+=("8gb-ncmoe18");                 ARGS+=("--n-cpu-moe 18")
NAMES+=("8gb-loopaware");               ARGS+=("-ot $(exps '0|1|2|3|4|5|16|17|18|19|20|21|22|23')")
# llama.cpp's own expert cache (--moe-cache-mib). All-host placement (--cpu-moe) needs the 15.8 GB of experts in
# RAM, which a 15 GB machine cannot hold (the page cache thrashes to the SSD, ~3 tok/s), so these keep the loop
# layers 6-17 on the GPU (6.9 GB) and the other 8.9 GB on the host: without a cache, then with the rest of a
# 12 GB budget (2.4 GB) as cache
NAMES+=("12gb-core-static");            ARGS+=("-ot $(exps '0|1|2|3|4|5|18|19|20|21|22|23')")
NAMES+=("12gb-core-moecache");          ARGS+=("-ot $(exps '0|1|2|3|4|5|18|19|20|21|22|23') --moe-cache-mib 2400")

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  if [ -n "${ONLY:-}" ] && ! [[ " $ONLY " == *" $name "* ]]; then continue; fi
  log="$OUT/$name.log"
  # mmap keeps CPU-side experts in the page cache, which the cap may reclaim but never turns into an OOM kill
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
    "$LLAMA" -m "$MODEL" -ngl 99 -c 8192 -np 1 -lv 4 --port "$PORT" --host 127.0.0.1 ${ARGS[$i]} > "$log" 2>&1 &
  srv=$!
  ok=0
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    python3 "$(dirname "$0")/workload.py" --url "http://127.0.0.1:$PORT" --trace /dev/null --out "$OUT/$name.jsonl" \
      --only cuda_kernel general_explain > /dev/null 2>&1
  fi
  kill -INT $srv 2>/dev/null; wait $srv 2>/dev/null  # SIGINT: a clean exit prints the MoE cache statistics
  if [ $ok = 1 ]; then
    rates=$(grep -E "^.*eval time" "$log" | grep -v prompt | sed -E 's/.*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)/\2/' | tr '\n' ' ')
    cache=$(grep -E "llama_moe_cache: .*hit rate" "$log" | tail -1 | sed -E 's/.*llama_moe_cache: //')
    echo "$name | decode tok/s: $rates | ${cache}" | tee -a "$OUT/results.txt"
  else
    echo "$name | failed to start" | tee -a "$OUT/results.txt"; tail -5 "$log"
  fi
  sleep 3
done

#!/usr/bin/env bash
# Speed and accuracy comparison of the current OpenJet engine (origin/master build in ../open-jet-bench)
# with llama.cpp b11982 on this machine.  Run it with nothing else on the GPU:
# it refuses to start when the GPU is busy (FORCE=1 overrides).  About 2 hours.
#
#   megakernel/bench/run_compare.sh [OUT_DIR]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH=/home/louis/open-jet-bench/megakernel
MODEL=/home/louis/open-jet/models/Qwen3.8-27B-Q4_K_M.gguf
LLAMACPP=/home/louis/bench-engines/llama.cpp-b11982/build/bin/llama-server
LLAMA_REF=/home/louis/open-jet/megakernel/tools/llama_ref
TOKENIZER=/home/louis/bench-engines/qwen_tok/tokenizer.json
OUT="${1:-$HERE/results/$(date +%F)-current}"
CTX=106496
mkdir -p "$OUT"

# 1. the GPU must be idle (a game or video on the Windows side swings results by up to 50%)
util=$(for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 1; done |
       awk '{s += $1} END {printf "%d", s / NR}')
echo "GPU utilisation before start: ${util}%"
if [ "$util" -gt 15 ] && [ "${FORCE:-0}" != 1 ]; then
    echo "The GPU is busy (${util}%). Close games/video and rerun, or set FORCE=1." >&2
    exit 1
fi

# 2. speed: 8 seeds in 4 interleaved rounds, short prompts and a ~98K-token prompt
MK="python3 $BENCH/server/mk_server.py -m $MODEL --port 8094 -c $CTX --kv q8 -d 4 --lib $BENCH/libqwen38mk.so"
COMMON="-m $MODEL --host 127.0.0.1 --port 8094 --parallel 1 -ngl 99 -c $CTX --jinja"
python3 "$HERE/engine_compare.py" --seeds 8 --rounds 4 -n 1024 --tokenizer "$TOKENIZER" --output "$OUT/speed.json" \
    --engine "openjet-attn8=8094=$MK" \
    --engine "openjet-attn2=8094=env MK_ATTN_S8=0 $MK" \
    --engine "llama.cpp-b11982=8094=$LLAMACPP $COMMON --flash-attn on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2" \
    2>&1 | tee "$OUT/speed.console.txt"

# 3. accuracy against llama.cpp generating token by token (README "Correctness"): short contexts and 96K
python3 - "$OUT" "$TOKENIZER" <<'EOF'
import glob, os, sys
from tokenizers import Tokenizer
out, tok = sys.argv[1], Tokenizer.from_file(sys.argv[2])
root = "/home/louis/open-jet-bench"
def take(paths, n):
    text = ""
    for p in paths:
        text += "\n\n" + open(p, errors="replace").read()
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= n, (paths[:2], len(ids), n)
    return tok.decode(ids[:n])
code = sorted(glob.glob(f"{root}/megakernel/*.cu*") + glob.glob(f"{root}/megakernel/*.h") + glob.glob(f"{root}/src/*.py"))
prose = sorted(glob.glob(f"{root}/docs/**/*.md", recursive=True) + [f"{root}/README.md", f"{root}/megakernel/README.md"])
open(f"{out}/code.txt", "w").write(take(code, 7 * 2048))
open(f"{out}/prose.txt", "w").write(take(prose, 6 * 2048))
open(f"{out}/long.txt", "w").write(take(code, 96000))
EOF
{
    echo "=== reference: llama.cpp token by token, q8_0 KV, 2048-token chunks scored from 1024"
    "$LLAMA_REF" -m "$MODEL" -f "$OUT/code.txt" -f "$OUT/prose.txt" --score-from 1024 -b 1 --ref-out "$OUT/short.ref"
    echo "=== reference: 96K, scored from 94000"
    "$LLAMA_REF" -m "$MODEL" -f "$OUT/long.txt" --chunk 96000 --score-from 94000 -b 1 --ref-out "$OUT/long.ref"
    for ref in short long; do
        ctx=$([ $ref = long ] && echo 98304 || echo 4096)
        echo "=== llama.cpp batched vs reference ($ref): the noise floor"
        "$LLAMA_REF" -m "$MODEL" --kld "$OUT/$ref.ref"
        for v in "pk_attn8 (default):" "pk_attn2:MK_ATTN_S8=0" "pk_attn (old):MK_ATTN=1"; do
            echo "=== megakernel ${v%%:*} vs reference ($ref)"
            env ${v#*:} "$BENCH/qwen38_mk" -m "$MODEL" -c $ctx --kv q8 --kld "$OUT/$ref.ref" --score-path mixed
        done
    done
} 2>&1 | grep -vE "^llama_model_loader|^load_tensors|^print_info|^llama_context|^llama_kv|^graph_|^sched_" | tee "$OUT/accuracy.txt"
echo "Done: $OUT"

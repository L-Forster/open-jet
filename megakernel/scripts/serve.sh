#!/usr/bin/env bash
# Serve Qwen3.8-27B through the megakernel's OpenAI-compatible server (Linux / WSL2). Usage: scripts/serve.sh [port] [ctx]
# Defaults: port 8080, 120,000 tokens of context, q8_0 KV cache, 7 MTP drafts, exact reuse on, images on when the mmproj
# file and libqwen38vision.so exist. Settings (environment):
#   MK_MODEL            the GGUF (default ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; Q4_K_M runs on the same engine). Keep it
#                       on a Linux filesystem under WSL2: /mnt/c or /mnt/g paths load several times slower.
#   MK_MMPROJ           the vision projector (default: mmproj-BF16.gguf, unsloth's file name, next to the model or in
#                       ~/models, else ~/models/mmproj-Qwen3.8-27B-BF16.gguf; MK_MMPROJ= turns images off)
#   MK_IMAGE_MAX_TOKENS tokens per image (default: the mmproj's, 4096); fewer tokens encode faster
#   MK_VIDEO_FPS        video input (input_video parts, with the mmproj; needs ffmpeg: on PATH, MK_FFMPEG, or
#                       pip install imageio-ffmpeg): frames sampled per second (default 4, llama.cpp's)
#   MK_VIDEO_MAX_TOKENS tokens per video frame pair (default: the image limit); fewer fit longer clips
#   MK_FFMPEG           the ffmpeg executable (or its directory) for video input
#   MK_HOST             listen address (default 127.0.0.1; 0.0.0.0 for other machines on the LAN)
#   MK_ALLOW_HOSTS      host names clients use besides localhost, IP addresses and MK_HOST (comma-separated; * = any):
#                       requests for other names get a 403 (DNS rebinding)
#   MK_CORS_ORIGINS     web origins whose pages may call the server (comma-separated, e.g. http://localhost:5173):
#                       requests from pages of other origins get a 403
#   MK_API_KEY          require this key on every request but /health (clients send it as their API key); default none
#   MK_DRAFT            MTP drafts per cycle (default 7)
#   MK_EXACT_REUSE=0    turn exact reuse off: a reused prompt then appends the previous reply instead of re-prefilling it
#                       (faster follow-up turns, but not bit-identical to the same request sent fresh)
#   MK_SIDE_MAX         side requests: a first-turn prompt of at most this many tokens (default 8192; 0 = off) that would
#                       discard a much longer cached conversation runs without evicting it (the KV rows it overwrites are
#                       kept in host memory and come back after its reply; with parking on: one without tools, such as
#                       a client's title or summary request, or with max_tokens <= 1024, and any later request without
#                       tools beside an agent's conversation; its own conversation is parked too, so its next turn
#                       continues beside the cached one)
#   MK_PARK_MIB         host memory for parked conversations (default 8192 MiB, allocated on first use; 0 = off): a request
#                       for another conversation parks the cached one, and it comes back bit for bit with its next request
#   MK_PF_CAP           prefill chunk tokens (default 2048; 1024 above 96k context: 0.34 GB less VRAM, prefill ~0.2-0.45% slower)
#   MK_KV, MK_KV_WIN    low-bit KV cache (opt-in), preset D': MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048 (6-bit K, 5-bit V,
#                       8-bit MTP layer and recent window: ~1.0 GB less VRAM at 120k, ~162k context in the same VRAM,
#                       up to 262,144 tokens on 24 GB; near-lossless, see README)
#   MK_KEEP_WARM_S      seconds the engine keeps the GPU's clocks up after each request (default 120, 0 = off)
#   MK_DRAFT_Q4=0       the draft head reads lm_head instead of its Q4_K copy (saves 90 MiB; the copy is on by default)
#   MK_MTP              the MTP draft layer from a separate GGUF of the same model (e.g. an mtp-*.gguf), for a model
#                       file without one or instead of the file's own
#   MK_NO_MTP=1         no MTP layer: plain decoding, depth 0 (also automatic for a model file without one)
#   MK_CHAT_TEMPLATE_FILE  a Jinja chat template file used instead of the GGUF's (--chat-template-file, as llama-server)
#   MK_CHAT_TEMPLATE_KWARGS  defaults under every request's chat_template_kwargs, e.g. '{"enable_thinking": false}'
#   MK_CONTENT_SPECIALS special-token text (<|im_end|>, ...) inside messages: text (default: tokenized as text, so content
#                       cannot forge turns) or parse (as the special tokens, as llama-server does)
#   MK_REASONING_FORMAT auto (default: the thinking as reasoning_content), none (left in the content), deepseek-legacy
#   MK_MAX_BODY_MB, MK_MAX_IMAGES, MK_IMAGE_MAX_PIXELS, MK_MAX_VIDEO_FRAMES
#                       request limits: body size (default 128 MB), images per request (32), pixels per image
#                       (100,000,000; checked before decoding), video frames per request (256)
#   MK_MAX_INFLIGHT_MB, MK_MAX_CONNECTIONS
#                       request bodies in memory at once (default 256 MB), open connections (256): past them a 503
#   MK_TIMEOUT_S        seconds a connection may make no progress before it closes (default 600, 0 = never)
#   MK_REQUEST_TIMEOUT_S seconds a request may take to arrive before its connection closes (default 300, 0 = no limit)
#   MK_PYTHON           Python with jinja2 (default python3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8080}"
CTX="${2:-120000}"
MODEL="${MK_MODEL:-$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
PY="${MK_PYTHON:-python3}"
[ -f "$HERE/libqwen38mk.so" ] || { echo "missing $HERE/libqwen38mk.so: run make (see README)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "missing model $MODEL (set MK_MODEL)" >&2; exit 1; }
if [ "$CTX" -gt 96000 ]; then export MK_PF_CAP="${MK_PF_CAP:-1024}"; fi
XR=(--exact-reuse) XS=on
[ "${MK_EXACT_REUSE:-1}" = 0 ] && XR=() XS=off
CT=()
[ -n "${MK_CHAT_TEMPLATE_FILE:-}" ] && CT=(--chat-template-file "$MK_CHAT_TEMPLATE_FILE")
if [ -n "${MK_MMPROJ+x}" ]; then MMPROJ="$MK_MMPROJ"
else
    MMPROJ=
    for f in "$(dirname "$MODEL")/mmproj-BF16.gguf" "$HOME/models/mmproj-BF16.gguf" "$HOME/models/mmproj-Qwen3.8-27B-BF16.gguf"; do
        if [ -f "$f" ]; then MMPROJ="$f"; break; fi
    done
fi
VA=() VS=off
if [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ] && [ -f "$HERE/libqwen38vision.so" ]; then
    VA=(--mmproj "$MMPROJ" ${MK_IMAGE_MAX_TOKENS:+--image-max-tokens "$MK_IMAGE_MAX_TOKENS"}
        ${MK_VIDEO_FPS:+--video-fps "$MK_VIDEO_FPS"} ${MK_VIDEO_MAX_TOKENS:+--video-max-tokens "$MK_VIDEO_MAX_TOKENS"})
    VS="on the CPU ($(basename "$MMPROJ"))"
fi
echo "megakernel: $MODEL, ctx $CTX, KV q8_0${MK_KV:+ + MK_KV=$MK_KV (window ${MK_KV_WIN:-off})}, ${MK_DRAFT:-7} MTP drafts," \
     "prefill chunk ${MK_PF_CAP:-2048}, exact reuse $XS, images $VS${MK_CHAT_TEMPLATE_FILE:+, chat template $MK_CHAT_TEMPLATE_FILE}," \
     "http://${MK_HOST:-127.0.0.1}:$PORT"
exec "$PY" "$HERE/server/mk_server.py" -m "$MODEL" --host "${MK_HOST:-127.0.0.1}" --port "$PORT" -c "$CTX" --kv q8 \
    -d "${MK_DRAFT:-7}" "${XR[@]}" "${VA[@]}" "${CT[@]}"

"""Request limits and defaults. main() rebinds MAX_BODY_MB, MAX_INFLIGHT_MB, MAX_CONNECTIONS, REQUEST_TIMEOUT_S,
MAX_IMAGES, MAX_VIDEO_FRAMES, IMAGE_MAX_PIXELS and REASONING_MAX_BYTES, so those are read as mk_limits.NAME, never
imported by name."""
import os
import sys

TOPK_MAX = 32
# sampling defaults (a request's absent / null fields): Unsloth's thinking-mode preset for Qwen3.8
DEF_TEMP, DEF_TOP_P, DEF_TOP_K, DEF_MIN_P = 1.0, 0.95, 20, 0.0
# request limits; main() reads them from the environment
MAX_BODY_MB = 128  # MK_MAX_BODY_MB: request body (several screenshots as base64 data: URLs fit)
MAX_INFLIGHT_MB = 256  # MK_MAX_INFLIGHT_MB: request bodies in memory at once, all connections (at least MK_MAX_BODY_MB)
MAX_CONNECTIONS = 256  # MK_MAX_CONNECTIONS: open connections (each holds a thread)
REQUEST_TIMEOUT_S = 300  # MK_REQUEST_TIMEOUT_S: a request's headers and body arrive within this long of its first byte
FLT_MAX = 3.4028234663852886e38  # (the engine takes sampling values as fp32)
WARN_KEYS, WARN_KEY_LEN, WARN_TEXT_LEN = 256, 128, 2000  # warn_once: its keys and texts can come from a request
MAX_IMAGES = 32  # MK_MAX_IMAGES: images per request
MAX_STOP, MAX_STOP_LEN = 64, 1024  # stop strings per request, characters per stop string
MAX_VIDEO_FRAMES = 256  # MK_MAX_VIDEO_FRAMES: video frames per request (all of its videos; 64 s at 4 fps)
IMAGE_MAX_PIXELS = 100_000_000  # MK_IMAGE_MAX_PIXELS (the vision library's too): larger video frames are not decoded
VIDEO_DECODE_S = 300  # a video's ffmpeg decode is stopped after this many seconds (400)
TIMEOUT_S = 600  # MK_TIMEOUT_S: a read or write stalled this long closes the connection (llama-server's --timeout)
# a stream's write that makes no progress this long: its client stopped reading and is gone (MK_TIMEOUT_S would hold the
# engine, and every request waiting for it, that long)
STREAM_WRITE_S = 60
VIDEO_TS_MS = 5000  # llama.cpp's video timestamp interval (mtmd_helper_video_init_params_default)
# lingering close (HTTPServer.shutdown_request): what a client still sends after the server's last answer is read and
# dropped for at most LINGER_S seconds / LINGER_BYTES bytes, LINGER_IDLE_S seconds without data (nginx's defaults)
LINGER_S, LINGER_IDLE_S, LINGER_BYTES = 30.0, 5.0, 1 << 30


# bytes all of a request's encrypted_content may inflate to: the model's full context x its longest token (set in main)
REASONING_MAX_BYTES = 262144 * 256


def env_int(name, default, minimum=1):
    """an integer setting >= minimum from the environment (default when unset or empty)"""
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    if not v.isdecimal() or int(v) < minimum:
        sys.exit(f"{name}: expected {'a positive integer' if minimum == 1 else f'an integer >= {minimum}'}, not {v!r}")
    return int(v)

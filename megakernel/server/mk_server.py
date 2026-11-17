#!/usr/bin/env python3
"""OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF, or the one given by --chat-template-file / --chat-template (llama-server's options), with
enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs and tools via `tools`;
reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured `tool_calls` (streamed as generated).
Sampling: temperature, top_p, top_k (1..32; <= 0: 32, the engine's maximum), min_p, seed; stop strings
(`stop`).  Successive requests of a conversation reuse the processed tokens: the new prompt is
matched against the sequence so far and only the rest is prefilled. When the new prompt diverges inside
what the engine already processed, it rewinds to the last DeltaNet checkpoint at or before the
divergence (saved during earlier prefills: at each generation prompt, each message start and every
--ck-every tokens; a pool of MK_CK_SLOTS, default 8).
POST /v1/messages (+ /v1/messages/count_tokens): the Anthropic Messages API over the same chat path.
POST /v1/responses (+ /v1/responses/input_tokens): the OpenAI Responses API over the same chat path, stateless (the client
sends the whole conversation each time, as Codex CLI does).
GET /health (503 "Loading model" until the model is loaded), /v1/models, /props, /slots, /metrics; POST /tokenize,
/detokenize, /apply-template, /v1/chat/completions/input_tokens (llama-server's shapes).  One request at a time; with
--agents N up to N at once, each in a parallel slot of the engine (its reply as when it runs alone; docs/SERVER.md).
After each request the engine keeps the GPU's clocks up for MK_KEEP_WARM_S seconds (default 120, 0 = off), so a
reply after a pause starts at full speed.
Limits (environment): MK_MAX_BODY_MB (request body, Content-Length or chunked, default 128), MK_MAX_IMAGES (images per
request, default 32),
MK_MAX_VIDEO_FRAMES (video frames per request, default 256), MK_TIMEOUT_S (a connection that sends and takes nothing
for this long is closed, default 600, 0 = never), MK_REQUEST_TIMEOUT_S (a request must arrive within this long of its
first byte, default 300, 0 = no limit), MK_MAX_INFLIGHT_MB (request bodies in memory at once, default 256: past it a
503), MK_MAX_CONNECTIONS (open connections, default 256: past it a 503); --api-key / MK_API_KEY requires that key on
every request but /health;
the image library refuses images over MK_IMAGE_MAX_PIXELS (default 100,000,000) before decoding them. A request
whose client disconnects stops at the next prefill chunk (2048 tokens) or engine cycle. A connection the server closes
while the client is still sending (an early 4xx for a body it does not read) is half-closed and drained first (bounded),
so the client reads the answer instead of a connection reset.
"""
import argparse
import json
import os
import struct
import sys
import threading
import traceback

import mk_limits
from mk_chat import REASONING_FORMATS, TEMPLATE_KWARGS
from mk_core import Server, SlotServer
from mk_gate import Gate, GatedHandler, exposure_warning
from mk_http import listen, make_handler, make_loading_handler
from mk_limits import TIMEOUT_S, env_int
from mk_log import CON, log
from mk_template import ChatTemplateArg, chat_template_source

HERE = os.path.dirname(os.path.abspath(__file__))


# the engine and encoder libraries next to the repo root: qwen38mk.dll / qwen38vision.dll on Windows
LIB_MK, LIB_VISION = ("qwen38mk.dll", "qwen38vision.dll") if os.name == "nt" else ("libqwen38mk.so", "libqwen38vision.so")


def gguf_text_limits(path):
    """(the GGUF's <arch>.context_length, the byte length of its longest token string), None where unreadable"""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # (GGUF scalar value types)
    ints = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    arch, ctxs, longest = None, {}, None
    try:
        with open(path, "rb") as f:
            def rd(fmt):
                b = f.read(struct.calcsize(fmt))
                if len(b) != struct.calcsize(fmt):
                    raise EOFError
                return struct.unpack(fmt, b)[0]

            def skip(vt, n=1):
                if vt == 8:  # (strings: u64 length + bytes)
                    for _ in range(n):
                        f.seek(rd("<Q"), 1)
                elif vt == 9:  # (arrays: element type, count, elements)
                    for _ in range(n):
                        et, m = rd("<I"), rd("<Q")
                        skip(et, m)
                else:
                    f.seek(sizes[vt] * n, 1)
            if f.read(4) != b"GGUF" or rd("<I") < 2:
                return None, None
            rd("<Q")  # (tensor count)
            for _ in range(rd("<Q")):
                key = f.read(rd("<Q")).decode("utf-8", "replace")
                vt = rd("<I")
                if key == "general.architecture" and vt == 8:
                    arch = f.read(rd("<Q")).decode("utf-8", "replace")
                elif key.endswith(".context_length") and vt in ints:
                    ctxs[key] = rd(ints[vt])
                elif key == "tokenizer.ggml.tokens" and vt == 9:
                    et, m = rd("<I"), rd("<Q")
                    if et != 8:
                        skip(et, m)
                        continue
                    longest = 0
                    for _ in range(m):  # (a token string's UTF-8 length >= its decoded piece's bytes)
                        n = rd("<Q")
                        f.seek(n, 1)
                        longest = max(longest, n)
                else:
                    skip(vt)
                if longest is not None and arch and f"{arch}.context_length" in ctxs:
                    break
    except (OSError, EOFError, KeyError, struct.error):
        pass
    return (ctxs.get(f"{arch}.context_length") if arch else None), longest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="a host name clients may reach the server under, besides localhost, IP addresses and --host's "
                         "(repeatable or comma-separated; '*': any name, e.g. behind a reverse proxy, which turns the "
                         "DNS-rebinding check off; env MK_ALLOW_HOSTS); requests for other names get a 403")
    ap.add_argument("--cors-origin", "--cors-origins", dest="cors_origin", action="append", default=[], metavar="ORIGIN",
                    help="a web origin (scheme://host[:port]) whose pages may call the server: its requests get CORS "
                         "headers (repeatable or comma-separated; env MK_CORS_ORIGINS); requests from pages of other origins "
                         "get a 403 (default: none, only the server's own origin; clients that are not web pages send no "
                         "Origin and are not affected)")
    ap.add_argument("--api-key", default=os.environ.get("MK_API_KEY") or None, metavar="KEY",
                    help="require this key on every request but /health, as 'Authorization: Bearer KEY' or 'x-api-key: "
                         "KEY'; others get a 401 (env MK_API_KEY, which keeps the key out of the process list)")
    ap.add_argument("-c", "--ctx", type=int, default=None,
                    help="context positions (default 32768; with --agents: per slot, 0 = the most that fit, the default "
                         "there)")
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--mtp", default=None,
                    help="the MTP draft layer from another GGUF of the same model (default: the model file's; env MK_MTP)")
    ap.add_argument("--no-mtp", action="store_true", help="no MTP layer: plain decoding, depth 0 (env MK_NO_MTP=1)")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), LIB_MK))
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit; with --agents: q8_0)")
    ap.add_argument("--agents", type=int, default=env_int("MK_AGENTS", 0, 0), metavar="N",
                    help="serve up to N requests at once (e.g. several coding agents), each in a parallel slot of the "
                         "engine with its own context of --ctx positions, decoded together in one batched launch per "
                         "cycle; each reply is the one the request gets alone. Text only; no presence / frequency "
                         "penalties, side requests, parked conversations or pinned checkpoints (0: one request at a "
                         "time; env MK_AGENTS)")
    ap.add_argument("--parallel", "-np", type=int, default=0, metavar="N",
                    help="--agents N with -c the context of each slot, default 32768 (llama-server's and open-jet's option)")
    ap.add_argument("--ck-every", type=int, default=4096,
                    help="DeltaNet checkpoint every N prompt tokens (0: off); see MK_CK_SLOTS for the pool size")
    ap.add_argument("--ck-msgs", type=int, default=1, help="DeltaNet checkpoint at each message start (1/0)")
    ap.add_argument("--ck-pin", type=int, default=1,
                    help="pin the DeltaNet checkpoint at the end of a prompt's system message (and where a new session "
                         "left the cached one inside it) against the pool's eviction, and park it with the conversation: "
                         "new sessions of a client resume after its system prompt and tools (1/0)")
    ap.add_argument("--exact-reuse", action="store_true",
                    help="reused requests answer exactly as fresh ones (MK_EXACT=1): restore a checkpoint and prefill "
                         "instead of appending to decoded rows (costs prompt time when a long reply is reused)")
    ap.add_argument("--side-max", type=int, default=env_int("MK_SIDE_MAX", 8192, 0), metavar="N",
                    help="side requests: a first-turn prompt (no assistant reply in it) of at most N tokens that would discard "
                         "at least twice its length of the cached conversation (with parking on: one without tools, "
                         "such as a client's title or summary request, or one that asks for at most 1024 tokens; and "
                         "any request without tools beside a conversation whose last request had tools) runs without "
                         "evicting it: the KV rows it overwrites go to host memory and come back after its reply; "
                         "with parking on, its own conversation is parked too (evicted first), so a follow-up without "
                         "tools or with a capped reply continues it the same way (0: off; env MK_SIDE_MAX)")
    ap.add_argument("--park-mib", type=int, default=env_int("MK_PARK_MIB", 8192, 0), metavar="MIB",
                    help="host memory for parked conversations: a request for another conversation parks the cached one "
                         "(KV, recurrent state, checkpoints) and it comes back bit for bit when its next request arrives; "
                         "least recently parked ones go first (allocated on first use; 0: off; env MK_PARK_MIB)")
    ap.add_argument("--park-min", type=int, default=1024, metavar="N",
                    help="conversations of fewer tokens are not parked; a parked one comes back when its prompt reuses at "
                         "least N more tokens than the cached one's")
    ap.add_argument("--park-cks", type=int, default=2, metavar="N",
                    help="checkpoints kept with a parked conversation (the newest, besides its pinned ones: --ck-pin; with "
                         "exact reuse any checkpoint gives the same bits, fewer only mean more re-prefill after an edit)")
    ap.add_argument("--content-specials", choices=["text", "parse"],
                    default=os.environ.get("MK_CONTENT_SPECIALS") or "text",
                    help="special-token text (<|im_end|>, <|im_start|>, ...) inside messages, tool results, tool calls and "
                         "tool definitions: text (tokenized as plain text, so content such as a file an agent reads cannot "
                         "forge turns) or parse (as those special tokens, as llama-server does) (env MK_CONTENT_SPECIALS)")
    ap.add_argument("--chat-template-file", dest="chat_template", action=ChatTemplateArg, metavar="PATH",
                    help="Jinja chat template file to use instead of the GGUF's (as llama-server), e.g. another "
                         "template for a fine-tune of the same model")
    ap.add_argument("--chat-template", dest="chat_template", action=ChatTemplateArg, metavar="TEMPLATE",
                    help="the chat template itself (Jinja source; 'chatml': llama.cpp's ChatML) instead of the GGUF's; "
                         "the last of --chat-template / --chat-template-file wins")
    ap.add_argument("--reasoning-format", default=os.environ.get("MK_REASONING_FORMAT") or "auto", metavar="FORMAT",
                    help="llama-server's: auto / deepseek (the thinking as reasoning_content, streamed too: the default), "
                         "deepseek-legacy (also left in the content; a stream: only there), none (left in the content); "
                         "a request's reasoning_format wins; Anthropic replies keep their thinking blocks "
                         "(env MK_REASONING_FORMAT)")
    ap.add_argument("--chat-template-kwargs", default=os.environ.get("MK_CHAT_TEMPLATE_KWARGS") or None, metavar="JSON",
                    help="defaults for every request's chat_template_kwargs (a JSON object of "
                         + ", ".join(TEMPLATE_KWARGS) + "; the request's own win), e.g. '{\"enable_thinking\": false}' "
                         "(env MK_CHAT_TEMPLATE_KWARGS)")
    ap.add_argument("--mmproj", default=None,
                    help="vision projector GGUF (e.g. mmproj-*.gguf): image_url inputs, encoded on the CPU (no VRAM)")
    ap.add_argument("--vision-lib", default=os.path.join(os.path.dirname(HERE), LIB_VISION))
    ap.add_argument("--vision-threads", type=int, default=0,
                    help="CPU threads of the image encoder (0: every logical CPU but two)")
    ap.add_argument("--vision-backend", choices=["auto", "cpu", "mtmd"], default="auto",
                    help="image encoder: cpu (AVX2 / FMA fp32 kernels), mtmd (llama.cpp's), auto (cpu if the CPU has AVX2 + FMA)")
    ap.add_argument("--image-min-tokens", type=int, default=0, help="minimum tokens per image (0: the mmproj's default)")
    ap.add_argument("--image-max-tokens", type=int, default=0,
                    help="maximum tokens per image (0: the mmproj's default, 4096 for Qwen3.x; fewer encode faster)")
    ap.add_argument("--image-cache", type=int, default=8,
                    help="images kept decoded / encoded for reuse (a request holds all of its own while it runs)")
    ap.add_argument("--video-fps", type=float, default=4.0,
                    help="video input (input_video / video_url parts; --mmproj and ffmpeg): frames sampled per second "
                         "(llama.cpp's default 4); every two frames are one grid of tokens")
    ap.add_argument("--video-max-tokens", type=int, default=0,
                    help="maximum tokens per video frame (pair) (0: --image-max-tokens, as llama.cpp; fewer encode faster "
                         "and fit longer clips; cpu vision backend)")
    ap.add_argument("--video-cache", type=int, default=2,
                    help="videos kept decoded / encoded for reuse by the next requests of a conversation")
    ap.add_argument("--ffmpeg", default=None,
                    help="ffmpeg executable (or its directory) for video input (env MK_FFMPEG; default: ffmpeg on PATH, "
                         "else the imageio-ffmpeg package's)")
    args = ap.parse_args()
    if args.parallel < 0 or args.agents < 0:
        ap.error("--agents / --parallel: expected a number of slots >= 0")
    if args.ctx is None:  # (--agents: the most that fits in each slot; --parallel, as open-jet's: 32768 each)
        args.ctx = 0 if args.agents > 0 and not args.parallel else 32768
    if args.parallel:
        args.agents = args.parallel
    if args.agents > 0:
        if args.kv == "f16":
            ap.error("--agents: the parallel slots need the q8_0 KV cache (--kv q8 or auto)")
        if args.ctx < 0:
            ap.error("--ctx: expected positions per slot >= 0 (0: the most that fit)")
        args.kv = "q8"
        # prefill chunks of 1024 (half the scratch of the default: more context for the slots, same prefill speed), and
        # prefill rows independent of how a prompt is cut into chunks (MK_EXACT): a prompt fed beside other slots'
        # cycles computes what it computes alone
        os.environ.setdefault("MK_PF_CAP", "1024")
        os.environ.setdefault("MK_EXACT", "1")
        # one checkpoint slot for the engine's own sequence (1024 positions, the warm-up only; each slot has its own pool):
        # the default 8 would pin 8 x 174 MiB of host memory for nothing
        os.environ.setdefault("MK_CK_SLOTS", "1")
        for k in ("MK_PF_MIX", "MK_PF_CHUNK", "MK_PF_SOLO"):  # (the slot scheduler's: checked before the model loads)
            env_int(k, 0, 0)
    if not 0 < args.video_fps <= 30:
        ap.error("--video-fps: expected a rate in (0, 30]")
    if args.video_max_tokens and args.video_max_tokens < max(8, args.image_min_tokens):
        ap.error("--video-max-tokens: expected 0 or at least 8 (and --image-min-tokens)")
    for k in ("side_max", "park_mib", "park_min", "park_cks"):
        if getattr(args, k) < 0:
            ap.error(f"--{k.replace('_', '-')}: expected a number >= 0")
    if args.content_specials not in ("text", "parse"):  # (an MK_CONTENT_SPECIALS default is not checked by choices)
        ap.error(f"MK_CONTENT_SPECIALS: expected text or parse, not {args.content_specials!r}")
    if args.reasoning_format not in REASONING_FORMATS:
        ap.error(f"--reasoning-format: one of {', '.join(REASONING_FORMATS)}, not {args.reasoning_format!r}")
    if args.chat_template_kwargs is not None:
        try:
            kw = json.loads(args.chat_template_kwargs)
        except ValueError as e:
            ap.error(f"--chat-template-kwargs: not JSON ({e})")
        if not isinstance(kw, dict) or set(kw) - set(TEMPLATE_KWARGS):
            ap.error(f"--chat-template-kwargs: a JSON object of {', '.join(TEMPLATE_KWARGS)} (the keys the server hands to "
                     "the template)")
        args.chat_template_kwargs = kw
        log(f"chat template kwargs: {json.dumps(kw)} (under each request's)")
    if args.reasoning_format != "auto":
        log(f"reasoning format: {args.reasoning_format}")
    args.chat_template_src = None  # (before the port opens: a bad template fails at once, not after the model load)
    if args.chat_template:
        try:
            args.chat_template_src = chat_template_source(*args.chat_template)
        except ValueError as e:
            ap.error(str(e))
        kind, value = args.chat_template
        log(f"chat template: {value if kind == 'file' else 'from --chat-template'} instead of the GGUF's")

    def listed(env, values):  # (the environment's comma-separated values, then the options')
        return [v.strip() for s in [os.environ.get(env, "")] + values for v in s.split(",") if v.strip()]
    try:
        gate = Gate(args.host, listed("MK_ALLOW_HOSTS", args.allow_host), listed("MK_CORS_ORIGINS", args.cors_origin),
                    args.api_key)
    except ValueError as e:
        ap.error(str(e))
    GatedHandler.timeout = env_int("MK_TIMEOUT_S", TIMEOUT_S, 0) or None  # (StreamRequestHandler applies it)
    mk_limits.REQUEST_TIMEOUT_S = env_int("MK_REQUEST_TIMEOUT_S", mk_limits.REQUEST_TIMEOUT_S, 0)
    mk_limits.MAX_BODY_MB = env_int("MK_MAX_BODY_MB", mk_limits.MAX_BODY_MB)
    mk_limits.MAX_INFLIGHT_MB = max(env_int("MK_MAX_INFLIGHT_MB", mk_limits.MAX_INFLIGHT_MB), mk_limits.MAX_BODY_MB)  # (a largest body always fits)
    mk_limits.MAX_CONNECTIONS = env_int("MK_MAX_CONNECTIONS", mk_limits.MAX_CONNECTIONS)
    mk_limits.MAX_IMAGES = env_int("MK_MAX_IMAGES", mk_limits.MAX_IMAGES)
    mk_limits.MAX_VIDEO_FRAMES = env_int("MK_MAX_VIDEO_FRAMES", mk_limits.MAX_VIDEO_FRAMES)
    n_train, longest = gguf_text_limits(args.model)
    mk_limits.REASONING_MAX_BYTES = max(n_train or 0, args.ctx) * (longest or 256)
    if args.mmproj:  # (the vision library checks images against it; the video decoder, frames)
        mk_limits.IMAGE_MAX_PIXELS = env_int("MK_IMAGE_MAX_PIXELS", mk_limits.IMAGE_MAX_PIXELS)
    if args.exact_reuse:
        os.environ["MK_EXACT"] = "1"  # read by the engine at construction
    # the port opens first (a busy port fails before the model loads); until the load is done every request gets a 503
    # (llama-server's "Loading model"), then the handler is swapped for the real one (new connections take it)
    httpd = listen(args.host, args.port, make_loading_handler())
    httpd.gate = gate
    log(f"listening on http://{args.host}:{args.port} (503 until the model is loaded)")
    if gate.any_host or gate.hosts != {"localhost"}:
        log("host names: any (--allow-host *)" if gate.any_host else f"host names: {', '.join(sorted(gate.hosts))} and IP addresses")
    if gate.origins:
        log(f"CORS: pages of {', '.join(sorted(gate.origins))} may call the server")
    if gate.api_key:
        log("API key: required on every request but /health")
    exposed = exposure_warning(args.host, gate.api_key)
    if exposed:
        log("warning:", exposed)
    box = {}

    def load():
        try:
            srv = (SlotServer if args.agents > 0 else Server)(args)
            if srv.eng.exact and not args.agents:
                log("exact reuse: on (MK_EXACT)")
            box["srv"] = srv
            httpd.RequestHandlerClass = make_handler(srv)
        except BaseException as e:  # (the server exits: serve_forever returns)
            box["error"] = e
            traceback.print_exc()
            httpd.shutdown()

    threading.Thread(target=load, name="mk-load", daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CON.status("")
        if "srv" in box:
            box["srv"].eng.keep_warm(False)
        httpd.server_close()
    if "error" in box:
        sys.exit(f"[mk-server] startup failed: {box['error']}")


if __name__ == "__main__":
    main()

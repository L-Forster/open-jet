#!/usr/bin/env python3
"""OpenJet Engine: OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF (enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs, tools
via `tools`), reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured
`tool_calls`.  Sampling: temperature, top_p, top_k (<= 32), min_p, seed.  Successive requests of a
conversation reuse the processed tokens: the new prompt is matched against the sequence so far and
only the rest is prefilled (rewinding to the end of the previous prompt if it diverges after that).
With --mmproj, OpenAI image_url parts (base64 data: URLs) are encoded on the CPU (vision/mk_vision.h).
GET /health, GET /v1/models.  One request at a time.
"""
import argparse
import base64
import binascii
import codecs
import ctypes
import hashlib
import json
import os
import random
import re
import struct
import sys
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK_MAX = 32


def log(*a):
    print("[mk-server]", *a, file=sys.stderr, flush=True)


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        # images (libraries before it: text only)
        self.images = hasattr(L, "mk_prefill_img")
        if self.images:
            L.mk_prefill_img.restype = ctypes.c_int
            L.mk_prefill_img.argtypes = [vp, ip, ctypes.c_int, ip, ip, ip, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            L.mk_n_embd.restype, L.mk_n_embd.argtypes = ctypes.c_int, []
        self.h = L.mk_new(model.encode(), ctx, n_draft, draft_vocab.encode() if draft_vocab else None, kv_mode)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.ctx = ctx
        self._pieces = {}

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text):
        b = text.encode()
        cap = len(b) + 16
        out = (ctypes.c_int * cap)()
        n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        if n < 0:
            cap = -n
            out = (ctypes.c_int * cap)()
            n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        return list(out[:n])

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def prefill(self, toks, images=()):
        """images: (pos, nx, ny, rows address) of each image the fed positions overlap (absolute positions)"""
        arr = (ctypes.c_int * len(toks))(*toks)
        if images:
            k = len(images)
            pos, nx, ny = ((ctypes.c_int * k)(*[im[j] for im in images]) for j in range(3))
            rows = (ctypes.c_void_p * k)(*[im[3] for im in images])
            r = self.L.mk_prefill_img(self.h, arr, len(toks), pos, nx, ny, rows, k)
        else:
            r = self.L.mk_prefill(self.h, arr, len(toks))
        if r < 0:
            raise self._err("prefill")
        return r

    def step(self):
        out = (ctypes.c_int * 8)()
        n = self.L.mk_step(self.h, out, 8)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0


class Vision:
    """ctypes binding of vision/mk_vision.h (libqwen38vision.so): the CPU image encoder, plus a cache of decoded /
    encoded images by content hash, so a conversation re-sending its images encodes each once"""

    class Image:
        def __init__(self, vision, handle, key):
            self.v, self.h, self.key = vision, handle, key
            nx, ny = ctypes.c_int(), ctypes.c_int()
            self.n = vision.L.mkv_image_grid(handle, ctypes.byref(nx), ctypes.byref(ny))
            self.nx, self.ny = nx.value, ny.value
            self.encode_s = None

        def rows(self):
            """address of the [n][n_embd] fp32 rows, encoding on the first call"""
            L = self.v.L
            if self.encode_s is None:
                if L.mkv_image_encode(self.v.ctx, self.h) != 0:
                    raise RuntimeError(f"image encode: {L.mkv_last_error().decode()}")
                self.encode_s = L.mkv_image_encode_ms(self.h) / 1e3
            return L.mkv_image_rows(self.h)

        def free(self):
            if self.h:
                self.v.L.mkv_image_free(self.h)
                self.h = None

    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("min_tokens", ctypes.c_int), ("max_tokens", ctypes.c_int),
                    ("flash_attn", ctypes.c_int), ("verbose", ctypes.c_int), ("backend", ctypes.c_int)]

    BACKENDS = {"auto": 0, "mtmd": 1, "cpu": 2}

    def __init__(self, lib, mmproj, model, threads=0, min_tokens=0, max_tokens=0, cache=8, backend="auto"):
        L = self.L = ctypes.CDLL(lib)
        vp, cp = ctypes.c_void_p, ctypes.c_char_p
        for name, res, args in (
                ("mkv_params_default", Vision.Params, []), ("mkv_new", vp, [cp, cp, Vision.Params]),
                ("mkv_free", None, [vp]), ("mkv_last_error", cp, []), ("mkv_n_embd", ctypes.c_int, [vp]),
                ("mkv_n_threads", ctypes.c_int, [vp]), ("mkv_backend", ctypes.c_int, [vp]),
                ("mkv_image_load", vp, [vp, cp, ctypes.c_size_t]),
                ("mkv_image_grid", ctypes.c_int, [vp, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]),
                ("mkv_image_encode", ctypes.c_int, [vp, vp]), ("mkv_image_rows", vp, [vp]),
                ("mkv_image_encode_ms", ctypes.c_double, [vp]), ("mkv_image_free", None, [vp])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        p = L.mkv_params_default()
        p.n_threads, p.min_tokens, p.max_tokens, p.backend = threads, min_tokens, max_tokens, Vision.BACKENDS[backend]
        self.ctx = L.mkv_new(mmproj.encode(), model.encode(), p)
        if not self.ctx:
            raise RuntimeError(f"vision: {L.mkv_last_error().decode()}")
        self.n_embd = L.mkv_n_embd(self.ctx)
        self.n_threads = L.mkv_n_threads(self.ctx)
        self.backend = {1: "libmtmd", 2: "vit_cpu"}.get(L.mkv_backend(self.ctx), "?")
        self.cache = {}  # key -> Image, least recently used first
        self.cache_max = max(1, cache)

    def image(self, data):
        """the Image of these file bytes (decoded and preprocessed; encoded lazily by rows())"""
        key = hashlib.sha256(data).hexdigest()
        im = self.cache.pop(key, None)
        if im is None:
            h = self.L.mkv_image_load(self.ctx, data, len(data))
            if not h:
                raise BadRequest(f"image: {self.L.mkv_last_error().decode()}")
            im = Vision.Image(self, h, key)
        self.cache[key] = im
        while len(self.cache) > self.cache_max:
            old = self.cache.pop(next(iter(self.cache)))
            old.free()
        return im

    def warm_up(self):
        """one encode of a 64 x 64 grey PNG (not cached): the first encode after the engine's load pays ~0.5 s once
        (every weight streamed in again, the workers' scratch), here at startup instead of in the first image request"""
        w = h = 64
        raw = b"".join(bytes([0]) + bytes([128, 128, 128]) * w for _ in range(h))

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        png = (bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        hd = self.L.mkv_image_load(self.ctx, png, len(png))
        if not hd or self.L.mkv_image_encode(self.ctx, hd) != 0:
            raise RuntimeError(f"vision warm-up: {self.L.mkv_last_error().decode()}")
        self.L.mkv_image_free(hd)


def image_bytes(url):
    """the file bytes of an OpenAI image_url (a data: URL, as llama-server takes them by default)"""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise BadRequest("image_url: only data: URLs (base64) are supported")
    head, sep, payload = url.partition(",")
    if not sep or ";base64" not in head:
        raise BadRequest("image_url: expected a base64 data: URL")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"image_url: bad base64 ({e})")


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


class BadRequest(Exception):
    pass


MEDIA_MARKER = "<__media__>"  # llama-server's (mtmd_default_marker)


def normalize_messages(messages, media=None):
    """media (a list: the server has a vision encoder): image_url parts become MEDIA_MARKER and their file bytes are
    appended to media in order. A message with images is flattened to one string as llama-server does (text parts
    joined by newlines, no newline next to an image), so the prompt around each image is llama-server's."""
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text", "image_url"):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text and image_url only)")
            if any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
                if media is None:
                    raise BadRequest("image input is not supported: start the server with --mmproj")
                text, after_media = "", False
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        iu = part.get("image_url")
                        media.append(image_bytes(iu.get("url") if isinstance(iu, dict) else iu))
                        text += MEDIA_MARKER
                        after_media = True
                    else:
                        if not after_media and text:
                            text += "\n"
                        text += str((part.get("text") if isinstance(part, dict) else part) or "")
                        after_media = False
                m["content"] = text
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*(?:</tool_call>|$)|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for p in PARAM_RE.finditer(m.group(2)):
            key, val = p.group(1).strip(), p.group(2)
            typ = (props.get(key) or {}).get("type")
            if typ == "string" or (typ is None and not val.strip()[:1] in "[{0123456789-tfn"):
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool-call markup."""

    def __init__(self, thinking, tools):
        self.mode = "reasoning" if thinking else "content_start"
        self.tools = tools
        self.buf = ""
        self.tool_text = ""

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content_start"
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        out.append(("content", self.buf[:i].rstrip()))
                        self.tool_text = self.buf[i:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: collect everything
            self.tool_text += self.buf
            self.buf = ""
            break
        return [(k, t) for k, t in out if t]


# ---------------------------------------------------------------------------------------------


class Server:
    vision = None   # Vision (--mmproj), else text only
    img_spans = ()  # the images of the engine's sequence: (pos, n, key), None: unknown; replaced, never mutated

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, args.ctx, args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv])
        self.template = make_template(self.eng.chat_template())
        if getattr(args, "mmproj", None):
            if not self.eng.images:
                raise RuntimeError(f"{args.lib} has no image input (mk_prefill_img): rebuild it")
            self.vision = Vision(args.vision_lib, args.mmproj, args.model, args.vision_threads, args.image_min_tokens,
                                 args.image_max_tokens, args.image_cache, args.vision_backend)
            if self.vision.n_embd != self.eng.L.mk_n_embd():
                raise RuntimeError(f"mmproj rows of {self.vision.n_embd} for a model of {self.eng.L.mk_n_embd()}")
            ids = [self.eng.tokenize(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError("the vocabulary has no <|vision_start|> / <|vision_end|> / <|image_pad|> tokens")
            self.vis_start, self.vis_end, self.img_pad = (i[0] for i in ids)
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.lock = threading.Lock()
        self.warned = set()
        vis = (f", vision on the CPU ({os.path.basename(args.mmproj)}, {self.vision.backend}, {self.vision.n_threads} threads)"
               if self.vision else "")
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts{vis}")

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and the first request ends the process with a CUDA out-of-memory (CK exits), which
        the client sees as a dropped connection. Warming up here moves that failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        self.eng.prefill(toks)
        self.eng.step()
        self.eng.reset()
        if self.vision:
            self.vision.warm_up()

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    def render(self, body):
        """(prompt text, thinking, image file bytes in prompt order: one MEDIA_MARKER in the text each)"""
        kw = dict(body.get("chat_template_kwargs") or {})
        media = [] if self.vision else None
        ctx = dict(messages=normalize_messages(body.get("messages") or [], media), add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        if media and text.count(MEDIA_MARKER) != len(media):
            raise BadRequest(f"the rendered prompt has {text.count(MEDIA_MARKER)} image markers for {len(media)} images")
        return text, thinking, media or []

    def tokenize_prompt(self, text, media):
        """the prompt's token ids and its images [(pos, Vision.Image)]: each MEDIA_MARKER becomes <|vision_start|>, the
        image's rows (<|image_pad|> placeholders, nx * ny of them), <|vision_end|>; the text between is tokenized piece
        by piece, as mtmd_tokenize does"""
        if not media:
            return self.eng.tokenize(text), []
        toks, images = [], []
        segs = text.split(MEDIA_MARKER)
        for i, seg in enumerate(segs):
            if seg:
                toks += self.eng.tokenize(seg)
            if i < len(media):
                im = self.vision.image(media[i])
                toks.append(self.vis_start)
                images.append((len(toks), im))
                toks += [self.img_pad] * im.n
                toks.append(self.vis_end)
        return toks, images

    def load_prompt(self, toks, images=()):
        """brings the engine to `toks` processed, reusing what it has; returns (first token, reused count).
        images: [(pos, Vision.Image)] of the prompt"""
        eng = self.eng
        seq = eng.sequence()
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        # images: their rows' placeholder ids are alike, so the processed sequence matches only up to the first image
        # that is not the same image at the same position in both (after a failed prefill: nothing is reused)
        spans = [(p, im.n, im.key) for p, im in images]
        if self.img_spans is None:
            lcp = 0
        else:
            for p, _, _ in set(spans) ^ set(self.img_spans):
                lcp = min(lcp, p)

        def feed(start):  # prefill toks[start:] with the images it overlaps (encoded now, each once: Vision's cache)
            imgs = [(p, im.nx, im.ny, im.rows()) for p, im in images if p + im.n > start]
            return eng.prefill(toks[start:], imgs), start

        self.img_spans = None  # (until the prefill succeeds)
        if seq and lcp >= eng.committed():
            eng.truncate(lcp)
            r = feed(lcp)
        else:
            ck = eng.checkpoint_pos()
            if 0 < ck <= lcp and eng.rewind_to_checkpoint():
                r = feed(ck)
            else:
                eng.reset()
                r = feed(0)
        self.img_spans = spans
        return r

    def sampling(self, body):
        temp = float(body.get("temperature", 0.8) if body.get("temperature") is not None else 0.8)
        top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
        top_k = int(body.get("top_k", 40) if body.get("top_k") is not None else 40)
        min_p = float(body.get("min_p", 0.05) if body.get("min_p") is not None else 0.05)
        seed = body.get("seed")
        seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        if top_k <= 0 or top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        for k in ("presence_penalty", "frequency_penalty"):
            if float(body.get(k) or 0) != 0:
                self.warn_once(k, f"{k} is not supported by the megakernel and is ignored")
        if float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0) != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            self.eng.set_sampling(0.0, 1.0, TOPK_MAX, 0.0, seed)
        else:
            self.eng.set_sampling(temp, top_p, top_k, min_p, seed)

    def generate(self, body, emit):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas.
        Returns (finish_reason, tool_calls, usage). emit returning False aborts."""
        text, thinking, media = self.render(body)
        toks, images = self.tokenize_prompt(text, media)
        max_new = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1
        max_new = int(max_new)
        limit = self.args.ctx - len(toks) - 16
        if limit <= 0:
            raise BadRequest(f"prompt of {len(toks)} tokens does not fit the context of {self.args.ctx}")
        max_new = limit if max_new <= 0 else min(max_new, limit)
        tools = body.get("tools") or []
        self.sampling(body)
        t0 = time.time()
        enc0 = [im.encode_s for _, im in images]
        tok, reused = self.load_prompt(toks, images)
        t1 = time.time()
        enc = [im.encode_s for (_, im), e in zip(images, enc0) if e is None and im.encode_s is not None]
        if images:
            log(f"images: {len(images)} ({', '.join(f'{im.nx}x{im.ny}' for _, im in images)} tokens), "
                f"{len(enc)} encoded on the CPU in {sum(enc):.2f} s")
        parser = OutputParser(thinking, tools)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, pending, aborted = 0, "length", [tok], False
        while n_gen < max_new and not aborted:
            if not pending:
                pending = self.eng.step()
            t = pending.pop(0)
            n_gen += 1
            if self.eng.is_eog(t):
                finish = "stop"
                break
            for kind, delta in parser.feed(dec.decode(self.eng.piece(t))):
                if emit(kind, delta) is False:
                    aborted = True
                    break
        for kind, delta in parser.feed(dec.decode(b"", final=True), final=True):
            emit(kind, delta)
        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
        if calls:
            finish = "tool_calls"
        t2 = time.time()
        log(f"prompt {len(toks)} tok ({reused} reused) in {t1 - t0:.2f} s, {n_gen} tok in {t2 - t1:.2f} s "
            f"({n_gen / max(t2 - t1, 1e-9):.1f} tok/s), {finish}")
        usage = {"prompt_tokens": len(toks), "completion_tokens": n_gen, "total_tokens": len(toks) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        timings = {"prompt_n": len(toks) - reused, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - t1) * 1e3, "predicted_per_second": n_gen / max(t2 - t1, 1e-9)}
        return finish, calls, usage, timings


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": srv.model_id, "object": "model", "owned_by": "openjet"}]})
            elif path == "/props":
                self._json(200, {"default_generation_settings": {"n_ctx": srv.args.ctx}, "total_slots": 1,
                                 "modalities": {"vision": srv.vision is not None, "audio": False}})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404 if not path.startswith("/slots") else 501,
                           {"error": {"message": f"{path} is not supported by the megakernel server", "type": "not_supported"}})
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            with srv.lock:
                try:
                    if body.get("stream"):
                        self._stream(body)
                    else:
                        self._complete(body)
                except BadRequest as e:
                    self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # engine failure: report it, keep serving
                    log("error:", repr(e))
                    try:
                        self._json(500, {"error": {"message": str(e), "type": "server_error"}})
                    except Exception:
                        pass

        def _complete(self, body):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(body, lambda k, t: parts[k].append(t))
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})

        def _stream(self, body):
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            started = [False]

            def chunk(delta, finish=None, usage=None, timings=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                data = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunk({"role": "assistant", "content": None})

            def emit(kind, text):
                head()
                try:
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            # render / tokenize errors are reported as a 400 before the stream starts
            srv.render(body)
            finish, calls, usage, timings = srv.generate(body, emit)
            head()
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            chunk({}, finish=finish, timings=timings)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            data = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    return H


def main():
    ap = argparse.ArgumentParser(
        description="OpenJet Engine: CUDA inference for Qwen3.8-27B with an OpenAI-compatible HTTP API."
    )
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), "libqwen38mk.so"))
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit)")
    ap.add_argument("--mmproj", default=None,
                    help="vision projector GGUF (e.g. mmproj-*.gguf): image_url inputs, encoded on the CPU (no VRAM)")
    ap.add_argument("--vision-lib", default=os.path.join(os.path.dirname(HERE), "libqwen38vision.so"))
    ap.add_argument("--vision-threads", type=int, default=0, help="CPU threads of the image encoder (0: all)")
    ap.add_argument("--vision-backend", choices=["auto", "cpu", "mtmd"], default="auto",
                    help="image encoder: cpu (AVX2 / FMA fp32 kernels), mtmd (llama.cpp's), auto (cpu if the CPU has AVX2 + FMA)")
    ap.add_argument("--image-min-tokens", type=int, default=0, help="minimum tokens per image (0: the mmproj's default)")
    ap.add_argument("--image-max-tokens", type=int, default=0,
                    help="maximum tokens per image (0: the mmproj's default, 4096 for Qwen3.x; fewer encode faster)")
    ap.add_argument("--image-cache", type=int, default=8, help="images kept decoded / encoded for reuse")
    args = ap.parse_args()
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

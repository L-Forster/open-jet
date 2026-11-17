"""Image and video input: the vision encoder's binding, content parts, ffmpeg frame decoding."""
import base64
import binascii
import ctypes
import hashlib
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import zlib

import mk_limits
from mk_errors import BadRequest
from mk_limits import VIDEO_DECODE_S, VIDEO_TS_MS


class Vision:
    """ctypes binding of vision/mk_vision.h (libqwen38vision.so): the CPU image encoder (llama.cpp's libmtmd), plus a
    cache of decoded / encoded images by content hash, so a conversation re-sending its images encodes each once.
    A request pins its images (image() .. release()): the cache never frees an image a request still holds, so it can
    hold more than --image-cache images while a request runs."""

    class Image:
        def __init__(self, vision, handle, key):
            self.v, self.h, self.key = vision, handle, key
            nx, ny = ctypes.c_int(), ctypes.c_int()
            self.n = vision.L.mkv_image_grid(handle, ctypes.byref(nx), ctypes.byref(ny))
            self.nx, self.ny = nx.value, ny.value
            self.encode_s = None
            self.pins = 0  # requests holding it (Vision.image / Vision.release)

        def rows(self):
            """address of the [n][n_embd] fp32 rows, encoding on the first call"""
            L = self.v.L
            if not self.h:
                raise RuntimeError("image encode: the image was freed")
            if self.encode_s is None:
                if L.mkv_image_encode(self.v.ctx, self.h) != 0:
                    raise RuntimeError(f"image encode: {L.mkv_last_error().decode()}")
                self.encode_s = L.mkv_image_encode_ms(self.h) / 1e3
            return L.mkv_image_rows(self.h)

        def free(self):
            if self.h:
                self.v.L.mkv_image_free(self.h)
                self.h = None

    class Video:
        """a decoded video in llama.cpp's layout: parts [str | Image], each Image one frame or a frame pair (Qwen-VL's
        temporal patch of 2: one grid of tokens); owned by Vision.videos (its images are pinned like images)"""

        def __init__(self, key, parts, n_frames):
            self.key, self.parts, self.n_frames = key, parts, n_frames
            self.images = [p for p in parts if not isinstance(p, str)]

        def free(self):
            for im in self.images:
                im.free()

    class Params(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("min_tokens", ctypes.c_int), ("max_tokens", ctypes.c_int),
                    ("flash_attn", ctypes.c_int), ("verbose", ctypes.c_int), ("backend", ctypes.c_int)]

    BACKENDS = {"auto": 0, "mtmd": 1, "cpu": 2}
    frames_ok = False  # the library has video frames (mkv_frames_load)
    videos = None      # key -> Video, least recently used first (__init__)
    video_cache_max = 0

    def __init__(self, lib, mmproj, model, threads=0, min_tokens=0, max_tokens=0, cache=8, backend="auto", video_cache=2):
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
        # video frames (libraries before it: images only)
        self.frames_ok = hasattr(L, "mkv_frames_load")
        if self.frames_ok:
            L.mkv_frames_load.restype = vp
            L.mkv_frames_load.argtypes = [vp, cp, cp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
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
        self.videos = {}  # key -> Video, least recently used first
        self.video_cache_max = max(0, video_cache)

    def image(self, data):
        """the Image of these file bytes (decoded and preprocessed; encoded lazily by rows()), pinned: the caller
        releases it (release) when its request is done"""
        key = hashlib.sha256(data).hexdigest()
        im = self.cache.pop(key, None)
        if im is None:
            h = self.L.mkv_image_load(self.ctx, data, len(data))
            if not h:
                raise BadRequest(f"image: {self.L.mkv_last_error().decode()}")
            im = Vision.Image(self, h, key)
        self.cache[key] = im
        im.pins += 1
        return im

    def video(self, data, decoder, fps, max_tokens, max_frames):
        """the Video of these file bytes (decoded by decoder (VideoDecoder) at fps, frames resized to at most max_tokens
        tokens (0: the image limit); its images are preprocessed, encoded lazily by rows()), pinned like image(). A
        video of more than max_frames frames is refused (BadRequest) while it decodes."""
        key = f"{hashlib.sha256(data).hexdigest()}/{fps!r}/{max_tokens}"
        v = self.videos.pop(key, None)
        if v is not None and v.n_frames > max_frames:
            self.videos[key] = v
            raise BadRequest(f"video of {v.n_frames} frames at {fps:g} fps: over the {max_frames} frames left for this "
                             f"request (MK_MAX_VIDEO_FRAMES {mk_limits.MAX_VIDEO_FRAMES})")
        if v is None:
            parts, n = [], 0
            try:
                for kind, x in video_parts(decoder.frames(data, fps, max_frames), fps):
                    if kind == "text":
                        parts.append(x)
                        continue
                    (w, h, a), b = x[0], x[1]
                    n += 2 if b else 1
                    hd = self.L.mkv_frames_load(self.ctx, a, b[2] if b else None, w, h, max_tokens)
                    if not hd:
                        raise BadRequest(f"video frame: {self.L.mkv_last_error().decode()}")
                    parts.append(Vision.Image(self, hd, f"{key}/{len(parts)}"))
            except BaseException:
                for im in parts:
                    if not isinstance(im, str):
                        im.free()
                raise
            v = Vision.Video(key, parts, n)
        self.videos[key] = v
        for im in v.images:
            im.pins += 1
        return v

    def release(self, images):
        """unpins the images of a request (image() / video() once per pin), then frees the least recently used unpinned
        images beyond --image-cache and videos beyond --video-cache"""
        for im in images:
            im.pins -= 1
        extra = len(self.cache) - self.cache_max
        for key in [k for k, im in self.cache.items() if im.pins <= 0][:max(extra, 0)]:
            self.cache.pop(key).free()
        extra = len(self.videos or ()) - self.video_cache_max
        for key in [k for k, v in (self.videos or {}).items() if all(im.pins <= 0 for im in v.images)][:max(extra, 0)]:
            self.videos.pop(key).free()

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
# video: frames from ffmpeg, laid out as llama.cpp's mtmd does (mtmd-helper.cpp video helpers + mtmd_tokenize)


class VideoBytes(bytes):
    """the file bytes of a video content part (normalize_messages' media list holds images as plain bytes)"""


VIDEO_PARTS = ("input_video", "video_url")  # llama-server's / vLLM's content part types


def video_bytes(part):
    """the file bytes of a video content part: llama-server's {"type": "input_video", "input_video": {"data": base64}}
    (or {"url": ...}), or {"type": "video_url", "video_url": {"url": ...}}; raw base64 or a base64 data: URL (no remote
    or file URLs)"""
    t = part.get("type")
    v = part.get(t)
    s = (v.get("data") or v.get("url")) if isinstance(v, dict) else v
    if not isinstance(s, str) or not s:
        raise BadRequest(f"{t}: expected base64 data")
    if s.startswith("data:"):
        head, sep, s = s.partition(",")
        if not sep or ";base64" not in head:
            raise BadRequest(f"{t}: expected a base64 data: URL")
    elif re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        raise BadRequest(f"{t}: only base64 data (or a base64 data: URL) is supported, not {s.split(':')[0]}:// URLs")
    try:
        data = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadRequest(f"{t}: bad base64 ({e})")
    if not data:
        raise BadRequest(f"{t}: empty video")
    return VideoBytes(data)


def f32(x):
    """x rounded to float32 (llama.cpp computes the video timestamps in float)"""
    return struct.unpack("f", struct.pack("f", x))[0]


def video_timestamp(k, fps, nxt):
    """mtmd-helper's timestamp text after frame k (0-based) at fps (float32) given the next threshold nxt (ms):
    (text or None, the next threshold); float32 arithmetic as there"""
    elapsed = f32(f32(f32(k) / fps) * 1000.0)
    if elapsed < nxt:
        return None, nxt
    s = f32(elapsed / 1000.0)
    minutes = int(f32(s / 60.0))
    sec = f32(s - f32(minutes * 60.0))
    return f"[{minutes}m{sec:.2f}s]", f32(nxt + VIDEO_TS_MS)


def video_parts(frames, fps):
    """a video's parts as llama.cpp lays them out: ("text", "Video:"), then each frame with a timestamp text
    ("[0m5.00s]") after every frame that reaches the next 5 s mark (frame 0 included); consecutive frames of one size
    (no text between) merge in pairs ("frames", (a, b)), a run's odd last frame stays alone ("frames", (a, None)).
    frames: (width, height, rgb bytes) in order; fps: the sampling rate (float32 in the timestamps)."""
    fps = f32(fps)
    yield "text", "Video:"
    nxt, pend = 0.0, None
    for k, fr in enumerate(frames):
        if pend is not None and pend[:2] != fr[:2]:  # (frames of different sizes do not merge)
            yield "frames", (pend, None)
            pend = None
        if pend is None:
            pend = fr
        else:
            yield "frames", (pend, fr)
            pend = None
        ts, nxt = video_timestamp(k, fps, nxt)
        if ts:
            if pend is not None:
                yield "frames", (pend, None)
                pend = None
            yield "text", ts
    if pend is not None:
        yield "frames", (pend, None)


class FrameTooLarge(ValueError):
    pass


def read_ppm(f, max_pixels=None):
    """the next binary PPM (P6, maxval 255) of a stream: (width, height, rgb bytes), None at its end; a frame of more
    than max_pixels pixels is refused from its header (FrameTooLarge), before its data is read"""
    def token():
        t = b""
        while True:
            c = f.read(1)
            if not c:
                return t or None
            if c == b"#" and not t:  # (a comment line)
                while c and c not in b"\r\n":
                    c = f.read(1)
                continue
            if c.isspace():
                if t:
                    return t
                continue
            t += c
            if len(t) > 16:
                raise ValueError("bad PPM header")
    magic = token()
    if magic is None:
        return None
    w, h, mx = token(), token(), token()
    if magic != b"P6" or not (w and h and mx) or not (w.isdigit() and h.isdigit()) or mx != b"255":
        raise ValueError("bad PPM header")
    w, h = int(w), int(h)
    if not (0 < w <= 65535 and 0 < h <= 65535):
        raise ValueError(f"bad PPM size {w} x {h}")
    if max_pixels and w * h > max_pixels:
        raise FrameTooLarge(f"frame of {w} x {h} pixels: over the limit of {max_pixels} pixels (MK_IMAGE_MAX_PIXELS)")
    data = f.read(w * h * 3)
    if len(data) != w * h * 3:
        raise ValueError("truncated frame")
    return w, h, data


def find_ffmpeg(explicit=None):
    """the ffmpeg executable for video input: explicit (--ffmpeg) or MK_FFMPEG (a file, or a directory with it), else
    ffmpeg on PATH, else the imageio-ffmpeg package's bundled binary (pip install imageio-ffmpeg); None: none found"""
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    p = explicit or os.environ.get("MK_FFMPEG")
    if p:
        p = os.path.join(p, exe) if os.path.isdir(p) else p
        return p if os.path.isfile(p) else None
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # optional
        p = imageio_ffmpeg.get_ffmpeg_exe()
        return p if p and os.path.isfile(p) else None
    except Exception:  # (not installed, or no binary for this platform)
        return None


class VideoDecoder:
    """video file bytes -> RGB24 frames through an ffmpeg executable, sampled by ffmpeg's fps filter as llama.cpp's
    mtmd-helper does (`-vf fps=F`, rgb24), but read as a PPM stream (each frame carries its size: no ffprobe needed).
    Frames over MK_IMAGE_MAX_PIXELS are refused as images are: read_ppm checks each frame's header before reading it,
    and ffmpeg's decoder gets `-max_pixels` (its default is INT_MAX pixels: a crafted clip of 40,000 x 40,000 frames
    would be decoded, 4.8 GB of rgb24 each) with MAX_PIXELS_SLACK more, as decoders check their row-aligned size
    (1100 x 1000 counts as 1152 x 1000), so that every frame within the limit decodes."""
    MAX_PIXELS_SLACK = 1 << 23  # (a 64-column and 64-row alignment of frames up to 65535 x 65535)
    # the demuxers ffmpeg may use (-format_whitelist; "mov,mp4,m4a,..." matches by any of its names): media containers
    # that read nothing but the file (mov's external tracks, enable_drefs, are off by default). Not DASH / HLS /
    # concat: a manifest or playlist opens other files and URLs (ffmpeg 6's dash outside -protocol_whitelist too)
    FORMATS = "mov,mp4,matroska,webm,avi,mpegts,mpeg,ogg,flv,gif"

    def __init__(self, exe):
        self.exe = exe
        self.version = ""

    def check(self, timeout=20):
        """runs `ffmpeg -version`; the first line of its output (RuntimeError if it does not run)"""
        try:
            r = subprocess.run([self.exe, "-hide_banner", "-version"], stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"{self.exe}: {e}")
        if r.returncode != 0:
            raise RuntimeError(f"{self.exe} -version: exit {r.returncode}")
        self.version = (r.stdout.decode(errors="replace").splitlines() or ["?"])[0]
        return self.version

    def frames(self, data, fps, max_frames, timeout=None):
        """yields (width, height, rgb bytes) per frame; BadRequest for a video ffmpeg cannot decode, one of more than
        max_frames frames (when the next frame arrives) or a decode over `timeout` seconds"""
        timeout = VIDEO_DECODE_S if timeout is None else timeout
        with tempfile.TemporaryDirectory(prefix="mkvideo") as d:
            # (FORMATS only: a "video" that is a manifest, a playlist or a concat list cannot make ffmpeg open other
            # files or URLs; the file protocol only and a name without an extension stay as a second layer)
            path = os.path.join(d, "input")
            with open(path, "wb") as f:
                f.write(data)
            cmd = [self.exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
                   "-format_whitelist", self.FORMATS,
                   "-max_pixels", str(min(mk_limits.IMAGE_MAX_PIXELS + self.MAX_PIXELS_SLACK, 2 ** 31 - 1)), "-i", path,
                   "-an", "-sn", "-vf", f"fps={fps:.6f}", "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24", "pipe:1"]
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as e:
                raise BadRequest(f"video: cannot run ffmpeg ({e})")
            err = bytearray()

            def drain():  # (stderr in a thread: a chatty ffmpeg must not block on a full pipe)
                for line in p.stderr:
                    err.extend(line)
                    del err[:-4096]
            th = threading.Thread(target=drain, name="mk-ffmpeg-stderr", daemon=True)
            th.start()
            killed = threading.Event()

            def kill():
                killed.set()
                p.kill()
            timer = threading.Timer(timeout, kill)
            timer.start()
            n = 0
            try:
                while True:
                    try:
                        fr = read_ppm(p.stdout, mk_limits.IMAGE_MAX_PIXELS)
                    except FrameTooLarge as e:
                        raise BadRequest(f"video: {e}")
                    except ValueError as e:
                        if killed.is_set():
                            break
                        raise BadRequest(f"video: bad frame stream from ffmpeg ({e})")
                    if fr is None:
                        break
                    n += 1
                    if n > max_frames:
                        raise BadRequest(f"video of more than {max_frames} frames at {fps:g} fps (MK_MAX_VIDEO_FRAMES "
                                         f"{mk_limits.MAX_VIDEO_FRAMES} per request)")
                    yield fr
            finally:
                timer.cancel()
                if p.poll() is None:
                    p.kill()
                p.stdout.close()
                p.wait()
                th.join(5)
                if not th.is_alive():
                    p.stderr.close()
            if killed.is_set():
                raise BadRequest(f"video: decoding took over {timeout} s")
            big = re.search(r"Picture size (\d+)x(\d+) exceeds specified max pixel count", err.decode(errors="replace"))
            if big:  # (-max_pixels: ffmpeg did not decode a frame, also after others of a smaller size)
                raise BadRequest(f"video: frames of about {big[1]} x {big[2]} pixels: over the limit of {mk_limits.IMAGE_MAX_PIXELS} "
                                 f"pixels (MK_IMAGE_MAX_PIXELS)")
            fmt = re.search(r"\[([\w,]+) @ \w+\] Format not on whitelist", err.decode(errors="replace"))
            if fmt and n == 0:  # (-format_whitelist: a manifest, a playlist, another container)
                raise BadRequest(f"video: {fmt[1][:40]} input is refused (the containers read: "
                                 f"{self.FORMATS.replace(',', ', ')})")
            if n == 0:
                msg = err.decode(errors="replace").strip().splitlines()
                raise BadRequest("video: no frames decoded" + (f" ({msg[-1][:300]})" if msg else ""))

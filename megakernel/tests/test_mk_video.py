"""Tests of server/mk_server.py's video input without a GPU: content parts, llama.cpp's frame layout and timestamps,
the PPM frame stream, the ffmpeg decoder (a fake process; the real ffmpeg when one is installed), the frame budget,
caching / pinning of a video's frame pairs, and requests end to end over a scripted engine and a fake vision library."""
import base64
import http.server
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from test_mk_server import _FakeVisionLib, _HttpCase, _load_mk_server, _ScriptEngine  # (tests/ is on sys.path)


def _ppm(w, h, fill=0):
    return b"P6\n%d %d\n255\n" % (w, h) + bytes([fill]) * (w * h * 3)


class VideoLayoutTests(unittest.TestCase):
    """video_parts / video_timestamp: llama.cpp b10246's layout (mtmd-helper.cpp read_next + mtmd_tokenize's merge)"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _layout(self, n, fps=4.0, sizes=None):
        frames = [((sizes or {}).get(i, (64, 48)) + (i,)) for i in range(n)]
        out = []
        for kind, x in self.mk.video_parts(iter(frames), fps):
            out.append(x if kind == "text" else tuple(f[2] for f in x if f is not None))
        return out

    def test_frame_0_is_alone_then_pairs_and_a_timestamp_after_each_5_s_mark(self) -> None:
        self.assertEqual(self._layout(1), ["Video:", (0,), "[0m0.00s]"])
        self.assertEqual(self._layout(3), ["Video:", (0,), "[0m0.00s]", (1, 2)])
        self.assertEqual(self._layout(16), ["Video:", (0,), "[0m0.00s]"] + [(i, i + 1) for i in range(1, 15, 2)] + [(15,)])
        pairs = [(i, i + 1) for i in range(1, 21, 2)]  # (frame 20 = 5.00 s ends the first run: its stamp follows it)
        self.assertEqual(self._layout(21), ["Video:", (0,), "[0m0.00s]"] + pairs + ["[0m5.00s]"])
        self.assertEqual(self._layout(24), ["Video:", (0,), "[0m0.00s]"] + pairs + ["[0m5.00s]", (21, 22), (23,)])

    def test_timestamps_use_float32_math_and_minutes(self) -> None:
        ts = self.mk.video_timestamp
        self.assertEqual(ts(240, self.mk.f32(4.0), 60000.0)[0], "[1m0.00s]")
        self.assertEqual(ts(13, self.mk.f32(2.5), 5000.0)[0], "[0m5.20s]")
        self.assertEqual(ts(19, self.mk.f32(4.0), 5000.0), (None, 5000.0))
        # a slow rate stamps every frame (each reaches the next 5 s mark), the threshold advancing 5 s at a time
        lay = self._layout(3, fps=0.1)
        self.assertEqual(lay, ["Video:", (0,), "[0m0.00s]", (1,), "[0m10.00s]", (2,), "[0m20.00s]"])

    def test_frames_of_different_sizes_do_not_merge(self) -> None:
        self.assertEqual(self._layout(5, sizes={2: (32, 32)}), ["Video:", (0,), "[0m0.00s]", (1,), (2,), (3, 4)])


class VideoPartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def test_input_video_and_video_url_parts(self) -> None:
        mk, b64 = self.mk, base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
        for part in ({"type": "input_video", "input_video": {"data": b64}},
                     {"type": "input_video", "input_video": {"url": "data:video/mp4;base64," + b64}},
                     {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}}):
            media = []
            msgs = mk.normalize_messages([{"role": "user", "content": [{"type": "text", "text": "what?"}, part]}], media)
            self.assertEqual(msgs[0]["content"], "what?" + mk.MEDIA_MARKER)
            self.assertIsInstance(media[0], mk.VideoBytes)
            self.assertEqual(bytes(media[0])[4:8], b"ftyp")

    def test_bad_video_parts_are_400(self) -> None:
        mk = self.mk
        for part, msg in (({"type": "input_video", "input_video": {"url": "https://example.com/a.mp4"}}, "https://"),
                          ({"type": "video_url", "video_url": {"url": "file:///etc/passwd"}}, "file://"),
                          ({"type": "input_video", "input_video": {}}, "expected base64"),
                          ({"type": "video_url", "video_url": {"url": "data:video/mp4,AAAA"}}, "base64 data: URL")):
            with self.assertRaises(mk.BadRequest) as cm:
                mk.normalize_messages([{"role": "user", "content": [part]}], [])
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(mk.BadRequest) as cm:  # (a server without --mmproj)
            mk.normalize_messages([{"role": "user", "content": [{"type": "input_video", "input_video": {"data": "AAAA"}}]}])
        self.assertIn("video input is not supported", str(cm.exception))


class _FakeProc:
    """subprocess.Popen of a fake ffmpeg: stdout / stderr from bytes"""

    def __init__(self, out, err=b"", code=0):
        self.stdout, self.stderr, self.returncode, self.killed = io.BufferedReader(io.BytesIO(out)), io.BytesIO(err), code, False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


class VideoDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def test_ppm_stream(self) -> None:
        mk = self.mk
        f = io.BytesIO(_ppm(4, 2, 7) + b"P6\n# comment\n3 1\n255\n" + bytes(range(9)))
        self.assertEqual(mk.read_ppm(f), (4, 2, bytes([7]) * 24))
        self.assertEqual(mk.read_ppm(f), (3, 1, bytes(range(9))))
        self.assertIsNone(mk.read_ppm(f))
        for bad in (b"P5\n1 1\n255\n\x00", b"P6\n2 2\n65535\n" + bytes(24), _ppm(2, 2)[:-1]):
            with self.assertRaises(ValueError):
                mk.read_ppm(io.BytesIO(bad))

    def _frames(self, proc, max_frames=10):
        dec = self.mk.VideoDecoder("ffmpeg")
        with mock.patch.object(self.mk.subprocess, "Popen", return_value=proc) as popen:
            out = list(dec.frames(b"video", 4.0, max_frames))
        return out, popen.call_args[0][0]

    def test_frames_through_ffmpegs_fps_filter_as_a_ppm_stream(self) -> None:
        out, cmd = self._frames(_FakeProc(_ppm(4, 2, 1) + _ppm(4, 2, 2)))
        self.assertEqual([(w, h, d[0]) for w, h, d in out], [(4, 2, 1), (4, 2, 2)])
        self.assertEqual(cmd[cmd.index("-vf") + 1], "fps=4.000000")  # (llama.cpp's filter)
        self.assertIn("ppm", cmd)
        self.assertEqual(cmd[cmd.index("-protocol_whitelist") + 1], "file")  # (no URLs from inside a "video")
        self.assertLess(cmd.index("-protocol_whitelist"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-format_whitelist") + 1], self.mk.VideoDecoder.FORMATS)  # (no manifests)
        self.assertLess(cmd.index("-format_whitelist"), cmd.index("-i"))

    def test_too_many_frames_no_frames_and_bad_streams_are_400(self) -> None:
        mk = self.mk
        with self.assertRaises(mk.BadRequest) as cm:
            self._frames(_FakeProc(_ppm(2, 2) * 3), max_frames=2)
        self.assertIn("more than 2 frames", str(cm.exception))
        with self.assertRaises(mk.BadRequest) as cm:
            self._frames(_FakeProc(b"", b"[in] moov atom not found\ninput: Invalid data\n", 1))
        self.assertIn("no frames decoded (input: Invalid data)", str(cm.exception))
        with self.assertRaises(mk.BadRequest) as cm:
            self._frames(_FakeProc(b"garbage!"))
        self.assertIn("bad frame stream", str(cm.exception))

    def test_ffmpeg_discovery(self) -> None:
        mk = self.mk
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            open(exe, "wb").close()
            self.assertEqual(mk.find_ffmpeg(exe), exe)
            self.assertEqual(mk.find_ffmpeg(d), exe)  # (a directory)
            with mock.patch.dict(os.environ, {"MK_FFMPEG": d}):
                self.assertEqual(mk.find_ffmpeg(), exe)
            self.assertIsNone(mk.find_ffmpeg(os.path.join(d, "missing")))
        with mock.patch.dict(os.environ, {"MK_FFMPEG": ""}), mock.patch.object(mk.shutil, "which", return_value=None), \
                mock.patch.dict("sys.modules", {"imageio_ffmpeg": None}):
            self.assertIsNone(mk.find_ffmpeg())  # (nothing installed: video input off)

    @unittest.skipUnless(shutil.which("ffmpeg"), "no ffmpeg on PATH")
    def test_a_real_clip(self) -> None:
        mk = self.mk
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "c.mkv")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
                            "-c:v", "ffv1", clip], check=True, timeout=60)
            data = open(clip, "rb").read()
        dec = mk.VideoDecoder(shutil.which("ffmpeg"))
        self.assertIn("ffmpeg", dec.check())
        frames = list(dec.frames(data, 4.0, 100))
        self.assertEqual([(w, h, len(b)) for w, h, b in frames], [(64, 48, 64 * 48 * 3)] * 8)
        with self.assertRaises(mk.BadRequest):
            list(dec.frames(b"not a video at all", 4.0, 100))

    @unittest.skipUnless(shutil.which("ffmpeg"), "no ffmpeg on PATH")
    def test_manifests_and_playlists_are_refused_media_containers_decode(self) -> None:
        mk, ff, hits = self.mk, shutil.which("ffmpeg"), []

        class Listener(http.server.BaseHTTPRequestHandler):  # (records what ffmpeg asks for)
            def do_GET(self):
                hits.append(self.path)
                self.send_error(404)

            def log_message(self, *a):
                pass
        lis = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Listener)
        threading.Thread(target=lis.serve_forever, daemon=True).start()
        self.addCleanup(lis.server_close)
        self.addCleanup(lis.shutdown)
        dec = mk.VideoDecoder(ff)
        with tempfile.TemporaryDirectory() as d:
            def clip(name, *args):  # (1 s; None: this ffmpeg has no encoder for it)
                p = os.path.join(d, name)
                src = f"testsrc=size=64x48:rate={25 if name.endswith('.mpg') else 4}:duration=1"  # (MPEG-1: no 4 fps)
                r = subprocess.run([ff, "-loglevel", "error", "-f", "lavfi", "-i", src, *args, p], capture_output=True,
                                   timeout=60)
                return open(p, "rb").read() if r.returncode == 0 else None

            def mpd(base):  # (a DASH manifest of one representation: the whole file at BaseURL)
                return ('<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
                        'mediaPresentationDuration="PT1S" minBufferTime="PT1S" '
                        'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"><Period>'
                        '<AdaptationSet mimeType="video/mp4"><Representation id="1" bandwidth="100000" width="64" '
                        f'height="48" codecs="mp4v.20.9"><BaseURL>{base}</BaseURL></Representation></AdaptationSet>'
                        '</Period></MPD>').encode()
            self.assertIsNotNone(clip("local.mp4", "-c:v", "mpeg4"))
            local = Path(d, "local.mp4").as_uri()
            url = f"http://127.0.0.1:{lis.server_address[1]}/probe.mp4"
            hls = f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{url}\n#EXT-X-ENDLIST\n".encode()
            for name, data in (("dash, a local file", mpd(local)), ("dash, a URL", mpd(url)), ("hls", hls),
                               ("concat", f"ffconcat version 1.0\nfile '{local}'\n".encode())):
                with self.assertRaises(mk.BadRequest, msg=name) as cm:
                    list(dec.frames(data, 4.0, 100))
                if name.startswith("dash"):
                    self.assertIn("video: dash input is refused (the containers read: mov, mp4, ", str(cm.exception))
            self.assertEqual(hits, [])  # (no request reached the listener)
            # the containers read (webm and ogg need libvpx / libtheora in this ffmpeg)
            codecs = {"mp4": "mpeg4", "mov": "mpeg4", "mkv": "ffv1", "webm": "libvpx", "avi": "mjpeg",
                      "ts": "mpeg2video", "mpg": "mpeg1video", "ogv": "libtheora", "flv": "flv1", "gif": "gif"}
            decoded = set()
            for ext, codec in codecs.items():
                data = clip("c." + ext, "-c:v", codec)
                if data is not None:
                    frames = list(dec.frames(data, 4.0, 100))
                    self.assertEqual({(w, h) for w, h, _ in frames}, {(64, 48)}, ext)
                    self.assertIn(len(frames), (4, 5), ext)
                    decoded.add(ext)
            self.assertLessEqual({"mp4", "mov", "mkv", "avi", "ts", "mpg", "flv", "gif"}, decoded)


class _VideoEngine(_ScriptEngine):
    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        self.calls.append(("images", tuple((p, nx, ny) for p, nx, ny, _ in images)))
        self.last_toks = list(toks)
        return super().prefill(toks, ck_at, cks)


class _FakeVideoLib(_FakeVisionLib):
    """+ mkv_frames_load: every frame / pair a 2 x 1 grid; the frames it got are recorded"""

    def __init__(self):
        super().__init__()
        self.loaded = []

    def mkv_frames_load(self, ctx, a, b, w, h, max_tokens):
        self.loaded.append((a[:1], b[:1] if b else None, w, h, max_tokens))
        return self.mkv_image_load(ctx, a + (b or b""), 0)

    def mkv_image_grid(self, h, nx, ny):
        nx._obj.value, ny._obj.value = 2, 1
        return 2


class _FakeDecoder:
    def __init__(self, n, bad_request):
        self.n, self.calls, self.bad = n, 0, bad_request

    def frames(self, data, fps, max_frames):
        self.calls += 1
        for i in range(self.n):
            if i + 1 > max_frames:
                raise self.bad(f"video of more than {max_frames} frames")
            yield 64, 48, bytes([i]) * 4


class VideoRequestTests(_HttpCase):
    """input_video end to end: token layout, image positions, caching, limits, and the 400s without a decoder"""

    def serve_video(self, n_frames=5, decoder=True):
        eng = _VideoEngine(["ok"])
        srv, http_ = self.serve(eng)
        lib = _FakeVideoLib()
        v = object.__new__(self.mk.Vision)
        v.L, v.ctx, v.cache, v.cache_max = lib, 1, {}, 8
        v.frames_ok, v.videos, v.video_cache_max = True, {}, 1
        srv.vision, srv.vis_start, srv.vis_end, srv.img_pad = v, 1001, 1002, 1003
        srv.args.video_fps, srv.args.video_max_tokens = 4.0, 0
        if decoder:
            srv.decoder, srv.video_ok = _FakeDecoder(n_frames, self.mk.BadRequest), True
        return srv, eng, v, lib, http_

    def _req(self, data=b"clip"):
        part = {"type": "input_video", "input_video": {"data": base64.b64encode(data).decode()}}
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": "Q"}, part]}], "max_tokens": 2}

    def test_a_video_is_laid_out_as_llama_cpp_does(self) -> None:
        srv, eng, v, lib, http_ = self.serve_video(n_frames=5)
        status, data = http_.post(self._req())
        self.assertEqual(status, 200, data[:300])
        # Video: [F0] [0m0.00s] [F1 F2] [F3 F4]: each text on its own, each frame / pair as an image of 2 x 1
        pre, t = [ord(c) for c in "<|im_start|>user\nQ"], [ord(c) for c in "Video:"]
        img = [1001, 1003, 1003, 1002]
        prompt = pre + t + img + [ord(c) for c in "[0m0.00s]"] + img + img
        toks = eng.last_toks
        self.assertEqual(toks[:len(prompt)], prompt)
        self.assertEqual([(a[0], b[0] if b else None) for a, b, *_ in lib.loaded], [(0, None), (1, 2), (3, 4)])
        p0 = len(pre) + len(t) + 1
        p1 = p0 + 3 + 9 + 1
        self.assertEqual([c for c in eng.calls if c[0] == "images"][-1], ("images", ((p0, 2, 1), (p1, 2, 1), (p1 + 4, 2, 1))))
        # the same video again: decoded once (cached), its images released after each request
        self.assertEqual(http_.post(self._req())[0], 200)
        self.assertEqual((srv.decoder.calls, len(lib.loaded)), (1, 3))
        self.assertTrue(all(im.pins == 0 for vid in v.videos.values() for im in vid.images))
        # another video: the cache keeps one, the first one's images are freed
        self.assertEqual(http_.post(self._req(b"clip2"))[0], 200)
        self.assertEqual((len(v.videos), len(lib.freed), lib.misuse), (1, 3, []))

    def test_frame_budget_is_a_400(self) -> None:
        srv, eng, v, lib, http_ = self.serve_video(n_frames=5)
        with mock.patch.object(self.mk, "MAX_VIDEO_FRAMES", 4):
            status, data = http_.post(self._req())
        self.assertEqual(status, 400)
        self.assertIn(b"more than 4 frames", data)
        self.assertEqual((lib.live, lib.misuse), ({}, []))  # (the frames loaded before the limit are freed)

    def test_no_decoder_is_a_400_that_names_ffmpeg(self) -> None:
        srv, eng, v, lib, http_ = self.serve_video(decoder=False)
        status, data = http_.post(self._req())
        self.assertEqual(status, 400)
        self.assertIn(b"ffmpeg", json.loads(data)["error"]["message"].encode())
        self.assertFalse(srv.video_ok)


if __name__ == "__main__":
    unittest.main()

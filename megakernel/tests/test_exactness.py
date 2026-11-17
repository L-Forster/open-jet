"""Tests of scripts/exactness.py without a GPU: stub programs stand in for qwen38_mk and for the Python that runs
server/mk_server.py, and STUB_* variables make them break one claim at a time."""
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "exactness.py")

# qwen38_mk: ids and a digest from the prompt (+ STUB_SALT); STUB_BREAK=d<n> / cap<n> changes them at that depth / chunk;
# depths above 0 report drafts unless STUB_NO_DRAFTS=1
STUB_CLI = r'''
import hashlib, os, re, shlex, sys

def run(args, out, err):
    prompt, depth, sampled, dig, policy = "default", "2", False, False, ""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-p": prompt = args[i + 1]; i += 1
        elif a == "-f": prompt = open(args[i + 1]).read(); i += 1
        elif a == "-d": depth = args[i + 1]; i += 1
        elif a == "--temp": sampled = True; i += 1
        elif a == "--digest": dig = True
        elif a == "--policy": policy = args[i + 1]; i += 1
        elif a in ("-n", "-o", "--top-p", "--top-k", "--seed"): i += 1
        i += 1
    kv = os.environ.get("MK_KV", "")
    brk = os.environ.get("STUB_BREAK", "") in ("d" + depth, "cap" + os.environ.get("MK_PF_CAP", "-"), "policy " + policy,
                                               kv + "/d" + depth if kv else "-", kv + "/cap" + os.environ.get("MK_PF_CAP", "-") if kv else "-")
    h = hashlib.sha256((os.environ.get("STUB_SALT", "") + kv + prompt + ("s" if sampled else "")).encode()).digest()
    ids = [str(b) for b in h[:12]]
    if brk: ids[5] = "99999"
    out.write("IN_IDS: 1 2 3\nIDS: " + " ".join(ids) + "\n")
    err.write("[mk] prompt %d tok in 0.1 s\n" % len(prompt.split()))
    drafts = 0 if depth == "0" or os.environ.get("STUB_NO_DRAFTS") else 6
    err.write("[mk] decode 12 tok in 0.1 s = 1.00 tok/s | cycles 4, accepted %d/%d drafts (50.0%%), 3.00 tok/cycle\n"
              % (drafts // 2, drafts))
    if dig: err.write("[mk] state digest after the prefill: %s (9 words)\n" % (h[12:20].hex() if not brk else "0" * 16))

if os.environ.get("STUB_FAIL"):
    sys.stderr.write("[mk] error: stub failure\n")
    sys.exit(1)
argv = sys.argv[1:]
if os.environ.get("MK_KV_WIN") and os.environ.get("STUB_BREAK") != "nowindow":
    sys.stderr.write("[mk] KV recent window: the last 2561..3200 positions of 32 low-bit sides stay q8\n")
if "--no-mtp" in argv:
    sys.stderr.write("[mk] MTP layer off (--no-mtp / MK_NO_MTP=1): plain decoding, depth 0\n")
if "--batch" in argv:
    for line in open(argv[argv.index("--batch") + 1]):
        job = shlex.split(line)
        prefix = job[job.index("-o") + 1]
        with open(prefix + ".out", "w") as o, open(prefix + ".err", "w") as e:
            run(job, o, e)
else:
    run(argv, sys.stdout, sys.stderr)
'''

# the server's Python: answers /health and chat requests, reporting reused prompt tokens from a server's second request
# on; STUB_BREAK=warm changes every reply after a server's first, STUB_BREAK=nocache reports no reuse, STUB_BREAK=solo<n>
# changes the replies of a server started with MK_PF_SOLO=n, STUB_BREAK=nolog logs no park or side request
STUB_SERVER = r'''
import hashlib, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
served = [0]
if os.environ.get("MK_KV_WIN") and os.environ.get("STUB_BREAK") != "nowindow":
    print("[mk] KV recent window", flush=True)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.send({"status": "ok"})
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/tokenize":
            self.send({"tokens": json.loads(body)["content"].split()})
            return
        served[0] += 1
        if served[0] > 1 and os.environ.get("STUB_BREAK") != "nolog":
            print("#2 parked the cached conversation\n#2 side request done", flush=True)
        key = os.environ.get("STUB_SALT", "") + os.environ.get("MK_KV", "") + json.dumps(json.loads(body), sort_keys=True)
        text = hashlib.sha256(key.encode()).hexdigest()[:24]
        if os.environ.get("STUB_BREAK") == "warm" and served[0] > 1: text += "!"
        if os.environ.get("STUB_BREAK") == "solo" + os.environ.get("MK_PF_SOLO", "-"): text += "?"
        cached = 0 if served[0] == 1 or os.environ.get("STUB_BREAK") == "nocache" else 2000
        self.send({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": cached}}})

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ExactnessScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.name == "nt":
            raise unittest.SkipTest("the stubs run as #! scripts")
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        cls.cli, cls.py, cls.model = (os.path.join(d, n) for n in ("qwen38_mk", "python", "model.gguf"))
        for path, src in ((cls.cli, STUB_CLI), (cls.py, STUB_SERVER)):
            with open(path, "w") as f:
                f.write(f"#!{sys.executable}\n{src}")
            os.chmod(path, 0o755)
        with open(cls.model, "wb") as f:
            f.write(b"GGUF stub")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_script(self, out, *args, **env):
        cmd = [sys.executable, SCRIPT, "-m", self.model, "--bin", self.cli, "--python", self.py, "--port", str(free_port()),
               "--out", os.path.join(self.tmp.name, out), "--timeout", "60", "--long-min", "5", "--deep-tokens", "400", "--full-tokens", "600", *args]
        env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")} | env
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        return r.returncode, r.stdout + r.stderr

    def test_every_check_passes_on_an_exact_build(self) -> None:
        rc, out = self.run_script("pass")
        self.assertEqual(rc, 0, out)
        # depth 2 + D, F; mtp 2; long q8 2 + D, F; policy 1; chunks 2 per cache; server 3 per cache; per cache: park 3 + side 2,
        # resume 3, deep 4; full 1; agents 1
        self.assertEqual(out.count("PASS "), 4 + 2 + 4 + 1 + 6 + 9 + 15 + 9 + 12 + 1 + 1, out)
        self.assertIn("64 passed, 0 failed", out)

    def test_a_depth_that_changes_the_ids_fails(self) -> None:
        rc, out = self.run_script("d4", "--legs", "depth,mtp", STUB_BREAK="d4")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL depth, code prompt: greedy ids identical at 0, 4, 7 drafts", out)
        self.assertIn("d4 vs d0: first difference at output token 5", out)
        self.assertIn("PASS mtp, code prompt", out)

    def test_a_depth_that_made_no_drafts_fails(self) -> None:
        rc, out = self.run_script("nodraft", "--legs", "depth,mtp", STUB_NO_DRAFTS="1")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL depth, "), 4, out)  # (two prompts, and the presets D and F)
        self.assertIn("d4, d7 made no drafts (no MTP layer in the model, or MK_NO_MTP is set), so depth independence is untested", out)
        self.assertEqual(out.count("FAIL mtp, "), 2, out)  # (both of its sides then ran plain decoding)
        self.assertIn("d7 made no drafts", out)

    def test_a_chunk_size_that_changes_the_state_fails(self) -> None:
        rc, out = self.run_script("cap", "--legs", "chunks", STUB_BREAK="cap1280")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL chunks: MK_EXACT state digest", out)
        self.assertIn("chunk 1280: 0000000000000000", out)
        self.assertIn("FAIL chunks D: MK_EXACT state digest", out)  # (STUB_BREAK has no cache in it: every cache's chunks break)
        self.assertIn("FAIL chunks: greedy ids identical", out)

    def test_a_warm_server_that_answers_differently_fails(self) -> None:
        rc, out = self.run_script("warm", "--legs", "server", STUB_BREAK="warm")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL server, "), 3, out)
        self.assertEqual(out.count("FAIL server D, "), 3, out)
        self.assertEqual(out.count("FAIL server F, "), 3, out)

    def test_a_warm_server_that_reused_nothing_fails(self) -> None:
        rc, out = self.run_script("nocache", "--legs", "server", STUB_BREAK="nocache")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL server, "), 3, out)
        self.assertIn("the warm server reused nothing (usage.prompt_tokens_details.cached_tokens 0, needed 1), so warm == fresh proves nothing", out)

    def test_a_preset_depth_that_changes_the_ids_fails(self) -> None:
        rc, out = self.run_script("presetd", "--legs", "depth", STUB_BREAK="K*=5r,V*=5/d7")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL depth, preset F past its window: greedy ids identical at 0, 4, 7 drafts", out)
        self.assertIn("PASS depth, preset D past its window", out)

    def test_a_preset_that_did_not_take_effect_fails(self) -> None:
        rc, out = self.run_script("nowin", "--legs", "chunks", "--presets", "F", STUB_BREAK="nowindow")
        self.assertEqual(rc, 1, out)
        self.assertIn("the engine reports no KV window: the preset did not take effect", out)
        rc, out = self.run_script("nowin2", "--legs", "server", "--presets", "D", STUB_BREAK="nowindow")
        self.assertEqual(rc, 1, out)
        self.assertIn("the engine reports no KV window: the preset did not take effect", out)

    def test_a_long_run_that_changes_the_ids_fails(self) -> None:
        rc, out = self.run_script("long", "--legs", "long", "--presets", "q8", STUB_BREAK="d7")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL long, q8, "), 2, out)

    def test_a_long_run_that_stopped_early_fails(self) -> None:
        rc, out = self.run_script("short", "--legs", "long", "--presets", "q8", "--long-min", "1000")
        self.assertEqual(rc, 1, out)
        self.assertIn("only 12 tokens at no drafts (--long-min 1000), so a late near-tie is untested", out)

    def test_a_draft_policy_that_changes_the_ids_fails(self) -> None:
        rc, out = self.run_script("pol", "--legs", "long,policy", "--presets", "q8", STUB_BREAK="policy stop=0.9,drop=0.9")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL policy: greedy ids under every draft policy equal no drafts", out)
        self.assertIn("tight gates vs no drafts: first difference at output token 5", out)

    def test_policy_needs_the_long_check(self) -> None:
        rc, out = self.run_script("bad2", "--legs", "policy")
        self.assertEqual(rc, 2, out)
        self.assertIn("policy compares against the long check's ids", out)

    def test_a_park_that_answers_differently_or_never_parked_fails(self) -> None:
        rc, out = self.run_script("parkw", "--legs", "park", "--presets", "q8", STUB_BREAK="warm")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL park, "), 5, out)
        rc, out = self.run_script("parkn", "--legs", "park", "--presets", "q8", STUB_BREAK="nolog")
        self.assertEqual(rc, 1, out)
        self.assertIn("the log shows no park, so nothing was tested", out)
        self.assertIn("the log shows no side request, so nothing was tested", out)

    def test_a_resume_that_reused_nothing_fails(self) -> None:
        rc, out = self.run_script("resn", "--legs", "resume", "--presets", "q8", STUB_BREAK="nocache")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL resume, "), 3, out)

    def test_a_deep_reuse_that_answers_differently_fails(self) -> None:
        rc, out = self.run_script("deep", "--legs", "deep", "--presets", "F", STUB_BREAK="warm")
        self.assertEqual(rc, 1, out)
        self.assertEqual(out.count("FAIL deep F, "), 4, out)
        self.assertIn("edit inside the prompt", out)

    def test_full_only_runs_with_preset_f(self) -> None:
        rc, out = self.run_script("fullq", "--legs", "full", "--presets", "q8")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("full F", out)
        rc, out = self.run_script("fullf", "--legs", "full", "--presets", "F", STUB_BREAK="warm")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL full F, follow-up", out)

    def test_a_prompt_fed_in_slot_chunks_that_answers_differently_fails(self) -> None:
        rc, out = self.run_script("solo", "--legs", "agents", STUB_BREAK="solo100")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL agents: the chunks prompt fed in slot prefill chunks of 100 (MK_PF_SOLO)", out)

    def test_a_failing_cli_is_reported_with_its_error(self) -> None:
        rc, out = self.run_script("fail", "--legs", "depth", STUB_FAIL="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL depth_d0: qwen38_mk ended with 1", out)
        self.assertIn("[mk] error: stub failure", out)

    def test_compare_finds_another_build_that_differs(self) -> None:
        self.assertEqual(self.run_script("ref")[0], 0)
        ref = os.path.join(self.tmp.name, "ref")
        rc, out = self.run_script("same", "--compare", ref)
        self.assertEqual(rc, 0, out)
        m = re.search(r"PASS compare: (\d+) of (\d+) results identical", out)
        self.assertTrue(m and m[1] == m[2], out)
        rc, out = self.run_script("other", "--compare", ref, STUB_SALT="x")
        self.assertEqual(rc, 1, out)
        self.assertIn(f"FAIL compare: 0 of {m[2]} results identical", out)

    def test_mtp_needs_the_depth_check(self) -> None:
        rc, out = self.run_script("bad", "--legs", "mtp")
        self.assertEqual(rc, 2, out)
        self.assertIn("mtp compares against the depth check's ids", out)


if __name__ == "__main__":
    unittest.main()

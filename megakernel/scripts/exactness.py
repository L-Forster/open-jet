#!/usr/bin/env python3
"""Checks the README's "Exact output" claims on this build, and with --compare that another build's results are identical."""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
LEGS = ("depth", "mtp", "long", "policy", "chunks", "server", "park", "resume", "deep", "full", "agents")
# the KV caches the launchers offer (scripts/serve.bat uses F above 240k); a prompt inside the 8-bit window never reads a low-bit row
PRESETS = {"q8": {}, "D": {"MK_KV": "K*=6,V*=5", "MK_KV_WIN": "2048"}, "F": {"MK_KV": "K*=5r,V*=5", "MK_KV_WIN": "2048"}}
# every MK_POLICY key only chooses drafts (docs/CONFIGURATION.md): the first is the defaults, so it comes first in a --batch
POLICIES = (("default", ""), ("no gates", "depth=0,drop=0,stop=0,chain=0,h1=0"), ("adaptive depth", "depth=2"),
            ("tight gates", "stop=0.9,drop=0.9"), ("loose gates", "stop=0.05,drop=0.05"))
ESSAY = "Write a detailed essay of at least 2,000 words on how lighthouses were built, lit and kept, from antiquity to automation."
GREEDY, SAMPLED_REQ = {"temperature": 0}, {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}
PROMPTS = {"code": "Write a C++ function that parses a GGUF header.",
           "story": "Tell me a short story about a lighthouse keeper."}
SAMPLED = ["--temp", "1", "--top-p", "0.95", "--top-k", "20", "--seed", "7"]
CHUNKS = (2048, 1280, 1024, 768)  # MK_PF_CAP values: multiples of 128, all >= 640 (smaller caps take another end-snapshot path); 1024 is the launchers' above 96k
SOLO = (100, 0)  # MK_PF_SOLO values: slot prefill chunks of 100 tokens (not a multiple of 32), the whole prompt
# the server checks' system message: with exact reuse a request resumes at a checkpoint (a 64-token grid and each message
# start), so the first turn must span several of them for the follow-up to reuse anything, as an agent's system prompt does
SYSTEM = " ".join([
    "You are a careful assistant for a lighthouse museum.",
    "Answer in plain English, without markdown, headings or lists unless the user asks for one.",
    "Keep answers short: a few lines unless more is requested, and never pad them with filler.",
    "When you are unsure of a fact about lighthouses, keepers or navigation, say so instead of guessing.",
    "Prefer concrete details: the name of a lighthouse, the year it was lit, the range of its light.",
    "When asked for verse, keep the meter simple and the rhymes natural, and do not explain the poem.",
    "Visitors range from children to retired sailors, so avoid jargon or explain it in a few words.",
    "If a request is outside the museum's subject, answer it briefly and politely all the same.",
] * 3)
U1 = "Name three uses of a lighthouse, one line each."
U2 = "Now write a four-line poem about the second one."
U2_EDIT = "Now write a four-line poem about the third one."
U3 = "Which lighthouse is the oldest still in use?"
U4 = "How did a keeper trim a lamp?"
U5 = "Write two lines of verse about fog."
TITLE = "Give a three-word title for a talk on lighthouses."

DESCRIPTION = """\
Checks the exactness claims of README.md ("Exact output") on this build:
  depth   greedy token ids are identical at every draft depth of --depths (two prompts; the depths above 0 must draft);
          with a preset, on the ~5k-token prompt, past its 8-bit window
  mtp     greedy ids without the MTP layer (--no-mtp, depth 0) equal the drafted ones
  long    the same at no drafts and the deepest over --long-tokens of output: a near-tie can take 1,000 tokens to come up
          (each cache; a run shorter than --long-min fails: it would prove nothing)
  policy  greedy ids under every draft policy of POLICIES equal the long check's without drafts
  chunks  with exact reuse (MK_EXACT=1) the state digest and greedy ids after a ~5k-token prompt do not depend on the
          prefill chunk size (MK_PF_CAP; each cache)
  server  with --exact-reuse, a follow-up turn, an edited turn and a sampled follow-up (fixed seed) on a warm server
          equal the same request sent to a fresh server (and the warm server did reuse its cache; each cache)
  park    conversations parked in host memory and brought back, and one kept beside a side request, answer like a fresh
          server (the log must show the park and the side request; each cache)
  resume  new sessions resumed at the pinned system-prompt checkpoint, live and after a park, answer like a fresh server
  deep    the server check at ~--deep-tokens of context, as the launchers run it: 120k, MK_PF_CAP 1024, plus an edit inside
          the prompt (each cache)
  full    a follow-up at ~--full-tokens on preset F at 262,144, serve.bat's default up there
  agents  with --agents 1 (MK_EXACT), the chunks prompt fed in slot prefill chunks of 100 tokens (MK_PF_SOLO) gets the
          greedy reply it gets fed whole
Every result goes to <out>/exactness.json. --compare DIR then checks that another build's results (Windows against Linux,
or the previous release) are identical. Needs the GPU to itself. --presets narrows the caches, --legs the checks; deep and
full take most of the time, each fresh server reading the whole prompt again."""


class LegFailed(Exception):
    def __init__(self, what, detail=""):
        super().__init__(what)
        self.what, self.detail = what, detail


def long_prompt(functions=240):
    lines = ["Explain what each of these functions computes.", "", "```python"]
    for i in range(1, functions + 1):
        lines += [f"def step_{i}(x, y):", f"    return (x * {i} + y) % {997 + i}", ""]
    return "\n".join(lines + ["```"])


def quoted(s):
    """one argument of a --batch line: '...' is taken literally there (main refuses an --out with a quote)"""
    assert "'" not in s, s
    return f"'{s}'"


def tail(path, n=8):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).rstrip() or "(empty)"
    except OSError:
        return "(no output)"


def first_difference(a, b):
    if a is None or b is None:
        return "no ids"
    x, y = a.split()[1:], b.split()[1:]  # (after "IDS:")
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return f"first difference at output token {i}"
    return "same" if len(x) == len(y) else f"lengths {len(x)} / {len(y)}"


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_info():
    info = os.path.join(HERE, "BUILD_INFO.txt")  # (release packages; PowerShell's utf8 starts it with a BOM)
    if os.path.isfile(info):
        with open(info, encoding="utf-8-sig", errors="replace") as f:
            return f.readline().strip()
    try:
        r = subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def user(text):
    return {"role": "user", "content": text}


def assistant(reply):
    return {"role": "assistant", "content": reply["content"] or ""}


SYS = {"role": "system", "content": SYSTEM}
SYS_LONG = {"role": "system", "content": SYSTEM + "\n\n" + long_prompt()}  # (past a preset's 8-bit window)


def sys_other(i):
    return {"role": "system", "content": f"You work at museum number {i}. {SYSTEM}\n\n{long_prompt(240 + 8 * i)}"}


class Server:
    """server/mk_server.py with exact reuse on --port, more options and environment; stops it (and waits for its exit)
    when the block ends"""

    def __init__(self, chk, name, args=(), env=None, ctx=32768):
        self.chk, self.name, self.args, self.env, self.ctx = chk, name, list(args), env or {}, ctx
        self.url = f"http://127.0.0.1:{chk.a.port}"
        self.log = chk.path(name, "log")

    def __enter__(self):
        a = self.chk.a
        if port_open(a.port):
            raise LegFailed(f"{self.name}: port {a.port} is in use (--port)")
        cmd = [a.python, os.path.join(HERE, "server", "mk_server.py"), "-m", a.model, "--port", str(a.port), "-c", str(self.ctx),
               "--kv", "q8", "-d", str(max(a.depths)), "--exact-reuse"] + self.args
        if a.draft_vocab:
            cmd += ["--draft-vocab", a.draft_vocab]
        env = {k: v for k, v in os.environ.items() if k != "MK_API_KEY"}  # (its own server: the requests carry no key)
        self.logf = open(self.log, "wb")
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                                      env={**env, "MK_KEEP_WARM_S": "0", **self.env})
        except OSError as e:
            self.logf.close()
            raise LegFailed(f"{self.name}: cannot start {a.python} ({e})")
        end = time.monotonic() + a.timeout
        while time.monotonic() < end:
            if self.p.poll() is not None:
                self.logf.close()
                raise LegFailed(f"{self.name}: the server exited with code {self.p.returncode}", tail(self.log))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        if "MK_KV_WIN" in self.env and not self.logged(WINDOW_ON):
                            self.stop()
                            raise LegFailed(f"{self.name}: the engine reports no KV window: the preset did not take effect", tail(self.log))
                        return self
            except OSError:  # (refused while it starts; 503 while the model loads)
                pass
            time.sleep(0.5)
        self.stop()
        raise LegFailed(f"{self.name}: not ready after {a.timeout} s", tail(self.log))

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.logf.close()
        # (Windows: a venv's python.exe is a launcher, and its child, the server, is still exiting when the launcher is gone)
        end = time.monotonic() + 120
        while port_open(self.chk.a.port) and time.monotonic() < end:
            time.sleep(0.5)

    def logged(self, pattern):
        """the pattern anywhere in this server's log so far"""
        self.logf.flush()
        with open(self.log, encoding="utf-8", errors="replace") as f:
            return re.search(pattern, f.read()) is not None

    def count(self, text):
        """tokens of the text, by the server's own tokenizer"""
        req = urllib.request.Request(self.url + "/tokenize", json.dumps({"content": text}).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.chk.a.timeout) as r:
                return len(json.load(r)["tokens"])
        except (OSError, ValueError, KeyError) as e:
            raise LegFailed(f"{self.name}: /tokenize failed ({e})", tail(self.log))

    def fit(self, target):
        """long_prompt of about target tokens: within 3%, else a LegFailed (too short would test the wrong depth)"""
        n = max(8, target // 24)
        for _ in range(6):
            text = long_prompt(n)
            got = self.count(text)
            if abs(got - target) <= target * 0.03:
                return text, got
            n = max(8, n * target // max(got, 1))
        raise LegFailed(f"{self.name}: no prompt within 3% of {target} tokens ({got} with {n} functions)")

    def chat(self, messages, opts):
        body = dict(messages=messages, max_tokens=128, stream=False, chat_template_kwargs={"enable_thinking": False}, **opts)
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.chk.a.timeout) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            raise LegFailed(f"{self.name}: HTTP {e.code}", e.read()[:500].decode(errors="replace"))
        except (OSError, ValueError) as e:
            raise LegFailed(f"{self.name}: the request failed ({e})", tail(self.log))
        c = d["choices"][0]
        usage = d.get("usage") or {}
        reply = {"content": c["message"].get("content"), "reasoning": c["message"].get("reasoning_content"),
                 "finish": c.get("finish_reason"), "tokens": usage.get("completion_tokens")}
        # (the reused prompt tokens are returned apart from the reply: a fresh server's are 0, so they are not compared)
        return reply, (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


WINDOW_ON = r"KV recent window"  # the engine's load line for a window ring (loader.cu)


class Check:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        self.rec = {}
        self.passed = self.failed = 0
        os.makedirs(self.out, exist_ok=True)

    def path(self, name, ext):
        return os.path.join(self.out, f"{name}.{ext}")

    def verdict(self, ok, what, detail=""):
        print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok and detail:
            print("     " + detail.replace("\n", "\n     "), flush=True)
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def cli(self, name, args, env=None):
        """one qwen38_mk process, stdout / stderr to <name>.out / .err"""
        a = self.a
        cmd = [a.bin, "-m", a.model, "--kv", "q8"] + (["--draft-vocab", a.draft_vocab] if a.draft_vocab else []) + args
        with open(self.path(name, "out"), "wb") as o, open(self.path(name, "err"), "wb") as e:
            try:
                rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=o, stderr=e, timeout=a.timeout,
                                    env=dict(os.environ, **(env or {}))).returncode
            except subprocess.TimeoutExpired:
                rc = "a timeout"
        if rc != 0:
            raise LegFailed(f"{name}: qwen38_mk ended with {rc}", tail(self.path(name, "err")))
        if env and "MK_KV_WIN" in env and not self.err_has(name, WINDOW_ON):
            raise LegFailed(f"{name}: the engine reports no KV window: the preset did not take effect", tail(self.path(name, "err")))

    def batch(self, name, jobs, extra=(), n=256, ctx=8192):
        """jobs [(output name, job arguments)] in one process (one model load)"""
        lines = [" ".join(quoted(x) for x in args + ["-n", str(n), "--ids", "--quiet", "-o", os.path.join(self.out, job)])
                 for job, args in jobs]
        with open(self.path(name, "jobs"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        self.cli(name, ["-c", str(ctx), "--batch", self.path(name, "jobs")] + list(extra))

    def tag(self, preset, key):
        """a result's key; q8_0's keep their names, so a reference recorded before the presets still compares"""
        return key if preset == "q8" else f"{preset}/{key}"

    def long_file(self, ask=""):
        prompt = self.path("long_ask" if ask else "chunks_prompt", "txt")
        with open(prompt, "w", encoding="utf-8", newline="\n") as f:
            f.write(long_prompt() + (f"\n\nSkip those functions. {ask}" if ask else ""))
        return prompt

    def ids(self, name):
        try:
            with open(self.path(name, "out"), encoding="utf-8", errors="replace") as f:
                return next((ln.strip() for ln in f if ln.startswith("IDS:")), None)
        except OSError:
            return None

    def err_match(self, name, pattern):
        m = re.search(pattern, tail(self.path(name, "err"), 200))
        return m.group(1) if m else None

    def err_has(self, name, pattern):
        """the pattern anywhere in <name>.err (the whole file: the engine's load lines are not in its tail)"""
        try:
            with open(self.path(name, "err"), encoding="utf-8", errors="replace") as f:
                return re.search(pattern, f.read()) is not None
        except OSError:
            return False

    def drafted(self, name):
        """True if the job's decode made drafts (its "accepted A/B drafts" line with B > 0)"""
        b = self.err_match(name, r"accepted \d+/(\d+) drafts")
        return b is not None and int(b) > 0

    def same(self, what, got):
        """a verdict on whether the labelled results in got are all present and identical"""
        vals = list(got.values())
        ok = None not in vals and len(set(vals)) == 1
        ref = next(iter(got))
        if all(v is None or v.startswith("IDS:") for v in vals):
            detail = "; ".join(f"{k} vs {ref}: {first_difference(got[ref], v)}" for k, v in got.items() if k != ref)
        else:
            detail = "; ".join(f"{k}: {v}" for k, v in got.items())
        self.verdict(ok, what, detail)

    def depth(self):
        d = self.a.depths
        for n in d:  # (a process per depth, as a user runs it: a --batch engine is built for its deepest job)
            jobs = [(f"depth_{p}_d{n}", ["-p", t, "-d", str(n)]) for p, t in PROMPTS.items()]
            if n == max(d):
                jobs.append(("sampled_story", ["-p", PROMPTS["story"], "-d", str(n)] + SAMPLED))
            self.batch(f"depth_d{n}", jobs)
        for p in PROMPTS:
            got = {f"d{n}": self.ids(f"depth_{p}_d{n}") for n in d}
            for k, v in got.items():
                self.rec[f"cli/{p}/{k}"] = v
            what = f"depth, {p} prompt: greedy ids identical at {', '.join(map(str, d))} drafts"
            undrafted = [f"d{n}" for n in d if n > 0 and not self.drafted(f"depth_{p}_d{n}")]
            if undrafted:  # (every depth then ran plain decoding: identical ids would prove nothing)
                self.verdict(False, what, f"{', '.join(undrafted)} made no drafts (no MTP layer in the model, or MK_NO_MTP is "
                                          "set), so depth independence is untested")
            else:
                self.same(what, got)
        self.rec["cli/story/sampled"] = self.ids("sampled_story")  # (--compare only: across depths only the distribution is kept)
        for preset in self.a.presets:
            if preset == "q8":
                continue
            got, args = {}, ["-c", "16384", "-f", self.long_file(), "-n", "256", "--ids", "--quiet"]
            for n in d:
                self.cli(f"depth_{preset}_d{n}", args + ["-d", str(n)], PRESETS[preset])
                got[f"d{n}"] = self.rec[f"cli/{preset}/d{n}"] = self.ids(f"depth_{preset}_d{n}")
            undrafted = [f"d{n}" for n in d if n > 0 and not self.drafted(f"depth_{preset}_d{n}")]
            what = f"depth, preset {preset} past its window: greedy ids identical at {', '.join(map(str, d))} drafts"
            if undrafted:
                self.verdict(False, what, f"{', '.join(undrafted)} made no drafts, so depth independence is untested")
            else:
                self.same(what, got)

    def mtp(self):
        top = f"d{max(self.a.depths)}"
        self.batch("nomtp", [(f"nomtp_{p}", ["-p", t, "-d", "0"]) for p, t in PROMPTS.items()], ["--no-mtp"])
        off = self.err_has("nomtp", r"MTP layer off")  # (the engine's load line: --no-mtp took effect)
        for p in PROMPTS:
            got = {"no MTP layer": self.ids(f"nomtp_{p}"), top: self.rec.get(f"cli/{p}/{top}")}
            self.rec[f"cli/{p}/nomtp"] = got["no MTP layer"]
            what = f"mtp, {p} prompt: greedy ids without the MTP layer equal those at {top[1:]} drafts"
            if not off:
                self.verdict(False, what, "the --no-mtp run did not report 'MTP layer off', so it ran with the layer: nothing is compared")
            elif not self.drafted(f"depth_{p}_{top}"):  # (the depth check's run, which mtp needs)
                self.verdict(False, what, f"{top} made no drafts (no MTP layer in the model, or MK_NO_MTP is set), so both sides "
                                          "ran plain decoding")
            else:
                self.same(what, got)

    def long(self):
        # ("code": past a preset's window, and with an essay to write: asked about the functions, the model stops near 600 tokens)
        top, sources = max(self.a.depths), {"code": ["-f", self.long_file(ESSAY)], "essay": ["-p", ESSAY]}
        for preset in self.a.presets:
            for p, src in sources.items():
                if p == "essay" and preset != "q8":  # (a short prompt: its first ~2k tokens never leave the 8-bit window)
                    continue
                got = {}
                for n in (0, top):
                    name = f"long_{preset}_{p}_d{n}"
                    self.cli(name, ["-c", "16384", *src, "-n", str(self.a.long_tokens), "-d", str(n), "--ids", "--quiet"], PRESETS[preset])
                    got[f"d{n}"] = self.rec[f"long/{preset}/{p}/d{n}"] = self.ids(name)
                made = len(got["d0"].split()) - 1 if got["d0"] else 0
                what = f"long, {preset}, {p}: greedy ids over {made} tokens identical at no drafts and {top}"
                if made < self.a.long_min:
                    self.verdict(False, what, f"only {made} tokens at no drafts (--long-min {self.a.long_min}), so a late near-tie is untested")
                elif not self.drafted(f"long_{preset}_{p}_d{top}"):
                    self.verdict(False, what, f"d{top} made no drafts, so both sides ran plain decoding")
                else:
                    self.same(what, got)

    def policy(self):
        top = max(self.a.depths)
        jobs = [(f"policy_{i}", ["-p", ESSAY, "-d", str(top)] + (["--policy", spec] if spec else [])) for i, (_, spec) in enumerate(POLICIES)]
        self.batch("policy", jobs, n=self.a.long_tokens, ctx=16384)
        got = {"no drafts": self.rec.get("long/q8/essay/d0")}
        for i, (label, _) in enumerate(POLICIES):
            got[label] = self.rec[f"policy/{label}"] = self.ids(f"policy_{i}")
        what = "policy: greedy ids under every draft policy equal no drafts"
        if not self.drafted("policy_0"):
            self.verdict(False, what, "the default policy made no drafts, so nothing is compared")
        else:
            self.same(what, got)

    def chunks(self):
        args = ["-c", "16384", "-f", self.long_file(), "-n", "64", "-d", str(max(self.a.depths)), "--digest", "--ids", "--quiet"]
        caps = ", ".join(map(str, CHUNKS))
        for preset in self.a.presets:
            label = "" if preset == "q8" else f" {preset}"
            for cap in CHUNKS:
                name = f"chunks_{preset}_{cap}"
                self.cli(name, args, {"MK_EXACT": "1", "MK_PF_CAP": str(cap), **PRESETS[preset]})
                self.rec[self.tag(preset, f"chunks/{cap}/digest")] = self.err_match(name, r"state digest after the prefill: ([0-9a-f]+)")
                self.rec[self.tag(preset, f"chunks/{cap}/ids")] = self.ids(name)
            n = self.err_match(f"chunks_{preset}_{CHUNKS[0]}", r"prompt (\d+) tok") or "?"
            self.same(f"chunks{label}: MK_EXACT state digest after a {n}-token prompt identical at prefill chunks of {caps}",
                      {f"chunk {c}": self.rec[self.tag(preset, f"chunks/{c}/digest")] for c in CHUNKS})
            self.same(f"chunks{label}: greedy ids identical at prefill chunks of {caps}",
                      {f"chunk {c}": self.rec[self.tag(preset, f"chunks/{c}/ids")] for c in CHUNKS})

    def against_fresh(self, leg, preset, cases, warm, cached, env=None, ctx=32768, min_cached=1):
        """each case's warm reply against the same request on a server of its own"""
        label = "" if preset == "q8" else f" {preset}"
        for k, (m, o) in cases.items():
            with Server(self, f"{leg}_{preset}_fresh_" + k.replace(" ", "_"), env={**PRESETS[preset], **(env or {})}, ctx=ctx) as s:
                fresh, _ = s.chat(m, o)
            self.rec[self.tag(preset, f"{leg}/{k}")] = fresh
            what = f"{leg}{label}, {k}: the warm server's reply (exact reuse) equals a fresh server's"
            if not cached[k] or cached[k] < min_cached:  # (nothing reused: the same reply would prove nothing)
                self.verdict(False, what, f"the warm server reused nothing (usage.prompt_tokens_details.cached_tokens "
                                          f"{cached[k]!r}, needed {min_cached}), so warm == fresh proves nothing")
            else:
                self.verdict(fresh == warm[k], what, f"warm:  {warm[k]!r}\nfresh: {fresh!r}")

    def turn_cases(self, hist):
        return {"follow-up": (hist + [user(U2)], GREEDY), "edit": (hist + [user(U2_EDIT)], GREEDY),
                "sampled follow-up": (hist + [user(U2)], SAMPLED_REQ)}

    def server(self):
        for preset in self.a.presets:
            first = [SYS if preset == "q8" else SYS_LONG, user(U1)]
            with Server(self, f"server_{preset}_warm", env=PRESETS[preset]) as s:
                r1, _ = s.chat(first, GREEDY)
                cases, warm, cached = self.turn_cases(first + [assistant(r1)]), {}, {}
                for k, (m, o) in cases.items():
                    warm[k], cached[k] = s.chat(m, o)
            self.rec[self.tag(preset, "server/first turn")] = r1
            self.against_fresh("server", preset, cases, warm, cached)

    def park(self):
        for preset in self.a.presets:
            label = "" if preset == "q8" else f" {preset}"
            first = {k: [sys_other(i), user(U1)] for i, k in enumerate(("A", "B", "C"))}
            with Server(self, f"park_{preset}_warm", env={**PRESETS[preset], "MK_SIDE_MAX": "0"}) as s:
                hist = {k: m + [assistant(s.chat(m, GREEDY)[0])] for k, m in first.items()}
                cases = {"A back": (hist["A"] + [user(U2)], GREEDY), "B back": (hist["B"] + [user(U2)], GREEDY),
                         "C back, sampled": (hist["C"] + [user(U2)], SAMPLED_REQ)}
                warm, cached = {}, {}
                for k, (m, o) in cases.items():
                    warm[k], cached[k] = s.chat(m, o)
                parked = s.logged(r"parked the cached conversation")
            if parked:
                self.against_fresh("park", preset, cases, warm, cached, min_cached=1024)
            else:
                self.verdict(False, f"park{label}: conversations parked and brought back", "the log shows no park, so nothing was tested")
            first = [SYS_LONG, user(U1)]
            with Server(self, f"side_{preset}_warm", env=PRESETS[preset]) as s:
                r1, _ = s.chat(first, GREEDY)
                title = [user(TITLE)]
                side, _ = s.chat(title, GREEDY)
                cases = {"after a side request": (first + [assistant(r1), user(U2)], GREEDY)}
                warm, cached = {}, {}
                for k, (m, o) in cases.items():
                    warm[k], cached[k] = s.chat(m, o)
                beside = s.logged(r"side request done")
            if beside:
                self.against_fresh("park", preset, {**cases, "side request": (title, GREEDY)}, {**warm, "side request": side},
                                   {**cached, "side request": 1024}, min_cached=1024)
            else:
                self.verdict(False, f"park{label}: a side request beside a conversation", "the log shows no side request, so nothing was tested")

    def resume(self):
        for preset in self.a.presets:
            with Server(self, f"resume_{preset}_warm", env={**PRESETS[preset], "MK_SIDE_MAX": "0"}) as s:
                s.chat([SYS_LONG, user(U1)], GREEDY)
                cases, warm, cached = {"new session": ([SYS_LONG, user(U3)], GREEDY)}, {}, {}
                warm["new session"], cached["new session"] = s.chat(*cases["new session"])
                s.chat([sys_other(1), user(U1)], GREEDY)
                for k, c in (("new session after a park", ([SYS_LONG, user(U4)], GREEDY)),
                             ("new session after a park, sampled", ([SYS_LONG, user(U5)], SAMPLED_REQ))):
                    cases[k] = c
                    warm[k], cached[k] = s.chat(*c)
            self.against_fresh("resume", preset, cases, warm, cached, min_cached=1024)

    def deep(self):
        env_all = {"MK_PF_CAP": "1024"}  # (what scripts/serve.sh sets above 96k)
        for preset in self.a.presets:
            env = {**PRESETS[preset], **env_all}
            with Server(self, f"deep_{preset}_warm", env=env, ctx=120000) as s:
                big, tokens = s.fit(self.a.deep_tokens)
                first = [SYS, user(big)]
                r1, _ = s.chat(first, GREEDY)
                cases, warm, cached = self.turn_cases(first + [assistant(r1)]), {}, {}
                # the last function but a few, so the divergence sits deep in the prompt
                names = re.findall(r"def (step_\d+)\(", big)
                cases["edit inside the prompt"] = ([SYS, user(big.replace(f"def {names[-4]}(", f"def {names[-4]}b("))], GREEDY)
                for k, (m, o) in cases.items():
                    warm[k], cached[k] = s.chat(m, o)
            print(f"     (deep {preset}: ~{tokens} tokens)", flush=True)
            self.rec[self.tag(preset, "deep/first turn")] = r1
            self.against_fresh("deep", preset, cases, warm, cached, env_all, 120000, tokens // 2)

    def full(self):
        if "F" not in self.a.presets:
            return
        env = {**PRESETS["F"], "MK_PF_CAP": "1024"}
        with Server(self, "full_F_warm", env=env, ctx=262144) as s:
            big, tokens = s.fit(self.a.full_tokens)
            first = [SYS, user(big)]
            r1, _ = s.chat(first, GREEDY)
            cases = {"follow-up": (first + [assistant(r1), user(U2)], GREEDY)}
            warm, cached = {}, {}
            warm["follow-up"], cached["follow-up"] = s.chat(*cases["follow-up"])
        print(f"     (full F: ~{tokens} tokens)", flush=True)
        self.rec["F/full/first turn"] = r1
        self.against_fresh("full", "F", cases, warm, cached, {"MK_PF_CAP": "1024"}, 262144, tokens // 2)

    def agents(self):
        msgs, got = [user(long_prompt())], {}
        for n in SOLO:
            with Server(self, f"agents_solo{n}", ["--agents", "1"], {"MK_PF_SOLO": str(n)}) as s:
                got[n], _ = s.chat(msgs, {"temperature": 0})
            self.rec[f"agents/solo {n}"] = got[n]
        a, b = SOLO
        self.verdict(got[a] == got[b], f"agents: the chunks prompt fed in slot prefill chunks of {a} (MK_PF_SOLO) gets the "
                                       "greedy reply it gets fed whole", f"chunks of {a}: {got[a]!r}\nwhole: {got[b]!r}")

    def save(self):
        a = self.a
        meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "build": build_info(), "bin_sha256": sha256(a.bin),
                "model": os.path.basename(a.model), "model_bytes": os.path.getsize(a.model), "depths": a.depths,
                "legs": a.legs, "presets": a.presets, "platform": sys.platform,
                "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MK_") and k != "MK_API_KEY"}}
        with open(self.path("exactness", "json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"meta": meta, "results": self.rec}, f, indent=1)
        return meta

    def compare(self, meta, other):
        path = os.path.join(other, "exactness.json") if os.path.isdir(other) else other
        with open(path, encoding="utf-8") as f:
            o = json.load(f)
        om, theirs = o["meta"], o["results"]
        if (om["model"], om["model_bytes"]) != (meta["model"], meta["model_bytes"]):
            self.verdict(False, f"compare: {path} ran another model file ({om['model']}, {om['model_bytes']} bytes)")
            return
        common = sorted(k for k in set(self.rec) & set(theirs) if self.rec[k] is not None and theirs[k] is not None)
        differ = [k for k in common if self.rec[k] != theirs[k]]
        self.verdict(bool(common) and not differ,
                     f"compare: {len(common) - len(differ)} of {len(common)} results identical to {path} "
                     f"({om['build']}, {om['platform']}, {om['time']})", "differ: " + ", ".join(differ))
        only = sorted(set(self.rec) ^ set(theirs))
        if only:
            print(f"     (in one run only, not compared: {', '.join(only)})")
        if om.get("env") != meta["env"]:
            print(f"     (the runs' MK_* settings differ: {om.get('env')} there, {meta['env']} here)")


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    dv = os.path.join(HERE, "draft_vocab_32k.bin")
    ap.add_argument("-m", "--model", default=os.environ.get("MK_MODEL"), help="the GGUF (default: MK_MODEL)")
    ap.add_argument("--bin", default=os.path.join(HERE, "qwen38_mk.exe" if WIN else "qwen38_mk"), help="the CLI")
    ap.add_argument("--draft-vocab", default=dv if os.path.isfile(dv) else "", help="(default: draft_vocab_32k.bin)")
    ap.add_argument("--depths", default="0,4,7", help="draft depths (default 0,4,7)")
    ap.add_argument("--legs", default=",".join(LEGS), help=f"the checks to run (default {','.join(LEGS)})")
    ap.add_argument("--presets", default=",".join(PRESETS), help=f"the KV caches of the per-cache checks (default {','.join(PRESETS)})")
    ap.add_argument("--long-tokens", type=int, default=2048, help="output tokens of the long and policy checks (default 2048)")
    ap.add_argument("--long-min", type=int, default=1000, help="fewest output tokens that count in the long check (default 1000)")
    ap.add_argument("--deep-tokens", type=int, default=100000, help="prompt size of the deep check (default 100000)")
    ap.add_argument("--full-tokens", type=int, default=258000, help="prompt size of the full check (default 258000)")
    ap.add_argument("--out", default="exactness-out", help="output directory (default exactness-out)")
    ap.add_argument("--compare", metavar="DIR", help="another run's output directory (or its exactness.json)")
    ap.add_argument("--port", type=int, default=18190, help="the server checks' port (default 18190)")
    ap.add_argument("--python", default=sys.executable, help="Python for the server, with jinja2 (default: this one)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per CLI run, server start or request")
    a = ap.parse_args()
    if not a.model or not os.path.isfile(a.model):
        ap.error(f"no model file {a.model!r} (-m or MK_MODEL)")
    if not os.path.isfile(a.bin):
        ap.error(f"no {a.bin}: build it (make / build.ps1) or pass --bin")
    if "'" in os.path.abspath(a.out):
        ap.error("--out: a path without quotes (the CLI's --batch lines cannot hold one)")
    a.model, a.bin = os.path.abspath(a.model), os.path.abspath(a.bin)
    a.draft_vocab = os.path.abspath(a.draft_vocab) if a.draft_vocab else ""
    try:
        a.depths = sorted({int(x) for x in a.depths.split(",")})
    except ValueError:
        ap.error("--depths: comma-separated integers")
    a.legs = [x for x in a.legs.split(",") if x]
    if set(a.legs) - set(LEGS):
        ap.error(f"--legs: some of {', '.join(LEGS)}")
    if "mtp" in a.legs and "depth" not in a.legs:
        ap.error("--legs: mtp compares against the depth check's ids, so it needs depth too")
    if "policy" in a.legs and "long" not in a.legs:
        ap.error("--legs: policy compares against the long check's ids, so it needs long too")
    a.presets = [x for x in a.presets.split(",") if x]
    if set(a.presets) - set(PRESETS) or not a.presets:
        ap.error(f"--presets: some of {', '.join(PRESETS)}")
    if "policy" in a.legs and "q8" not in a.presets:
        ap.error("--legs: policy compares on q8_0, so --presets needs q8")
    a.legs = [x for x in LEGS if x in a.legs]
    chk = Check(a)
    t0 = time.monotonic()
    for leg in a.legs:
        print(f"== {leg} ({time.monotonic() - t0:.0f} s)", flush=True)
        try:
            getattr(chk, leg)()
        except LegFailed as e:
            chk.verdict(False, e.what, e.detail)
    meta = chk.save()
    if a.compare:
        chk.compare(meta, a.compare)
    print(f"exactness: {chk.passed} passed, {chk.failed} failed in {time.monotonic() - t0:.0f} s "
          f"({meta['build']}; {chk.path('exactness', 'json')})")
    return 1 if chk.failed else 0


if __name__ == "__main__":
    sys.exit(main())

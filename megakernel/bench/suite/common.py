"""Shared helpers of the benchmark suite: paths, the item manifest, requests, nvidia-smi, the question prompts.

Item files come from the data directory (downloads, untrusted): their text only goes into prompts and the scorer's
string checks; nothing in them is executed except HumanEval under score.py --exec-humaneval."""
import hashlib
import json
import os
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MANIFEST = os.path.join(HERE, "items.json")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
GREEDY = {"temperature": 0.0, "top_k": 1}
LET = "ABCDEFGHIJ"


def canon_sha(obj):
    """sha256 of an item's normalised content: its JSON with sorted keys, UTF-8, no ASCII escapes, no spaces."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path):
    """A prompt file as the clients send it (UTF-8, universal newlines)."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_manifest(path=MANIFEST):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_set(data, name, manifest=None):
    """The items of one set from the data directory, each checked against the manifest (same ids, order, content)."""
    m = manifest or load_manifest()
    if name not in m["sets"]:
        raise SystemExit(f"unknown set {name!r} (sets: {', '.join(m['sets'])})")
    spec = m["sets"][name]
    p = os.path.join(data, *spec["file"].split("/"))
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing: run fetch.py --data {data} first")
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    ids = [it.get("id") for it in items]
    want = [w["id"] for w in spec["items"]]
    if ids != want:
        raise SystemExit(f"{p}: the item ids differ from the manifest's: run fetch.py again")
    for it, w in zip(items, spec["items"]):
        if canon_sha(it) != w["sha256"]:
            raise SystemExit(f"{p}: item {it['id']} differs from the manifest: run fetch.py again")
    return items


def load_prompt(data, name, manifest=None):
    """pf38k (committed under prompts/) or pf101k / oj100k (built by fetch.py in the data directory), hash-checked."""
    m = manifest or load_manifest()
    spec = m["prompts"][name]
    p = os.path.join(HERE if spec["where"] == "repo" else data, *spec["file"].split("/"))
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing" + ("" if spec["where"] == "repo" else f": run fetch.py --data {data} first"))
    text = read_text(p)
    if text_sha(text) != spec["sha256"]:
        raise SystemExit(f"{p} differs from the manifest (sha256): " +
                         ("restore it from git" if spec["where"] == "repo" else "run fetch.py again"))
    return text


def chat_url(base):
    """The chat completions URL from a base URL (http://host:port, .../v1 or the full path)."""
    b = base.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return b + ("/chat/completions" if b.endswith("/v1") else "/v1/chat/completions")


def post(url, body, timeout):
    """POST a JSON body; returns (reply, wall seconds including reading and parsing the reply)."""
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.perf_counter() - t0


class Gpu:
    """nvidia-smi queries for one GPU (whole-card numbers, other processes included); off without nvidia-smi."""

    def __init__(self, index):
        self.index = None if str(index).lower() == "none" else str(index)

    def query(self, fields):
        if self.index is None:
            return None
        try:
            r = subprocess.run(["nvidia-smi", "-i", self.index, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
            vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            return vals if r.returncode == 0 and len(vals) == len(fields.split(",")) else None
        except (OSError, IndexError, subprocess.SubprocessError):
            return None

    def vram_mib(self):
        v = self.query("memory.used")
        try:
            return int(v[0]) if v else None
        except ValueError:
            return None


def add_common_args(ap, gpu=True):
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="server base URL (default http://127.0.0.1:8080)")
    ap.add_argument("--label", required=True, help="label written into every record (one per server configuration)")
    ap.add_argument("--out", required=True, help="results file (JSON lines, appended)")
    ap.add_argument("--data", default=DATA, help="data directory written by fetch.py (default bench/suite/data)")
    if gpu:
        ap.add_argument("--gpu", default="0", help="nvidia-smi GPU index for VRAM / clock records, or none (default 0)")


# ---- question prompts: (prompt, max_tokens, thinking) per set, zero-shot, the same for every engine

def p_gsm8k(it):
    return (f"Solve the following math problem. Show your work briefly, then give the final answer on the last line as "
            f"'Answer: <number>'.\n\n{it['question']}"), 768, False


def p_mmlu_pro(it):
    opts = "\n".join(f"({LET[i]}) {o}" for i, o in enumerate(it["options"]))
    return (f"The following is a multiple choice question about {it['category']}. Think step by step, then finish with "
            f"'The answer is (X)' where X is the letter of the correct option.\n\nQuestion: {it['question']}\nOptions:\n{opts}"), 1024, False


def p_arc_c(it):
    opts = "\n".join(f"({lab}) {t}" for lab, t in zip(it["labels"], it["texts"]))
    return f"{it['question']}\n{opts}\n\nAnswer with the letter of the correct option only.", 8, False


def p_math(it):
    return f"Solve the following problem. Put the final answer in \\boxed{{}}.\n\n{it['problem']}", 1536, False


def p_math_think(it):
    return f"Solve the following problem. Put the final answer in \\boxed{{}}.\n\n{it['problem']}", 8192, True


def p_humaneval(it):
    return ("Complete the following Python function. Reply with the complete function (including the signature and any "
            f"imports it needs) in one ```python code block.\n\n```python\n{it['prompt']}```"), 768, False


def longbench_prompt(it):  # LongBench v2's own prompt (no chain of thought)
    return (f"Please read the following text and answer the question below.\n\n<text>\n{it['context']}\n</text>\n\n"
            f"What is the correct answer to this question: {it['question']}\nChoices:\n(A) {it['choice_A']}\n"
            f"(B) {it['choice_B']}\n(C) {it['choice_C']}\n(D) {it['choice_D']}\n\n"
            "Format your response as follows: \"The correct answer is (insert answer here)\".")


def p_long(it):
    return longbench_prompt(it), 128, False


def p_niah(it):
    return it["prompt"], 64, False


# the order the 2026-10-20 run asked them in
QA_SETS = [("gsm8k", p_gsm8k), ("mmlu_pro", p_mmlu_pro), ("arc_c", p_arc_c), ("math500", p_math), ("humaneval", p_humaneval),
           ("niah", p_niah), ("longbench_v2", p_long), ("math500_think", p_math_think)]

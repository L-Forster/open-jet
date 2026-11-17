"""Download the benchmark items into a data directory and check every item against the manifest (items.json).
usage: python -I bench/suite/fetch.py [--data DIR] [--sets NAME ...] [--check]
  --data   where the items go (default bench/suite/data, ignored by git); downloads are kept under DIR/raw
  --sets   only these (default: every set and the built prompts): gsm8k mmlu_pro arc_c math500 math500_think humaneval
           niah niah_max longbench_v2 longbench_max pf101k oj100k
  --check  download nothing: check the files already in DIR against the manifest
The items are selected by the manifest's ids and row numbers (no re-sampling); the NIAH prompts, pf101k.txt and
oj100k.txt are built from the manifest's construction parameters. An item whose content differs from the manifest's
hash is an error that names it (the dataset changed upstream). Downloads are untrusted data: nothing in them is executed
here (run with python -I, so no module is imported from the current directory)."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

UA = {"User-Agent": "qwen38-mk-bench-suite/1"}
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROWS_MAX = 100  # rows per request the API allows


def get(url, path=None, tries=6, timeout=600):
    """Bytes of a URL, or the URL streamed into path (via path.part). Retries with backoff (the rows API rate-limits)."""
    last = None
    for i in range(tries):
        wait = min(120, 5 * 2 ** i)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                if path is None:
                    return r.read()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                os.replace(tmp, path)
                return path
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                raise SystemExit(f"{url}: HTTP {e.code}")
            last = e
            ra = (e.headers or {}).get("Retry-After", "")
            if ra.isdigit():
                wait = min(300, max(wait, int(ra)))
        except (urllib.error.URLError, OSError) as e:
            last = e
        if i + 1 < tries:
            print(f"  retry {i + 1}/{tries - 1} in {wait} s: {url[:110]}: {last}", flush=True)
            time.sleep(wait)
    hint = ""
    if "CERTIFICATE_VERIFY_FAILED" in str(last):
        hint = (" (Python could not verify the site's certificate. On Windows, opening the URL once in a browser or with"
                " `curl -I <url>` makes Windows install the missing root certificate; then run fetch.py again)")
    raise SystemExit(f"download failed: {url}: {last}{hint}")


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_file(data, src, rel=None, url=None, want=None):
    """A source file under data/raw, downloaded unless an equal copy is there. want: (bytes, sha256) or None."""
    p = os.path.join(data, *(rel or src["file"]).split("/"))
    want = want or (src.get("bytes"), src.get("sha256"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not (os.path.exists(p) and os.path.getsize(p) == want[0] and file_sha(p) == want[1]):
        print(f"downloading {url or src['url']} ...", flush=True)
        get(url or src["url"], p)
    sha = file_sha(p)
    return p, sha == want[1], sha


# ---- HF datasets-server rows -> items, as the 2026-10-13 fetch converted them

def t_gsm8k(r, e):
    return {"id": e["id"], "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}


def t_mmlu_pro(r, e):
    return {"id": f"mmlupro_{r['question_id']}", "category": r["category"], "question": r["question"],
            "options": r["options"], "answer": r["answer"]}


def t_arc_c(r, e):
    return {"id": r["id"], "question": r["question"], "labels": r["choices"]["label"], "texts": r["choices"]["text"],
            "answer": r["answerKey"]}


def t_math(r, e):
    return {"id": r["unique_id"], "level": r["level"], "subject": r["subject"], "problem": r["problem"], "answer": r["answer"]}


def t_humaneval(r, e):
    return {"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}


TRANSFORM = {"gsm8k": t_gsm8k, "mmlu_pro": t_mmlu_pro, "arc_c": t_arc_c, "math500": t_math, "humaneval": t_humaneval}


def windows(rows):
    """Covering requests (offset, length <= ROWS_MAX) for a set of row numbers."""
    out, rows = [], sorted(set(rows))
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] - rows[i] < ROWS_MAX:
            j += 1
        out.append((rows[i], rows[j] - rows[i] + 1))
        i = j + 1
    return out


def hf_rows(src, rows):
    got = {}
    for off, n in windows(rows):
        q = urllib.parse.urlencode({"dataset": src["dataset"], "config": src["config"], "split": src["split"],
                                    "offset": off, "length": n})
        d = json.loads(get(f"{ROWS_API}?{q}"))
        for r in d.get("rows", []):
            got[r["row_idx"]] = r["row"]
    return got


def lb_item(d, e):
    return {"id": d["_id"], "domain": d["domain"], "sub_domain": d["sub_domain"], "difficulty": d["difficulty"],
            "length": d["length"], "approx_tokens": e["approx_tokens"], "question": d["question"],
            "choice_A": d["choice_A"], "choice_B": d["choice_B"], "choice_C": d["choice_C"], "choice_D": d["choice_D"],
            "answer": d["answer"], "context": d["context"]}


def haystack(path, hs):
    """The novel's text from its first chapter heading to the end marker, newlines normalised."""
    raw = re.sub(r"\r\n?", "\n", open(path, "rb").read().decode("utf-8"))
    a, b = raw.find(hs["start"]), raw.rfind(hs["end"])
    if a < 0:
        raise SystemExit(f"{path}: {hs['start']!r} not found: the file changed upstream")
    return raw[a:b if b > 0 else len(raw)]


def niah_item(hay, build, e):
    """A RULER-style multi-key needle prompt: the haystack's first body_chars characters with the needles inserted at
    the given character offsets, then the question for one key."""
    body = hay[:e["body_chars"]]
    parts, last = [build["head"]], 0
    for pos, key, val in e["needles"]:
        parts += [body[last:pos], build["needle"].format(key=key, value=val)]
        last = pos
    parts += [body[last:], build["question"].format(key=e["key"])]
    return {"id": e["id"], "length": e["length"], "tokens": e["tokens"], "key": e["key"], "answer": e["answer"],
            "depth": e["depth"], "prompt": "".join(parts)}


def write_json(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--sets", nargs="+")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    m = C.load_manifest()
    built = [k for k, v in m["prompts"].items() if v["where"] == "data"]
    names = list(m["sets"]) + built
    want = a.sets or names
    bad = [s for s in want if s not in names]
    if bad:
        raise SystemExit(f"unknown set(s) {bad}; sets: {' '.join(names)}")
    data = os.path.abspath(a.data)
    os.makedirs(data, exist_ok=True)
    errors = []

    if a.check:
        for s in want:
            try:
                if s in built:
                    C.load_prompt(data, s, m)
                else:
                    C.load_set(data, s, m)
                print(f"{s}: ok", flush=True)
            except SystemExit as e:
                errors.append(f"{s}: {e}")
        for e in errors:
            print("ERROR", e)
        sys.exit(1 if errors else 0)

    def finish(name, items, why):
        spec = m["sets"][name]
        errs = []
        for it, e in zip(items, spec["items"]):
            if it is None:
                errs.append(f"{name} {e['id']}: {why}")
            elif it["id"] != e["id"]:
                errs.append(f"{name} {e['id']}: upstream now has {it['id']} there")
            elif C.canon_sha(it) != e["sha256"]:
                errs.append(f"{name} {e['id']}: content differs from the manifest")
        if errs:
            errors.extend(errs)
            print(f"{name}: {len(errs)} of {len(items)} items differ from the manifest: not written", flush=True)
            return
        write_json(os.path.join(data, *spec["file"].split("/")), items)
        print(f"{name}: {len(items)} items, all equal to the manifest -> {spec['file']}", flush=True)

    # small sets: HF datasets-server rows, by row number (sets that share a source are fetched together)
    by_src = {}
    for s in want:
        if s in m["sets"] and m["sources"][m["sets"][s]["source"]]["type"] == "hf_rows":
            by_src.setdefault(m["sets"][s]["source"], []).append(s)
    for src_name, sets in by_src.items():
        src = m["sources"][src_name]
        rows = hf_rows(src, [e["row"] for s in sets for e in m["sets"][s]["items"]])
        for s in sets:
            fn = TRANSFORM[src_name]
            finish(s, [fn(rows[e["row"]], e) if e["row"] in rows else None for e in m["sets"][s]["items"]],
                   f"its row is missing from {src['dataset']} (the dataset changed upstream)")

    # LongBench v2: one JSON file at a pinned revision
    lbs = [s for s in want if s in m["sets"] and m["sets"][s]["source"] == "longbench_v2"]
    if lbs:
        p, ok, sha = fetch_file(data, m["sources"]["longbench_v2"])
        if not ok:
            print(f"note: {p} sha256 {sha} is not the manifest's (the pinned revision should not change)", flush=True)
        with open(p, encoding="utf-8") as f:
            byid = {d["_id"]: d for d in json.load(f)}
        for s in lbs:
            finish(s, [lb_item(byid[e["id"]], e) if e["id"] in byid else None for e in m["sets"][s]["items"]],
                   "not in data.json")
        del byid

    # NIAH: built from the Project Gutenberg text
    nis = [s for s in want if s in m["sets"] and m["sets"][s]["source"] == "pg2600"]
    if nis:
        src = m["sources"]["pg2600"]
        p, ok, sha = fetch_file(data, src)
        if not ok:
            print(f"note: {p} is not byte-equal to the copy of 2026-10-13 (sha256 {sha}); the prompts' hashes decide",
                  flush=True)
        hay = haystack(p, src["haystack"])
        for s in nis:
            spec = m["sets"][s]
            items = []
            for e in spec["items"]:
                it = niah_item(hay, spec["build"], e) if len(hay) >= e["body_chars"] else None
                items.append(it if it is not None and C.text_sha(it["prompt"]) == e["prompt_sha256"] else None)
            finish(s, items, "the built prompt differs from the manifest's (the haystack text changed upstream)")

    # built prompts: files of this repository and of pinned upstream sources, concatenated and cut. pf101k.txt (pf38k.txt +
    # llama.cpp sources) is cut by bytes; oj100k.txt (open-jet's src/**/*.py, read as its parallel_bench.py reads them:
    # UTF-8 with replacement, universal newlines) by characters
    for name in [s for s in built if s in want]:
        spec = m["prompts"][name]
        chars = "chars" in spec
        buf = "" if chars else b""
        for part in spec["parts"]:
            kind, rel = part.split(":", 1)
            if kind == "repo":
                p = os.path.join(C.HERE, *rel.split("/"))
            else:
                src = m["sources"][kind]
                fs = src["files"][rel]
                p, ok, sha = fetch_file(data, src, rel=f"{src['dir']}/{rel}", url=src["base"] + rel,
                                        want=(fs["bytes"], fs["sha256"]))
                if not ok:
                    errors.append(f"{name}: {rel} at {src['base']} has sha256 {sha}, not the manifest's")
            if chars:
                with open(p, encoding="utf-8", errors="replace") as f:
                    buf += f.read()
            else:
                with open(p, "rb") as f:
                    b = f.read()
                buf += b.replace(b"\r\n", b"\n") if kind == "repo" else b
        buf = buf[:spec["chars"]] if chars else buf[:spec["bytes"]]
        out = os.path.join(data, *spec["file"].split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if chars:
            with open(out, "w", encoding="utf-8", newline="") as f:
                f.write(buf)
        else:
            with open(out, "wb") as f:
                f.write(buf)
        try:
            C.load_prompt(data, name, m)
            print(f"{name}: {len(buf)} {'characters' if chars else 'bytes'}, equal to the manifest -> {spec['file']}",
                  flush=True)
        except SystemExit as e:
            errors.append(f"{name}: {e}")

    for e in errors:
        print("ERROR", e)
    if errors:
        print(f"{len(errors)} error(s): the items above differ from the ones the published numbers were measured on.")
        sys.exit(1)
    print("FETCH_OK", flush=True)


if __name__ == "__main__":
    main()

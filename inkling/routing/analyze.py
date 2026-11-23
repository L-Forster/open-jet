"""Analyse an Inkling 2 Mini routing trace: how uneven expert use is, how the loop passes and
consecutive tokens overlap, how well expert caches would hit, and what that means for decode
speed on small devices.

    python inkling/routing/analyze.py --trace T.bin --requests R.jsonl --gguf model.gguf --out results/

Only decode tokens are analysed (prompt tokens are read in large batches where every expert is
used anyway). Caches are evaluated leave-one-request-out: a static profile for a request is
built from the other requests only, as Strata does for its hit-rate curve.
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_PHYS, CORE_START, CORE_END, N_PASSES, N_EXPERT, TOPK = 24, 6, 18, 3, 128, 8
N_CORE = CORE_END - CORE_START
N_APPS = N_PHYS + (N_PASSES - 1) * N_CORE  # 48 layer applications per token


def slot_phys_pass(slot: int) -> tuple[int, int]:
    if slot < N_PHYS:
        return slot, 0
    return CORE_START + (slot - N_PHYS) % N_CORE, 1 + (slot - N_PHYS) // N_CORE


def app_order() -> list[int]:
    """Slots in execution order: prelude, pass 0..2 of the core, tail."""
    core = list(range(CORE_START, CORE_END))
    loops = [N_PHYS + p * N_CORE + i for p in range(N_PASSES - 1) for i in range(N_CORE)]
    return list(range(CORE_START)) + core + loops + list(range(CORE_END, N_PHYS))


APP_SLOTS = app_order()
APP_PHYS = np.array([slot_phys_pass(s)[0] for s in APP_SLOTS])
APP_PASS = np.array([slot_phys_pass(s)[1] for s in APP_SLOTS])


# ---------------------------------------------------------------- trace parsing

def read_evals(path: Path, start: int, end: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns a list of (ids [n_tok, 48 apps, 8], weights [n_tok, 48, 8]) per graph evaluation."""
    data = path.read_bytes()[start:end]
    off = 0
    evals: list[dict] = []
    cur: dict | None = None
    while off < len(data):
        slot, kind, n_used, n_tok = np.frombuffer(data, np.int32, 4, off)
        off += 16
        n = int(n_used) * int(n_tok)
        arr = np.frombuffer(data, np.int32 if kind == 0 else np.float32, n, off).reshape(n_tok, n_used)
        off += 4 * n
        if slot == 0 and kind == 0:
            cur = {"ids": {}, "w": {}, "n": int(n_tok)}
            evals.append(cur)
        assert cur is not None and int(n_tok) == cur["n"], "trace record outside an evaluation"
        (cur["ids"] if kind == 0 else cur["w"])[int(slot)] = arr
    out = []
    for e in evals:
        if len(e["ids"]) != N_APPS:
            continue  # incomplete evaluation (e.g. aborted request)
        ids = np.stack([e["ids"][s] for s in APP_SLOTS], axis=1)
        w = np.stack([e["w"][s] for s in APP_SLOTS], axis=1) if len(e["w"]) == N_APPS else np.zeros(ids.shape, np.float32)
        out.append((ids, w))
    return out


@dataclass
class Request:
    name: str
    prompt_tokens: int
    ids: np.ndarray      # decode tokens [T, 48, 8]
    weights: np.ndarray  # [T, 48, 8]


def load_requests(trace: Path, reqs_path: Path) -> list[Request]:
    out = []
    for line in reqs_path.read_text().splitlines():
        r = json.loads(line)
        evals = read_evals(trace, r["trace_start"], r["trace_end"])
        last_batch = max((i for i, (ids, _) in enumerate(evals) if ids.shape[0] > 1), default=-1)
        dec = evals[last_batch + 1:]
        if not dec:
            continue
        out.append(Request(r["name"], r.get("prompt_tokens") or 0,
                           np.concatenate([d[0] for d in dec]), np.concatenate([d[1] for d in dec])))
    return out


# ---------------------------------------------------------------- model sizes

def gguf_sizes(path: Path, gguf_py: Path) -> dict:
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # noqa: E402

    rd = GGUFReader(str(path))
    expert = np.zeros(N_PHYS)       # bytes of one expert (gate+up+down) per physical layer
    dense_layer = np.zeros(N_PHYS)  # non-expert bytes per physical layer
    loop_extra = 0.0                # loop controller weights (read in passes 1..2)
    other = {}
    for t in rd.tensors:
        name, nbytes = t.name, int(t.n_bytes)
        if name.startswith("blk."):
            il = int(name.split(".")[1])
            if "_exps" in name:
                expert[il] += nbytes / N_EXPERT
            elif "loop" in name:
                loop_extra += nbytes
            else:
                dense_layer[il] += nbytes
        else:
            other[name] = nbytes
    return {"expert": expert, "dense_layer": dense_layer, "loop_extra": loop_extra, "other": other,
            "file_bytes": path.stat().st_size}


# ---------------------------------------------------------------- statistics

def usage_counts(reqs: list[Request]) -> np.ndarray:
    """Reads per (physical layer, expert); each pass of a looped layer counts as a read."""
    c = np.zeros((N_PHYS, N_EXPERT))
    for r in reqs:
        for a in range(N_APPS):
            np.add.at(c[APP_PHYS[a]], r.ids[:, a, :].ravel(), 1)
    return c


def coverage(counts: np.ndarray, expert_bytes: np.ndarray, shares=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Smallest share of expert bytes that serves each share of reads, picking the most-read experts first."""
    reads = counts.ravel()
    sizes = np.repeat(expert_bytes, N_EXPERT)
    order = np.argsort(-reads)
    cum_reads = np.cumsum(reads[order]) / reads.sum()
    cum_bytes = np.cumsum(sizes[order]) / sizes.sum()
    return {f"{int(s*100)}%_reads": float(cum_bytes[np.searchsorted(cum_reads, s)]) for s in shares}


def loop_stats(reqs: list[Request]) -> dict:
    uniq, overlap01, overlap12 = [], [], []
    for r in reqs:
        for i in range(N_CORE):
            p = [r.ids[:, np.where((APP_PHYS == CORE_START + i) & (APP_PASS == k))[0][0], :] for k in range(N_PASSES)]
            for t in range(r.ids.shape[0]):
                s = [set(x[t].tolist()) for x in p]
                uniq.append(len(s[0] | s[1] | s[2]))
                overlap01.append(len(s[0] & s[1]) / TOPK)
                overlap12.append(len(s[1] & s[2]) / TOPK)
    uniform_uniq = N_EXPERT * (1 - (1 - TOPK / N_EXPERT) ** N_PASSES)
    return {"unique_per_core_layer": float(np.mean(uniq)), "uniform_expectation": uniform_uniq,
            "pass0_pass1_overlap": float(np.mean(overlap01)), "pass1_pass2_overlap": float(np.mean(overlap12))}


def temporal_stats(reqs: list[Request], windows=(1, 4, 16, 64)) -> dict:
    """Share of a token's (layer, expert) needs already needed by one of the previous w tokens."""
    res = {}
    for w in windows:
        hit = tot = 0
        for r in reqs:
            T = r.ids.shape[0]
            need = [set() for _ in range(T)]
            for t in range(T):
                for a in range(N_APPS):
                    for e in r.ids[t, a]:
                        need[t].add(int(APP_PHYS[a]) * N_EXPERT + int(e))
            for t in range(1, T):
                recent = set().union(*need[max(0, t - w):t])
                hit += len(need[t] & recent)
                tot += len(need[t])
        res[f"prev_{w}"] = hit / tot
    return res


def tail_weight_share(reqs: list[Request]) -> dict:
    w = np.concatenate([r.weights.reshape(-1, TOPK) for r in reqs])
    w = -np.sort(-w, axis=1)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    share = (w / s).mean(axis=0)
    return {f"rank{i+1}": float(x) for i, x in enumerate(share)}


# ---------------------------------------------------------------- cache replay

def token_needs(r: Request) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per token: (unique keys, reads per key). key = phys*128 + expert."""
    out = []
    for t in range(r.ids.shape[0]):
        keys = (APP_PHYS[:, None] * N_EXPERT + r.ids[t]).ravel()
        u, c = np.unique(keys, return_counts=True)
        out.append((u, c))
    return out


def simulate(policy: str, reqs: list[Request], profile_counts: list[np.ndarray], cap_bytes: float,
             key_bytes: np.ndarray, swap_every: int = 4, swap_max: int = 96, decay: float = 0.97) -> dict:
    """One cache over all experts. Returns read hit rate (reads served from the cache, each pass a read)
    and missed unique-expert bytes per token (what a slower tier has to deliver once per token)."""
    hits = reads = 0.0
    miss_bytes = 0.0
    n_tok = 0
    for r, prof in zip(reqs, profile_counts):
        needs = token_needs(r)
        order = np.argsort(-prof.ravel())

        def fill(rank: np.ndarray) -> set:
            s, used = set(), 0.0
            for k in rank:
                b = key_bytes[k]
                if used + b > cap_bytes:
                    continue
                s.add(int(k))
                used += b
            return s

        if policy == "static":
            cache = fill(order)
        elif policy == "adaptive":
            cache = fill(order)
            score = prof.ravel().astype(float) / max(prof.sum(), 1) * 50.0  # profile prior, worth ~50 tokens
        elif policy == "lru":
            lru: OrderedDict[int, None] = OrderedDict((int(k), None) for k in fill(order))
            used = sum(key_bytes[k] for k in lru)
        elif policy == "belady":
            pos: dict[int, list[int]] = {}
            for tt, (u, _) in enumerate(needs):
                for k in u.tolist():
                    pos.setdefault(k, []).append(tt)
            INF = 1 << 30

            def next_use(x: int, after: int) -> int:
                p = pos.get(x)
                if not p:
                    return INF
                i = bisect.bisect_right(p, after)
                return p[i] if i < len(p) else INF

            cache = fill(order)
            used = sum(key_bytes[k] for k in cache)
            heap = [(-next_use(k, -1), k) for k in cache]
            heapq.heapify(heap)
            cur_nu = {k: -h for h, k in heap}

        for t, (u, c) in enumerate(needs):
            n_tok += 1
            reads += c.sum()
            if policy in ("static", "adaptive"):
                inc = np.fromiter((k in cache for k in u.tolist()), bool, len(u))
                hits += c[inc].sum()
                miss_bytes += key_bytes[u[~inc]].sum()
                if policy == "adaptive":
                    score *= decay
                    np.add.at(score, u, c)
                    if (t + 1) % swap_every == 0:
                        cand = [int(k) for k in np.argsort(-score)[: len(cache) + swap_max] if int(k) not in cache][:swap_max]
                        victims = sorted(cache, key=lambda k: score[k])
                        for new, old in zip(cand, victims):
                            if score[new] > score[old] * 1.5 and key_bytes[new] <= key_bytes[old] * 1.01:
                                cache.remove(old)
                                cache.add(new)
            elif policy == "lru":
                for k, cnt in zip(u.tolist(), c.tolist()):
                    if k in lru:
                        hits += cnt
                        lru.move_to_end(k)
                    else:
                        miss_bytes += key_bytes[k]
                        lru[k] = None
                        used += key_bytes[k]
                        while used > cap_bytes:
                            old, _ = lru.popitem(last=False)
                            used -= key_bytes[old]
            elif policy == "belady":
                ul = u.tolist()
                for k, cnt in zip(ul, c.tolist()):
                    if k in cache:
                        hits += cnt
                    else:
                        miss_bytes += key_bytes[k]
                        cache.add(k)
                        used += key_bytes[k]
                for k in ul:  # refresh next use of everything this token touched
                    nu = next_use(k, t)
                    cur_nu[k] = nu
                    heapq.heappush(heap, (-nu, k))
                while used > cap_bytes:
                    # evict the cached expert needed furthest in the future (lazy heap; stale entries skipped)
                    neg, victim = heapq.heappop(heap)
                    if victim not in cache or cur_nu.get(victim) != -neg:
                        continue
                    cache.remove(victim)
                    used -= key_bytes[victim]
    return {"read_hit": hits / reads, "miss_bytes_per_token": miss_bytes / n_tok}


# ---------------------------------------------------------------- device model

DEVICES = [
    # name, VRAM GiB, GPU GB/s, RAM GiB free for experts, CPU expert GB/s, SSD GB/s
    ("6 GB GPU + 16 GB RAM (DDR4)", 6, 336, 9, 30, 3.0),
    ("8 GB GPU + 16 GB RAM (DDR4)", 8, 448, 9, 30, 3.0),
    ("8 GB GPU + 8 GB RAM (DDR4)", 8, 448, 3, 30, 3.0),
    ("12 GB GPU + 16 GB RAM (DDR5)", 12, 360, 9, 45, 5.0),
    ("12 GB GPU + 32 GB RAM (DDR5)", 12, 504, 25, 45, 5.0),
    ("16 GB GPU + 16 GB RAM (DDR5)", 16, 448, 9, 45, 5.0),
]
GPU_EFF = 0.75
RESERVE_GIB = 1.0  # CUDA context, compute buffers, KV at a few K tokens


def device_estimates(curves: dict, sizes: dict) -> list[dict]:
    exp_total = float(sizes["expert"].sum() * N_EXPERT)
    dense_resident = float(sizes["dense_layer"].sum() + sizes["loop_extra"]
                           + sum(v for k, v in sizes["other"].items() if not k.startswith("token_embd")))
    # dense bytes read per token: core layers' non-expert weights are read once per pass
    passes = np.ones(N_PHYS)
    passes[CORE_START:CORE_END] = N_PASSES
    dense_per_tok = float((sizes["dense_layer"] * passes).sum() + sizes["loop_extra"]
                          + sizes["other"].get("output.weight", 0))
    caps = np.array(curves["cap_share"])
    out = []
    for name, vram, gbw, ram, cbw, ssd in DEVICES:
        v_cap = max(0.0, vram * 2**30 - dense_resident - RESERVE_GIB * 2**30)
        r_cap = ram * 2**30
        v_share = min(1.0, v_cap / exp_total)
        vr_share = min(1.0, (v_cap + r_cap) / exp_total)
        best = np.maximum(curves["adaptive_read_hit"], curves["static_read_hit"])  # engine keeps whichever works
        hit_v = float(np.interp(v_share, caps, best))
        # experts missed by VRAM: read by the CPU from RAM (or from the page cache after an SSD fetch)
        reads_per_tok = curves["expert_read_bytes_per_token"]
        gpu_ms = (dense_per_tok + hit_v * reads_per_tok) / (gbw * 1e9 * GPU_EFF) * 1e3
        cpu_ms = (1 - hit_v) * reads_per_tok / (cbw * 1e9) * 1e3
        ssd_bytes = float(np.interp(vr_share, caps, curves["lru_miss_bytes"])) if vr_share < 1 else 0.0
        ssd_ms = ssd_bytes / (ssd * 1e9) * 1e3
        tok_s_overlap = 1e3 / (max(gpu_ms, cpu_ms) + ssd_ms + 1.0)  # +1 ms for 48 GPU/CPU handoffs
        out.append({"device": name, "experts_in_vram": round(v_share, 3), "experts_in_vram_or_ram": round(vr_share, 3),
                    "vram_read_hit": round(hit_v, 3), "gpu_ms": round(gpu_ms, 2), "cpu_ms": round(cpu_ms, 2),
                    "ssd_mb_per_token": round(ssd_bytes / 1e6, 1), "ssd_ms": round(ssd_ms, 2),
                    "decode_tok_s": round(tok_s_overlap, 1)})
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--gguf-py", type=Path, default=Path("/home/louis/bench-engines/llama.cpp-inkling/gguf-py"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-belady", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reqs = load_requests(args.trace, args.requests)
    sizes = gguf_sizes(args.gguf, args.gguf_py)
    key_bytes = np.repeat(sizes["expert"], N_EXPERT)
    exp_total = float(key_bytes.sum())
    n_tok = sum(r.ids.shape[0] for r in reqs)

    counts = usage_counts(reqs)
    per_req = [usage_counts([r]) for r in reqs]
    loo = [counts - c for c in per_req]  # profile for each request from the others

    reads_per_tok = float(sizes["expert"][APP_PHYS].sum() * TOPK)
    summary = {
        "requests": [{"name": r.name, "prompt_tokens": r.prompt_tokens, "decode_tokens": int(r.ids.shape[0])} for r in reqs],
        "decode_tokens": n_tok,
        "expert_bytes_per_layer_MB": [round(x / 1e6, 3) for x in sizes["expert"]],
        "expert_pool_GB": round(exp_total / 1e9, 2),
        "dense_layer_GB": round(float(sizes["dense_layer"].sum()) / 1e9, 3),
        "loop_extra_GB": round(sizes["loop_extra"] / 1e9, 3),
        "other_GB": {k: round(v / 1e9, 3) for k, v in sizes["other"].items() if v > 5e6},
        "expert_read_bytes_per_token_GB": round(reads_per_tok / 1e9, 3),
        "coverage_all_requests": coverage(counts, sizes["expert"]),
        "coverage_loo_mean": {},
        "loop": loop_stats(reqs),
        "temporal": temporal_stats(reqs),
        "gate_weight_share_by_rank": tail_weight_share(reqs),
    }
    # per-layer skew: share of reads served by the 25% most-read experts of that layer
    top25 = []
    for il in range(N_PHYS):
        c = np.sort(counts[il])[::-1]
        top25.append(round(float(c[: N_EXPERT // 4].sum() / c.sum()), 3))
    summary["top25pct_experts_read_share_per_layer"] = top25

    caps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    curves = {"cap_share": caps, "expert_read_bytes_per_token": reads_per_tok}
    for pol in ["static", "adaptive", "lru"] + ([] if args.no_belady else ["belady"]):
        hr, mb = [], []
        for cs in caps:
            res = simulate(pol, reqs, loo, cs * exp_total + 1, key_bytes)
            hr.append(round(res["read_hit"], 4))
            mb.append(res["miss_bytes_per_token"])
            print(f"{pol:9s} cache {cs:4.0%}: read hit {res['read_hit']:.3f}, miss {res['miss_bytes_per_token']/1e6:7.1f} MB/token", flush=True)
        curves[f"{pol}_read_hit"] = hr
        curves[f"{pol}_miss_bytes"] = mb
    summary["cache_curves"] = {k: v for k, v in curves.items()}
    summary["devices"] = device_estimates(curves, sizes)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("decode_tokens", "expert_pool_GB", "coverage_all_requests", "loop", "temporal")}, indent=2))
    for d in summary["devices"]:
        print(d)


if __name__ == "__main__":
    main()

"""The C++ engine (libinkling.so) against the reference forward pass, token by token, on the validation prompts.

    python inkling/engine/tests/check_engine.py ref     # reference logits -> results/ref_logits.pt (needs ~6 GB VRAM)
    python inkling/engine/tests/check_engine.py engine  # engine logits, compared (needs ~19 GB VRAM)

Reports per prompt: top-1 agreement, max |logit difference|, KL(reference || engine), and decode speed.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "engine":
    # the engine links libllama, whose CUDA backend needs a newer libstdc++ than conda's (which numpy and PyTorch
    # load): load the engine first so the system's libstdc++ is the one in the process
    ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "libinkling.so"))
import numpy as np  # noqa: E402

import torch  # noqa: E402

ENGINE = Path(__file__).resolve().parents[1]
_DEFAULT_GGUFS = (
    "/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf",
    "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf",
)
GGUF = os.environ.get("IK_GGUF") or next((p for p in _DEFAULT_GGUFS if Path(p).exists()), _DEFAULT_GGUFS[0])
VALIDATION = ENGINE.parent / "reference" / "results" / "2026-11-21-validate.llama.json"
_stem = Path(GGUF).stem
REF = ENGINE / "tests" / "results" / ("ref_logits.pt" if "Q4_K_XL" in _stem else f"ref_logits_{_stem}.pt")


def sequences():
    runs = json.loads(VALIDATION.read_text())
    return [(r["prompt"], r["prompt_tokens"] + r["gen_tokens"][:-1]) for r in runs]


def run_gen(prompt):
    """the generated tokens (minus the last) of the validation run with this prompt"""
    for r in json.loads(VALIDATION.read_text()):
        if r["prompt"] == prompt:
            return r["gen_tokens"][:-1]
    return []


def ref_phase():
    sys.path.insert(0, str(ENGINE.parent / "reference"))
    from inkling_ref import InklingRef

    m = InklingRef(GGUF)
    out = [m.forward(seq)[0].cpu() for _, seq in sequences()]
    REF.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, REF)
    print(f"saved {REF}")


def engine_phase():
    L = ctypes.CDLL(str(ENGINE / "libinkling.so"))
    L.ik_new.restype = ctypes.c_void_p
    L.ik_new.argtypes = [ctypes.c_char_p, ctypes.c_int]
    L.ik_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    L.ik_reset.argtypes = [ctypes.c_void_p]
    L.ik_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    L.ik_n_vocab.argtypes = [ctypes.c_void_p]
    L.ik_last_error.restype = ctypes.c_char_p
    L.ik_new2.restype = ctypes.c_void_p
    L.ik_new2.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.ik_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    import os
    vram = os.environ.get("IK_VRAM_MIB")
    stats = (ctypes.c_longlong * 10)()
    t0 = time.time()
    if vram is not None:  # offload mode
        prof = os.environ.get("IK_PROFILE", str(ENGINE / "profiles" / "default.f32"))
        h = L.ik_new2(GGUF.encode(), 4096, int(vram), prof.encode() if prof else None, int(os.environ.get("IK_THREADS", "0")))
    else:
        h = L.ik_new(GGUF.encode(), 4096)
    if not h:
        raise SystemExit("ik_new: " + L.ik_last_error().decode())
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    nv = L.ik_n_vocab(h)
    refs = torch.load(REF)
    buf = np.empty(nv, dtype=np.float32)
    only = int(__import__("os").environ.get("IK_ONLY", "0")) or None
    for (prompt, seq), ref in list(zip(sequences(), refs))[:only]:
        assert L.ik_reset(h) == 0
        out = []
        t0 = time.time()
        # IK_PREFILL=1: the prompt (all but its last token) batched, then token by token from there
        n_pf = len(seq) - (len(seq) - len(run_gen(prompt))) - 1 if os.environ.get("IK_PREFILL") else 0
        if n_pf > 0:
            arr = np.array(seq[:n_pf], dtype=np.int32)
            if L.ik_prefill(h, arr.ctypes.data, n_pf) != 0:
                raise SystemExit("ik_prefill: " + L.ik_last_error().decode())
        for tok in seq[n_pf:]:
            if L.ik_decode(h, tok, buf.ctypes.data) != 0:
                raise SystemExit("ik_decode: " + L.ik_last_error().decode())
            out.append(torch.from_numpy(buf.copy()))
        dt = (time.time() - t0) / len(seq)
        eng = torch.stack(out)
        ref = ref[n_pf:]
        top1 = (eng.argmax(-1) == ref.argmax(-1)).float().mean().item()
        lp_r, lp_e = torch.log_softmax(ref.double(), -1), torch.log_softmax(eng.double(), -1)
        kl = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
        L.ik_stats(h, stats) if vram is not None else None
        print(json.dumps({"prompt": prompt[:30], "vram_reads": stats[0], "cpu_experts": stats[1], "swaps": stats[2], "tokens": len(seq), "top1_agreement": round(top1, 4),
                          "max_abs_logit_diff": round((eng - ref).abs().max().item(), 5),
                          "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "ms_per_token": round(dt * 1e3, 2)},
                         ensure_ascii=False), flush=True)
    # decode speed over a longer run (greedy continuation of the first prompt); IK_REPEAT runs it again
    for rep in range(int(os.environ.get("IK_REPEAT", "1"))):
        assert L.ik_reset(h) == 0
        if vram is not None:
            L.ik_stats(h, stats)
        tok = sequences()[0][1][0]
        t0 = time.time()
        n = 256
        for _ in range(n):
            L.ik_decode(h, tok, buf.ctypes.data)
            tok = int(buf.argmax())
        dt = (time.time() - t0) / n
        if rep + 1 < int(os.environ.get("IK_REPEAT", "1")):
            print(f"run {rep + 1}: {1 / dt:.1f} tok/s")
    if vram is not None:
        L.ik_stats(h, stats)
        tot = stats[0] + stats[1] + stats[8]
        print(f"offload {vram} MiB: VRAM hit {stats[0] / max(tot, 1):.3f}, swaps {stats[2]}, CPU experts/token {stats[1] / n:.1f}, "
              f"CPU compute {stats[3] / 1e6 / n:.2f} ms/token over {stats[4] / n:.1f} layer applications "
              f"({stats[3] / max(stats[1], 1) / 1e3:.0f} us per expert)"
              + (f", SSD reads {stats[5] / n:.1f}/token at {stats[6] / max(stats[7], 1):.2f} GB/s" if stats[5] else "")
              + (f", copied to VRAM {stats[8] / n:.1f}/token ({stats[9] / 1e6 / n:.2f} ms/token)" if stats[8] else ""))
    print(f"decode: {1 / dt:.1f} tok/s over {n} tokens ({dt * 1e3:.2f} ms/token)")


if __name__ == "__main__":
    {"ref": ref_phase, "engine": engine_phase}[sys.argv[1]]()

"""Long-sequence check of batched prefill: 1,300 tokens (several 256-token chunks, past the 1,024-position sliding window
and relative-bias extent) prefilled, then the last 8 decoded, against the reference forward pass at those positions.

    python inkling/engine/tests/check_long.py ref      # reference logits (fp32, ~10 GB VRAM)
    python inkling/engine/tests/check_long.py engine   # engine: prefill + decode, compared
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "engine":
    ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "libinkling.so"))  # before numpy / torch (libstdc++)
import numpy as np  # noqa: E402
import torch  # noqa: E402

ENGINE = Path(__file__).resolve().parents[1]
_DEFAULT_GGUFS = (
    "/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf",
    "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf",
)
GGUF = os.environ.get("IK_GGUF") or next((p for p in _DEFAULT_GGUFS if Path(p).exists()), _DEFAULT_GGUFS[0])
N, TAIL = int(os.environ.get("IK_LONG_N", "1300")), 8
OUT = ENGINE / "tests" / "results" / f"long_{Path(GGUF).stem}_{N}.pt"


def tokens():
    rng = np.random.default_rng(7)
    return rng.integers(1000, 150000, size=N).tolist()


def ref_phase():
    sys.path.insert(0, str(ENGINE.parent / "reference"))
    from inkling_ref import InklingRef
    m = InklingRef(GGUF)
    logits, _ = m.forward(tokens())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits[-TAIL:].cpu(), OUT)
    print(f"saved {OUT}")


def engine_phase():
    L = ctypes.CDLL(str(ENGINE / "libinkling.so"))
    vp, ip = ctypes.c_void_p, ctypes.c_int
    L.ik_new.restype = vp; L.ik_new.argtypes = [ctypes.c_char_p, ip]
    L.ik_new2.restype = vp; L.ik_new2.argtypes = [ctypes.c_char_p, ip, ip, ctypes.c_char_p, ip]
    L.ik_prefill.argtypes = [vp, vp, ip]; L.ik_decode.argtypes = [vp, ip, vp]; L.ik_n_vocab.argtypes = [vp]
    L.ik_last_error.restype = ctypes.c_char_p
    vram = os.environ.get("IK_VRAM_MIB")
    if vram:
        h = L.ik_new2(GGUF.encode(), 4096, int(vram), str(ENGINE / "profiles" / "default.f32").encode(), 0)
    else:
        h = L.ik_new(GGUF.encode(), 4096)
    if not h:
        raise SystemExit(L.ik_last_error().decode())
    toks = tokens()
    arr = np.array(toks[:N - TAIL], dtype=np.int32)
    t0 = time.time()
    if L.ik_prefill(h, arr.ctypes.data, N - TAIL) != 0:
        raise SystemExit("ik_prefill: " + L.ik_last_error().decode())
    dt = time.time() - t0
    buf = np.empty(L.ik_n_vocab(h), np.float32)
    out = []
    for tok in toks[N - TAIL:]:
        assert L.ik_decode(h, tok, buf.ctypes.data) == 0, L.ik_last_error()
        out.append(torch.from_numpy(buf.copy()))
    eng, ref = torch.stack(out), torch.load(OUT)
    lp_r, lp_e = torch.log_softmax(ref.double(), -1), torch.log_softmax(eng.double(), -1)
    kl = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
    print(json.dumps({"tokens": N, "prefill_tok_s": round((N - TAIL) / dt, 1), "top1_agreement": float((eng.argmax(-1) == ref.argmax(-1)).float().mean()),
                      "max_abs_logit_diff": round(float((eng - ref).abs().max()), 5), "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}))


if __name__ == "__main__":
    {"ref": ref_phase, "engine": engine_phase}[sys.argv[1]]()

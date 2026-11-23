"""Conversation checkpoint: prompt -> save -> 400 other tokens (past the sliding-window ring's margin) -> restore ->
continuation must give the same logits, bit for bit, as prompt -> continuation from a fresh start.

    python inkling/engine/tests/check_checkpoint.py [IK_VRAM_MIB=... for offload mode]
"""
import ctypes, os, random, sys
from pathlib import Path

L = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "libinkling.so"))
vp, ip = ctypes.c_void_p, ctypes.c_int
for name, res, args in (("ik_new", vp, [ctypes.c_char_p, ip]), ("ik_new2", vp, [ctypes.c_char_p, ip, ip, ctypes.c_char_p, ip]),
                        ("ik_prefill", ip, [vp, vp, ip]), ("ik_decode", ip, [vp, ip, vp]), ("ik_reset", ip, [vp]),
                        ("ik_n_vocab", ip, [vp]), ("ik_checkpoint_save", ip, [vp]), ("ik_checkpoint_restore", ip, [vp]),
                        ("ik_last_error", ctypes.c_char_p, [])):
    getattr(L, name).restype, getattr(L, name).argtypes = res, args
_DEFAULT_GGUFS = (
    "/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf",
    "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf",
)
GGUF = (os.environ.get("IK_GGUF") or next((p for p in _DEFAULT_GGUFS if Path(p).exists()), _DEFAULT_GGUFS[0])).encode()
vram = os.environ.get("IK_VRAM_MIB")
h = L.ik_new2(GGUF, 4096, int(vram), None, 0) if vram else L.ik_new(GGUF, 4096)
assert h, L.ik_last_error()
nv = L.ik_n_vocab(h)
rng = random.Random(3)
prompt = [rng.randrange(1000, 150000) for _ in range(600)]
other = [rng.randrange(1000, 150000) for _ in range(400)]
cont = [rng.randrange(1000, 150000) for _ in range(50)]
buf = (ctypes.c_float * nv)()


def run_prompt():
    arr = (ctypes.c_int * (len(prompt) - 1))(*prompt[:-1])
    assert L.ik_prefill(h, arr, len(prompt) - 1) == 0, L.ik_last_error()
    assert L.ik_decode(h, prompt[-1], buf) == 0


def feed(toks):
    out = []
    for t in toks:
        assert L.ik_decode(h, t, buf) == 0, L.ik_last_error()
        out.append(list(buf[i] for i in range(0, nv, 997)))
    return out


run_prompt()
assert L.ik_checkpoint_save(h) == 0
feed(other)
assert L.ik_checkpoint_restore(h) == 0
a = feed(cont)
L.ik_reset(h)
run_prompt()
b = feed(cont)
same = a == b
diff = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
print(f"checkpoint restore vs fresh: identical {same}, max |logit difference| {diff:.2e}")
sys.exit(0 if same else 1)

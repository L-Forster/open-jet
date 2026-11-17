#!/usr/bin/env python3
"""GPU checks of the C API: engine errors and their roll-backs, penalties, policies, the state digest after a reuse
(scripts/gate.sh runs them).
usage: capi_checks.py <libqwen38mk.so> <model.gguf> <draft_vocab> [case ...]

It runs the given cases (default: every case), each in its own process, and prints one verdict each: PASS, FAIL, or
SKIP with the reason (a case that cannot test what it is for on this machine). Exit status 1 when any case FAILed or
crashed, 2 on an unknown case, else 0.

The out-of-memory cases never ask the device for more than its free VRAM, so they need no spill and give the same answer on
every run (on WDDM / WSL2 the driver's CUDA sysmem fallback lets an allocation past the VRAM succeed in system memory, charged
to Windows commit, so "ask for 30 GB and expect a failure" spilled 20-28 GB and failed only when Windows commit ran out):
  slots_oom / ctor_oom      the out-of-memory comes from the device's current memory pool, capped (cuMemPoolCreate maxSize)
                            far below the free VRAM: the engine's own cudaMallocAsync fails part way, on any driver.
  slots_devoom / ctor_devoom  a device out-of-memory: the VRAM is filled to a known margin (a cuMemAlloc sized from
                            cuMemGetInfo); the pool is capped at the margin + 1 GiB, so with the sysmem fallback on at most
                            1 GiB goes past the VRAM, and then the case SKIPs ("sysmem fallback active").
On WSL2 every device allocation is charged to Windows commit (also inside the VRAM): the devoom cases start only with
CAPI_COMMIT_MIN_GB (default 45) of Windows commit free. CAPI_DEVOOM=0 skips them."""
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

LIB, MODEL, DV = sys.argv[1:4]
CASES = ("penalty", "slots_dup", "slot_ck", "slots_oom", "slots_fit", "ctor_oom", "slots_devoom", "ctor_devoom", "policy",
         "model_values", "digest_q8", "digest_D", "digest_F", "digest_W1024", "park_step")
PROMPT = b"<|im_start|>user\nWrite a short poem about a lighthouse.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
vp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
SKIP_RC = 77  # a case's exit code for SKIP (automake's convention)
MiB = 1 << 20
GiB = 1 << 30
# the slots of the out-of-memory cases: 2 x 40000 positions, ~1.7 GB a slot on the 27B (q8 KV), so with ~2.5 GiB left the
# first slot is built and the second fails part way; the retry (2 x 4096, ~0.37 GB a slot) fits in what the first freed
SLOT_N, SLOT_CTX = 2, 40000
CAP_SLOTS = 2560 * MiB   # slots_oom: the pool cap
CAP_CTOR = 3 * GiB       # ctor_oom: the pool cap (engine 2 needs ~17 GB)
MARGIN = 2560 * MiB      # the devoom cases: free VRAM left beside the filler
OVER = 1 * GiB           # the devoom cases: the pool cap over the margin (the most that can go past the VRAM)
HEADROOM = 1536 * MiB    # free VRAM a capped case keeps outside its cap (non-pool allocations, the runtime)
LEFT_MIB = 64            # device memory a failed call may leave behind (MiB; rounding, the runtime's own)
CTX_MIB = 512            # device memory the first engine may leave behind after its free (MiB: the engine's module code
                         # and the context's local memory stay loaded in the process)


# ---------------------------------------------------------------------------------------------------------------- driver API
class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class PoolProps(ctypes.Structure):  # CUmemPoolProps (cuda.h, CUDA 12.2+: maxSize; 88 bytes)
    _fields_ = [("allocType", ctypes.c_int), ("handleTypes", ctypes.c_int), ("location", Loc),
                ("win32SecurityAttributes", ctypes.c_void_p), ("maxSize", ctypes.c_size_t), ("usage", ctypes.c_ushort),
                ("reserved", ctypes.c_ubyte * 54)]


class Drv:
    """the CUDA driver API on the device's primary context (the one the engine's runtime uses)"""

    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.dev = ctypes.c_int()
        self.ctx = ctypes.c_void_p()
        for r in (self.cu.cuInit(0), self.cu.cuDeviceGet(ctypes.byref(self.dev), 0),
                  self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), self.cu.cuCtxSetCurrent(self.ctx)):
            if r != 0:
                raise RuntimeError(f"CUDA driver API setup failed ({r})")
        self.dflt = ctypes.c_void_p()
        self.cu.cuDeviceGetDefaultMemPool(ctypes.byref(self.dflt), self.dev)
        self.pool = None

    def free(self):
        """free device memory, bytes (after the device is idle)"""
        self.cu.cuCtxSynchronize()
        fr, tot = ctypes.c_size_t(), ctypes.c_size_t()
        self.cu.cuMemGetInfo_v2(ctypes.byref(fr), ctypes.byref(tot))
        return fr.value

    def stats(self, pool=None):
        """the pool's reserved / used, current and high-water (MiB)"""
        p = pool or self.pool
        vals = []
        for a in (5, 6, 7, 8):  # CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, _HIGH, USED_MEM_CURRENT, _HIGH
            v = ctypes.c_uint64()
            self.cu.cuMemPoolGetAttribute(p, a, ctypes.byref(v))
            vals.append(v.value // MiB)
        return dict(zip(("reserved", "reserved_high", "used", "used_high"), vals))

    def _pool(self, nbytes):
        pr = PoolProps()
        pr.allocType, pr.handleTypes, pr.location.type, pr.location.id, pr.maxSize = 1, 0, 1, self.dev.value, nbytes
        pool = ctypes.c_void_p()
        return pool if self.cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(pr)) == 0 else None

    def cap_works(self):
        """does the driver enforce a pool's maxSize? (a 64 MiB pool must refuse one 128 MiB allocation; nothing past the
        free VRAM is asked for either way)"""
        pool = self._pool(64 * MiB)
        if not pool:
            return False
        p = ctypes.c_uint64()
        r = self.cu.cuMemAllocFromPoolAsync(ctypes.byref(p), ctypes.c_size_t(128 * MiB), pool, None)
        if r == 0:
            self.cu.cuMemFreeAsync(p, None)
        self.cu.cuCtxSynchronize()
        self.cu.cuMemPoolDestroy(pool)
        return r != 0

    def cap(self, nbytes):
        """a pool of at most nbytes on the device, made its current pool (the engine's cudaMallocAsync source). None when
        the driver does not enforce the cap (checked first, see cap_works) or the engine does not allocate from pools."""
        if os.environ.get("MK_CUDAMALLOC", "")[:1] == "1" or not self.cap_works():
            return None
        pool = self._pool(nbytes)
        if not pool:
            return None
        self.cu.cuDeviceSetMemPool(self.dev, pool)
        self.pool = pool
        return pool

    def trim(self):
        """the capped pool's free memory back to the device"""
        self.cu.cuCtxSynchronize()
        if self.pool:
            self.cu.cuMemPoolTrimTo(self.pool, ctypes.c_size_t(0))

    def uncap(self):
        """the default pool current again, the capped one destroyed (call after everything allocated from it is freed)"""
        self.cu.cuCtxSynchronize()
        self.cu.cuDeviceSetMemPool(self.dev, self.dflt)
        if self.pool:
            self.cu.cuMemPoolDestroy(self.pool)
            self.pool = None

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        return p if self.cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0 else None

    def release(self, p):
        if p:
            self.cu.cuMemFree_v2(p)


# ------------------------------------------------------------------------------------------------------------------ engine
def smi(field):
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def vram():
    return smi("memory.used")


def lib():
    L = ctypes.CDLL(LIB)
    L.mk_new2.restype, L.mk_new2.argtypes = vp, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    L.mk_last_error.restype = ctypes.c_char_p
    L.mk_free.argtypes = [vp]
    L.mk_set_penalty.argtypes = [vp, ctypes.c_float, ctypes.c_float]
    L.mk_prefill.argtypes = [vp, ip, ctypes.c_int]
    L.mk_step.argtypes = [vp, ip, ctypes.c_int]
    L.mk_slots_open2.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mk_slot_begin2.argtypes = [vp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                 ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]
    L.mk_slot_prefill.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    L.mk_slots_step.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]
    L.mk_slots_step_mixed.argtypes = [vp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int, ip, ctypes.c_int, ip, ip]
    L.mk_slot_sequence.argtypes = [vp, ctypes.c_int, ip, ctypes.c_int]
    L.mk_tokenize.argtypes = [vp, ctypes.c_char_p, ctypes.c_int, ip, ctypes.c_int]
    return L


def err(L):
    return L.mk_last_error().decode(errors="replace")


def new(L, ctx=8192):
    return L.mk_new2(MODEL.encode(), ctx, 7, DV.encode(), 1, 0)  # (kv_mode 1: q8_0, which the slots need)


def ids(L, h):
    out = (ctypes.c_int * 512)()
    n = L.mk_tokenize(h, PROMPT, len(PROMPT), out, 512)
    assert n > 0, err(L)
    return (ctypes.c_int * n)(*out[:n]), n


def check(ok, what, L=None):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok or L is None else f" ({err(L)})"), flush=True)
    return ok


def note(s):
    print("     " + s, flush=True)


def skip(why):
    """the case cannot test what it is for here: SKIP (the caller returns None)"""
    print("SKIP " + why, flush=True)


def main_steps(L, h, k):
    out = (ctypes.c_int * 16)()
    return all(L.mk_step(h, out, 16) > 0 for _ in range(k))


def slot_steps(L, h, slots, k):
    sa, out, cnt = (ctypes.c_int * len(slots))(*slots), (ctypes.c_int * (16 * len(slots)))(), (ctypes.c_int * len(slots))()
    for _ in range(k):
        if L.mk_slots_step(h, sa, len(slots), out, 16, cnt) != 0 or not all(cnt[i] > 0 for i in range(len(slots))):
            return False
    return True


def begin(L, h, slot, toks, n):
    return L.mk_slot_begin2(h, slot, toks, n, 0.0, 1.0, 20, 0.0, 1, 0, 0) >= 0 and L.mk_slot_prefill(h, slot, 1 << 30) == 0


def is_oom(L):
    return "out of memory" in err(L)


NOCAP = ("the engine's allocations cannot be capped here: {}")


def nocap_why():
    if os.environ.get("MK_CUDAMALLOC", "")[:1] == "1":
        return NOCAP.format("MK_CUDAMALLOC=1 (one cudaMalloc per buffer, outside any memory pool), so the request is not bounded")
    return NOCAP.format("the driver does not enforce a memory pool's maxSize (a 128 MiB allocation from a 64 MiB pool succeeded)")


# ------------------------------------------------------------------------------------------------------------------- cases
def case_penalty(L):
    # penalties on the engine's own sequence, then slots: the lone-slot step used to apply them to the lane (exit 4) and a
    # slot reset used to clear the lane's missing counts (-1, slot failed)
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    toks, n = ids(L, h)
    ok = check(L.mk_set_penalty(h, 0.5, 0.2) == 0, "set_penalty 0.5 / 0.2", L)
    ok &= check(L.mk_prefill(h, toks, n) >= 0 and main_steps(L, h, 3), "own sequence: prefill + 3 penalised steps", L)
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "slots_open 2 x 4096", L)
    ok &= check(begin(L, h, 0, toks, n), "slot 0: begin + prefill", L)
    ok &= check(slot_steps(L, h, [0], 4), "slot 0 alone: 4 steps (the solo launch)", L)
    ok &= check(begin(L, h, 0, toks, n), "slot 0 again: begin (truncate / rewind) + prefill", L)
    ok &= check(begin(L, h, 1, toks, n) and slot_steps(L, h, [0, 1], 3), "slots 0 + 1: 3 batched steps", L)
    ok &= check(main_steps(L, h, 2), "own sequence after the slot calls: 2 penalised steps", L)
    L.mk_free(h)
    return ok


def slot_seq(L, h, slot):
    out = (ctypes.c_int * 8192)()
    return out[:L.mk_slot_sequence(h, slot, out, 8192)]


def case_slots_dup(L):
    # a slot listed twice is refused before anything runs: five ids used to write past Launch::ln, and two rows of one
    # batched launch updated one slot's KV cache; mk_slots_step_mixed refuses before it feeds the prompt chunk
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    toks, n = ids(L, h)
    ok = check(L.mk_slots_open2(h, 4, 4096, 0) > 0, "slots_open 4 x 4096", L)
    for s in range(4):
        ok &= check(begin(L, h, s, toks, n), f"slot {s}: begin + prefill", L)
    ok &= check(slot_steps(L, h, [0, 1, 2, 3], 2), "slots 0-3: 2 batched steps", L)
    before = [slot_seq(L, h, s) for s in range(4)]

    def arrays(slots):
        return (ctypes.c_int * len(slots))(*slots), (ctypes.c_int * (16 * len(slots)))(), (ctypes.c_int * len(slots))()
    for slots in ([0, 0], [3, 1, 3], [0, 1, 2, 3, 0]):
        sa, out, cnt = arrays(slots)
        ok &= check(L.mk_slots_step(h, sa, len(slots), out, 16, cnt) == -1 and "listed twice" in err(L), f"step {slots}: refused", L)
    ok &= check([slot_seq(L, h, s) for s in range(4)] == before, "the refused steps changed no slot")
    ok &= check(L.mk_slot_begin2(h, 0, toks, n, 0.0, 1.0, 20, 0.0, 1, 0, 0) >= 0, "slot 0: begin a prompt", L)
    fed = len(slot_seq(L, h, 0))
    sa, out, cnt = arrays([1, 1])
    left = ctypes.c_int(-1)
    ok &= check(L.mk_slots_step_mixed(h, 0, 64, sa, 2, out, 16, cnt, ctypes.byref(left)) == -1 and "listed twice" in err(L),
                "mixed step [1, 1]: refused", L)
    ok &= check(len(slot_seq(L, h, 0)) == fed, "the refused mixed step fed no prompt chunk")
    ok &= check(L.mk_slot_prefill(h, 0, 1 << 30) == 0, "slot 0: its prompt still prefills", L)
    ok &= check(slot_steps(L, h, [0, 1, 2, 3], 2), "slots 0-3: 2 more batched steps", L)
    L.mk_free(h)
    return ok


def slot_ids(L, h, slot, k):
    """the next k tokens slot emits alone (None: a step failed or its context is full)"""
    sa, out, cnt, got = (ctypes.c_int * 1)(slot), (ctypes.c_int * 16)(), (ctypes.c_int * 1)(), []
    while len(got) < k:
        if L.mk_slots_step(h, sa, 1, out, 16, cnt) != 0 or cnt[0] <= 0:
            return None
        got += out[:cnt[0]]
    return got[:k]


def case_slot_ck(L):
    # MK_EXACT: a slot prompt fed in chunks whose last chunk would be <= 32 tokens past a grid point (it used to save no
    # checkpoint, so a repeat of the prompt reused nothing): the chunk before it takes them, which changes no row
    os.environ["MK_EXACT"] = "1"
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    text = reuse_text(20)
    buf = (ctypes.c_int * 4096)()
    n = L.mk_tokenize(h, text, len(text), buf, 4096)
    ok = check(n >= 140, f"the text has {n} >= 140 tokens", L)
    toks = (ctypes.c_int * 100)(*buf[:100])
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "slots_open 2 x 4096", L)
    ok &= check(L.mk_slot_begin2(h, 0, toks, 100, 0.0, 1.0, 20, 0.0, 1, 0, 0) == 0, "slot 0: begin 100 tokens", L)
    left = [L.mk_slot_prefill(h, 0, 32) for _ in range(3)]
    ok &= check(left == [68, 36, 0], f"slot 0 fed with budget 32: {left} left after each call (68, 36, 0: the third "
                                     "chunk takes the 4 tokens past 96)", L)
    ok &= check(begin(L, h, 1, toks, 100), "slot 1: begin + the whole prompt in one call", L)
    got = [slot_ids(L, h, s, 16) for s in (0, 1)]
    ok &= check(got[0] is not None and got[0] == got[1], f"chunked and whole: the same 16 greedy ids ({got[0]} / {got[1]})", L)
    for s in (0, 1):
        r = L.mk_slot_begin2(h, s, toks, 100, 0.0, 1.0, 20, 0.0, 1, 0, 0)
        ok &= check(r == 96, f"slot {s}: the prompt again resumes at its checkpoint at 96 (reused {r})", L)
    # a follow-up holding everything slot 0 decoded (its last prompt + reply + a new turn), sent with no slot_begin of the
    # last prompt first: the engine goes back to the checkpoint inside that prompt itself (it used to keep the reply's
    # decoded rows), so the reply is the one of a slot that prefills the follow-up from scratch
    ok &= check(L.mk_slot_prefill(h, 0, 1 << 30) == 0 and slot_ids(L, h, 0, 16) is not None, "slot 0: 16 more tokens", L)
    fu = slot_seq(L, h, 0) + list(buf[100:140])
    fa = (ctypes.c_int * len(fu))(*fu)
    r = L.mk_slot_begin2(h, 0, fa, len(fu), 0.0, 1.0, 20, 0.0, 1, 0, 0)
    ok &= check(r == 96, f"slot 0: the follow-up resumes at 96, inside its last prompt (reused {r} of {len(fu)})", L)
    ok &= check(L.mk_slot_prefill(h, 0, 1 << 30) == 0, "slot 0: the follow-up prefilled", L)
    other = (ctypes.c_int * 50)(*buf[50:100])
    ok &= check(begin(L, h, 1, other, 50) and begin(L, h, 1, fa, len(fu)), "slot 1: another prompt, then the follow-up", L)
    got = [slot_ids(L, h, s, 16) for s in (0, 1)]
    ok &= check(got[0] is not None and got[0] == got[1], f"follow-up: the same 16 greedy ids as from scratch ({got[0]} / {got[1]})", L)
    L.mk_free(h)
    return ok


def case_model_values(L):
    # a model whose RoPE base or RMS epsilon is not the one compiled in is refused. A sparse copy of the model with one key
    # changed: the same size and header, no tensor data, as neither the vocabulary nor the refusal reads any
    with open(MODEL, "rb") as f:
        head = f.read(64 * MiB)
    ok = True
    for key, val in ((b"qwen35.rope.freq_base", 10000.0), (b"qwen35.attention.layer_norm_rms_epsilon", 1e-5)):
        at = head.find(struct.pack("<Q", len(key)) + key)
        if not check(at >= 0, f"{key.decode()} is in the model's first 64 MiB"):
            ok = False
            continue
        at += 8 + len(key)
        if not check(struct.unpack_from("<I", head, at)[0] == 6, f"{key.decode()} is a float32"):
            ok = False
            continue
        changed = bytearray(head)
        struct.pack_into("<f", changed, at + 4, val)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "changed.gguf")
            with open(path, "wb") as f:
                f.write(changed)
                f.truncate(os.path.getsize(MODEL))
            h = L.mk_new2(path.encode(), 8192, 7, DV.encode(), 1, 0)
            ok &= check(not h and key.decode() in err(L), f"{key.decode()} = {val:g}: refused ({err(L)})")
            if h:
                L.mk_free(h)
    h = new(L)
    ok &= check(bool(h), "the model itself loads", L)
    if h:
        L.mk_free(h)
    return ok


def slots_rollback(L, D, h, what, devoom=False):
    """slots_open(SLOT_N, SLOT_CTX) under the current limit: fails part way (out of memory), everything it allocated goes
    back, a retry of 2 x 4096 succeeds and runs. True / False; None when the open did not fail, or (devoom) when its
    allocations went past the free VRAM (the caller decides)."""
    f0 = D.free()
    r = L.mk_slots_open2(h, SLOT_N, SLOT_CTX, 0)
    e = err(L) if r < 0 else "opened"
    st = D.stats()
    note(f"slots_open {SLOT_N} x {SLOT_CTX}: {r} ({e}); capped pool after it (MiB): {st}; device free before it {f0 // MiB} MiB")
    if r > 0 or (devoom and st["reserved_high"] > (f0 + 64 * MiB) // MiB):
        return None
    ok = check(is_oom(L), f"slots_open {SLOT_N} x {SLOT_CTX} fails with out of memory ({what})", L)
    ok &= check(st["used_high"] >= 1024, f"it failed part way: {st['used_high']} MiB of it was allocated first (>= 1024: slot 0 built)")
    ok &= check(st["used"] == 0, f"roll-back: {st['used']} MiB of it still allocated (0)")
    D.trim()
    f1 = D.free()
    ok &= check((f0 - f1) // MiB < LEFT_MIB, f"roll-back: device free memory {(f0 - f1) // MiB} MiB below the value before the open (< {LEFT_MIB})")
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "then slots_open 2 x 4096 succeeds", L)
    toks, n = ids(L, h)
    ok &= check(begin(L, h, 0, toks, n) and slot_steps(L, h, [0], 3), "slot 0: begin + prefill + 3 steps", L)
    return ok


def case_slots_oom(L):
    # slots_open failing part way through its allocations rolls back: a second open succeeds (it used to say "slots are open
    # already" over the half-built lanes). The failure: the pool the engine allocates from is capped below the free VRAM.
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < CAP_SLOTS + HEADROOM:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine: the capped pool needs {(CAP_SLOTS + HEADROOM) // MiB}")
        return None
    if not D.cap(CAP_SLOTS):
        L.mk_free(h)
        skip(nocap_why())
        return None
    note(f"device free {free // MiB} MiB beside the engine; the engine's slots now come from a pool capped at {CAP_SLOTS // MiB} MiB")
    ok = slots_rollback(L, D, h, f"the pool's {CAP_SLOTS // MiB} MiB cap")
    if ok is None:
        ok = check(False, f"slots_open {SLOT_N} x {SLOT_CTX} fails (it opened inside a {CAP_SLOTS // MiB} MiB pool)")
    L.mk_free(h)
    D.uncap()
    return ok


def case_slots_fit(L):
    # slots that cannot fit in the free device memory are refused before anything is allocated (they used to be
    # allocated until the device ran out, and with the sysmem fallback went on into system memory). Bounded on any build:
    # the pool is capped below the free VRAM, so a build without the check fails at the cap instead of spilling.
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < CAP_SLOTS + HEADROOM:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine: the capped pool needs {(CAP_SLOTS + HEADROOM) // MiB}")
        return None
    if not D.cap(free - HEADROOM):
        L.mk_free(h)
        skip(nocap_why())
        return None
    r = L.mk_slots_open2(h, 4, 200000, 0)
    e = err(L) if r < 0 else "opened"
    st = D.stats()
    note(f"device free {free // MiB} MiB beside the engine; slots_open 4 x 200000 (~30 GB): {r} ({e}); capped pool after it (MiB): {st}")
    ok = check(r < 0, "slots_open 4 x 200000 fails")
    ok &= check(st["used_high"] == 0, f"it was refused before allocating anything ({st['used_high']} MiB allocated first)")
    ok &= check("of device memory" in e, "the error says what it needs and what is free")
    ok &= check(L.mk_slots_open2(h, 2, 4096, 0) > 0, "then slots_open 2 x 4096 succeeds", L)
    toks, n = ids(L, h)
    ok &= check(begin(L, h, 0, toks, n) and slot_steps(L, h, [0], 3), "slot 0: begin + prefill + 3 steps", L)
    L.mk_free(h)
    D.uncap()
    return ok


def ctor_rollback(L, D, v0, h1, what, devoom=False, fill=None):
    """engine 2 under the current limit fails part way; everything it allocated goes back; engine 1's free gives everything
    back; engine 3 starts. None (devoom) when engine 2's allocations went past the free VRAM (the sysmem fallback)."""
    v1 = D.free()
    h2 = new(L)
    st = D.stats()
    e2 = err(L)
    D.trim()
    v2 = D.free()
    note(f"engine 2: {'up' if h2 else e2}; capped pool after it: {st}; device free MiB: before engine 1 {v0 // MiB}, "
         f"engine 1 up {v1 // MiB}, after engine 2's attempt {v2 // MiB}")
    if h2:
        L.mk_free(h2)
        D.uncap()
        D.release(fill)
        L.mk_free(h1)
        return check(False, f"engine 2 fails (it came up despite {what})")
    if devoom and st["reserved_high"] > (v1 + 64 * MiB) // MiB:
        D.uncap()
        D.release(fill)
        L.mk_free(h1)
        return "fallback", st
    ok = check(is_oom(L), f"engine 2 fails with out of memory ({what})", L)
    ok &= check(st["used_high"] >= 1024, f"it failed part way: {st['used_high']} MiB of it was allocated first (>= 1024)")
    ok &= check(st["used"] == 0, f"engine 2's pool allocations all freed ({st['used']} MiB left)")
    ok &= check((v1 - v2) // MiB < LEFT_MIB, f"engine 2's attempt left {(v1 - v2) // MiB} MiB of device memory behind (< {LEFT_MIB})")
    D.uncap()
    D.release(fill)
    vu = D.free()
    L.mk_free(h1)
    v3 = D.free()
    note(f"device free MiB: the capped pool (and the filler) gone {vu // MiB}, after freeing engine 1 {v3 // MiB}")
    ok &= check((v0 - v3) // MiB < CTX_MIB, f"after freeing engine 1: device free memory {(v0 - v3) // MiB} MiB below the start "
                f"(< {CTX_MIB}: module code, local memory)")
    h3 = new(L)
    ok &= check(bool(h3), "engine 3 starts", L)
    if h3:
        L.mk_free(h3)
    return ok


def case_ctor_oom(L):
    # a second engine that runs out of memory part way frees what it allocated (it used to keep it), the first one's free
    # gives everything back, and a third starts. The failure: engine 2 allocates from a pool capped below the free VRAM.
    D = Drv()
    v0 = D.free()
    h1 = new(L)
    if not check(bool(h1), "engine 1", L):
        return False
    free = D.free()
    if free < CAP_CTOR + HEADROOM:
        L.mk_free(h1)
        skip(f"only {free // MiB} MiB of device memory free beside engine 1: the capped pool needs {(CAP_CTOR + HEADROOM) // MiB}")
        return None
    if not D.cap(CAP_CTOR):
        L.mk_free(h1)
        skip(nocap_why())
        return None
    note(f"device free {free // MiB} MiB beside engine 1; engine 2 allocates from a pool capped at {CAP_CTOR // MiB} MiB")
    return ctor_rollback(L, D, v0, h1, f"the pool's {CAP_CTOR // MiB} MiB cap")


FALLBACK = ("sysmem fallback active: an OOM cannot be forced ({}: the allocations went on past the VRAM into system memory, "
            "the Windows driver's CUDA sysmem fallback). To run this case: NVIDIA Control Panel > Manage 3D settings > CUDA - "
            "Sysmem Fallback Policy = Prefer No Sysmem Fallback (WSL2 follows the Windows driver), or a native Linux driver")


def devoom_off():
    if os.environ.get("CAPI_DEVOOM", "1") == "0":
        skip("CAPI_DEVOOM=0: the device out-of-memory cases are off")
        return True
    return False


def case_slots_devoom(L):
    # as slots_oom, with a device out-of-memory: the VRAM filled to MARGIN free; the pool capped at MARGIN + OVER, so with
    # the sysmem fallback on at most OVER goes past the VRAM (the open then succeeds or stops at the cap: SKIP)
    if devoom_off():
        return None
    D = Drv()
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    free = D.free()
    if free < MARGIN + 256 * MiB:
        L.mk_free(h)
        skip(f"only {free // MiB} MiB of device memory free beside the engine (the case leaves {MARGIN // MiB})")
        return None
    fill = D.alloc(free - MARGIN)
    if not fill:
        L.mk_free(h)
        skip(f"the filler ({(free - MARGIN) // MiB} MiB, inside the free VRAM) could not be allocated")
        return None
    if not D.cap(D.free() + OVER):
        D.release(fill)
        L.mk_free(h)
        skip(nocap_why())
        return None
    note(f"VRAM filled to {D.free() // MiB} MiB free; the engine's slots now come from a pool capped {OVER // MiB} MiB above that")
    os.environ["MK_SLOT_SPILL"] = "1"  # (past slots_open's up-front check, so the device itself runs out)
    ok = slots_rollback(L, D, h, "the VRAM is full", devoom=True)
    del os.environ["MK_SLOT_SPILL"]
    st = D.stats()
    L.mk_free(h)
    D.uncap()
    D.release(fill)
    if ok is None:
        skip(FALLBACK.format(f"slots_open {SLOT_N} x {SLOT_CTX} got {st['reserved_high']} MiB of pool with {MARGIN // MiB} MiB of VRAM free"))
    return ok


def case_ctor_devoom(L):
    # as ctor_oom, with a device out-of-memory: the VRAM filled to MARGIN free beside engine 1, engine 2's pool capped at
    # MARGIN + OVER (with the sysmem fallback on, engine 2 stops at the cap after at most OVER past the VRAM: SKIP)
    if devoom_off():
        return None
    D = Drv()
    v0 = D.free()
    h1 = new(L)
    if not check(bool(h1), "engine 1", L):
        return False
    free = D.free()
    if free < MARGIN + 256 * MiB:
        L.mk_free(h1)
        skip(f"only {free // MiB} MiB of device memory free beside engine 1 (the case leaves {MARGIN // MiB})")
        return None
    fill = D.alloc(free - MARGIN)
    if not fill:
        L.mk_free(h1)
        skip(f"the filler ({(free - MARGIN) // MiB} MiB, inside the free VRAM) could not be allocated")
        return None
    if not D.cap(D.free() + OVER):
        D.release(fill)
        L.mk_free(h1)
        skip(nocap_why())
        return None
    note(f"VRAM filled to {D.free() // MiB} MiB free beside engine 1; engine 2 allocates from a pool capped {OVER // MiB} MiB above that")
    r = ctor_rollback(L, D, v0, h1, "the VRAM is full", devoom=True, fill=fill)
    if isinstance(r, tuple):
        skip(FALLBACK.format(f"engine 2's pool grew to {r[1]['reserved_high']} MiB with {MARGIN // MiB} MiB of VRAM free"))
        return None
    return r


def case_policy(L):
    # a constructor that throws after the engine is built (a bad MK_POLICY) deletes it (it used to leak the whole engine)
    v0 = vram()
    os.environ["MK_POLICY"] = "garbage"  # (no key=value: set_policy throws after the Impl is built)
    h = new(L)
    v1 = vram()
    note(f"VRAM MiB: before {v0}, after the failed start {v1}; engine: {'up' if h else err(L)}")
    ok = check(not h, "MK_POLICY=garbage: the engine refuses to start")
    ok &= check(v1 - v0 < 1024, f"the failed start left {v1 - v0} MiB behind (< 1024: the CUDA context)")
    if h:
        L.mk_free(h)
    del os.environ["MK_POLICY"]
    h = new(L)
    ok &= check(bool(h), "without it the engine starts", L)
    if h:
        L.mk_free(h)
    return ok


def case_park_step(L):
    # mk_step right after mk_unpark / mk_side_end with drafts the last cycle accepted not yet in the DeltaNet state: the
    # device holds another sequence's replay rows, so the step is refused until a prefill, which then continues the
    # conversation exactly as without the park (qwen38_mk.h, park)
    L.mk_store_budget.argtypes = [vp, ctypes.c_longlong]
    L.mk_set_sampling.argtypes = [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong]
    L.mk_park.argtypes = [vp, ctypes.c_int]
    L.mk_unpark.argtypes = [vp, ctypes.c_int]
    L.mk_reset.argtypes = [vp]
    L.mk_side_begin.argtypes = [vp]
    L.mk_side_end.argtypes = [vp]
    L.mk_committed.argtypes = [vp]
    L.mk_sequence.argtypes = [vp, ip, ctypes.c_int]
    h = new(L)
    if not check(bool(h), "engine", L):
        return False
    L.mk_store_budget(h, 2 * GiB)
    L.mk_set_sampling(h, 0.0, 1.0, 0, 0.0, 0)
    out = (ctypes.c_int * 16)()

    def tok(text):
        buf = (ctypes.c_int * (len(text) + 16))()
        n = L.mk_tokenize(h, text, len(text), buf, len(text) + 16)
        assert n > 0, err(L)
        return list(buf[:n])

    def feed(toks):
        first = L.mk_prefill(h, (ctypes.c_int * len(toks))(*toks), len(toks))
        assert first >= 0, err(L)
        return first

    def decode(first, n=48):
        got = [first]
        while len(got) < n:
            k = L.mk_step(h, out, 16)
            assert k > 0, err(L)
            got += list(out[:k])
        return got[:n]

    def pending():  # accepted drafts the next step replays from the DeltaNet ring (position - committed)
        return L.mk_sequence(h, None, 0) - 1 - L.mk_committed(h)

    toks, _ = ids(L, h)
    feed(list(toks))
    for _ in range(32):
        assert L.mk_step(h, out, 16) > 0, err(L)
        if pending() > 0:
            break
    if pending() <= 0:
        skip("no decode cycle accepted a draft in 32 steps (no MTP layer?): nothing pending to test")
        L.mk_free(h)
        return None
    note(f"{pending()} accepted drafts pending")
    pids = [L.mk_park(h, 0), L.mk_park(h, 0)]
    if not check(min(pids) > 0, f"parked twice (ids {pids})", L):
        L.mk_free(h)
        return False
    more = tok(b"<|im_end|>\n<|im_start|>user\nNow one about the sea.<|im_end|>\n<|im_start|>assistant\n")
    other = tok(b"<|im_start|>user\nName three rivers.<|im_end|>\n<|im_start|>assistant\n")
    want = decode(feed(more))  # (the conversation continued without a park)
    L.mk_reset(h)
    decode(feed(other), 16)
    ok = check(L.mk_unpark(h, pids[0]) == 0, "another conversation ran, the parked one is back", L)
    ok &= check(L.mk_step(h, out, 16) < 0 and "prefill first" in err(L), "mk_step right away: refused", L)
    ok &= check(decode(feed(more)) == want, "after a prefill: the 48 greedy ids of the conversation never parked")
    ok &= check(L.mk_unpark(h, pids[1]) == 0 and L.mk_side_begin(h) == 0, "back again, kept aside for a side request", L)
    decode(feed(other), 16)
    ok &= check(L.mk_side_end(h) == 1, "the side request ran, the conversation is back", L)
    ok &= check(L.mk_step(h, out, 16) < 0 and "prefill first" in err(L), "mk_step right away: refused", L)
    ok &= check(decode(feed(more)) == want, "after a prefill: the 48 greedy ids of the conversation never parked")
    L.mk_free(h)
    return ok


def reuse_text(functions):
    lines = ["Explain what each of these functions computes.", "", "```python"]
    for i in range(1, functions + 1):
        lines += [f"def step_{i}(x, y):", f"    return (x * {i} + y) % {997 + i}", ""]
    return "\n".join(lines + ["```"]).encode()


def digest_reuse(L, env):
    """Exact reuse in the sequence state itself: every K / V row, window-ring row and DeltaNet / conv state after a resume at
    a checkpoint, a resume after a branch, a park and restore and a side request, bitwise against a fresh prefill of the
    same tokens, and the next 64 greedy ids."""
    os.environ.update({"MK_EXACT": "1", "MK_CK_SLOTS": "8", **env})
    L.mk_prefill_cks.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]
    L.mk_state_digest.argtypes = [vp, ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
    L.mk_checkpoints.argtypes = [vp, ip, ctypes.c_int]
    L.mk_rewind_to.argtypes = [vp, ctypes.c_int]
    L.mk_reset.argtypes = [vp]
    L.mk_set_sampling.argtypes = [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong]
    L.mk_store_budget.argtypes = [vp, ctypes.c_longlong]
    L.mk_park.argtypes = [vp, ctypes.c_int]
    L.mk_unpark.argtypes = [vp, ctypes.c_int]
    L.mk_side_begin.argtypes = [vp]
    L.mk_side_end.argtypes = [vp]
    L.mk_kv_sizes.argtypes = [vp] + [ctypes.POINTER(ctypes.c_longlong)] * 4
    h = new(L, 32768)
    if not check(bool(h), "the engine starts", L):
        return False
    sizes = [ctypes.c_longlong() for _ in range(4)]
    L.mk_kv_sizes(h, *[ctypes.byref(x) for x in sizes])
    if env and not check(sizes[1].value > 0, "the preset took effect: the window rings exist"):
        return False
    L.mk_store_budget(h, 8 * GiB)
    L.mk_set_sampling(h, 0.0, 1.0, 0, 0.0, 0)

    def tok(text):
        buf = (ctypes.c_int * (len(text) + 16))()
        n = L.mk_tokenize(h, text, len(text), buf, len(text) + 16)
        assert n > 0, err(L)
        return list(buf[:n])

    def feed(toks, cks=()):
        arr = (ctypes.c_int * len(toks))(*toks)
        ca = (ctypes.c_int * max(len(cks), 1))(*cks)
        first = L.mk_prefill_cks(h, arr, len(toks), ca, len(cks), -1)
        assert first >= 0, err(L)
        return first

    def digest():
        n = L.mk_state_digest(h, None, 0)
        assert n > 0, err(L)
        a = (ctypes.c_ulonglong * n)()
        L.mk_state_digest(h, a, n)
        return list(a)

    def decode(first, n=64):
        out, got = (ctypes.c_int * 16)(), [first]
        for _ in range(n):
            if len(got) >= n:
                break
            k = L.mk_step(h, out, 16)
            assert k >= 0, err(L)
            got += list(out[:k])
        return got[:n]

    def resume(toks, at_most):
        """back to the last checkpoint at or before at_most, the rest of toks again: the position"""
        n = L.mk_checkpoints(h, None, 0)
        a = (ctypes.c_int * max(n, 1))()
        L.mk_checkpoints(h, a, n)
        ck = max([c for c in a[:n] if 0 < c <= at_most], default=0)
        assert ck and L.mk_rewind_to(h, ck) == 0, f"no checkpoint at or before {at_most}: {list(a[:n])} ({err(L)})"
        return ck, feed(toks[ck:])

    toks = tok(reuse_text(400))
    N = len(toks)
    grid = list(range(1024, N, 1024))
    note(f"{N} tokens, checkpoints asked at every 1024, rings {sizes[1].value / MiB:.0f} MiB")
    ok = check(N > 4096, "the prompt is past the window")
    first = feed(toks, grid)
    fresh, ids0 = digest(), decode(first)
    ck, first = resume(toks, N - 1)
    ok &= check(digest() == fresh, f"after decoding, a resume at {ck} and the rest again: state bitwise equal to the fresh prefill")
    ok &= check(decode(first) == ids0, "and the next 64 greedy ids equal")
    M = N - 900  # a branch: the same prompt up to M, then other tokens
    other = toks[:M] + tok(reuse_text(40))[:900]
    ck, first = resume(other, M)
    branched = digest()
    decode(first)
    ck2, first = resume(toks, M)
    ok &= check(digest() == fresh, f"back from a branch at {M} by the checkpoint at {ck2}: state bitwise equal to the fresh prefill")
    ok &= check(decode(first) == ids0, "and the next 64 greedy ids equal")
    L.mk_reset(h)
    feed(other, grid)
    ok &= check(digest() == branched, f"the branch resumed at {ck}: state bitwise equal to its own fresh prefill")
    L.mk_reset(h)
    first = feed(toks, grid)
    ok &= check(digest() == fresh, "a second fresh prefill: state bitwise equal to the first")
    pid = L.mk_park(h, 2)
    if not check(pid > 0, f"parked (id {pid})", L):
        L.mk_free(h)
        return False
    L.mk_reset(h)
    decode(feed(tok(reuse_text(60))))
    if not check(L.mk_unpark(h, pid) >= 0, "restored", L):
        L.mk_free(h)
        return False
    ok &= check(digest() == fresh, "parked, another conversation run, restored: state bitwise equal to the fresh prefill")
    ok &= check(decode(first) == ids0, "and the next 64 greedy ids equal")
    L.mk_reset(h)
    first = feed(toks, grid)
    if not check(L.mk_side_begin(h) == 0, "the conversation is kept aside", L):
        L.mk_free(h)
        return False
    decode(feed(tok(reuse_text(60))))
    ok &= check(L.mk_side_end(h) == 1, "a side request ran and the conversation is back", L)
    ok &= check(digest() == fresh, "after the side request: state bitwise equal to the fresh prefill")
    ok &= check(decode(first) == ids0, "and the next 64 greedy ids equal")
    L.mk_free(h)
    return ok


def case_digest_q8(L):
    return digest_reuse(L, {})


def case_digest_D(L):
    return digest_reuse(L, {"MK_KV": "K*=6,V*=5", "MK_KV_WIN": "2048"})


def case_digest_F(L):
    return digest_reuse(L, {"MK_KV": "K*=5r,V*=5", "MK_KV_WIN": "2048"})


def case_digest_W1024(L):
    """MK_EXACT with MK_KV_WIN=1024: window rings of 1952 rows, shorter than the default prefill chunk (2048), which the
    engine then caps at 1920. Resumes at checkpoints inside a chunk (1024: its window starts at 0; 3072), each bitwise
    against the fresh prefill: a checkpoint's ring rows of a chunk longer than the ring held other positions' rows."""
    os.environ.update({"MK_EXACT": "1", "MK_CK_SLOTS": "8", "MK_KV": "K*=6,V*=5", "MK_KV_WIN": "1024"})
    os.environ.pop("MK_PF_CAP", None)
    L.mk_prefill_cks.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]
    L.mk_state_digest.argtypes = [vp, ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
    L.mk_checkpoints.argtypes = [vp, ip, ctypes.c_int]
    L.mk_rewind_to.argtypes = [vp, ctypes.c_int]
    L.mk_set_sampling.argtypes = [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong]
    L.mk_kv_sizes.argtypes = [vp] + [ctypes.POINTER(ctypes.c_longlong)] * 4
    h = new(L, 32768)
    if not check(bool(h), "the engine starts", L):
        return False
    sizes = [ctypes.c_longlong() for _ in range(4)]
    L.mk_kv_sizes(h, *[ctypes.byref(x) for x in sizes])
    if not check(sizes[1].value > 0, "the preset took effect: the window rings exist"):
        L.mk_free(h)
        return False
    L.mk_set_sampling(h, 0.0, 1.0, 0, 0.0, 0)
    text = reuse_text(400)
    buf = (ctypes.c_int * (len(text) + 16))()
    N = L.mk_tokenize(h, text, len(text), buf, len(text) + 16)
    assert N > 0, err(L)
    toks = list(buf[:N])

    def feed(t, cks=()):
        arr = (ctypes.c_int * len(t))(*t)
        ca = (ctypes.c_int * max(len(cks), 1))(*cks)
        first = L.mk_prefill_cks(h, arr, len(t), ca, len(cks), -1)
        assert first >= 0, err(L)
        return first

    def digest():
        n = L.mk_state_digest(h, None, 0)
        assert n > 0, err(L)
        a = (ctypes.c_ulonglong * n)()
        L.mk_state_digest(h, a, n)
        return list(a)

    def decode(first, n=64):
        out, got = (ctypes.c_int * 16)(), [first]
        for _ in range(n):
            if len(got) >= n:
                break
            k = L.mk_step(h, out, 16)
            assert k >= 0, err(L)
            got += list(out[:k])
        return got[:n]

    def checkpoints():
        n = L.mk_checkpoints(h, None, 0)
        a = (ctypes.c_int * max(n, 1))()
        L.mk_checkpoints(h, a, n)
        return list(a[:n])

    ok = check(N > 4096, f"the prompt ({N} tokens) is past two prefill chunks")
    first = feed(toks, (1024, 3072))
    fresh, ids0 = digest(), decode(first)
    for P in (3072, 1024):  # (the later one first: a rewind keeps the checkpoints before it)
        if not check(P in checkpoints() and L.mk_rewind_to(h, P) == 0, f"back to the checkpoint at {P}", L):
            L.mk_free(h)
            return False
        first = feed(toks[P:])
        ok &= check(digest() == fresh, f"a resume at {P} and the rest again: state bitwise equal to the fresh prefill")
        ok &= check(decode(first) == ids0, "and the next 64 greedy ids equal")
    L.mk_free(h)
    return ok


def win_commit_free_gb():
    """Windows commit free (GB) from WSL / Windows, None where it cannot be read (native Linux)"""
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                            "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1)"],
                           capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def commit_gate(c):
    """the devoom cases may go up to OVER past the VRAM: on Windows / WSL2 they start only with CAPI_COMMIT_MIN_GB (default
    45) of Windows commit free, waiting up to 120 s for it (an earlier case's teardown). A reason to SKIP, or None."""
    need = float(os.environ.get("CAPI_COMMIT_MIN_GB", "45"))
    if not c.endswith("devoom") or need <= 0 or os.environ.get("CAPI_DEVOOM", "1") == "0":
        return None
    for _ in range(25):
        g = win_commit_free_gb()
        if g is None or g >= need:
            return None
        subprocess.run(["sleep", "5"])
    return f"commit guard: Windows commit free {g} GB < {need} GB (CAPI_COMMIT_MIN_GB) after 120 s"


if __name__ == "__main__":
    if sys.argv[4:5] == ["--child"]:  # (the driver's own form: one case in this process)
        r = globals()["case_" + sys.argv[5]](lib())
        sys.stdout.flush()
        os._exit(SKIP_RC if r is None else 0 if r is True else 1)  # (no interpreter teardown after the engine is gone)
    todo = list(dict.fromkeys(sys.argv[4:])) or list(CASES)
    unknown = [c for c in todo if c not in CASES]
    if unknown:
        print(f"capi_checks: unknown case {' '.join(unknown)} (cases: {' '.join(CASES)})", file=sys.stderr, flush=True)
        sys.exit(2)
    verdict = {}
    for c in todo:
        print(f"== {c}", flush=True)
        why = commit_gate(c)
        if why:
            print("SKIP " + why, flush=True)
            verdict[c] = "SKIP"
            print(f"-- {c}: SKIP (not started)", flush=True)
            continue
        r = subprocess.run([sys.executable, __file__, LIB, MODEL, DV, "--child", c], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        lines = r.stdout.splitlines()
        print("\n".join(ln for ln in lines if ln.startswith(("PASS", "FAIL", "SKIP", "     ", "Traceback", "[mk] error"))
                        or "Error" in ln or "exiting with code" in ln or "out of memory" in ln))
        fails = sum(ln.startswith("FAIL") for ln in lines)
        if r.returncode == SKIP_RC and not fails:
            verdict[c] = "SKIP"
        elif r.returncode == 0 and not fails:
            verdict[c] = "PASS"
        else:
            verdict[c] = "FAIL"
        sig = f", killed by signal {-r.returncode}" if r.returncode < 0 else ""
        print(f"-- {c}: {verdict[c]} (exit {r.returncode}{sig})", flush=True)
    failed = [c for c in todo if verdict[c] == "FAIL"]
    skipped = [c for c in todo if verdict[c] == "SKIP"]
    passed = [c for c in todo if verdict[c] == "PASS"]
    print(f"CAPI_CHECKS_DONE: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped of {len(todo)}"
          + (f"; FAILED: {' '.join(failed)}" if failed else "") + (f"; SKIPPED: {' '.join(skipped)}" if skipped else ""))
    sys.exit(1 if failed else 0)

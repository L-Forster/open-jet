"""Time each per-token kernel (libinkling_test.so) and the expert/matvec kernels at Inkling 2 Mini sizes."""
import ctypes, sys, time
from pathlib import Path
import torch
ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "tests"))
import test_ik_kernels as T  # noqa: E402  (structs, library)
L = T.L
P, fl = T.P, ctypes.c_float
E, H, HK, D = 2304, 24, 8, 128
dev = "cuda"

def timeit(name, fn, n=300, per_token=1):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize()
    us = a.elapsed_time(b) / n * 1e3
    print(f"{name:28s} {us:8.1f} us   x{per_token:3d} per token = {us * per_token / 1e3:6.2f} ms")
    return us * per_token

tot = 0
x = torch.randn(E, device=dev); w = torch.randn(E, device=dev); out = torch.empty(4 * E, device=dev)
states = [torch.randn(E, device=dev) for _ in range(5)]
ptr3 = (ctypes.c_void_p * 3)(*[s.data_ptr() for s in states[:3]])
ptr5 = (ctypes.c_void_p * 5)(*[s.data_ptr() for s in states])
q3 = torch.randn(4, E, device=dev)
tot += timeit("mix 3 states, 1 query", lambda: L.ikt_mix(ptr3, 3, P(q3), 1, P(out), E), per_token=48 * 2)
tot += timeit("mix 5 states, 4 queries", lambda: L.ikt_mix(ptr5, 5, P(q3), 4, P(out), E), per_token=24)
down = torch.randn(64, 2 * E, device=dev); gain = torch.randn(E, 64, device=dev); forget = torch.randn(E, 64, device=dev)
tot += timeit("control (1)", lambda: L.ikt_control(P(x), P(w), P(down), P(gain), P(forget), P(out), E, 64), per_token=24 * 2)
tot += timeit("rmsnorm", lambda: L.ikt_rmsnorm(P(x), P(w), P(out), E, fl(1e-6)), per_token=48 * 2)
W24 = torch.randn(24, E, device=dev); W128 = torch.randn(128, E, device=dev)
tot += timeit("matvec_f32 24 rows (sinks)", lambda: L.ikt_matvec_f32(P(W24), P(x), P(out), 24, E), per_token=48)
ids = torch.empty(8, dtype=torch.int32, device=dev); wts = torch.empty(8, device=dev); b = torch.randn(128, device=dev)
tot += timeit("router", lambda: L.ikt_router(P(W128), P(b), P(x), P(ids), P(wts), 128, E, 8, fl(8.0)), per_token=48)
hist = torch.randn(3, E, device=dev); cw = torch.randn(E, 4, device=dev)
tot += timeit("conv_residual", lambda: L.ikt_conv_residual(P(x), P(hist), P(cw), P(w), P(out), E), per_token=96)
print(f"total of these: {tot / 1e3:.2f} ms per token")

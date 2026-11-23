"""Achieved memory bandwidth of the expert kernels: 8 random experts per call, as in decode."""
import ctypes, sys, time
from pathlib import Path
import numpy as np, torch
ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE.parent / "reference"))
from inkling_ref import InklingRef
GGUF = sys.argv[1] if len(sys.argv) > 1 else "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf"
L = ctypes.CDLL(str(ENGINE / "libinkling_kernels.so"))
vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
L.ik_experts_gate_up.argtypes = [ip, vp, vp, vp, ip, vp, vp, ip, ip, fl, fl, vp]
L.ik_experts_down.argtypes = [ip, vp, vp, vp, vp, ip, vp, vp, vp, ip, ip, vp]
sys.path.insert(0, str(ENGINE / "tests"))
L.ik_matrix_bytes.argtypes = [ip, ip, ip]; L.ik_matrix_bytes.restype = ctypes.c_size_t
L.ik_repack.argtypes = [ip, vp, vp, ip, ip, vp]
def repacked(raw, typ, rows, n):
    flat = raw.reshape(raw.shape[0], -1); bufs = []
    for e in range(raw.shape[0]):
        dst = torch.empty(L.ik_matrix_bytes(typ, rows, n), dtype=torch.uint8, device="cuda")
        L.ik_repack(typ, ctypes.c_void_p(flat[e].data_ptr()), ctypes.c_void_p(dst.data_ptr()), rows, n, None); bufs.append(dst)
    return bufs, torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device="cuda")
ref = InklingRef(GGUF)
E, F = ref.n_embd, ref.n_ff
P = lambda t: ctypes.c_void_p(t.data_ptr() if t is not None else 0)
for il in (6, 21):
    def w(n):
        t = ref.t[f"blk.{il}.{n}.weight"]; s = f"blk.{il}.{n}.scale"
        return (torch.from_numpy(np.ascontiguousarray(t.data)).cuda(),
                torch.from_numpy(np.ascontiguousarray(ref.t[s].data)).cuda() if s in ref.t else None,
                0 if t.tensor_type.name == "NVFP4" else 1)
    gu, gus, typ = w("ffn_gate_up_exps"); dn, dns, _ = w("ffn_down_exps")
    per_expert = (gu.numel() + dn.numel()) / 128
    gu_bytes, dn_bytes = gu.numel() / 128, dn.numel() / 128
    gub, gu = repacked(gu, typ, 2 * F, E); dnb, dn = repacked(dn, typ, E, F)
    x = torch.randn(E, device="cuda"); h = torch.empty(8, F, device="cuda"); y = torch.empty(E, device="cuda"); part = torch.empty(8, E, device="cuda")
    wts = torch.rand(8, device="cuda")
    idsets = [torch.randperm(128, device="cuda")[:8].to(torch.int32) for _ in range(64)]
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 400
        for i in range(n):
            ids = idsets[i % 64]
            L.ik_experts_gate_up(typ, P(gu), P(gus), P(ids), 8, P(x), P(h), E, F, 4.0, 25.0, None)
            L.ik_experts_down(typ, P(dn), P(dns), P(ids), P(wts), 8, P(h), P(part), P(y), E, F, None)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    ev[0].record()
    for i in range(200):
        L.ik_experts_gate_up(typ, P(gu), P(gus), P(idsets[i % 64]), 8, P(x), P(h), E, F, 4.0, 25.0, None)
    ev[1].record()
    for i in range(200):
        L.ik_experts_down(typ, P(dn), P(dns), P(idsets[i % 64]), P(wts), 8, P(h), P(part), P(y), E, F, None)
    ev[2].record(); torch.cuda.synchronize()
    tg, td = ev[0].elapsed_time(ev[1]) / 200e3, ev[1].elapsed_time(ev[2]) / 200e3
    gb, db = 8 * gu_bytes, 8 * dn_bytes
    print(f"  gate_up {tg*1e6:.1f} us {gb/tg/1e9:.0f} GB/s | down {td*1e6:.1f} us {db/td/1e9:.0f} GB/s")
    print(f"layer {il} ({'NVFP4' if typ == 0 else 'Q8_0'}): {dt*1e6:.1f} us per 8 experts, "
          f"{8*per_expert/1e6:.1f} MB, {8*per_expert/dt/1e9:.0f} GB/s")
    del gu, dn

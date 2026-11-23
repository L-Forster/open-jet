"""CPU expert kernels against the reference dequantisation, on real Inkling experts read from the mmap'd GGUF."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE.parent / "reference"))
_DEFAULT_GGUFS = (
    "/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf",
    "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf",
)
GGUF = os.environ.get("INKLING_GGUF") or os.environ.get("IK_GGUF") or next(
    (p for p in _DEFAULT_GGUFS if Path(p).exists()), _DEFAULT_GGUFS[0]
)
if not Path(GGUF).exists() or not (ENGINE / "libinkling_cpu.so").exists():
    pytest.skip("needs the GGUF and libinkling_cpu.so", allow_module_level=True)
from inkling_ref import InklingRef  # noqa: E402

L = ctypes.CDLL(str(ENGINE / "libinkling_cpu.so"))
vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
L.ik_cpu_gate_up.argtypes = [ip, vp, fl, vp, vp, ip, ip, fl, fl, ip, ip]
L.ik_cpu_down.argtypes = [ip, vp, fl, vp, vp, ip, ip, ip, ip]
L.ik_cpu_gate_up_batch.argtypes = [ip, vp, fl, vp, vp, ip, ip, ip, fl, fl, ip, ip]
L.ik_cpu_down_batch.argtypes = [ip, vp, vp, vp, vp, ip, ip, ip, ip, ip]


@pytest.fixture(scope="module")
def ref():
    return InklingRef(GGUF, device="cpu")


@pytest.mark.parametrize("il,e", [(6, 5), (6, 99), (21, 40)])
def test_cpu_expert(ref, il, e):
    E, F = ref.n_embd, ref.n_ff
    gu_t, dn_t = ref.t[f"blk.{il}.ffn_gate_up_exps.weight"], ref.t[f"blk.{il}.ffn_down_exps.weight"]
    gu_typ, dn_typ = int(gu_t.tensor_type), int(dn_t.tensor_type)
    gu = np.ascontiguousarray(gu_t.data[e]); dn = np.ascontiguousarray(dn_t.data[e])
    sg = float(ref.t[f"blk.{il}.ffn_gate_up_exps.scale"].data[e]) if f"blk.{il}.ffn_gate_up_exps.scale" in ref.t else 1.0
    sd = float(ref.t[f"blk.{il}.ffn_down_exps.scale"].data[e]) if f"blk.{il}.ffn_down_exps.scale" in ref.t else 1.0
    rng = np.random.default_rng(il * 1000 + e)
    x = rng.standard_normal(E).astype(np.float32)
    h = np.zeros(F, np.float32); y = np.zeros(E, np.float32)
    t0 = time.perf_counter()
    L.ik_cpu_gate_up(gu_typ, gu.ctypes.data, sg, x.ctypes.data, h.ctypes.data, E, F, ref.situ_beta, ref.situ_lb, 0, F)
    L.ik_cpu_down(dn_typ, dn.ctypes.data, 0.7 * sd, h.ctypes.data, y.ctypes.data, E, F, 0, E)
    dt = time.perf_counter() - t0
    G = ref.experts(f"blk.{il}.ffn_gate_up_exps.weight", [e])[0].double()
    Dn = ref.experts(f"blk.{il}.ffn_down_exps.weight", [e])[0].double()
    hh = G @ torch.from_numpy(x).double()
    g, u = hh[:F], hh[F:]
    h_ref = ref.situ_beta * torch.tanh(g / ref.situ_beta) * torch.sigmoid(g) * (ref.situ_lb * torch.tanh(u / ref.situ_lb))
    y_ref = 0.7 * (Dn @ h_ref)
    rh = float(np.linalg.norm(h - h_ref.numpy()) / np.linalg.norm(h_ref.numpy()))
    ry = float(np.linalg.norm(y - y_ref.numpy()) / np.linalg.norm(y_ref.numpy()))
    mb = (gu.nbytes + dn.nbytes) / 1e6
    print(f"layer {il} expert {e}: one thread {dt*1e3:.2f} ms for {mb:.1f} MB = {mb/dt/1e3:.1f} GB/s; rel err h {rh:.1e} y {ry:.1e}")
    assert rh < 1e-5 and ry < 1e-5


@pytest.mark.parametrize("il,e,ntok", [(6, 5, 4), (6, 5, 8), (21, 40, 4)])
def test_cpu_expert_batch(ref, il, e, ntok):
    E, F = ref.n_embd, ref.n_ff
    gu_t, dn_t = ref.t[f"blk.{il}.ffn_gate_up_exps.weight"], ref.t[f"blk.{il}.ffn_down_exps.weight"]
    gu_typ, dn_typ = int(gu_t.tensor_type), int(dn_t.tensor_type)
    gu = np.ascontiguousarray(gu_t.data[e]); dn = np.ascontiguousarray(dn_t.data[e])
    sg = float(ref.t[f"blk.{il}.ffn_gate_up_exps.scale"].data[e]) if f"blk.{il}.ffn_gate_up_exps.scale" in ref.t else 1.0
    sd = float(ref.t[f"blk.{il}.ffn_down_exps.scale"].data[e]) if f"blk.{il}.ffn_down_exps.scale" in ref.t else 1.0
    rng = np.random.default_rng(il * 1000 + e + ntok)
    xs = [np.ascontiguousarray(rng.standard_normal(E).astype(np.float32)) for _ in range(ntok)]
    hs = [np.zeros(F, np.float32) for _ in range(ntok)]
    ys = [np.zeros(E, np.float32) for _ in range(ntok)]
    cs = np.array([0.3 + 0.1 * t for t in range(ntok)], dtype=np.float32) * sd
    x_ptrs = (vp * ntok)(*[x.ctypes.data for x in xs])
    h_ptrs = (vp * ntok)(*[h.ctypes.data for h in hs])
    y_ptrs = (vp * ntok)(*[y.ctypes.data for y in ys])
    L.ik_cpu_gate_up_batch(gu_typ, gu.ctypes.data, sg, x_ptrs, h_ptrs, ntok, E, F, ref.situ_beta, ref.situ_lb, 0, F)
    L.ik_cpu_down_batch(dn_typ, dn.ctypes.data, cs.ctypes.data, h_ptrs, y_ptrs, ntok, E, F, 0, E)
    for t in range(ntok):
        h1 = np.zeros(F, np.float32); y1 = np.zeros(E, np.float32)
        L.ik_cpu_gate_up(gu_typ, gu.ctypes.data, sg, xs[t].ctypes.data, h1.ctypes.data, E, F, ref.situ_beta, ref.situ_lb, 0, F)
        L.ik_cpu_down(dn_typ, dn.ctypes.data, float(cs[t]), h1.ctypes.data, y1.ctypes.data, E, F, 0, E)
        rh = float(np.linalg.norm(hs[t] - h1) / (np.linalg.norm(h1) + 1e-12))
        ry = float(np.linalg.norm(ys[t] - y1) / (np.linalg.norm(y1) + 1e-12))
        assert rh < 1e-6 and ry < 1e-6

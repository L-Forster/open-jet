"""Expert kernels against the reference dequantisation, on real Inkling weights.

    INKLING_GGUF=/path/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf python -m pytest inkling/engine/tests -q
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE.parent / "reference"))
from inkling_ref import InklingRef  # noqa: E402

GGUF = os.environ.get("INKLING_GGUF", "/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf")
if not Path(GGUF).exists():
    pytest.skip("set INKLING_GGUF to a mainline-layout Inkling 2 Mini GGUF", allow_module_level=True)


@pytest.fixture(scope="module")
def lib():
    so = ENGINE / "libinkling_kernels.so"
    if not so.exists():
        pytest.skip("build inkling/engine first (make)")
    L = ctypes.CDLL(str(so))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_experts_gate_up.argtypes = [ip, vp, vp, vp, ip, vp, vp, ip, ip, fl, fl, vp]
    L.ik_experts_down.argtypes = [ip, vp, vp, vp, vp, ip, vp, vp, vp, ip, ip, vp]
    L.ik_matrix_bytes.argtypes = [ip, ip, ip]
    L.ik_matrix_bytes.restype = ctypes.c_size_t
    L.ik_repack.argtypes = [ip, vp, vp, ip, ip, vp]
    return L


def repacked(lib, raw: "torch.Tensor", typ: int, rows: int, n: int):
    """Repack every expert of a GGUF expert tensor into its own buffer; returns (buffers, device pointer array)."""
    n_expert = raw.shape[0]
    flat = raw.reshape(n_expert, -1)
    bufs = []
    for e in range(n_expert):
        dst = torch.empty(lib.ik_matrix_bytes(typ, rows, n), dtype=torch.uint8, device="cuda")
        assert lib.ik_repack(typ, ctypes.c_void_p(flat[e].data_ptr()), ctypes.c_void_p(dst.data_ptr()), rows, n, None) == 0
        bufs.append(dst)
    ptrs = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device="cuda")
    return bufs, ptrs


@pytest.fixture(scope="module")
def ref():
    return InklingRef(GGUF)


def layer_weights(ref, il, name):
    t = ref.t[f"blk.{il}.{name}.weight"]
    raw = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    sname = f"blk.{il}.{name}.scale"
    scale = torch.from_numpy(np.ascontiguousarray(ref.t[sname].data)).cuda() if sname in ref.t else None
    return raw, scale, 0 if t.tensor_type.name == "NVFP4" else 1


@pytest.mark.parametrize("il", [6, 21])  # NVFP4 experts, Q8_0 experts
def test_experts_match_reference(lib, ref, il):
    torch.manual_seed(il)
    E, F = ref.n_embd, ref.n_ff
    gu_raw, gu_scale, typ = layer_weights(ref, il, "ffn_gate_up_exps")
    dn_raw, dn_scale, typ2 = layer_weights(ref, il, "ffn_down_exps")
    assert typ == typ2
    ids = torch.tensor([3, 17, 42, 64, 77, 90, 101, 127], dtype=torch.int32, device="cuda")
    wts = torch.rand(8, device="cuda")
    x = torch.randn(E, device="cuda")
    h = torch.empty(8, F, device="cuda")
    y = torch.empty(E, device="cuda")
    part = torch.empty(8, E, device="cuda")
    ptr = lambda t: ctypes.c_void_p(t.data_ptr() if t is not None else 0)  # noqa: E731
    gu_bufs, gu_ptrs = repacked(lib, gu_raw, typ, 2 * F, E)
    dn_bufs, dn_ptrs = repacked(lib, dn_raw, typ, E, F)
    assert lib.ik_experts_gate_up(typ, ptr(gu_ptrs), ptr(gu_scale), ptr(ids), 8, ptr(x), ptr(h), E, F,
                                  ref.situ_beta, ref.situ_lb, None) == 0
    assert lib.ik_experts_down(typ, ptr(dn_ptrs), ptr(dn_scale), ptr(ids), ptr(wts), 8, ptr(h), ptr(part), ptr(y), E, F, None) == 0
    torch.cuda.synchronize()

    sel = ids.tolist()
    gu = ref.experts(f"blk.{il}.ffn_gate_up_exps.weight", sel)
    dn = ref.experts(f"blk.{il}.ffn_down_exps.weight", sel)
    hh = torch.einsum("kre,e->kr", gu, x)
    g, u = hh[:, :F], hh[:, F:]
    h_ref = ref.situ_beta * torch.tanh(g / ref.situ_beta) * torch.sigmoid(g) * (ref.situ_lb * torch.tanh(u / ref.situ_lb))
    y_ref = torch.einsum("k,kre,ke->r", wts, dn, h_ref)

    rel_h = ((h - h_ref).norm() / h_ref.norm()).item()
    rel_y = ((y - y_ref).norm() / y_ref.norm()).item()
    assert rel_h < 1e-5, rel_h
    assert rel_y < 1e-5, rel_y

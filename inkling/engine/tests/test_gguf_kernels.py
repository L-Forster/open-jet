"""GGUF-layout kernels (gguf_kernels.cu) and the CPU's generic path against gguf-py's dequantisation, for every ggml
type in an Inkling GGUF that the fast paths don't cover (UD-IQ4_XS: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q5_0).

    INKLING_IQ4_GGUF=/path/Inkling-2-Mini-UD-IQ4_XS.gguf python -m pytest inkling/engine/tests/test_gguf_kernels.py -q
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
GGUF = os.environ.get("INKLING_IQ4_GGUF", "/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf")
if not Path(GGUF).exists():
    pytest.skip("set INKLING_IQ4_GGUF to a mainline-layout Inkling 2 Mini UD-IQ4_XS GGUF", allow_module_level=True)
from inkling_ref import InklingRef  # noqa: E402

FAST = {"NVFP4", "Q8_0", "F32", "BF16"}


@pytest.fixture(scope="module")
def lib():
    L = ctypes.CDLL(str(ENGINE / "libinkling_kernels.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_matvec_gguf.argtypes = [ip, vp, vp, vp, ip, ip, vp]
    L.ik_experts_gate_up_gguf.argtypes = [ip, vp, vp, vp, ip, vp, vp, ip, ip, fl, fl, vp]
    L.ik_experts_down_gguf.argtypes = [ip, vp, vp, vp, vp, ip, vp, vp, vp, ip, ip, vp]
    return L


@pytest.fixture(scope="module")
def cpu():
    L = ctypes.CDLL(str(ENGINE / "libinkling_cpu.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_cpu_gate_up.argtypes = [ip, vp, fl, vp, vp, ip, ip, fl, fl, ip, ip]
    L.ik_cpu_down.argtypes = [ip, vp, fl, vp, vp, ip, ip, ip, ip]
    return L


@pytest.fixture(scope="module")
def ref():
    return InklingRef(GGUF)


def P(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def dense_cases(ref):
    seen = {}
    for t in ref.r.tensors:
        if len(t.shape) == 2 and "_exps" not in t.name and t.tensor_type.name not in FAST and t.name != "token_embd.weight":
            seen.setdefault(t.tensor_type.name, t.name)
    return sorted(seen.items())


def test_dense_matvec(lib, ref):
    cases = dense_cases(ref)
    assert cases, "no dense tensors of other types in this file"
    for typ, name in cases:
        t = ref.t[name]
        n, rows = int(t.shape[0]), int(t.shape[1])
        if rows > 50000:  # the output head: a slice of its rows
            raw = np.ascontiguousarray(t.data[:4096]); rows = 4096
        else:
            raw = np.ascontiguousarray(t.data)
        W = ref._deq_gguf(raw, t.tensor_type).reshape(rows, n)
        x = torch.randn(n, device="cuda")
        y = torch.empty(rows, device="cuda")
        w = torch.from_numpy(raw).cuda()
        assert lib.ik_matvec_gguf(int(t.tensor_type), P(w), P(x), P(y), rows, n, None) == 0
        torch.cuda.synchronize()
        err = ((y - W @ x).norm() / (W @ x).norm()).item()
        print(f"{typ} {name}: rel err {err:.1e}")
        assert err < 1e-5, (typ, name, err)


def expert_layers(ref):
    out = []
    for il in range(ref.n_layer):
        gu, dn = ref.t[f"blk.{il}.ffn_gate_up_exps.weight"], ref.t[f"blk.{il}.ffn_down_exps.weight"]
        if gu.tensor_type.name not in FAST or dn.tensor_type.name not in FAST:
            out.append(il)
    return out


def test_experts(lib, cpu, ref):
    layers = expert_layers(ref)
    assert layers, "no experts of other types in this file"
    E, F = ref.n_embd, ref.n_ff
    for il in layers:
        gu_t, dn_t = ref.t[f"blk.{il}.ffn_gate_up_exps.weight"], ref.t[f"blk.{il}.ffn_down_exps.weight"]
        ids_l = [1, 20, 33, 47, 64, 80, 100, 126]
        gu_raw = torch.from_numpy(np.ascontiguousarray(gu_t.data)).cuda()
        dn_raw = torch.from_numpy(np.ascontiguousarray(dn_t.data)).cuda()
        gu_ptrs = torch.tensor([gu_raw[e].data_ptr() for e in range(ref.n_expert)], dtype=torch.int64, device="cuda")
        dn_ptrs = torch.tensor([dn_raw[e].data_ptr() for e in range(ref.n_expert)], dtype=torch.int64, device="cuda")
        sg = f"blk.{il}.ffn_gate_up_exps.scale"
        sd = f"blk.{il}.ffn_down_exps.scale"
        gu_s = torch.from_numpy(np.array(ref.t[sg].data)).cuda() if sg in ref.t else None
        dn_s = torch.from_numpy(np.array(ref.t[sd].data)).cuda() if sd in ref.t else None
        ids = torch.tensor(ids_l, dtype=torch.int32, device="cuda")
        wts = torch.rand(8, device="cuda")
        x = torch.randn(E, device="cuda")
        h = torch.empty(8, F, device="cuda"); part = torch.empty(8, E, device="cuda"); y = torch.empty(E, device="cuda")
        assert lib.ik_experts_gate_up_gguf(int(gu_t.tensor_type), P(gu_ptrs), P(gu_s), P(ids), 8, P(x), P(h), E, F,
                                           ref.situ_beta, ref.situ_lb, None) == 0
        assert lib.ik_experts_down_gguf(int(dn_t.tensor_type), P(dn_ptrs), P(dn_s), P(ids), P(wts), 8, P(h), P(part), P(y),
                                        E, F, None) == 0
        torch.cuda.synchronize()
        G = ref.experts(f"blk.{il}.ffn_gate_up_exps.weight", ids_l)
        Dn = ref.experts(f"blk.{il}.ffn_down_exps.weight", ids_l)
        hh = torch.einsum("kre,e->kr", G, x)
        g, u = hh[:, :F], hh[:, F:]
        h_ref = ref.situ_beta * torch.tanh(g / ref.situ_beta) * torch.sigmoid(g) * (ref.situ_lb * torch.tanh(u / ref.situ_lb))
        y_ref = torch.einsum("k,kre,ke->r", wts, Dn, h_ref)
        eh = ((h - h_ref).norm() / h_ref.norm()).item()
        ey = ((y - y_ref).norm() / y_ref.norm()).item()
        # the CPU's generic path, one expert
        e0 = ids_l[0]
        xc = x.cpu().numpy(); hc = np.zeros(F, np.float32); yc = np.zeros(E, np.float32)
        gub = np.ascontiguousarray(gu_t.data[e0]); dnb = np.ascontiguousarray(dn_t.data[e0])
        cpu.ik_cpu_gate_up(int(gu_t.tensor_type), gub.ctypes.data, float(gu_s[e0]) if gu_s is not None else 1.0, xc.ctypes.data,
                           hc.ctypes.data, E, F, ref.situ_beta, ref.situ_lb, 0, F)
        cpu.ik_cpu_down(int(dn_t.tensor_type), dnb.ctypes.data, float(dn_s[e0]) if dn_s is not None else 1.0, hc.ctypes.data,
                        yc.ctypes.data, E, F, 0, E)
        yc_ref = (Dn[0] @ h_ref[0]).cpu().numpy()
        ec = float(np.linalg.norm(yc - yc_ref) / np.linalg.norm(yc_ref))
        print(f"layer {il} ({gu_t.tensor_type.name}/{dn_t.tensor_type.name}): GPU rel err h {eh:.1e} y {ey:.1e}, CPU {ec:.1e}")
        assert eh < 1e-5 and ey < 1e-5 and ec < 1e-5, (il, eh, ey, ec)

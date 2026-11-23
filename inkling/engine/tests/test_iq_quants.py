"""Bit-accuracy tests for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, GGUF NVFP4/Q8_0, and Q4_K/Q5_K/Q6_K/Q8_0 embed.

Verifies every GPU kernel (libinkling_kernels.so, libinkling_test.so) and CPU AVX2 kernel (libinkling_cpu.so)
against llama.cpp's official dequantize_row_* implementations in inkling/engine/lib/libggml-base.so on synthetic
random blocks (no 11.6 GB model download required).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

ENGINE = Path(__file__).resolve().parents[1]
GGML_LIB = ENGINE / "lib" / "libggml-base.so"
if not GGML_LIB.exists() or not (ENGINE / "libinkling_kernels.so").exists():
    pytest.skip("needs built libinkling_kernels.so and lib/libggml-base.so", allow_module_level=True)

# (name, ggml_type_id, block_values, block_bytes, dequant_symbol, fp16_offsets)
QUANT_SPECS = [
    ("IQ2_XXS", 16, 256, 66, "dequantize_row_iq2_xxs", (0,)),
    ("IQ2_XS", 17, 256, 74, "dequantize_row_iq2_xs", (0,)),
    ("IQ3_XXS", 18, 256, 98, "dequantize_row_iq3_xxs", (0,)),
    ("IQ2_S", 22, 256, 82, "dequantize_row_iq2_s", (0,)),
    ("NVFP4", 40, 64, 36, "dequantize_row_nvfp4", ()),
    ("Q8_0", 8, 32, 34, "dequantize_row_q8_0", (0,)),
]

EMBED_SPECS = [
    ("Q8_0", 8, 32, 34, "dequantize_row_q8_0", (0,)),
    ("Q4_K", 12, 256, 144, "dequantize_row_q4_K", (0, 2)),
    ("Q5_K", 13, 256, 176, "dequantize_row_q5_K", (0, 2)),
    ("Q6_K", 14, 256, 210, "dequantize_row_q6_K", (208,)),
]


@pytest.fixture(scope="module")
def ggml():
    L = ctypes.CDLL(str(GGML_LIB))
    for _, _, _, _, sym, _ in QUANT_SPECS + EMBED_SPECS:
        fn = getattr(L, sym)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
    return L


@pytest.fixture(scope="module")
def kernels():
    L = ctypes.CDLL(str(ENGINE / "libinkling_kernels.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_gguf_type_supported.argtypes = [ip]
    L.ik_gguf_type_supported.restype = ip
    L.ik_gguf_row_bytes.argtypes = [ip, ip]
    L.ik_gguf_row_bytes.restype = ctypes.c_size_t
    L.ik_matvec_gguf.argtypes = [ip, vp, vp, vp, ip, ip, vp]
    L.ik_pf_matvec_gguf.argtypes = [ip, vp, vp, vp, ip, ip, ip, vp]
    L.ik_experts_gate_up_gguf.argtypes = [ip, vp, vp, vp, ip, vp, vp, ip, ip, fl, fl, vp]
    L.ik_experts_down_gguf.argtypes = [ip, vp, vp, vp, vp, ip, vp, vp, vp, ip, ip, vp]
    L.ik_pf_experts_gguf.argtypes = [ip, ip, vp, vp, vp, vp, vp, vp, ip, vp, vp, vp, vp, ip, ip, ip, fl, fl, vp]
    return L


@pytest.fixture(scope="module")
def cpu():
    L = ctypes.CDLL(str(ENGINE / "libinkling_cpu.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_cpu_gate_up.argtypes = [ip, vp, fl, vp, vp, ip, ip, fl, fl, ip, ip]
    L.ik_cpu_down.argtypes = [ip, vp, fl, vp, vp, ip, ip, ip, ip]
    L.ik_cpu_gate_up_batch.argtypes = [ip, vp, fl, vp, vp, ip, ip, ip, fl, fl, ip, ip]
    L.ik_cpu_down_batch.argtypes = [ip, vp, vp, vp, vp, ip, ip, ip, ip, ip]
    return L


@pytest.fixture(scope="module")
def test_api():
    L = ctypes.CDLL(str(ENGINE / "libinkling_test.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ikt_embed_type.argtypes = [ip, vp, vp, vp, ip, fl]
    return L


def P(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def make_random_blocks(rng: np.random.Generator, rows: int, n: int, bvals: int, bbytes: int, fp16_offsets: tuple[int, ...], gtype: int) -> np.ndarray:
    nblk = (rows * n) // bvals
    raw = rng.integers(0, 256, size=(nblk, bbytes), dtype=np.uint8)
    for off in fp16_offsets:
        scales = rng.uniform(-0.5, 0.5, size=nblk).astype(np.float16).view(np.uint8).reshape(nblk, 2)
        raw[:, off : off + 2] = scales
    if gtype == 40:  # NVFP4: valid UE4M3 scales (0..126; avoid 0x7F/0xFF NaN sentinel)
        raw[:, :4] = rng.integers(8, 120, size=(nblk, 4), dtype=np.uint8)
    return np.ascontiguousarray(raw.reshape(rows, (n // bvals) * bbytes))


def dequant_ggml(ggml, sym: str, raw: np.ndarray, rows: int, n: int) -> np.ndarray:
    out = np.empty((rows, n), dtype=np.float32)
    fn = getattr(ggml, sym)
    rb = raw.shape[-1]
    flat = np.ascontiguousarray(raw).reshape(rows, rb)
    for r in range(rows):
        fn(flat[r].ctypes.data, out[r].ctypes.data, ctypes.c_int64(n))
    return out


def situ_np(g: np.ndarray, u: np.ndarray, beta: float, lb: float) -> np.ndarray:
    act = beta * np.tanh(g / beta) / (1.0 + np.exp(-g))
    if lb > 0.0:
        u = lb * np.tanh(u / lb)
    return act * u


@pytest.mark.parametrize("name,gtype,bvals,bbytes,sym,fp16_offs", QUANT_SPECS)
def test_gpu_matvec_and_prefill(ggml, kernels, name, gtype, bvals, bbytes, sym, fp16_offs):
    assert kernels.ik_gguf_type_supported(gtype) == 1
    rows, n = 128, 1024
    assert kernels.ik_gguf_row_bytes(gtype, n) == (n // bvals) * bbytes
    rng = np.random.default_rng(100 + gtype)
    raw = make_random_blocks(rng, rows, n, bvals, bbytes, fp16_offs, gtype)
    W = dequant_ggml(ggml, sym, raw, rows, n)
    W_t = torch.from_numpy(W).cuda()
    w_d = torch.from_numpy(raw).cuda()

    # T = 1 (ik_matvec_gguf)
    x1 = torch.randn(n, device="cuda")
    y1 = torch.empty(rows, device="cuda")
    assert kernels.ik_matvec_gguf(gtype, P(w_d), P(x1), P(y1), rows, n, None) == 0
    torch.cuda.synchronize()
    ref1 = W_t @ x1
    err1 = ((y1 - ref1).norm() / ref1.norm()).item()
    assert err1 < 1e-5, (name, "T=1", err1)

    # T = 4 and T = 7 (ik_pf_matvec_gguf)
    for T in (4, 7):
        X = torch.randn(T, n, device="cuda")
        Y = torch.empty(T, rows, device="cuda")
        assert kernels.ik_pf_matvec_gguf(gtype, P(w_d), P(X), P(Y), rows, n, T, None) == 0
        torch.cuda.synchronize()
        refT = X @ W_t.T
        errT = ((Y - refT).norm() / refT.norm()).item()
        assert errT < 1e-5, (name, f"T={T}", errT)


@pytest.mark.parametrize("gu_spec,dn_spec", [
    (QUANT_SPECS[0], QUANT_SPECS[4]),  # IQ2_XXS gate_up + NVFP4 down (blk.0 of UD-IQ3_XXS)
    (QUANT_SPECS[1], QUANT_SPECS[4]),  # IQ2_XS  gate_up + NVFP4 down (blk.4 of UD-IQ3_XXS)
    (QUANT_SPECS[2], QUANT_SPECS[4]),  # IQ3_XXS gate_up + NVFP4 down (blk.1..19 of UD-IQ3_XXS)
    (QUANT_SPECS[3], QUANT_SPECS[4]),  # IQ2_S   gate_up + NVFP4 down (core layers of UD-IQ3_XXS)
    (QUANT_SPECS[2], QUANT_SPECS[2]),  # IQ3_XXS gate_up + IQ3_XXS down
    (QUANT_SPECS[3], QUANT_SPECS[3]),  # IQ2_S   gate_up + IQ2_S down
])
def test_gpu_and_cpu_experts(ggml, kernels, cpu, gu_spec, dn_spec):
    gu_name, gu_t, gu_bv, gu_bb, gu_sym, gu_fp16 = gu_spec
    dn_name, dn_t, dn_bv, dn_bb, dn_sym, dn_fp16 = dn_spec
    n_expert, n_sel, E, F = 6, 4, 512, 256
    beta, lb = 3.0, 10.0
    rng = np.random.default_rng(200 + gu_t * 31 + dn_t)

    gu_raw = np.stack([make_random_blocks(rng, 2 * F, E, gu_bv, gu_bb, gu_fp16, gu_t) for _ in range(n_expert)])
    dn_raw = np.stack([make_random_blocks(rng, E, F, dn_bv, dn_bb, dn_fp16, dn_t) for _ in range(n_expert)])
    gu_s_np = rng.uniform(0.5, 1.5, size=n_expert).astype(np.float32)
    dn_s_np = rng.uniform(0.5, 1.5, size=n_expert).astype(np.float32)

    gu_W = np.stack([dequant_ggml(ggml, gu_sym, gu_raw[e], 2 * F, E) * gu_s_np[e] for e in range(n_expert)])
    dn_W = np.stack([dequant_ggml(ggml, dn_sym, dn_raw[e], E, F) * dn_s_np[e] for e in range(n_expert)])

    gu_d = torch.from_numpy(gu_raw).cuda()
    dn_d = torch.from_numpy(dn_raw).cuda()
    gu_ptrs = torch.tensor([gu_d[e].data_ptr() for e in range(n_expert)], dtype=torch.int64, device="cuda")
    dn_ptrs = torch.tensor([dn_d[e].data_ptr() for e in range(n_expert)], dtype=torch.int64, device="cuda")
    gu_s = torch.from_numpy(gu_s_np).cuda()
    dn_s = torch.from_numpy(dn_s_np).cuda()

    # 1. GPU single-token decode experts
    ids_l = [0, 2, 4, 5]
    ids = torch.tensor(ids_l, dtype=torch.int32, device="cuda")
    wts = torch.rand(n_sel, device="cuda")
    x = torch.randn(E, device="cuda")
    h = torch.empty(n_sel, F, device="cuda")
    part = torch.empty(n_sel, E, device="cuda")
    y = torch.empty(E, device="cuda")
    assert kernels.ik_experts_gate_up_gguf(gu_t, P(gu_ptrs), P(gu_s), P(ids), n_sel, P(x), P(h), E, F, beta, lb, None) == 0
    assert kernels.ik_experts_down_gguf(dn_t, P(dn_ptrs), P(dn_s), P(ids), P(wts), n_sel, P(h), P(part), P(y), E, F, None) == 0
    torch.cuda.synchronize()

    x_np = x.cpu().numpy()
    wts_np = wts.cpu().numpy()
    h_ref = np.zeros((n_sel, F), np.float32)
    y_ref = np.zeros(E, np.float32)
    for k, e in enumerate(ids_l):
        hh = gu_W[e] @ x_np
        h_ref[k] = situ_np(hh[:F], hh[F:], beta, lb)
        y_ref += wts_np[k] * (dn_W[e] @ h_ref[k])
    eh = float(np.linalg.norm(h.cpu().numpy() - h_ref) / np.linalg.norm(h_ref))
    ey = float(np.linalg.norm(y.cpu().numpy() - y_ref) / np.linalg.norm(y_ref))
    assert eh < 1e-5 and ey < 1e-5, (gu_name, dn_name, "GPU decode", eh, ey)

    # 2. GPU grouped prefill experts exercising cnt=1, cnt=3, cnt=6
    T = 5
    # 5 tokens x 2 slots = 10 entries: expert 0 gets 1 entry, expert 1 gets 3 entries, expert 2 gets 6 entries
    entry_experts = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    k_pf = 2
    offsets_l = [0, 1, 4, 10, 10, 10, 10]
    list_l = list(range(10))
    offsets_d = torch.tensor(offsets_l, dtype=torch.int32, device="cuda")
    list_d = torch.tensor(list_l, dtype=torch.int32, device="cuda")
    X_pf = torch.randn(T, E, device="cuda")
    wts_pf = torch.rand(T * k_pf, device="cuda")
    H_pf = torch.empty(T * k_pf, F, device="cuda")
    part_pf = torch.empty(T * k_pf, E, device="cuda")
    assert kernels.ik_pf_experts_gguf(
        gu_t, dn_t, P(gu_ptrs), P(dn_ptrs), P(gu_s), P(dn_s), P(offsets_d), P(list_d), k_pf,
        P(wts_pf), P(X_pf), P(H_pf), P(part_pf), E, F, n_expert, beta, lb, None
    ) == 0
    torch.cuda.synchronize()
    X_pf_np = X_pf.cpu().numpy()
    wts_pf_np = wts_pf.cpu().numpy()
    part_pf_np = part_pf.cpu().numpy()
    for en, e in enumerate(entry_experts):
        hh = gu_W[e] @ X_pf_np[en // k_pf]
        he = situ_np(hh[:F], hh[F:], beta, lb)
        pe_ref = wts_pf_np[en] * (dn_W[e] @ he)
        ep = float(np.linalg.norm(part_pf_np[en] - pe_ref) / np.linalg.norm(pe_ref))
        assert ep < 1e-5, (gu_name, dn_name, f"GPU pf entry {en} exp {e}", ep)

    # 3. CPU single-token and batched (n_tok=4) experts
    e0 = ids_l[0]
    hc = np.zeros(F, np.float32)
    yc = np.zeros(E, np.float32)
    cpu.ik_cpu_gate_up(gu_t, gu_raw[e0].ctypes.data, float(gu_s_np[e0]), x_np.ctypes.data, hc.ctypes.data, E, F, beta, lb, 0, F)
    cpu.ik_cpu_down(dn_t, dn_raw[e0].ctypes.data, float(wts_np[0] * dn_s_np[e0]), hc.ctypes.data, yc.ctypes.data, E, F, 0, E)
    yc_ref = wts_np[0] * (dn_W[e0] @ h_ref[0])
    ech = float(np.linalg.norm(hc - h_ref[0]) / np.linalg.norm(h_ref[0]))
    ecy = float(np.linalg.norm(yc - yc_ref) / np.linalg.norm(yc_ref))
    assert ech < 1e-5 and ecy < 1e-5, (gu_name, dn_name, "CPU single", ech, ecy)

    ntok = 4
    vp = ctypes.c_void_p
    xs = [np.ascontiguousarray(rng.standard_normal(E).astype(np.float32)) for _ in range(ntok)]
    hs = [np.zeros(F, np.float32) for _ in range(ntok)]
    ys = [np.zeros(E, np.float32) for _ in range(ntok)]
    cs = np.array([0.4 + 0.1 * t for t in range(ntok)], dtype=np.float32) * dn_s_np[e0]
    x_ptrs = (vp * ntok)(*[v.ctypes.data for v in xs])
    h_ptrs = (vp * ntok)(*[v.ctypes.data for v in hs])
    y_ptrs = (vp * ntok)(*[v.ctypes.data for v in ys])
    cpu.ik_cpu_gate_up_batch(gu_t, gu_raw[e0].ctypes.data, float(gu_s_np[e0]), x_ptrs, h_ptrs, ntok, E, F, beta, lb, 0, F)
    cpu.ik_cpu_down_batch(dn_t, dn_raw[e0].ctypes.data, cs.ctypes.data, h_ptrs, y_ptrs, ntok, E, F, 0, E)
    for t in range(ntok):
        hh = gu_W[e0] @ xs[t]
        hr = situ_np(hh[:F], hh[F:], beta, lb)
        yr = (0.4 + 0.1 * t) * (dn_W[e0] @ hr)
        assert float(np.linalg.norm(hs[t] - hr) / np.linalg.norm(hr)) < 1e-5
        assert float(np.linalg.norm(ys[t] - yr) / np.linalg.norm(yr)) < 1e-5


@pytest.mark.parametrize("name,gtype,bvals,bbytes,sym,fp16_offs", EMBED_SPECS)
def test_gpu_embed_types(ggml, test_api, name, gtype, bvals, bbytes, sym, fp16_offs):
    n = 2304
    eps = 1e-6
    rng = np.random.default_rng(300 + gtype)
    raw = make_random_blocks(rng, 1, n, bvals, bbytes, fp16_offs, gtype)[0]
    w_np = rng.standard_normal(n).astype(np.float32)
    deq = dequant_ggml(ggml, sym, raw[None, :], 1, n)[0]
    ref = deq * (1.0 / np.sqrt(np.mean(deq * deq) + eps)) * w_np

    row_d = torch.from_numpy(raw).cuda()
    w_d = torch.from_numpy(w_np).cuda()
    out_d = torch.empty(n, device="cuda")
    test_api.ikt_embed_type(gtype, P(row_d), P(w_d), P(out_d), n, eps)
    torch.cuda.synchronize()
    err = float(np.linalg.norm(out_d.cpu().numpy() - ref) / np.linalg.norm(ref))
    assert err < 1e-5, (name, err)

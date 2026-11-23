"""Each per-token kernel against the PyTorch op it replaces (reference/inkling_ref.py math), on random inputs.

    make -C inkling/engine libinkling_test.so && python -m pytest inkling/engine/tests/test_ik_kernels.py -q
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

ENGINE = Path(__file__).resolve().parents[1]
SO = ENGINE / "libinkling_test.so"
if not SO.exists():
    pytest.skip("build inkling/engine/libinkling_test.so first", allow_module_level=True)

L = ctypes.CDLL(str(SO))
vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
E, H, HK, D, NREL = 2304, 24, 8, 128, 16
dev = "cuda"


def P(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


class AttnPrepArgs(ctypes.Structure):
    _fields_ = [(n, vp) for n in ("q", "k", "v", "r", "kdyn", "vdyn", "k_conv", "v_conv", "hist_k", "hist_v", "q_clip", "rel_clip",
                                  "q_out", "r_out", "k_cache", "v_cache", "k_q", "v_q", "k_s", "v_s", "pos")] + \
               [(n, ip) for n in ("window", "global_", "n_head", "n_head_kv", "d_head", "n_rel", "ring", "kv_q8")] + \
               [(n, fl) for n in ("qk_clip", "rel_clip_thr", "eps", "log_alpha", "log_floor")]


class AttnArgs(ctypes.Structure):
    _fields_ = [(n, vp) for n in ("q", "r", "sinks", "gate", "rel_bias", "k_cache", "v_cache", "k_q", "v_q", "k_s", "v_s", "out", "pos")] + \
               [(n, ip) for n in ("window", "extent", "n_head", "n_head_kv", "ring", "kv_q8")] + [("scratch", vp)]


def test_struct_sizes():
    assert L.ikt_sizeof_attn_prep() == ctypes.sizeof(AttnPrepArgs)
    assert L.ikt_sizeof_attn() == ctypes.sizeof(AttnArgs)


def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def close(a, b, tol=1e-5):
    err = ((a - b).norm() / (b.norm() + 1e-30)).item()
    assert err < tol, err


def test_embed():
    torch.manual_seed(0)
    nb = E // 32
    d = (torch.rand(nb) * 0.1).half()
    q = torch.randint(-127, 128, (nb, 32), dtype=torch.int8)
    row = torch.cat([d.view(torch.uint8).view(nb, 2), q.view(torch.uint8)], dim=1).reshape(-1).to(dev)
    w = torch.randn(E, device=dev)
    out = torch.empty(E, device=dev)
    L.ikt_embed(P(row), P(w), P(out), E, fl(1e-6))
    x = (q.float() * d.float()[:, None]).reshape(-1).to(dev)
    close(out, rms(x, 1e-6) * w)


@pytest.mark.parametrize("n_states,n_q", [(1, 1), (3, 3), (5, 4)])
def test_mix(n_states, n_q):
    torch.manual_seed(n_states)
    states = [torch.randn(E, device=dev) * (i + 1) for i in range(n_states)]
    ptrs = (ctypes.c_void_p * n_states)(*[s.data_ptr() for s in states])
    query = torch.randn(n_q, E, device=dev)
    out = torch.empty(n_q, E, device=dev)
    L.ikt_mix(ptrs, n_states, P(query), n_q, P(out), E)
    st = rms(torch.stack(states), 1e-6)
    p = torch.softmax(query @ st.T / math.sqrt(E), dim=-1)
    close(out, p @ st)


def test_control():
    torch.manual_seed(1)
    cur, msg = torch.randn(E, device=dev), torch.randn(E, device=dev)
    down = torch.randn(64, 2 * E, device=dev) * 0.02
    gain, forget = torch.randn(E, 64, device=dev) * 0.1, torch.randn(E, 64, device=dev) * 0.1
    out = torch.empty(E, device=dev)
    L.ikt_control(P(cur), P(msg), P(down), P(gain), P(forget), P(out), E, 64)
    g = down @ torch.cat([rms(cur, 1e-6), rms(msg, 1e-6)])
    ga, fo = torch.tanh(gain @ g), torch.tanh(forget @ g)
    close(out, cur - fo * cur + ga * msg)


def test_rmsnorm_matvec():
    torch.manual_seed(2)
    x, w = torch.randn(E, device=dev), torch.randn(E, device=dev)
    out = torch.empty(E, device=dev)
    L.ikt_rmsnorm(P(x), P(w), P(out), E, fl(1e-6))
    close(out, rms(x, 1e-6) * w)
    W = torch.randn(128, E, device=dev)
    y = torch.empty(128, device=dev)
    L.ikt_matvec_f32(P(W), P(x), P(y), 128, E)
    close(y, W @ x)


def conv_ref(cur, hist, w, dyn=None):
    """cur [C]; hist [3, C] oldest first; w [C, 4]; dyn [heads*4] per-head taps by lag."""
    x = torch.cat([hist, cur[None]])
    out = (x * w.T).sum(0)
    if dyn is not None:
        nh = dyn.numel() // 4
        dd = dyn.view(nh, 4)
        C = cur.numel()
        for lag in range(4):
            out = out + (x[3 - lag].view(nh, C // nh) * dd[:, lag:lag + 1]).view(C)
    return (cur + out) / math.sqrt((2 if dyn is not None else 1) * 4 + 1), x[1:].clone()


@pytest.mark.parametrize("glob,kv_q8", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_attn_prep(glob, kv_q8):
    torch.manual_seed(3 + glob)
    kv = HK * D
    q, k, v, r = (torch.randn(H * D, device=dev) * 3, torch.randn(kv, device=dev) * 3, torch.randn(kv, device=dev),
                  torch.randn(H * NREL, device=dev) * 3)
    kdyn = torch.randn(HK * 4, device=dev) if glob else None
    vdyn = torch.randn(HK * 4, device=dev) if glob else None
    kconv, vconv = torch.randn(kv, 4, device=dev), torch.randn(kv, 4, device=dev)
    hk, hv = torch.randn(3, kv, device=dev), torch.randn(3, kv, device=dev)
    hk0, hv0 = hk.clone(), hv.clone()
    qclip, rclip = torch.rand(H, D, device=dev) + 0.5, torch.rand(H, NREL, device=dev) + 0.5
    qo, ro = torch.empty(H * D, device=dev), torch.empty(H * NREL, device=dev)
    window, pos = (0, 9000) if glob else (1024, 1500)
    rows = 10000 if glob else 1024
    kc, vc = torch.zeros(rows, kv, device=dev), torch.zeros(rows, kv, device=dev)
    kq, vq = torch.zeros(rows, kv, dtype=torch.int8, device=dev), torch.zeros(rows, kv, dtype=torch.int8, device=dev)
    ks, vs = torch.zeros(rows, kv // 32, dtype=torch.float16, device=dev), torch.zeros(rows, kv // 32, dtype=torch.float16, device=dev)
    dpos = torch.tensor([pos], dtype=torch.int32, device=dev)
    a = AttnPrepArgs(q=q.data_ptr(), k=k.data_ptr(), v=v.data_ptr(), r=r.data_ptr(),
                     kdyn=kdyn.data_ptr() if glob else None, vdyn=vdyn.data_ptr() if glob else None,
                     k_conv=kconv.data_ptr(), v_conv=vconv.data_ptr(), hist_k=hk.data_ptr(), hist_v=hv.data_ptr(),
                     q_clip=qclip.data_ptr(), rel_clip=rclip.data_ptr(), q_out=qo.data_ptr(), r_out=ro.data_ptr(),
                     k_cache=kc.data_ptr(), v_cache=vc.data_ptr(), k_q=kq.data_ptr(), v_q=vq.data_ptr(),
                     k_s=ks.data_ptr(), v_s=vs.data_ptr(), pos=dpos.data_ptr(), window=window, ring=window,
                     global_=glob, kv_q8=kv_q8, n_head=H, n_head_kv=HK, d_head=D, n_rel=NREL,
                     qk_clip=22.6274, rel_clip_thr=8.0, eps=1e-6, log_alpha=0.1, log_floor=8192.0)
    L.ikt_attn_prep(ctypes.byref(a))
    torch.cuda.synchronize()

    def clip(x, scale, thr):
        if scale is not None:
            x = x * scale
        n = torch.sqrt((x * x).sum(-1, keepdim=True)) + 1e-6
        return x * torch.clamp(thr / n, 0, 1)

    kk, hk1 = conv_ref(k, hk0, kconv, kdyn)
    vv, hv1 = conv_ref(v, hv0, vconv, vdyn)
    tau = 1 + 0.1 * math.log(max((pos + 1) / 8192, 1)) if glob else 1.0
    slot = pos % window if window else pos
    k_ref = clip(kk.view(HK, D), None, 22.6274).reshape(-1)
    v_ref = rms(vv.view(HK, D), 1e-6).reshape(-1)
    if kv_q8:
        k_got = (kq[slot].view(-1, 32).float() * ks[slot].float()[:, None]).reshape(-1)
        v_got = (vq[slot].view(-1, 32).float() * vs[slot].float()[:, None]).reshape(-1)
        close(k_got, k_ref, tol=1e-2)
        close(v_got, v_ref, tol=1e-2)
    else:
        close(kc[slot], k_ref)
        close(vc[slot], v_ref)
    close(hk, hk1)
    close(hv, hv1)
    close(qo, (clip(q.view(H, D), qclip, 22.6274) * tau).reshape(-1))
    close(ro, (clip(r.view(H, NREL), rclip, 8.0) * tau).reshape(-1))


@pytest.mark.parametrize("glob,pos,kv_q8", [(1, 0, 0), (1, 700, 0), (1, 3000, 0), (0, 300, 0), (0, 2500, 0),
                                            (1, 0, 1), (1, 3000, 1), (0, 2500, 1)])
def test_attention(glob, pos, kv_q8):
    torch.manual_seed(pos + glob)
    kv = HK * D
    window = 0 if glob else 1024
    extent = 1024
    rows = 4096 if glob else 1024
    kc, vc = torch.randn(rows, kv, device=dev), torch.randn(rows, kv, device=dev)
    if kv_q8:
        def q8(t):
            b = t.view(rows, kv // 32, 32)
            am = b.abs().amax(-1)
            sc = (am / 127.0).half()
            q = torch.round(b / (am.clamp(min=1e-30)[..., None] / 127.0)).clamp(-127, 127).to(torch.int8)
            deq = (q.float() * sc.float()[..., None]).reshape(rows, kv)
            return q.reshape(rows, kv).contiguous(), sc.contiguous(), deq
        kq, ks, kc_ref = q8(kc)
        vq, vs, vc_ref = q8(vc)
    else:
        kq = vq = torch.empty(1, dtype=torch.int8, device=dev)
        ks = vs = torch.empty(1, dtype=torch.float16, device=dev)
        kc_ref, vc_ref = kc, vc
    q = torch.randn(H, D, device=dev)
    r = torch.randn(H, NREL, device=dev)
    sinks = torch.randn(H, device=dev)
    gate = torch.randn(H * D, device=dev)
    relb = torch.randn(extent, NREL, device=dev) * 0.3
    out = torch.empty(H * D, device=dev)
    dpos = torch.tensor([pos], dtype=torch.int32, device=dev)
    scratch = torch.empty(H * 16 * 130, device=dev)
    a = AttnArgs(scratch=scratch.data_ptr(), q=q.data_ptr(), r=r.data_ptr(), sinks=sinks.data_ptr(), gate=gate.data_ptr(), rel_bias=relb.data_ptr(),
                 k_cache=kc.data_ptr(), v_cache=vc.data_ptr(), k_q=kq.data_ptr(), v_q=vq.data_ptr(), k_s=ks.data_ptr(), v_s=vs.data_ptr(),
                 out=out.data_ptr(), pos=dpos.data_ptr(), window=window, ring=window, kv_q8=kv_q8,
                 extent=extent, n_head=H, n_head_kv=HK)
    L.ikt_attention(ctypes.byref(a))
    torch.cuda.synchronize()

    lo = max(0, pos - window + 1) if window else 0
    ts = torch.arange(lo, pos + 1, device=dev)
    slots = ts % window if window else ts
    K = kc_ref[slots].view(-1, HK, D).repeat_interleave(H // HK, dim=1)
    V = vc_ref[slots].view(-1, HK, D).repeat_interleave(H // HK, dim=1)
    dist = pos - ts
    table = torch.cat([relb @ r.T, torch.zeros(1, H, device=dev)]) - sinks[None]
    bias = table[torch.clamp(dist, max=extent)]
    s = torch.einsum("hd,thd->th", q, K) / math.sqrt(D) + bias
    mx = torch.clamp(s.amax(0, keepdim=True), min=0)
    ex = torch.exp(s - mx)
    att = ex / (torch.exp(-mx) + ex.sum(0, keepdim=True))
    o = torch.einsum("th,thd->hd", att, V).reshape(-1) * torch.sigmoid(gate)
    close(out, o)


def test_conv_residual():
    torch.manual_seed(5)
    x, hist, w, add = torch.randn(E, device=dev), torch.randn(3, E, device=dev), torch.randn(E, 4, device=dev), torch.randn(E, device=dev)
    h0 = hist.clone()
    out = torch.empty(E, device=dev)
    L.ikt_conv_residual(P(x), P(hist), P(w), P(add), P(out), E)
    y, h1 = conv_ref(x, h0, w)
    close(out, add + y)
    close(hist, h1)


def test_router():
    torch.manual_seed(6)
    W, b, x = torch.randn(128, E, device=dev) * 0.05, torch.randn(128, device=dev) * 0.1, torch.randn(E, device=dev)
    ids = torch.empty(8, dtype=torch.int32, device=dev)
    wts = torch.empty(8, device=dev)
    L.ikt_router(P(W), P(b), P(x), P(ids), P(wts), 128, E, 8, fl(8.0))
    probs = torch.sigmoid(W @ x)
    ref_ids = torch.topk(probs + b, 8).indices
    assert set(ids.tolist()) == set(ref_ids.tolist())
    w = probs[ids.long()]
    close(wts, w / w.sum() * 8.0)

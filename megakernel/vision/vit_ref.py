"""fp64 reference of the Qwen3.x vision tower (qwen3vl_merger mmproj), from the GGUF weights, on vit_check's dumped
preprocessed image (<prefix>.<i>.img.f32 + .txt), compared with its dumped libmtmd rows (.mtmd.f32) and vit_cpu rows
(.vit.f32). Same graph as llama.cpp's clip (models/qwen3vl.cpp), every op in float64.
Writes the reference rows (rounded to fp32) to <prefix>.<i>.f64.f32. Needs numpy and gguf-py (pip install gguf, or
LLAMA=<llama.cpp checkout>). Usage: python vit_ref.py <mmproj.gguf> <dump prefix> <i> [<i> ...]
(the dumps: tools/vit_check ... --dump <prefix> image...)"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("LLAMA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llama.cpp")), "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

mm = gguf.GGUFReader(sys.argv[1])
T = {t.name: t for t in mm.tensors}


def W(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type).astype(np.float64)
    return a.reshape([int(x) for x in reversed(t.shape)])


D, nH, HD, nL = 1152, 16, 72, 27
pe0 = W("v.patch_embd.weight") + W("v.patch_embd.weight.1")  # [1152][3][16][16]
Wp = pe0.reshape(D, 768)
bp = W("v.patch_embd.bias")
pos = W("v.position_embd.weight")  # [2304][1152]
S = int(round(np.sqrt(pos.shape[0])))
pos = pos.reshape(S, S, D)  # [y][x][e]
L = []
for l in range(nL):
    p = f"v.blk.{l}."
    L.append({k: W(p + k) for k in ("ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight",
                                     "attn_out.bias", "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias",
                                     "ffn_down.weight", "ffn_down.bias")})
post_w, post_b = W("v.post_ln.weight"), W("v.post_ln.bias")
m0w, m0b, m2w, m2b = W("mm.0.weight"), W("mm.0.bias"), W("mm.2.weight"), W("mm.2.bias")


def ln(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu(x):  # tanh form (ggml_gelu)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def encode(img, tw, th):
    pw, ph = tw // 16, th // 16
    M = pw * ph
    t = np.arange(M)
    g, q = t >> 2, t & 3
    px = 2 * (g % (pw // 2)) + (q & 1)
    py = 2 * (g // (pw // 2)) + (q >> 1)
    im = img.reshape(th, tw, 3).astype(np.float64)
    patches = np.stack([im[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16, :].transpose(2, 0, 1).reshape(768)
                        for x, y in zip(px, py)])
    # learned position embedding, bilinear with align corners from S x S to pw x ph
    sf0 = (pw - 1) / (S - 1) if pw > 1 else pw / S
    sf1 = (ph - 1) / (S - 1) if ph > 1 else ph / S
    sx, sy = px / sf0, py / sf1
    x0 = np.clip(np.floor(sx).astype(int), 0, S - 1)
    y0 = np.clip(np.floor(sy).astype(int), 0, S - 1)
    x1, y1 = np.clip(x0 + 1, 0, S - 1), np.clip(y0 + 1, 0, S - 1)
    dx = np.clip(sx - x0, 0, 1)[:, None]
    dy = np.clip(sy - y0, 0, 1)[:, None]
    pemb = (pos[y0, x0] * (1 - dx) * (1 - dy) + pos[y0, x1] * dx * (1 - dy) + pos[y1, x0] * (1 - dx) * dy
            + pos[y1, x1] * dx * dy)
    x = patches @ Wp.T + bp + pemb
    # vision RoPE: pairs (i, i + 36); i < 18 by row (py), i >= 18 by column (px)
    i = np.arange(36)
    freq = 10000.0 ** (-2.0 * (i % 18) / 36)
    ang = np.where(i < 18, py[:, None], px[:, None]) * freq[None, :]  # [M][36]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]

    def rope(v):  # [M][nH][72]
        a, b = v[..., :36], v[..., 36:]
        return np.concatenate([a * c - b * s, a * s + b * c], -1)

    for l in range(nL):
        y = L[l]
        h = ln(x, y["ln1.weight"], y["ln1.bias"])
        qkv = h @ y["attn_qkv.weight"].T + y["attn_qkv.bias"]
        qh = rope(qkv[:, :D].reshape(M, nH, HD))
        kh = rope(qkv[:, D:2 * D].reshape(M, nH, HD))
        vh = qkv[:, 2 * D:].reshape(M, nH, HD)
        att = np.empty((M, nH, HD))
        for hh in range(nH):
            sc = qh[:, hh] @ kh[:, hh].T / np.sqrt(HD)
            sc -= sc.max(-1, keepdims=True)
            pr = np.exp(sc)
            pr /= pr.sum(-1, keepdims=True)
            att[:, hh] = pr @ vh[:, hh]
        x = x + att.reshape(M, D) @ y["attn_out.weight"].T + y["attn_out.bias"]
        h = ln(x, y["ln2.weight"], y["ln2.bias"])
        x = x + gelu(h @ y["ffn_up.weight"].T + y["ffn_up.bias"]) @ y["ffn_down.weight"].T + y["ffn_down.bias"]
    h = ln(x, post_w, post_b).reshape(M // 4, 4 * D)
    return gelu(h @ m0w.T + m0b) @ m2w.T + m2b


def cmp(name, ref, got):
    d = got - ref
    rel = np.linalg.norm(d) / np.linalg.norm(ref)
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    print(f"   {name:8s} vs fp64: rel L2 {rel:.3e}  max |d| {np.abs(d).max():.3e}  min row cos {cos.min():.6f}  "
          f"mean row cos {cos.mean():.6f}", flush=True)


prefix = sys.argv[2]
for i in sys.argv[3:]:
    p = f"{prefix}.{i}"
    tw, th, nt, ne = map(int, open(p + ".txt").read().split())
    img = np.fromfile(p + ".img.f32", np.float32)
    t0 = time.time()
    ref = encode(img, tw, th)
    print(f"image {i}: {tw} x {th}, {nt} tokens, fp64 reference in {time.time() - t0:.0f} s", flush=True)
    ref.astype(np.float32).tofile(p + ".f64.f32")  # (the reference rows, rounded to fp32)
    for name in ("mtmd", "vit"):
        cmp(name, ref, np.fromfile(f"{p}.{name}.f32", np.float32).reshape(nt, ne).astype(np.float64))

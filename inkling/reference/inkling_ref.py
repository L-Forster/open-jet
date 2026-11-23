"""Reference forward pass of Inkling 2 Mini, read from a mainline-layout (inkling2_loop) GGUF.

It follows llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp (base 19bc41d) op for op, in fp32 on
the GPU, processing a whole token sequence per layer (no KV cache). The engine's kernels are checked against it,
and it against llama.cpp.

    from inkling_ref import InklingRef
    m = InklingRef("Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf")
    logits, routing = m.forward(token_ids)  # logits [T, 200058]; routing[app] = expert ids [T, 8]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

GGUF_PY = Path("/home/louis/bench-engines/llama.cpp-inkling/gguf-py")

# E2M1 values doubled (ggml kvalues_mxfp4); the UE4M3 decode below halves them back
_KV_FP4 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)


def _ue4m3(x: torch.Tensor) -> torch.Tensor:
    """Unsigned E4M3 (bias 7) to float, times 0.5 (ggml_ue4m3_to_fp32)."""
    x = x.to(torch.int32)
    e = (x >> 3) & 0xF
    m = (x & 0x7).to(torch.float32)
    raw = torch.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * torch.pow(2.0, (e - 7).to(torch.float32)))
    return torch.where((x == 0) | (x == 0x7F), torch.zeros_like(raw), raw * 0.5)


def deq_nvfp4(raw: torch.Tensor, n: int) -> torch.Tensor:
    """raw: uint8 [rows, n/64*36] -> float32 [rows, n]. Block: 4 UE4M3 scales, 32 bytes of nibbles;
    sub-block s holds elements s*16..s*16+15, low nibbles first then high nibbles."""
    rows = raw.shape[0]
    b = raw.view(rows, n // 64, 36)
    d = _ue4m3(b[:, :, :4])                                  # [rows, nb, 4]
    qs = b[:, :, 4:].view(rows, n // 64, 4, 8).to(torch.int64)
    lo = _KV_FP4.to(raw.device)[qs & 0xF]
    hi = _KV_FP4.to(raw.device)[qs >> 4]
    v = torch.cat([lo, hi], dim=-1) * d[..., None]          # [rows, nb, 4, 16]
    return v.reshape(rows, n)


def deq_q8_0(raw: torch.Tensor, n: int) -> torch.Tensor:
    rows = raw.shape[0]
    b = raw.view(rows, n // 32, 34)
    d = b[:, :, :2].contiguous().view(torch.float16).to(torch.float32)  # [rows, nb, 1]
    q = b[:, :, 2:].contiguous().view(torch.int8).to(torch.float32)
    return (q * d).reshape(rows, n)


class InklingRef:
    def __init__(self, gguf_path: str | Path, device: str = "cuda"):
        sys.path.insert(0, str(GGUF_PY))
        from gguf import GGUFReader

        self.dev = device
        self.r = GGUFReader(str(gguf_path))
        self.t = {t.name: t for t in self.r.tensors}
        f = {k: v for k, v in self.r.fields.items()}

        def kv(name):
            return f[f"inkling2_loop.{name}"].contents()

        self.n_layer = kv("block_count")
        self.n_embd = kv("embedding_length")
        self.n_head = kv("attention.head_count")[0]
        self.n_head_kv = kv("attention.head_count_kv")[0]
        self.d_head = kv("attention.key_length")
        self.n_expert = kv("expert_count")
        self.n_used = kv("expert_used_count")
        self.n_ff = kv("expert_feed_forward_length")
        self.eps = kv("attention.layer_norm_rms_epsilon")
        self.is_swa = kv("attention.sliding_window_pattern")
        self.n_swa = kv("attention.sliding_window")
        self.n_conv = kv("shortconv.l_cache")
        self.w_scale = kv("expert_weights_scale")
        self.situ_beta = kv("activation.situ_beta")
        self.situ_lb = kv("activation.situ_linear_beta")
        self.block = kv("attn_res.block_size")
        self.softcap = kv("final_logit_softcapping")
        self.n_rel = kv("attention.relative_length")
        self.rel_extent = kv("attention.relative_extent")
        self.qk_clip = kv("attention.qk_clip")
        self.rel_clip = kv("attention.relative_clip")
        self.log_floor = kv("attention.log_scale_floor")
        self.log_alpha = kv("attention.log_scale_alpha")
        self.core_start, self.core_end, self.n_passes = kv("loop.start"), kv("loop.end"), kv("loop.count")
        self.n_vocab = kv("vocab_size_unpadded")
        self._cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------ weights

    def raw(self, name: str) -> np.ndarray:
        return self.t[name].data

    def w(self, name: str) -> torch.Tensor:
        """Dense weight as float32 [ne1, ne0] (rows = outputs), cached on the device."""
        if name in self._cache:
            return self._cache[name]
        t = self.t[name]
        ne = [int(x) for x in t.shape]
        typ = t.tensor_type.name
        data = torch.from_numpy(np.ascontiguousarray(t.data)).to(self.dev)
        if typ == "F32":
            out = data.view(torch.float32).reshape(list(reversed(ne)))
        elif typ == "BF16":
            out = data.view(torch.bfloat16).to(torch.float32).reshape(list(reversed(ne)))
        elif typ == "Q8_0":
            out = deq_q8_0(data.reshape(-1, data.shape[-1]), ne[0]).reshape(list(reversed(ne)))
        else:  # other ggml types (K-quants, IQ4, Q5_0): gguf-py's reference dequantisation
            out = self._deq_gguf(t.data, t.tensor_type).reshape(list(reversed(ne)))
        if out.dim() == 1:
            out = out.contiguous()
        self._cache[name] = out
        return out

    def experts(self, name: str, ids: list[int]) -> torch.Tensor:
        """Expert matrices [len(ids), out, in] in float32, times their .scale."""
        t = self.t[name]
        ne = [int(x) for x in t.shape]  # [in, out, n_expert]
        raw = t.data  # [n_expert, out, row_bytes]
        sub = torch.from_numpy(np.ascontiguousarray(raw[ids])).to(self.dev)
        rows = sub.reshape(-1, sub.shape[-1])
        typ = t.tensor_type.name
        if typ == "NVFP4":
            m = deq_nvfp4(rows, ne[0])
        elif typ == "Q8_0":
            m = deq_q8_0(rows, ne[0])
        else:
            m = self._deq_gguf(np.ascontiguousarray(raw[ids]), t.tensor_type)
        m = m.reshape(len(ids), ne[1], ne[0])
        sname = name.replace(".weight", ".scale")
        if sname in self.t:
            s = torch.from_numpy(np.ascontiguousarray(self.t[sname].data)).to(self.dev)[ids]
            m = m * s[:, None, None]
        return m

    def _deq_gguf(self, raw: np.ndarray, qtype) -> torch.Tensor:
        from gguf.quants import dequantize
        return torch.from_numpy(np.ascontiguousarray(dequantize(np.ascontiguousarray(raw), qtype), dtype=np.float32)).to(self.dev)

    # ------------------------------------------------------------ ops

    def rms(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def norm(self, x, wname):
        return self.rms(x, self.eps) * self.w(wname)

    def mix(self, blocks: list[torch.Tensor], cur: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Depth attention: [T, n_q, n_embd] = softmax over states of (q . rms(state)) / sqrt(d), times rms(state)."""
        states = torch.stack([*blocks, cur], dim=1)              # [T, S, E]
        states = self.rms(states, 1e-6)
        q = query if query.dim() == 2 else query[None]           # [n_q, E]
        scores = torch.einsum("tse,qe->tqs", states, q) / math.sqrt(self.n_embd)
        p = torch.softmax(scores, dim=-1)
        return torch.einsum("tqs,tse->tqe", p, states)

    @staticmethod
    def clip(x: torch.Tensor, scale: torch.Tensor | None, thr: float) -> torch.Tensor:
        if scale is not None:
            x = x * scale
        n = torch.sqrt((x * x).sum(-1, keepdim=True)) + 1e-6
        return x * torch.clamp(thr / n, 0.0, 1.0)

    def convolve(self, cur: torch.Tensor, wname: str, dynamic: torch.Tensor | None) -> torch.Tensor:
        """Causal short convolution over time with a residual. cur [T, C]; weight [C, n_conv] (tap k reads x[t-3+k]);
        dynamic [T, n_heads*n_conv] adds per-head input-dependent taps (tap lag reads x[t-lag])."""
        T, C = cur.shape
        k = self.n_conv
        w = self.w(wname)                                        # [C, n_conv]
        x = torch.cat([torch.zeros(k - 1, C, device=cur.device), cur], dim=0)  # empty history at sequence start
        out = torch.zeros_like(cur)
        for j in range(k):
            out = out + x[j:j + T] * w[:, j]
        if dynamic is not None:
            nh = dynamic.shape[1] // k
            dyn = dynamic.view(T, nh, k)                         # [T, head, lag]
            for lag in range(k):
                win = x[k - 1 - lag:k - 1 - lag + T].view(T, nh, C // nh)
                out = out + (win * dyn[:, :, lag:lag + 1]).view(T, C)
        return (cur + out) / math.sqrt((2 if dynamic is not None else 1) * k + 1)

    def control(self, cur, message, il_phys: int, ctrl: str):
        g = torch.cat([self.rms(cur, 1e-6), self.rms(message, 1e-6)], dim=-1)
        g = g @ self.w(f"blk.{il_phys}.loop_{ctrl}_ctrl_down.weight").T
        gain = torch.tanh(g @ self.w(f"blk.{il_phys}.loop_{ctrl}_ctrl_gain.weight").T)
        forget = torch.tanh(g @ self.w(f"blk.{il_phys}.loop_{ctrl}_ctrl_forget.weight").T)
        return cur - forget * cur + gain * message

    # ------------------------------------------------------------ layer

    def layer(self, il: int, ip: int, cur: torch.Tensor, blocks: list, pos: torch.Tensor,
              messages: torch.Tensor | None, routing: list):
        T = cur.shape[0]
        E, H, HK, D = self.n_embd, self.n_head, self.n_head_kv, self.d_head
        p = f"blk.{il}."
        swa = self.is_swa[il]

        inpSA = cur
        inputs = [cur, cur, cur]
        if il > 0:
            mixed = self.mix(blocks, cur, self.w(p + "attn_res_query.weight"))  # [T, 1 or 3, E]
            inputs = [mixed[:, 0 if swa else j] for j in range(3)]
        if messages is not None:
            inputs = [self.control(inputs[j], messages[:, j], il, "qkv"[j]) for j in range(3)]
        if swa and messages is not None:
            inputs = [self.norm(x, p + "attn_norm.weight") for x in inputs]
        elif swa:
            n = self.norm(inputs[0], p + "attn_norm.weight")
            inputs = [n, n, n]
        else:
            inputs = [self.norm(inputs[0], p + "attn_q_inp_norm.weight"),
                      self.norm(inputs[1], p + "attn_k_inp_norm.weight"),
                      self.norm(inputs[2], p + "attn_v_inp_norm.weight")]

        q = inputs[0] @ self.w(p + "attn_q.weight").T
        k = self.convolve(inputs[1] @ self.w(p + "attn_k.weight").T, p + "attn_k_conv.weight",
                          None if swa else inputs[1] @ self.w(p + "attn_k_conv_proj.weight").T)
        v = self.convolve(inputs[2] @ self.w(p + "attn_v.weight").T, p + "attn_v_conv.weight",
                          None if swa else inputs[2] @ self.w(p + "attn_v_conv_proj.weight").T)
        r = inputs[0] @ self.w(p + "attn_rel.weight").T

        q = self.clip(q.view(T, H, D), self.w(p + "attn_q_clip.weight"), self.qk_clip)
        k = self.clip(k.view(T, HK, D), None, self.qk_clip)
        v = self.rms(v.view(T, HK, D), self.eps)
        r = self.clip(r.view(T, H, self.n_rel), self.w(p + "attn_rel_clip.weight"), self.rel_clip)
        if not swa:
            tau = 1.0 + self.log_alpha * torch.log(torch.clamp((pos.float() + 1.0) / self.log_floor, min=1.0))
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]

        sinks = inputs[0] @ self.w(p + "attn_sink_proj.weight").T          # [T, H]
        extent = self.n_swa if swa else self.rel_extent
        relb = self.w(p + "attn_rel_bias.weight")                          # [extent, n_rel]
        relb = torch.cat([relb, torch.zeros(1, self.n_rel, device=relb.device)], dim=0)
        table = torch.einsum("dk,thk->thd", relb, r) - sinks[:, :, None]  # [T, H, extent+1]
        dist = pos[:, None] - pos[None, :]                                 # [Tq, Tk]
        idx = torch.where((dist >= 0) & (dist < extent), dist, torch.full_like(dist, extent))
        bias = torch.gather(table, 2, idx[:, None, :].expand(T, H, T))     # [Tq, H, Tk]
        kk = k.repeat_interleave(H // HK, dim=1)                           # [T, H, D]
        vv = v.repeat_interleave(H // HK, dim=1)
        s = torch.einsum("qhd,khd->qhk", q, kk) / math.sqrt(D) + bias
        mask = dist < 0
        if swa:
            mask = mask | (dist >= self.n_swa)
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        # sink logit 0: exp(s) / (1 + sum exp(s))
        mx = torch.clamp(s.amax(-1, keepdim=True), min=0.0)
        ex = torch.exp(s - mx)
        att = ex / (torch.exp(-mx) + ex.sum(-1, keepdim=True))
        o = torch.einsum("qhk,khd->qhd", att, vv).reshape(T, H * D)
        o = o * torch.sigmoid(inputs[0] @ self.w(p + "attn_gate.weight").T)
        o = self.convolve(o @ self.w(p + "attn_output.weight").T, p + "attn_post_conv.weight", None)

        if ip % self.block == 0:
            blocks.append(inpSA)
            cur = o
        else:
            cur = inpSA + o
        ffn_inp = cur

        x = self.mix(blocks, cur, self.w(p + "ffn_res_query.weight"))[:, 0]
        if messages is not None:
            x = self.control(x, messages[:, 3], il, "ffn")
        x = self.norm(x, p + "ffn_norm.weight")

        logits = x @ self.w(p + "ffn_gate_inp.weight").T                   # [T, n_expert]
        probs = torch.sigmoid(logits)
        sel = probs + self.w(p + "exp_probs_b.bias") if (p + "exp_probs_b.bias") in self.t else probs
        ids = torch.topk(sel, self.n_used, dim=-1).indices                # [T, 8]
        wts = torch.gather(probs, 1, ids)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * self.w_scale
        routing.append(ids.cpu())

        out = torch.zeros(T, E, device=x.device)
        uniq = torch.unique(ids).tolist()
        for c in range(0, len(uniq), 16):
            chunk = uniq[c:c + 16]
            gu = self.experts(p + "ffn_gate_up_exps.weight", chunk)       # [n, 2*n_ff, E]
            dn = self.experts(p + "ffn_down_exps.weight", chunk)          # [n, E, n_ff]
            for j, e in enumerate(chunk):
                hit = (ids == e)
                rows = hit.any(-1).nonzero().squeeze(-1)
                if rows.numel() == 0:
                    continue
                h = x[rows] @ gu[j].T
                g, u = h[:, :self.n_ff], h[:, self.n_ff:]
                act = self.situ_beta * torch.tanh(g / self.situ_beta) * torch.sigmoid(g)
                u = self.situ_lb * torch.tanh(u / self.situ_lb)
                y = (act * u) @ dn[j].T
                wj = (wts[rows] * hit[rows].float()).sum(-1, keepdim=True)
                out[rows] += wj * y
            del gu, dn
        out = self.convolve(out, p + "ffn_post_conv.weight", None)
        return ffn_inp + out

    # ------------------------------------------------------------ model

    @torch.no_grad()
    def forward(self, tokens: list[int]):
        T = len(tokens)
        pos = torch.arange(T, device=self.dev)
        emb = self.t["token_embd.weight"]
        rows = torch.from_numpy(np.ascontiguousarray(emb.data[tokens])).to(self.dev)
        cur = deq_q8_0(rows, self.n_embd)
        cur = self.norm(cur, "token_embd_norm.weight")

        routing: list[torch.Tensor] = []
        blocks: list[torch.Tensor] = []
        for il in range(self.core_start):
            cur = self.layer(il, il, cur, blocks, pos, None, routing)
        prelude, prelude_blocks = cur, list(blocks)
        for pss in range(self.n_passes):
            msgs = None
            if pss > 0:
                bank = blocks[len(prelude_blocks) + 1:]
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
            for ip in range(self.core_start, self.core_end):
                if pss > 0:
                    msgs = self.mix(bank, bank_out, self.w(f"blk.{ip}.loop_msg_query.weight"))  # [T, 4, E]
                cur = self.layer(ip, ip, cur, blocks, pos, msgs, routing)
        for il in range(self.core_end, self.n_layer):
            cur = self.layer(il, il, cur, blocks, pos, None, routing)

        x = self.mix(blocks, cur, self.w("output_res_query.weight"))[:, 0]
        x = self.norm(x, "output_norm.weight")
        out = self.t["output.weight"]
        logits = torch.empty(T, self.n_vocab, device=self.dev)
        step = 16384
        for a in range(0, self.n_vocab, step):
            b = min(a + step, self.n_vocab)
            rows = np.ascontiguousarray(out.data[a:b])
            if out.tensor_type.name == "Q8_0":
                wchunk = deq_q8_0(torch.from_numpy(rows).to(self.dev), self.n_embd)
            else:
                wchunk = self._deq_gguf(rows, out.tensor_type).reshape(b - a, self.n_embd)
            logits[:, a:b] = x @ wchunk.T
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits, routing  # routing in execution order: 6 prelude, 12 x 3 core passes, 6 tail

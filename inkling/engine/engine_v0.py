"""Inkling 2 Mini decode engine, prototype: PyTorch orchestration around the engine's CUDA kernels.

Everything is in VRAM (24 GB cards). Quantized matrices (Q8_0 dense, NVFP4/Q8_0 experts) are repacked once and run
through libinkling_kernels.so; the small fp32/bf16 tensors and the per-token glue (norms, depth mixing, loop control,
attention over the cache) run as fp32 torch ops. It decodes one token at a time with per-slot KV caches and
convolution histories, and is checked against reference/inkling_ref.py, which recomputes the whole sequence.

It fixes the decode algorithm (state per cache slot, the loop passes, the relative bias over the cache) before the
C++/CUDA engine is written, and is the baseline that engine is measured against.
"""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE.parent / "reference"))
from inkling_ref import InklingRef, deq_q8_0  # noqa: E402

TYPE = {"NVFP4": 0, "Q8_0": 1}


def _lib():
    L = ctypes.CDLL(str(ENGINE / "libinkling_kernels.so"))
    vp, ip, fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.ik_matrix_bytes.argtypes = [ip, ip, ip]
    L.ik_matrix_bytes.restype = ctypes.c_size_t
    L.ik_repack.argtypes = [ip, vp, vp, ip, ip, vp]
    L.ik_matvec_q8_0.argtypes = [vp, vp, vp, ip, ip, vp]
    L.ik_experts_gate_up.argtypes = [ip, vp, vp, vp, ip, vp, vp, ip, ip, fl, fl, vp]
    L.ik_experts_down.argtypes = [ip, vp, vp, vp, vp, ip, vp, vp, vp, ip, ip, vp]
    return L


def _p(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


class Slot:
    """State of one cache slot (a layer application): KV cache and convolution histories."""

    def __init__(self, ctx: int, hk: int, d: int, e: int, dev: str):
        self.k = torch.zeros(ctx, hk, d, device=dev)
        self.v = torch.zeros(ctx, hk, d, device=dev)
        kv = hk * d
        self.hist = {name: torch.zeros(3, c, device=dev) for name, c in (("k", kv), ("v", kv), ("o", e), ("f", e))}


class InklingEngineV0:
    def __init__(self, gguf: str | Path, ctx: int = 2048, device: str = "cuda"):
        self.L = _lib()
        self.ref = InklingRef(gguf, device)  # hyperparameters and small-tensor loading
        self.dev = device
        self.ctx = ctx
        r = self.ref
        self.q8 = {}  # name -> (buffer, rows, n)
        self.experts = {}  # layer -> dict
        self.stream = None
        for t in r.r.tensors:
            if t.tensor_type.name == "Q8_0" and "_exps" not in t.name and t.name != "token_embd.weight":
                n, rows = int(t.shape[0]), int(t.shape[1])
                self.q8[t.name] = (self._repack(t.data, 1, rows, n), rows, n)
        for il in range(r.n_layer):
            ex = {}
            for name, rows, n in (("ffn_gate_up_exps", 2 * r.n_ff, r.n_embd), ("ffn_down_exps", r.n_embd, r.n_ff)):
                t = r.t[f"blk.{il}.{name}.weight"]
                typ = TYPE[t.tensor_type.name]
                bufs = [self._repack(t.data[e], typ, rows, n) for e in range(r.n_expert)]
                sname = f"blk.{il}.{name}.scale"
                scale = torch.from_numpy(np.array(r.t[sname].data)).to(device) if sname in r.t else None
                ex[name] = (typ, bufs, torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=device), scale)
            self.experts[il] = ex
        self.h = torch.empty(r.n_used, r.n_ff, device=device)
        self.part = torch.empty(r.n_used, r.n_embd, device=device)
        self.reset()

    def _repack(self, data: np.ndarray, typ: int, rows: int, n: int) -> torch.Tensor:
        src = torch.from_numpy(np.array(data)).to(self.dev)
        dst = torch.empty(self.L.ik_matrix_bytes(typ, rows, n), dtype=torch.uint8, device=self.dev)
        assert self.L.ik_repack(typ, _p(src), _p(dst), rows, n, None) == 0
        return dst

    def reset(self):
        r = self.ref
        n_slots = r.n_layer + (r.n_passes - 1) * (r.core_end - r.core_start)
        self.slots = [Slot(self.ctx, r.n_head_kv, r.d_head, r.n_embd, self.dev) for _ in range(n_slots)]
        self.pos = 0

    # ------------------------------------------------------------ ops

    def mv(self, name: str, x: torch.Tensor) -> torch.Tensor:
        buf, rows, n = self.q8[name]
        y = torch.empty(rows, device=self.dev)
        assert self.L.ik_matvec_q8_0(_p(buf), _p(x.contiguous()), _p(y), rows, n, None) == 0
        return y

    def conv(self, cur: torch.Tensor, slot: Slot, key: str, wname: str, dynamic: torch.Tensor | None) -> torch.Tensor:
        r = self.ref
        k = r.n_conv
        hist = slot.hist[key]
        x = torch.cat([hist, cur[None]], dim=0)                 # [4, C]: oldest first, current last
        w = r.w(wname)                                          # [C, 4]
        out = (x * w.T).sum(0)
        if dynamic is not None:
            nh = dynamic.numel() // k
            dyn = dynamic.view(nh, k)
            C = cur.numel()
            for lag in range(k):
                out = out + (x[k - 1 - lag].view(nh, C // nh) * dyn[:, lag:lag + 1]).view(C)
        slot.hist[key] = x[1:].clone()
        return (cur + out) / math.sqrt((2 if dynamic is not None else 1) * k + 1)

    def moe(self, il: int, x: torch.Tensor) -> torch.Tensor:
        r = self.ref
        p = f"blk.{il}."
        logits = r.w(p + "ffn_gate_inp.weight") @ x
        probs = torch.sigmoid(logits)
        sel = probs + r.w(p + "exp_probs_b.bias")
        ids = torch.topk(sel, r.n_used).indices
        wts = probs[ids]
        wts = (wts / (wts.sum() + 1e-20) * r.w_scale).contiguous()
        ids32 = ids.to(torch.int32)
        self.last_ids.append(ids.cpu())
        typ, _, gu_ptrs, gu_scale = self.experts[il]["ffn_gate_up_exps"]
        typ2, _, dn_ptrs, dn_scale = self.experts[il]["ffn_down_exps"]
        y = torch.empty(r.n_embd, device=self.dev)
        assert self.L.ik_experts_gate_up(typ, _p(gu_ptrs), _p(gu_scale), _p(ids32), r.n_used, _p(x.contiguous()), _p(self.h),
                                         r.n_embd, r.n_ff, r.situ_beta, r.situ_lb, None) == 0
        assert self.L.ik_experts_down(typ2, _p(dn_ptrs), _p(dn_scale), _p(ids32), _p(wts), r.n_used, _p(self.h), _p(self.part),
                                      _p(y), r.n_embd, r.n_ff, None) == 0
        return y

    # ------------------------------------------------------------ layer

    def layer(self, il: int, ip: int, slot_idx: int, cur: torch.Tensor, blocks: list, messages: torch.Tensor | None):
        r = self.ref
        E, H, HK, D = r.n_embd, r.n_head, r.n_head_kv, r.d_head
        p = f"blk.{il}."
        swa = r.is_swa[il]
        slot = self.slots[slot_idx]
        pos = self.pos

        inpSA = cur
        inputs = [cur, cur, cur]
        if il > 0:
            mixed = r.mix([b[None] for b in blocks], cur[None], r.w(p + "attn_res_query.weight"))[0]
            inputs = [mixed[0 if swa else j] for j in range(3)]
        if messages is not None:
            inputs = [r.control(inputs[j][None], messages[j][None], il, "qkv"[j])[0] for j in range(3)]
        if swa and messages is not None:
            inputs = [r.norm(x, p + "attn_norm.weight") for x in inputs]
        elif swa:
            n = r.norm(inputs[0], p + "attn_norm.weight")
            inputs = [n, n, n]
        else:
            inputs = [r.norm(inputs[0], p + "attn_q_inp_norm.weight"),
                      r.norm(inputs[1], p + "attn_k_inp_norm.weight"),
                      r.norm(inputs[2], p + "attn_v_inp_norm.weight")]

        q = self.mv(p + "attn_q.weight", inputs[0])
        k = self.conv(self.mv(p + "attn_k.weight", inputs[1]), slot, "k", p + "attn_k_conv.weight",
                      None if swa else r.w(p + "attn_k_conv_proj.weight") @ inputs[1])
        v = self.conv(self.mv(p + "attn_v.weight", inputs[2]), slot, "v", p + "attn_v_conv.weight",
                      None if swa else r.w(p + "attn_v_conv_proj.weight") @ inputs[2])
        rr = self.mv(p + "attn_rel.weight", inputs[0])

        q = r.clip(q.view(H, D), r.w(p + "attn_q_clip.weight"), r.qk_clip)
        k = r.clip(k.view(HK, D), None, r.qk_clip)
        v = r.rms(v.view(HK, D), r.eps)
        rr = r.clip(rr.view(H, r.n_rel), r.w(p + "attn_rel_clip.weight"), r.rel_clip)
        if not swa:
            tau = 1.0 + r.log_alpha * math.log(max((pos + 1.0) / r.log_floor, 1.0))
            q = q * tau
            rr = rr * tau
        slot.k[pos] = k
        slot.v[pos] = v

        sinks = r.w(p + "attn_sink_proj.weight") @ inputs[0]                  # [H]
        extent = r.n_swa if swa else r.rel_extent
        lo = max(0, pos - r.n_swa + 1) if swa else 0
        kk = slot.k[lo:pos + 1].repeat_interleave(H // HK, dim=1)             # [T, H, D]
        vv = slot.v[lo:pos + 1].repeat_interleave(H // HK, dim=1)
        dist = pos - torch.arange(lo, pos + 1, device=self.dev)               # [T]
        relb = r.w(p + "attn_rel_bias.weight")                                # [extent, n_rel]
        idx = torch.clamp(dist, max=extent)
        table = torch.cat([relb @ rr.T, torch.zeros(1, H, device=self.dev)], dim=0) - sinks[None]  # [extent+1, H]
        bias = table[idx]                                                     # [T, H]
        s = torch.einsum("hd,thd->th", q, kk) / math.sqrt(D) + bias
        mx = torch.clamp(s.amax(0, keepdim=True), min=0.0)
        ex = torch.exp(s - mx)
        att = ex / (torch.exp(-mx) + ex.sum(0, keepdim=True))
        o = torch.einsum("th,thd->hd", att, vv).reshape(H * D)
        o = o * torch.sigmoid(self.mv(p + "attn_gate.weight", inputs[0]))
        o = self.conv(self.mv(p + "attn_output.weight", o), slot, "o", p + "attn_post_conv.weight", None)

        if ip % r.block == 0:
            blocks.append(inpSA)
            cur = o
        else:
            cur = inpSA + o
        ffn_inp = cur

        x = r.mix([b[None] for b in blocks], cur[None], r.w(p + "ffn_res_query.weight"))[0, 0]
        if messages is not None:
            x = r.control(x[None], messages[3][None], il, "ffn")[0]
        x = r.norm(x, p + "ffn_norm.weight")
        out = self.conv(self.moe(il, x), slot, "f", p + "ffn_post_conv.weight", None)
        return ffn_inp + out

    @torch.no_grad()
    def decode(self, token: int) -> torch.Tensor:
        """Feed one token, return the logits for the next one."""
        r = self.ref
        assert self.pos < self.ctx
        self.last_ids = []
        emb = r.t["token_embd.weight"]
        cur = deq_q8_0(torch.from_numpy(np.array(emb.data[token:token + 1])).to(self.dev), r.n_embd)[0]
        cur = r.norm(cur, "token_embd_norm.weight")
        blocks: list[torch.Tensor] = []
        for il in range(r.core_start):
            cur = self.layer(il, il, il, cur, blocks, None)
        prelude, prelude_blocks = cur, list(blocks)
        n_core = r.core_end - r.core_start
        for pss in range(r.n_passes):
            msgs = None
            if pss > 0:
                bank = blocks[len(prelude_blocks) + 1:]
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
            for ip in range(r.core_start, r.core_end):
                slot = ip if pss == 0 else r.n_layer + (pss - 1) * n_core + (ip - r.core_start)
                if pss > 0:
                    msgs = r.mix([b[None] for b in bank], bank_out[None], r.w(f"blk.{ip}.loop_msg_query.weight"))[0]
                cur = self.layer(ip, ip, slot, cur, blocks, msgs)
        for il in range(r.core_end, r.n_layer):
            cur = self.layer(il, il, il, cur, blocks, None)
        x = r.mix([b[None] for b in blocks], cur[None], r.w("output_res_query.weight"))[0, 0]
        x = r.norm(x, "output_norm.weight")
        logits = self.mv("output.weight", x)[: r.n_vocab]
        self.pos += 1
        return r.softcap * torch.tanh(logits / r.softcap)

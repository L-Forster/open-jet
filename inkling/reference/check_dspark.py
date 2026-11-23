"""PyTorch reference verification of Inkling 2 Mini's DSpark (mtp.safetensors) drafter against UD-IQ4_XS.

Verifies the exact tap points [5, 23, 41, 48], DFlare feature mixing, 3-layer bidirectional block attention,
and recurrent Markov head, and measures mean accepted draft tokens per 7-token proposal on real sequences.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REF_DIR))
from inkling_ref import InklingRef, deq_q8_0  # noqa: E402


def load_safetensors(path: Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + hlen])
    base = 8 + hlen
    out: dict[str, torch.Tensor] = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[v["dtype"]]
        t = torch.frombuffer(bytearray(raw[base + a : base + b]), dtype=dt).reshape(v["shape"])
        out[k] = t.to(device=device, dtype=torch.float32)
    return out


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_neox(x: torch.Tensor, pos: torch.Tensor, theta: float = 1_000_000.0) -> torch.Tensor:
    """x: [T, H, D], pos: [T] -> neox half-split RoPE."""
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    angles = pos.float()[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@torch.no_grad()
def forward_with_taps(m: InklingRef, tokens: list[int], tap_visits=(5, 23, 41, 48)):
    T = len(tokens)
    pos = torch.arange(T, device=m.dev)
    emb = m.t["token_embd.weight"]
    rows = torch.from_numpy(np.ascontiguousarray(emb.data[tokens])).to(m.dev)
    cur = m.norm(deq_q8_0(rows, m.n_embd), "token_embd_norm.weight")

    taps: list[torch.Tensor] = []
    routing: list[torch.Tensor] = []
    blocks: list[torch.Tensor] = []
    visit = 0
    for il in range(m.core_start):
        if visit in tap_visits:
            taps.append(cur.clone())
        cur = m.layer(il, il, cur, blocks, pos, None, routing)
        visit += 1
    prelude, prelude_blocks = cur, list(blocks)
    for pss in range(m.n_passes):
        msgs = None
        if pss > 0:
            bank = blocks[len(prelude_blocks) + 1 :]
            bank_out = cur
            cur = prelude
            blocks = list(prelude_blocks)
        for ip in range(m.core_start, m.core_end):
            if pss > 0:
                msgs = m.mix(bank, bank_out, m.w(f"blk.{ip}.loop_msg_query.weight"))
            if visit in tap_visits:
                taps.append(cur.clone())
            cur = m.layer(ip, ip, cur, blocks, pos, msgs, routing)
            visit += 1
    for il in range(m.core_end, m.n_layer):
        if visit in tap_visits:
            taps.append(cur.clone())
        cur = m.layer(il, il, cur, blocks, pos, None, routing)
        visit += 1

    x = m.mix(blocks, cur, m.w("output_res_query.weight"))[:, 0]
    x = m.norm(x, "output_norm.weight")
    if visit in tap_visits:
        taps.append(x.clone())
    assert len(taps) == 4
    return torch.stack(taps, dim=1)  # [T, 4, E]


@torch.no_grad()
def dspark_draft(
    m: InklingRef,
    w: dict[str, torch.Tensor],
    lm_head: torch.Tensor,
    ctx_taps: torch.Tensor,
    anchor_tok: int,
    pos0: int,
    block_size: int = 8,
) -> list[int]:
    """Draft block_size - 1 tokens starting at pos0+1 given context taps [pos0, 4, E] and anchor_tok at pos0."""
    dev = ctx_taps.device
    H, HK, D = 24, 8, 128
    eps = 1e-6
    alpha = torch.softmax(w["fuse_weight"], dim=-1)  # [4, 4]
    ctx_pos = torch.arange(pos0, device=dev)
    blk_pos = torch.arange(pos0, pos0 + block_size, device=dev)

    # Embed anchor + 7 masks
    emb = m.t["token_embd.weight"]
    arow = torch.from_numpy(np.ascontiguousarray(emb.data[anchor_tok : anchor_tok + 1])).to(dev)
    aemb = m.norm(deq_q8_0(arow, m.n_embd), "token_embd_norm.weight")[0]
    x = w["mask_embed"][None, :].expand(block_size, -1).clone()
    x[0] = aemb

    local_layers = {0, 2}
    for l in range(3):
        p = f"layers.{l}."
        # Context K/V from target taps
        mixed = torch.einsum("tfe,f->te", ctx_taps, alpha[l])
        ctx = rmsnorm(mixed, w[p + "dflare_norm.weight"], eps)
        kc = (ctx @ w[p + "self_attn.k_proj_ctx.weight"].T).view(pos0, HK, D)
        vc = (ctx @ w[p + "self_attn.v_proj_ctx.weight"].T).view(pos0, HK, D)
        kc = rope_neox(rmsnorm(kc, w[p + "self_attn.k_norm_ctx.weight"], eps), ctx_pos)

        residual = x
        xn = rmsnorm(x, w[p + "input_layernorm.weight"], eps)
        q = (xn @ w[p + "self_attn.q_proj.weight"].T).view(block_size, H, D)
        kb = (xn @ w[p + "self_attn.k_proj.weight"].T).view(block_size, HK, D)
        vb = (xn @ w[p + "self_attn.v_proj.weight"].T).view(block_size, HK, D)
        q = rope_neox(rmsnorm(q, w[p + "self_attn.q_norm.weight"], eps), blk_pos)
        kb = rope_neox(rmsnorm(kb, w[p + "self_attn.k_norm.weight"], eps), blk_pos)

        # Local layers see last 1024 context tokens + block
        if l in local_layers and pos0 > 1024:
            kc, vc = kc[-1024:], vc[-1024:]
        k_all = torch.cat([kc, kb], dim=0).repeat_interleave(H // HK, dim=1)  # [S, H, D]
        v_all = torch.cat([vc, vb], dim=0).repeat_interleave(H // HK, dim=1)
        scores = torch.einsum("qhd,shd->qhs", q, k_all) / math.sqrt(D)
        att = torch.softmax(scores, dim=-1)
        out = torch.einsum("qhs,shd->qhd", att, v_all).reshape(block_size, H * D)
        x = residual + out @ w[p + "self_attn.o_proj.weight"].T

        residual = x
        xn = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
        g = xn @ w[p + "mlp.gate_proj.weight"].T
        u = xn @ w[p + "mlp.up_proj.weight"].T
        x = residual + (F.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T

    h_draft = rmsnorm(x, w["norm.weight"], eps)[1:]  # [7, E]

    # Recurrent Markov head
    f_rnn = rmsnorm(torch.einsum("fe,f->e", ctx_taps[-1], alpha[3]), w["markov_head.dflare_norm.weight"], eps)
    rank = 256
    rnn_w = w["markov_head.rnn_proj.weight"]  # [768, 5120]
    rnn_b = w["markov_head.rnn_proj.bias"]
    w_rec = rnn_w[:, : 2 * rank]
    w_ctx = rnn_w[:, 2 * rank :]
    hid_w = w["markov_head.hidden_out.weight"]  # [2304, 256]
    hid_b = w["markov_head.hidden_out.bias"]
    mw1 = w["markov_head.markov_w1.weight"]  # [201024, 256]

    state = torch.zeros(rank, device=dev)
    prev = anchor_tok
    drafts: list[int] = []
    for i in range(block_size - 1):
        ctx_in = torch.cat([h_draft[i], f_rnn], dim=-1)
        rec_in = torch.cat([state, mw1[prev]], dim=-1)
        raw = rnn_b + ctx_in @ w_ctx.T + rec_in @ w_rec.T
        gate = torch.sigmoid(raw[:rank])
        cand = torch.tanh(raw[rank : 2 * rank])
        latent = torch.tanh(raw[2 * rank :])
        state = gate * state + (1.0 - gate) * cand
        h_corr = h_draft[i] + latent @ hid_w.T + hid_b
        logits = 30.0 * torch.tanh((h_corr @ lm_head.T) / 30.0)
        prev = int(logits[: m.n_vocab].argmax().item())
        drafts.append(prev)
    return drafts


def main():
    gguf = Path("/home/louis/models/inkling/Inkling-2-Mini-UD-IQ4_XS-mainline.gguf")
    mtp = Path("/home/louis/models/inkling/mtp.safetensors")
    val = json.loads((REF_DIR / "results" / "2026-11-21-validate.llama.json").read_text())

    m = InklingRef(gguf)
    w = load_safetensors(mtp, m.dev)
    # Dequant lm_head once (n_vocab x 2304 in fp32 ~1.8 GB)
    out_t = m.t["output.weight"]
    lm_head = m._deq_gguf(out_t.data[: m.n_vocab], out_t.tensor_type).reshape(m.n_vocab, m.n_embd)

    total_acc = 0
    total_cycles = 0
    for run in val:
        seq = run["prompt_tokens"] + run["gen_tokens"]
        p_len = len(run["prompt_tokens"])
        taps = forward_with_taps(m, seq)
        # Simulate speculative decoding from p_len over the generated tokens
        pos = p_len
        acc_list = []
        while pos + 1 < len(seq):
            anchor = seq[pos - 1]
            # Wait: at position pos-1, target saw seq[0..pos-1] and emitted seq[pos] = anchor!
            # So context taps are seq[0..pos-1] (length pos), anchor is seq[pos] at position pos!
            ctx_taps = taps[:pos]
            anchor = seq[pos]
            drafts = dspark_draft(m, w, lm_head, ctx_taps, anchor, pos)
            # Compare against ground-truth target greedy continuation seq[pos+1 .. pos+7]
            acc = 0
            for k, d in enumerate(drafts):
                if pos + 1 + k < len(seq) and d == seq[pos + 1 + k]:
                    acc += 1
                else:
                    break
            acc_list.append(acc)
            total_acc += acc
            total_cycles += 1
            pos += 1 + acc
        mean_acc = sum(acc_list) / max(len(acc_list), 1)
        print(f"prompt={run['prompt'][:30]!r} cycles={len(acc_list)} acc={acc_list} mean_acc={mean_acc:.2f}/7 (mean_emit={1+mean_acc:.2f})")
    print(f"OVERALL: cycles={total_cycles}, mean_acc={total_acc/total_cycles:.2f}/7, mean_emitted={1+total_acc/total_cycles:.2f} tok/cycle")


if __name__ == "__main__":
    main()

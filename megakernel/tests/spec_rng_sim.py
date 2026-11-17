#!/usr/bin/env python3
"""Does keying the sampling random numbers by position bias the engine's speculative sampling? A CPU replica of the
sampled decode loop of qwen38_mk.cu: sample_targets (per-token rule with sampled drafts), main_control, the MTP draft
chain with the confidence gates (the default gates at 7 drafts; chain=1 or chain=2) and draft_sample, plus a policy
the engine does not have, blk=1 (block verification of sampled drafts), with the device's counter-based urand
(splitmix64) and its keys:
  acceptance of row i (position n + i): urand(seed, 2 (n + i)); residual / bonus draw: urand(seed, 2 (n + i) + 1);
  blk eta_i: urand(seed, 2 (n + i - 1)); a draft at absolute position x: urand(seed ^ 0xD1B54A32D192ED03, x).
The "fix" configurations salt the blk etas and the drafts with the launch's start position (mix); "tok_chain2_fix" is
chain=2 without drop gates (the engine's behaviour).

Target p(x_t | x_(t-1)) and draft q(x_t | x_(t-1), depth in the chain) are synthetic Markov kernels over V tokens, so
the exact law of the emitted sequence is known. Tests (chi-square goodness of fit, valid for any exact sampler):
  law:     per position t, the counts of x_t given x_(t-1) = a against p(. | a), summed over a.
  classes: (per-token rule only, where the class is fixed by the random numbers before t) the same for the positions
           whose verified draft had been drafted before (a re-draft: position keys reuse its number), for first drafts
           and for rows without a draft.
Results on 2026-10-17 (200k seeds per configuration; 1M for blk_pos / blk_fix / tok_chain2_fix): the per-token rule
with position keys (tok_pos, the default) fits everywhere, the re-draft class included (396k positions, p = 0.28);
blk=1 with position keys is biased (blk_pos_ng, no gates: p = 4e-10; with the default gates the effect is below what
1M seeds resolve) and fits with the salted keys; chain=2 with a drop gate is biased (tok_chain2: p = 2e-7;
tok_chain2_d1: total variation up to 0.07) and fits without drop gates.
The blk configurations are the record of this analysis.
usage: spec_rng_sim.py [--seeds N] [--configs a,b,..] [--L 12] [--V 8]   (needs scipy; ~20 s per 100k seeds and configuration)"""
import argparse
import math
import random
import sys

from scipy.stats import chi2 as chi2d

M64 = (1 << 64) - 1
DKEY = 0xD1B54A32D192ED03


def urand(seed, ctr):
    z = (seed + (ctr + 1) * 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    z ^= z >> 31
    return (z >> 40) / 16777216.0


def mix(n):  # the fix's salt: launch start position -> 64-bit word (any injective map would do)
    return ((n + 1) * 0xD6E8FEB86659FD93) & M64


def draw(p, u, excl=-1, positive=False):
    """cand_draw: inverse CDF over p (optionally without excl, renormalised). positive: the residual walks of
    sample_targets / sample_targets_blk, which skip entries <= 0, so when rounding leaves x >= 0 after the whole walk
    the draw is the last entry with a positive residual (cand_draw's: the last entry not excluded)"""
    s = sum(x for j, x in enumerate(p) if j != excl)
    x = u * s
    last = 0
    for j, pj in enumerate(p):
        if j == excl or (positive and not pj > 0):
            continue
        last = j
        x -= pj
        if x < 0:
            return j
    return last


def make_model(V, rs, alpha=0.25, mixw=0.3):
    def dirichlet(al):
        g = [max(rs.gammavariate(al, 1.0), 1e-12) for _ in range(V)]
        s = sum(g)
        return [x / s for x in g]
    P = [dirichlet(alpha) for _ in range(V)]  # peaked rows (LLM-like top-1 mass)
    Qb = []
    for a in range(V):  # draft = target pushed toward another Dirichlet draw
        o = dirichlet(alpha)
        Qb.append([(1 - mixw) * P[a][j] + mixw * o[j] for j in range(V)])

    def q_of(a, k):  # flatter with depth (the MTP's chain drifts from the catch-up's hidden state)
        e = 1.0 / (1.0 + 0.1 * k)
        r = [Qb[a][j] ** e for j in range(V)]
        s = sum(r)
        return [x / s for x in r]
    Q = {(a, k): q_of(a, k) for a in range(V) for k in range(8)}
    return P, Q


STATS = {}
G7_STOP = [0.0, 0.25, 0.25, 0.25, 0.3, 0.3, 0.4]
G7_DROP = [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]


def run_one(seed, cfg, P, Q, L, x0, nd_max=7):
    """One generation of positions 1..L. Returns (tokens[0..L], per position: (verified-draft flag, redraft flag))."""
    blk, fix, chain2, gates = cfg["blk"], cfg["fix"], cfg["chain2"], cfg["gates"]
    stop = G7_STOP if gates in (1, 3) else [0.0] * 7  # 3: the g7 stop gates without drop gates (the chain=2 fix)
    drop = G7_DROP if gates == 1 else [0.5] + [0.0] * 6 if gates == 2 else [0.0] * 7
    seq = [x0]
    cls = [None]
    drafted = set()  # positions some chain has drafted (its draft uniform consumed)
    # prefill's last launch (n = 0): the first token from p(. | x0) with the bonus key 2 * 0 + 1
    seq.append(draw(P[x0], urand(seed, 1)))
    cls.append((0, 0))
    n = 1
    tok = [seq[1]]
    dq = []      # dq[i]: proposal of draft tok[i + 1]
    dflag = []   # redraft flag of draft tok[i + 1]

    def chain(Pc, bonus):
        nonlocal tok, dq, dflag
        tok, dq, dflag = [bonus], [], []
        ch = 1.0
        nd_prod = 0
        for k in range(nd_max):
            q = Q[(tok[k], min(k, 7))]
            conf = max(q)
            pos = Pc + 1 + k
            key = (seed ^ DKEY ^ mix(Pc)) if fix else (seed ^ DKEY)
            d = draw(q, urand(key, pos))
            flag = pos in drafted
            drafted.add(pos)
            ch = ch * (q[d] if chain2 else conf)
            f = ch
            nd_prod = k if f < drop[k] else k + 1
            tok.append(d)
            dq.append(q)
            dflag.append(flag)
            if f < stop[k] or f < drop[k]:
                break
        tok, dq, dflag = tok[:nd_prod + 1], dq[:nd_prod], dflag[:nd_prod]

    chain(n, tok[0])
    while len(seq) <= L:
        T = len(tok)
        rows_p = [P[tok[i]] for i in range(T)]
        tg = [None] * T
        ekey = (seed ^ mix(n)) if fix else seed
        if blk and T >= 2:
            g = T - 1
            w, wt, tau = 1.0, 1.0, 0
            for i in range(g):
                p, q = rows_p[i], dq[i]
                if i >= 1:
                    r = sum(max(0.0, w * p[j] - q[j]) for j in range(len(p)))
                    den = r + 1.0 - w
                    h = r / den if den > 0 else 1.0
                    if urand(ekey, 2 * (n + i - 1)) < h:
                        tau, wt = i, w
                d = tok[i + 1]
                pd, qd = p[d], q[d]
                w = min(w * pd / qd, 1.0) if qd > 0 else 0.0
            if urand(ekey, 2 * (n + g - 1)) < w:
                tau, wt = g, w
            for i in range(tau):
                tg[i] = tok[i + 1]
            u = urand(seed, 2 * (n + tau) + 1)
            p = rows_p[tau]
            if tau < g:
                q = dq[tau]
                res = [max(0.0, wt * p[j] - q[j]) for j in range(len(p))]
            else:
                res = None
            if res is None or not sum(res) > 0:
                tg[tau] = draw(p, u)
            else:
                tg[tau] = draw(res, u, positive=True)
            last = tau
            i = tau + 1
            while i < T and tg[i - 1] == tok[i]:
                tg[i] = draw(rows_p[i], urand(seed, 2 * (n + i) + 1))
                last = i
                i += 1
        else:
            last = T - 1
            for i in range(T):
                p = rows_p[i]
                if i < T - 1:
                    d, q = tok[i + 1], dq[i]
                    if urand(ekey, 2 * (n + i)) * q[d] < p[d]:
                        tg[i] = d
                        continue
                    res = [max(0.0, p[j] - q[j]) for j in range(len(p))]
                    u = urand(seed, 2 * (n + i) + 1)
                    tg[i] = draw(p, u, d) if not sum(res) > 0 else draw(res, u, positive=True)
                    last = i
                    break
                tg[i] = draw(p, urand(seed, 2 * (n + i) + 1))
        # main_control
        a = 0
        while a < T - 1 and tg[a] == tok[a + 1]:
            a += 1
        STATS["launch"] = STATS.get("launch", 0) + 1
        STATS["T"] = STATS.get("T", 0) + T
        STATS["acc"] = STATS.get("acc", 0) + a
        for i in range(a + 1):
            seq.append(tg[i])
            if i < T - 1:
                cls.append((1, 1 if dflag[i] else 0))  # verified draft at this position (accepted or rejected)
            else:
                cls.append((0, 0))  # bonus row: no draft
        n = n + a + 1
        chain(n, tg[a])
    return seq[:L + 1], cls[:L + 1]


def exact_marg(P, x0, L):
    V = len(P)
    m = [[0.0] * V for _ in range(L + 1)]
    m[0][x0] = 1.0
    for t in range(1, L + 1):
        for a in range(V):
            for b in range(V):
                m[t][b] += m[t - 1][a] * P[a][b]
    return m


def gof(counts, probs):
    """chi-square GOF, pooling expected < 5 cells into one; returns (chi2, dof)."""
    n = sum(counts)
    if n == 0:
        return 0.0, 0
    cells, po, pe = [], 0, 0.0
    for c, p in zip(counts, probs):
        e = n * p
        if e >= 5:
            cells.append((c, e))
        else:
            po += c
            pe += e
    if pe > 0:
        cells.append((po, pe))
    if len(cells) < 2:
        return 0.0, 0
    return sum((o - e) ** 2 / e for o, e in cells), len(cells) - 1


CONFIGS = {
    "tok_pos": dict(blk=0, fix=0, chain2=0, gates=1),    # default policy, position keys
    "tok_fix": dict(blk=0, fix=1, chain2=0, gates=1),    # default policy, launch-salted keys
    "blk_pos": dict(blk=1, fix=0, chain2=0, gates=1),    # policy blk=1, position keys
    "blk_fix": dict(blk=1, fix=1, chain2=0, gates=1),    # policy blk=1, launch-salted keys (the fix)
    "blk_pos_ng": dict(blk=1, fix=0, chain2=0, gates=0),  # blk=1 without gates (fixed depth 7), position keys
    "blk_fix_ng": dict(blk=1, fix=1, chain2=0, gates=0),  # blk=1 without gates, the fix
    "tok_chain2": dict(blk=0, fix=0, chain2=1, gates=1),  # policy chain=2 (drop gate on the draft's own q')
    "tok_chain2_fix": dict(blk=0, fix=0, chain2=1, gates=3),  # chain=2 with the drop gates off (the engine's policy)
    "tok_d1": dict(blk=0, fix=0, chain2=0, gates=2),      # no stops, drop draft 1 below 0.5 (chain=1: top-1 q')
    "tok_chain2_d1": dict(blk=0, fix=0, chain2=1, gates=2),  # the same with chain=2 (the draft's own q')
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100000)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--model-seed", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.03)  # Dirichlet concentration of the rows (peaked: top-1 ~0.85)
    ap.add_argument("--mix", type=float, default=0.15)    # draft = (1 - mix) target + mix other
    ap.add_argument("--seed-mode", default="rand", help="rand: random 64-bit seeds (default); ap: an arithmetic progression")
    a = ap.parse_args()
    rs = random.Random(a.model_seed)
    P, Q = make_model(a.V, rs, a.alpha, a.mix)
    x0 = 0
    V, L = a.V, a.L
    em = exact_marg(P, x0, L)
    print(f"V={V} L={L} seeds={a.seeds} model-seed={a.model_seed} alpha={a.alpha} mix={a.mix}; acceptance of draft 1 "
          f"E[min(1,p/q)] = {sum(em[1][x] * sum(min(P[x][j], Q[(x, 0)][j]) for j in range(V)) for x in range(V)):.3f}, "
          f"mean top-1 q = {sum(em[1][x] * max(Q[(x, 0)]) for x in range(V)):.3f}")
    for name in a.configs.split(","):
        cfg = CONFIGS[name]
        cnt = [[[0] * V for _ in range(V)] for _ in range(L + 1)]  # [t][prev][tok]
        cls_cnt = {}  # (verified, redraft) -> [prev][tok]
        nred = 0
        STATS.clear()
        srng = random.Random(12345)
        for s in range(a.seeds):
            seed = srng.getrandbits(64) if a.seed_mode == "rand" else (0x1234567 + 7919 * s) & M64
            seq, cl = run_one(seed, cfg, P, Q, L, x0)
            for t in range(1, L + 1):
                cnt[t][seq[t - 1]][seq[t]] += 1
                if t >= 2:
                    k = cl[t]
                    c = cls_cnt.setdefault(k, [[0] * V for _ in range(V)])
                    c[seq[t - 1]][seq[t]] += 1
                    nred += k == (1, 1)
        # law test per position
        worst = (1.0, 0)
        tot_c, tot_d = 0.0, 0
        line = []
        for t in range(1, L + 1):
            c2, df = 0.0, 0
            for prev in range(V):
                x, d = gof(cnt[t][prev], P[prev])
                c2 += x
                df += d
            pv = chi2d.sf(c2, df) if df else 1.0
            tv = 0.5 * sum(abs(sum(cnt[t][pr][b] for pr in range(V)) / a.seeds - em[t][b]) for b in range(V))
            line.append(f"t{t}:p={pv:.3f}/TV={tv:.4f}")
            worst = min(worst, (pv, t))
            tot_c += c2
            tot_d += df
        print(f"== {name} {cfg}: {STATS['launch']} launches, mean T {STATS['T'] / max(1, STATS['launch']):.2f}, "
              f"accepted {STATS['acc']} / drafted {STATS['T'] - STATS['launch']}")
        print("   law per position: " + " ".join(line))
        print(f"   law: min p {worst[0]:.2e} at t{worst[1]} (Bonferroni x{L}: {min(1.0, worst[0] * L):.2e}); "
              f"pooled chi2 {tot_c:.1f} / dof {tot_d} (sum over positions, indicative only)")
        if not cfg["blk"]:
            for k, lab in (((1, 1), "verified re-drafts"), ((1, 0), "verified first drafts"), ((0, 0), "bonus rows")):
                if k not in cls_cnt:
                    continue
                c2, df, n = 0.0, 0, 0
                for prev in range(V):
                    x, d = gof(cls_cnt[k][prev], P[prev])
                    c2 += x
                    df += d
                    n += sum(cls_cnt[k][prev])
                pv = chi2d.sf(c2, df) if df else 1.0
                print(f"   class {lab:22s} n={n:7d} chi2 {c2:8.1f} dof {df:3d} p = {pv:.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

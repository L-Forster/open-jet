// Batched prefill: the prompt (all but its last token) is processed in chunks of up to `cap`
// tokens with ordinary kernel launches, mostly fp16 tensor-core GEMMs (prefill_gemm.cuh) against
// the same quantised weights the decode megakernel streams.  It leaves exactly the state the
// megakernel expects: K/V caches (trunk + MTP), committed delta-net state, conv ring, h history.
// Included inside qwen38_mk.cu's anonymous namespace.

#include "prefill_gemm.cuh"

__constant__ PBufs cP;

// The megakernel TU is built with -dlcm=cg (plain loads become strong L2 loads, which are very
// slow when scattered); prefill kernels read their inputs through the read-only path (__ldg),
// which is valid because every input was written by an earlier launch.
__device__ __forceinline__ float dequant_q4k_ldg(const uint8_t * row, int e) {
    const int b = e >> 8, w = e & 255, sb = w >> 5, l = w & 31;
    const uint8_t * blk = row + b * 144;
    const float d = __half2float(__ldg(reinterpret_cast<const __half *>(blk)));
    const float dmin = __half2float(__ldg(reinterpret_cast<const __half *>(blk + 2)));
    const uint8_t * sc = blk + 4;
    int s, m;
    if (sb < 4) {
        s = __ldg(sc + sb) & 63;
        m = __ldg(sc + sb + 4) & 63;
    } else {
        s = (__ldg(sc + sb + 4) & 0xF) | ((__ldg(sc + sb - 4) >> 6) << 4);
        m = (__ldg(sc + sb + 4) >> 4) | ((__ldg(sc + sb) >> 6) << 4);
    }
    const uint8_t qb = __ldg(blk + 16 + 32 * (sb >> 1) + l);
    const int qv = (sb & 1) ? (qb >> 4) : (qb & 0xF);
    return d * (float) s * (float) qv - dmin * (float) m;
}

// lane i holds element i of a 32-block starting at dst: store it as fp16 in GEMM order (saturating)
// (SITE: MK_PF_INFCHK's input site, 0 norm_quant, 1 the delta net's output, 2 the MTP block's input)
template <int SITE = 0> __device__ __forceinline__ void hstore32(float v, __half * dst) {
#if MK_PF_INFCHK
    pf::chk_clamp(v, pf::CHK_IN_NORM + SITE);
#endif
    dst[pf::hperm(threadIdx.x & 31)] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
}

__device__ __forceinline__ float block_sum256(float v, float * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < 8 ? red[lane] : 0.f;
    return wsum(r);
}

// ET: token_embd's type (ModelW::embd_t; T_Q4K, or T_Q6K / T_Q8_0: dequant_embd, the decode path's values)
template <int ET>
__device__ __forceinline__ float embd_elem_pf(const uint8_t * row, int e) {
    if constexpr (ET == T_Q4K) return dequant_q4k_ldg(row, e);
    else return dequant_embd<ET>(row, e);
}
template <int ET>
__global__ void pk_embed(int T) { MK_VIEW_INIT();
    const int t = blockIdx.x;
    const uint8_t * row = cW.embd + (size_t) cP.tok[t] * embd_rb<ET>();
    for (int e = threadIdx.x; e < D; e += blockDim.x) cP.x[(size_t) t * D + e] = embd_elem_pf<ET>(row, e);
}

// pk_embed for a chunk with image rows (MKImage): a token with erow >= 0 takes that row of cP.emb (the vision encoder's
// fp32 embedding, which llama.cpp feeds as is) instead of its token_embd row; the other tokens as pk_embed
template <int ET>
__global__ void pk_embed_img(int T) { MK_VIEW_INIT();
    const int t = blockIdx.x, r = __ldg(cP.erow + t);
    if (r < 0) {
        const uint8_t * row = cW.embd + (size_t) cP.tok[t] * embd_rb<ET>();
        for (int e = threadIdx.x; e < D; e += blockDim.x) cP.x[(size_t) t * D + e] = embd_elem_pf<ET>(row, e);
        return;
    }
    const float4 * src = reinterpret_cast<const float4 *>(cP.emb + (size_t) r * D);
    float4 * dst = reinterpret_cast<float4 *>(cP.x + (size_t) t * D);
    for (int e = threadIdx.x; e < D / 4; e += blockDim.x) dst[e] = __ldg(src + e);
}

// RMS-norm (weight w) of rows of src [T][D] -> fp16 xh (K = D); rs per token
__global__ void __launch_bounds__(256) pk_norm_quant(const float * src, const float * w) { MK_VIEW_INIT();
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float * x = src + (size_t) t * D;
    float ss = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = __ldg(x + e);
        ss += v * v;
    }
    const float rs = rsqrtf(block_sum256(ss, red) / (float) D + EPS);
    if (threadIdx.x == 0) cP.rs[t] = rs;
    for (int j = warp; j < D / 32; j += 8) {
        const int e = 32 * j + lane;
        hstore32(__ldg(x + e) * rs * __ldg(w + e), cP.xh + (size_t) t * D + 32 * j);
    }
}

// MK_K8 bit 2: pk_norm_quant with x read once. Thread tid reads e = tid + 256 i (i = 0 .. D / 256 - 1) in the sum of squares
// and, as warp w = tid >> 5 / lane, writes the 32-blocks j = w + 8 i (e = 32 j + lane = the same e): the values stay in
// registers. Same sums in the same order, same products: bit-identical to pk_norm_quant.
__global__ void __launch_bounds__(256) pk_norm_quant1(const float * src, const float * w) { MK_VIEW_INIT();
    __shared__ float red[8];
    constexpr int NE = D / 256;
    static_assert(D % 256 == 0, "pk_norm_quant1");
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float * x = src + (size_t) t * D;
    float v[NE];
#pragma unroll
    for (int i = 0; i < NE; i++) v[i] = __ldg(x + threadIdx.x + 256 * i);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NE; i++) ss += v[i] * v[i];
    const float rs = rsqrtf(block_sum256(ss, red) / (float) D + EPS);
    if (threadIdx.x == 0) cP.rs[t] = rs;
#pragma unroll
    for (int i = 0; i < NE; i++) {
        const int j = warp + 8 * i, e = 32 * j + lane;
        hstore32(v[i] * rs * __ldg(w + e), cP.xh + (size_t) t * D + 32 * j);
    }
}

// delta-net alpha / beta: ab[c][t][r] = rs[t] * sum_{k in split c} Wab[r][k] x[t][k] (attn_norm folded
// into Wab).  Block = 64 tokens x 96 rows x one of AB_SPLITS K-splits; thread = 4 tokens x 6 rows.
constexpr int AB_SPLITS = 8, AB_KS = D / AB_SPLITS;
// Q8_0 alpha / beta (abq != nullptr): weights (q * d) * attn_norm formed while filling sw, the folded fp32 values
__global__ void __launch_bounds__(256) pk_ab(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T) { MK_VIEW_INIT();
    __shared__ __align__(16) float sx[32][64 + 4]; // read as float4
    __shared__ __align__(16) float sw[32][96 + 4];
    const int tid = threadIdx.x, tg = tid & 15, rg = tid >> 4; // tokens 4 tg .. , rows 6 rg ..
    const int t0 = blockIdx.x * 64, k00 = blockIdx.y * AB_KS;
    float acc[4][6];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 6; j++) acc[i][j] = 0.f;
    for (int k0 = k00; k0 < k00 + AB_KS; k0 += 32) {
        for (int i = tid; i < 64 * 8; i += 256) { // x tile, transposed to [k][token]
            const int tt = i >> 3, kk = (i & 7) * 4;
            const float4 v = t0 + tt < T ? __ldg(reinterpret_cast<const float4 *>(cP.x + (size_t) (t0 + tt) * D + k0 + kk))
                                         : make_float4(0.f, 0.f, 0.f, 0.f);
            sx[kk][tt] = v.x; sx[kk + 1][tt] = v.y; sx[kk + 2][tt] = v.z; sx[kk + 3][tt] = v.w;
        }
        for (int i = tid; i < 96 * 8; i += 256) {
            const int rr = i >> 3, kk = (i & 7) * 4;
            float4 v;
            if (abq) {
                const int qw = __ldg(reinterpret_cast<const int *>(abq + (size_t) rr * D + k0 + kk));
                const float d = __half2float(abd[(size_t) rr * (D / 32) + ((k0 + kk) >> 5)]);
                const float4 f = __ldg(reinterpret_cast<const float4 *>(fn + k0 + kk));
                v.x = ((float) (int8_t) (qw) * d) * f.x;
                v.y = ((float) (int8_t) (qw >> 8) * d) * f.y;
                v.z = ((float) (int8_t) (qw >> 16) * d) * f.z;
                v.w = ((float) (int8_t) (qw >> 24) * d) * f.w;
            } else {
                v = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) rr * D + k0 + kk));
            }
            sw[kk][rr] = v.x; sw[kk + 1][rr] = v.y; sw[kk + 2][rr] = v.z; sw[kk + 3][rr] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (int kk = 0; kk < 32; kk++) {
            const float4 xv = *reinterpret_cast<const float4 *>(&sx[kk][4 * tg]);
            float wv[6];
#pragma unroll
            for (int j = 0; j < 6; j++) wv[j] = sw[kk][6 * rg + j];
            const float xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 6; j++) acc[i][j] = fmaf(xa[i], wv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int t = t0 + 4 * tg + i;
        if (t >= T) continue;
        const float rs = __ldg(cP.rs + t);
#pragma unroll
        for (int j = 0; j < 6; j++) cP.ab[((size_t) blockIdx.y * cP.cap + t) * 96 + 6 * rg + j] = acc[i][j] * rs;
    }
}

// pk_ab with blocks of 128 threads of 8 tokens x 6 rows (MK_PFX bit 16 off; Q8: abq != nullptr). Every output is the same
// chain as pk_ab's (from 0, fmaf(x, w, acc) over its split's k in order, times rs; w formed by the same expression), but
// a k step costs 2 x float4 + 3 x float2 shared loads per 48 FMAs (pk_ab: 1 + 3 per 24; it is bound by its shared-memory
// wavefronts), the x tile is stored without bank conflicts, and the next tile's global loads are in flight during the
// FMAs (pk_ab loads, stores, then computes). Thread (tg, rg): tokens 4 tg + {0..3} and 32 + 4 tg + {0..3} (8 lanes read 128
// contiguous bytes), rows 6 rg + {0..5}.
template <bool Q8>
__global__ void __launch_bounds__(128, 4) pk_ab2(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T) { MK_VIEW_INIT();
    __shared__ __align__(16) float sx[32][64 + 4]; // [k][token]
    __shared__ __align__(16) float sw[32][96 + 4]; // [k][row]
    const int tid = threadIdx.x, tg = tid & 7, rg = tid >> 3;
    const int t0 = blockIdx.x * 64, k00 = blockIdx.y * AB_KS;
    const int xt = tid & 63, xk = 4 * (tid >> 6); // x fill: token xt, k quads xk + 8 m (lanes along tokens)
    const int wk = 4 * (tid & 7), wr = tid >> 3;  // w fill: rows wr + 16 m, k quad wk (as pk_ab)
    float acc[8][6];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 6; j++) acc[i][j] = 0.f;
    float4 px[4], pw[Q8 ? 1 : 6]; // Q8: pw[0] = fn of the quad
    int pq[Q8 ? 6 : 1];
    __half pd[Q8 ? 6 : 1];
    const bool xin = t0 + xt < T;
    auto load = [&](int k0) {
#pragma unroll
        for (int m = 0; m < 4; m++)
            px[m] = xin ? __ldg(reinterpret_cast<const float4 *>(cP.x + (size_t) (t0 + xt) * D + k0 + xk + 8 * m))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int m = 0; m < 6; m++) {
            const int rr = wr + 16 * m;
            if constexpr (Q8) {
                pq[m] = __ldg(reinterpret_cast<const int *>(abq + (size_t) rr * D + k0 + wk));
                pd[m] = abd[(size_t) rr * (D / 32) + ((k0 + wk) >> 5)];
            } else {
                pw[m] = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) rr * D + k0 + wk));
            }
        }
        if constexpr (Q8) pw[0] = __ldg(reinterpret_cast<const float4 *>(fn + k0 + wk));
    };
    auto store = [&] {
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int kk = xk + 8 * m;
            sx[kk][xt] = px[m].x; sx[kk + 1][xt] = px[m].y; sx[kk + 2][xt] = px[m].z; sx[kk + 3][xt] = px[m].w;
        }
#pragma unroll
        for (int m = 0; m < 6; m++) {
            const int rr = wr + 16 * m;
            float4 v;
            if constexpr (Q8) {
                const int qw = pq[m];
                const float d = __half2float(pd[m]);
                const float4 f = pw[0];
                v.x = ((float) (int8_t) (qw) * d) * f.x;
                v.y = ((float) (int8_t) (qw >> 8) * d) * f.y;
                v.z = ((float) (int8_t) (qw >> 16) * d) * f.z;
                v.w = ((float) (int8_t) (qw >> 24) * d) * f.w;
            } else {
                v = pw[m];
            }
            sw[wk][rr] = v.x; sw[wk + 1][rr] = v.y; sw[wk + 2][rr] = v.z; sw[wk + 3][rr] = v.w;
        }
    };
    load(k00);
    for (int k0 = k00; k0 < k00 + AB_KS; k0 += 32) {
        store();
        __syncthreads();
        if (k0 + 32 < k00 + AB_KS) load(k0 + 32);
#pragma unroll 4
        for (int kk = 0; kk < 32; kk++) {
            const float4 xa = *reinterpret_cast<const float4 *>(&sx[kk][4 * tg]);
            const float4 xb = *reinterpret_cast<const float4 *>(&sx[kk][32 + 4 * tg]);
            const float2 w0 = *reinterpret_cast<const float2 *>(&sw[kk][6 * rg]);
            const float2 w1 = *reinterpret_cast<const float2 *>(&sw[kk][6 * rg + 2]);
            const float2 w2 = *reinterpret_cast<const float2 *>(&sw[kk][6 * rg + 4]);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            const float wv[6] = {w0.x, w0.y, w1.x, w1.y, w2.x, w2.y};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 6; j++) acc[i][j] = fmaf(xv[i], wv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int t = t0 + (i < 4 ? 4 * tg + i : 32 + 4 * tg + i - 4);
        if (t >= T) continue;
        const float rs = __ldg(cP.rs + t);
#pragma unroll
        for (int j = 0; j < 6; j++) cP.ab[((size_t) blockIdx.y * cP.cap + t) * 96 + 6 * rg + j] = acc[i][j] * rs;
    }
}

// conv1d (kernel 4, causal, continuing from the conv ring) + silu, l2-norm of q / k, log g / beta
__global__ void __launch_bounds__(256) pk_dn_prep(int T, int n0, int dl, const float * convw, const float * dt,
                                                  const float * A) { MK_VIEW_INIT();
    const int t = blockIdx.x, pos = n0 + t, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float * ring = cB.conv + (size_t) dl * CONV_RING * QKV;
    auto raw = [&](int j, int c) -> float { // input of position pos - 3 + j
        const int tt = t - 3 + j, p = pos - 3 + j;
        if (tt >= 0) return __ldg(cP.y0 + (size_t) tt * QKV + c);
        if (p >= 0) return __ldg(ring + (size_t) (p % CONV_RING) * QKV + c);
        return 0.f;
    };
    auto conv = [&](int c) {
        const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + c * 4));
        return silu(w.x * raw(0, c) + w.y * raw(1, c) + w.z * raw(2, c) + w.w * raw(3, c));
    };
    __half * out = cP.dq + (size_t) t * QKV;
    for (int hh = warp; hh < 2 * NKH; hh += 8) { // q heads then k heads
        float v[4];
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = conv(hh * 128 + lane + 32 * i);
            ss += v[i] * v[i];
        }
        float sc = rsqrtf(fmaxf(wsum(ss), EPS * EPS));
        if (hh < NKH) sc *= 0.08838834764831845f; // 1/sqrt(128)
#pragma unroll
        for (int i = 0; i < 4; i++) out[hh * 128 + lane + 32 * i] = __float2half_rn(v[i] * sc);
    }
    for (int c = 2 * NKH * 128 + threadIdx.x; c < QKV; c += 256) out[c] = __float2half_rn(conv(c));
    if (threadIdx.x < NVH) {
        const int h = threadIdx.x;
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int c = 0; c < AB_SPLITS; c++) {
            al += __ldg(cP.ab + ((size_t) c * cP.cap + t) * 96 + h);
            be += __ldg(cP.ab + ((size_t) c * cP.cap + t) * 96 + NVH + h);
        }
        float sp = al + __ldg(dt + h);
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        cP.gb[(size_t) t * 96 + h] = sp * __ldg(A + h); // log g: finite even when g underflows
        cP.gb[(size_t) t * 96 + NVH + h] = 1.0f / (1.0f + expf(-be));
    }
}

// MK_K8 bit 4: pk_dn_prep for NT tokens per block (t0 = NT blockIdx.x ..): each thread loads the NT + 3 conv inputs of a
// channel once (pk_dn_prep's blocks re-read every input row 4 times through L2) and forms the NT outputs with the same
// expressions (same products, sums and order per element): bit-identical to pk_dn_prep.
template <int NT>
__global__ void __launch_bounds__(256) pk_dn_prep_t(int T, int n0, int dl, const float * convw, const float * dt,
                                                    const float * A) { MK_VIEW_INIT();
    const int t0 = blockIdx.x * NT, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nt = min(NT, T - t0);
    const float * ring = cB.conv + (size_t) dl * CONV_RING * QKV;
    auto raw = [&](int tt, int c) -> float { // input of chunk row tt (position n0 + tt; tt < 0: the conv ring)
        const int p = n0 + tt;
        if (tt >= 0) return __ldg(cP.y0 + (size_t) tt * QKV + c);
        if (p >= 0) return __ldg(ring + (size_t) (p % CONV_RING) * QKV + c);
        return 0.f;
    };
    // conv outputs of channel c for rows t0 .. t0 + NT - 1 (rows >= T: inputs 0, outputs unused)
    auto conv = [&](int c, float (&o)[NT]) {
        const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + c * 4));
        float r[NT + 3];
#pragma unroll
        for (int k = 0; k < NT + 3; k++) r[k] = k - 3 < nt ? raw(t0 - 3 + k, c) : 0.f;
#pragma unroll
        for (int u = 0; u < NT; u++) o[u] = silu(w.x * r[u] + w.y * r[u + 1] + w.z * r[u + 2] + w.w * r[u + 3]);
    };
    for (int hh = warp; hh < 2 * NKH; hh += 8) { // q heads then k heads
        float v[4][NT];
#pragma unroll
        for (int i = 0; i < 4; i++) conv(hh * 128 + lane + 32 * i, v[i]);
#pragma unroll
        for (int u = 0; u < NT; u++) {
            if (u >= nt) break;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) ss += v[i][u] * v[i][u];
            float sc = rsqrtf(fmaxf(wsum(ss), EPS * EPS));
            if (hh < NKH) sc *= 0.08838834764831845f; // 1/sqrt(128)
            __half * out = cP.dq + (size_t) (t0 + u) * QKV;
#pragma unroll
            for (int i = 0; i < 4; i++) out[hh * 128 + lane + 32 * i] = __float2half_rn(v[i][u] * sc);
        }
    }
    for (int c = 2 * NKH * 128 + threadIdx.x; c < QKV; c += 256) {
        float o[NT];
        conv(c, o);
#pragma unroll
        for (int u = 0; u < NT; u++)
            if (u < nt) cP.dq[(size_t) (t0 + u) * QKV + c] = __float2half_rn(o[u]);
    }
    if (threadIdx.x < NVH) {
        const int h = threadIdx.x;
        for (int u = 0; u < nt; u++) {
            const int t = t0 + u;
            float al = 0.f, be = 0.f;
#pragma unroll
            for (int c = 0; c < AB_SPLITS; c++) {
                al += __ldg(cP.ab + ((size_t) c * cP.cap + t) * 96 + h);
                be += __ldg(cP.ab + ((size_t) c * cP.cap + t) * 96 + NVH + h);
            }
            float sp = al + __ldg(dt + h);
            sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
            cP.gb[(size_t) t * 96 + h] = sp * __ldg(A + h); // log g: finite even when g underflows
            cP.gb[(size_t) t * 96 + NVH + h] = 1.0f / (1.0f + expf(-be));
        }
    }
}

// raw qkv rows of the chunk's last positions -> conv ring (after pk_dn_prep consumed the old ones). F4 (MK_PFX bit 128 off, 256
// threads): a row as QKV / 1024 float4 per thread, every load issued before the first store (the float loop: ~40 dependent
// round trips per thread); the same bytes to the same slots.
template <bool F4>
__global__ void pk_conv_ring(int T, int n0, int dl) { MK_VIEW_INIT();
    float * ring = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int nt = min(T, CONV_RING), t = T - nt + blockIdx.x;
    if constexpr (F4) {
        static_assert(QKV % 1024 == 0, "pk_conv_ring F4");
        constexpr int NV = QKV / 1024;
        const float4 * src = reinterpret_cast<const float4 *>(cP.y0 + (size_t) t * QKV) + threadIdx.x;
        float4 * dst = reinterpret_cast<float4 *>(ring + (size_t) ((n0 + t) % CONV_RING) * QKV) + threadIdx.x;
        float4 v[NV];
#pragma unroll
        for (int i = 0; i < NV; i++) v[i] = __ldg(src + 256 * i);
#pragma unroll
        for (int i = 0; i < NV; i++) dst[256 * i] = v[i];
    } else {
        for (int c = threadIdx.x; c < QKV; c += blockDim.x)
            ring[(size_t) ((n0 + t) % CONV_RING) * QKV + c] = __ldg(cP.y0 + (size_t) t * QKV + c);
    }
}

// Checkpoint at position P inside the chunk at n0 (pf_run): the conv ring as it is after position P - 1, i.e. slot
// p % CONV_RING = raw qkv of p for p in [P - CONV_RING, P): the chunk's rows (y0) for p >= n0, else the ring before
// the chunk (launched before pk_conv_ring overwrites it; slots of p < 0 are copied as they are, never read).
__global__ void pk_ck_conv(int n0, int P, int dl, float * dst) { MK_VIEW_INIT();
    const int p = P - CONV_RING + (int) blockIdx.x, s = (p % CONV_RING + CONV_RING) % CONV_RING;
    const float * src = p >= n0 ? cP.y0 + (size_t) (p - n0) * QKV : cB.conv + ((size_t) dl * CONV_RING + s) * QKV;
    for (int c = threadIdx.x; c < QKV; c += blockDim.x) dst[(size_t) s * QKV + c] = __ldg(src + c);
}
// ... and the h ring: slot p % H_RING = final h of p for p in lo .. lo + gridDim.x - 1, from the ring before the chunk
// (from_ring: launched before pk_final_h overwrites it) or from the chunk's h rows (after pk_final_h).  dst: the
// checkpoint's ring (mapped host memory, or device memory with MK_CK_DEVICE=1)
__global__ void pk_ck_hist(int lo, int n0, int from_ring, float * dst) { MK_VIEW_INIT();
    const int p = lo + (int) blockIdx.x, s = (p % H_RING + H_RING) % H_RING;
    const float * src = from_ring ? cB.h_hist + (size_t) s * D : cP.h + (size_t) (p - n0 + 1) * D;
    for (int e = threadIdx.x; e < D; e += blockDim.x) dst[(size_t) s * D + e] = __ldg(src + e);
}

// Gated delta rule over the chunk in the chunked (WY) form on tensor cores.  Per 32-token chunk,
// with a_i = sum_{j <= i} log g_j, gamma_i = exp(a_i), state S (128 x 128, rows = key dim) at the
// chunk start and u_i = beta_i (v_i - g_i S_{i-1}^T k_i) the delta-rule updates:
//   A_ij = beta_i exp(a_i - a_j) k_i.k_j (j < i),  T = (I + A)^-1
//   U    = T diag(beta) V - T diag(beta gamma) (K S)
//   O    = diag(gamma) Q S + P U,  P_ij = exp(a_i - a_j) q_i.k_j (j <= i)
//   S'   = gamma_31 S + K^T diag(exp(a_31 - a_i)) U
// Value columns are independent: block = (value head, 64 columns), warp = 8 columns, whose 128 x 8
// slice of S lives in fp32 MMA accumulators; it becomes an fp16 B operand in registers through
// movmatrix (as do K S and U), so the state never touches shared memory.  K K^T / Q K^T and the
// triangular solve (one warp) are per block.  fp16 operands, fp32 accumulation; inputs of the next
// chunk are prefetched into registers.
constexpr int DC = 32, DC_LD = 136, DC_LV = 72, DC_LP = 40, DC_LA = 36;
__device__ __forceinline__ unsigned movt(unsigned x) {
    unsigned y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
__device__ __forceinline__ unsigned pk2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm2t(unsigned (&r)[2], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma4(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    mma16816(c, a[0], a[1], a[2], a[3], b0, b1);
}

// Rows t0 .. t0 + T - 1 of the chunk (32-token chunks from row t0): the state is read from st_in and written to st_out
// (one layer's [NVH][128 cols][128 rows]). Normally both are the layer's live state and t0 = 0; a checkpoint at row
// t0 + T splits the scan there (the first part writes a staging copy that the second part reads; pf_run).
__global__ void __launch_bounds__(256, 2) pk_dn_scan(int T, int t0, const float * st_in, float * st_out) { MK_VIEW_INIT();
    __shared__ __align__(16) __half sK[DC][DC_LD], sQ[DC][DC_LD], sV[DC][DC_LV];
    __shared__ __align__(16) __half sT0[DC][DC_LP], sT1[DC][DC_LP], sP[DC][DC_LP];
    __shared__ __align__(16) float sAT[DC][DC_LA]; // A transposed: sAT[j][i] = A_ij
    __shared__ float sa[DC], sb[DC], sg[DC], sd[DC];
    const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int col0 = blockIdx.y * 64 + 8 * w; // this warp's 8 value columns
    const float * Si = st_in + (size_t) h * 128 * 128; // element (col, row) at dn_sidx(col, row)
    float S[8][4];                                    // rows 16 mt + g (+8), columns col0 + 2 t4 (+1)
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        const int c = col0 + 2 * t4, rw = 16 * mt + g;
        S[mt][0] = __ldcg(Si + dn_sidx(c, rw));
        S[mt][1] = __ldcg(Si + dn_sidx(c + 1, rw));
        S[mt][2] = __ldcg(Si + dn_sidx(c, rw + 8));
        S[mt][3] = __ldcg(Si + dn_sidx(c + 1, rw + 8));
    }

    uint4 pqk[4], pv;
    float pbeta = 0.f, plg = 0.f;
    auto fetch = [&](int tb) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = tid + 256 * k, tt = i >> 5, c = i & 31, t = tb + tt;
            pqk[k] = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) (t0 + t) * QKV + (c < 16 ? 0 : NKH * 128) + kh * 128) + (c & 15))
                           : make_uint4(0, 0, 0, 0);
        }
        const int t = tb + (tid >> 3);
        pv = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) (t0 + t) * QKV + 2 * NKH * 128 + h * 128 + blockIdx.y * 64) + (tid & 7))
                   : make_uint4(0, 0, 0, 0);
        if (w == 0) {
            const int tl = tb + lane;
            pbeta = tl < T ? __ldg(cP.gb + (size_t) (t0 + tl) * 96 + NVH + h) : 0.f;
            plg = tl < T ? __ldg(cP.gb + (size_t) (t0 + tl) * 96 + h) : 0.f;
        }
    };
    auto put = [&] {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = tid + 256 * k, tt = i >> 5, c = i & 31;
            *reinterpret_cast<uint4 *>(c < 16 ? &sQ[tt][8 * c] : &sK[tt][8 * (c - 16)]) = pqk[k];
        }
        *reinterpret_cast<uint4 *>(&sV[tid >> 3][8 * (tid & 7)]) = pv;
        if (w == 0) { // cumulative log gates (padding rows: log g = 0, beta = 0)
            float a = plg;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, a, o);
                if (lane >= o) a += y;
            }
            const float al = __shfl_sync(0xffffffffu, a, 31);
            sa[lane] = a;
            sb[lane] = pbeta;
            sg[lane] = __expf(a);
            sd[lane] = __expf(al - a);
        }
    };

    fetch(0);
    for (int tb = 0; tb < T; tb += DC) {
        __syncthreads(); // previous chunk done with shared memory
        put();
        __syncthreads();
        if (tb + DC < T) fetch(tb + DC);

        // K K^T (warps 0-3) and Q K^T (warps 4-7): 2 n8 tiles each -> A (fp32, transposed) and P (fp16)
        {
            const int isq = w >> 2, mt = (w >> 1) & 1, n0 = 16 * (w & 1);
            const __half (*X)[DC_LD] = isq ? sQ : sK;
            float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < 8; s++) {
                unsigned a[4], b[4];
                ldsm4(a, &X[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                ldsm4(b, &sK[n0 + (lane & 7) + ((lane >> 4) << 3)][16 * s + ((lane >> 3) & 1) * 8]);
                mma4(c0, a, b[0], b[1]);
                mma4(c1, a, b[2], b[3]);
            }
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int i = 16 * mt + g + ((e & 2) ? 8 : 0), j = n0 + 8 * (e >> 2) + 2 * t4 + (e & 1);
                const float v = (e >> 2) ? c1[e & 3] : c0[e & 3];
                const float f = j <= i ? __expf(sa[i] - sa[j]) * v : 0.f;
                if (isq) sP[i][j] = __float2half_rn(f);
                else sAT[j][i] = j < i ? sb[i] * f : 0.f;
            }
        }
        __syncthreads();

        // warp 0: T = (I + A)^-1 (lane = column, right-looking); every warp: K S and Q S for its columns
        if (w == 0) {
            float x[DC];
#pragma unroll
            for (int i = 0; i < DC; i++) x[i] = i == lane ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < DC - 1; j++) {
#pragma unroll
                for (int i4 = (j + 1) / 4; i4 < DC / 4; i4++) {
                    const float4 a4 = *reinterpret_cast<const float4 *>(&sAT[j][4 * i4]);
                    const float av[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
                    for (int q = 0; q < 4; q++)
                        if (4 * i4 + q > j) x[4 * i4 + q] = fmaf(-av[q], x[j], x[4 * i4 + q]);
                }
            }
            const float bc = sb[lane], bgc = bc * sg[lane];
#pragma unroll
            for (int i = 0; i < DC; i++) {
                sT0[i][lane] = __float2half_rn(x[i] * bc);
                sT1[i][lane] = __float2half_rn(x[i] * bgc);
            }
        }
        float ks[2][4] = {}, qs[2][4] = {};
#pragma unroll
        for (int s = 0; s < 8; s++) { // k = key dims 16 s .. : B = S tile s
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(S[s][e], pf::CHK_DN_S);
#endif
            const unsigned b0 = movt(pk2(S[s][0], S[s][1])), b1 = movt(pk2(S[s][2], S[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                ldsm4(a, &sK[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(ks[mt], a, b0, b1);
                ldsm4(a, &sQ[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(qs[mt], a, b0, b1);
            }
        }
        __syncthreads();

        // U = T diag(beta) V - T diag(beta gamma) (K S)
        float u[2][4] = {}, ws[2][4] = {};
#pragma unroll
        for (int s = 0; s < 2; s++) { // k = tokens 16 s ..
            unsigned bv[2];
            ldsm2t(bv, &sV[16 * s + (lane & 15)][8 * w]);
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(ks[s][e], pf::CHK_DN_KS);
#endif
            const unsigned bk0 = movt(pk2(ks[s][0], ks[s][1])), bk1 = movt(pk2(ks[s][2], ks[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                if (s > mt) continue; // T is lower triangular
                unsigned a[4];
                ldsm4(a, &sT0[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(u[mt], a, bv[0], bv[1]);
                ldsm4(a, &sT1[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(ws[mt], a, bk0, bk1);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) u[mt][e] -= ws[mt][e];

        // O = diag(gamma) Q S + P U
        float o[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            const float ga = sg[16 * mt + g], gb_ = sg[16 * mt + g + 8];
            o[mt][0] = ga * qs[mt][0]; o[mt][1] = ga * qs[mt][1];
            o[mt][2] = gb_ * qs[mt][2]; o[mt][3] = gb_ * qs[mt][3];
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(u[s][e], pf::CHK_DN_U);
#endif
            const unsigned bu0 = movt(pk2(u[s][0], u[s][1])), bu1 = movt(pk2(u[s][2], u[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                if (s > mt) continue;
                unsigned a[4];
                ldsm4(a, &sP[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(o[mt], a, bu0, bu1);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int t = tb + 16 * mt + g + 8 * hf;
                if (t < T)
                    *reinterpret_cast<float2 *>(cP.y2 + (size_t) (t0 + t) * VD + h * 128 + col0 + 2 * t4) =
                        make_float2(o[mt][2 * hf], o[mt][2 * hf + 1]);
            }

        // S = gamma_31 S + K^T diag(exp(a_31 - a)) U
        const float gl = sg[DC - 1];
#pragma unroll
        for (int mt = 0; mt < 8; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) S[mt][e] *= gl;
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const float da = sd[16 * s + g], db = sd[16 * s + g + 8];
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(u[s][e] * (e < 2 ? da : db), pf::CHK_DN_UD);
#endif
            const unsigned b0 = movt(pk2(u[s][0] * da, u[s][1] * da)), b1 = movt(pk2(u[s][2] * db, u[s][3] * db));
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                unsigned a[4];
                ldsm4t(a, &sK[16 * s + (lane & 7) + ((lane >> 4) << 3)][16 * mt + ((lane >> 3) & 1) * 8]);
                mma4(S[mt], a, b0, b1);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        float * p = st_out + (size_t) h * 128 * 128;
        const int c = col0 + 2 * t4, rw = 16 * mt + g;
        p[dn_sidx(c, rw)] = S[mt][0];
        p[dn_sidx(c + 1, rw)] = S[mt][1];
        p[dn_sidx(c, rw + 8)] = S[mt][2];
        p[dn_sidx(c + 1, rw + 8)] = S[mt][3];
    }
}

// pk_dn_scan's S-independent per-chunk matrices for every 32-row chunk of rows 0 .. T - 1 at once (MK_PFX bit 2 off):
// per value head h and chunk c the fp16 [32][32] matrices T0 = T diag(beta), T1 = T diag(beta gamma) and P at
// tri + ((c NVH + h) 3 + {0, 1, 2}) 1024. Block = (key head, chunk): K K^T and Q K^T once for the DN_GV value heads that
// share the key head (pk_dn_scan makes them per value head and column half with the same MMAs: the same values), then
// per value head pk_dn_scan's exp scaling, triangular solve and fp16 roundings. Rows >= T are zero, as pk_dn_scan's.
constexpr int DN_GV = NVH / NKH; // value heads h = kh + NKH j of key head kh
__global__ void __launch_bounds__(256) pk_dn_tri(int T, __half * tri) { MK_VIEW_INIT();
    __shared__ __align__(16) __half sK[DC][DC_LD], sQ[DC][DC_LD];
    __shared__ __align__(16) float sAT[DN_GV][DC][DC_LA]; // A transposed per value head: sAT[v][j][i] = A_ij
    __shared__ float sa[DN_GV][DC], sb[DN_GV][DC], sg[DN_GV][DC];
    const int kh = blockIdx.x, c = blockIdx.y, tb = DC * c, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = tid + 256 * k, tt = i >> 5, cc = i & 31, t = tb + tt;
        const uint4 v = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) t * QKV + (cc < 16 ? 0 : NKH * 128) + kh * 128) + (cc & 15))
                              : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4 *>(cc < 16 ? &sQ[tt][8 * cc] : &sK[tt][8 * (cc - 16)]) = v;
    }
    if (w < DN_GV) { // value head kh + NKH w: cumulative log gates as in pk_dn_scan's put (padding rows: log g = 0, beta = 0)
        const int h = kh + NKH * w, tl = tb + lane;
        const float beta = tl < T ? __ldg(cP.gb + (size_t) tl * 96 + NVH + h) : 0.f;
        float a = tl < T ? __ldg(cP.gb + (size_t) tl * 96 + h) : 0.f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, a, o);
            if (lane >= o) a += y;
        }
        sa[w][lane] = a;
        sb[w][lane] = beta;
        sg[w][lane] = __expf(a);
    }
    __syncthreads();
    { // K K^T (warps 0-3) and Q K^T (warps 4-7): 2 n8 tiles each, as pk_dn_scan; then A (fp32, transposed) and P per value head
        const int isq = w >> 2, mt = (w >> 1) & 1, n0 = 16 * (w & 1);
        const __half (*X)[DC_LD] = isq ? sQ : sK;
        float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 8; s++) {
            unsigned a[4], b[4];
            ldsm4(a, &X[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
            ldsm4(b, &sK[n0 + (lane & 7) + ((lane >> 4) << 3)][16 * s + ((lane >> 3) & 1) * 8]);
            mma4(c0, a, b[0], b[1]);
            mma4(c1, a, b[2], b[3]);
        }
        __half * pg = tri + ((size_t) (c * NVH + kh) * 3 + 2) * (DC * DC); // P of value head kh (+ NKH 3 DC DC per j)
#pragma unroll
        for (int v = 0; v < DN_GV; v++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int i = 16 * mt + g + ((e & 2) ? 8 : 0), j = n0 + 8 * (e >> 2) + 2 * t4 + (e & 1);
                const float x = (e >> 2) ? c1[e & 3] : c0[e & 3];
                const float f = j <= i ? __expf(sa[v][i] - sa[v][j]) * x : 0.f;
                if (isq) pg[(size_t) v * NKH * 3 * DC * DC + i * DC + j] = __float2half_rn(f);
                else sAT[v][j][i] = j < i ? sb[v][i] * f : 0.f;
            }
    }
    __syncthreads();
    if (w < DN_GV) { // T = (I + A)^-1 of value head kh + NKH w (lane = column, right-looking), as pk_dn_scan's warp 0
        float x[DC];
#pragma unroll
        for (int i = 0; i < DC; i++) x[i] = i == lane ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < DC - 1; j++) {
#pragma unroll
            for (int i4 = (j + 1) / 4; i4 < DC / 4; i4++) {
                const float4 a4 = *reinterpret_cast<const float4 *>(&sAT[w][j][4 * i4]);
                const float av[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
                for (int q = 0; q < 4; q++)
                    if (4 * i4 + q > j) x[4 * i4 + q] = fmaf(-av[q], x[j], x[4 * i4 + q]);
            }
        }
        const float bc = sb[w][lane], bgc = bc * sg[w][lane];
        __half * tg = tri + (size_t) (c * NVH + kh + NKH * w) * 3 * (DC * DC);
#pragma unroll
        for (int i = 0; i < DC; i++) {
            tg[i * DC + lane] = __float2half_rn(x[i] * bc);
            tg[DC * DC + i * DC + lane] = __float2half_rn(x[i] * bgc);
        }
    }
}

// pk_dn_scan with pk_dn_tri's matrices (MK_PFX bit 2 off): rows t0 .. t0 + T - 1 must lie on pk_dn_tri's 32-row grid and
// end at its T (t0 % 32 == 0; a piece ending elsewhere has its padding elsewhere: pk_dn_scan). Block = (value head, 32
// value columns), warp = 8 columns with pk_dn_scan's per-column code. 192 blocks of 4 warps (<= 3 per SM: <= 3 warps of
// fp32-accumulating MMAs per SMSP; pk_dn_scan's 96 blocks of 8 warps put 4 per SMSP on 14 SMs), and per chunk two block
// barriers instead of four (no K K^T / Q K^T, no solve).
constexpr int DC_LV4 = 40; // sV row stride (halves): 32 columns + 8 (ldmatrix rows on distinct bank quads)
__global__ void __launch_bounds__(128, 3) pk_dn_scan4(int T, int t0, const float * st_in, float * st_out, const __half * tri) { MK_VIEW_INIT();
    __shared__ __align__(16) __half sK[DC][DC_LD], sQ[DC][DC_LD], sV[DC][DC_LV4];
    __shared__ __align__(16) __half sT0[DC][DC_LP], sT1[DC][DC_LP], sP[DC][DC_LP];
    __shared__ float sg[DC], sd[DC];
    const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int col0 = blockIdx.y * 32 + 8 * w; // this warp's 8 value columns
    const float * Si = st_in + (size_t) h * 128 * 128; // element (col, row) at dn_sidx(col, row)
    float S[8][4];                                    // rows 16 mt + g (+8), columns col0 + 2 t4 (+1)
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        const int c = col0 + 2 * t4, rw = 16 * mt + g;
        S[mt][0] = __ldcg(Si + dn_sidx(c, rw));
        S[mt][1] = __ldcg(Si + dn_sidx(c + 1, rw));
        S[mt][2] = __ldcg(Si + dn_sidx(c, rw + 8));
        S[mt][3] = __ldcg(Si + dn_sidx(c + 1, rw + 8));
    }

    uint4 pqk[8], pv, pm[3];
    float plg = 0.f;
    auto fetch = [&](int tb) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int i = tid + 128 * k, tt = i >> 5, c = i & 31, t = tb + tt;
            pqk[k] = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) (t0 + t) * QKV + (c < 16 ? 0 : NKH * 128) + kh * 128) + (c & 15))
                           : make_uint4(0, 0, 0, 0);
        }
        const int t = tb + (tid >> 2);
        pv = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) (t0 + t) * QKV + 2 * NKH * 128 + h * 128 + blockIdx.y * 32) + (tid & 3))
                   : make_uint4(0, 0, 0, 0);
        const uint4 * tg = reinterpret_cast<const uint4 *>(tri + (size_t) ((t0 + tb) / DC * NVH + h) * 3 * (DC * DC));
#pragma unroll
        for (int m = 0; m < 3; m++) pm[m] = tb < T ? __ldg(tg + DC * DC / 8 * m + tid) : make_uint4(0, 0, 0, 0); // row tid / 4, cols 8 (tid % 4) ..
        if (w == 0) {
            const int tl = tb + lane;
            plg = tl < T ? __ldg(cP.gb + (size_t) (t0 + tl) * 96 + h) : 0.f;
        }
    };
    auto put = [&] {
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int i = tid + 128 * k, tt = i >> 5, c = i & 31;
            *reinterpret_cast<uint4 *>(c < 16 ? &sQ[tt][8 * c] : &sK[tt][8 * (c - 16)]) = pqk[k];
        }
        *reinterpret_cast<uint4 *>(&sV[tid >> 2][8 * (tid & 3)]) = pv;
        *reinterpret_cast<uint4 *>(&sT0[tid >> 2][8 * (tid & 3)]) = pm[0];
        *reinterpret_cast<uint4 *>(&sT1[tid >> 2][8 * (tid & 3)]) = pm[1];
        *reinterpret_cast<uint4 *>(&sP[tid >> 2][8 * (tid & 3)]) = pm[2];
        if (w == 0) { // cumulative log gates (padding rows: log g = 0)
            float a = plg;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, a, o);
                if (lane >= o) a += y;
            }
            const float al = __shfl_sync(0xffffffffu, a, 31);
            sg[lane] = __expf(a);
            sd[lane] = __expf(al - a);
        }
    };

    fetch(0);
    for (int tb = 0; tb < T; tb += DC) {
        __syncthreads(); // previous chunk done with shared memory
        put();
        __syncthreads();
        if (tb + DC < T) fetch(tb + DC);

        // K S and Q S for this warp's columns
        float ks[2][4] = {}, qs[2][4] = {};
#pragma unroll
        for (int s = 0; s < 8; s++) { // k = key dims 16 s .. : B = S tile s
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(S[s][e], pf::CHK_DN_S);
#endif
            const unsigned b0 = movt(pk2(S[s][0], S[s][1])), b1 = movt(pk2(S[s][2], S[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                ldsm4(a, &sK[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(ks[mt], a, b0, b1);
                ldsm4(a, &sQ[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(qs[mt], a, b0, b1);
            }
        }

        // U = T diag(beta) V - T diag(beta gamma) (K S)
        float u[2][4] = {}, ws[2][4] = {};
#pragma unroll
        for (int s = 0; s < 2; s++) { // k = tokens 16 s ..
            unsigned bv[2];
            ldsm2t(bv, &sV[16 * s + (lane & 15)][8 * w]);
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(ks[s][e], pf::CHK_DN_KS);
#endif
            const unsigned bk0 = movt(pk2(ks[s][0], ks[s][1])), bk1 = movt(pk2(ks[s][2], ks[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                if (s > mt) continue; // T is lower triangular
                unsigned a[4];
                ldsm4(a, &sT0[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(u[mt], a, bv[0], bv[1]);
                ldsm4(a, &sT1[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(ws[mt], a, bk0, bk1);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) u[mt][e] -= ws[mt][e];

        // O = diag(gamma) Q S + P U
        float o[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            const float ga = sg[16 * mt + g], gb_ = sg[16 * mt + g + 8];
            o[mt][0] = ga * qs[mt][0]; o[mt][1] = ga * qs[mt][1];
            o[mt][2] = gb_ * qs[mt][2]; o[mt][3] = gb_ * qs[mt][3];
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(u[s][e], pf::CHK_DN_U);
#endif
            const unsigned bu0 = movt(pk2(u[s][0], u[s][1])), bu1 = movt(pk2(u[s][2], u[s][3]));
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                if (s > mt) continue;
                unsigned a[4];
                ldsm4(a, &sP[16 * mt + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                mma4(o[mt], a, bu0, bu1);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int t = tb + 16 * mt + g + 8 * hf;
                if (t < T)
                    *reinterpret_cast<float2 *>(cP.y2 + (size_t) (t0 + t) * VD + h * 128 + col0 + 2 * t4) =
                        make_float2(o[mt][2 * hf], o[mt][2 * hf + 1]);
            }

        // S = gamma_31 S + K^T diag(exp(a_31 - a)) U
        const float gl = sg[DC - 1];
#pragma unroll
        for (int mt = 0; mt < 8; mt++)
#pragma unroll
            for (int e = 0; e < 4; e++) S[mt][e] *= gl;
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const float da = sd[16 * s + g], db = sd[16 * s + g + 8];
#if MK_PF_INFCHK
#pragma unroll
            for (int e = 0; e < 4; e++) pf::chk_clamp(u[s][e] * (e < 2 ? da : db), pf::CHK_DN_UD);
#endif
            const unsigned b0 = movt(pk2(u[s][0] * da, u[s][1] * da)), b1 = movt(pk2(u[s][2] * db, u[s][3] * db));
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                unsigned a[4];
                ldsm4t(a, &sK[16 * s + (lane & 7) + ((lane >> 4) << 3)][16 * mt + ((lane >> 3) & 1) * 8]);
                mma4(S[mt], a, b0, b1);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        float * p = st_out + (size_t) h * 128 * 128;
        const int c = col0 + 2 * t4, rw = 16 * mt + g;
        p[dn_sidx(c, rw)] = S[mt][0];
        p[dn_sidx(c + 1, rw)] = S[mt][1];
        p[dn_sidx(c, rw + 8)] = S[mt][2];
        p[dn_sidx(c + 1, rw + 8)] = S[mt][3];
    }
}

// gated RMS norm of the delta-net output (z = y1) -> fp16 xh (K = VD)
__global__ void __launch_bounds__(256) pk_dn_out(const float * snorm) { MK_VIEW_INIT();
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int hh = warp; hh < NVH; hh += 8) {
        const float * o = cP.y2 + (size_t) t * VD + hh * 128;
        const float * z = cP.y1 + (size_t) t * VD + hh * 128;
        float v[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = __ldg(o + lane + 32 * i);
            ss += v[i] * v[i];
        }
        const float rs = rsqrtf(wsum(ss) / 128.0f + EPS);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = lane + 32 * i;
            hstore32<1>(v[i] * rs * __ldg(snorm + c) * silu(__ldg(z + c)), cP.xh + (size_t) t * VD + hh * 128 + 32 * i);
        }
    }
}

// attention q / k / v post-processing: RMS norms, RoPE, K / V -> cache (fp16), q -> qh
// (r0: first chunk row of this launch; MK_EXACT with KV window rings runs the chunk in 640-position pieces)
__global__ void __launch_bounds__(256) pk_attn_prep(int n0, int slot, const float * qn, const float * kn, int do_q, int r0 = 0) { MK_VIEW_INIT();
    const int t = r0 + blockIdx.x, pos = n0 + t, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t ctx = cB.ctx;
    for (int it = warp; it < 32; it += 8) {
        if (it < NH && !do_q) continue;
        float v[8];
        if (it < NH) load8(cP.y0 + (size_t) t * QG + it * 512 + lane * 8, v);
        else load8(cP.y1 + (size_t) t * 2 * KVD + (it >= NH + NKV ? KVD : 0) + ((it - NH) % NKV) * HD + lane * 8, v);
        const bool isv = it >= NH + NKV;
        if (!isv) {
            const float * nw = it < NH ? qn : kn;
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * __ldg(nw + lane * 8 + k);
            rope8(v, lane, pos);
            if (it < NH && cB.kvq8 && cB.kvr[slot]) fwht256(v); // the slot's K is stored rotated (MK_KV "4r")
        }
        const float sc = it < NH ? 0.0625f : 1.0f;
        __half2 hv[4];
#pragma unroll
        for (int k = 0; k < 4; k++) hv[k] = __floats2half2_rn(v[2 * k] * sc, v[2 * k + 1] * sc);
        if (it < NH) {
            *reinterpret_cast<int4 *>(cP.qh + ((size_t) t * NH + it) * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        } else {
            const size_t row = ((size_t) slot * NKV + (it - NH) % NKV) * ctx + pos;
#if MK_KV_EMU
            if (cB.kvq8 && cB.kvemu[isv][slot])
                kv_store_emu(static_cast<int8_t *>(isv ? cB.vc : cB.kc) + row * HD, (isv ? cB.vsc : cB.ksc) + row * (HD / 32), v, cB.kvemu[isv][slot], pos);
            else
#endif
            if (cB.kvq8) kv_store(isv, slot, (it - NH) % NKV, pos, v);
            else *reinterpret_cast<int4 *>(static_cast<__half *>(isv ? cB.vc : cB.kc) + row * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        }
    }
}

#include "prefill_attn.cuh"

// attention output * sigmoid(gate) -> fp16 xh (K = VD, hperm order within each 16-group). One warp per head, 8 dims per
// lane: a slot whose V is stored rotated (MK_KV "r" on V) gets its output rotated back here (WHT, its own inverse).
__global__ void __launch_bounds__(256) pk_attn_out(int slot) { MK_VIEW_INIT();
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const bool rotv = cB.kvq8 && cB.kvrv[slot];
    for (int h = warp; h < NH; h += 8) {
        float v[8];
        load8(cP.y2 + (size_t) t * VD + h * HD + lane * 8, v);
        if (rotv) fwht256(v);
        __half * dst = cP.xh + (size_t) t * VD + h * HD;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int d = lane * 8 + k;
            const float gate = __ldg(cP.y0 + (size_t) t * QG + h * 512 + 256 + d);
#if MK_PF_INFCHK
            pf::chk_clamp(v[k] / (1.0f + expf(-gate)), pf::CHK_IN_ATTN);
#endif
            dst[pf::hperm(d)] = __float2half_rn(fminf(fmaxf(v[k] / (1.0f + expf(-gate)), -65504.f), 65504.f));
        }
    }
}

// final RMS norm -> h rows (h[t + 1] = position n0 + t) and the decode h ring (last 16 positions)
__global__ void __launch_bounds__(256) pk_final_h(int T, int n0) { MK_VIEW_INIT();
    __shared__ float red[8];
    const int t = blockIdx.x;
    const float * x = cP.x + (size_t) t * D;
    float ss = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = __ldg(x + e);
        ss += v * v;
    }
    const float rs = rsqrtf(block_sum256(ss, red) / (float) D + EPS);
    const bool ring = t >= T - H_RING;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = __ldg(x + e) * rs * __ldg(cW.onorm + e);
        cP.h[(size_t) (t + 1) * D + e] = v;
        if (ring) cB.h_hist[(size_t) ((n0 + t) % H_RING) * D + e] = v;
    }
}

// MTP input rows: concat(enorm(embed(tok_p)), hnorm(h_{p-1})) -> fp16 xh (K = 2 D)
// IMG (a chunk with image rows): embed(tok_p) of an image position is its input-embedding row (as llama.cpp's MTP
// draft takes the vision embedding there, #30257)
template <bool IMG, int ET>
__global__ void __launch_bounds__(256) pk_mtp_in() { MK_VIEW_INIT();
    __shared__ float se[D];
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t * row = cW.embd + (size_t) cP.tok[t] * embd_rb<ET>();
    const float * er = nullptr;
    if (IMG) {
        const int r = __ldg(cP.erow + t);
        if (r >= 0) er = cP.emb + (size_t) r * D;
    }
    const float * hp = cP.h + (size_t) t * D;
    float ae = 0.f, ah = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = IMG && er ? __ldg(er + e) : embd_elem_pf<ET>(row, e), hv = __ldg(hp + e);
        se[e] = v;
        ae += v * v;
        ah += hv * hv;
    }
    const float rse = rsqrtf(block_sum256(ae, red) / (float) D + EPS);
    const float rsh = rsqrtf(block_sum256(ah, red) / (float) D + EPS);
    for (int j = warp; j < 2 * D / 32; j += 8) {
        const int e = 32 * (j % (D / 32)) + lane;
        const float v = j < D / 32 ? se[e] * rse * __ldg(cW.enorm + e) : __ldg(hp + e) * rsh * __ldg(cW.hnorm + e);
        hstore32<2>(v, cP.xh + (size_t) t * 2 * D + 32 * j);
    }
}

// argmax over logits rows y0[t][0..VOCAB) (scoring)
__global__ void __launch_bounds__(256) pk_argmax(int t0) { MK_VIEW_INIT();
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float * l = cP.y0 + (size_t) t * VOCAB;
    float b = -INFINITY;
    int ix = 0;
    for (int i = threadIdx.x; i < VOCAB; i += 256)
        if (__ldg(l + i) > b) { b = __ldg(l + i); ix = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, b, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
    }
    if (lane == 0) { bv[warp] = b; bi[warp] = ix; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++)
            if (bv[w] > b || (bv[w] == b && bi[w] < ix)) { b = bv[w]; ix = bi[w]; }
        cP.amax[t0 + t] = ix;
    }
}

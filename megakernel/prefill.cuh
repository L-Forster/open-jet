// Batched prefill: the prompt (all but its last token) is processed in chunks of up to `cap`
// tokens with ordinary kernel launches, mostly fp16 tensor-core GEMMs (prefill_gemm.cuh) against
// the same quantised weights the decode megakernel streams.  It leaves exactly the state the
// megakernel expects: K/V caches (trunk + MTP), committed delta-net state, conv ring, h history.
// Included inside qwen38_mk.cu's anonymous namespace.

#include "prefill_gemm.cuh"

struct PBufs {
    int cap;      // token capacity of a chunk (multiple of 128)
    float * x;    // [cap][D] residual stream
    float * xm;   // [cap][D] MTP stream
    float * rs;   // [cap] RMS scale of the last normed input
    __half * xh;  // [cap][FF] fp16 GEMM input, elements of each 16-group in pf::hperm order
    __half * xh2; // [cap][FF] fp16 silu(gate) * up (ffn_down input), same order
    float * y0;   // [cap][FF]  qkv | q+gate | ffn gate | logits
    float * y1;   // [cap][FF]  z | k,v | ffn up
    float * y2;   // [cap][VD]  delta-net / attention output
    __half * dq;  // [cap][QKV] conv'd + normalised q | k | v (fp16)
    float * gb;   // [cap][2 * NVH] log g | beta
    float * ab;   // [AB_SPLITS][cap][2 * NVH] alpha | beta pre-activations (split-K partials)
    float * h;    // [cap + 1][D] final normed hidden of positions n0 - 1 .. n0 + T - 1
    __half * qh;  // [cap][NH][HD] attention queries (normed, rotated, pre-scaled)
    int * tok;    // [cap]
    int * amax;   // [cap] argmax output (scoring)
};

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
__device__ __forceinline__ void hstore32(float v, __half * dst) {
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

__global__ void pk_embed(int T) {
    const int t = blockIdx.x;
    const uint8_t * row = cW.embd + (size_t) cP.tok[t] * (D / 256 * 144);
    for (int e = threadIdx.x; e < D; e += blockDim.x) cP.x[(size_t) t * D + e] = dequant_q4k_ldg(row, e);
}

// RMS-norm (weight w) of rows of src [T][D] -> fp16 xh (K = D); rs per token
__global__ void __launch_bounds__(256) pk_norm_quant(const float * src, const float * w) {
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

// delta-net alpha / beta: ab[c][t][r] = rs[t] * sum_{k in split c} Wab[r][k] x[t][k] (attn_norm folded
// into Wab).  Block = 64 tokens x 96 rows x one of AB_SPLITS K-splits; thread = 4 tokens x 6 rows.
constexpr int AB_SPLITS = 8, AB_KS = D / AB_SPLITS;
__global__ void __launch_bounds__(256) pk_ab(const float * Wab, int T) {
    __shared__ float sx[32][64 + 4];
    __shared__ float sw[32][96 + 4];
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
            const float4 v = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) rr * D + k0 + kk));
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

// conv1d (kernel 4, causal, continuing from the conv ring) + silu, l2-norm of q / k, log g / beta
__global__ void __launch_bounds__(256) pk_dn_prep(int T, int n0, int dl, const float * convw, const float * dt,
                                                  const float * A) {
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

// raw qkv rows of the chunk's last positions -> conv ring (after pk_dn_prep consumed the old ones)
__global__ void pk_conv_ring(int T, int n0, int dl) {
    float * ring = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int nt = min(T, CONV_RING), t = T - nt + blockIdx.x;
    for (int c = threadIdx.x; c < QKV; c += blockDim.x)
        ring[(size_t) ((n0 + t) % CONV_RING) * QKV + c] = __ldg(cP.y0 + (size_t) t * QKV + c);
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

__global__ void __launch_bounds__(256, 2) pk_dn_scan(int T, int dl) {
    __shared__ __align__(16) __half sK[DC][DC_LD], sQ[DC][DC_LD], sV[DC][DC_LV];
    __shared__ __align__(16) __half sT0[DC][DC_LP], sT1[DC][DC_LP], sP[DC][DC_LP];
    __shared__ __align__(16) float sAT[DC][DC_LA]; // A transposed: sAT[j][i] = A_ij
    __shared__ float sa[DC], sb[DC], sg[DC], sd[DC];
    const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int col0 = blockIdx.y * 64 + 8 * w; // this warp's 8 value columns
    float * Sg = cB.dstate + ((size_t) dl * NVH + h) * 128 * 128; // [col][row]
    float S[8][4];                                                // rows 16 mt + g (+8), columns col0 + 2 t4 (+1)
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        const float * p = Sg + (size_t) (col0 + 2 * t4) * 128 + 16 * mt + g;
        S[mt][0] = __ldcg(p);
        S[mt][1] = __ldcg(p + 128);
        S[mt][2] = __ldcg(p + 8);
        S[mt][3] = __ldcg(p + 136);
    }

    uint4 pqk[4], pv;
    float pbeta = 0.f, plg = 0.f;
    auto fetch = [&](int tb) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = tid + 256 * k, tt = i >> 5, c = i & 31, t = tb + tt;
            pqk[k] = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) t * QKV + (c < 16 ? 0 : NKH * 128) + kh * 128) + (c & 15))
                           : make_uint4(0, 0, 0, 0);
        }
        const int t = tb + (tid >> 3);
        pv = t < T ? __ldg(reinterpret_cast<const uint4 *>(cP.dq + (size_t) t * QKV + 2 * NKH * 128 + h * 128 + blockIdx.y * 64) + (tid & 7))
                   : make_uint4(0, 0, 0, 0);
        if (w == 0) {
            const int tl = tb + lane;
            pbeta = tl < T ? __ldg(cP.gb + (size_t) tl * 96 + NVH + h) : 0.f;
            plg = tl < T ? __ldg(cP.gb + (size_t) tl * 96 + h) : 0.f;
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
                    *reinterpret_cast<float2 *>(cP.y2 + (size_t) t * VD + h * 128 + col0 + 2 * t4) =
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
        float * p = Sg + (size_t) (col0 + 2 * t4) * 128 + 16 * mt + g;
        p[0] = S[mt][0];
        p[128] = S[mt][1];
        p[8] = S[mt][2];
        p[136] = S[mt][3];
    }
}

// gated RMS norm of the delta-net output (z = y1) -> fp16 xh (K = VD)
__global__ void __launch_bounds__(256) pk_dn_out(const float * snorm) {
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
            hstore32(v[i] * rs * __ldg(snorm + c) * silu(__ldg(z + c)), cP.xh + (size_t) t * VD + hh * 128 + 32 * i);
        }
    }
}

// attention q / k / v post-processing: RMS norms, RoPE, K / V -> cache (fp16), q -> qh
__global__ void __launch_bounds__(256) pk_attn_prep(int n0, int slot, const float * qn, const float * kn, int do_q) {
    const int t = blockIdx.x, pos = n0 + t, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
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
        }
        const float sc = it < NH ? 0.0625f : 1.0f;
        __half2 hv[4];
#pragma unroll
        for (int k = 0; k < 4; k++) hv[k] = __floats2half2_rn(v[2 * k] * sc, v[2 * k + 1] * sc);
        if (it < NH) {
            *reinterpret_cast<int4 *>(cP.qh + ((size_t) t * NH + it) * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        } else {
            const size_t row = ((size_t) slot * NKV + (it - NH) % NKV) * ctx + pos;
            if (cB.kvq8) kv_store_q8(static_cast<int8_t *>(isv ? cB.vc : cB.kc) + row * HD, (isv ? cB.vsc : cB.ksc) + row * (HD / 32), v);
            else *reinterpret_cast<int4 *>(static_cast<__half *>(isv ? cB.vc : cB.kc) + row * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        }
    }
}

// Causal flash attention over the cache for the chunk's queries.  CTA = (16 query positions, kv
// head); warp = one of the 6 query heads of the group (16 rows).  K / V tiles of 32 positions in
// shared memory, S = Q K^T and O += P V on tensor cores (fp16 operands, fp32 accumulation).
constexpr int AT_Q = 16, AT_K = 32, AT_LD = HD + 8;
constexpr int AT_SMEM = (6 * AT_Q + 2 * AT_K) * AT_LD * 2; // Q, K/V tile
template <bool Q8>
__global__ void __launch_bounds__(192) pk_attn(int T, int n0, int slot) {
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sQ = reinterpret_cast<__half *>(asm_); // [96][AT_LD]
    __half * sK = sQ + 6 * AT_Q * AT_LD;            // [32][AT_LD]
    __half * sV = sK + AT_K * AT_LD;                // [32][AT_LD]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int t0 = blockIdx.x * AT_Q, g = blockIdx.y;
    const size_t ctx = cB.ctx;
    const size_t row0 = ((size_t) slot * NKV + g) * ctx;
    const __half * Kc = static_cast<const __half *>(cB.kc) + row0 * HD;
    const __half * Vc = static_cast<const __half *>(cB.vc) + row0 * HD;
    const int nq = min(AT_Q, T - t0);
    for (int i = tid; i < 6 * AT_Q * 32; i += 192) { // 96 rows x 32 chunks of 16 B
        const int r = i >> 5, c = i & 31, hq = r / AT_Q, tt = r % AT_Q;
        __half * d = sQ + r * AT_LD + 8 * c;
        if (tt < nq) cp_async16(d, cP.qh + ((size_t) (t0 + tt) * NH + g * 6 + hq) * HD + 8 * c);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
    const int qpos0 = n0 + t0, pmax = qpos0 + nq - 1;
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f; // rows g8, g8 + 8
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const __half * qa = sQ + (warp * AT_Q + g8) * AT_LD + 2 * t4;
    for (int kb = 0; kb <= pmax; kb += AT_K) {
        const int np = min(AT_K, pmax + 1 - kb);
        if constexpr (Q8) { // q8 K/V tile -> fp16 in shared memory (all loads issued before the first conversion)
            constexpr int PER = (2 * AT_K * 32 + 191) / 192;
            uint2 q[PER];
            float d[PER];
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, c = i & 31;
                q[k] = make_uint2(0, 0);
                d[k] = 0.f;
                if (i < 2 * AT_K * 32 && r < np) {
                    const size_t row = row0 + kb + r;
                    q[k] = __ldg(reinterpret_cast<const uint2 *>(static_cast<const int8_t *>(which ? cB.vc : cB.kc) + row * HD) + c);
                    d[k] = __half2float((which ? cB.vsc : cB.ksc)[row * (HD / 32) + (c >> 2)]);
                }
            }
            pf::cpa_wait<0>(); // Q rows (first tile)
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, c = i & 31;
                if (i >= 2 * AT_K * 32) break;
                __half2 h[4];
                q8_to_f16(q[k], d[k], h);
                *reinterpret_cast<int4 *>((which ? sV : sK) + r * AT_LD + 8 * c) = *reinterpret_cast<int4 *>(h);
            }
        } else {
            for (int i = tid; i < AT_K * 32; i += 192) {
                const int p = i >> 5, c = i & 31;
                if (p < np) {
                    cp_async16(sK + p * AT_LD + 8 * c, Kc + (size_t) (kb + p) * HD + 8 * c);
                    cp_async16(sV + p * AT_LD + 8 * c, Vc + (size_t) (kb + p) * HD + 8 * c);
                } else {
                    *reinterpret_cast<int4 *>(sK + p * AT_LD + 8 * c) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4 *>(sV + p * AT_LD + 8 * c) = make_int4(0, 0, 0, 0);
                }
            }
            pf::cpa_commit();
            pf::cpa_wait<0>();
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks);
            const unsigned a1 = *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 16 * ks);
            const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8);
            const unsigned a3 = *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 16 * ks + 8);
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const __half * kp = sK + (8 * ni + g8) * AT_LD + 2 * t4 + 16 * ks;
                mma16816(s[ni], a0, a1, a2, a3, *reinterpret_cast<const unsigned *>(kp),
                         *reinterpret_cast<const unsigned *>(kp + 8));
            }
        }
        // causal mask + online softmax (rows g8 and g8 + 8; a row's 4 lanes share it)
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int p = kb + 8 * ni + 2 * t4;
            if (p > qr0) s[ni][0] = -INFINITY;
            if (p + 1 > qr0) s[ni][1] = -INFINITY;
            if (p > qr1) s[ni][2] = -INFINITY;
            if (p + 1 > qr1) s[ni][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[ni][0], s[ni][1]));
            mx1 = fmaxf(mx1, fmaxf(s[ni][2], s[ni][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = mn0 == -INFINITY ? 1.f : expf(m0 - mn0), al1 = mn1 == -INFINITY ? 1.f : expf(m1 - mn1);
        float ps0 = 0.f, ps1 = 0.f;
        unsigned pa[4][2]; // P as fp16 pairs: [n8 tile][row g8 / g8 + 8]
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const float p0 = mn0 == -INFINITY ? 0.f : expf(s[ni][0] - mn0);
            const float p1 = mn0 == -INFINITY ? 0.f : expf(s[ni][1] - mn0);
            const float p2 = mn1 == -INFINITY ? 0.f : expf(s[ni][2] - mn1);
            const float p3 = mn1 == -INFINITY ? 0.f : expf(s[ni][3] - mn1);
            ps0 += p0 + p1;
            ps1 += p2 + p3;
            __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pa[ni][0] = *reinterpret_cast<unsigned *>(&h01);
            pa[ni][1] = *reinterpret_cast<unsigned *>(&h23);
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * al0 + ps0;
        l1 = l1 * al1 + ps1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            o[i][0] *= al0;
            o[i][1] *= al0;
            o[i][2] *= al1;
            o[i][3] *= al1;
        }
        // O += P V: k = positions (2 steps of 16), n = 32 tiles of 8 dims
#pragma unroll
        for (int ks = 0; ks < AT_K / 16; ks++) {
            const unsigned a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
#pragma unroll
            for (int dt2 = 0; dt2 < 16; dt2++) { // two 8-dim tiles per ldmatrix.x4
                const __half * vb = sV + (16 * ks + (lane & 15)) * AT_LD + 16 * dt2 + (lane >> 4) * 8;
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                             : "r"((unsigned) __cvta_generic_to_shared(vb)));
                mma16816(o[2 * dt2], a0, a1, a2, a3, b0, b1);
                mma16816(o[2 * dt2 + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();
    }
    const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int hq = g * 6 + warp;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int d = 8 * i + 2 * t4;
        if (g8 < nq) {
            float * dst = cP.y2 + (size_t) (t0 + g8) * VD + hq * HD + d;
            dst[0] = o[i][0] * il0;
            dst[1] = o[i][1] * il0;
        }
        if (g8 + 8 < nq) {
            float * dst = cP.y2 + (size_t) (t0 + g8 + 8) * VD + hq * HD + d;
            dst[0] = o[i][2] * il1;
            dst[1] = o[i][3] * il1;
        }
    }
}

// attention output * sigmoid(gate) -> fp16 xh (K = VD)
__global__ void __launch_bounds__(256) pk_attn_out() {
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int j = warp; j < VD / 32; j += 8) {
        const int e = 32 * j + lane, h = e / HD, d = e % HD;
        const float gate = __ldg(cP.y0 + (size_t) t * QG + h * 512 + 256 + d);
        hstore32(__ldg(cP.y2 + (size_t) t * VD + e) / (1.0f + expf(-gate)), cP.xh + (size_t) t * VD + 32 * j);
    }
}

// final RMS norm -> h rows (h[t + 1] = position n0 + t) and the decode h ring (last 16 positions)
__global__ void __launch_bounds__(256) pk_final_h(int T, int n0) {
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
__global__ void __launch_bounds__(256) pk_mtp_in() {
    __shared__ float se[D];
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t * row = cW.embd + (size_t) cP.tok[t] * (D / 256 * 144);
    const float * hp = cP.h + (size_t) t * D;
    float ae = 0.f, ah = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = dequant_q4k_ldg(row, e), hv = __ldg(hp + e);
        se[e] = v;
        ae += v * v;
        ah += hv * hv;
    }
    const float rse = rsqrtf(block_sum256(ae, red) / (float) D + EPS);
    const float rsh = rsqrtf(block_sum256(ah, red) / (float) D + EPS);
    for (int j = warp; j < 2 * D / 32; j += 8) {
        const int e = 32 * (j % (D / 32)) + lane;
        const float v = j < D / 32 ? se[e] * rse * __ldg(cW.enorm + e) : __ldg(hp + e) * rsh * __ldg(cW.hnorm + e);
        hstore32(v, cP.xh + (size_t) t * 2 * D + 32 * j);
    }
}

// argmax over logits rows y0[t][0..VOCAB) (scoring)
__global__ void __launch_bounds__(256) pk_argmax(int t0) {
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

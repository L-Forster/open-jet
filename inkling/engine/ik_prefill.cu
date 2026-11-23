// Batched (prefill) kernels: T tokens per launch, activations stored [T][dim]. Each mirrors a decode kernel of
// ik_kernels.cu / expert_kernels.cu / gguf_kernels.cu with the token as an extra dimension. Weight-heavy kernels read a
// weight once per group of tokens (dense matrices) or once per expert for all the chunk's tokens that chose it
// (experts, after grouping), which is where prefill gains over decoding token by token.

#include "ik_prefill.cuh"

#include <cmath>
#include <cuda_fp16.h>

namespace ik {
namespace pf {

namespace {

constexpr int WARP = 32;
constexpr int TG = 16;  // tokens per pass in the weight-reuse kernels (accumulators held in registers)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int LPR>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = (blockDim.x + WARP - 1) / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    return warp_sum(lane < nw ? red[lane] : 0.f);
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
    const float act = beta * tanhf(g / beta) * sigmoidf(g);
    if (lb > 0.f) u = lb * tanhf(u / lb);
    return act * u;
}

__device__ __forceinline__ float ue4m3(uint32_t x) {
    x &= 0x7F;
    const float normal = __int_as_float(int((x << 20) + (119u << 23)));
    return x == 0x7F ? 0.f : (x < 8 ? float(x) * 0.0009765625f : normal);
}

__device__ const float k_fp4_init[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, -0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

// ---------------------------------------------------------------- per-token kernels, token = blockIdx.y (or x)

__device__ __forceinline__ void pf_k_scale_min(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ float pf_emb_dequant_elem(const uint8_t * row, int i, int emb_type) {
    if (emb_type == 8) {  // Q8_0
        const uint8_t * blk = row + (i / 32) * 34;
        return __half2float(*reinterpret_cast<const __half *>(blk)) * float(int8_t(blk[2 + (i & 31)]));
    }
    if (emb_type == 12) {  // Q4_K
        const uint8_t * b = row + (i / 256) * 144;
        const int rem = i & 255, is = rem >> 5, p = is >> 1, l = rem & 31;
        int sc, m;
        pf_k_scale_min(is, b + 4, sc, m);
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(b + 2));
        const uint8_t q = b[16 + 32 * p + l];
        return d * float(sc) * float((is & 1) ? (q >> 4) : (q & 0xF)) - dmin * float(m);
    }
    if (emb_type == 13) {  // Q5_K
        const uint8_t * b = row + (i / 256) * 176;
        const int rem = i & 255, is = rem >> 5, p = is >> 1, l = rem & 31;
        int sc, m;
        pf_k_scale_min(is, b + 4, sc, m);
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(b + 2));
        const uint8_t ql = b[48 + 32 * p + l];
        const int lo = (is & 1) ? (ql >> 4) : (ql & 0xF);
        const int hi = ((b[16 + l] >> is) & 1) << 4;
        return d * float(sc) * float(lo + hi) - dmin * float(m);
    }
    // Q6_K (14)
    const uint8_t * b = row + (i / 256) * 210;
    const int rem = i & 255, k8 = rem >> 5, half = k8 >> 2, k = k8 & 3, l = rem & 31;
    const uint8_t * ql = b + 64 * half;
    const uint8_t * qh = b + 128 + 32 * half;
    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * half;
    const int lo = (k & 1) ? ql[l + 32] : ql[l];
    const int q = (((k & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
    return __half2float(*reinterpret_cast<const __half *>(b + 208)) * float(sc[(l >> 4) + 2 * k] * q);
}

__global__ void embed_kernel(const uint8_t * __restrict__ rows, const float * __restrict__ w, float * __restrict__ out,
                             int n, float eps, int emb_type, size_t row_bytes) {
    __shared__ float red[32];
    extern __shared__ float x[];
    const int t = blockIdx.x;
    const uint8_t * row = rows + size_t(t) * row_bytes;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = pf_emb_dequant_elem(row, i, emb_type);
        x[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[size_t(t) * n + i] = x[i] * inv * w[i];
}

__global__ void mix_kernel(MixStates st, const float * __restrict__ query, int n_q, float * __restrict__ out, int n) {
    constexpr int NS = MAX_STATES, NQ = 4, NACC = NS * (NQ + 1);
    __shared__ float red[NACC][32];
    __shared__ float tot[NACC];
    __shared__ float coef[NQ][NS];
    const int t = blockIdx.x;
    const size_t off = size_t(t) * n;
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    float acc[NACC];
#pragma unroll
    for (int j = 0; j < NACC; ++j) acc[j] = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float qv[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) qv[q] = q < n_q ? query[q * n + i] : 0.f;
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            if (s < st.n) {
                const float x = st.p[s][off + i];
                acc[s] += x * x;
#pragma unroll
                for (int q = 0; q < NQ; ++q) acc[NS + q * NS + s] += x * qv[q];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) red[j][w] = v;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < NACC; j += blockDim.x) {
        float s = 0.f;
        for (int k = 0; k < nw; ++k) s += red[j][k];
        tot[j] = s;
    }
    __syncthreads();
    if (threadIdx.x < n_q) {
        const int q = threadIdx.x;
        float sc[NS], mx = -INFINITY;
        for (int s = 0; s < st.n; ++s) {
            sc[s] = tot[NS + q * NS + s] * rsqrtf(tot[s] / n + 1e-6f) * rsqrtf(float(n));
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < st.n; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < st.n; ++s) coef[q][s] = sc[s] / sum * rsqrtf(tot[s] / n + 1e-6f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xs[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) xs[s] = s < st.n ? st.p[s][off + i] : 0.f;
        for (int q = 0; q < n_q; ++q) {
            float v = 0.f;
            // only the st.n states that exist: coef[q][s >= st.n] is uninitialised shared memory, and 0 * NaN = NaN
            for (int s = 0; s < st.n; ++s) v += coef[q][s] * xs[s];
            out[(size_t(t) * n_q + q) * n + i] = v;
        }
    }
}

// g[t][c][r] = down[c][r] . [rms(cur[t]), rms(msg[t])]; one warp per (t, c, r)
__global__ void control_g_kernel(ControlArgs a, size_t cur_stride, size_t msg_stride, float * __restrict__ g, int n, int rank, int T) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int r = gw % rank, c = (gw / rank) % a.count, t = gw / (rank * a.count);
    if (t >= T) return;
    const float * cur = a.cur[c] + size_t(t) * cur_stride;
    const float * msg = a.msg[c] + size_t(t) * msg_stride;
    const float * row = a.down[c] + size_t(r) * 2 * n;
    float s1 = 0.f, s2 = 0.f, d1 = 0.f, d2 = 0.f;
    for (int i = lane; i < n; i += WARP) {
        const float x = cur[i], y = msg[i];
        s1 += x * x; s2 += y * y;
        d1 += row[i] * x; d2 += row[n + i] * y;
    }
    s1 = warp_sum(s1); s2 = warp_sum(s2); d1 = warp_sum(d1); d2 = warp_sum(d2);
    if (lane == 0) g[(size_t(t) * a.count + c) * 64 + r] = d1 * rsqrtf(s1 / n + 1e-6f) + d2 * rsqrtf(s2 / n + 1e-6f);
}

__global__ void control_out_kernel(ControlArgs a, size_t cur_stride, size_t msg_stride, size_t out_stride, const float * __restrict__ g,
                                   int n, int rank, int T) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int i = gw % n, c = (gw / n) % a.count, t = gw / (n * a.count);
    if (t >= T) return;
    const float * gt = g + (size_t(t) * a.count + c) * 64;
    const float * gr = a.gain[c] + size_t(i) * rank;
    const float * fr = a.forget[c] + size_t(i) * rank;
    float ga = 0.f, fo = 0.f;
    for (int k = lane; k < rank; k += WARP) { ga += gr[k] * gt[k]; fo += fr[k] * gt[k]; }
    ga = tanhf(warp_sum(ga));
    fo = tanhf(warp_sum(fo));
    if (lane == 0) {
        const float x = a.cur[c][size_t(t) * cur_stride + i];
        a.out[c][size_t(t) * out_stride + i] = x - fo * x + ga * a.msg[c][size_t(t) * msg_stride + i];
    }
}

__global__ void rmsnorm_kernel(NormArgs a, size_t x_stride, size_t out_stride, int n, float eps) {
    __shared__ float red[32];
    const int v = blockIdx.x, t = blockIdx.y;
    const float * x = a.x[v] + size_t(t) * x_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a.out[v][size_t(t) * out_stride + i] = x[i] * inv * a.w[v][i];
}

// ---------------------------------------------------------------- dense matrices, weights read once per TG tokens

// repacked Q8_0 [rows][N]: q int8 [rows][N], d fp16 [rows][N/32]; Y[t][r] = W[r] . X[t]
template <int N, int TMAX = TG>
__global__ void __launch_bounds__(256) matvec_q8r_kernel(const uint8_t * __restrict__ m, const float * __restrict__ X,
                                                         float * __restrict__ Y, int rows, int T) {
    constexpr int STEPS = N / 256;
    const int lane = threadIdx.x % WARP;
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const uint2 * q = reinterpret_cast<const uint2 *>(m + size_t(r) * N);
        const __half * d = reinterpret_cast<const __half *>(m + size_t(rows) * N) + size_t(r) * (N / 32);
        for (int t0 = 0; t0 < T; t0 += TMAX) {
            float acc[TMAX];
#pragma unroll
            for (int j = 0; j < TMAX; ++j) acc[j] = 0.f;
#pragma unroll
            for (int s = 0; s < STEPS; ++s) {
                const int w = s * WARP + lane;
                const uint2 qw = __ldg(q + w);
                const float sc = __half2float(__ldg(d + (w >> 2)));
                float wv[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) { wv[i] = float(int8_t(qw.x >> (8 * i))); wv[4 + i] = float(int8_t(qw.y >> (8 * i))); }
#pragma unroll
                for (int j = 0; j < TMAX; ++j) {
                    if (t0 + j < T) {
                        const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(t0 + j) * N + w * 8);
                        const float4 a = __ldg(xv), b = __ldg(xv + 1);
                        acc[j] += (wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w +
                                   wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w) * sc;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < TMAX; ++j) {
                if (t0 + j < T) {
                    const float v = warp_sum(acc[j]);
                    if (lane == 0) Y[size_t(t0 + j) * rows + r] = v;
                }
            }
        }
    }
}

struct MultiQ8PfArgs {
    const uint8_t * w[5];
    float * y[5];
    int rows[5];
    int off[6];
    int count;
};

// Fused shared-input Q8_0 matvecs for T <= 8 (up to 5 projections sharing X: Q/K/V/Rel/Gate)
template <int N, int TMAX = 8>
__global__ void __launch_bounds__(256) matvec_q8r_multi_kernel(MultiQ8PfArgs a, const float * __restrict__ X, int T) {
    constexpr int STEPS = N / 256;
    const int lane = threadIdx.x % WARP;
    const int total = a.off[a.count];
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int g = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; g < total; g += stride) {
        int mi = 0;
        while (mi + 1 < a.count && g >= a.off[mi + 1]) ++mi;
        const int r = g - a.off[mi];
        const int rows = a.rows[mi];
        const uint8_t * m = a.w[mi];
        const uint2 * q = reinterpret_cast<const uint2 *>(m + size_t(r) * N);
        const __half * d = reinterpret_cast<const __half *>(m + size_t(rows) * N) + size_t(r) * (N / 32);
        float acc[TMAX];
#pragma unroll
        for (int j = 0; j < TMAX; ++j) acc[j] = 0.f;
#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            const int w = s * WARP + lane;
            const uint2 qw = __ldg(q + w);
            const float sc = __half2float(__ldg(d + (w >> 2)));
            float wv[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) { wv[i] = float(int8_t(qw.x >> (8 * i))); wv[4 + i] = float(int8_t(qw.y >> (8 * i))); }
#pragma unroll
            for (int j = 0; j < TMAX; ++j) {
                if (j < T) {
                    const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(j) * N + w * 8);
                    const float4 xa = __ldg(xv), xb = __ldg(xv + 1);
                    acc[j] += (wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w +
                               wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w) * sc;
                }
            }
        }
        float * Y = a.y[mi];
#pragma unroll
        for (int j = 0; j < TMAX; ++j) {
            if (j < T) {
                const float v = warp_sum(acc[j]);
                if (lane == 0) Y[size_t(j) * rows + r] = v;
            }
        }
    }
}

// fp32 [rows][n]: small matrices (sinks, convolution taps, router)
__global__ void matvec_f32_kernel(const float * __restrict__ w, const float * __restrict__ X, float * __restrict__ Y, int rows, int n, int T) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int r = gw % rows, t = gw / rows;
    if (t >= T) return;
    const float * row = w + size_t(r) * n;
    const float * x = X + size_t(t) * n;
    float d = 0.f;
    for (int i = lane; i < n; i += WARP) d += row[i] * x[i];
    d = warp_sum(d);
    if (lane == 0) Y[size_t(t) * rows + r] = d;
}

// ---------------------------------------------------------------- attention

// causal short convolutions over the chunk: thread per channel; out[t] uses inputs t-3..t (history before the chunk).
// hist_snaps (optional, verify mode): [T][3][C] snapshots of the post-token history for exact rollback on partial acceptance.
__global__ void conv_kernel(const float * __restrict__ X, float * __restrict__ hist, const float * __restrict__ w,
                            const float * __restrict__ dyn, int heads, float * __restrict__ out, int C, int T, float norm,
                            float * __restrict__ hist_snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    const float * wc = w + size_t(c) * 4;
    const int head = dyn ? c / (C / heads) : 0;
    for (int t = 0; t < T; ++t) {
        const float x = X[size_t(t) * C + c];
        float o = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3];
        if (dyn) {
            const float * dk = dyn + size_t(t) * heads * 4 + head * 4;
            o += dk[0] * x + dk[1] * h2 + dk[2] * h1 + dk[3] * h0;
        }
        out[size_t(t) * C + c] = (x + o) * norm;
        h0 = h1; h1 = h2; h2 = x;
        if (hist_snaps) {
            float * dst = hist_snaps + size_t(t) * 3 * C;
            dst[c] = h0; dst[C + c] = h1; dst[2 * C + c] = h2;
        }
    }
    hist[c] = h0; hist[C + c] = h1; hist[2 * C + c] = h2;
}

// per token: K clip and V rms into the caches at position pos0 + t; Q and R scaled, clipped and tau-scaled
__global__ void attn_prep2_kernel(AttnPrepArgs a, const float * __restrict__ kc_in, const float * __restrict__ vc_in, int pos0,
                                  const int * __restrict__ d_pos) {
    const int t = blockIdx.x, pos = (d_pos ? *d_pos : pos0) + t;
    const int kv = a.n_head_kv * a.d_head;
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int slot = a.ring > 0 ? pos % a.ring : pos;
    const float * ks = kc_in + size_t(t) * kv;
    const float * vs = vc_in + size_t(t) * kv;
    for (int h = w; h < a.n_head_kv; h += nw) {
        const float4 xk = *reinterpret_cast<const float4 *>(ks + h * 128 + lane * 4);
        const float4 xv = *reinterpret_cast<const float4 *>(vs + h * 128 + lane * 4);
        const float sk = warp_sum(xk.x * xk.x + xk.y * xk.y + xk.z * xk.z + xk.w * xk.w);
        const float sv = warp_sum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w);
        const float fk = fminf(fmaxf(a.qk_clip / (sqrtf(sk) + 1e-6f), 0.f), 1.f);
        const float fv = rsqrtf(sv / 128.f + a.eps);
        const float vk[4] = {xk.x * fk, xk.y * fk, xk.z * fk, xk.w * fk};
        const float vv[4] = {xv.x * fv, xv.y * fv, xv.z * fv, xv.w * fv};
        if (a.kv_q8) {
            const size_t qoff = size_t(slot) * kv + size_t(h) * 128;
            const size_t soff = size_t(slot) * (kv / 32) + size_t(h) * 4;
            kv_store_q8_128(a.k_q + qoff, a.k_s + soff, vk);
            kv_store_q8_128(a.v_q + qoff, a.v_s + soff, vv);
        } else {
            const size_t off = size_t(slot) * kv + size_t(h) * 128 + lane * 4;
            *reinterpret_cast<float4 *>(a.k_cache + off) = make_float4(vk[0], vk[1], vk[2], vk[3]);
            *reinterpret_cast<float4 *>(a.v_cache + off) = make_float4(vv[0], vv[1], vv[2], vv[3]);
        }
    }
    const float tau = a.global ? 1.f + a.log_alpha * logf(fmaxf((pos + 1.f) / a.log_floor, 1.f)) : 1.f;
    const float * q = a.q + size_t(t) * a.n_head * a.d_head;
    const float * r = a.r + size_t(t) * a.n_head * a.n_rel;
    float * qo = a.q_out + size_t(t) * a.n_head * a.d_head;
    float * ro = a.r_out + size_t(t) * a.n_head * a.n_rel;
    for (int h = w; h < a.n_head; h += nw) {
        float sq = 0.f;
        for (int i = lane; i < a.d_head; i += WARP) { const float x = q[h * a.d_head + i] * a.q_clip[h * a.d_head + i]; sq += x * x; }
        sq = warp_sum(sq);
        const float fq = fminf(fmaxf(a.qk_clip / (sqrtf(sq) + 1e-6f), 0.f), 1.f) * tau;
        for (int i = lane; i < a.d_head; i += WARP) qo[h * a.d_head + i] = q[h * a.d_head + i] * a.q_clip[h * a.d_head + i] * fq;
        float sr = 0.f;
        for (int i = lane; i < a.n_rel; i += WARP) { const float x = r[h * a.n_rel + i] * a.rel_clip[h * a.n_rel + i]; sr += x * x; }
        sr = warp_sum(sr);
        const float fr = fminf(fmaxf(a.rel_clip_thr / (sqrtf(sr) + 1e-6f), 0.f), 1.f) * tau;
        for (int i = lane; i < a.n_rel; i += WARP) ro[h * a.n_rel + i] = r[h * a.n_rel + i] * a.rel_clip[h * a.n_rel + i] * fr;
    }
}

// one block per (head, token); the query at pos0 + t sees the cache up to its own position
template <bool Q8>
__global__ void attention_kernel(AttnArgs a, int pos0, const int * __restrict__ d_pos) {
    constexpr int D = 128;
    __shared__ float s_m[32], s_l[32];
    __shared__ float s_acc[32][D];
    const int h = blockIdx.x, t = blockIdx.y, hk = h / (a.n_head / a.n_head_kv);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int pos = (d_pos ? *d_pos : pos0) + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int kv = a.n_head_kv * D;
    const float * qp = a.q + size_t(t) * a.n_head * D;
    float q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = qp[h * D + lane * 4 + j] * rsqrtf(float(D));
    float r[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) r[j] = a.r[size_t(t) * a.n_head * 16 + h * 16 + j];
    const float sink = a.sinks[size_t(t) * a.n_head + h];
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int p = lo + w; p <= pos; p += nw) {
        const int slot = a.ring > 0 ? p % a.ring : p;
        float s;
        if constexpr (Q8) {
            const float4 kk = kv_load_q8_raw(a.k_q + size_t(slot) * kv + hk * D, lane);
            const float dk = kv_load_q8_scale(a.k_s + size_t(slot) * (kv / 32) + hk * 4, lane);
            s = warp_sum((q[0] * kk.x + q[1] * kk.y + q[2] * kk.z + q[3] * kk.w) * dk);
        } else {
            const float4 kk = *reinterpret_cast<const float4 *>(a.k_cache + size_t(slot) * kv + hk * D + lane * 4);
            s = warp_sum(q[0] * kk.x + q[1] * kk.y + q[2] * kk.z + q[3] * kk.w);
        }
        const int d = pos - p;
        float b = 0.f;
        if (d < a.extent) {
            const float * rb = a.rel_bias + size_t(d) * 16;
#pragma unroll
            for (int j = 0; j < 16; ++j) b += rb[j] * r[j];
        }
        s += b - sink;
        const float mn = fmaxf(m, s), corr = expf(m - mn), pr = expf(s - mn);
        l = l * corr + pr;
        if constexpr (Q8) {
            const float4 vv = kv_load_q8_raw(a.v_q + size_t(slot) * kv + hk * D, lane);
            const float pd = pr * kv_load_q8_scale(a.v_s + size_t(slot) * (kv / 32) + hk * 4, lane);
            acc[0] = acc[0] * corr + pd * vv.x; acc[1] = acc[1] * corr + pd * vv.y;
            acc[2] = acc[2] * corr + pd * vv.z; acc[3] = acc[3] * corr + pd * vv.w;
        } else {
            const float4 vv = *reinterpret_cast<const float4 *>(a.v_cache + size_t(slot) * kv + hk * D + lane * 4);
            acc[0] = acc[0] * corr + pr * vv.x; acc[1] = acc[1] * corr + pr * vv.y;
            acc[2] = acc[2] * corr + pr * vv.z; acc[3] = acc[3] * corr + pr * vv.w;
        }
        m = mn;
    }
    if (lane == 0) { s_m[w] = m; s_l[w] = l; }
#pragma unroll
    for (int j = 0; j < 4; ++j) s_acc[w][lane * 4 + j] = acc[j];
    __syncthreads();
    if (w == 0) {
        float M = 0.f;
        for (int i = 0; i < nw; ++i) M = fmaxf(M, s_m[i]);
        float L = expf(-M);
        float o[4] = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < nw; ++i) {
            if (s_m[i] == -INFINITY) continue;
            const float c = expf(s_m[i] - M);
            L += s_l[i] * c;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] += s_acc[i][lane * 4 + j] * c;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t i = size_t(t) * a.n_head * D + h * D + lane * 4 + j;
            a.out[i] = o[j] / L * sigmoidf(a.gate[i]);
        }
    }
}

// out[t] = (add ? add[t] : 0) + (x[t] + conv)/sqrt(5), thread per channel, in token order
__global__ void conv_res_kernel(const float * __restrict__ X, float * __restrict__ hist, const float * __restrict__ w,
                                const float * __restrict__ add, float * __restrict__ out, int n, int T,
                                float * __restrict__ hist_snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float h0 = hist[c], h1 = hist[n + c], h2 = hist[2 * n + c];
    const float * wc = w + size_t(c) * 4;
    for (int t = 0; t < T; ++t) {
        const float x = X[size_t(t) * n + c];
        const float y = (x + h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3]) * rsqrtf(5.f);
        out[size_t(t) * n + c] = add ? add[size_t(t) * n + c] + y : y;
        h0 = h1; h1 = h2; h2 = x;
        if (hist_snaps) {
            float * dst = hist_snaps + size_t(t) * 3 * n;
            dst[c] = h0; dst[n + c] = h1; dst[2 * n + c] = h2;
        }
    }
    hist[c] = h0; hist[n + c] = h1; hist[2 * n + c] = h2;
}

// ---------------------------------------------------------------- router and grouping

__global__ void router_logits_kernel(const float * __restrict__ w, const float * __restrict__ X, float * __restrict__ probs, int n_expert, int n, int T) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int e = gw % n_expert, t = gw / n_expert;
    if (t >= T) return;
    const float4 * row = reinterpret_cast<const float4 *>(w + size_t(e) * n);
    const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(t) * n);
    float d = 0.f;
    for (int i = lane; i < n / 4; i += WARP) {
        const float4 a = row[i], b = xv[i];
        d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    d = warp_sum(d);
    if (lane == 0) probs[size_t(t) * n_expert + e] = sigmoidf(d);
}

__global__ void router_topk_kernel(const float * __restrict__ probs_all, const float * __restrict__ bias, int32_t * __restrict__ ids_all,
                                   float * __restrict__ wts_all, int n_expert, int k, float scale) {
    constexpr int PER = 8;
    const int t = blockIdx.x, lane = threadIdx.x;
    const float * probs = probs_all + size_t(t) * n_expert;
    int32_t * ids = ids_all + size_t(t) * k;
    float * wts = wts_all + size_t(t) * k;
    float sel[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + j * WARP;
        sel[j] = e < n_expert ? probs[e] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
        float best = -INFINITY;
        int bi = n_expert - 1;  // (an all-NaN row still yields a valid id)
#pragma unroll
        for (int j = 0; j < PER; ++j) if (sel[j] > best) { best = sel[j]; bi = lane + j * WARP; }
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        if ((bi % WARP) == lane) sel[bi / WARP] = -INFINITY;
        const float p = probs[bi];
        sum += p;
        if (lane == 0) { ids[i] = bi; wts[i] = p; }
    }
    __syncwarp();
    if (lane < k) wts[lane] *= scale / (sum + 1e-20f);
}

// entries of each expert in token order: offsets[e] .. offsets[e+1] index list[] = t * k + slot
// 128-thread parallel histogram + prefix sum + scatter (+ optional compacted active-expert list)
__global__ void group_kernel(const int32_t * __restrict__ ids, int T, int k, int n_expert, int * __restrict__ offsets,
                             int * __restrict__ list, int * __restrict__ active, int * __restrict__ n_active) {
    const int total = T * k;
    const int tid = threadIdx.x;
    if (n_expert <= 128 && total <= 2048) {
        __shared__ int32_t s_ids[2048];
        __shared__ int s_wsum[4], s_asum[4];
        for (int i = tid; i < total; i += 128) s_ids[i] = ids[i];
        __syncthreads();
        int cnt = 0;
        if (tid < n_expert) {
            for (int i = 0; i < total; ++i) cnt += (s_ids[i] == tid);
        }
        // Warp-level inclusive scan of cnt and is_act = (cnt > 0)
        const int lane = tid & 31, warp = tid >> 5;
        int v = cnt, a = cnt > 0 ? 1 : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int nv = __shfl_up_sync(0xffffffffu, v, o);
            const int na = __shfl_up_sync(0xffffffffu, a, o);
            if (lane >= o) { v += nv; a += na; }
        }
        if (lane == 31) { s_wsum[warp] = v; s_asum[warp] = a; }
        __syncthreads();
        int base_v = 0, base_a = 0;
        for (int w = 0; w < warp; ++w) { base_v += s_wsum[w]; base_a += s_asum[w]; }
        const int excl_v = base_v + v - cnt;
        const int excl_a = base_a + a - (cnt > 0 ? 1 : 0);
        if (tid < n_expert) {
            offsets[tid] = excl_v;
            if (active && cnt > 0) active[excl_a] = tid;
        }
        if (tid == n_expert - 1) {
            offsets[n_expert] = base_v + v;
            if (n_active) *n_active = base_a + a;
        }
        if (tid < n_expert && cnt > 0) {
            int o = excl_v;
            for (int i = 0; i < total; ++i)
                if (s_ids[i] == tid) list[o++] = i;
        }
        return;
    }
    if (tid != 0) return;
    for (int e = 0; e <= n_expert; ++e) offsets[e] = 0;
    for (int i = 0; i < total; ++i) offsets[ids[i] + 1]++;
    int na = 0;
    for (int e = 0; e < n_expert; ++e) {
        if (active && offsets[e + 1] > 0) active[na++] = e;
        offsets[e + 1] += offsets[e];
    }
    if (n_active) *n_active = na;
    for (int e = 0; e < n_expert; ++e) {
        int o = offsets[e];
        for (int i = 0; i < total; ++i) if (ids[i] == e) list[o++] = i;
    }
}

// ---------------------------------------------------------------- grouped experts

// 8 unscaled weights of row r, lane-step w, plus their sub-block scale (factoring *sc out of multi-token dots)
template <bool NVFP4>
__device__ __forceinline__ float weights8_unscaled(const uint8_t * m, int rows, int n, int r, int w, const float * lut, float * wv) {
    if constexpr (NVFP4) {
        const uint32_t qw = __ldg(reinterpret_cast<const uint32_t *>(m) + size_t(r) * (n / 8) + w);
        const float sc = ue4m3(__ldg(m + size_t(rows) * n / 2 + size_t(r) * (n / 16) + (w >> 1)));
#pragma unroll
        for (int i = 0; i < 8; ++i) wv[i] = lut[(qw >> (4 * i)) & 0xF];
        return sc;
    } else {
        const uint2 qw = __ldg(reinterpret_cast<const uint2 *>(m + size_t(r) * n) + w);
        const float sc = __half2float(__ldg(reinterpret_cast<const __half *>(m + size_t(rows) * n) + size_t(r) * (n / 32) + (w >> 2)));
#pragma unroll
        for (int i = 0; i < 4; ++i) { wv[i] = float(int8_t(qw.x >> (8 * i))); wv[4 + i] = float(int8_t(qw.y >> (8 * i))); }
        return sc;
    }
}

template <bool NVFP4>
__device__ __forceinline__ void weights8(const uint8_t * m, int rows, int n, int r, int w, const float * lut, float * wv) {
    const float sc = weights8_unscaled<NVFP4>(m, rows, n, r, w, lut, wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] *= sc;
}

__device__ __forceinline__ float dot8(const float * wv, const float * x) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(x)), b = __ldg(reinterpret_cast<const float4 *>(x) + 1);
    return wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
}

__device__ __forceinline__ float dot8_ab(const float * wv, const float4 & a, const float4 & b) {
    return wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
}

// h[entry][r] = situ(s (W_gu[e][r] . x[t]), s (W_gu[e][r + n_ff] . x[t])) for every entry of expert e; grid (blocks, n_expert or max_active)
template <bool NVFP4>
__global__ void __launch_bounds__(256) gate_up_fast_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                           const int * __restrict__ offsets, const int * __restrict__ list, int k,
                                                           const float * __restrict__ X, float * __restrict__ H, int n_embd, int n_ff,
                                                           float beta, float lb, const int * __restrict__ active,
                                                           const int * __restrict__ n_active) {
    int e = blockIdx.y;
    if (active) {
        if (e >= *n_active) return;
        e = active[e];
    }
    const int b0 = offsets[e], cnt = offsets[e + 1] - b0;
    const uint8_t * m = experts[e];
    if (!m || cnt == 0) return;
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = k_fp4_init[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x % WARP, steps = n_embd / 256;
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP);
    if (cnt == 1) {
        const int en0 = list[b0];
        const float * x0 = X + size_t(en0 / k) * n_embd;
        if (n_embd == 2304) {
            float xr[9][8];
#pragma unroll
            for (int st = 0; st < 9; ++st) {
                const float4 * xp = reinterpret_cast<const float4 *>(x0 + (st * WARP + lane) * 8);
                const float4 a = __ldg(xp), b = __ldg(xp + 1);
                xr[st][0] = a.x; xr[st][1] = a.y; xr[st][2] = a.z; xr[st][3] = a.w;
                xr[st][4] = b.x; xr[st][5] = b.y; xr[st][6] = b.z; xr[st][7] = b.w;
            }
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
                float ag = 0.f, au = 0.f;
#pragma unroll
                for (int st = 0; st < 9; ++st) {
                    const int w = st * WARP + lane;
                    float wg[8], wu[8];
                    const float sg = weights8_unscaled<NVFP4>(m, 2 * n_ff, 2304, r, w, lut, wg);
                    const float su = weights8_unscaled<NVFP4>(m, 2 * n_ff, 2304, r + n_ff, w, lut, wu);
                    float dg = 0.f, du = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) { dg += wg[i] * xr[st][i]; du += wu[i] * xr[st][i]; }
                    ag += dg * sg;
                    au += du * su;
                }
                const float g = warp_sum(ag), u = warp_sum(au);
                if (lane == 0) H[size_t(en0) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
            return;
        }
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
            float ag = 0.f, au = 0.f;
            for (int st = 0; st < steps; ++st) {
                const int w = st * WARP + lane;
                float wg[8], wu[8];
                const float sg = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r, w, lut, wg);
                const float su = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r + n_ff, w, lut, wu);
                const float4 * xp = reinterpret_cast<const float4 *>(x0 + w * 8);
                const float4 a = __ldg(xp), b = __ldg(xp + 1);
                ag += dot8_ab(wg, a, b) * sg;
                au += dot8_ab(wu, a, b) * su;
            }
            const float g = warp_sum(ag), u = warp_sum(au);
            if (lane == 0) H[size_t(en0) * n_ff + r] = situ(g * s, u * s, beta, lb);
        }
        return;
    }
    if (cnt <= 4) {
        int en[4];
        const float * xp[4];
        for (int j = 0; j < cnt; ++j) { en[j] = list[b0 + j]; xp[j] = X + size_t(en[j] / k) * n_embd; }
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
            float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
            for (int st = 0; st < steps; ++st) {
                const int w = st * WARP + lane;
                float wg[8], wu[8];
                const float sg = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r, w, lut, wg);
                const float su = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r + n_ff, w, lut, wu);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j < cnt) {
                        const float4 * xv = reinterpret_cast<const float4 *>(xp[j] + w * 8);
                        const float4 a = __ldg(xv), b = __ldg(xv + 1);
                        ag[j] += dot8_ab(wg, a, b) * sg;
                        au[j] += dot8_ab(wu, a, b) * su;
                    }
                }
            }
            for (int j = 0; j < cnt; ++j) {
                const float g = warp_sum(ag[j]), u = warp_sum(au[j]);
                if (lane == 0) H[size_t(en[j]) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
        }
        return;
    }
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
        for (int j0 = 0; j0 < cnt; j0 += TG) {
            float ag[TG], au[TG];
#pragma unroll
            for (int j = 0; j < TG; ++j) ag[j] = au[j] = 0.f;
            for (int st = 0; st < steps; ++st) {
                const int w = st * WARP + lane;
                float wg[8], wu[8];
                const float sg = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r, w, lut, wg);
                const float su = weights8_unscaled<NVFP4>(m, 2 * n_ff, n_embd, r + n_ff, w, lut, wu);
#pragma unroll
                for (int j = 0; j < TG; ++j) {
                    if (j0 + j < cnt) {
                        const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(list[b0 + j0 + j] / k) * n_embd + w * 8);
                        const float4 a = __ldg(xv), b = __ldg(xv + 1);
                        ag[j] += dot8_ab(wg, a, b) * sg;
                        au[j] += dot8_ab(wu, a, b) * su;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < TG; ++j) {
                if (j0 + j < cnt) {
                    const float g = warp_sum(ag[j]), u = warp_sum(au[j]);
                    if (lane == 0) H[size_t(list[b0 + j0 + j]) * n_ff + r] = situ(g * s, u * s, beta, lb);
                }
            }
        }
    }
}

// partial[entry][r] = w[entry] s (W_dn[e][r] . h[entry]); LPR 16 lanes per row, two rows per warp
template <bool NVFP4>
__global__ void __launch_bounds__(256) down_fast_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                        const int * __restrict__ offsets, const int * __restrict__ list,
                                                        const float * __restrict__ wts, const float * __restrict__ H,
                                                        float * __restrict__ partial, int n_embd, int n_ff,
                                                        const int * __restrict__ active, const int * __restrict__ n_active) {
    constexpr int LPR = 16;
    int e = blockIdx.y;
    if (active) {
        if (e >= *n_active) return;
        e = active[e];
    }
    const int b0 = offsets[e], cnt = offsets[e + 1] - b0;
    const uint8_t * m = experts[e];
    if (!m || cnt == 0) return;
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = k_fp4_init[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x % WARP, jl = lane % LPR, steps = n_ff / (8 * LPR);
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP) * 2;
    if (cnt == 1) {
        const int en0 = list[b0];
        const float ws0 = wts[en0] * s;
        const float * h0 = H + size_t(en0) * n_ff;
        if (n_ff == 1152) {
            float hr[9][8];
#pragma unroll
            for (int st = 0; st < 9; ++st) {
                const float4 * hp = reinterpret_cast<const float4 *>(h0 + (st * LPR + jl) * 8);
                const float4 a = __ldg(hp), b = __ldg(hp + 1);
                hr[st][0] = a.x; hr[st][1] = a.y; hr[st][2] = a.z; hr[st][3] = a.w;
                hr[st][4] = b.x; hr[st][5] = b.y; hr[st][6] = b.z; hr[st][7] = b.w;
            }
            for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * 2 + lane / LPR; r < n_embd; r += stride) {
                float a = 0.f;
#pragma unroll
                for (int st = 0; st < 9; ++st) {
                    const int w = st * LPR + jl;
                    float wv[8];
                    const float sc = weights8_unscaled<NVFP4>(m, n_embd, 1152, r, w, lut, wv);
                    float d = 0.f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) d += wv[i] * hr[st][i];
                    a += d * sc;
                }
                const float v = group_sum<LPR>(a);
                if (jl == 0) partial[size_t(en0) * n_embd + r] = ws0 * v;
            }
            return;
        }
        for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * 2 + lane / LPR; r < n_embd; r += stride) {
            float a = 0.f;
            for (int st = 0; st < steps; ++st) {
                const int w = st * LPR + jl;
                float wv[8];
                const float sc = weights8_unscaled<NVFP4>(m, n_embd, n_ff, r, w, lut, wv);
                a += dot8(wv, h0 + w * 8) * sc;
            }
            const float v = group_sum<LPR>(a);
            if (jl == 0) partial[size_t(en0) * n_embd + r] = ws0 * v;
        }
        return;
    }
    if (cnt <= 4) {
        int en[4];
        float ws[4];
        const float * hp[4];
        for (int j = 0; j < cnt; ++j) { en[j] = list[b0 + j]; ws[j] = wts[en[j]] * s; hp[j] = H + size_t(en[j]) * n_ff; }
        for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * 2 + lane / LPR; r < n_embd; r += stride) {
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int st = 0; st < steps; ++st) {
                const int w = st * LPR + jl;
                float wv[8];
                const float sc = weights8_unscaled<NVFP4>(m, n_embd, n_ff, r, w, lut, wv);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (j < cnt) a[j] += dot8(wv, hp[j] + w * 8) * sc;
            }
            for (int j = 0; j < cnt; ++j) {
                const float v = group_sum<LPR>(a[j]);
                if (jl == 0) partial[size_t(en[j]) * n_embd + r] = ws[j] * v;
            }
        }
        return;
    }
    for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * 2 + lane / LPR; r < n_embd; r += stride) {
        for (int j0 = 0; j0 < cnt; j0 += TG) {
            float a[TG];
#pragma unroll
            for (int j = 0; j < TG; ++j) a[j] = 0.f;
            for (int st = 0; st < steps; ++st) {
                const int w = st * LPR + jl;
                float wv[8];
                const float sc = weights8_unscaled<NVFP4>(m, n_embd, n_ff, r, w, lut, wv);
#pragma unroll
                for (int j = 0; j < TG; ++j)
                    if (j0 + j < cnt) a[j] += dot8(wv, H + size_t(list[b0 + j0 + j]) * n_ff + w * 8) * sc;
            }
#pragma unroll
            for (int j = 0; j < TG; ++j) {
                if (j0 + j < cnt) {
                    const float v = group_sum<LPR>(a[j]);
                    if (jl == 0) {
                        const int en = list[b0 + j0 + j];
                        partial[size_t(en) * n_embd + r] = wts[en] * s * v;
                    }
                }
            }
        }
    }
}

// y[t][r] = sum over the token's k entries
__global__ void sum_entries_kernel(const float * __restrict__ partial, float * __restrict__ Y, int n_embd, int k, int T) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(T) * n_embd) return;
    const size_t t = i / n_embd, r = i % n_embd;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += partial[(t * k + j) * n_embd + r];
    Y[i] = acc;
}

}  // namespace

// ---------------------------------------------------------------- launchers

void embed(const uint8_t * rows, const float * w, float * out, int n, float eps, int T, cudaStream_t st, int emb_type) {
    const size_t rb = emb_type == 8 ? size_t(n / 32) * 34
                    : emb_type == 12 ? size_t(n / 256) * 144
                    : emb_type == 13 ? size_t(n / 256) * 176
                    : size_t(n / 256) * 210;
    embed_kernel<<<T, 256, sizeof(float) * n, st>>>(rows, w, out, n, eps, emb_type, rb);
}
void mix(const MixStates & s, const float * query, int n_q, float * out, int n, int T, cudaStream_t st) {
    mix_kernel<<<T, 768, 0, st>>>(s, query, n_q, out, n);
}
void control(const ControlArgs & a_in, int count, size_t cur_stride, size_t msg_stride, size_t out_stride, int n, int rank, int T,
             float * scratch, cudaStream_t st) {
    ControlArgs a = a_in;
    a.count = count;
    const int wg = T * count * rank, wo = T * count * n;
    control_g_kernel<<<(wg + 7) / 8, 256, 0, st>>>(a, cur_stride, msg_stride, scratch, n, rank, T);
    control_out_kernel<<<(wo + 7) / 8, 256, 0, st>>>(a, cur_stride, msg_stride, out_stride, scratch, n, rank, T);
}
void rmsnorm(const NormArgs & a, int count, size_t x_stride, size_t out_stride, int n, float eps, int T, cudaStream_t st) {
    rmsnorm_kernel<<<dim3(count, T), 512, 0, st>>>(a, x_stride, out_stride, n, eps);
}
int matvec_q8r(const void * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st) {
    const int blocks = (rows + 7) / 8, grid = blocks < 1024 ? blocks : 1024;
    if (T <= 8) {
        if (n == 2304) matvec_q8r_kernel<2304, 8><<<grid, 256, 0, st>>>((const uint8_t *) w, X, Y, rows, T);
        else if (n == 3072) matvec_q8r_kernel<3072, 8><<<grid, 256, 0, st>>>((const uint8_t *) w, X, Y, rows, T);
        else return (int) cudaErrorInvalidValue;
    } else {
        if (n == 2304) matvec_q8r_kernel<2304, TG><<<grid, 256, 0, st>>>((const uint8_t *) w, X, Y, rows, T);
        else if (n == 3072) matvec_q8r_kernel<3072, TG><<<grid, 256, 0, st>>>((const uint8_t *) w, X, Y, rows, T);
        else return (int) cudaErrorInvalidValue;
    }
    return (int) cudaGetLastError();
}
int matvec_q8r_multi(int count, const void * const * ws, const float * X, float * const * ys, const int * rows, int n, int T,
                     cudaStream_t st) {
    if (count < 1 || count > 5 || T < 1 || T > 8) return (int) cudaErrorInvalidValue;
    MultiQ8PfArgs a{};
    a.count = count;
    a.off[0] = 0;
    for (int i = 0; i < count; ++i) {
        a.w[i] = (const uint8_t *) ws[i];
        a.y[i] = ys[i];
        a.rows[i] = rows[i];
        a.off[i + 1] = a.off[i] + rows[i];
    }
    const int blocks = (a.off[count] + 7) / 8, grid = blocks < 1024 ? blocks : 1024;
    if (n == 2304) matvec_q8r_multi_kernel<2304, 8><<<grid, 256, 0, st>>>(a, X, T);
    else if (n == 3072) matvec_q8r_multi_kernel<3072, 8><<<grid, 256, 0, st>>>(a, X, T);
    else return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}
void matvec_f32(const float * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st) {
    const int warps = rows * T;
    matvec_f32_kernel<<<(warps + 7) / 8, 256, 0, st>>>(w, X, Y, rows, n, T);
}
void conv(const float * X, float * hist, const float * w, const float * dyn, int heads, float * out, int C, int T, float norm, cudaStream_t st,
          float * hist_snaps) {
    conv_kernel<<<(C + 255) / 256, 256, 0, st>>>(X, hist, w, dyn, heads, out, C, T, norm, hist_snaps);
}
void attn_prep2(const AttnPrepArgs & a, const float * k_in, const float * v_in, int pos0, int T, cudaStream_t st, const int * d_pos) {
    attn_prep2_kernel<<<T, 512, 0, st>>>(a, k_in, v_in, pos0, d_pos);
}
void attention(const AttnArgs & a, int pos0, int T, cudaStream_t st, const int * d_pos) {
    if (a.kv_q8) attention_kernel<true><<<dim3(a.n_head, T), 256, 0, st>>>(a, pos0, d_pos);
    else attention_kernel<false><<<dim3(a.n_head, T), 256, 0, st>>>(a, pos0, d_pos);
}
void conv_res(const float * X, float * hist, const float * w, const float * add, float * out, int n, int T, cudaStream_t st,
              float * hist_snaps) {
    conv_res_kernel<<<(n + 255) / 256, 256, 0, st>>>(X, hist, w, add, out, n, T, hist_snaps);
}
void router(const float * w, const float * bias, const float * X, int32_t * ids, float * wts, float * probs, int n_expert, int n, int k,
            float scale, int T, cudaStream_t st) {
    const int warps = n_expert * T;
    router_logits_kernel<<<(warps + 7) / 8, 256, 0, st>>>(w, X, probs, n_expert, n, T);
    router_topk_kernel<<<T, 32, 0, st>>>(probs, bias, ids, wts, n_expert, k, scale);
}
void group(const int32_t * ids, int T, int k, int n_expert, int * offsets, int * list, cudaStream_t st,
           int * active, int * n_active) {
    group_kernel<<<1, 128, 0, st>>>(ids, T, k, n_expert, offsets, list, active, n_active);
}
int experts_fast(int type, const void * const * gu, const void * const * dn, const float * gu_scale, const float * dn_scale,
                 const int * offsets, const int * list, int k, const float * wts, const float * X, float * H, float * partial,
                 int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st,
                 const int * active, const int * n_active, int max_active) {
    const int ny = (active && n_active && max_active > 0) ? (max_active < n_expert ? max_active : n_expert) : n_expert;
    const dim3 grid(36, ny);
    const auto * g = (const uint8_t * const *) gu;
    const auto * d = (const uint8_t * const *) dn;
    if (type == 0) {
        gate_up_fast_kernel<true><<<grid, 256, 0, st>>>(g, gu_scale, offsets, list, k, X, H, n_embd, n_ff, beta, lb, active, n_active);
        down_fast_kernel<true><<<grid, 256, 0, st>>>(d, dn_scale, offsets, list, wts, H, partial, n_embd, n_ff, active, n_active);
    } else {
        gate_up_fast_kernel<false><<<grid, 256, 0, st>>>(g, gu_scale, offsets, list, k, X, H, n_embd, n_ff, beta, lb, active, n_active);
        down_fast_kernel<false><<<grid, 256, 0, st>>>(d, dn_scale, offsets, list, wts, H, partial, n_embd, n_ff, active, n_active);
    }
    return (int) cudaGetLastError();
}
void sum_entries(const float * partial, float * Y, int n_embd, int k, int T, cudaStream_t st) {
    const size_t n = size_t(T) * n_embd;
    sum_entries_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(partial, Y, n_embd, k, T);
}

}  // namespace pf
}  // namespace ik

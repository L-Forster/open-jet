// Per-token kernels of the Inkling 2 Mini decode step (everything but the quantized matrix products, which live in
// expert_kernels.cu). Each mirrors an op of reference/inkling_ref.py, in fp32. Kernels that depend on the position
// read it from device memory, so one captured CUDA graph serves every token.

#include "ik_kernels.cuh"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace ik {

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block; every thread gets the result. red: shared scratch of 32 floats
__device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = (blockDim.x + WARP - 1) / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// ---------------------------------------------------------------- embedding

__device__ __forceinline__ void emb_k_scale_min(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ float emb_dequant_elem(const uint8_t * row, int i, int emb_type) {
    if (emb_type == 8) {  // Q8_0: 34 B / 32 vals
        const uint8_t * blk = row + (i / 32) * 34;
        return __half2float(*reinterpret_cast<const __half *>(blk)) * float(int8_t(blk[2 + (i & 31)]));
    }
    if (emb_type == 12) {  // Q4_K: 144 B / 256 vals
        const uint8_t * b = row + (i / 256) * 144;
        const int rem = i & 255, is = rem >> 5, p = is >> 1, l = rem & 31;
        int sc, m;
        emb_k_scale_min(is, b + 4, sc, m);
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(b + 2));
        const uint8_t q = b[16 + 32 * p + l];
        return d * float(sc) * float((is & 1) ? (q >> 4) : (q & 0xF)) - dmin * float(m);
    }
    if (emb_type == 13) {  // Q5_K: 176 B / 256 vals
        const uint8_t * b = row + (i / 256) * 176;
        const int rem = i & 255, is = rem >> 5, p = is >> 1, l = rem & 31;
        int sc, m;
        emb_k_scale_min(is, b + 4, sc, m);
        const float d = __half2float(*reinterpret_cast<const __half *>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(b + 2));
        const uint8_t ql = b[48 + 32 * p + l];
        const int lo = (is & 1) ? (ql >> 4) : (ql & 0xF);
        const int hi = ((b[16 + l] >> is) & 1) << 4;
        return d * float(sc) * float(lo + hi) - dmin * float(m);
    }
    // Q6_K (14): 210 B / 256 vals
    const uint8_t * b = row + (i / 256) * 210;
    const int rem = i & 255, k8 = rem >> 5, half = k8 >> 2, k = k8 & 3, l = rem & 31;
    const uint8_t * ql = b + 64 * half;
    const uint8_t * qh = b + 128 + 32 * half;
    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * half;
    const int lo = (k & 1) ? ql[l + 32] : ql[l];
    const int q = (((k & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
    return __half2float(*reinterpret_cast<const __half *>(b + 208)) * float(sc[(l >> 4) + 2 * k] * q);
}

// out = rms(dequant row) * w
__global__ void embed_kernel(const uint8_t * __restrict__ row, const float * __restrict__ w, float * __restrict__ out, int n, float eps, int emb_type) {
    __shared__ float red[32];
    extern __shared__ float x[];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = emb_dequant_elem(row, i, emb_type);
        x[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

// ---------------------------------------------------------------- depth mixing

// out[q] = sum_s softmax_s(query_q . rms(state_s) / sqrt(n)) * rms(state_s), for n_q queries.
// One pass accumulates every sum of squares and query dot product; one reduction; then the weighted sum.
__global__ void mix_kernel(MixStates st, const float * __restrict__ query, int n_q, float * __restrict__ out, int n) {
    constexpr int NS = MAX_STATES, NQ = 4, NACC = NS * (NQ + 1);
    __shared__ float red[NACC][32];
    __shared__ float tot[NACC];
    __shared__ float coef[NQ][NS];
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
                const float x = st.p[s][i];
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
        float t = 0.f;
        for (int k = 0; k < nw; ++k) t += red[j][k];
        tot[j] = t;
    }
    __syncthreads();
    if (threadIdx.x < n_q) {
        const int q = threadIdx.x;
        float sc[NS], mx = -INFINITY;
        for (int s = 0; s < st.n; ++s) {
            const float inv = rsqrtf(tot[s] / n + 1e-6f);
            sc[s] = tot[NS + q * NS + s] * inv * rsqrtf(float(n));
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < st.n; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < st.n; ++s) coef[q][s] = sc[s] / sum * rsqrtf(tot[s] / n + 1e-6f);  // fold the state's rms in
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xs[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) xs[s] = s < st.n ? st.p[s][i] : 0.f;
        for (int q = 0; q < n_q; ++q) {
            float v = 0.f;
            // only the st.n states that exist: coef[q][s >= st.n] is uninitialised shared memory, and 0 * NaN = NaN
            for (int s = 0; s < st.n; ++s) v += coef[q][s] * xs[s];
            out[q * n + i] = v;
        }
    }
}

// ---------------------------------------------------------------- loop control

// out = cur - forget*cur + gain*msg, with g = down . [rms(cur), rms(msg)], gain = tanh(G g), forget = tanh(F g).
// Kernel 1: one warp per (controller, row of down) computes g; each warp also derives the two rms factors (the inputs
// are 9 KB, read from L2). Kernel 2: one warp per (controller, output row) reads its 64-wide gain and forget rows.
__global__ void control_g_kernel(ControlArgs a, float * __restrict__ g, int n, int rank) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int c = gw / rank, r = gw % rank;
    if (c >= a.count) return;
    const float * cur = a.cur[c];
    const float * msg = a.msg[c];
    const float * row = a.down[c] + size_t(r) * 2 * n;
    float s1 = 0.f, s2 = 0.f, d1 = 0.f, d2 = 0.f;
    for (int i = lane; i < n; i += WARP) {
        const float x = cur[i], y = msg[i];
        s1 += x * x; s2 += y * y;
        d1 += row[i] * x; d2 += row[n + i] * y;
    }
    s1 = warp_sum(s1); s2 = warp_sum(s2); d1 = warp_sum(d1); d2 = warp_sum(d2);
    if (lane == 0) g[c * 64 + r] = d1 * rsqrtf(s1 / n + 1e-6f) + d2 * rsqrtf(s2 / n + 1e-6f);
}

__global__ void control_out_kernel(ControlArgs a, const float * __restrict__ g, int n, int rank) {
    const int lane = threadIdx.x % WARP;
    const int gw = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int c = gw / n, i = gw % n;
    if (c >= a.count) return;
    const float * gr = a.gain[c] + size_t(i) * rank;
    const float * fr = a.forget[c] + size_t(i) * rank;
    float ga = 0.f, fo = 0.f;
    for (int k = lane; k < rank; k += WARP) { ga += gr[k] * g[c * 64 + k]; fo += fr[k] * g[c * 64 + k]; }
    ga = tanhf(warp_sum(ga));
    fo = tanhf(warp_sum(fo));
    if (lane == 0) {
        const float x = a.cur[c][i];
        a.out[c][i] = x - fo * x + ga * a.msg[c][i];
    }
}

// ---------------------------------------------------------------- norms and small products

// out[v] = rms(x[v]) * w[v] for up to 3 vectors (blockIdx.x)
__global__ void rmsnorm_kernel(NormArgs a, int n, float eps) {
    __shared__ float red[32];
    const int v = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += a.x[v][i] * a.x[v][i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) a.out[v][i] = a.x[v][i] * inv * a.w[v][i];
}

// y = W x for a small fp32 or bf16 matrix [rows][n]; one warp per row
template <typename T>
__global__ void matvec_small_kernel(const T * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, int rows, int n) {
    const int lane = threadIdx.x % WARP;
    const int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (r >= rows) return;
    const T * row = w + size_t(r) * n;
    float d = 0.f;
    for (int i = lane; i < n; i += WARP) d += float(row[i]) * x[i];
    d = warp_sum(d);
    if (lane == 0) y[r] = d;
}

// ---------------------------------------------------------------- attention preparation

// causal short convolution with residual: out[c] = (x[c] + sum_j xin[j][c] w[c][j] (+ dynamic taps)) / sqrt(norm),
// xin = [hist[0], hist[1], hist[2], x]; shifts the history
__device__ __forceinline__ float conv1(float x, float * hist, int C, int c, const float * w, int n_conv) {
    const float h0 = hist[c], h1 = hist[C + c], h2 = hist[2 * C + c];
    const float * wc = w + size_t(c) * n_conv;
    float out = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x * wc[3];
    hist[c] = h1; hist[C + c] = h2; hist[2 * C + c] = x;
    return out;
}

// one block; thread per channel for k and v (1024 each), then per-head clips. Writes q, r (clipped, tau-scaled)
// and the new k, v row into the slot's cache at the current position.
__global__ void attn_prep_kernel(AttnPrepArgs a) {
    extern __shared__ float sm[];  // k[kv], v[kv]
    const int kv = a.n_head_kv * a.d_head;
    float * ks = sm;
    float * vs = sm + kv;
    const int pos = *a.pos;
    const float norm_k = rsqrtf(float((a.kdyn ? 2 : 1) * 4 + 1));
    for (int c = threadIdx.x; c < kv; c += blockDim.x) {
        const int head = c / a.d_head;
        // the history holds inputs before the time step: [t-3, t-2, t-1]
        const float hk0 = a.hist_k[c], hk1 = a.hist_k[kv + c], hk2 = a.hist_k[2 * kv + c];
        const float hv0 = a.hist_v[c], hv1 = a.hist_v[kv + c], hv2 = a.hist_v[2 * kv + c];
        const float xk = a.k[c], xv = a.v[c];
        float ok = conv1(xk, a.hist_k, kv, c, a.k_conv, 4);
        float ov = conv1(xv, a.hist_v, kv, c, a.v_conv, 4);
        if (a.kdyn) {  // tap lag reads the input lag steps back: lag 0 = current, 3 = oldest
            const float * dk = a.kdyn + head * 4;
            const float * dv = a.vdyn + head * 4;
            ok += dk[0] * xk + dk[1] * hk2 + dk[2] * hk1 + dk[3] * hk0;
            ov += dv[0] * xv + dv[1] * hv2 + dv[2] * hv1 + dv[3] * hv0;
        }
        ks[c] = (xk + ok) * norm_k;
        vs[c] = (xv + ov) * norm_k;
    }
    __syncthreads();
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int slot = a.ring > 0 ? pos % a.ring : pos;
    // k: L2 clip per head; v: rms per head (d_head == 128: 4 contiguous elements per lane)
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
    // q: scale by q_clip, L2 clip, tau; r: scale by rel_clip, L2 clip, tau
    const float tau = a.global ? 1.f + a.log_alpha * logf(fmaxf((pos + 1.f) / a.log_floor, 1.f)) : 1.f;
    for (int h = w; h < a.n_head; h += nw) {
        float sq = 0.f;
        for (int i = lane; i < a.d_head; i += WARP) {
            const float x = a.q[h * a.d_head + i] * a.q_clip[h * a.d_head + i];
            sq += x * x;
        }
        sq = warp_sum(sq);
        const float fq = fminf(fmaxf(a.qk_clip / (sqrtf(sq) + 1e-6f), 0.f), 1.f) * tau;
        for (int i = lane; i < a.d_head; i += WARP) a.q_out[h * a.d_head + i] = a.q[h * a.d_head + i] * a.q_clip[h * a.d_head + i] * fq;
        float sr = 0.f;
        for (int i = lane; i < a.n_rel; i += WARP) {
            const float x = a.r[h * a.n_rel + i] * a.rel_clip[h * a.n_rel + i];
            sr += x * x;
        }
        sr = warp_sum(sr);
        const float fr = fminf(fmaxf(a.rel_clip_thr / (sqrtf(sr) + 1e-6f), 0.f), 1.f) * tau;
        for (int i = lane; i < a.n_rel; i += WARP) a.r_out[h * a.n_rel + i] = a.r[h * a.n_rel + i] * a.rel_clip[h * a.n_rel + i] * fr;
    }
}

// ---------------------------------------------------------------- attention

// Split-K decode attention ("flash decoding"): the visible positions of each head are divided among SPLITS blocks; each
// keeps an online softmax over its range (bias(d) = rel_bias[min(d, extent)] . r_h, zero past the extent, minus the
// sink logit) and writes (max, sum, weighted V) to scratch; attention_combine merges them with the sink, which adds
// exp(0) to the denominator, and applies the output gate. The split count is fixed, so one captured graph serves
// every position; each block derives its range from the device position.
constexpr int SPLITS = 16;

template <bool Q8>
__global__ void attention_split_kernel(AttnArgs a, float * __restrict__ part) {
    constexpr int D = 128;
    __shared__ float s_m[32], s_l[32];
    __shared__ float s_acc[32][D];
    const int h = blockIdx.x, sp = blockIdx.y, hk = h / (a.n_head / a.n_head_kv);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int pos = *a.pos;
    const int lo0 = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int n = pos - lo0 + 1, per = (n + SPLITS - 1) / SPLITS;
    const int lo = lo0 + sp * per, hi = min(pos, lo + per - 1);
    const int kv = a.n_head_kv * D;
    float q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = a.q[h * D + lane * 4 + j] * rsqrtf(float(D));
    float r[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) r[j] = a.r[h * 16 + j];
    const float sink = a.sinks[h];
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = lo + w; t <= hi; t += nw) {
        const int slot = a.ring > 0 ? t % a.ring : t;
        float s;
        if constexpr (Q8) {
            const float4 kk = kv_load_q8_raw(a.k_q + size_t(slot) * kv + hk * D, lane);
            const float dk = kv_load_q8_scale(a.k_s + size_t(slot) * (kv / 32) + hk * 4, lane);
            s = warp_sum((q[0] * kk.x + q[1] * kk.y + q[2] * kk.z + q[3] * kk.w) * dk);
        } else {
            const float4 kk = *reinterpret_cast<const float4 *>(a.k_cache + size_t(slot) * kv + hk * D + lane * 4);
            s = warp_sum(q[0] * kk.x + q[1] * kk.y + q[2] * kk.z + q[3] * kk.w);
        }
        const int d = pos - t;
        float b = 0.f;
        if (d < a.extent) {
            const float * rb = a.rel_bias + size_t(d) * 16;
#pragma unroll
            for (int j = 0; j < 16; ++j) b += rb[j] * r[j];
        }
        s += b - sink;
        const float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
        l = l * corr + p;
        if constexpr (Q8) {
            const float4 vv = kv_load_q8_raw(a.v_q + size_t(slot) * kv + hk * D, lane);
            const float pd = p * kv_load_q8_scale(a.v_s + size_t(slot) * (kv / 32) + hk * 4, lane);
            acc[0] = acc[0] * corr + pd * vv.x; acc[1] = acc[1] * corr + pd * vv.y;
            acc[2] = acc[2] * corr + pd * vv.z; acc[3] = acc[3] * corr + pd * vv.w;
        } else {
            const float4 vv = *reinterpret_cast<const float4 *>(a.v_cache + size_t(slot) * kv + hk * D + lane * 4);
            acc[0] = acc[0] * corr + p * vv.x; acc[1] = acc[1] * corr + p * vv.y;
            acc[2] = acc[2] * corr + p * vv.z; acc[3] = acc[3] * corr + p * vv.w;
        }
        m = mn;
    }
    if (lane == 0) { s_m[w] = m; s_l[w] = l; }
#pragma unroll
    for (int j = 0; j < 4; ++j) s_acc[w][lane * 4 + j] = acc[j];
    __syncthreads();
    if (w == 0) {
        float M = -INFINITY;
        for (int i = 0; i < nw; ++i) M = fmaxf(M, s_m[i]);
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
        if (M != -INFINITY) {
            for (int i = 0; i < nw; ++i) {
                if (s_m[i] == -INFINITY) continue;
                const float c = expf(s_m[i] - M);
                L += s_l[i] * c;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] += s_acc[i][lane * 4 + j] * c;
            }
        }
        float * out = part + (size_t(h) * SPLITS + sp) * (D + 2);
        if (lane == 0) { out[0] = M; out[1] = L; }
#pragma unroll
        for (int j = 0; j < 4; ++j) out[2 + lane * 4 + j] = o[j];
    }
}

// one warp per head: merge the splits and the sink logit 0, then out = (sum p v) / L * sigmoid(gate)
__global__ void attention_combine_kernel(AttnArgs a, const float * __restrict__ part) {
    constexpr int D = 128;
    const int h = blockIdx.x, lane = threadIdx.x;
    float M = 0.f;  // the sink logit 0 takes part in the max
    for (int sp = 0; sp < SPLITS; ++sp) M = fmaxf(M, part[(size_t(h) * SPLITS + sp) * (D + 2)]);
    float L = expf(-M), o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int sp = 0; sp < SPLITS; ++sp) {
        const float * p = part + (size_t(h) * SPLITS + sp) * (D + 2);
        if (p[0] == -INFINITY) continue;
        const float c = expf(p[0] - M);
        L += p[1] * c;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] += p[2 + lane * 4 + j] * c;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = h * D + lane * 4 + j;
        a.out[i] = o[j] / L * sigmoidf(a.gate[i]);
    }
}

// ---------------------------------------------------------------- convolution + residual

// out = (add ? add : 0) + (x + conv(x)) / sqrt(5), shifting the history
__global__ void conv_residual_kernel(const float * __restrict__ x, float * __restrict__ hist, const float * __restrict__ w,
                                     const float * __restrict__ add, float * __restrict__ out, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float xv = x[c];
    const float o = conv1(xv, hist, n, c, w, 4);
    const float y = (xv + o) * rsqrtf(5.f);
    out[c] = add ? add[c] + y : y;
}

// ---------------------------------------------------------------- router

// probs = sigmoid(W x) (fp32 [n_expert][n]); one warp per expert, spread over blocks
__global__ void router_logits_kernel(const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ probs, int n_expert, int n) {
    const int lane = threadIdx.x % WARP;
    const int e = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (e >= n_expert) return;
    const float4 * row = reinterpret_cast<const float4 *>(w + size_t(e) * n);
    const float4 * xv = reinterpret_cast<const float4 *>(x);
    float d = 0.f;
    for (int i = lane; i < n / 4; i += WARP) {
        const float4 a = row[i], b = xv[i];
        d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    d = warp_sum(d);
    if (lane == 0) probs[e] = sigmoidf(d);
}

// top-k of probs + bias (one warp); weights = probs / sum * scale
__global__ void router_topk_kernel(const float * __restrict__ probs, const float * __restrict__ bias, int32_t * __restrict__ ids,
                                   float * __restrict__ wts, int n_expert, int k, float scale) {
    constexpr int PER = 8;  // n_expert <= 256
    const int lane = threadIdx.x;
    float sel[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + j * WARP;
        sel[j] = e < n_expert ? probs[e] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    float sum = 0.f;
    for (int t = 0; t < k; ++t) {
        float best = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + j * WARP;
            if (sel[j] > best) { best = sel[j]; bi = e; }
        }
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        if ((bi % WARP) == lane) sel[bi / WARP] = -INFINITY;
        const float p = probs[bi];
        sum += p;
        if (lane == 0) { ids[t] = bi; wts[t] = p; }
    }
    __syncwarp();
    if (lane < k) wts[lane] *= scale / (sum + 1e-20f);
}

// ---------------------------------------------------------------- output

__global__ void softcap_kernel(const float * __restrict__ x, float * __restrict__ y, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(x[i] / cap);
}

__global__ void add_pos_kernel(int * pos, int d) { *pos += d; }

// ---------------------------------------------------------------- CPU handoff (mapped host memory)

// copy the router's choice and the expert input to the host, then raise the flag (value pos + 1)
__global__ void publish_kernel(const int32_t * __restrict__ ids, const float * __restrict__ wts, const float * __restrict__ x,
                               HostSlot h, const int * __restrict__ pos, int k, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) h.x[i] = x[i];
    if (threadIdx.x < k) { h.ids[threadIdx.x] = ids[threadIdx.x]; h.wts[threadIdx.x] = wts[threadIdx.x]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *h.ready = *pos + 1;
}

// the GPU's share of the experts missing from the VRAM cache: their pages are pinned and mapped (zc tables), and the
// m-th missing expert (in selection order) is the GPU's when zc is set and floor((m+1)*share) > floor(m*share).
// Writes out_gu / out_dn [ids[k]] = the zc pointer for the GPU's experts, null for the others (the CPU applies the
// same rule). The GGUF-layout expert kernels then read those experts over PCIe.
__global__ void zc_select_kernel(const void * const * __restrict__ table, const void * const * __restrict__ zc_gu,
                                 const void * const * __restrict__ zc_dn, const int32_t * __restrict__ ids, int k, float share,
                                 const void ** __restrict__ out_gu, const void ** __restrict__ out_dn) {
    if (threadIdx.x != 0) return;
    int m = 0;
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        const void * g = nullptr, * d = nullptr;
        if (table[e] == nullptr) {
            const bool mine = zc_gu[e] != nullptr && floorf((m + 1) * share) > floorf(m * share);
            if (mine) { g = zc_gu[e]; d = zc_dn[e]; }
            ++m;
        }
        out_gu[e] = g;
        out_dn[e] = d;
    }
}

__global__ void copy_wait_kernel(const void * const * __restrict__ table, const int32_t * __restrict__ ids, int k,
                                 const int * __restrict__ pos, const volatile int * flag, const CopySel * sel, uint8_t * base,
                                 size_t slot, size_t gu_area, const void ** __restrict__ out_gu, const void ** __restrict__ out_dn) {
    if (threadIdx.x != 0) return;
    int miss = 0;
    for (int j = 0; j < k; ++j) {
        miss |= table[ids[j]] == nullptr;
        out_gu[ids[j]] = nullptr;
        out_dn[ids[j]] = nullptr;
    }
    if (!miss) return;
    const int want = *pos + 1;
    while (*flag != want) {}
    __threadfence_system();
    const volatile CopySel * s = sel;
    const int n = s->n;
    for (int i = 0; i < n; ++i) {
        const int e = ids[s->k[i]];
        out_gu[e] = base + i * slot;
        out_dn[e] = base + i * slot + gu_area;
    }
}

__global__ void add_kernel(float * __restrict__ y, const float * __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// wait until the CPU has published its experts for this position, then y += y_cpu. When every selected expert is in
// the VRAM cache (table entry set) the CPU has nothing to add and the round trip is skipped.
__global__ void wait_add_kernel(HostSlot h, const void * const * __restrict__ table, const int32_t * __restrict__ ids, int k,
                                const int * __restrict__ pos, float * __restrict__ y, int n) {
    __shared__ int need;
    if (threadIdx.x == 0) {
        int miss = 0;
        for (int j = 0; j < k; ++j) miss |= table[ids[j]] == nullptr;
        need = miss;
        if (miss) {
            const int want = *pos + 1;
            while (*h.done != want) {}
            __threadfence_system();
        }
    }
    __syncthreads();
    if (!need) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] += h.y[i];
}


}  // namespace

// ---------------------------------------------------------------- launchers

void embed(const uint8_t * row, const float * w, float * out, int n, float eps, cudaStream_t st, int emb_type) {
    embed_kernel<<<1, 256, sizeof(float) * n, st>>>(row, w, out, n, eps, emb_type);
}

// fused depth mixing + per-query RMSNorm (when no loop controller sits between them): avoids 1 launch + global round-trip
struct MixNormW { const float * w[4]; };

__global__ void mix_norm_kernel(MixStates st, const float * __restrict__ query, int n_q, MixNormW nw_w, float eps, float * __restrict__ out, int n) {
    constexpr int NS = MAX_STATES, NQ = 4, NACC = NS * (NQ + 1);
    __shared__ float red[NACC][32];
    __shared__ float tot[NACC];
    __shared__ float coef[NQ][NS];
    __shared__ float nred[NQ][32];
    __shared__ float ninv[NQ];
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
                const float x = st.p[s][i];
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
        float t = 0.f;
        for (int k = 0; k < nw; ++k) t += red[j][k];
        tot[j] = t;
    }
    __syncthreads();
    if (threadIdx.x < n_q) {
        const int q = threadIdx.x;
        float sc[NS], mx = -INFINITY;
        for (int s = 0; s < st.n; ++s) {
            const float inv = rsqrtf(tot[s] / n + 1e-6f);
            sc[s] = tot[NS + q * NS + s] * inv * rsqrtf(float(n));
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < st.n; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < st.n; ++s) coef[q][s] = sc[s] / sum * rsqrtf(tot[s] / n + 1e-6f);
    }
    __syncthreads();
    // compute mixed[q][i] (at most 3 elements per thread for n=2304, blockDim=768) in registers and reduce ss[q]
    float mv[3][NQ];
    float ss[NQ] = {0.f, 0.f, 0.f, 0.f};
    int cnt = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x, ++cnt) {
        float xs[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) xs[s] = s < st.n ? st.p[s][i] : 0.f;
        for (int q = 0; q < n_q; ++q) {
            float v = 0.f;
            for (int s = 0; s < st.n; ++s) v += coef[q][s] * xs[s];
            mv[cnt][q] = v;
            ss[q] += v * v;
        }
    }
    for (int q = 0; q < n_q; ++q) {
        const float v = warp_sum(ss[q]);
        if (lane == 0) nred[q][w] = v;
    }
    __syncthreads();
    if (threadIdx.x < n_q) {
        const int q = threadIdx.x;
        float t = 0.f;
        for (int k = 0; k < nw; ++k) t += nred[q][k];
        ninv[q] = rsqrtf(t / n + eps);
    }
    __syncthreads();
    cnt = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x, ++cnt) {
        for (int q = 0; q < n_q; ++q) out[q * n + i] = mv[cnt][q] * ninv[q] * nw_w.w[q][i];
    }
}

void mix(const MixStates & s, const float * query, int n_q, float * out, int n, cudaStream_t st) {
    mix_kernel<<<1, 768, 0, st>>>(s, query, n_q, out, n);
}

void mix_norm(const MixStates & s, const float * query, int n_q, const float * const * norm_w, float eps, float * out, int n, cudaStream_t st) {
    MixNormW w{};
    for (int q = 0; q < n_q; ++q) w.w[q] = norm_w[q];
    mix_norm_kernel<<<1, 768, 0, st>>>(s, query, n_q, w, eps, out, n);
}

void control(const ControlArgs & a_in, int count, int n, int rank, float * scratch, cudaStream_t st) {
    ControlArgs a = a_in;
    a.count = count;
    control_g_kernel<<<(count * rank + 7) / 8, 256, 0, st>>>(a, scratch, n, rank);
    control_out_kernel<<<(count * n + 7) / 8, 256, 0, st>>>(a, scratch, n, rank);
}

void rmsnorm(const NormArgs & a, int count, int n, float eps, cudaStream_t st) {
    rmsnorm_kernel<<<count, 512, 0, st>>>(a, n, eps);
}

void matvec_f32(const float * w, const float * x, float * y, int rows, int n, cudaStream_t st) {
    matvec_small_kernel<float><<<(rows + 7) / 8, 256, 0, st>>>(w, x, y, rows, n);
}

void matvec_bf16(const __nv_bfloat16 * w, const float * x, float * y, int rows, int n, cudaStream_t st) {
    matvec_small_kernel<__nv_bfloat16><<<(rows + 7) / 8, 256, 0, st>>>(w, x, y, rows, n);
}

void attn_prep(const AttnPrepArgs & a, cudaStream_t st) {
    attn_prep_kernel<<<1, 1024, sizeof(float) * 2 * a.n_head_kv * a.d_head, st>>>(a);
}

void attention(const AttnArgs & a, cudaStream_t st) {
    // scratch for the splits' partial results ([n_head][SPLITS][2 + 128]), one per launch site (graph-safe)
    float * part = a.scratch;
    if (a.kv_q8) attention_split_kernel<true><<<dim3(a.n_head, SPLITS), 256, 0, st>>>(a, part);
    else attention_split_kernel<false><<<dim3(a.n_head, SPLITS), 256, 0, st>>>(a, part);
    attention_combine_kernel<<<a.n_head, 32, 0, st>>>(a, part);
}

void conv_residual(const float * x, float * hist, const float * w, const float * add, float * out, int n, cudaStream_t st) {
    conv_residual_kernel<<<(n + 255) / 256, 256, 0, st>>>(x, hist, w, add, out, n);
}

void router(const float * w, const float * bias, const float * x, int32_t * ids, float * wts, int n_expert, int n, int k, float scale,
            float * scratch, cudaStream_t st) {
    router_logits_kernel<<<(n_expert + 7) / 8, 256, 0, st>>>(w, x, scratch, n_expert, n);
    router_topk_kernel<<<1, 32, 0, st>>>(scratch, bias, ids, wts, n_expert, k, scale);
}

void softcap(const float * x, float * y, int n, float cap, cudaStream_t st) {
    softcap_kernel<<<(n + 255) / 256, 256, 0, st>>>(x, y, n, cap);
}

namespace {

__global__ void softcap_pen_argmax_p1_kernel(const float * __restrict__ raw, float * __restrict__ logits,
                                             const int32_t * __restrict__ pen_counts, const float * __restrict__ pen_params,
                                             int n, float cap, float * __restrict__ bval, int32_t * __restrict__ bidx) {
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const float pres = pen_params ? pen_params[0] : 0.f;
    const float freq = pen_params ? pen_params[1] : 0.f;
    const bool has_pen = (pres != 0.f || freq != 0.f) && pen_counts != nullptr;
    float bv = -INFINITY;
    int32_t bi = n;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = raw[i];
        if (cap > 0.f) v = cap * tanhf(v / cap);
        if (has_pen) {
            const int32_t c = pen_counts[i];
            if (c > 0) v -= pres + freq * float(c);
        }
        logits[i] = v;
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        bv = lane < 8 ? sv[lane] : -INFINITY;
        bi = lane < 8 ? si[lane] : n;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { bval[blockIdx.x] = bv; bidx[blockIdx.x] = bi; }
    }
}

__global__ void argmax_p2_kernel(const float * __restrict__ bval, const int32_t * __restrict__ bidx,
                                 int nb, int32_t * __restrict__ out_idx, float * __restrict__ out_val) {
    const int lane = threadIdx.x;
    float bv = -INFINITY;
    int32_t bi = 0x7fffffff;
    for (int i = lane; i < nb; i += 32) {
        const float v = bval[i];
        const int32_t idx = bidx[i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
        *out_idx = bi;
        *out_val = bv;
    }
}

__global__ void filter_logits_kernel(const float * __restrict__ logits, int n, const float * __restrict__ max_val,
                                     float margin, int32_t * __restrict__ out_ids, float * __restrict__ out_vals,
                                     int * __restrict__ cnt_out, int cap) {
    const float thr = *max_val - margin;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = logits[i];
        if (v >= thr) {
            const int pos = atomicAdd(cnt_out, 1);
            if (pos < cap) { out_ids[pos] = i; out_vals[pos] = v; }
        }
    }
}

__global__ void pen_inc_kernel(int32_t * __restrict__ pen_counts, int tok) {
    pen_counts[tok] += 1;
}

}  // namespace

void softcap_pen_argmax(const float * raw, float * logits, const int32_t * pen_counts, const float * pen_params,
                        int n, float cap, float * bval, int32_t * bidx, int32_t * out_idx, float * out_val, cudaStream_t st) {
    softcap_pen_argmax_p1_kernel<<<64, 256, 0, st>>>(raw, logits, pen_counts, pen_params, n, cap, bval, bidx);
    argmax_p2_kernel<<<1, 32, 0, st>>>(bval, bidx, 64, out_idx, out_val);
}

void filter_logits(const float * logits, int n, const float * max_val, float margin,
                   int32_t * out_ids, float * out_vals, int * cnt_out, int cap, cudaStream_t st) {
    cudaMemsetAsync(cnt_out, 0, sizeof(int), st);
    filter_logits_kernel<<<64, 256, 0, st>>>(logits, n, max_val, margin, out_ids, out_vals, cnt_out, cap);
}

void pen_inc(int32_t * pen_counts, int tok, cudaStream_t st) {
    pen_inc_kernel<<<1, 1, 0, st>>>(pen_counts, tok);
}

void add_pos(int * pos, int d, cudaStream_t st) { add_pos_kernel<<<1, 1, 0, st>>>(pos, d); }

void publish(const int32_t * ids, const float * wts, const float * x, const HostSlot & h, const int * pos, int k, int n, cudaStream_t st) {
    publish_kernel<<<1, 1024, 0, st>>>(ids, wts, x, h, pos, k, n);
}

void zc_select(const void * const * table, const void * const * zc_gu, const void * const * zc_dn, const int32_t * ids, int k,
               float share, const void ** out_gu, const void ** out_dn, cudaStream_t st) {
    zc_select_kernel<<<1, 32, 0, st>>>(table, zc_gu, zc_dn, ids, k, share, out_gu, out_dn);
}

void add(float * y, const float * x, int n, cudaStream_t st) { add_kernel<<<(n + 255) / 256, 256, 0, st>>>(y, x, n); }

void copy_wait(const void * const * table, const int32_t * ids, int k, const int * pos, const volatile int * flag,
               const CopySel * sel, uint8_t * base, size_t slot, size_t gu_area, const void ** out_gu, const void ** out_dn,
               cudaStream_t st) {
    copy_wait_kernel<<<1, 32, 0, st>>>(table, ids, k, pos, flag, sel, base, slot, gu_area, out_gu, out_dn);
}

void wait_add(const HostSlot & h, const void * const * table, const int32_t * ids, int k, const int * pos, float * y, int n, cudaStream_t st) {
    wait_add_kernel<<<1, 1024, 0, st>>>(h, table, ids, k, pos, y, n);
}

}  // namespace ik

// Inkling 2 Mini DSpark (mtp.safetensors) CUDA implementation.
// Matches vllm/models/inkling2/nvidia/dspark.py + ops/dspark.py:
//   - 4 target hidden taps [5, 23, 41, 48] mixed via softmax(fuse_weight[row]) + dflare_norm
//   - 3 dense BF16 transformer layers (layers 0 & 2 local SWA=1024, layer 1 global) with bidirectional
//     block attention over [context KV (0..pos-1), block KV (pos..pos+7)] and neox RoPE (theta=1e6)
//   - Recurrent rnn_hidden Markov head (rank=256) with folded LM-head projection [n_vocab, 256] in fp16
//     for fast on-GPU argmax across all 7 draft steps with a single host sync.

#include "ik_dspark.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "compat_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ik {
namespace {

constexpr int WARP = 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_THETA = 1000000.0f;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("dspark ") + what + ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------- minimal safetensors reader

struct STEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t off0 = 0, off1 = 0;
};

struct SafetensorsFile {
    uint8_t * base = nullptr;
    size_t size = 0;
    size_t data_base = 0;
    std::unordered_map<std::string, STEntry> tensors;
#ifdef _WIN32
    mk_win_mmap_t wmap_{};
#endif

    explicit SafetensorsFile(const std::string & path) {
#ifdef _WIN32
        if (mk_win_mmap(path.c_str(), &wmap_) != 0) throw std::runtime_error("cannot open " + path);
        base = wmap_.base;
        size = wmap_.size;
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 16) { close(fd); throw std::runtime_error("invalid safetensors " + path); }
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        base = (uint8_t *) p;
#endif
        uint64_t hlen = 0;
        memcpy(&hlen, base, 8);
        if (hlen == 0 || 8 + hlen > size) throw std::runtime_error("corrupt safetensors header: " + path);
        data_base = 8 + (size_t) hlen;
        parse_header(reinterpret_cast<const char *>(base + 8), (size_t) hlen);
    }

    ~SafetensorsFile() {
        if (!base) return;
#ifdef _WIN32
        mk_win_munmap(&wmap_);
#else
        munmap(base, size);
#endif
    }

    const uint8_t * data(const std::string & name, const STEntry ** out_e = nullptr) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("mtp.safetensors missing tensor: " + name);
        if (out_e) *out_e = &it->second;
        return base + data_base + it->second.off0;
    }

private:
    static void skip_ws(const char * s, size_t n, size_t & i) {
        while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    static std::string parse_str(const char * s, size_t n, size_t & i) {
        skip_ws(s, n, i);
        if (i >= n || s[i] != '"') throw std::runtime_error("safetensors JSON: expected string");
        ++i;
        std::string r;
        while (i < n && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < n) { ++i; }
            r.push_back(s[i++]);
        }
        if (i < n) ++i;
        return r;
    }
    static int64_t parse_int(const char * s, size_t n, size_t & i) {
        skip_ws(s, n, i);
        bool neg = false;
        if (i < n && s[i] == '-') { neg = true; ++i; }
        int64_t v = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        return neg ? -v : v;
    }
    static void skip_val(const char * s, size_t n, size_t & i) {
        skip_ws(s, n, i);
        if (i >= n) return;
        if (s[i] == '"') { parse_str(s, n, i); return; }
        if (s[i] == '{' || s[i] == '[') {
            const char close = s[i] == '{' ? '}' : ']';
            int depth = 1; ++i;
            while (i < n && depth > 0) {
                if (s[i] == '"') parse_str(s, n, i);
                else if (s[i] == '{' || s[i] == '[') { ++depth; ++i; }
                else if (s[i] == '}' || s[i] == ']') { --depth; ++i; }
                else ++i;
            }
            (void) close;
            return;
        }
        while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    }
    void parse_header(const char * s, size_t n) {
        size_t i = 0;
        skip_ws(s, n, i);
        if (i >= n || s[i] != '{') throw std::runtime_error("safetensors JSON: expected {");
        ++i;
        while (i < n) {
            skip_ws(s, n, i);
            if (i < n && s[i] == '}') break;
            std::string key = parse_str(s, n, i);
            skip_ws(s, n, i);
            if (i < n && s[i] == ':') ++i;
            if (key == "__metadata__") {
                skip_val(s, n, i);
            } else {
                skip_ws(s, n, i);
                if (i >= n || s[i] != '{') throw std::runtime_error("safetensors JSON: expected tensor object");
                ++i;
                STEntry e;
                while (i < n) {
                    skip_ws(s, n, i);
                    if (i < n && s[i] == '}') { ++i; break; }
                    std::string f = parse_str(s, n, i);
                    skip_ws(s, n, i);
                    if (i < n && s[i] == ':') ++i;
                    if (f == "dtype") {
                        e.dtype = parse_str(s, n, i);
                    } else if (f == "shape") {
                        skip_ws(s, n, i); ++i;  // '['
                        while (i < n) {
                            skip_ws(s, n, i);
                            if (s[i] == ']') { ++i; break; }
                            e.shape.push_back(parse_int(s, n, i));
                            skip_ws(s, n, i);
                            if (i < n && s[i] == ',') ++i;
                        }
                    } else if (f == "data_offsets") {
                        skip_ws(s, n, i); ++i;  // '['
                        e.off0 = (size_t) parse_int(s, n, i);
                        skip_ws(s, n, i);
                        if (i < n && s[i] == ',') ++i;
                        e.off1 = (size_t) parse_int(s, n, i);
                        skip_ws(s, n, i);
                        if (i < n && s[i] == ']') ++i;
                    } else {
                        skip_val(s, n, i);
                    }
                    skip_ws(s, n, i);
                    if (i < n && s[i] == ',') ++i;
                }
                tensors.emplace(std::move(key), std::move(e));
            }
            skip_ws(s, n, i);
            if (i < n && s[i] == ',') ++i;
        }
    }
};

// ---------------------------------------------------------------- CUDA kernels

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    return warp_sum(lane < nw ? red[lane] : 0.f);
}

// Mix 4 target taps [T][4][E] with alpha[4] and RMSNorm with weight[E] -> out[T][E]
__global__ void mix_rmsnorm_kernel(const float * __restrict__ taps, const float * __restrict__ alpha,
                                   const float * __restrict__ w, float * __restrict__ out, int E, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];
    const float * tp = taps + size_t(t) * 4 * E;
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = a0 * tp[i] + a1 * tp[E + i] + a2 * tp[2 * E + i] + a3 * tp[3 * E + i];
        out[size_t(t) * E + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / float(E) + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        out[size_t(t) * E + i] = out[size_t(t) * E + i] * inv * w[i];
    }
}

// RMSNorm: out[t][i] = x[t][i] * inv_rms * w[i]
__global__ void rmsnorm_kernel(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ out, int E, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const float * xp = x + size_t(t) * E;
    float * op = out + size_t(t) * E;
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) ss += xp[i] * xp[i];
    const float inv = rsqrtf(block_sum(ss, red) / float(E) + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) op[i] = xp[i] * inv * w[i];
}

// residual[t][i] += add[t][i]; out[t][i] = rmsnorm(residual[t], w)
__global__ void add_rmsnorm_kernel(float * __restrict__ residual, const float * __restrict__ add,
                                   const float * __restrict__ w, float * __restrict__ out, int E, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    float * rp = residual + size_t(t) * E;
    const float * ap = add + size_t(t) * E;
    float * op = out + size_t(t) * E;
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = rp[i] + ap[i];
        rp[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / float(E) + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) op[i] = rp[i] * inv * w[i];
}

// Build block[0..7]: row 0 = anchor_emb, rows 1..7 = mask_embed
__global__ void init_block_kernel(const float * __restrict__ anchor_emb, const float * __restrict__ mask_embed,
                                  float * __restrict__ out, int E) {
    const int t = blockIdx.x;
    const float * src = t == 0 ? anchor_emb : mask_embed;
    for (int i = threadIdx.x; i < E; i += blockDim.x) out[size_t(t) * E + i] = src[i];
}

// Quantize BF16 [rows, K] (K multiple of 32) -> repacked Q8_0: int8_t q[rows][K] + __half d[rows][K/32]
__global__ void quantize_bf16_to_q8_kernel(const __nv_bfloat16 * __restrict__ src, int8_t * __restrict__ dst_q,
                                           __half * __restrict__ dst_d, size_t n_blocks) {
    const size_t blk = (size_t(blockIdx.x) * (blockDim.x / WARP)) + (threadIdx.x / WARP);
    if (blk >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float v = __bfloat162float(src[blk * 32 + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax / 127.f;
    const float inv = amax > 0.f ? (127.f / amax) : 0.f;
    int q = __float2int_rn(v * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst_q[blk * 32 + lane] = int8_t(q);
    if (lane == 0) dst_d[blk] = __float2half_rn(scale);
}

// Quantize FP16 [rows, K] (K multiple of 32) -> repacked Q8_0: int8_t q[rows][K] + __half d[rows][K/32]
__global__ void quantize_f16_to_q8_kernel(const __half * __restrict__ src, int8_t * __restrict__ dst_q,
                                          __half * __restrict__ dst_d, size_t n_blocks) {
    const size_t blk = (size_t(blockIdx.x) * (blockDim.x / WARP)) + (threadIdx.x / WARP);
    if (blk >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float v = __half2float(src[blk * 32 + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax / 127.f;
    const float inv = amax > 0.f ? (127.f / amax) : 0.f;
    int q = __float2int_rn(v * inv);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst_q[blk * 32 + lane] = int8_t(q);
    if (lane == 0) dst_d[blk] = __float2half_rn(scale);
}

// Vectorized Q8_0 GEMM for small T: Y[t][r] = W_q8[r] . X[t]
// 1 warp per output row, 8 int8 (uint2) per lane per step -> 256 elements/step, half the bandwidth of BF16!
template <int TMAX>
__global__ void __launch_bounds__(256) q8_gemm_kernel(const int8_t * __restrict__ Wq, const __half * __restrict__ Wd,
                                                      const float * __restrict__ X, float * __restrict__ Y, int rows, int K, int T) {
    const int lane = threadIdx.x % WARP;
    const int stride = gridDim.x * (blockDim.x / WARP);
    const int steps = K / 256;
    const int n_sb = K / 32;
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const uint2 * qr = reinterpret_cast<const uint2 *>(Wq + size_t(r) * K);
        const __half * dr = Wd + size_t(r) * n_sb;
        for (int t0 = 0; t0 < T; t0 += TMAX) {
            float acc[TMAX];
#pragma unroll
            for (int j = 0; j < TMAX; ++j) acc[j] = 0.f;
            for (int s = 0; s < steps; ++s) {
                const int w = s * WARP + lane;
                const uint2 qw = __ldg(qr + w);
                const float sc = __half2float(__ldg(dr + (w >> 2)));
                float wv[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wv[i] = float(int8_t(qw.x >> (8 * i)));
                    wv[4 + i] = float(int8_t(qw.y >> (8 * i)));
                }
#pragma unroll
                for (int j = 0; j < TMAX; ++j) {
                    if (t0 + j < T) {
                        const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(t0 + j) * K + w * 8);
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

// Fused Q8_0 gate_proj + up_proj + SiLU(g)*u for T <= 8: reads X once per pair of rows, half the weight bandwidth of BF16!
__global__ void __launch_bounds__(256) q8_glu_kernel(const int8_t * __restrict__ Wgq, const __half * __restrict__ Wgd,
                                                     const int8_t * __restrict__ Wuq, const __half * __restrict__ Wud,
                                                     const float * __restrict__ X, float * __restrict__ Y, int rows, int K, int T) {
    constexpr int TMAX = 8;
    const int lane = threadIdx.x % WARP;
    const int stride = gridDim.x * (blockDim.x / WARP);
    const int steps = K / 256;
    const int n_sb = K / 32;
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const uint2 * qg = reinterpret_cast<const uint2 *>(Wgq + size_t(r) * K);
        const __half * dg = Wgd + size_t(r) * n_sb;
        const uint2 * qu = reinterpret_cast<const uint2 *>(Wuq + size_t(r) * K);
        const __half * du = Wud + size_t(r) * n_sb;
        float ag[TMAX], au[TMAX];
#pragma unroll
        for (int j = 0; j < TMAX; ++j) ag[j] = au[j] = 0.f;
        for (int s = 0; s < steps; ++s) {
            const int w = s * WARP + lane;
            const uint2 vg = __ldg(qg + w), vu = __ldg(qu + w);
            const float sg = __half2float(__ldg(dg + (w >> 2))), su = __half2float(__ldg(du + (w >> 2)));
            float wg[8], wu[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wg[i] = float(int8_t(vg.x >> (8 * i))); wg[4 + i] = float(int8_t(vg.y >> (8 * i)));
                wu[i] = float(int8_t(vu.x >> (8 * i))); wu[4 + i] = float(int8_t(vu.y >> (8 * i)));
            }
#pragma unroll
            for (int j = 0; j < TMAX; ++j) {
                if (j < T) {
                    const float4 * xv = reinterpret_cast<const float4 *>(X + size_t(j) * K + w * 8);
                    const float4 a = __ldg(xv), b = __ldg(xv + 1);
                    ag[j] += (wg[0] * a.x + wg[1] * a.y + wg[2] * a.z + wg[3] * a.w +
                              wg[4] * b.x + wg[5] * b.y + wg[6] * b.z + wg[7] * b.w) * sg;
                    au[j] += (wu[0] * a.x + wu[1] * a.y + wu[2] * a.z + wu[3] * a.w +
                              wu[4] * b.x + wu[5] * b.y + wu[6] * b.z + wu[7] * b.w) * su;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < TMAX; ++j) {
            if (j < T) {
                const float g = warp_sum(ag[j]), u = warp_sum(au[j]);
                if (lane == 0) Y[size_t(j) * rows + r] = (g / (1.f + expf(-g))) * u;
            }
        }
    }
}

// Per-head RMSNorm + neox RoPE (theta=1e6) in-place on [T, n_heads, 128]; reads pos0 from d_pos when non-null.
__global__ void qk_norm_rope_kernel(float * __restrict__ X, const float * __restrict__ w, int n_heads, int pos0,
                                    const int * __restrict__ d_pos, float eps) {
    constexpr int D = 128, HALF = 64;
    const int t = blockIdx.y, h = blockIdx.x;
    const int i = threadIdx.x;  // 0..63
    const int p0 = d_pos ? *d_pos : pos0;
    float * row = X + (size_t(t) * n_heads + h) * D;
    __shared__ float red[4];
    const float a = row[i], b = row[i + HALF];
    float ss = a * a + b * b;
    ss = warp_sum(ss);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    const float tot = red[0] + red[1];
    const float inv = rsqrtf(tot / float(D) + eps);
    const float x1 = a * inv * w[i];
    const float x2 = b * inv * w[i + HALF];
    const float freq = powf(ROPE_THETA, -float(2 * i) / float(D));
    const float ang = float(p0 + t) * freq;
    float s, c;
    sincosf(ang, &s, &c);
    row[i] = x1 * c - x2 * s;
    row[i + HALF] = x2 * c + x1 * s;
}

// Context KV store: K_in/V_in [T, 8, 128] -> k_norm_ctx + neox RoPE -> write half to k_cache/v_cache at (pos0 + t) % ring (if ring > 0)
__global__ void store_ctx_kv_kernel(const float * __restrict__ K_in, const float * __restrict__ V_in,
                                    const float * __restrict__ k_norm_w, __half * __restrict__ K_cache, __half * __restrict__ V_cache,
                                    int pos0, const int * __restrict__ d_pos, int ring, float eps) {
    constexpr int HK = 8, D = 128, HALF = 64;
    const int t = blockIdx.y, h = blockIdx.x;
    const int i = threadIdx.x;  // 0..63
    const int pos = (d_pos ? *d_pos : pos0) + t;
    const int slot = ring > 0 ? (pos % ring) : pos;
    const float * kr = K_in + (size_t(t) * HK + h) * D;
    const float * vr = V_in + (size_t(t) * HK + h) * D;
    __shared__ float red[4];
    const float a = kr[i], b = kr[i + HALF];
    float ss = warp_sum(a * a + b * b);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1]) / float(D) + eps);
    const float x1 = a * inv * k_norm_w[i];
    const float x2 = b * inv * k_norm_w[i + HALF];
    const float freq = powf(ROPE_THETA, -float(2 * i) / float(D));
    float s, c;
    sincosf(float(pos) * freq, &s, &c);
    __half * kd = K_cache + (size_t(slot) * HK + h) * D;
    __half * vd = V_cache + (size_t(slot) * HK + h) * D;
    kd[i] = __float2half_rn(x1 * c - x2 * s);
    kd[i + HALF] = __float2half_rn(x2 * c + x1 * s);
    vd[i] = __float2half_rn(vr[i]);
    vd[i + HALF] = __float2half_rn(vr[i + HALF]);
}

// Bidirectional block attention for DSpark:
// Query Q[b, h, 128] (b = 0..7, h = 0..23) attends to:
//   1. Context KV cache positions p in [lo, pos - 1]  (lo = max(0, pos - SWA) if local else 0)
//   2. All 8 block KV entries K_blk[0..7], V_blk[0..7]
// Grid: dim3(24, 8), 256 threads (8 warps).
__global__ void block_attn_kernel(const float * __restrict__ Q, const __half * __restrict__ K_ctx, const __half * __restrict__ V_ctx,
                                  const float * __restrict__ K_blk, const float * __restrict__ V_blk, float * __restrict__ O,
                                  int pos_arg, const int * __restrict__ d_pos, int ring, int window) {
    constexpr int H = 24, HK = 8, D = 128, B = 8;
    __shared__ float s_m[8], s_l[8], s_acc[8][D];
    const int h = blockIdx.x, b = blockIdx.y, hk = h / (H / HK);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, nw = blockDim.x / WARP;
    const int pos = d_pos ? *d_pos : pos_arg;
    const float scale = rsqrtf(float(D));
    const float * qp = Q + (size_t(b) * H + h) * D;
    float q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = qp[lane * 4 + j] * scale;

    const int lo = window > 0 ? max(0, pos - window) : 0;
    const int n_ctx = pos - lo;
    const int total = n_ctx + B;

    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int idx = w; idx < total; idx += nw) {
        float4 kk, vv;
        if (idx < n_ctx) {
            const int p = lo + idx;
            const int slot = ring > 0 ? (p % ring) : p;
            const __half2 * kh = reinterpret_cast<const __half2 *>(K_ctx + (size_t(slot) * HK + hk) * D + lane * 4);
            const __half2 * vh = reinterpret_cast<const __half2 *>(V_ctx + (size_t(slot) * HK + hk) * D + lane * 4);
            const float2 k0 = __half22float2(__ldg(kh)), k1 = __half22float2(__ldg(kh + 1));
            const float2 v0 = __half22float2(__ldg(vh)), v1 = __half22float2(__ldg(vh + 1));
            kk = make_float4(k0.x, k0.y, k1.x, k1.y);
            vv = make_float4(v0.x, v0.y, v1.x, v1.y);
        } else {
            const int bb = idx - n_ctx;
            kk = *reinterpret_cast<const float4 *>(K_blk + (size_t(bb) * HK + hk) * D + lane * 4);
            vv = *reinterpret_cast<const float4 *>(V_blk + (size_t(bb) * HK + hk) * D + lane * 4);
        }
        const float s = warp_sum(q[0] * kk.x + q[1] * kk.y + q[2] * kk.z + q[3] * kk.w);
        const float mn = fmaxf(m, s), corr = expf(m - mn), pr = expf(s - mn);
        l = l * corr + pr;
        acc[0] = acc[0] * corr + pr * vv.x;
        acc[1] = acc[1] * corr + pr * vv.y;
        acc[2] = acc[2] * corr + pr * vv.z;
        acc[3] = acc[3] * corr + pr * vv.w;
        m = mn;
    }
    if (lane == 0) { s_m[w] = m; s_l[w] = l; }
#pragma unroll
    for (int j = 0; j < 4; ++j) s_acc[w][lane * 4 + j] = acc[j];
    __syncthreads();
    if (w == 0) {
        float M = -INFINITY;
        for (int i = 0; i < nw; ++i) M = fmaxf(M, s_m[i]);
        float L = 0.f;
        float o[4] = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < nw; ++i) {
            if (s_m[i] == -INFINITY) continue;
            const float c = expf(s_m[i] - M);
            L += s_l[i] * c;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] += s_acc[i][lane * 4 + j] * c;
        }
        float * op = O + (size_t(b) * H + h) * D;
        const float invL = L > 0.f ? 1.f / L : 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) op[lane * 4 + j] = o[j] * invL;
    }
}

// Transpose [C, n_vocab_pad] fp32 columns into [n_vocab, 256] fp16 fold_w at column offset col0
__global__ void scatter_fold_cols_kernel(const float * __restrict__ Y, __half * __restrict__ fold_w,
                                         int n_vocab, int n_vocab_pad, int col0, int C) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (v >= n_vocab || c >= C) return;
    fold_w[size_t(v) * DSpark::RANK + (col0 + c)] = __float2half_rn(Y[size_t(c) * n_vocab_pad + v]);
}

// Precompute Markov context[0..6][768] = rnn_bias + rnn_w_ctx @ concat(h_draft[i], f_rnn)
// Grid: (768/8 = 96, 7), 256 threads (1 warp per output row r in 0..767, blockIdx.y = step i in 0..6)
__global__ void markov_ctx_kernel(const __nv_bfloat16 * __restrict__ rnn_w, const float * __restrict__ rnn_b,
                                  const float * __restrict__ h_draft, const float * __restrict__ f_rnn,
                                  float * __restrict__ ctx_out) {
    constexpr int E = DSpark::E, RANK = DSpark::RANK, ROW_K = 2 * RANK + 2 * E;
    const int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int step = blockIdx.y;
    const int lane = threadIdx.x % WARP;
    if (r >= 3 * RANK) return;
    const __nv_bfloat16 * row = rnn_w + size_t(r) * ROW_K + 2 * RANK;
    const float * h = h_draft + size_t(step) * E;
    float acc = 0.f;
    for (int k = lane; k < E; k += WARP) {
        acc += __bfloat162float(row[k]) * h[k] + __bfloat162float(row[E + k]) * f_rnn[k];
    }
    acc = warp_sum(acc);
    if (lane == 0) ctx_out[size_t(step) * 3 * RANK + r] = rnn_b[r] + acc;
}

// Single Markov recurrence step on GPU (1 block, 256 threads, vectorized uint4 = 8 BF16 per load):
__global__ void markov_recur_kernel(const __nv_bfloat16 * __restrict__ rnn_w, const __nv_bfloat16 * __restrict__ mw1,
                                    const float * __restrict__ ctx_step, float * __restrict__ state, float * __restrict__ latent,
                                    const int32_t * __restrict__ prev_tok_ptr, int step) {
    constexpr int RANK = DSpark::RANK, ROW_K = 2 * RANK + 2 * DSpark::E;
    __shared__ __align__(16) float s_in[2 * RANK];
    const int c = threadIdx.x;  // 0..255
    const int prev = *prev_tok_ptr;
    const float st = (step == 0) ? 0.f : state[c];
    s_in[c] = st;
    s_in[RANK + c] = __bfloat162float(__ldg(mw1 + size_t(prev) * RANK + c));
    __syncthreads();

    float raw[3];
    for (int part = 0; part < 3; ++part) {
        const int r = part * RANK + c;
        const uint4 * wv = reinterpret_cast<const uint4 *>(rnn_w + size_t(r) * ROW_K);
        float acc = ctx_step[r];
#pragma unroll 4
        for (int k = 0; k < (2 * RANK) / 8; ++k) {
            const uint4 qw = __ldg(wv + k);
            const __nv_bfloat162 * b2 = reinterpret_cast<const __nv_bfloat162 *>(&qw);
            const float2 f0 = __bfloat1622float2(b2[0]), f1 = __bfloat1622float2(b2[1]);
            const float2 f2 = __bfloat1622float2(b2[2]), f3 = __bfloat1622float2(b2[3]);
            const float4 x0 = *reinterpret_cast<const float4 *>(s_in + k * 8);
            const float4 x1 = *reinterpret_cast<const float4 *>(s_in + k * 8 + 4);
            acc += f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w +
                   f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
        }
        raw[part] = acc;
    }
    const float g = 1.f / (1.f + expf(-raw[0]));
    const float cand = tanhf(raw[1]);
    state[c] = g * st + (1.f - g) * cand;
    latent[c] = tanhf(raw[2]);
}

// Stage 1 of fused Q8_0 fold_w @ latent + base_logits + fold_b + softcap -> block argmax
// 16 threads per row (LPR=16): 1 single uint4 = 16 int8 weights per lane (256 elems/row, half the bandwidth of FP16!).
__global__ void __launch_bounds__(256) markov_fold_argmax1_kernel(
    const float * __restrict__ base_logits, const int8_t * __restrict__ fold_q, const __half * __restrict__ fold_d,
    const float * __restrict__ fold_b, const float * __restrict__ latent, int n_vocab, float softcap,
    float * __restrict__ blk_val, int32_t * __restrict__ blk_idx) {
    constexpr int LPR = 16;  // 16 lanes * 16 elems = 256 = RANK
    __shared__ float s_lat[DSpark::RANK];
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    s_lat[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x % LPR;
    const int rpb = blockDim.x / LPR;  // 16 rows per block per iter
    const int row0 = blockIdx.x * rpb + threadIdx.x / LPR;
    const int stride = gridDim.x * rpb;

    float best_v = -INFINITY;
    int32_t best_i = 0;
    const float4 l0 = *reinterpret_cast<const float4 *>(s_lat + lane * 16);
    const float4 l1 = *reinterpret_cast<const float4 *>(s_lat + lane * 16 + 4);
    const float4 l2 = *reinterpret_cast<const float4 *>(s_lat + lane * 16 + 8);
    const float4 l3 = *reinterpret_cast<const float4 *>(s_lat + lane * 16 + 12);

    for (int v = row0; v < n_vocab; v += stride) {
        const uint4 q = __ldg(reinterpret_cast<const uint4 *>(fold_q + size_t(v) * DSpark::RANK + lane * 16));
        const float sc = __half2float(__ldg(fold_d + size_t(v) * (DSpark::RANK / 32) + (lane >> 1)));
        float d0 = float(int8_t(q.x)) * l0.x + float(int8_t(q.x >> 8)) * l0.y + float(int8_t(q.x >> 16)) * l0.z + float(int8_t(q.x >> 24)) * l0.w;
        float d1 = float(int8_t(q.y)) * l1.x + float(int8_t(q.y >> 8)) * l1.y + float(int8_t(q.y >> 16)) * l1.z + float(int8_t(q.y >> 24)) * l1.w;
        float d2 = float(int8_t(q.z)) * l2.x + float(int8_t(q.z >> 8)) * l2.y + float(int8_t(q.z >> 16)) * l2.z + float(int8_t(q.z >> 24)) * l2.w;
        float d3 = float(int8_t(q.w)) * l3.x + float(int8_t(q.w >> 8)) * l3.y + float(int8_t(q.w >> 16)) * l3.z + float(int8_t(q.w >> 24)) * l3.w;
        float dot = (d0 + d1 + d2 + d3) * sc;
#pragma unroll
        for (int o = LPR / 2; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) {
            const float raw = __ldg(base_logits + v) + __ldg(fold_b + v) + dot;
            const float val = softcap * tanhf(raw / softcap);
            if (val > best_v || (val == best_v && v < best_i)) { best_v = val; best_i = v; }
        }
    }
    s_val[threadIdx.x] = (lane == 0) ? best_v : -INFINITY;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float ov = s_val[threadIdx.x + o];
            const int32_t oi = s_idx[threadIdx.x + o];
            if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { blk_val[blockIdx.x] = s_val[0]; blk_idx[blockIdx.x] = s_idx[0]; }
}

// Stage 2: reduce 256 block winners -> out_tok[0]
__global__ void markov_fold_argmax2_kernel(const float * __restrict__ blk_val, const int32_t * __restrict__ blk_idx,
                                           int n_blocks, int32_t * __restrict__ out_tok) {
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    const int i = threadIdx.x;
    s_val[i] = i < n_blocks ? blk_val[i] : -INFINITY;
    s_idx[i] = i < n_blocks ? blk_idx[i] : 0;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (i < o) {
            const float ov = s_val[i + o];
            const int32_t oi = s_idx[i + o];
            if (ov > s_val[i] || (ov == s_val[i] && oi < s_idx[i])) { s_val[i] = ov; s_idx[i] = oi; }
        }
        __syncthreads();
    }
    if (i == 0) *out_tok = s_idx[0];
}

// Copy single token tap [4][E]
__global__ void copy_f32_kernel(const float * __restrict__ src, float * __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Stream-ordered scalar writers (pass values by kernel arg -> zero host-DMA race when chaining verify_commit -> draft!)
__global__ void set_i32_kernel(int32_t * dst, int32_t v) { *dst = v; }
__global__ void set_i32_2_kernel(int32_t * d0, int32_t v0, int32_t * d1, int32_t v1) { *d0 = v0; *d1 = v1; }

struct Q8Mat {
    int8_t * q = nullptr;
    __half * d = nullptr;
};

void launch_q8_gemm(const Q8Mat & W, const float * X, float * Y, int rows, int K, int T, cudaStream_t st) {
    const int blocks = (rows + 7) / 8, grid = blocks < 1024 ? blocks : 1024;
    if (T == 1) q8_gemm_kernel<1><<<grid, 256, 0, st>>>(W.q, W.d, X, Y, rows, K, 1);
    else if (T <= 4) q8_gemm_kernel<4><<<grid, 256, 0, st>>>(W.q, W.d, X, Y, rows, K, T);
    else if (T <= 8) q8_gemm_kernel<8><<<grid, 256, 0, st>>>(W.q, W.d, X, Y, rows, K, T);
    else q8_gemm_kernel<16><<<grid, 256, 0, st>>>(W.q, W.d, X, Y, rows, K, T);
}

}  // namespace

struct DSpark::Impl {
    int ctx = 0, n_vocab = 0, n_vocab_pad = 0;
    float softcap = 30.f;
    std::vector<void *> owned;

    struct LayerW {
        float * dflare_norm = nullptr, * in_norm = nullptr, * post_norm = nullptr;
        float * q_norm = nullptr, * k_norm = nullptr, * k_norm_ctx = nullptr;
        Q8Mat q_proj, k_proj, v_proj, k_proj_ctx, v_proj_ctx, o_proj, gate_proj, up_proj, down_proj;
        __half * k_cache = nullptr, * v_cache = nullptr;
        int ring = 0, window = 0;
    };
    LayerW L[N_LAYERS];

    float * fuse_alpha = nullptr;          // [4][4]
    float * mask_embed = nullptr;          // [E]
    float * norm_w = nullptr;              // [E]
    float * mh_dflare_norm = nullptr;      // [E]
    float * mh_rnn_bias = nullptr;         // [768]
    __nv_bfloat16 * mh_rnn_w = nullptr;    // [768, 5120]
    __nv_bfloat16 * mh_w1 = nullptr;       // [vocab_pad, 256]
    int8_t * fold_q = nullptr;             // [n_vocab, 256]
    __half * fold_d = nullptr;             // [n_vocab, 8]
    float * fold_b = nullptr;              // [n_vocab]

    // Runtime scratch buffers
    float * last_taps = nullptr;           // [4][E]
    float * buf_x = nullptr;               // [BLOCK][E]
    float * buf_xn = nullptr;              // [BLOCK][E]
    float * buf_q = nullptr;               // [BLOCK][H*D]
    float * buf_k = nullptr;               // [BLOCK][KV]
    float * buf_v = nullptr;               // [BLOCK][KV]
    float * buf_o = nullptr;               // [BLOCK][H*D]
    float * buf_ff = nullptr;              // [BLOCK][FF]
    float * buf_ctx_tmp = nullptr;         // [256][E] for store_context chunks
    float * buf_ctx_k = nullptr;           // [256][KV]
    float * buf_ctx_v = nullptr;           // [256][KV]
    float * buf_f_rnn = nullptr;           // [E]
    float * buf_mh_ctx = nullptr;          // [N_SPEC][768]
    float * buf_state = nullptr;           // [RANK]
    float * buf_latent = nullptr;          // [RANK]
    float * buf_base_logits = nullptr;     // [N_SPEC][n_vocab_pad]
    float * buf_blk_val = nullptr;         // [256]
    int32_t * buf_blk_idx = nullptr;       // [256]
    int32_t * d_drafts = nullptr;          // [N_SPEC]
    int32_t * h_drafts = nullptr;          // pinned [N_SPEC]
    int32_t * h_scalars = nullptr;         // pinned [2]: [pos, anchor_tok]
    int32_t * d_pos_buf = nullptr;         // [1] for CUDA-graphed draft & ctx store
    int32_t * d_anchor_tok = nullptr;      // [1] for CUDA-graphed draft
    float * d_anchor_buf = nullptr;        // [E] fixed device copy of anchor_emb for graphed draft
    float * d_ctx_taps_buf = nullptr;      // [BLOCK * N_TAPS * E] fixed device copy for graphed ctx store (T <= 8)
    cudaGraph_t draft_g[N_SPEC + 1] = {};
    cudaGraphExec_t draft_e[N_SPEC + 1] = {};
    cudaGraph_t ctx_g[BLOCK + 1] = {};
    cudaGraphExec_t ctx_e[BLOCK + 1] = {};

    template <typename T> T * dalloc(size_t n) {
        void * p = nullptr;
        ck(cudaMalloc(&p, sizeof(T) * n), "malloc");
        owned.push_back(p);
        return (T *) p;
    }

    __nv_bfloat16 * load_bf16(const SafetensorsFile & sf, const std::string & name, size_t expected_elems, cudaStream_t st) {
        const STEntry * e = nullptr;
        const uint8_t * src = sf.data(name, &e);
        if (e->dtype != "BF16" || (e->off1 - e->off0) != expected_elems * 2)
            throw std::runtime_error("mtp.safetensors unexpected shape/dtype for " + name);
        auto * dst = dalloc<__nv_bfloat16>(expected_elems);
        ck(cudaMemcpyAsync(dst, src, expected_elems * 2, cudaMemcpyHostToDevice, st), "upload bf16");
        ck(cudaStreamSynchronize(st), "upload bf16");
        return dst;
    }

    Q8Mat load_q8(const SafetensorsFile & sf, const std::string & name, size_t rows, size_t K, __nv_bfloat16 * tmp_bf16,
                  cudaStream_t st) {
        const size_t elems = rows * K;
        const STEntry * e = nullptr;
        const uint8_t * src = sf.data(name, &e);
        if (e->dtype != "BF16" || (e->off1 - e->off0) != elems * 2)
            throw std::runtime_error("mtp.safetensors unexpected shape/dtype for " + name);
        ck(cudaMemcpyAsync(tmp_bf16, src, elems * 2, cudaMemcpyHostToDevice, st), "upload bf16 tmp");
        Q8Mat m;
        m.q = dalloc<int8_t>(elems);
        m.d = dalloc<__half>(elems / 32);
        const size_t n_blocks = elems / 32;
        quantize_bf16_to_q8_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, st>>>(tmp_bf16, m.q, m.d, n_blocks);
        ck(cudaStreamSynchronize(st), "quant q8");
        return m;
    }

    float * load_f32(const SafetensorsFile & sf, const std::string & name, size_t expected_elems, cudaStream_t st) {
        const STEntry * e = nullptr;
        const uint8_t * src = sf.data(name, &e);
        if (e->dtype != "BF16" || (e->off1 - e->off0) != expected_elems * 2)
            throw std::runtime_error("mtp.safetensors unexpected shape/dtype for " + name);
        std::vector<float> h(expected_elems);
        const uint16_t * s = reinterpret_cast<const uint16_t *>(src);
        for (size_t i = 0; i < expected_elems; ++i) {
            uint32_t bits = uint32_t(s[i]) << 16;
            memcpy(&h[i], &bits, 4);
        }
        float * dst = dalloc<float>(expected_elems);
        ck(cudaMemcpyAsync(dst, h.data(), expected_elems * 4, cudaMemcpyHostToDevice, st), "upload f32");
        ck(cudaStreamSynchronize(st), "upload f32");
        return dst;
    }

    ~Impl() {
        for (int i = 0; i <= N_SPEC; ++i) {
            if (draft_e[i]) cudaGraphExecDestroy(draft_e[i]);
            if (draft_g[i]) cudaGraphDestroy(draft_g[i]);
        }
        for (int i = 0; i <= BLOCK; ++i) {
            if (ctx_e[i]) cudaGraphExecDestroy(ctx_e[i]);
            if (ctx_g[i]) cudaGraphDestroy(ctx_g[i]);
        }
        for (void * p : owned) cudaFree(p);
        if (h_drafts) cudaFreeHost(h_drafts);
        if (h_scalars) cudaFreeHost(h_scalars);
    }
};

DSpark::DSpark(const std::string & path, int ctx, int n_vocab, int n_vocab_pad, float softcap,
               const LmHeadFn & lm_head, cudaStream_t st)
    : p_(new Impl()) {
    try {
        p_->ctx = ctx;
        p_->n_vocab = n_vocab;
        p_->n_vocab_pad = n_vocab_pad;
        p_->softcap = softcap;

        SafetensorsFile sf(path);

        // Softmax over fuse_weight[4][4]
        {
            const STEntry * e = nullptr;
            const uint16_t * src = reinterpret_cast<const uint16_t *>(sf.data("fuse_weight", &e));
            float a[16];
            for (int row = 0; row < 4; ++row) {
                float mx = -1e30f, v[4];
                for (int c = 0; c < 4; ++c) {
                    uint32_t bits = uint32_t(src[row * 4 + c]) << 16;
                    memcpy(&v[c], &bits, 4);
                    mx = std::max(mx, v[c]);
                }
                float sum = 0.f;
                for (int c = 0; c < 4; ++c) { v[c] = std::exp(v[c] - mx); sum += v[c]; }
                for (int c = 0; c < 4; ++c) a[row * 4 + c] = v[c] / sum;
            }
            p_->fuse_alpha = p_->dalloc<float>(16);
            ck(cudaMemcpyAsync(p_->fuse_alpha, a, sizeof(a), cudaMemcpyHostToDevice, st), "fuse_alpha");
        }

        p_->mask_embed = p_->load_f32(sf, "mask_embed", E, st);
        p_->norm_w = p_->load_f32(sf, "norm.weight", E, st);

        __nv_bfloat16 * tmp_bf16 = nullptr;
        ck(cudaMalloc(&tmp_bf16, sizeof(__nv_bfloat16) * size_t(FF) * E), "tmp bf16");
        try {
            for (int l = 0; l < N_LAYERS; ++l) {
                const std::string pre = "layers." + std::to_string(l) + ".";
                Impl::LayerW & W = p_->L[l];
                W.dflare_norm = p_->load_f32(sf, pre + "dflare_norm.weight", E, st);
                W.in_norm = p_->load_f32(sf, pre + "input_layernorm.weight", E, st);
                W.post_norm = p_->load_f32(sf, pre + "post_attention_layernorm.weight", E, st);
                W.q_norm = p_->load_f32(sf, pre + "self_attn.q_norm.weight", D, st);
                W.k_norm = p_->load_f32(sf, pre + "self_attn.k_norm.weight", D, st);
                W.k_norm_ctx = p_->load_f32(sf, pre + "self_attn.k_norm_ctx.weight", D, st);
                W.q_proj = p_->load_q8(sf, pre + "self_attn.q_proj.weight", H * D, E, tmp_bf16, st);
                W.k_proj = p_->load_q8(sf, pre + "self_attn.k_proj.weight", KV, E, tmp_bf16, st);
                W.v_proj = p_->load_q8(sf, pre + "self_attn.v_proj.weight", KV, E, tmp_bf16, st);
                W.k_proj_ctx = p_->load_q8(sf, pre + "self_attn.k_proj_ctx.weight", KV, E, tmp_bf16, st);
                W.v_proj_ctx = p_->load_q8(sf, pre + "self_attn.v_proj_ctx.weight", KV, E, tmp_bf16, st);
                W.o_proj = p_->load_q8(sf, pre + "self_attn.o_proj.weight", E, H * D, tmp_bf16, st);
                W.gate_proj = p_->load_q8(sf, pre + "mlp.gate_proj.weight", FF, E, tmp_bf16, st);
                W.up_proj = p_->load_q8(sf, pre + "mlp.up_proj.weight", FF, E, tmp_bf16, st);
                W.down_proj = p_->load_q8(sf, pre + "mlp.down_proj.weight", E, FF, tmp_bf16, st);

                const bool is_local = (l == 0 || l == 2);
                W.window = is_local ? SWA : 0;
                W.ring = is_local ? RING : 0;
                const size_t rows = is_local ? (size_t) RING : (size_t) ctx;
                W.k_cache = p_->dalloc<__half>(rows * KV);
                W.v_cache = p_->dalloc<__half>(rows * KV);
            }
        } catch (...) {
            cudaFree(tmp_bf16);
            throw;
        }
        cudaFree(tmp_bf16);

        p_->mh_dflare_norm = p_->load_f32(sf, "markov_head.dflare_norm.weight", E, st);
        p_->mh_rnn_bias = p_->load_f32(sf, "markov_head.rnn_proj.bias", 3 * RANK, st);
        p_->mh_rnn_w = p_->load_bf16(sf, "markov_head.rnn_proj.weight", size_t(3 * RANK) * (2 * RANK + 2 * E), st);
        {
            const STEntry * e = nullptr;
            sf.data("markov_head.markov_w1.weight", &e);
            const size_t elems = (e->off1 - e->off0) / 2;
            p_->mh_w1 = p_->load_bf16(sf, "markov_head.markov_w1.weight", elems, st);
        }

        // Fold hidden_out ([E, RANK] + bias [E]) into target lm_head -> Q8_0 fold_q [n_vocab, 256] + fold_d [n_vocab, 8] + fold_b [n_vocab]
        {
            const STEntry * ew = nullptr, * eb = nullptr;
            const uint16_t * hw = reinterpret_cast<const uint16_t *>(sf.data("markov_head.hidden_out.weight", &ew));
            const uint16_t * hb = reinterpret_cast<const uint16_t *>(sf.data("markov_head.hidden_out.bias", &eb));
            // Transpose hidden_out.weight [E, RANK] -> [RANK, E] on host
            std::vector<float> h_cols(size_t(RANK) * E);
            for (int r = 0; r < E; ++r) {
                for (int c = 0; c < RANK; ++c) {
                    uint32_t bits = uint32_t(hw[size_t(r) * RANK + c]) << 16;
                    float f; memcpy(&f, &bits, 4);
                    h_cols[size_t(c) * E + r] = f;
                }
            }
            std::vector<float> h_bias(E);
            for (int r = 0; r < E; ++r) {
                uint32_t bits = uint32_t(hb[r]) << 16;
                memcpy(&h_bias[r], &bits, 4);
            }
            p_->fold_q = p_->dalloc<int8_t>(size_t(n_vocab) * RANK);
            p_->fold_d = p_->dalloc<__half>(size_t(n_vocab) * (RANK / 32));
            p_->fold_b = p_->dalloc<float>(n_vocab);
            constexpr int CBLK = 16;
            float * d_in = nullptr, * d_out = nullptr;
            __half * tmp_fold = nullptr;
            ck(cudaMalloc(&d_in, sizeof(float) * CBLK * E), "fold tmp in");
            ck(cudaMalloc(&d_out, sizeof(float) * CBLK * n_vocab_pad), "fold tmp out");
            ck(cudaMalloc(&tmp_fold, sizeof(__half) * size_t(n_vocab) * RANK), "fold tmp f16");
            try {
                // fold_b = lm_head(hidden_out.bias)
                ck(cudaMemcpyAsync(d_in, h_bias.data(), sizeof(float) * E, cudaMemcpyHostToDevice, st), "fold bias");
                lm_head(d_in, d_out, 1, st);
                ck(cudaMemcpyAsync(p_->fold_b, d_out, sizeof(float) * n_vocab, cudaMemcpyDeviceToDevice, st), "fold bias out");
                // fold_w columns in chunks of 16 -> temporary fp16, then quantize to Q8_0
                for (int c0 = 0; c0 < RANK; c0 += CBLK) {
                    const int C = std::min(CBLK, RANK - c0);
                    ck(cudaMemcpyAsync(d_in, h_cols.data() + size_t(c0) * E, sizeof(float) * C * E, cudaMemcpyHostToDevice, st), "fold col");
                    lm_head(d_in, d_out, C, st);
                    scatter_fold_cols_kernel<<<dim3((n_vocab + 255) / 256, C), 256, 0, st>>>(d_out, tmp_fold, n_vocab, n_vocab_pad, c0, C);
                }
                const size_t n_blocks = size_t(n_vocab) * (RANK / 32);
                quantize_f16_to_q8_kernel<<<(unsigned) ((n_blocks + 7) / 8), 256, 0, st>>>(tmp_fold, p_->fold_q, p_->fold_d, n_blocks);
                ck(cudaStreamSynchronize(st), "fold sync");
            } catch (...) {
                cudaFree(d_in);
                cudaFree(d_out);
                cudaFree(tmp_fold);
                throw;
            }
            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(tmp_fold);
        }

        // Runtime scratch
        p_->last_taps = p_->dalloc<float>(N_TAPS * E);
        p_->buf_x = p_->dalloc<float>(BLOCK * E);
        p_->buf_xn = p_->dalloc<float>(BLOCK * E);
        p_->buf_q = p_->dalloc<float>(BLOCK * H * D);
        p_->buf_k = p_->dalloc<float>(BLOCK * KV);
        p_->buf_v = p_->dalloc<float>(BLOCK * KV);
        p_->buf_o = p_->dalloc<float>(BLOCK * H * D);
        p_->buf_ff = p_->dalloc<float>(BLOCK * FF);
        p_->buf_ctx_tmp = p_->dalloc<float>(256 * E);
        p_->buf_ctx_k = p_->dalloc<float>(256 * KV);
        p_->buf_ctx_v = p_->dalloc<float>(256 * KV);
        p_->buf_f_rnn = p_->dalloc<float>(E);
        p_->buf_mh_ctx = p_->dalloc<float>(N_SPEC * 3 * RANK);
        p_->buf_state = p_->dalloc<float>(RANK);
        p_->buf_latent = p_->dalloc<float>(RANK);
        p_->buf_base_logits = p_->dalloc<float>(size_t(N_SPEC) * n_vocab_pad);
        p_->buf_blk_val = p_->dalloc<float>(256);
        p_->buf_blk_idx = p_->dalloc<int32_t>(256);
        p_->d_drafts = p_->dalloc<int32_t>(N_SPEC);
        p_->d_pos_buf = p_->dalloc<int32_t>(1);
        p_->d_anchor_tok = p_->dalloc<int32_t>(1);
        p_->d_anchor_buf = p_->dalloc<float>(E);
        p_->d_ctx_taps_buf = p_->dalloc<float>(BLOCK * N_TAPS * E);
        ck(cudaHostAlloc((void **) &p_->h_drafts, sizeof(int32_t) * N_SPEC, cudaHostAllocDefault), "h_drafts");
        ck(cudaHostAlloc((void **) &p_->h_scalars, sizeof(int32_t) * 2, cudaHostAllocDefault), "h_scalars");
        fprintf(stderr, "inkling: DSpark MTP loaded (%s, 3 Q8_0 layers + rank-%d Q8_0 Markov head, block=%d)\n",
                path.c_str(), RANK, BLOCK);
    } catch (...) {
        delete p_;
        p_ = nullptr;
        throw;
    }
}

DSpark::~DSpark() { delete p_; }

void DSpark::reset() {}

void DSpark::store_context(const float * d_taps, int pos0, int T, cudaStream_t st) {
    if (T <= 0) return;
    if (T <= BLOCK) {
        // Fast CUDA-graphed path for decode (T=1) and verify_commit (T=1..8): 1 graph launch instead of 13 eager kernels!
        set_i32_kernel<<<1, 1, 0, st>>>(p_->d_pos_buf, pos0);
        ck(cudaMemcpyAsync(p_->d_ctx_taps_buf, d_taps, sizeof(float) * size_t(T) * N_TAPS * E, cudaMemcpyDeviceToDevice, st), "ctx taps");
        if (!p_->ctx_e[T]) {
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "ctx capture");
            for (int l = 0; l < N_LAYERS; ++l) {
                Impl::LayerW & W = p_->L[l];
                mix_rmsnorm_kernel<<<T, 256, 0, st>>>(p_->d_ctx_taps_buf, p_->fuse_alpha + l * 4, W.dflare_norm, p_->buf_ctx_tmp, E, EPS);
                launch_q8_gemm(W.k_proj_ctx, p_->buf_ctx_tmp, p_->buf_ctx_k, KV, E, T, st);
                launch_q8_gemm(W.v_proj_ctx, p_->buf_ctx_tmp, p_->buf_ctx_v, KV, E, T, st);
                store_ctx_kv_kernel<<<dim3(HK, T), 64, 0, st>>>(p_->buf_ctx_k, p_->buf_ctx_v, W.k_norm_ctx, W.k_cache, W.v_cache,
                                                                0, p_->d_pos_buf, W.ring, EPS);
            }
            const float * last = p_->d_ctx_taps_buf + size_t(T - 1) * N_TAPS * E;
            copy_f32_kernel<<<(N_TAPS * E + 255) / 256, 256, 0, st>>>(last, p_->last_taps, N_TAPS * E);
            ck(cudaStreamEndCapture(st, &p_->ctx_g[T]), "ctx end capture");
            ck(cudaGraphInstantiate(&p_->ctx_e[T], p_->ctx_g[T], 0), "ctx instantiate");
        }
        ck(cudaGraphLaunch(p_->ctx_e[T], st), "ctx launch");
        return;
    }
    // Prefill chunks (T > 8): process in chunks of at most 256 tokens
    for (int off = 0; off < T; off += 256) {
        const int n = std::min(256, T - off);
        const float * tp = d_taps + size_t(off) * N_TAPS * E;
        for (int l = 0; l < N_LAYERS; ++l) {
            Impl::LayerW & W = p_->L[l];
            mix_rmsnorm_kernel<<<n, 256, 0, st>>>(tp, p_->fuse_alpha + l * 4, W.dflare_norm, p_->buf_ctx_tmp, E, EPS);
            launch_q8_gemm(W.k_proj_ctx, p_->buf_ctx_tmp, p_->buf_ctx_k, KV, E, n, st);
            launch_q8_gemm(W.v_proj_ctx, p_->buf_ctx_tmp, p_->buf_ctx_v, KV, E, n, st);
            store_ctx_kv_kernel<<<dim3(HK, n), 64, 0, st>>>(p_->buf_ctx_k, p_->buf_ctx_v, W.k_norm_ctx, W.k_cache, W.v_cache,
                                                            pos0 + off, nullptr, W.ring, EPS);
        }
    }
    const float * last = d_taps + size_t(T - 1) * N_TAPS * E;
    copy_f32_kernel<<<(N_TAPS * E + 255) / 256, 256, 0, st>>>(last, p_->last_taps, N_TAPS * E);
}

void DSpark::draft(const float * d_anchor_emb, int anchor_tok, int pos, int n_draft,
                   const LmHeadFn & lm_head, int * out_drafts, cudaStream_t st) {
    const int nd = std::max(1, std::min(N_SPEC, n_draft));
    set_i32_2_kernel<<<1, 1, 0, st>>>(p_->d_pos_buf, pos, p_->d_anchor_tok, anchor_tok);
    ck(cudaMemcpyAsync(p_->d_anchor_buf, d_anchor_emb, sizeof(float) * E, cudaMemcpyDeviceToDevice, st), "draft emb");

    if (!p_->draft_e[nd]) {
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "draft capture");
        // 1. Build 8-token block: [anchor_emb, mask_embed x 7]
        init_block_kernel<<<BLOCK, 256, 0, st>>>(p_->d_anchor_buf, p_->mask_embed, p_->buf_x, E);

        // 2. Run 3 dense Q8_0 transformer layers over [context KV (0..pos-1), block KV (pos..pos+7)]
        rmsnorm_kernel<<<BLOCK, 256, 0, st>>>(p_->buf_x, p_->L[0].in_norm, p_->buf_xn, E, EPS);
        for (int l = 0; l < N_LAYERS; ++l) {
            Impl::LayerW & W = p_->L[l];
            launch_q8_gemm(W.q_proj, p_->buf_xn, p_->buf_q, H * D, E, BLOCK, st);
            launch_q8_gemm(W.k_proj, p_->buf_xn, p_->buf_k, KV, E, BLOCK, st);
            launch_q8_gemm(W.v_proj, p_->buf_xn, p_->buf_v, KV, E, BLOCK, st);
            qk_norm_rope_kernel<<<dim3(H, BLOCK), 64, 0, st>>>(p_->buf_q, W.q_norm, H, 0, p_->d_pos_buf, EPS);
            qk_norm_rope_kernel<<<dim3(HK, BLOCK), 64, 0, st>>>(p_->buf_k, W.k_norm, HK, 0, p_->d_pos_buf, EPS);
            block_attn_kernel<<<dim3(H, BLOCK), 256, 0, st>>>(p_->buf_q, W.k_cache, W.v_cache, p_->buf_k, p_->buf_v, p_->buf_o,
                                                              0, p_->d_pos_buf, W.ring, W.window);
            launch_q8_gemm(W.o_proj, p_->buf_o, p_->buf_xn, E, H * D, BLOCK, st);
            add_rmsnorm_kernel<<<BLOCK, 256, 0, st>>>(p_->buf_x, p_->buf_xn, W.post_norm, p_->buf_xn, E, EPS);
            {
                const int blocks = (FF + 7) / 8, grid = blocks < 1024 ? blocks : 1024;
                q8_glu_kernel<<<grid, 256, 0, st>>>(W.gate_proj.q, W.gate_proj.d, W.up_proj.q, W.up_proj.d,
                                                    p_->buf_xn, p_->buf_ff, FF, E, BLOCK);
            }
            launch_q8_gemm(W.down_proj, p_->buf_ff, p_->buf_xn, E, FF, BLOCK, st);
            const float * next_w = (l + 1 < N_LAYERS) ? p_->L[l + 1].in_norm : p_->norm_w;
            add_rmsnorm_kernel<<<BLOCK, 256, 0, st>>>(p_->buf_x, p_->buf_xn, next_w, p_->buf_xn, E, EPS);
        }

        // 3. Base logits for positions 1..nd (skipping anchor at index 0)
        const float * h_draft = p_->buf_xn + E;
        lm_head(h_draft, p_->buf_base_logits, nd, st);

        // 4. Recurrent Markov head: f_rnn -> ctx[0..nd-1] -> nd sequential steps 100% on GPU
        mix_rmsnorm_kernel<<<1, 256, 0, st>>>(p_->last_taps, p_->fuse_alpha + 3 * 4, p_->mh_dflare_norm, p_->buf_f_rnn, E, EPS);
        markov_ctx_kernel<<<dim3((3 * RANK + 7) / 8, nd), 256, 0, st>>>(p_->mh_rnn_w, p_->mh_rnn_bias, h_draft, p_->buf_f_rnn, p_->buf_mh_ctx);

        for (int i = 0; i < nd; ++i) {
            markov_recur_kernel<<<1, RANK, 0, st>>>(p_->mh_rnn_w, p_->mh_w1, p_->buf_mh_ctx + size_t(i) * 3 * RANK,
                                                    p_->buf_state, p_->buf_latent,
                                                    i == 0 ? p_->d_anchor_tok : p_->d_drafts + (i - 1), i);
            markov_fold_argmax1_kernel<<<256, 256, 0, st>>>(p_->buf_base_logits + size_t(i) * p_->n_vocab_pad,
                                                            p_->fold_q, p_->fold_d, p_->fold_b, p_->buf_latent,
                                                            p_->n_vocab, p_->softcap, p_->buf_blk_val, p_->buf_blk_idx);
            markov_fold_argmax2_kernel<<<1, 256, 0, st>>>(p_->buf_blk_val, p_->buf_blk_idx, 256, p_->d_drafts + i);
        }
        ck(cudaMemcpyAsync(p_->h_drafts, p_->d_drafts, sizeof(int32_t) * nd, cudaMemcpyDeviceToHost, st), "drafts d2h");
        ck(cudaStreamEndCapture(st, &p_->draft_g[nd]), "draft end capture");
        ck(cudaGraphInstantiate(&p_->draft_e[nd], p_->draft_g[nd], 0), "draft instantiate");
    }
    ck(cudaGraphLaunch(p_->draft_e[nd], st), "draft launch");
    ck(cudaStreamSynchronize(st), "draft sync");
    for (int i = 0; i < nd; ++i) out_drafts[i] = p_->h_drafts[i];
}

size_t DSpark::ck_bytes() const {
    size_t b = sizeof(float) * N_TAPS * E;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (p_->L[l].ring) b += 2 * sizeof(__half) * size_t(p_->L[l].ring) * KV;
    }
    return b;
}

void DSpark::ck_save(uint8_t * dst, cudaStream_t st) const {
    const size_t tb = sizeof(float) * N_TAPS * E;
    ck(cudaMemcpyAsync(dst, p_->last_taps, tb, cudaMemcpyDeviceToHost, st), "dspark ck"); dst += tb;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (p_->L[l].ring) {
            const size_t kb = sizeof(__half) * size_t(p_->L[l].ring) * KV;
            ck(cudaMemcpyAsync(dst, p_->L[l].k_cache, kb, cudaMemcpyDeviceToHost, st), "dspark ck"); dst += kb;
            ck(cudaMemcpyAsync(dst, p_->L[l].v_cache, kb, cudaMemcpyDeviceToHost, st), "dspark ck"); dst += kb;
        }
    }
}

void DSpark::ck_restore(const uint8_t * src, cudaStream_t st) {
    const size_t tb = sizeof(float) * N_TAPS * E;
    ck(cudaMemcpyAsync(p_->last_taps, src, tb, cudaMemcpyHostToDevice, st), "dspark restore"); src += tb;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (p_->L[l].ring) {
            const size_t kb = sizeof(__half) * size_t(p_->L[l].ring) * KV;
            ck(cudaMemcpyAsync(p_->L[l].k_cache, src, kb, cudaMemcpyHostToDevice, st), "dspark restore"); src += kb;
            ck(cudaMemcpyAsync(p_->L[l].v_cache, src, kb, cudaMemcpyHostToDevice, st), "dspark restore"); src += kb;
        }
    }
}

}  // namespace ik

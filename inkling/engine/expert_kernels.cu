// Expert kernels for Inkling 2 Mini decode: NVFP4 and Q8_0 matrix-vector products over the selected experts.
//
//   ik_experts_gate_up: h[k][r] = situ(scale_gu[e] * (W_gu[e][r] . x), scale_gu[e] * (W_gu[e][r + n_ff] . x)), e = ids[k]
//   ik_experts_down:    y[r]    = sum_k w[k] * scale_dn[e] * (W_dn[e][r] . h[k])
//
// Weights are kept in the engine's repacked layout (ik_repack), one contiguous blob per expert matrix [rows][n]:
//   NVFP4: q  uint32 [rows][n/8]  word w holds the E2M1 codes of values 8w..8w+7, value 8w+i in bits 4i..4i+3
//          d  uint8  [rows][n/16] UE4M3 scale of values 16s..16s+15
//   Q8_0:  q  int8   [rows][n]
//          d  fp16   [rows][n/32]
// The byte count equals the GGUF blocks'. LPR lanes share a row; lane j of them handles values 8j..8j+7 of every
// 8*LPR-value step, so weight loads are contiguous, and the matching slice of the input vector stays in the lane's
// registers across rows. Steps divide evenly: 2304 values = 9 steps of 32 lanes, 1152 = 9 steps of 16 lanes.
// Accumulation is fp32 with fp32 activations; the down product writes per-expert partial sums that a second kernel
// adds in a fixed order, so results are deterministic.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARP = 32;
constexpr int VPL = 8;  // values per lane per step

// UE4M3 (bias 7) to float, halved (ggml_ue4m3_to_fp32). Normal values: the 4 exponent and 3 mantissa bits drop into
// the fp32 fields, rebiased by 120 (7 -> 127) minus 1 for the halving. Subnormals (exponent 0): m * 2^-10.
__device__ __forceinline__ float ue4m3(uint32_t x) {
    x &= 0x7F;
    const float normal = __int_as_float(int((x << 20) + (119u << 23)));
    const float sub = float(x) * 0.0009765625f;  // 2^-10
    return x == 0x7F ? 0.f : (x < 8 ? sub : normal);
}

// E2M1 code to its value doubled (ggml kvalues_mxfp4: 0 1 2 3 4 6 8 12, bit 3 = sign); ue4m3() halves the scale to
// match. The table lives in shared memory: its 16 entries sit in 16 banks, so lanes reading different entries do not
// conflict (a __constant__ table would serialise them), and no int-to-float conversion is needed (quarter rate on
// sm_86).
__device__ const float k_fp4_init[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, -0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

__device__ __forceinline__ void init_lut(float * lut) {
    if (threadIdx.x < 16) lut[threadIdx.x] = k_fp4_init[threadIdx.x];
    __syncthreads();
}

template <int LPR>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <bool NVFP4>
__host__ __device__ inline size_t matrix_bytes(int rows, int n) {
    return NVFP4 ? size_t(rows) * n / 2 + size_t(rows) * n / 16 : size_t(rows) * n + size_t(rows) * (n / 32) * 2;
}

// partial dot of row r (n values, matrix of `rows` rows) with the lane's x slice; j = lane index within the row group
template <bool NVFP4, int LPR, int STEPS>
__device__ __forceinline__ float dot(const uint8_t * m, int rows, int n, int r, const float (&xr)[STEPS][VPL], int j,
                                     const float * lut) {
    float acc = 0.f;
    if constexpr (NVFP4) {
        const uint32_t * q = reinterpret_cast<const uint32_t *>(m) + size_t(r) * (n / 8);
        const uint8_t * d = m + size_t(rows) * n / 2 + size_t(r) * (n / 16);
#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            const int w = s * LPR + j;
            const uint32_t qw = __ldg(q + w);
            const float sc = ue4m3(__ldg(d + (w >> 1)));
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a0 += lut[(qw >> (4 * i)) & 0xF] * xr[s][i];
                a1 += lut[(qw >> (4 * i + 16)) & 0xF] * xr[s][4 + i];
            }
            acc += (a0 + a1) * sc;
        }
    } else {
        const uint2 * q = reinterpret_cast<const uint2 *>(m + size_t(r) * n);
        const __half * d = reinterpret_cast<const __half *>(m + size_t(rows) * n) + size_t(r) * (n / 32);
#pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            const int w = s * LPR + j;
            const uint2 qw = __ldg(q + w);
            const float sc = __half2float(__ldg(d + (w >> 2)));
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a0 += float(int8_t(qw.x >> (8 * i))) * xr[s][i];
                a1 += float(int8_t(qw.y >> (8 * i))) * xr[s][4 + i];
            }
            acc += (a0 + a1) * sc;
        }
    }
    return acc;
}

template <int LPR, int STEPS>
__device__ __forceinline__ void load_slice(const float * x, int j, float (&xr)[STEPS][VPL]) {
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
        const float4 * p = reinterpret_cast<const float4 *>(x + (s * LPR + j) * VPL);
        const float4 a = p[0], b = p[1];
        xr[s][0] = a.x; xr[s][1] = a.y; xr[s][2] = a.z; xr[s][3] = a.w;
        xr[s][4] = b.x; xr[s][5] = b.y; xr[s][6] = b.z; xr[s][7] = b.w;
    }
}

__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
    const float act = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
    if (lb > 0.f) u = lb * tanhf(u / lb);
    return act * u;
}

// GROUPS = 32/LPR rows per warp at a time; grid (blocks, n_sel); rows walked with a grid stride
// EMBD: input length (2304); the gate/up row pair r, r + n_ff is computed by the same lanes
template <bool NVFP4, int LPR, int EMBD>
__global__ void __launch_bounds__(256) gate_up_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                      const int32_t * __restrict__ ids, const float * __restrict__ x,
                                                      float * __restrict__ h, int n_ff, float beta, float lb) {
    constexpr int STEPS = EMBD / (VPL * LPR), GROUPS = WARP / LPR;
    __shared__ float lut[16];
    init_lut(lut);
    const int lane = threadIdx.x % WARP, j = lane % LPR;
    const int k = blockIdx.y, e = ids[k];
    const uint8_t * m = experts[e];
    if (!m) return;  // not in the VRAM cache: the CPU computes this expert
    float xr[STEPS][VPL];
    load_slice<LPR, STEPS>(x, j, xr);
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP) * GROUPS;
    for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * GROUPS + lane / LPR; r < n_ff; r += stride) {
        const float g = group_sum<LPR>(dot<NVFP4, LPR, STEPS>(m, 2 * n_ff, EMBD, r, xr, j, lut));
        const float u = group_sum<LPR>(dot<NVFP4, LPR, STEPS>(m, 2 * n_ff, EMBD, r + n_ff, xr, j, lut));
        if (j == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
    }
}

// partial[k][r] = w[k] * scale_dn[e] * (W_dn[e][r] . h[k]); grid (blocks, n_sel)
template <bool NVFP4, int LPR, int FF>
__global__ void __launch_bounds__(256) down_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                   const int32_t * __restrict__ ids, const float * __restrict__ wts,
                                                   const float * __restrict__ h, float * __restrict__ partial, int n_embd) {
    constexpr int STEPS = FF / (VPL * LPR), GROUPS = WARP / LPR;
    __shared__ float lut[16];
    init_lut(lut);
    const int lane = threadIdx.x % WARP, j = lane % LPR;
    const int k = blockIdx.y, e = ids[k];
    const uint8_t * m = experts[e];
    const int stride = gridDim.x * (blockDim.x / WARP) * GROUPS;
    if (!m) {  // not in the VRAM cache: contributes nothing here (the CPU adds it)
        for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * GROUPS + lane / LPR; r < n_embd; r += stride)
            if (j == 0) partial[size_t(k) * n_embd + r] = 0.f;
        return;
    }
    float xr[STEPS][VPL];
    load_slice<LPR, STEPS>(h + size_t(k) * FF, j, xr);
    const float c = wts[k] * (scale ? scale[e] : 1.f);
    for (int r = (blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP) * GROUPS + lane / LPR; r < n_embd; r += stride) {
        const float v = group_sum<LPR>(dot<NVFP4, LPR, STEPS>(m, n_embd, FF, r, xr, j, lut));
        if (j == 0) partial[size_t(k) * n_embd + r] = c * v;
    }
}

// y[r] = W[r] . x for a dense repacked Q8_0 matrix [rows][N]; one warp per row
template <int N>
__global__ void __launch_bounds__(256) matvec_q8_0_kernel(const uint8_t * __restrict__ m, const float * __restrict__ x,
                                                          float * __restrict__ y, int rows) {
    constexpr int LPR = 32, STEPS = N / (VPL * LPR);
    const int lane = threadIdx.x % WARP;
    float xr[STEPS][VPL];
    load_slice<LPR, STEPS>(x, lane, xr);
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const float v = group_sum<LPR>(dot<false, LPR, STEPS>(m, rows, N, r, xr, lane, nullptr));
        if (lane == 0) y[r] = v;
    }
}

// fused shared-input Q8_0 matvecs: up to 5 projections sharing one x (loaded into registers once per warp)
struct MultiQ8Args {
    const uint8_t * w[5];
    float * y[5];
    int rows[5];
    int off[6];  // prefix sums of rows
    int count;
};

template <int N>
__global__ void __launch_bounds__(256) matvec_q8_0_multi_kernel(MultiQ8Args a, const float * __restrict__ x) {
    constexpr int LPR = 32, STEPS = N / (VPL * LPR);
    const int lane = threadIdx.x % WARP;
    float xr[STEPS][VPL];
    load_slice<LPR, STEPS>(x, lane, xr);
    const int total = a.off[a.count];
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int g = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; g < total; g += stride) {
        int m = 0;
        while (m + 1 < a.count && g >= a.off[m + 1]) ++m;
        const int r = g - a.off[m];
        const float v = group_sum<LPR>(dot<false, LPR, STEPS>(a.w[m], a.rows[m], N, r, xr, lane, nullptr));
        if (lane == 0) a.y[m][r] = v;
    }
}

__global__ void sum_partials_kernel(const float * __restrict__ partial, float * __restrict__ y, int n_embd, int n_sel) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_embd) return;
    float acc = 0.f;
    for (int k = 0; k < n_sel; ++k) acc += partial[size_t(k) * n_embd + r];
    y[r] = acc;
}

// GGUF blocks of one matrix [rows][n] -> repacked layout
__global__ void repack_nvfp4_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int n) {
    const size_t n_sub = size_t(rows) * n / 16;
    uint32_t * q = reinterpret_cast<uint32_t *>(dst);
    uint8_t * d = dst + size_t(rows) * n / 2;
    for (size_t g = blockIdx.x * size_t(blockDim.x) + threadIdx.x; g < n_sub; g += size_t(gridDim.x) * blockDim.x) {
        // sub-block g = 16 values; GGUF: block g/4, scale byte g%4, 8 bytes of nibbles (low = value j, high = value 8+j)
        const uint8_t * blk = src + (g / 4) * 36;
        const int s = int(g % 4);
        d[g] = blk[s];
        const uint8_t * qs = blk + 4 + s * 8;
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            lo |= uint32_t(qs[j] & 0xF) << (4 * j);
            hi |= uint32_t(qs[j] >> 4) << (4 * j);
        }
        q[2 * g] = lo;
        q[2 * g + 1] = hi;
    }
}

__global__ void repack_q8_0_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int n) {
    const size_t nb = size_t(rows) * n / 32;
    int8_t * q = reinterpret_cast<int8_t *>(dst);
    __half * d = reinterpret_cast<__half *>(dst + size_t(rows) * n);
    for (size_t b = blockIdx.x * size_t(blockDim.x) + threadIdx.x; b < nb; b += size_t(gridDim.x) * blockDim.x) {
        const uint8_t * blk = src + b * 34;
        d[b] = *reinterpret_cast<const __half *>(blk);  // 34-byte blocks keep the fp16 2-aligned
#pragma unroll
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = int8_t(blk[2 + j]);
    }
}

constexpr int TPB = 256;
constexpr int N_EMBD = 2304, N_FF = 1152;  // Inkling 2 Mini; the register slices are sized for these

}  // namespace

extern "C" {

// bytes of one repacked expert matrix; type: 0 = NVFP4, 1 = Q8_0
size_t ik_matrix_bytes(int type, int rows, int n) { return type == 0 ? matrix_bytes<true>(rows, n) : matrix_bytes<false>(rows, n); }

// repack one matrix (GGUF blocks, rows x n) into the engine layout; src and dst are device pointers
int ik_repack(int type, const void * src, void * dst, int rows, int n, cudaStream_t st) {
    if (type == 0) repack_nvfp4_kernel<<<1024, TPB, 0, st>>>((const uint8_t *) src, (uint8_t *) dst, rows, n);
    else           repack_q8_0_kernel<<<1024, TPB, 0, st>>>((const uint8_t *) src, (uint8_t *) dst, rows, n);
    return (int) cudaGetLastError();
}

// y = W x for a dense repacked Q8_0 matrix [rows][n], n = 2304 or 3072; device pointers
int ik_matvec_q8_0(const void * w, const float * x, float * y, int rows, int n, cudaStream_t st) {
    const int blocks = (rows + TPB / WARP - 1) / (TPB / WARP);
    const int grid = blocks < 1024 ? blocks : 1024;
    if (n == 2304)      matvec_q8_0_kernel<2304><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows);
    else if (n == 3072) matvec_q8_0_kernel<3072><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows);
    else return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

int ik_matvec_q8_0_multi(int count, const void * const * ws, const float * x, float * const * ys, const int * rows, int n, cudaStream_t st) {
    if (count < 1 || count > 5 || n != 2304) return (int) cudaErrorInvalidValue;
    MultiQ8Args a{};
    a.count = count;
    a.off[0] = 0;
    for (int i = 0; i < count; ++i) {
        a.w[i] = (const uint8_t *) ws[i];
        a.y[i] = ys[i];
        a.rows[i] = rows[i];
        a.off[i + 1] = a.off[i] + rows[i];
    }
    const int blocks = (a.off[count] + TPB / WARP - 1) / (TPB / WARP);
    const int grid = blocks < 1024 ? blocks : 1024;
    matvec_q8_0_multi_kernel<2304><<<grid, TPB, 0, st>>>(a, x);
    return (int) cudaGetLastError();
}

// experts: device array of n_expert device pointers to repacked gate_up matrices [2*n_ff][n_embd];
// ids [n_sel], x [n_embd], h [n_sel][n_ff] on the device. Returns a cudaError_t.
int ik_experts_gate_up(int type, const void * const * experts, const float * scale, const int32_t * ids, int n_sel,
                       const float * x, float * h, int n_embd, int n_ff, float beta, float lb, cudaStream_t st) {
    if (n_embd != N_EMBD || n_ff != N_FF) return (int) cudaErrorInvalidValue;
    const dim3 grid(36, n_sel);  // 288 warps per expert, 4 row pairs each
    const auto * ex = (const uint8_t * const *) experts;
    if (type == 0) gate_up_kernel<true, 32, N_EMBD><<<grid, TPB, 0, st>>>(ex, scale, ids, x, h, n_ff, beta, lb);
    else           gate_up_kernel<false, 32, N_EMBD><<<grid, TPB, 0, st>>>(ex, scale, ids, x, h, n_ff, beta, lb);
    return (int) cudaGetLastError();
}

// experts: device array of device pointers to repacked down matrices [n_embd][n_ff]; partial: [n_sel][n_embd] scratch
int ik_experts_down(int type, const void * const * experts, const float * scale, const int32_t * ids, const float * wts, int n_sel,
                    const float * h, float * partial, float * y, int n_embd, int n_ff, cudaStream_t st) {
    if (n_embd != N_EMBD || n_ff != N_FF) return (int) cudaErrorInvalidValue;
    const dim3 grid(36, n_sel);  // 288 warps per expert, 2 rows at a time, 8 rows each
    const auto * ex = (const uint8_t * const *) experts;
    if (type == 0) down_kernel<true, 16, N_FF><<<grid, TPB, 0, st>>>(ex, scale, ids, wts, h, partial, n_embd);
    else           down_kernel<false, 16, N_FF><<<grid, TPB, 0, st>>>(ex, scale, ids, wts, h, partial, n_embd);
    sum_partials_kernel<<<(n_embd + TPB - 1) / TPB, TPB, 0, st>>>(partial, y, n_embd, n_sel);
    return (int) cudaGetLastError();
}

}  // extern "C"

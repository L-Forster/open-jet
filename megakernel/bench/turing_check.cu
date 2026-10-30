// Model-free checks of the actual decode/prefill kernels, not replacement implementations.
// nvcc -O3 -std=c++17 -arch=sm_75 -DQWEN_MK_SM75=1 -I$LLAMA/ggml/include \
//   turing_check.cu -L$LLAMA/build/bin -lggml-base -o turing_check
#include "../qwen38_mk.cu"

namespace {

std::vector<void *> allocations;

template <class T> T * buffer(size_t n) {
    T * p;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    allocations.push_back(p);
    return p;
}

template <class T> T * upload(const std::vector<T> & v) {
    T * p = buffer<T>(v.size());
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

template <class T> std::vector<T> download(const T * p, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

void release() {
    for (void * p : allocations) CK(cudaFree(p));
    allocations.clear();
}

void close(float got, float expected, float tolerance, const char * label) {
    if (!std::isfinite(got) || fabsf(got - expected) > tolerance)
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(got) +
                                 ", expected " + std::to_string(expected));
}

__device__ unsigned pair(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}

__device__ float av(int m, int k) { return float((m * 3 + k * 5) % 11 - 5); }
__device__ float bv(int k, int n) { return float((k * 7 + n * 2) % 13 - 6); }

__global__ void check_mma(float * f, float * h) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const unsigned a[4] = {pair(av(g, 2 * t), av(g, 2 * t + 1)),
                           pair(av(g + 8, 2 * t), av(g + 8, 2 * t + 1)),
                           pair(av(g, 2 * t + 8), av(g, 2 * t + 9)),
                           pair(av(g + 8, 2 * t + 8), av(g + 8, 2 * t + 9))};
    const unsigned b0 = pair(bv(2 * t, g), bv(2 * t + 1, g));
    const unsigned b1 = pair(bv(2 * t + 8, g), bv(2 * t + 9, g));
    float c[4] = {1.f, 2.f, 3.f, 4.f};
    unsigned d[2] = {pair(1, 2), pair(3, 4)};
    mma16816(c, a[0], a[1], a[2], a[3], b0, b1);
    pf::hmma16(d, a, b0, b1);
    const float2 lo = __half22float2(pf::u2h(d[0])), hi = __half22float2(pf::u2h(d[1]));
    const float hv[4] = {lo.x, lo.y, hi.x, hi.y};
    for (int e = 0; e < 4; e++) {
        const int m = g + (e / 2) * 8, n = 2 * t + (e & 1);
        f[m * 8 + n] = c[e];
        h[m * 8 + n] = hv[e];
    }
}

void mma_test() {
    float * df = buffer<float>(128), * dh = buffer<float>(128);
    check_mma<<<1, 32>>>(df, dh);
    CK(cudaGetLastError());
    const auto f = download(df, 128), h = download(dh, 128);
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            float expected = 1 + 2 * (m / 8) + (n & 1);
            for (int k = 0; k < 16; k++)
                expected += ((m * 3 + k * 5) % 11 - 5) * ((k * 7 + n * 2) % 13 - 6);
            close(f[m * 8 + n], expected, 0, "fp32 MMA lane mapping");
            close(h[m * 8 + n], expected, 0, "fp16 MMA lane mapping");
        }
    puts("PASS fp32/fp16 MMA k16 split, nonzero accumulation, all lanes");
    release();
}

QBuf quant_buffer(int units, int tokens) {
    QBuf q{};
    q.U = units;
    q.xa = buffer<int8_t>((size_t) tokens * units * 16);
    q.xb = buffer<int8_t>((size_t) tokens * units * 16);
    q.meta = buffer<uint2>((size_t) tokens * units);
    q.ssq = buffer<float>((size_t) tokens * units);
    return q;
}

template <int T, int TYPE>
__global__ void __launch_bounds__(NT, 1) check_resid(Mat w, QBuf q, float * x, const float * norm) {
    resid32<T, 17, TYPE>(w, q, x, false, norm, smem);
}

template <int T, int TYPE> void resid_test() {
    constexpr int units = U_FF, blocks = FF / 256, bsz = TYPE == T_Q4K ? 144 : Q6_BLK;
    std::vector<uint8_t> w((size_t) D * blocks * bsz, 0);
    std::vector<uint8_t> md(TYPE == T_Q6K ? (size_t) D * blocks * Q6_META : 0, 0); // Q6_K meta stream
    const __half one = __float2half(1), small = __float2half(1.f / 32);
    for (int r = 0; r < D; r++)
        for (int b = 0; b < blocks; b++) {
            uint8_t * p = w.data() + ((size_t) r * blocks + b) * bsz;
            const int value = 1 + (r + b) % 7;
            if constexpr (TYPE == T_Q4K) {
                memcpy(p, &one, 2);
                for (int j = 0; j < 4; j++) p[4 + j] = 1;
                for (int j = 8; j < 12; j++) p[4 + j] = 1;
                memset(p + 16, value * 17, 128);
            } else {
                memset(p, value * 17, 128);
                uint8_t * mp = md.data() + ((size_t) r * blocks + (b & ~1)) * Q6_META;
                memset(mp + 16 * (b & 1), 1, 16);
                memcpy(mp + 32 + 2 * (b & 1), &small, 2);
            }
        }
    QBuf q = quant_buffer(units, T);
    std::vector<int8_t> vals((size_t) T * units * 16);
    std::vector<uint2> meta((size_t) T * units);
    const __half2 ds = __floats2half2_rn(1, 1);
    unsigned dsbits;
    memcpy(&dsbits, &ds, 4);
    for (int t = 0; t < T; t++)
        for (int u = 0; u < units; u++) {
            const int v = (u + 2 * t) % 5 - 2;
            memset(vals.data() + ((size_t) t * units + u) * 16, (uint8_t) v, 16);
            const unsigned sum = (unsigned short) (16 * v);
            meta[t * units + u] = make_uint2(dsbits, sum | (sum << 16));
        }
    CK(cudaMemcpy(q.xa, vals.data(), vals.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q.xb, vals.data(), vals.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(q.meta, meta.data(), meta.size() * sizeof(uint2), cudaMemcpyHostToDevice));
    Bufs bufs{};
    bufs.qn = quant_buffer(U_D, T);
    CK(cudaMemcpyToSymbol(cB, &bufs, sizeof(bufs)));
    float * x = buffer<float>((size_t) T * D);
    const float * norm = upload(std::vector<float>(D, 1));
    const __half * mdev = nullptr;
    if constexpr (TYPE == T_Q6K) mdev = reinterpret_cast<const __half *>(upload(md));
    const Mat mat{upload(w), mdev, TYPE, FF, D, blocks * bsz, TYPE == T_Q6K ? blocks * (Q6_META / 2) : 0};
    CK(cudaFuncSetAttribute(check_resid<T, TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
    check_resid<T, TYPE><<<4, NT, SM_TOTAL>>>(mat, q, x, norm);
    CK(cudaGetLastError());
    const auto out = download(x, (size_t) T * D);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < D; r++) {
            float expected = 0;
            for (int u = 0; u < units; u++) {
                float weight = 1 + (r + u / 8) % 7;
                if constexpr (TYPE == T_Q6K) weight = (weight - 32) / 32;
                expected += 32 * weight * ((u + 2 * t) % 5 - 2);
            }
            close(out[(size_t) t * D + r], expected, 0, "sliced residual GEMV");
        }
    printf("PASS residual GEMV type=%d T=%d, K=%d, multi-slice and remainder\n", TYPE, T, FF);
    release();
}

float value(int p, int d) { return float((p + d % 17) % 13 - 6) / 16; }

Bufs kv_buffers(int ctx, bool q8) {
    Bufs b{};
    b.ctx = ctx;
    b.kvq8 = q8;
    if (q8) {
        std::vector<int8_t> v((size_t) NKV * ctx * HD);
        for (int g = 0; g < NKV; g++)
            for (int p = 0; p < ctx; p++)
                for (int d = 0; d < HD; d++) v[((size_t) g * ctx + p) * HD + d] = (int8_t) (value(p, d) * 16);
        b.kc = buffer<int8_t>(v.size());
        b.vc = upload(v);
        b.ksc = b.vsc = upload(std::vector<__half>((size_t) NKV * ctx * (HD / 32), __float2half(1.f / 16)));
    } else {
        std::vector<__half> v((size_t) NKV * ctx * HD);
        for (int g = 0; g < NKV; g++)
            for (int p = 0; p < ctx; p++)
                for (int d = 0; d < HD; d++) v[((size_t) g * ctx + p) * HD + d] = __float2half(value(p, d));
        b.kc = buffer<__half>(v.size());
        b.vc = upload(v);
    }
    return b;
}

template <bool Q8> void prefill_attention_test() {
    constexpr int tokens = 19, n0 = 7, ctx = 37;
    Bufs b = kv_buffers(ctx, Q8);
    PBufs p{};
    p.qh = buffer<__half>((size_t) tokens * NH * HD);
    p.y2 = buffer<float>((size_t) tokens * VD);
    CK(cudaMemcpyToSymbol(cB, &b, sizeof(b)));
    CK(cudaMemcpyToSymbol(cP, &p, sizeof(p)));
    CK(cudaFuncSetAttribute(pk_attn<Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
    pk_attn<Q8><<<dim3((tokens + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(tokens, n0, 0);
    CK(cudaGetLastError());
    const auto out = download(p.y2, (size_t) tokens * VD);
    for (int t = 0; t < tokens; t++)
        for (int d = 0; d < VD; d++) {
            float expected = 0;
            for (int k = 0; k <= n0 + t; k++) expected += value(k, d % HD);
            close(out[(size_t) t * VD + d], expected / (n0 + t + 1), .001f, "prefill attention");
        }
    printf("PASS prefill attention %s, causal/prefix/partial query and KV tiles\n", Q8 ? "q8" : "fp16");
    release();
}

template <bool Q8> __global__ void __launch_bounds__(NT, 1) check_decode_attention(LayerW w, int n) {
    attn_core_t<5, Q8>(w, 0, n);
}

template <bool Q8> void decode_attention_test() {
    constexpr int n = 70, tokens = 5, ctx = 80;
    Bufs b = kv_buffers(ctx, Q8);
    b.qbuf = buffer<float>((size_t) tokens * QG);
    b.kbuf = buffer<float>((size_t) tokens * KVD);
    std::vector<float> v((size_t) tokens * KVD);
    for (int t = 0; t < tokens; t++)
        for (int d = 0; d < KVD; d++) v[t * KVD + d] = value(n + t, d % HD);
    b.vbuf = upload(v);
    b.rope = upload(std::vector<float2>((size_t) ctx * 32, make_float2(1, 0)));
    b.apart = buffer<float>((size_t) NKV * MAX_CH * AR * HD);
    b.aml = buffer<float2>((size_t) NKV * MAX_CH * AR);
    LayerW w{};
    w.qn = w.kn = upload(std::vector<float>(HD, 1));
    CK(cudaMemcpyToSymbol(cB, &b, sizeof(b)));
    CK(cudaFuncSetAttribute(check_decode_attention<Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
    check_decode_attention<Q8><<<1, NT, SM_TOTAL>>>(w, n);
    CK(cudaGetLastError());
    int nch, cs;
    // Host equivalent of attn_dims.
    nch = std::max(1, std::min(20, (n + tokens + 63) / 64));
    cs = (n + tokens + nch - 1) / nch;
    const auto out = download(b.apart, (size_t) NKV * nch * AR * HD);
    const auto ml = download(b.aml, (size_t) NKV * nch * AR);
    for (int item = 0; item < NKV * nch; item++)
        for (int r = 0; r < 6 * tokens; r++) {
            const int start = (item % nch) * cs, end = std::min(start + cs, n + r % tokens + 1);
            const float2 stat = ml[item * AR + r];
            close(stat.x, 0, 0, "decode attention maximum");
            close(stat.y, end - start, .001f, "decode attention softmax denominator");
            for (int d = 0; d < HD; d++) {
                float expected = 0;
                for (int k = start; k < end; k++) expected += value(k, d);
                close(out[((size_t) item * AR + r) * HD + d], expected, Q8 ? .02f : .001f, "decode attention");
            }
        }
    printf("PASS decode attention %s, T=5, split-K, causal and KV tile tails\n", Q8 ? "q8" : "fp16");
    release();
}

} // namespace

int main() {
    try {
        CK(cudaFuncSetAttribute(mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
        int active = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, mk_cycle, NT, SM_TOTAL));
        if (active < 1) throw std::runtime_error("decode kernel occupancy is zero");
        cudaFuncAttributes attr{};
        CK(cudaFuncGetAttributes(&attr, mk_cycle));
        printf("PASS decode residency: %d CTA/SM, dynamic=%d static=%zu registers=%d\n",
               active, SM_TOTAL, attr.sharedSizeBytes, attr.numRegs);
        mma_test();
        resid_test<3, T_Q4K>(); resid_test<4, T_Q4K>(); resid_test<5, T_Q4K>();
        resid_test<3, T_Q6K>(); resid_test<4, T_Q6K>(); resid_test<5, T_Q6K>();
        prefill_attention_test<false>(); prefill_attention_test<true>();
        decode_attention_test<false>(); decode_attention_test<true>();
        CK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        release();
        return 1;
    }
}

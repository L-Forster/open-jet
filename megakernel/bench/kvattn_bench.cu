// Decode attention on low-bit K / V (MK_KV presets D, F, ...): the engine's kernels at a base revision against the working
// tree's, cut verbatim out of qwen38_mk.cu (bench/kvattn_gen.py), on one random KV cache. Compares partials, (m, l) and the
// rows written bit for bit, and times both with the same launch shape as the engine (82 blocks x 512 threads).
//   python3 kvattn_gen.py <repo> fbb8efb <gen>/kvattn_old.inc; python3 kvattn_gen.py <repo> WORK <gen>/kvattn_new.inc
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-dlcm=cg,-v -I.. -I<gen> kvattn_bench.cu -o kvattn_bench
//   ./kvattn_bench check [smoke]                               exactness matrix (formats x n x T), exit 1 on any difference
//   ./kvattn_bench time <F|D|q8> <n> <T> <rounds> <reps>       interleaved old / new, ms per call (KVB_Q8M=1: q8 data through new's MIX kernel)
//   ./kvattn_bench stress <F|D|q8> <n> <T> <runs>              new many times against one run of old: races one-shot check cannot see
// Good for exactness and coarse checks; ptxas schedules the same functions differently in the engine: decide role splits on engine stamps.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define MK_SHARED_NS // (anonymous)
#include "mk_shared.cuh"

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));        \
            exit(1);                                                                                                \
        }                                                                                                           \
    } while (0)

namespace {
__constant__ Bufs cBs[NVIEW];
__constant__ int cView;
__constant__ LayerW cL;
#define MK_ATT_PF_CTX (2 * ATT_BUCKET)
__shared__ int s_view;
#define cB cBs[s_view]
#define MK_VIEW_INIT()                          \
    do {                                        \
        if (threadIdx.x == 0) s_view = cView;   \
        __syncthreads();                        \
    } while (0)
extern __shared__ __align__(16) uint8_t smem[];
__shared__ int s_args[4]; // the call's scalars through shared memory, as in the engine (ml_n(), s_mtp): ptxas must not see them as uniform

namespace kold {
__device__ __forceinline__ void attn_idle_pf(const Mat &, int, int) {}
#undef MK_W48X
#define MK_W48X 0
#include "kvattn_old.inc"
} // namespace kold

namespace knew {
__device__ __forceinline__ void attn_idle_pf(const Mat &, int, int) {}
#undef MK_W48X
#ifndef KVB_NEW_X
#define KVB_NEW_X 1
#endif
#define MK_W48X KVB_NEW_X
#include "kvattn_new.inc"
#undef MK_W48X
} // namespace knew

#define KVB_KERNEL(NAME, NS)                                                                                        \
    __global__ void __launch_bounds__(NT, 1) NAME(int q8, int T, int slot_, int n_, int b0_, int reps) {            \
        MK_VIEW_INIT();                                                                                             \
        const LayerW W = cL;                                                                                        \
        if (threadIdx.x == 0) { s_args[0] = slot_; s_args[1] = n_; s_args[2] = b0_; }                               \
        __syncthreads();                                                                                            \
        for (int rep = 0; rep < reps; rep++) {                                                                      \
            const int slot = s_args[0], n = s_args[1], b0 = s_args[2];                                              \
            if (q8) {                                                                                               \
                switch (T) {                                                                                        \
                case 1: NS::attn_core_q8<1>(W, slot, n, b0); break;                                                 \
                case 2: NS::attn_core_q8<2>(W, slot, n, b0); break;                                                 \
                case 3: NS::attn_core_q8<3>(W, slot, n, b0); break;                                                 \
                case 4: NS::attn_core_q8<4>(W, slot, n, b0); break;                                                 \
                case 5: NS::attn_core_q8<5>(W, slot, n, b0); break;                                                 \
                default: NS::attn_core_w48q<8>(W, slot, n, -1, b0); break;                                          \
                }                                                                                                   \
            } else {                                                                                                \
                switch (T) {                                                                                        \
                case 1: NS::attn_core_q8m<1>(W, slot, n, b0); break;                                                \
                case 2: NS::attn_core_q8m<2>(W, slot, n, b0); break;                                                \
                case 3: NS::attn_core_q8m<3>(W, slot, n, b0); break;                                                \
                case 4: NS::attn_core_q8m<4>(W, slot, n, b0); break;                                                \
                case 5: NS::attn_core_q8m<5>(W, slot, n, b0); break;                                                \
                default: NS::attn_core_w48m<8>(W, slot, n, b0); break;                                              \
                }                                                                                                   \
            }                                                                                                       \
            __syncthreads();                                                                                        \
        }                                                                                                           \
    }
KVB_KERNEL(k_old, kold)
KVB_KERNEL(k_new, knew)
} // namespace

namespace {
constexpr int NB = 82; // the engine's blocks: 82 SMs

int g_ctx;
Bufs hb;
LayerW hL;

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

template <class X>
X * dalloc(size_t n) {
    X * p;
    CK(cudaMalloc(&p, n * sizeof(X)));
    return p;
}

void fill_bytes(void * d, size_t bytes, uint64_t seed, std::vector<uint8_t> & tmp) {
    Rng r{seed * 0x9E3779B97F4A7C15ull + 88172645463325252ull};
    tmp.resize(bytes);
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        const uint64_t x = r.next();
        memcpy(&tmp[i], &x, 8);
    }
    CK(cudaMemcpy(d, tmp.data(), bytes, cudaMemcpyHostToDevice));
}

// fp16 scales: magnitudes around 0.1, both signs (q4_0-style scales are negative at times), a few zeros
void fill_scales(__half * d, size_t n, uint64_t seed) {
    Rng r{seed * 0xD1342543DE82EF95ull + 1234567ull};
    std::vector<uint16_t> t(n);
    for (size_t i = 0; i < n; i++) {
        const uint64_t x = r.next();
        const uint16_t e = 9 + (x >> 8) % 8;
        const uint16_t m = (uint16_t) ((x >> 16) & 0x3ff), s = (x >> 32) & 1;
        t[i] = (x & 255) == 0 ? 0 : (uint16_t) ((s << 15) | (e << 10) | m);
    }
    CK(cudaMemcpy(d, t.data(), n * 2, cudaMemcpyHostToDevice));
}

void fill_floats(float * d, size_t n, uint64_t seed) {
    Rng r{seed * 0xA0761D6478BD642Full + 777ull};
    std::vector<float> t(n);
    for (size_t i = 0; i < n; i++) {
        const uint64_t x = r.next();
        const double u = ((x & 0xffffff) + 0.5) / (double) (1 << 24), v = (((x >> 24) & 0xffffff) + 0.5) / (double) (1 << 24);
        t[i] = (float) (sqrt(-2 * log(u)) * cos(6.283185307 * v));
    }
    CK(cudaMemcpy(d, t.data(), n * 4, cudaMemcpyHostToDevice));
}

struct Fmt {
    int kb, vb, kr, win;
    const char * name;
};

// the KV buffers are big enough for q8 at this context and hold the same random bytes for every format
int8_t * g_kq, * g_vq, * g_krq, * g_vrq;
__half * g_ks, * g_vs, * g_krs, * g_vrs;

void set_fmt(const Fmt & f) {
    for (int s = 0; s < N_ATT_SLOTS; s++) {
        hb.kvp[0][s] = g_kq;
        hb.kvp[1][s] = g_vq;
        hb.kvs[0][s] = g_ks;
        hb.kvs[1][s] = g_vs;
        hb.kvb[0][s] = f.kb;
        hb.kvb[1][s] = f.vb;
        hb.kvr[s] = f.kr;
        hb.kvrv[s] = 0;
        hb.kvrp[0][s] = f.win && f.kb != 8 ? g_krq : nullptr;
        hb.kvrp[1][s] = f.win && f.vb != 8 ? g_vrq : nullptr;
        hb.kvrs[0][s] = f.win && f.kb != 8 ? g_krs : nullptr;
        hb.kvrs[1][s] = f.win && f.vb != 8 ? g_vrs : nullptr;
    }
    hb.kvwin = f.win;
    const int wb = (f.win + ATT_BUCKET - 1) / ATT_BUCKET;
    hb.kvrc = f.win ? (wb + 1) * ATT_BUCKET + ATP : 0;
    CK(cudaMemcpyToSymbol(cBs, &hb, sizeof hb));
}

void launch(bool nw, bool q8, int T, int slot, int n, int b0, int reps) {
    int a0 = q8, a1 = T, a2 = slot, a3 = n, a4 = b0, a5 = reps;
    void * args[] = {&a0, &a1, &a2, &a3, &a4, &a5};
    if (nw) CK(cudaLaunchCooperativeKernel((void *) k_new, dim3(NB), dim3(NT), args, SM_TOTAL, 0));
    else CK(cudaLaunchCooperativeKernel((void *) k_old, dim3(NB), dim3(NT), args, SM_TOTAL, 0));
    CK(cudaDeviceSynchronize());
}

constexpr size_t APART_ITEMS = NKV * (ATT_NCH + 1);

struct Out {
    std::vector<float> ap;
    std::vector<float2> ml;
    std::vector<int8_t> rows;
};

void fetch_rows(std::vector<int8_t> & o, const Fmt & f, int T, int n) {
    // the rows this call stored: main buffers, scales and rings
    o.clear();
    auto add = [&](const void * p, size_t b) {
        const size_t z = o.size();
        o.resize(z + b);
        CK(cudaMemcpy(o.data() + z, p, b, cudaMemcpyDeviceToHost));
    };
    for (int g = 0; g < NKV; g++) {
        const size_t r = (size_t) g * g_ctx + n;
        add(g_kq + r * (HD * f.kb / 8), (size_t) T * (HD * f.kb / 8));
        add(g_vq + r * (HD * f.vb / 8), (size_t) T * (HD * f.vb / 8));
        add(g_ks + r * (HD / 32), (size_t) T * (HD / 32) * 2);
        add(g_vs + r * (HD / 32), (size_t) T * (HD / 32) * 2);
        if (hb.kvrc)
            for (int t = 0; t < T; t++) {
                const size_t rr = (size_t) g * hb.kvrc + (n + t) % hb.kvrc;
                if (hb.kvrp[0][0]) add(g_krq + rr * HD, HD);
                if (hb.kvrp[1][0]) add(g_vrq + rr * HD, HD);
            }
    }
}

void run_one(bool nw, const Fmt & f, bool q8, int T, int slot, int n, int b0, Out & o) { // q8: the all-q8 kernel, else MIX
    CK(cudaMemset(hb.apart, 0xA5, APART_ITEMS * AR * 256 * 4));
    CK(cudaMemset(hb.aml, 0xA5, APART_ITEMS * AR * 8));
    launch(nw, q8, T, slot, n, b0, 1);
    o.ap.resize(APART_ITEMS * AR * 256);
    o.ml.resize(APART_ITEMS * AR);
    CK(cudaMemcpy(o.ap.data(), hb.apart, o.ap.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.ml.data(), hb.aml, o.ml.size() * 8, cudaMemcpyDeviceToHost));
    fetch_rows(o.rows, f, T, n);
}

int check_case(const Fmt & f, bool q8, int T, int slot, int n, int b0, bool q8n) { // q8 / q8n: old / new kernel's form
    Out a, b;
    run_one(false, f, q8, T, slot, n, b0, a);
    run_one(true, f, q8n, T, slot, n, b0, b);
    const bool ok_ap = !memcmp(a.ap.data(), b.ap.data(), a.ap.size() * 4), ok_ml = !memcmp(a.ml.data(), b.ml.data(), a.ml.size() * 8);
    const bool ok_rows = a.rows == b.rows;
    size_t nd = 0, wr = 0;
    for (size_t i = 0; i < a.ap.size(); i++) {
        uint32_t u;
        memcpy(&u, &a.ap[i], 4);
        wr += u != 0xA5A5A5A5u;
        nd += memcmp(&a.ap[i], &b.ap[i], 4) != 0;
    }
    printf("%-3s>%-3s K%d V%d r%d w%d T%d slot %2d n %6d b0 %2d: %s (%zu of %zu written partial floats differ, ml %s, rows %s)\n", q8 ? "q8" : "mix", q8n ? "q8" : "mix", f.kb, f.vb,
           f.kr, f.win, T, slot, n, b0, ok_ap && ok_ml && ok_rows ? "OK" : "DIFF", nd, wr, ok_ml ? "same" : "DIFF", ok_rows ? "same" : "DIFF");
    fflush(stdout);
    return !(ok_ap && ok_ml && ok_rows);
}

void setup(int ctx_positions) {
    g_ctx = ctx_positions + KV_PAD;
    memset(&hb, 0, sizeof hb);
    std::vector<uint8_t> tmp;
    const size_t rows = (size_t) NKV * g_ctx;
    g_kq = dalloc<int8_t>(rows * HD);
    g_vq = dalloc<int8_t>(rows * HD);
    g_ks = dalloc<__half>(rows * (HD / 32));
    g_vs = dalloc<__half>(rows * (HD / 32));
    fill_bytes(g_kq, rows * HD, 1, tmp);
    fill_bytes(g_vq, rows * HD, 2, tmp);
    fill_scales(g_ks, rows * (HD / 32), 3);
    fill_scales(g_vs, rows * (HD / 32), 4);
    const size_t rr = (size_t) NKV * (6 * ATT_BUCKET + ATP);
    g_krq = dalloc<int8_t>(rr * HD);
    g_vrq = dalloc<int8_t>(rr * HD);
    g_krs = dalloc<__half>(rr * (HD / 32));
    g_vrs = dalloc<__half>(rr * (HD / 32));
    fill_bytes(g_krq, rr * HD, 5, tmp);
    fill_bytes(g_vrq, rr * HD, 6, tmp);
    fill_scales(g_krs, rr * (HD / 32), 7);
    fill_scales(g_vrs, rr * (HD / 32), 8);
    hb.ctx = g_ctx;
    hb.kvq8 = 1;
    hb.dattn = 1;
    hb.qbuf = dalloc<float>((size_t) MAXD * QG + 64);
    hb.kbuf = dalloc<float>((size_t) MAXD * KVD + 64);
    hb.vbuf = dalloc<float>((size_t) MAXD * KVD + 64);
    fill_floats(hb.qbuf, (size_t) MAXD * QG, 9);
    fill_floats(hb.kbuf, (size_t) MAXD * KVD, 10);
    fill_floats(hb.vbuf, (size_t) MAXD * KVD, 11);
    hb.apart = dalloc<float>(APART_ITEMS * AR * 256);
    hb.aml = dalloc<float2>(APART_ITEMS * AR);
    // RoPE [ctx][32] (cos, sin): any unit pairs do
    const size_t nr = (size_t) g_ctx * 32;
    std::vector<float2> rp(nr);
    for (size_t i = 0; i < nr; i++) {
        const double a = (double) (i / 32) * pow(1e7, -2.0 * (double) (i % 32) / 64.0);
        rp[i] = make_float2((float) cos(a), (float) sin(a));
    }
    float2 * drp = dalloc<float2>(nr);
    CK(cudaMemcpy(drp, rp.data(), nr * 8, cudaMemcpyHostToDevice));
    hb.rope = drp;
    memset(&hL, 0, sizeof hL);
    float * qn = dalloc<float>(HD), * kn = dalloc<float>(HD);
    std::vector<float> w(HD);
    for (int i = 0; i < HD; i++) w[i] = 0.5f + 1.0f * (float) ((i * 2654435761u) >> 24) / 256.0f;
    CK(cudaMemcpy(qn, w.data(), HD * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < HD; i++) w[i] = 0.5f + 1.0f * (float) ((i * 40503u + 7) & 255) / 256.0f;
    CK(cudaMemcpy(kn, w.data(), HD * 4, cudaMemcpyHostToDevice));
    hL.qn = qn;
    hL.kn = kn;
    CK(cudaMemcpyToSymbol(cL, &hL, sizeof hL));
    int v0 = 0;
    CK(cudaMemcpyToSymbol(cView, &v0, sizeof v0));
    CK(cudaFuncSetAttribute(k_old, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
    CK(cudaFuncSetAttribute(k_new, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: kvattn_bench check [smoke] | time <F|D|q8> <n> <T> <rounds> <reps>\n");
        return 2;
    }
    const Fmt F{5, 5, 1, 2048, "F"}, D{6, 5, 0, 2048, "D"}, Q{8, 8, 0, 0, "q8"};
    if (!strcmp(argv[1], "time")) {
        const char * p = argv[2];
        const int n = atoi(argv[3]), T = atoi(argv[4]), rounds = atoi(argv[5]), reps = atoi(argv[6]);
        const int slot = T == 1 ? 16 : 0;
        setup(n + 16);
        const Fmt f = !strcmp(p, "F") ? F : !strcmp(p, "D") ? D : Q;
        set_fmt(f);
        const bool q8 = f.kb == 8 && f.vb == 8, q8n = q8 && !getenv("KVB_Q8M"); // KVB_Q8M: new = the MIX kernel on all-q8 data
        // warm up past the memory clock's ~2-3 s ramp (P3 until sustained load), then old / new interleaved
        const auto tw = std::chrono::steady_clock::now();
        const double warm_s = getenv("KVB_WARM") ? atof(getenv("KVB_WARM")) : 4.5; // (0 under ncu: every pass re-runs the program)
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count() < warm_s) {
            launch(false, q8, T, slot, n, 0, reps);
            launch(true, q8n, T, slot, n, 0, reps);
        }
        std::vector<double> t[2];
        for (int r = 0; r < rounds; r++)
            for (int v = 0; v < 2; v++) {
                const bool nw = (r & 1) ? v == 0 : v == 1;
                const auto t0 = std::chrono::steady_clock::now();
                launch(nw, nw ? q8n : q8, T, slot, n, 0, reps);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
                t[nw].push_back(ms);
            }
        for (int v = 0; v < 2; v++) std::sort(t[v].begin(), t[v].end());
        int nch, cs;
        { const int L = n + T; cs = ATP * std::max(1, (L + ATT_BUCKET - 1) / ATT_BUCKET); nch = std::max(1, (L + cs - 1) / cs); }
        const int tiles = cs / ATP;
        printf("%s n %d T %d: old min %.3f med %.3f ms | new min %.3f med %.3f ms | delta min %+.3f med %+.3f ms (%+.1f%%) | us per tile old %.3f new %.3f (%d tiles, %d chunks)\n", p, n, T,
               t[0][0], t[0][t[0].size() / 2], t[1][0], t[1][t[1].size() / 2], t[1][0] - t[0][0], t[1][t[1].size() / 2] - t[0][t[0].size() / 2],
               100 * (t[1][t[1].size() / 2] / t[0][t[0].size() / 2] - 1), t[0][0] * 1e3 / tiles, t[1][0] * 1e3 / tiles, tiles, nch);
        return 0;
    }
    if (!strcmp(argv[1], "stress")) { // stress <F|D|q8> <n> <T> <runs>: the new kernel over and over against one run of the old: races
        const char * p = argv[2];
        const int n = atoi(argv[3]), T = atoi(argv[4]), runs = atoi(argv[5]);
        const int slot = T == 1 ? 16 : 0;
        setup(n + 16);
        const Fmt f = !strcmp(p, "F") ? F : !strcmp(p, "D") ? D : Q;
        set_fmt(f);
        const bool q8 = f.kb == 8 && f.vb == 8;
        Out a, b;
        run_one(false, f, q8, T, slot, n, 0, a);
        long bad = 0;
        for (int r = 0; r < runs; r++) {
            if (getenv("KVB_STRESS_OLD")) run_one(false, f, q8, T, slot, n, 0, b);
            else run_one(true, f, false, T, slot, n, 0, b);
            size_t nd = 0, first = 0;
            for (size_t i = 0; i < a.ap.size(); i++)
                if (memcmp(&a.ap[i], &b.ap[i], 4)) {
                    if (!nd) first = i;
                    nd++;
                }
            const bool mlok = !memcmp(a.ml.data(), b.ml.data(), a.ml.size() * 8);
            if (nd || !mlok) {
                bad++;
                if (bad <= 40) {
                    long ndr = 0;
                    std::vector<int> dims;
                    const size_t irow = first / 256;
                    for (int d = 0; d < 256; d++)
                        if (memcmp(&a.ap[irow * 256 + d], &b.ap[irow * 256 + d], 4)) { ndr++; if (dims.size() < 8) dims.push_back(d); }
                    printf("run %d: %zu floats differ, ml %s; first: item %zu row %zu dim %zu (%d dims in that row: %ld;", r, nd, mlok ? "same" : "DIFF", first / 256 / AR, irow % AR, first % 256, (int) dims.size(), ndr);
                    for (int d : dims) printf(" %d", d);
                    float x = a.ap[first], y = b.ap[first];
                    printf(") old %g new %g\n", x, y);
                    fflush(stdout);
                }
            }
        }
        printf("STRESS %s n %d T %d: %ld of %d runs differ\n", p, n, T, bad, runs);
        return bad ? 1 : 0;
    }
    const bool smoke = argc > 2 && !strcmp(argv[2], "smoke");
    setup(smoke ? 42000 : 258100);
    const std::vector<Fmt> fmts = smoke ? std::vector<Fmt>{F, D} : std::vector<Fmt>{F, D, {4, 4, 0, 2048, "44"}, {7, 6, 0, 0, "76"}, {8, 5, 0, 2048, "85"}, {5, 8, 1, 2048, "58"}, {6, 6, 1, 0, "66"}, {4, 7, 0, 2048, "47"}};
    const std::vector<int> ns = smoke ? std::vector<int>{700, 5000, 40013} : std::vector<int>{66, 650, 1300, 2049, 5000, 10007, 40013, 101600, 200031, 257700};
    const std::vector<int> Ts = {1, 2, 3, 4, 5, 8};
    long bad = 0;
    for (const Fmt & f : fmts) {
        set_fmt(f);
        for (int n : ns)
            for (int T : Ts) {
                if (T <= 5 && n + T <= ATT_BUCKET) continue; // short chunks: attn_core_t, not these kernels
                const int b0 = n % 7 == 0 ? 3 : 0;
                bad += check_case(f, false, T, T == 1 ? 16 : 0, n, b0, false);
            }
    }
    set_fmt(Q);
    for (int n : ns)
        for (int T : Ts) {
            if (T <= 5 && n + T <= ATT_BUCKET) continue;
            bad += check_case(Q, true, T, T == 1 ? 16 : 0, n, 0, true);
            bad += check_case(Q, true, T, T == 1 ? 16 : 0, n, 0, false); // the MIX kernel on all-q8 slots
        }
    printf(bad ? "CHECK FAILED: %ld cases differ\n" : "CHECK OK (%ld differ)\n", bad);
    return bad ? 1 : 0;
}

// mmq_bench: correctness + throughput of the prefill fp16 tensor-core GEMM (prefill_gemm.cuh) on random data in the engine's
// on-device layouts (loader.cu's repacks are not run: the bytes are random in the layouts they produce; wv_relayout is the engine's).
//   make bench/mmq_bench          every kernel listed below (~20 s); MMQFLAGS=-DMMQ_SET=1: plain Q5_K only
//   bench/mmq_bench check [ntok]  each kernel against a scalar fp64 reference on the pre-wv rows, and each candidate (MMQ_CAND)
//                                 bit for bit against its old kernel on normal and on wide-range data (exit status 1 on a failure)
//   bench/mmq_bench time <kernel-regex> <K> <N> [ntok] [reps] [nlay]
//                                 launch times (nlay weight copies, cycled: L2 cannot hold them, as in the engine), old and candidate
//                                 interleaved launch by launch; warmed up >= 4 s first (the memory clock drops at idle). Cycles, not seconds:
//                                 a prefill runs at the power cap, and interleaved launches share one clock
//   bench/mmq_bench sustain <kernel-regex> <K> <N> [ntok] [secs] [rounds]
//                                 old and candidate in turns of secs seconds: launches / s with the power cap in the loop, the number that counts
// A plain q5k `time` run under ncu matches a served mmh<13, 0> launch of the same shape (K 5120 N 6144, 1024 tokens: Compute Throughput 74.26 vs 74.15%,
// 620 vs 623 us, the same stall mix and instruction counts, L2 hit rate 89.48 in both).
#define MK_SHARED_NS // (anonymous, as in qwen38_mk.cu)
#include "../mk_shared.cuh"
#include "../mk_launch.h" // MK_GLU_MIX_PAIRS

#include <regex>

namespace {
// wv_host.cuh's host loop (the engine's is loader.cuh's pool)
template <typename F> void parallel_rows(int64_t n, F f) {
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (unsigned i = 0, nt = std::max(1u, std::thread::hardware_concurrency()); i < nt; i++)
        th.emplace_back([&] {
            for (int64_t r; (r = next.fetch_add(1)) < n;) f(r);
        });
    for (auto & t : th) t.join();
}
#include "../wv_host.cuh"

// what prefill_gemm.cuh takes from qwen38_mk.cu
__device__ __forceinline__ void iq4_lut(unsigned q, unsigned t0, unsigned t1, unsigned t2, unsigned t3, int & lo, int & hi) {
    const unsigned sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const unsigned a = __byte_perm(__byte_perm(t0, t1, q), __byte_perm(t2, t3, q), sel);
    const unsigned b = __byte_perm(__byte_perm(t0, t1, q >> 16), __byte_perm(t2, t3, q >> 16), sel >> 16);
    lo = (int) __byte_perm(a, b, 0x6420);
    hi = (int) __byte_perm(a, b, 0x7531);
}
__device__ uint32_t g_iq3s_grid[512];
__device__ __forceinline__ unsigned iq3s_grid(unsigned idx) {
    unsigned r;
    asm("ld.global.nc.L1::evict_last.u32 %0, [%1];" : "=r"(r) : "l"(g_iq3s_grid + idx));
    return r;
}
#include "../prefill_gemm.cuh"
} // namespace
using namespace pf;

#ifndef MMQ_SET
#define MMQ_SET 0
#endif

// ---- the kernels ---------------------------------------------------------------------------------------------------
struct KR {
    const char * name;
    int type, lay, type2, lay2;
    bool glu;
    int nb;
    void (*attr)();
    void (*run)(const WMat &, const WMat &, const __half *, int, const Out &);
    const char * oldn; // a candidate: the name of the kernel it must match bit for bit
};
template <int TYPE, int NB, int TY2> constexpr int ksm() {
    if constexpr (TY2 >= 0) return hm_smem2<TYPE, TY2, NB>();
    else return hm_smem<TYPE, NB>();
}
template <int TYPE, bool GLU, int NB, int LAY, int TY2, int LY2> void k_attr() {
    CK(cudaFuncSetAttribute(mmh<TYPE, GLU, NB, LAY, TY2, LY2>, cudaFuncAttributeMaxDynamicSharedMemorySize, ksm<TYPE, NB, TY2>()));
}
template <int TYPE, bool GLU, int NB, int LAY, int TY2, int LY2>
void k_run(const WMat & W, const WMat & Wu, const __half * X, int ntok, const Out & O) {
    const int nrt = W.N / (GLU ? BM / 2 : BM);
    mmh<TYPE, GLU, NB, LAY, TY2, LY2><<<dim3(nrt, (ntok + NB - 1) / NB), 256, ksm<TYPE, NB, TY2>()>>>(
        W, X, ntok, O, GLU && TY2 < 0 ? Wu.w : nullptr, TY2 >= 0 ? Wu : WMat{});
}
#define MMQ_K(nm, TYPE, GLU, NB, LAY, TY2, LY2, oldn)                                                                     \
    {nm, TYPE, LAY, TY2, LY2, GLU, NB, k_attr<TYPE, GLU, NB, LAY, TY2, LY2>, k_run<TYPE, GLU, NB, LAY, TY2, LY2>, oldn},
#define MMQ_PLAIN(nm, TYPE, NB, LAY) MMQ_K(nm, TYPE, false, NB, LAY, -1, LAY, nullptr)
#define MMQ_FUSED(nm, TYPE, LAY) MMQ_K(nm, TYPE, true, BN, LAY, -1, LAY, nullptr)
#define MMQ_MIX(TG, LG, TU, LU) MMQ_K(#TG "_" #LG "__" #TU "_" #LU, TG, true, BN, LG, TU, LU, nullptr)
static const KR g_k[] = {
    MMQ_PLAIN("q5k", TQ5K, BN, 1)
#if MMQ_SET == 0
    MMQ_PLAIN("q4k", TQ4K, BN, 1) MMQ_PLAIN("iq4x", TIQ4, BN, 1) MMQ_PLAIN("iq4nl", TIQ4NL, BN, 1) MMQ_PLAIN("q6k", TQ6K, BN, 1)
    MMQ_PLAIN("q80", TQ80, BN, 1) MMQ_PLAIN("q3k", TQ3K, BN, 1) MMQ_PLAIN("iq3s", TIQ3S, BN, 1)
    MMQ_PLAIN("q5k_l0", TQ5K, BN, 0) MMQ_PLAIN("q4k_l0", TQ4K, BN, 0) MMQ_PLAIN("iq4x_l0", TIQ4, BN, 0) MMQ_PLAIN("iq4nl_l0", TIQ4NL, BN, 0)
    MMQ_PLAIN("q6k_l0", TQ6K, BN, 0) MMQ_PLAIN("q80_l0", TQ80, BN, 0) MMQ_PLAIN("q3k_l0", TQ3K, BN, 0) MMQ_PLAIN("iq3s_l0", TIQ3S, BN, 0)
    MMQ_K("q5k_nb64", TQ5K, false, 64, 1, -1, 1, "q5k") MMQ_K("q5k_nb32", TQ5K, false, 32, 1, -1, 1, "q5k") // (any NB: the same bits)
    MMQ_FUSED("glu_q5k", TQ5K, 1) MMQ_FUSED("glu_q4k", TQ4K, 1) MMQ_FUSED("glu_iq4x", TIQ4, 1) MMQ_FUSED("glu_q5k_l0", TQ5K, 0)
    MK_GLU_MIX_PAIRS(MMQ_MIX)
#endif
#ifdef MMQ_CAND // candidates: -DMMQ_CAND='"file"', lines like MMQ_K("q5k_nb64", TQ5K, false, 64, 1, -1, 1, "q5k") (the last name: the kernel it must match)
#include MMQ_CAND
#endif
};

// ---- random weights ------------------------------------------------------------------------------------------------
struct XS {
    uint64_t s;
    uint64_t operator()() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    double u() { return (double) ((*this)() >> 11) * (1.0 / 9007199254740992.0); }
};
static void fill(uint8_t * p, size_t n, XS & r) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint64_t v = r();
        memcpy(p + i, &v, 8);
    }
    for (; i < n; i++) p[i] = (uint8_t) r();
}
static uint16_t fh(float f) {
    __half x = __float2half(f);
    uint16_t u;
    memcpy(&u, &x, 2);
    return u;
}
static float hf(const uint8_t * p) {
    __half x;
    memcpy(&x, p, 2);
    return __half2float(x);
}
static uint32_t g_grid[512];

// One matrix: w / meta in the loader's pre-wv layout, dw / dd on the device in the kernel's layout.
struct Mx {
    int type = 0, K = 0, N = 0, lay = 0, rb = 0, nd = 0;
    std::vector<uint8_t> w, meta;
    uint8_t * dw = nullptr, * dd = nullptr;
    WMat W{};
    void free_dev() {
        if (dw) CK(cudaFree(dw));
        if (dd) CK(cudaFree(dd));
        dw = dd = nullptr;
    }
};
static Mx gen(int type, int lay, int K, int N, XS & r, bool wide) {
    Mx m;
    m.type = type;
    m.K = K;
    m.N = N;
    m.lay = lay;
    const int nb = K / 256;
    int mrow = 0; // meta bytes per row
    switch (type) {
        case TQ4K: case TIQ4NL: m.rb = nb * 144; break;
        case TQ5K: m.rb = nb * 176; break;
        case TIQ4: m.rb = nb * 136; break;
        case TQ6K: m.rb = nb * Q6_BLK; mrow = nb * Q6_META; m.nd = mrow / 2; break;
        case TQ3K: m.rb = nb * Q3_BLK; mrow = nb * Q6_META; m.nd = mrow / 2; break;
        case TIQ3S: m.rb = nb * Q3_BLK; mrow = nb * IQ3S_META; m.nd = mrow / 2; break;
        default: m.rb = K; mrow = K / 32 * 2; m.nd = K / 32; break; // Q8_0
    }
    m.w.resize((size_t) N * m.rb);
    m.meta.resize((size_t) N * mrow);
    fill(m.w.data(), m.w.size(), r);
    fill(m.meta.data(), m.meta.size(), r);
    auto d = [&]() {
        const double l = wide ? -15.0 + 17.0 * r.u() : -11.0 + 3.0 * r.u(); // 2^l
        return fh((float) ((r() & 1) && wide ? -1.0 : 1.0) * (float) pow(2.0, l));
    };
    auto put = [&](uint8_t * p) {
        const uint16_t v = d();
        memcpy(p, &v, 2);
    };
    for (int row = 0; row < N; row++) {
        uint8_t * rw = m.w.data() + (size_t) row * m.rb;
        uint8_t * mt = m.meta.data() + (size_t) row * mrow;
        if (type == TQ4K || type == TQ5K)
            for (int b = 0; b < nb; b++) {
                put(rw + (size_t) b * m.rb / nb);
                put(rw + (size_t) b * m.rb / nb + 2);
            }
        else if (type == TIQ4)
            for (int b = 0; b < nb; b += 2) {
                put(rw + (size_t) (b / 2) * 272);
                put(rw + (size_t) (b / 2) * 272 + 8);
            }
        else if (type == TIQ4NL)
            for (int b = 0; b < nb; b++)
                for (int s = 0; s < 8; s++) put(rw + (size_t) b * 144 + 2 * s);
        else if (type == TQ6K || type == TQ3K)
            for (int b = 0; b < nb; b += 2) {
                put(mt + (size_t) (b / 2) * 36 + 32);
                put(mt + (size_t) (b / 2) * 36 + 34);
            }
        else if (type == TIQ3S)
            for (int b = 0; b < nb; b++) put(mt + (size_t) b * IQ3S_META + 12);
        else if (type == TQ80)
            for (int e = 0; e < K / 32; e++) put(mt + 2 * e);
    }
    std::vector<uint8_t> o, om;
    const uint8_t * dwp = m.w.data(), * dmp = m.meta.data();
    size_t dwn = m.w.size(), dmn = m.meta.size();
    if (lay) {
        o.resize(type == TQ6K ? m.w.size() + m.meta.size() : m.w.size());
        om.resize(type == TQ6K ? 0 : m.meta.size());
        dmn = wv_relayout(type, N, K, m.rb, m.w.data(), m.meta.data(), m.meta.size(), o.data(), om.data());
        dwp = o.data();
        dwn = o.size();
        dmp = om.data();
    }
    CK(cudaMalloc(&m.dw, dwn));
    CK(cudaMemcpy(m.dw, dwp, dwn, cudaMemcpyHostToDevice));
    if (dmn) {
        CK(cudaMalloc(&m.dd, dmn));
        CK(cudaMemcpy(m.dd, dmp, dmn, cudaMemcpyHostToDevice));
    }
    m.W = WMat{m.dw, reinterpret_cast<const __half *>(m.dd), type, K, N, m.rb, m.nd};
    return m;
}

static double kv_iq4(unsigned q) { // kvalues_iq4nl (the bytes of IQ4U_T* minus 128)
    const unsigned t[4] = {IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3};
    return (double) ((t[q >> 2] >> (8 * (q & 3))) & 0xff) - 128.0;
}
// Scalar value of element e of a row, from the pre-wv bytes: d * sc * (q - 32) etc. as ggml dequantizes.
static double deq(const Mx & m, int row, int e) {
    const uint8_t * rw = m.w.data() + (size_t) row * m.rb;
    const int b = e >> 8, w = e & 255, sb = w >> 5, l = w & 31;
    const int p = w >> 6, ww = w & 63, sub = ww >> 5, i = ww & 31, hh = i >> 4, ii = i & 15, tq = ii >> 2, mm = ii & 3;
    const uint8_t * mp = m.meta.empty() ? nullptr : m.meta.data() + (size_t) row * (m.meta.size() / m.N);
    if (m.type == TQ4K || m.type == TQ5K) {
        const uint8_t * blk = rw + (size_t) b * (m.type == TQ4K ? 144 : 176), * sc = blk + 4;
        const double d = hf(blk), dmin = hf(blk + 2);
        int s, mn;
        if (sb < 4) {
            s = sc[sb] & 63;
            mn = sc[sb + 4] & 63;
        } else {
            s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
            mn = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
        }
        const int qs = m.type == TQ4K ? 16 : 48, qb = blk[qs + 32 * (sb >> 1) + l];
        int q = (sb & 1) ? qb >> 4 : qb & 0xF;
        if (m.type == TQ5K) q |= ((blk[16 + l] >> sb) & 1) << 4;
        return d * s * q - dmin * mn;
    }
    if (m.type == TIQ4) {
        const uint8_t * pr = rw + (size_t) (b >> 1) * 272, * hd = pr + 8 * (b & 1);
        const unsigned shi = hd[2] | (hd[3] << 8), slo = hd[4 + (sb >> 1)];
        const int sl = (slo >> (4 * (sb & 1))) & 0xF, sh = (shi >> (4 * (sb >> 1) + 2 * (sb & 1))) & 3;
        const int qb = pr[16 + 128 * (b & 1) + 32 * (sb >> 1) + l];
        return (double) hf(hd) * ((sl | (sh << 4)) - 32) * kv_iq4((sb & 1) ? qb >> 4 : qb & 0xF);
    }
    if (m.type == TIQ4NL) {
        const uint8_t * blk = rw + (size_t) b * 144;
        const int qb = blk[16 + 32 * (sb >> 1) + l];
        return (double) hf(blk + 2 * sb) * kv_iq4((sb & 1) ? qb >> 4 : qb & 0xF);
    }
    if (m.type == TQ80) return (double) hf(mp + 2 * (e >> 5)) * (double) (int8_t) rw[32 * (8 * b + 2 * p + hh) + 16 * sub + ii];
    if (m.type == TQ6K || m.type == TQ3K) {
        const uint8_t * pr = mp + (size_t) (b >> 1) * 36;
        const double s = (double) hf(pr + 32 + 2 * (b & 1)) * (double) (int8_t) pr[16 * (b & 1) + (w >> 4)];
        const uint8_t * blk = rw + (size_t) b * (m.type == TQ6K ? Q6_BLK : Q3_BLK);
        if (m.type == TQ6K) {
            const int lo = sub ? blk[32 * p + i] >> 4 : blk[32 * p + i] & 0xF;
            unsigned hw;
            memcpy(&hw, blk + 128 + 8 * (2 * p + hh) + 4 * sub, 4);
            return s * ((lo | (int) (((hw >> (8 * (ii % 4) + 2 * (ii / 4))) & 3) << 4)) - 32);
        }
        const int u3 = ((blk[16 * p + 8 * hh + 4 * sub + mm] >> (2 * tq)) & 3) | (((blk[64 + 8 * p + 4 * hh + mm] >> (tq + 4 * sub)) & 1) << 2);
        return s * (u3 - 4);
    }
    // IQ3_S
    const uint8_t * blk = rw + (size_t) b * Q3_BLK, * mb = mp + (size_t) b * IQ3S_META;
    const int qs = blk[16 * p + 8 * hh + 4 * sub + tq], qh = (mb[2 * p + hh] >> (4 * sub + tq)) & 1;
    const int gv = (g_grid[qs | (qh << 8)] >> (8 * mm)) & 0xff, sg = (blk[64 + 8 * p + 4 * hh + mm] >> (4 * sub + tq)) & 1;
    return (double) hf(mb + 12) * (1 + 2 * ((mb[8 + p] >> (4 * sub)) & 0xF)) * (sg ? -gv : gv);
}

static std::vector<__half> gen_x(int ntok, int K, XS & r, bool wide) {
    std::vector<__half> x((size_t) ntok * K);
    for (auto & v : x) {
        if (wide) { // the fp16 inputs' whole range, denormals included (the engine clamps them to +-65504)
            const double l = -25.0 + 41.0 * r.u();
            v = __float2half((float) ((r() & 1) ? -1.0 : 1.0) * (float) std::min(pow(2.0, l), 65504.0));
        } else { // about N(0, 1)
            const double s = r.u() + r.u() + r.u() + r.u() + r.u() + r.u() - 3.0;
            v = __float2half((float) (s * 1.41));
        }
    }
    return x;
}

static const KR * find_k(const char * n) {
    for (const KR & k : g_k)
        if (!strcmp(k.name, n)) return &k;
    return nullptr;
}

// ---- check ---------------------------------------------------------------------------------------------------------
static int check_one(const KR & k, int ntok) {
    const KR * old = k.oldn ? find_k(k.oldn) : nullptr;
    int fail = 0;
    for (int wide = 0; wide < 2; wide++) {
        if (wide && !old) break; // (wide-range data: old vs new only, the reference's rounding differs too much)
        const int K = 1024, N = 1024; // nb even, N a multiple of every BX
        XS r{0x9E3779B97F4A7C15ull ^ ((uint64_t) k.type << 8) ^ (uint64_t) wide};
        Mx a = gen(k.type, k.lay, K, N, r, wide);
        Mx u;
        if (k.glu) u = gen(k.type2 >= 0 ? k.type2 : k.type, k.type2 >= 0 ? k.lay2 : k.lay, K, N, r, wide);
        auto xh = gen_x(ntok, K, r, wide);
        __half * dx;
        CK(cudaMalloc(&dx, xh.size() * 2));
        CK(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
        const int ldy = N + 64;
        const size_t esz = k.glu ? 2 : 4, ysz = (size_t) ntok * ldy * esz;
        float * dy, * dy2;
        CK(cudaMalloc(&dy, ysz));
        CK(cudaMalloc(&dy2, ysz));
        // the padding of y stays at this pattern, and an old-kernel run of OUT_ADD starts from it too
        std::vector<uint8_t> init(ysz), y1(ysz), y2(ysz);
        fill(init.data(), init.size(), r);
        if (!k.glu) { // finite fp32 values, so OUT_ADD does something
            float * f = reinterpret_cast<float *>(init.data());
            for (size_t i = 0; i < ysz / 4; i++) f[i] = (float) (r.u() - 0.5);
        }
        auto run1 = [&](const KR & kk, float * y, int mode) {
            kk.attr();
            CK(cudaMemcpy(y, init.data(), ysz, cudaMemcpyHostToDevice));
            const Out O = k.glu ? Out{nullptr, ldy, OUT_STORE, reinterpret_cast<__half *>(y)} : Out{y, ldy, mode, nullptr};
            kk.run(a.W, u.W, dx, ntok, O);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
        };
        double num = 0, den = 0;
        if (!wide) { // scalar reference, a sample of rows and tokens
            run1(k, dy, OUT_STORE);
            CK(cudaMemcpy(y1.data(), dy, ysz, cudaMemcpyDeviceToHost));
            const int nts = std::min(ntok, 24);
            parallel_rows(N / 8, [&](int64_t rr) {
                const int row = (int) rr * 8 + (int) (rr % 8);
                double n1 = 0, d1 = 0;
                for (int t0 = 0; t0 < nts; t0++) {
                    const int t = t0 * ntok / nts;
                    const __half * xp = xh.data() + (size_t) t * K;
                    double g = 0, up = 0;
                    for (int e = 0; e < K; e++) {
                        const double x = __half2float(xp[hperm(e)]);
                        g += deq(a, row, e) * x;
                        if (k.glu) up += deq(u, row, e) * x;
                    }
                    double ref, got;
                    if (k.glu) {
                        ref = g / (1.0 + exp(-g)) * up;
                        got = __half2float(reinterpret_cast<const __half *>(y1.data())[(size_t) t * ldy + hperm(row)]);
                    } else {
                        ref = g;
                        got = reinterpret_cast<const float *>(y1.data())[(size_t) t * ldy + row];
                    }
                    n1 += (got - ref) * (got - ref);
                    d1 += ref * ref;
                }
                static std::mutex mu;
                std::lock_guard<std::mutex> l(mu);
                num += n1;
                den += d1;
            });
            const double err = sqrt(num / den);
            printf("%-34s rel err vs the scalar reference %.2e%s\n", k.name, err, err > 3e-3 ? "  FAIL" : "");
            if (err > 3e-3) fail = 1;
        }
        if (old) {
            for (int mode = 0; mode < (k.glu ? 1 : 2); mode++) {
                run1(*old, dy, mode);
                run1(k, dy2, mode);
                CK(cudaMemcpy(y1.data(), dy, ysz, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(y2.data(), dy2, ysz, cudaMemcpyDeviceToHost));
                const bool same = !memcmp(y1.data(), y2.data(), ysz);
                size_t firstd = 0, nd = 0;
                if (!same)
                    for (size_t i = 0; i < ysz; i += esz)
                        if (memcmp(y1.data() + i, y2.data() + i, esz) && !nd++) firstd = i / esz;
                printf("%-34s vs %-12s %s %s: %s", k.name, k.oldn, wide ? "wide  " : "normal", mode ? "add  " : "store", same ? "bit-identical\n" : "DIFFERENT");
                if (!same) {
                    printf(" %zu elements, the first at token %zu row %zu\n", nd, firstd / ldy, firstd % ldy);
                    fail = 1;
                }
            }
        }
        CK(cudaFree(dx));
        CK(cudaFree(dy));
        CK(cudaFree(dy2));
        a.free_dev();
        u.free_dev();
    }
    return fail;
}

// ---- time ----------------------------------------------------------------------------------------------------------
static void time_k(const KR & k, int K, int N, int ntok, int reps, int nlay) {
    const KR * old = k.oldn ? find_k(k.oldn) : nullptr;
    XS r{0x123456789ABCDEFull};
    std::vector<Mx> ms, us;
    for (int l = 0; l < nlay; l++) {
        ms.push_back(gen(k.type, k.lay, K, N, r, false));
        if (k.glu) us.push_back(gen(k.type2 >= 0 ? k.type2 : k.type, k.type2 >= 0 ? k.lay2 : k.lay, K, N, r, false));
    }
    auto xh = gen_x(ntok, K, r, false);
    __half * dx;
    CK(cudaMalloc(&dx, xh.size() * 2));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    float * dy;
    const size_t ysz = (size_t) ntok * N * 4;
    CK(cudaMalloc(&dy, ysz));
    const Out O = k.glu ? Out{nullptr, N, OUT_STORE, reinterpret_cast<__half *>(dy)} : Out{dy, N, OUT_STORE, nullptr};
    k.attr();
    if (old) old->attr();
    static const WMat W0{};
    auto launch = [&](const KR & kk, int l) {
        kk.run(ms[l].W, k.glu ? us[l].W : W0, dx, ntok, O);
    };
    cudaEvent_t ea, eb;
    CK(cudaEventCreate(&ea));
    CK(cudaEventCreate(&eb));
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 4.0; i++) { // memory clock up
        launch(k, i % nlay);
        CK(cudaDeviceSynchronize());
    }
    auto once = [&](const KR & kk, int l) {
        CK(cudaEventRecord(ea));
        launch(kk, l);
        CK(cudaEventRecord(eb));
        CK(cudaEventSynchronize(eb));
        float ms_;
        CK(cudaEventElapsedTime(&ms_, ea, eb));
        return (double) ms_;
    };
    std::vector<double> tn, to;
    for (int i = 0; i < reps; i++)
        for (int l = 0; l < nlay; l++) {
            if (old) to.push_back(once(*old, l));
            tn.push_back(once(k, l));
        }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double fl = 2.0 * K * (double) N * ntok * (k.glu ? 2.0 : 1.0), mn = med(tn);
    // (GLU: N gate rows and N up rows)
    printf("%-22s K %5d N %5d tok %4d grid %3d x %2d nb %3d: median %8.3f ms min %8.3f  %6.1f TFLOPS  %5.1f%% of the 1.665 GHz fp16 peak", k.name, K, N, ntok,
           N / (k.glu ? BM / 2 : BM), (ntok + k.nb - 1) / k.nb, k.nb, mn, *std::min_element(tn.begin(), tn.end()), fl / (mn * 1e-3) / 1e12,
           100.0 * fl / (mn * 1e-3) / (328 * 1.665e9 * 256));
    if (old) {
        const double mo = med(to);
        printf(" | old %s %8.3f ms min %8.3f: %+.2f%%", k.oldn, mo, *std::min_element(to.begin(), to.end()), 100.0 * (mn / mo - 1.0));
    }
    printf("\n");
    fflush(stdout);
    CK(cudaEventDestroy(ea));
    CK(cudaEventDestroy(eb));
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    for (auto & m : ms) m.free_dev();
    for (auto & m : us) m.free_dev();
}

// ---- sustain -------------------------------------------------------------------------------------------------------
// Old and candidate in turns of [secs] s each, back to back: launches / s with the power cap in the loop (interleaved launches share one clock, so `time`
// cannot show that a kernel with a higher tensor duty draws more power and loses clock). nvidia-smi samples per segment.
static void sustain_k(const KR & k, int K, int N, int ntok, double secs, int nlay, int rounds) {
    const KR * old = k.oldn ? find_k(k.oldn) : nullptr;
    XS r{0x123456789ABCDEFull};
    std::vector<Mx> ms, us;
    for (int l = 0; l < nlay; l++) {
        ms.push_back(gen(k.type, k.lay, K, N, r, false));
        if (k.glu) us.push_back(gen(k.type2 >= 0 ? k.type2 : k.type, k.type2 >= 0 ? k.lay2 : k.lay, K, N, r, false));
    }
    auto xh = gen_x(ntok, K, r, false);
    __half * dx;
    CK(cudaMalloc(&dx, xh.size() * 2));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    float * dy;
    CK(cudaMalloc(&dy, (size_t) ntok * N * 4));
    const Out O = k.glu ? Out{nullptr, N, OUT_STORE, reinterpret_cast<__half *>(dy)} : Out{dy, N, OUT_STORE, nullptr};
    k.attr();
    if (old) old->attr();
    static const WMat W0{};
    std::atomic<bool> stop{false};
    std::mutex mu;
    double clk = 0, pw = 0;
    int nclk = 0;
    std::thread smi([&] {
        while (!stop) {
            FILE * f = popen("nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null", "r");
            double c = 0, p = 0;
            if (f) {
                if (fscanf(f, "%lf, %lf", &c, &p) == 2) {
                    std::lock_guard<std::mutex> l(mu);
                    clk += c;
                    pw += p;
                    nclk++;
                }
                pclose(f);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    });
    double tot[2] = {0, 0}, fl = 2.0 * K * (double) N * ntok * (k.glu ? 2.0 : 1.0);
    long cnt[2] = {0, 0};
    auto seg = [&](const KR & kk, int w) {
        // the first third of a segment: the power loop settles
        for (int phase = 0; phase < 2; phase++) {
            const auto ts = std::chrono::steady_clock::now();
            long n = 0;
            const double dur = phase ? secs : secs / 3;
            {
                std::lock_guard<std::mutex> l(mu);
                clk = pw = 0;
                nclk = 0;
            }
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count() < dur) {
                for (int i = 0; i < 8; i++, n++) kk.run(ms[n % nlay].W, k.glu ? us[n % nlay].W : W0, dx, ntok, O);
                CK(cudaDeviceSynchronize());
            }
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            if (phase) {
                std::lock_guard<std::mutex> l(mu);
                tot[w] += el;
                cnt[w] += n;
                printf("  %-14s %6.1f launches/s  %6.1f TFLOPS  %5.0f MHz %5.0f W\n", kk.name, n / el, n * fl / el / 1e12, nclk ? clk / nclk : 0.0, nclk ? pw / nclk : 0.0);
                fflush(stdout);
            }
        }
    };
    for (int rr = 0; rr < rounds; rr++) {
        if (old) seg(*old, 0);
        seg(k, 1);
    }
    stop = true;
    smi.join();
    if (old) printf("%-22s K %5d N %5d tok %4d: sustained %+.2f%% time vs %s\n", k.name, K, N, ntok, 100.0 * ((tot[1] / cnt[1]) / (tot[0] / cnt[0]) - 1.0), k.oldn);
    fflush(stdout);
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    for (auto & m : ms) m.free_dev();
    for (auto & m : us) m.free_dev();
}

int main(int argc, char ** argv) {
    XS gr{42};
    for (auto & g : g_grid) { // four odd magnitudes, as ggml's iq3s_grid
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) v |= (uint32_t) (1 + 2 * (gr() & 7)) << (8 * i);
        g = v;
    }
    CK(cudaMemcpyToSymbol(g_iq3s_grid, g_grid, sizeof g_grid));
    const std::string mode = argc > 1 ? argv[1] : "check";
    if (mode == "check") {
        const int ntok = argc > 2 ? atoi(argv[2]) : 300;
        int fail = 0;
        for (const KR & k : g_k) fail |= check_one(k, ntok);
        printf(fail ? "mmq_bench check: FAILED\n" : "mmq_bench check: all good\n");
        return fail;
    }
    if (mode == "time" && argc >= 5) {
        const std::regex re(argv[2]);
        const int K = atoi(argv[3]), N = atoi(argv[4]), ntok = argc > 5 ? atoi(argv[5]) : 1024, reps = argc > 6 ? atoi(argv[6]) : 5, nlay = argc > 7 ? atoi(argv[7]) : 4;
        for (const KR & k : g_k)
            if (std::regex_search(k.name, re)) time_k(k, K, N, ntok, reps, nlay);
        return 0;
    }
    if (mode == "sustain" && argc >= 5) {
        const std::regex re(argv[2]);
        const int K = atoi(argv[3]), N = atoi(argv[4]), ntok = argc > 5 ? atoi(argv[5]) : 1024, rounds = argc > 7 ? atoi(argv[7]) : 2;
        const double secs = argc > 6 ? atof(argv[6]) : 6.0;
        for (const KR & k : g_k)
            if (std::regex_search(k.name, re)) sustain_k(k, K, N, ntok, secs, 4, rounds);
        return 0;
    }
    fprintf(stderr, "usage: %s check [ntok] | time <kernel-regex> <K> <N> [ntok] [reps] [nlay] | sustain <kernel-regex> <K> <N> [ntok] [secs] [rounds]\n", argv[0]);
    return 2;
}

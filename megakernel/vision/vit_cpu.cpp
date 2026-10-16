// Qwen3.x vision tower on the CPU, fp32, AVX2 / FMA (see vit_cpu.h). Built with -mavx2 -mfma (x86-64-v3): Zen 2 /
// 3 / 4 / 5, Intel Haswell and later; mk_vision checks the CPU before using it.
#include "vit_cpu.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace mkv {
namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 64-byte aligned fp32 storage that only grows (scratch keeps its capacity between images)
struct Buf {
    float * p = nullptr;
    size_t n = 0;
    Buf() = default;
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;
    Buf(Buf && o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
    Buf & operator=(Buf && o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    ~Buf() { free(p); }
    float * get(size_t m) {
        if (m > n) {
            free(p);
            p = nullptr;
            n = 0;
            if (posix_memalign(reinterpret_cast<void **>(&p), 64, std::max<size_t>(m, 16) * sizeof(float))) throw std::bad_alloc();
            n = m;
        }
        return p;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Thread pool: one worker per physical core, pinned; run(n, f) hands out tasks 0 .. n - 1 dynamically (an atomic
// counter), so a core that is busy with something else delays only its own task, not a fixed share of the work.

std::vector<int> cpu_order() {
    // the first SMT sibling of each core, then the second siblings (Linux sysfs topology)
    std::vector<int> first, second;
    std::set<std::string> seen;
    for (int c = 0; c < 4096; c++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!f) {
            if (c >= (int) std::thread::hardware_concurrency()) break;
            continue;
        }
        std::string s;
        std::getline(f, s);
        (seen.insert(s).second ? first : second).push_back(c);
    }
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

int physical_cores() {
    std::set<std::string> cores;
    for (int c = 0; c < 4096; c++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!f) {
            if (c >= (int) std::thread::hardware_concurrency()) break;
            continue;
        }
        std::string s;
        std::getline(f, s);
        cores.insert(s);
    }
    return cores.empty() ? (int) std::max(1u, std::thread::hardware_concurrency()) : (int) cores.size();
}

// The caller of run() works too: n - 1 workers, pinned to the CPUs after the first one in cpu_order(), which stays free
// for the (unpinned) caller, so that the n threads never outnumber the CPUs (a spinning worker on the caller's CPU
// would delay every parallel region's start).
class Pool {
  public:
    Pool(int n, bool pin) {
        const std::vector<int> order = pin ? cpu_order() : std::vector<int>{};
        for (int i = 1; i < n; i++) {
            th_.emplace_back([this] { loop(); });
            if (!order.empty()) {
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(order[i % order.size()], &cs);
                pthread_setaffinity_np(th_.back().native_handle(), sizeof(cs), &cs); // (best effort)
            }
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            gen_++;
        }
        cv_.notify_all();
        for (auto & t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }
    void run(int ntask, const std::function<void(int)> & f) {
        if (ntask <= 0) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            f_ = &f;
            ntask_ = ntask;
            next_.store(0, std::memory_order_relaxed);
            busy_.store((int) th_.size(), std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all(); // (wakes only the workers that went to sleep)
        for (int t; (t = next_.fetch_add(1, std::memory_order_relaxed)) < ntask;) f(t);
        // the workers' last tasks: spin, then sleep
        for (int s = 0; s < 200000 && busy_.load(std::memory_order_acquire) != 0; s++) _mm_pause();
        if (busy_.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lk(m_);
            done_.wait(lk, [&] { return busy_.load(std::memory_order_acquire) == 0; });
        }
    }

  private:
    void loop() {
        long seen = 0;
        for (;;) {
            // spin a little before sleeping: the encoder runs hundreds of short parallel regions back to back
            for (int s = 0; s < 20000 && gen_.load(std::memory_order_acquire) == seen; s++) _mm_pause();
            if (gen_.load(std::memory_order_acquire) == seen) {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_acquire) != seen; });
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_) return;
            const std::function<void(int)> & f = *f_;
            for (int t; (t = next_.fetch_add(1, std::memory_order_relaxed)) < ntask_;) f(t);
            if (busy_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_);
                done_.notify_one();
            }
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)> * f_ = nullptr;
    int ntask_ = 0;
    std::atomic<int> next_{0}, busy_{0};
    std::atomic<long> gen_{0};
    bool stop_ = false;
};

// ---------------------------------------------------------------------------------------------------------------
// vector math

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
inline float hmax(__m256 v) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// e^x, Cephes expf's reduction and polynomial (~1 ulp); x clamped to [-87.3, 88]
inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.33654f)), _mm256_set1_ps(88.0f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

// GELU, tanh form (the graph's ggml_gelu), as x * sigmoid(2u): 0.5 x (1 + tanh u) without the cancellation
inline __m256 gelu8(__m256 x) {
    const __m256 u = _mm256_mul_ps(_mm256_set1_ps(0.7978845608028654f),
                                   _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715f), x), _mm256_mul_ps(x, x), x));
    const __m256 e = exp8(_mm256_mul_ps(_mm256_set1_ps(-2.0f), u));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM: C[M][N] (op)= A[M][K] W^T + bias. W ([N][K] as in the GGUF) is packed once into panels of 16 outputs,
// [N / 16][K][16]; A into panels of 6 rows, [ceil(M / 6)][K][6]; a 6 x 16 FMA micro-kernel (12 accumulators) runs over
// K in blocks of 256 (the B panel, 16 KB, stays in L1 while 16 A panels stream from L2) into a 96 x 128 tile.

constexpr int MR = 6, NR = 16, KC = 256, MBP = 16, NBP = 8, LDT = NBP * NR;

struct PW {
    int N = 0, K = 0;
    Buf w;
    std::vector<float> b;
};

enum Epi { STORE, ADD, GELU };

inline void ukr(const float * __restrict a, const float * __restrict b, int kc, float * __restrict c, bool acc) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (acc) {
        c00 = _mm256_load_ps(c + 0 * LDT), c01 = _mm256_load_ps(c + 0 * LDT + 8);
        c10 = _mm256_load_ps(c + 1 * LDT), c11 = _mm256_load_ps(c + 1 * LDT + 8);
        c20 = _mm256_load_ps(c + 2 * LDT), c21 = _mm256_load_ps(c + 2 * LDT + 8);
        c30 = _mm256_load_ps(c + 3 * LDT), c31 = _mm256_load_ps(c + 3 * LDT + 8);
        c40 = _mm256_load_ps(c + 4 * LDT), c41 = _mm256_load_ps(c + 4 * LDT + 8);
        c50 = _mm256_load_ps(c + 5 * LDT), c51 = _mm256_load_ps(c + 5 * LDT + 8);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
    }
#pragma GCC unroll 4
    for (int k = 0; k < kc; k++) {
        const __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
        __m256 t = _mm256_broadcast_ss(a + 0);
        c00 = _mm256_fmadd_ps(t, b0, c00), c01 = _mm256_fmadd_ps(t, b1, c01);
        t = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(t, b0, c10), c11 = _mm256_fmadd_ps(t, b1, c11);
        t = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(t, b0, c20), c21 = _mm256_fmadd_ps(t, b1, c21);
        t = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(t, b0, c30), c31 = _mm256_fmadd_ps(t, b1, c31);
        t = _mm256_broadcast_ss(a + 4);
        c40 = _mm256_fmadd_ps(t, b0, c40), c41 = _mm256_fmadd_ps(t, b1, c41);
        t = _mm256_broadcast_ss(a + 5);
        c50 = _mm256_fmadd_ps(t, b0, c50), c51 = _mm256_fmadd_ps(t, b1, c51);
        a += MR;
        b += NR;
    }
    _mm256_store_ps(c + 0 * LDT, c00), _mm256_store_ps(c + 0 * LDT + 8, c01);
    _mm256_store_ps(c + 1 * LDT, c10), _mm256_store_ps(c + 1 * LDT + 8, c11);
    _mm256_store_ps(c + 2 * LDT, c20), _mm256_store_ps(c + 2 * LDT + 8, c21);
    _mm256_store_ps(c + 3 * LDT, c30), _mm256_store_ps(c + 3 * LDT + 8, c31);
    _mm256_store_ps(c + 4 * LDT, c40), _mm256_store_ps(c + 4 * LDT + 8, c41);
    _mm256_store_ps(c + 5 * LDT, c50), _mm256_store_ps(c + 5 * LDT + 8, c51);
}

void pack_w(Pool & pool, const std::vector<float> & W, int N, int K, PW & out) {
    if (N % NR) throw std::runtime_error("vit: output size not a multiple of 16");
    out.N = N;
    out.K = K;
    float * dst = out.w.get((size_t) N * K);
    pool.run(N / NR, [&](int jp) {
        float * d = dst + (size_t) jp * K * NR;
        for (int j = 0; j < NR; j++) {
            const float * s = W.data() + (size_t) (jp * NR + j) * K;
            for (int k = 0; k < K; k++) d[(size_t) k * NR + j] = s[k];
        }
    });
}

// The packed A layout of an M x K matrix: element (r, k) at pa(r, k, K); rows past M up to a multiple of 6 are zero.
// The producers of GEMM inputs (LayerNorm, im2col, attention, a GELU epilogue) write it directly.
inline size_t pa(int r, int k, int K) { return ((size_t) (r / MR) * K + k) * MR + r % MR; }

// C = A W^T + bias with epilogue epi into C (row-major, ldc); with `next` (STORE / GELU only) into next instead, packed
// as the A of a GEMM with K = W.N (rows >= M zero)
void gemm(Pool & pool, const float * ap, int M, const PW & W, float * C, int ldc, Epi epi, float * next = nullptr) {
    const int K = W.K, N = W.N, mp = (M + MR - 1) / MR, np = N / NR;
    // tiles of mb x nb panels (at most MBP x NBP): smaller ones when a large tile would leave fewer than ~6 tasks per
    // thread (small images), so that the last tasks do not leave most threads idle
    int mb = MBP, nb = NBP;
    while (((mp + mb - 1) / mb) * ((np + nb - 1) / nb) < 6 * pool.size() && (mb > 4 || nb > 4)) {
        if (mb >= nb) mb /= 2;
        else nb /= 2;
    }
    const int tm = (mp + mb - 1) / mb, tn = (np + nb - 1) / nb;
    pool.run(tm * tn, [&](int t) {
        alignas(64) static thread_local float ct[MBP * MR * LDT];
        const int bi = t / tn, bj = t % tn;
        const int i0 = bi * mb, i1 = std::min(mp, i0 + mb), j0 = bj * nb, j1 = std::min(np, j0 + nb);
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = std::min(KC, K - k0);
            for (int j = j0; j < j1; j++) {
                const float * b = W.w.p + ((size_t) j * K + k0) * NR;
                for (int i = i0; i < i1; i++) ukr(ap + ((size_t) i * K + k0) * MR, b, kc, ct + (i - i0) * MR * LDT + (j - j0) * NR, k0 > 0);
            }
        }
        const int c0 = j0 * NR, nc = (j1 - j0) * NR;
        const float * bias = W.b.data() + c0;
        if (next) {
            for (int i = i0; i < i1; i++)
                for (int c = 0; c < nc; c += 8) {
                    alignas(32) float v[MR][8];
                    for (int rr = 0; rr < MR; rr++) {
                        __m256 x = _mm256_add_ps(_mm256_load_ps(ct + ((i - i0) * MR + rr) * LDT + c), _mm256_loadu_ps(bias + c));
                        if (epi == GELU) x = gelu8(x);
                        if (i * MR + rr >= M) x = _mm256_setzero_ps();
                        _mm256_store_ps(v[rr], x);
                    }
                    float * d = next + ((size_t) i * N + c0 + c) * MR;
                    for (int cc = 0; cc < 8; cc++)
                        for (int rr = 0; rr < MR; rr++) d[cc * MR + rr] = v[rr][cc];
                }
            return;
        }
        const int r1 = std::min(M, i1 * MR);
        for (int r = i0 * MR; r < r1; r++) {
            const float * s = ct + (r - i0 * MR) * LDT;
            float * d = C + (size_t) r * ldc + c0;
            for (int c = 0; c < nc; c += 8) {
                const __m256 v = _mm256_add_ps(_mm256_load_ps(s + c), _mm256_loadu_ps(bias + c));
                if (epi == STORE) _mm256_storeu_ps(d + c, v);
                else if (epi == ADD) _mm256_storeu_ps(d + c, _mm256_add_ps(_mm256_loadu_ps(d + c), v));
                else _mm256_storeu_ps(d + c, gelu8(v));
            }
        }
    });
}

// LayerNorm of the M rows of x (ld D) written packed (pa) as the A of a GEMM: G consecutive rows form one row of
// K = G * D (G = 1: the blocks; G = 4: the merger's 2x2 groups); output rows past M / G zero
void layer_norm_packed(Pool & pool, const float * x, float * ap, int M, int D, int G, const float * w, const float * b, float eps) {
    const int Mo = M / G, K = G * D, panels = (Mo + MR - 1) / MR;
    pool.run(panels, [&](int p) {
        for (int ro = p * MR; ro < (p + 1) * MR; ro++) {
            float * d0 = ap + ((size_t) p * K) * MR + ro % MR;
            if (ro >= Mo) {
                for (int k = 0; k < K; k++) d0[(size_t) k * MR] = 0.f;
                continue;
            }
            for (int g = 0; g < G; g++) {
                const float * xr = x + (size_t) (ro * G + g) * D;
                __m256 s = _mm256_setzero_ps();
                for (int e = 0; e < D; e += 8) s = _mm256_add_ps(s, _mm256_loadu_ps(xr + e));
                const __m256 vm = _mm256_set1_ps(hsum(s) / D);
                __m256 q = _mm256_setzero_ps();
                for (int e = 0; e < D; e += 8) {
                    const __m256 dd = _mm256_sub_ps(_mm256_loadu_ps(xr + e), vm);
                    q = _mm256_fmadd_ps(dd, dd, q);
                }
                const __m256 rs = _mm256_set1_ps(1.0f / std::sqrt(hsum(q) / D + eps));
                float * d = d0 + (size_t) g * D * MR;
                for (int e = 0; e < D; e += 8) {
                    const __m256 dd = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + e), vm), rs);
                    alignas(32) float v[8];
                    _mm256_store_ps(v, _mm256_fmadd_ps(dd, _mm256_loadu_ps(w + e), _mm256_loadu_ps(b + e)));
                    for (int i = 0; i < 8; i++) d[(size_t) (e + i) * MR] = v[i];
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------------------------

constexpr int HD = 72, KB = 64, QP = 8, VP = (HD + NR - 1) / NR; // head size, keys per block, query panels per task, V panels

// attention, one head, a block of up to QP * 6 query rows, with both products on the GEMM micro-kernel: S = Q K^T
// (A: the rows' query panels, K = HD; B: the head's key panels kp, [ceil(M / 16)][HD][16]), an online softmax over key
// blocks of KB, O += P V (A: P packed per query panel, K = the block's keys; B: the head's value panels vb,
// [VP][Mk][16] (dimensions past HD and keys past M zero), Mk = keys rounded up to 16). The K / V block (~40 KB) is
// read once for all the query panels. qa: [QP][HD][6] (rows past `rows` zero); out: rows r0 .. of the
// output-projection GEMM's packed A (pa, K), columns col0 .. col0 + HD - 1.
void attn_block(const float * qa, int rows, const float * kp, const float * vb, int Mk, int M, float * out, int r0, int col0, int K) {
    alignas(64) static thread_local float S[QP * MR * LDT]; // scores, then probabilities (rows of LDT)
    alignas(64) static thread_local float O[QP * MR * LDT]; // output accumulators (VP * 16 columns used)
    alignas(64) static thread_local float Pa[QP * KB * MR]; // P packed as GEMM A panels
    const int np = (rows + MR - 1) / MR, R = np * MR;
    float m[QP * MR], l[QP * MR];
    for (int r = 0; r < R; r++) {
        m[r] = -INFINITY;
        l[r] = 0.f;
        for (int c = 0; c < VP * NR; c += 8) _mm256_store_ps(O + (size_t) r * LDT + c, _mm256_setzero_ps());
    }
    for (int k0 = 0; k0 < M; k0 += KB) {
        const int nk = std::min(KB, M - k0), npan = (nk + 15) / 16, nkp = npan * 16;
        for (int p = 0; p < np; p++)
            for (int j = 0; j < npan; j++)
                ukr(qa + (size_t) p * HD * MR, kp + (size_t) (k0 / 16 + j) * HD * 16, HD, S + (size_t) p * MR * LDT + j * NR, false);
        for (int r = 0; r < R; r++) {
            float * s = S + (size_t) r * LDT;
            for (int j = nk; j < nkp; j++) s[j] = -INFINITY; // (keys past M in the last panel)
            __m256 mx = _mm256_set1_ps(-INFINITY);
            for (int j = 0; j < nkp; j += 8) mx = _mm256_max_ps(mx, _mm256_load_ps(s + j));
            const float mn = std::max(m[r], hmax(mx));
            const float alpha = std::exp(m[r] - mn); // (0 on the first block)
            const __m256 vmn = _mm256_set1_ps(mn), va = _mm256_set1_ps(alpha);
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < nkp; j += 8) {
                const __m256 pv = exp8(_mm256_sub_ps(_mm256_load_ps(s + j), vmn));
                _mm256_store_ps(s + j, pv);
                sum = _mm256_add_ps(sum, pv);
            }
            l[r] = l[r] * alpha + hsum(sum);
            m[r] = mn;
            float * o = O + (size_t) r * LDT;
            for (int c = 0; c < VP * NR; c += 8) _mm256_store_ps(o + c, _mm256_mul_ps(va, _mm256_load_ps(o + c)));
            // P row r into its A panel: Pa[p][key][r % 6]
            float * pd = Pa + (size_t) (r / MR) * nkp * MR + r % MR;
            for (int j = 0; j < nkp; j++) pd[(size_t) j * MR] = s[j];
        }
        for (int p = 0; p < np; p++)
            for (int d = 0; d < VP; d++)
                ukr(Pa + (size_t) p * nkp * MR, vb + ((size_t) d * Mk + k0) * NR, nkp, O + (size_t) p * MR * LDT + d * NR, true);
    }
    for (int r = 0; r < rows; r++) {
        const float inv = 1.0f / l[r];
        const float * o = O + (size_t) r * LDT;
        float * dst = out + pa(r0 + r, col0, K);
        for (int e = 0; e < HD; e++) dst[(size_t) e * MR] = o[e] * inv;
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------

struct Vit::Impl {
    Pool pool;
    int n_embd = 0, D = 0, nH = 0, FF = 0, nL = 0, patch = 0, merge = 0, pside = 0;
    float eps = 1e-6f, mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    int min_pixels = 0, max_pixels = 0;
    struct Layer {
        PW qkv, out, up, down;
        std::vector<float> ln1w, ln1b, ln2w, ln2b;
    };
    std::vector<Layer> L;
    PW patch_w, mm0, mm2;
    std::vector<float> pos_embd, post_w, post_b;
    // scratch
    Buf x, qkv, ap0, ap1, kp, vp, rope;

    explicit Impl(int n, bool pin) : pool(n, pin) {}
};

namespace {

struct GG {
    gguf_context * g = nullptr;
    FILE * f = nullptr;
    size_t data = 0;
    ~GG() {
        if (g) gguf_free(g);
        if (f) fclose(f);
    }
    int64_t key(const char * k) const { return gguf_find_key(g, k); }
    uint32_t u32(const char * k) const {
        const int64_t i = key(k);
        if (i < 0) throw std::runtime_error(std::string("mmproj: missing ") + k);
        return gguf_get_val_u32(g, i);
    }
    float f32(const char * k, float def) const {
        const int64_t i = key(k);
        return i < 0 ? def : gguf_get_val_f32(g, i);
    }
    // a tensor widened to fp32 (row-major, ne0 fastest)
    std::vector<float> get(const std::string & name) const {
        const int64_t t = gguf_find_tensor(g, name.c_str());
        if (t < 0) throw std::runtime_error("mmproj: missing tensor " + name);
        const ggml_type type = gguf_get_tensor_type(g, t);
        const size_t nbytes = gguf_get_tensor_size(g, t);
        const size_t bs = ggml_blck_size(type), ts = ggml_type_size(type);
        const size_t n = nbytes / ts * bs;
        std::vector<uint8_t> raw(nbytes);
        if (fseeko(f, (off_t) (data + gguf_get_tensor_offset(g, t)), SEEK_SET) != 0 || fread(raw.data(), 1, nbytes, f) != nbytes)
            throw std::runtime_error("mmproj: cannot read " + name);
        std::vector<float> v(n);
        if (type == GGML_TYPE_F32) memcpy(v.data(), raw.data(), n * sizeof(float));
        else {
            const auto * tr = ggml_get_type_traits(type);
            if (!tr || !tr->to_float) throw std::runtime_error("mmproj: unsupported tensor type for " + name);
            tr->to_float(raw.data(), v.data(), (int64_t) n);
        }
        return v; // (the callers check its size against the hyperparameters)
    }
};

} // namespace

Vit::Vit(const std::string & mmproj, const Params & prm) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) throw std::runtime_error("vit: needs AVX2 + FMA");
    const int nt = prm.n_threads > 0 ? prm.n_threads : physical_cores();
    p_.reset(new Impl(nt, prm.pin));
    Impl & I = *p_;
    GG gg;
    gguf_init_params gp = {true, nullptr};
    gg.g = gguf_init_from_file(mmproj.c_str(), gp);
    if (!gg.g) throw std::runtime_error("vit: cannot read " + mmproj);
    gg.f = fopen(mmproj.c_str(), "rb");
    if (!gg.f) throw std::runtime_error("vit: cannot open " + mmproj);
    gg.data = gguf_get_data_offset(gg.g);
    {
        const int64_t k = gg.key("clip.projector_type");
        if (k < 0 || std::string(gguf_get_val_str(gg.g, k)) != "qwen3vl_merger")
            throw std::runtime_error("vit: only the qwen3vl_merger projector (Qwen3.x vision) is implemented");
    }
    I.D = (int) gg.u32("clip.vision.embedding_length");
    I.nH = (int) gg.u32("clip.vision.attention.head_count");
    I.FF = (int) gg.u32("clip.vision.feed_forward_length");
    I.nL = (int) gg.u32("clip.vision.block_count");
    I.patch = (int) gg.u32("clip.vision.patch_size");
    I.n_embd = (int) gg.u32("clip.vision.projection_dim");
    I.merge = gg.key("clip.vision.spatial_merge_size") >= 0 ? (int) gg.u32("clip.vision.spatial_merge_size") : 2;
    I.eps = gg.f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    for (const char * k : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        const int64_t i = gg.key(k);
        if (i >= 0 && gguf_get_arr_n(gg.g, i) == 3) {
            const float * v = static_cast<const float *>(gguf_get_arr_data(gg.g, i));
            memcpy(std::string(k).find("mean") != std::string::npos ? I.mean : I.stdv, v, 3 * sizeof(float));
        }
    }
    {
        const int64_t i = gg.key("clip.vision.is_deepstack_layers");
        if (i >= 0)
            for (size_t j = 0; j < gguf_get_arr_n(gg.g, i); j++)
                if (static_cast<const int8_t *>(gguf_get_arr_data(gg.g, i))[j]) throw std::runtime_error("vit: deepstack layers are not implemented");
    }
    if (I.D / I.nH != HD || I.merge != 2 || I.patch != 16) throw std::runtime_error("vit: unexpected vision tower shape");
    // llama.cpp's limits for this projector (clip.cpp: set_limit_image_tokens(8, 4096)), per merged token 32 x 32 pixels
    const int area = I.patch * I.patch * I.merge * I.merge;
    I.min_pixels = (prm.min_tokens > 0 ? prm.min_tokens : 8) * area;
    I.max_pixels = (prm.max_tokens > 0 ? prm.max_tokens : 4096) * area;
    Pool & pool = I.pool;
    const int D = I.D;
    auto vec = [&](const std::string & n, size_t expect) {
        std::vector<float> v = gg.get(n);
        if (v.size() != expect) throw std::runtime_error("vit: " + n + " has an unexpected size");
        return v;
    };
    auto mat = [&](const std::string & n, int N, int K, PW & out, const std::string & bias) {
        pack_w(pool, vec(n, (size_t) N * K), N, K, out);
        out.b = vec(bias, (size_t) N);
    };
    // patch embedding: the two temporal kernels act on the same frame for a still image, so their sum is the kernel
    {
        std::vector<float> w0 = vec("v.patch_embd.weight", (size_t) D * 3 * 256), w1 = vec("v.patch_embd.weight.1", w0.size());
        for (size_t i = 0; i < w0.size(); i++) w0[i] += w1[i];
        pack_w(pool, w0, D, 3 * 256, I.patch_w); // [out][c][ky][kx]
        I.patch_w.b = vec("v.patch_embd.bias", D);
    }
    {
        std::vector<float> pe = gg.get("v.position_embd.weight");
        I.pside = (int) std::lround(std::sqrt((double) (pe.size() / D)));
        if ((size_t) I.pside * I.pside * D != pe.size()) throw std::runtime_error("vit: position embedding is not square");
        I.pos_embd = std::move(pe);
    }
    I.L.resize(I.nL);
    for (int l = 0; l < I.nL; l++) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        Impl::Layer & y = I.L[l];
        mat(p + "attn_qkv.weight", 3 * D, D, y.qkv, p + "attn_qkv.bias");
        mat(p + "attn_out.weight", D, D, y.out, p + "attn_out.bias");
        mat(p + "ffn_up.weight", I.FF, D, y.up, p + "ffn_up.bias");
        mat(p + "ffn_down.weight", D, I.FF, y.down, p + "ffn_down.bias");
        y.ln1w = vec(p + "ln1.weight", D), y.ln1b = vec(p + "ln1.bias", D);
        y.ln2w = vec(p + "ln2.weight", D), y.ln2b = vec(p + "ln2.bias", D);
    }
    I.post_w = vec("v.post_ln.weight", D), I.post_b = vec("v.post_ln.bias", D);
    mat("mm.0.weight", 4 * D, 4 * D, I.mm0, "mm.0.bias");
    mat("mm.2.weight", I.n_embd, 4 * D, I.mm2, "mm.2.bias");
}

Vit::~Vit() = default;

int Vit::n_embd() const { return p_->n_embd; }
int Vit::n_threads() const { return p_->pool.size(); }

void Vit::target_size(int width, int height, int & tw, int & th) const {
    // mtmd-image.cpp calc_size_preserved_ratio (smart resize), float arithmetic as there
    const int f = p_->patch * p_->merge;
    auto round_by = [f](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
    auto ceil_by = [f](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
    auto floor_by = [f](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };
    int h_bar = std::max(f, round_by((float) height));
    int w_bar = std::max(f, round_by((float) width));
    if (h_bar * w_bar > p_->max_pixels) {
        const auto beta = std::sqrt(static_cast<float>(height * width) / p_->max_pixels);
        h_bar = std::max(f, floor_by(height / beta));
        w_bar = std::max(f, floor_by(width / beta));
    } else if (h_bar * w_bar < p_->min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(p_->min_pixels) / (height * width));
        h_bar = ceil_by(height * beta);
        w_bar = ceil_by(width * beta);
    }
    tw = w_bar;
    th = h_bar;
}

void Vit::preprocess(const uint8_t * rgb, int nx, int ny, std::vector<float> & img, int & tw, int & th) const {
    target_size(nx, ny, tw, th);
    // mtmd-image.cpp img_tool::resize with PAD_CEIL (the projector's default): an aspect-preserving bilinear resize,
    // centred on black; the same size: a copy
    std::vector<uint8_t> dst((size_t) tw * th * 3, 0);
    if (nx == tw && ny == th) {
        memcpy(dst.data(), rgb, dst.size());
    } else {
        const float scale = std::min(static_cast<float>(tw) / nx, static_cast<float>(th) / ny);
        const int nw = std::min(static_cast<int>(std::ceil(nx * scale)), tw), nh = std::min(static_cast<int>(std::ceil(ny * scale)), th);
        const int ox = (tw - nw) / 2, oy = (th - nh) / 2;
        const float x_ratio = nw > 1 ? static_cast<float>(nx - 1) / (nw - 1) : 0.0f;
        const float y_ratio = nh > 1 ? static_cast<float>(ny - 1) / (nh - 1) : 0.0f;
        auto lerp = [](float s, float e, float t) { return s + (e - s) * t; };
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x) {
                const float px = x * x_ratio, py = y * y_ratio;
                const int x0 = std::min(static_cast<int>(px), nx - 1), y0 = std::min(static_cast<int>(py), ny - 1);
                const int x1 = std::min(x0 + 1, nx - 1), y1 = std::min(y0 + 1, ny - 1);
                const float xf = px - x0, yf = py - y0;
                const uint8_t * p00 = rgb + ((size_t) y0 * nx + x0) * 3, * p10 = rgb + ((size_t) y0 * nx + x1) * 3;
                const uint8_t * p01 = rgb + ((size_t) y1 * nx + x0) * 3, * p11 = rgb + ((size_t) y1 * nx + x1) * 3;
                uint8_t * d = dst.data() + ((size_t) (y + oy) * tw + (x + ox)) * 3;
                for (int c = 0; c < 3; ++c) {
                    const float top = lerp(static_cast<float>(p00[c]), static_cast<float>(p10[c]), xf);
                    const float bottom = lerp(static_cast<float>(p01[c]), static_cast<float>(p11[c]), xf);
                    d[c] = static_cast<uint8_t>(lerp(top, bottom, yf));
                }
            }
    }
    img.resize(dst.size());
    for (size_t i = 0; i < dst.size(); i++) {
        const int c = (int) (i % 3);
        img[i] = ((float) dst[i] / 255.0f - p_->mean[c]) / p_->stdv[c];
    }
}

void Vit::encode(const float * img, int tw, int th, float * out) {
    Impl & I = *p_;
    Pool & pool = I.pool;
    const double t0 = now_ms();
    const int D = I.D, FF = I.FF, nH = I.nH, ps = I.patch;
    if (tw % 32 || th % 32 || tw <= 0 || th <= 0) throw std::runtime_error("vit: image size must be a positive multiple of 32");
    const int pw = tw / ps, ph = th / ps, M = pw * ph, Mp = (M + 11) / 12 * 12, nk16 = (M + 15) / 16;
    float * x = I.x.get((size_t) Mp * D);           // residual stream
    float * qkv = I.qkv.get((size_t) Mp * 3 * D);
    const size_t apn = (size_t) Mp * std::max(std::max(FF, 4 * D), 768);
    float * ap0 = I.ap0.get(apn); // packed GEMM inputs: im2col, LayerNorm outputs, attention output, merger input
    float * ap1 = I.ap1.get(apn); // packed GEMM inputs written by a GELU epilogue (MLP hidden, merger hidden)
    const int Mk = nk16 * 16;
    float * kp = I.kp.get((size_t) nH * nk16 * 16 * HD); // per head: key panels [nk16][HD][16]
    float * vp = I.vp.get((size_t) nH * VP * Mk * NR);  // per head: value panels [VP][Mk][16]
    float * rope = I.rope.get((size_t) M * HD); // per patch: 36 cos, then 36 sin
    // patch t in the merged order: 2x2 groups in raster order, each group's patches row by row
    auto patch_xy = [pw](int t, int & px, int & py) {
        const int g = t >> 2, q = t & 3, gx = g % (pw / 2), gy = g / (pw / 2);
        px = 2 * gx + (q & 1);
        py = 2 * gy + (q >> 1);
    };
    // x = learned position embedding, bilinear (align corners) from its pside x pside grid to pw x ph (ggml_interpolate)
    {
        const int S = I.pside;
        const float sf0 = pw > 1 && S > 1 ? (float) (pw - 1) / (S - 1) : (float) pw / S;
        const float sf1 = ph > 1 && S > 1 ? (float) (ph - 1) / (S - 1) : (float) ph / S;
        pool.run(M, [&](int t) {
            int px, py;
            patch_xy(t, px, py);
            const float sx = (float) px / sf0, sy = (float) py / sf1;
            int x0 = (int) std::floor(sx), y0 = (int) std::floor(sy);
            int x1 = x0 + 1, y1 = y0 + 1;
            x0 = std::max(0, std::min(x0, S - 1)), x1 = std::max(0, std::min(x1, S - 1));
            y0 = std::max(0, std::min(y0, S - 1)), y1 = std::max(0, std::min(y1, S - 1));
            const float dx = std::max(0.0f, std::min(sx - (float) x0, 1.0f)), dy = std::max(0.0f, std::min(sy - (float) y0, 1.0f));
            const float * a = I.pos_embd.data() + ((size_t) y0 * S + x0) * D, * b = I.pos_embd.data() + ((size_t) y0 * S + x1) * D;
            const float * c = I.pos_embd.data() + ((size_t) y1 * S + x0) * D, * d = I.pos_embd.data() + ((size_t) y1 * S + x1) * D;
            float * xr = x + (size_t) t * D;
            for (int e = 0; e < D; e++)
                xr[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy + d[e] * dx * dy;
            // vision RoPE angles of this patch: pairs (i, i + 36), i < 18 by its row, i >= 18 by its column, base 10000
            float * cs = rope + (size_t) t * HD;
            for (int i = 0; i < HD / 2; i++) {
                const int sec = HD / 4;
                const double th_ = (double) (i < sec ? py : px) * std::pow(10000.0, -2.0 * (i % sec) / (HD / 2));
                cs[i] = (float) std::cos(th_);
                cs[HD / 2 + i] = (float) std::sin(th_);
            }
        });
    }
    // patch embedding (+ bias) added to it: im2col rows [c][ky][kx] of each 16 x 16 patch, packed
    pool.run((M + MR - 1) / MR, [&](int p) {
        for (int rr = 0; rr < MR; rr++) {
            const int t = p * MR + rr;
            float * d = ap0 + (size_t) p * 768 * MR + rr;
            if (t >= M) {
                for (int j = 0; j < 768; j++) d[(size_t) j * MR] = 0.f;
                continue;
            }
            int px, py;
            patch_xy(t, px, py);
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < ps; ky++)
                    for (int kx = 0; kx < ps; kx++)
                        d[(size_t) (c * 256 + ky * 16 + kx) * MR] = img[((size_t) (py * ps + ky) * tw + (px * ps + kx)) * 3 + c];
        }
    });
    gemm(pool, ap0, M, I.patch_w, x, D, ADD);
    const double t1 = now_ms();
    double t_att = 0, t_ln = 0, t_gemm = 0;
    auto timed = [](double & acc, auto && f) {
        const double a0 = now_ms();
        f();
        acc += now_ms() - a0;
    };
    const float kq_scale = 1.0f / std::sqrt((float) HD);
    for (int l = 0; l < I.nL; l++) {
        const Impl::Layer & y = I.L[l];
        timed(t_ln, [&] { layer_norm_packed(pool, x, ap0, M, D, 1, y.ln1w.data(), y.ln1b.data(), I.eps); });
        timed(t_gemm, [&] { gemm(pool, ap0, M, y.qkv, qkv, 3 * D, STORE); });
        const double ta = now_ms();
        // RoPE on q and k (q also scaled), then per head: k transposed into panels of 16 keys, v into panels of 16
        // dimensions (both GEMM B layouts)
        pool.run(nH * nk16, [&](int t) {
            const int hh = t / nk16, kp16 = t % nk16;
            float * kd = kp + ((size_t) hh * nk16 + kp16) * HD * 16;
            float * vd = vp + (size_t) hh * VP * Mk * NR;
            for (int j = 0; j < 16; j++) {
                const int k = kp16 * 16 + j;
                if (k >= M) {
                    for (int d = 0; d < HD; d++) kd[d * 16 + j] = 0.f;
                    for (int d = 0; d < VP * NR; d++) vd[((size_t) (d / NR) * Mk + k) * NR + d % NR] = 0.f;
                    continue;
                }
                const float * cs = rope + (size_t) k * HD;
                float * qr = qkv + (size_t) k * 3 * D + hh * HD;
                const float * kr = qr + D, * vr = qr + 2 * D;
                for (int i = 0; i < HD / 2; i++) {
                    const float c = cs[i], s = cs[HD / 2 + i];
                    const float q0 = qr[i], q1 = qr[i + HD / 2], k0 = kr[i], k1 = kr[i + HD / 2];
                    qr[i] = (q0 * c - q1 * s) * kq_scale;
                    qr[i + HD / 2] = (q0 * s + q1 * c) * kq_scale;
                    kd[i * 16 + j] = k0 * c - k1 * s;
                    kd[(i + HD / 2) * 16 + j] = k0 * s + k1 * c;
                }
                for (int d = 0; d < VP * NR; d++) vd[((size_t) (d / NR) * Mk + k) * NR + d % NR] = d < HD ? vr[d] : 0.f;
            }
        });
        // attention per head and block of QP * 6 query rows (attn_block); the output goes packed into ap0 (its rows past
        // M, if any, zeroed here)
        for (int r = M; r < (M + MR - 1) / MR * MR; r++)
            for (int j = 0; j < D; j++) ap0[pa(r, j, D)] = 0.f;
        const int qb = QP * MR, nqb = (M + qb - 1) / qb;
        pool.run(nH * nqb, [&](int t) {
            const int hh = t / nqb, r0 = (t % nqb) * qb, rows = std::min(qb, M - r0);
            alignas(64) float qa[QP * HD * MR]; // this head's rotated query rows as GEMM A panels
            for (int r = 0; r < (rows + MR - 1) / MR * MR; r++) {
                const float * q = qkv + (size_t) (r0 + r) * 3 * D + hh * HD;
                float * d = qa + (size_t) (r / MR) * HD * MR + r % MR;
                for (int e = 0; e < HD; e++) d[(size_t) e * MR] = r < rows ? q[e] : 0.f;
            }
            attn_block(qa, rows, kp + (size_t) hh * nk16 * 16 * HD, vp + (size_t) hh * VP * Mk * NR, Mk, M, ap0, r0, hh * HD, D);
        });
        t_att += now_ms() - ta;
        timed(t_gemm, [&] { gemm(pool, ap0, M, y.out, x, D, ADD); });
        timed(t_ln, [&] { layer_norm_packed(pool, x, ap0, M, D, 1, y.ln2w.data(), y.ln2b.data(), I.eps); });
        timed(t_gemm, [&] { gemm(pool, ap0, M, y.up, nullptr, 0, GELU, ap1); });
        timed(t_gemm, [&] { gemm(pool, ap1, M, y.down, x, D, ADD); });
    }
    const double t2 = now_ms();
    // post-LN, then the merger on each 2x2 group (4 consecutive rows = one 4D row): fc1, GELU, fc2
    const int G = M / 4;
    layer_norm_packed(pool, x, ap0, M, D, 4, I.post_w.data(), I.post_b.data(), I.eps);
    gemm(pool, ap0, G, I.mm0, nullptr, 0, GELU, ap1);
    gemm(pool, ap1, G, I.mm2, out, I.n_embd, STORE);
    const double t3 = now_ms();
    ms[0] = t1 - t0;
    ms[1] = t2 - t1 - t_att;
    ms[2] = t_att;
    ms[3] = t3 - t2;
    ms[4] = t3 - t0;
    if (getenv("VIT_PROF")) {
        const double gf = 2.0 * M * ((double) D * 3 * D + (double) D * D + 2.0 * D * FF) * I.nL / 1e9;
        const double af = 4.0 * M * (double) M * HD * nH * I.nL / 1e9;
        fprintf(stderr, "[vit] %d patches: blocks gemm %.0f ms (%.0f GFLOP/s), ln %.0f, attention %.0f ms (%.0f GFLOP/s), "
                        "patch %.0f, merger %.0f, total %.0f ms\n", M, t_gemm, gf / t_gemm * 1e3, t_ln, t_att,
                af / t_att * 1e3, ms[0], ms[3], ms[4]);
    }
}

} // namespace mkv

// MKEngine::Impl: loading the model (GGUF tensors -> device layouts: repacks, re-encodes, the wv re-layout; the MTP
// file) and the constructor.

#include "engine_impl.h"

namespace {

// GGUF Q6_K block (210 B: ql[128] | qh[64] | int8 scales[16] | fp16 d) from 256 integers q in [-32, 31],
// one scale per 16 of them and the bits of d.
void put_q6k(uint8_t * blk, const int * q, const int8_t * sc, uint16_t dbits) {
    uint8_t *ql = blk, *qh = blk + 128;
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < 32; l++) {
            const int v0 = q[128 * n + l] + 32, v1 = q[128 * n + l + 32] + 32;
            const int v2 = q[128 * n + l + 64] + 32, v3 = q[128 * n + l + 96] + 32;
            ql[64 * n + l] = (uint8_t) ((v0 & 0xF) | ((v2 & 0xF) << 4));
            ql[64 * n + l + 32] = (uint8_t) ((v1 & 0xF) | ((v3 & 0xF) << 4));
            qh[32 * n + l] = (uint8_t) (((v0 >> 4) & 3) | (((v1 >> 4) & 3) << 2) | (((v2 >> 4) & 3) << 4) | (((v3 >> 4) & 3) << 6));
        }
    memcpy(blk + 192, sc, 16);
    memcpy(blk + 208, &dbits, 2);
}

// Q3_K or IQ3_S row -> Q6_K with the same values. Both store a small integer times d times an integer
// sub-block scale (Q3_K: d (sc - 32) q, q in [-4, 3], per 16; IQ3_S: d (1 + 2 s) q, |q| <= 15, per 32),
// so q = value / (d * scale) is exact and Q6_K holds it with the same d and scale. f: the row dequantized.
void reencode_q6k(uint32_t type, const uint8_t * src, const float * f, uint8_t * out, int64_t K) {
    for (int64_t b = 0; b < K / 256; b++) {
        const uint8_t * blk = src + b * 110;
        const float * y = f + b * 256;
        uint16_t dbits;
        int8_t sc[16];
        if (type == gguf::Q3_K) { // hmask[32] | qs[64] | scales[12] (6 bits each) | fp16 d
            memcpy(&dbits, blk + 108, 2);
            uint32_t aux[4];
            memcpy(aux, blk + 96, 12);
            const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
            aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
            aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
            aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
        } else { // IQ3_S: fp16 d | qs[64] | qh[8] | signs[32] | scales[4] (4 bits per 32)
            memcpy(&dbits, blk, 2);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) (1 + 2 * ((blk[106 + i / 4] >> (4 * ((i / 2) & 1))) & 0xf));
        }
        const float d = ggml_fp16_to_fp32(dbits);
        int q[256];
        for (int e = 0; e < 256; e++) {
            const float s = d * (float) sc[e / 16];
            q[e] = s == 0.0f ? 0 : (int) lrintf(y[e] / s);
        }
        put_q6k(out + b * 210, q, sc, dbits);
    }
}

// IQ4_NL row (18 B blocks: fp16 d | 16 bytes of 4-bit indices) -> Q8_0 (fp16 d | 32 int8) with the same
// d and q = kvalues_iq4nl[index]: the same values.
void reencode_iq4nl_q80(const uint8_t * src, uint8_t * out, int64_t K) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * s = src + b * 18;
        uint8_t * o = out + b * 34;
        memcpy(o, s, 2);
        for (int j = 0; j < 16; j++) {
            o[2 + j] = (uint8_t) kv[s[2 + j] & 0xf];
            o[2 + j + 16] = (uint8_t) kv[s[2 + j] >> 4];
        }
    }
}

// Q4_0 row (18 B blocks: fp16 d | 16 bytes, element j in the low nibble of byte j and element j + 16 in its high nibble,
// value d * (q - 8)) -> Q8_0 with the same d and q - 8 as int8: the same values (Q4_0 tensors outside the MTP block)
void reencode_q40_q80(const uint8_t * src, uint8_t * out, int64_t K) {
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * s = src + b * 18;
        uint8_t * o = out + b * 34;
        memcpy(o, s, 2);
        for (int j = 0; j < 16; j++) {
            o[2 + j] = (uint8_t) (int8_t) ((s[2 + j] & 0xf) - 8);
            o[2 + j + 16] = (uint8_t) (int8_t) ((s[2 + j] >> 4) - 8);
        }
    }
}

// GGUF IQ4_XS blocks (136 B: fp16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]) -> the layout rows_iq4x and the
// prefill GEMM read: pairs of blocks in 272 B (16-byte aligned, no padding): the two 8-byte GGUF headers, then
// each block's 128 index bytes in Q4_K order (see rows_iq4x). nblocks must be even.
void repack_iq4xs(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * 136, *qs = s + 8;
        uint8_t * pr = dst + (i >> 1) * 272;
        memcpy(pr + 8 * (i & 1), s, 8);
        uint8_t * o = pr + 16 + 128 * (i & 1);
        auto idx = [&](int v) {
            const int ib = v >> 5, j = v & 31;
            return j < 16 ? qs[16 * ib + j] & 0xf : qs[16 * ib + j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

// GGUF IQ4_NL (8 blocks of 18 B: fp16 d, u8 qs[16]) per 256 weights -> 144 B: the eight fp16 d, then the 128 index
// bytes in the order of repack_iq4xs (rows_iq4nl, prefill TIQ4NL)
void repack_iq4nl(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t * s = src + i * 144;
        uint8_t * o = dst + i * 144;
        for (int ib = 0; ib < 8; ib++) memcpy(o + 2 * ib, s + 18 * ib, 2);
        auto idx = [&](int v) {
            const uint8_t * qs = s + 18 * (v >> 5) + 2;
            const int j = v & 31;
            return j < 16 ? qs[j] & 0xf : qs[j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[16 + 32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

// Q3_K's 12 bytes of packed 6-bit scales -> sc - 32 (ggml's unpacking)
void q3k_scales(const uint8_t * packed, int8_t (&sc)[16]) {
    uint32_t aux[4];
    memcpy(aux, packed, 12);
    const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
    aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
    aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
    aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
    for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
}

// GGUF Q3_K row (110 B blocks: hmask[32] | qs[64] | scales[12] | fp16 d) -> the native layout of rows_q3k: values (96 B
// per block) and the Q6_K meta stream (18 B per block in pairs). y: the row as ggml dequantizes it, for the self-check
// (the native values, dequantized as ggml does, must reproduce it bit for bit). nb must be even.
bool repack_q3k(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *hm = blk, *qs = blk + 32;
        uint8_t u3[256];
        for (int e = 0; e < 256; e++) {
            const int n = e >> 7, j = (e >> 5) & 3, l = e & 31;
            u3[e] = (uint8_t) (((qs[32 * n + l] >> (2 * j)) & 3) | (((hm[l] >> (4 * n + j)) & 1) << 2));
        }
        uint8_t * o = q + (size_t) b * 96;
        memset(o, 0, 96);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int i = 0; i < 16; i++) {
                const int ea = 64 * p + 16 * hh + i, eb = ea + 32, m = i & 3, j = i >> 2;
                o[8 * w + m] |= (uint8_t) ((u3[ea] & 3) << (2 * j));
                o[8 * w + 4 + m] |= (uint8_t) ((u3[eb] & 3) << (2 * j));
                o[64 + 4 * w + m] |= (uint8_t) (((u3[ea] >> 2) << j) | ((u3[eb] >> 2) << (4 + j)));
            }
        }
        int8_t sc[16];
        q3k_scales(blk + 96, sc);
        uint16_t dbits;
        memcpy(&dbits, blk + 108, 2);
        uint8_t * mp = meta + (size_t) (b & ~1) * 18;
        memcpy(mp + 16 * (b & 1), sc, 16);
        memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
        // self-check: decode the native bytes as rows_q3k does, dequantize as ggml (dl = d (sc - 32), dl q)
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, j = i >> 2;
            const int lo = (o[8 * w + 4 * isb + m] >> (2 * j)) & 3, hb = (o[64 + 4 * w + m] >> (4 * isb + j)) & 1;
            const int8_t s = (int8_t) mp[16 * (b & 1) + 4 * p + 2 * isb + ((r >> 4) & 1)];
            const float v = (d * (float) s) * (float) (lo + 4 * hb - 4);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

// ggml's iq3s_grid, recovered with ggml's own IQ3_S dequantization (d = 1, scale 1, no signs: index -> 4 magnitudes)
std::vector<uint32_t> iq3s_grid_host() {
    std::vector<uint32_t> g(512);
    const auto * tr = ggml_get_type_traits(GGML_TYPE_IQ3_S);
    uint8_t blk[110] = {};
    const uint16_t one = 0x3c00;
    float y[256];
    for (int idx = 0; idx < 512; idx++) {
        memset(blk, 0, sizeof blk);
        memcpy(blk, &one, 2);
        blk[2] = (uint8_t) (idx & 0xff); // qs[0]: index 0 of sub-block 0
        blk[66] = (uint8_t) (idx >> 8);  // qh[0] bit 0
        tr->to_float(blk, y, 256);
        uint32_t v = 0;
        for (int m = 0; m < 4; m++) {
            const int gm = (int) y[m];
            if ((float) gm != y[m] || gm < 1 || gm > 15) throw std::runtime_error("iq3s_grid: unexpected IQ3_S dequantization");
            v |= (uint32_t) gm << (8 * m);
        }
        g[idx] = v;
    }
    return g;
}

// GGUF IQ3_S row (110 B blocks: fp16 d | qs[64] | qh[8] | signs[32] | scales[4]) -> the native layout of rows_iq3s:
// values (96 B per block) and meta (16 B per block). Self-check against y as repack_q3k.
bool repack_iq3s(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y, const std::vector<uint32_t> & grid) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        uint8_t * o = q + (size_t) b * 96;
        uint8_t * mt = meta + (size_t) b * 16;
        memset(o, 0, 96);
        memset(mt, 0, 16);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int isb = 0; isb < 2; isb++) {
                const int ib = 2 * p + isb; // the 32-weight sub-block of A (isb 0) or B (isb 1)
                for (int c = 0; c < 4; c++) {
                    const int k = 4 * hh + c; // grid index k of the sub-block (its weights 4k .. 4k + 3)
                    o[8 * w + 4 * isb + c] = qs[8 * ib + k];
                    mt[w] |= (uint8_t) (((qh[ib] >> k) & 1) << (4 * isb + c));
                }
                for (int i = 0; i < 16; i++) {
                    const int off = 16 * hh + i; // weight of the sub-block
                    const int sgn = (signs[4 * ib + off / 8] >> (off % 8)) & 1;
                    o[64 + 4 * w + (i & 3)] |= (uint8_t) (sgn << (4 * isb + (i >> 2)));
                }
            }
        }
        memcpy(mt + 8, scales, 4);
        memcpy(mt + 12, blk, 2);
        uint16_t dbits;
        memcpy(&dbits, blk, 2);
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, c = i >> 2;
            const int idx = o[8 * w + 4 * isb + c] | (((mt[w] >> (4 * isb + c)) & 1) << 8);
            const int g = (grid[idx] >> (8 * m)) & 0xff;
            const int sgn = (o[64 + 4 * w + m] >> (4 * isb + c)) & 1;
            const float db = d * (float) (1 + 2 * ((mt[8 + p] >> (4 * isb)) & 0xf));
            const float v = db * (float) g * (sgn ? -1.f : 1.f);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

} // namespace

void MKEngine::Impl::set_draft_q4(int on) {
    if (!dq4_rows) throw std::runtime_error("policy dq4 needs the draft head's Q4_K copy (MK_DRAFT_Q4, on by default) and a draft vocabulary");
    hw.dq4_n = on ? dq4_rows : 0;
    CK(mkl::set_cW(&hw.dq4_n, sizeof(int), offsetof(ModelW, dq4_n)));
}

bool MKEngine::Impl::is_mtp_name(const std::string & n) {
    static const std::string p = "blk." + std::to_string(NL) + ".";
    return n.compare(0, p.size(), p) == 0;
}

const gguf::Tensor & MKEngine::Impl::tget(const std::string & name) {
    const gguf::File & f = src_of(name);
    const auto & t = f.get(name);
    if (Prefetch * px = &f == &file ? pfx.get() : pfx_mtp.get()) px->need(t);
    return t;
}

void MKEngine::Impl::requant_q4k(const gguf::Tensor & t, uint8_t * out) {
    const int64_t K = (int64_t) t.ne[0];
    int64_t rows = 1;
    for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
    const size_t rs_from = ggml_row_size((ggml_type) t.type, K), rs_to = ggml_row_size(GGML_TYPE_Q4_K, K);
    const auto * tr = ggml_get_type_traits((ggml_type) t.type);
    parallel_rows(rows, [&](int64_t r) {
        thread_local std::vector<float> f;
        f.resize((size_t) K);
        tr->to_float(t.data + (size_t) r * rs_from, f.data(), K);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), out + (size_t) r * rs_to, 0, 1, K, nullptr); // lossy by design (the MTP block only)
    });
}

void MKEngine::Impl::rq_start() {
    rq_bg = has_mtp && !(getenv("MK_LOAD_RQBG") && atoi(getenv("MK_LOAD_RQBG")) == 0);
    if (!rq_bg) return;
    std::vector<const gguf::Tensor *> ts;
    const gguf::File & mf = mtpf ? *mtpf : file;
    for (const char * r : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                           "ffn_down.weight", "nextn.eh_proj.weight"}) {
        const auto it = mf.tensors.find("blk." + std::to_string(NL) + "." + r);
        if (it != mf.tensors.end() && requant_type(it->second.type)) ts.push_back(&it->second);
    }
    rq_th.t = std::thread([this, ts] {
        const auto t0 = lclk::now();
        try {
            for (const gguf::Tensor * t : ts) {
                int64_t rows = 1;
                for (size_t d = 1; d < t->ne.size(); d++) rows *= (int64_t) t->ne[d];
                std::vector<uint8_t> v(ggml_row_size(GGML_TYPE_Q4_K, (int64_t) t->ne[0]) * (size_t) rows);
                requant_q4k(*t, v.data());
                rq_res[t->name] = std::move(v);
            }
        } catch (...) { // (want() requantizes a tensor with no result here itself)
        }
        rq_s = lsec(t0);
    });
}

auto MKEngine::Impl::want(const std::string & name, int to) -> Src {
    const auto & t = tget(name);
    Src s;
    s.type = t.type;
    s.ne = t.ne;
    s.data = t.data;
    s.nbytes = t.nbytes;
    const bool to_f32 = to == T_F32 && t.type != gguf::F32;
    const bool to_q4 = !to_f32 && req_q4 && requant_type(t.type);
    const bool to_q6 = !to_f32 && !to_q4 && reenc3 && (t.type == gguf::Q3_K || t.type == gguf::IQ3_S);
    // native IQ4_NL unless MK_IQ4NL_Q8; Q4_0 (outside the MTP block) always (the kernels do not read it)
    const bool to_q8 = !to_f32 && !to_q4 && ((t.type == gguf::IQ4_NL && getenv("MK_IQ4NL_Q8")) || t.type == gguf::Q4_0);
    if (!to_f32 && !to_q4 && !to_q6 && !to_q8) return s; // anything the kernels cannot read is refused by up_mat / need()
    const ggml_type from = (ggml_type) t.type;
    const ggml_type dst = to_f32 ? GGML_TYPE_F32 : to_q4 ? GGML_TYPE_Q4_K : to_q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
    const int64_t K = (int64_t) t.ne[0];
    int64_t rows = 1;
    for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
    const size_t rs_from = ggml_row_size(from, K), rs_to = ggml_row_size(dst, K);
    const auto t0 = std::chrono::steady_clock::now();
    if (to_q4 && rq_bg) { // the background thread's result (rq_start)
        if (rq_th.t.joinable()) {
            auto _ = lc.scope("MTP Q4_K requant (wait for the background thread)");
            rq_th.t.join();
        }
        const auto it = rq_res.find(name);
        if (it != rq_res.end()) s.own = std::move(it->second);
        if (it != rq_res.end()) rq_res.erase(it);
    }
    if (to_q4 && s.own.empty()) {
        s.own.resize(rs_to * (size_t) rows);
        requant_q4k(t, s.own.data());
    } else if (!to_q4) {
    s.own.resize(rs_to * (size_t) rows);
    const auto * tr = ggml_get_type_traits(from);
    const auto * tr_dst = ggml_get_type_traits(dst);
    std::atomic<bool> exact{true};
    parallel_rows(rows, [&](int64_t r) {
        thread_local std::vector<float> f, g;
        f.resize((size_t) K);
        const uint8_t * src = t.data + (size_t) r * rs_from;
        uint8_t * out = s.own.data() + (size_t) r * rs_to;
        tr->to_float(src, f.data(), K);
        if (to_f32) {
            memcpy(out, f.data(), (size_t) K * 4);
            return;
        }
        if (to_q6) reencode_q6k(t.type, src, f.data(), out, K);
        else if (t.type == gguf::Q4_0) reencode_q40_q80(src, out, K);
        else reencode_iq4nl_q80(src, out, K);
        g.resize((size_t) K);
        tr_dst->to_float(out, g.data(), K);
        if (memcmp(f.data(), g.data(), (size_t) K * 4) != 0) exact = false;
    });
    if (!exact) throw std::runtime_error(name + ": re-encoding is not exact");
    }
    s.type = (uint32_t) dst;
    s.data = s.own.data();
    s.nbytes = s.own.size();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (to_q4) {
        n_req++;
        req_s += dt;
    } else {
        n_conv++;
        conv_s += dt;
    }
    return s;
}

auto MKEngine::Impl::hset() -> HSet & {
    HSet & S = hs[hs_i];
    hs_i ^= 1;
    auto _ = lc.scope("upload stall (host buffer set busy)");
    upl.wait(S.seq);
    return S;
}

void MKEngine::Impl::put(void * d, const void * s, size_t n, HSet * S) {
    const uint64_t q = upl.put(d, s, n);
    if (S && q) S->seq = q;
    wbytes += n;
}

const float * MKEngine::Impl::up_f32(const std::string & name, size_t expect) {
    const auto & t = tget(name);
    if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
    size_t n = t.nbytes / 4;
    if (expect && n != expect) throw std::runtime_error(name + " bad size");
    return dput<float>(t.data, n);
}

Mat MKEngine::Impl::up_mat(const std::string & name, int K, int N, int to) {
    Src t = want(name, to < 0 ? (int) src_of(name).get(name).type : to);
    if (!shape2(t.ne, (uint64_t) K, (uint64_t) N))
        throw std::runtime_error(name + " has shape " + gguf::shape_str(t.ne) + ", expected " + gguf::shape_str({(uint64_t) K, (uint64_t) N}));
    Mat m{};
    m.K = K;
    m.N = N;
    m.type = (int) t.type;
    const int nb = K / 256;
    // a source owned by want() (re-encoded / re-quantised copy) must stay alive until its upload is done
    auto keep = [&](Src & s) {
        if (!s.own.empty()) own_keep.push_back(std::move(s.own));
    };
    if (t.type == gguf::Q4_K || t.type == gguf::Q5_K) {
        m.rb = (int) gguf::type_row_bytes(t.type, K);
        if (wv_lay) {
            HSet & S = hset();
            wv_place(m, t.data, t.nbytes, nullptr, 0, S);
        } else
        {
            m.w = dput<uint8_t>(t.data, t.nbytes);
            keep(t);
        }
    } else if (t.type == gguf::Q6_K) {
        // two streams (rows_q6k): 192 B of values per block, and per row a meta stream of 18 B per block in
        // pairs (sc even | sc odd | d even | d odd)
        if (nb & 1) throw std::runtime_error(name + ": Q6_K rows need an even number of 256-blocks");
        m.rb = nb * Q6_BLK;
        m.nd = nb * (Q6_META / 2); // halves
        HSet & S = hset();
        const size_t qn = (size_t) N * m.rb, mn = (size_t) N * nb * Q6_META;
        uint8_t * qd = S.q.get(qn);
        uint8_t * md = S.m.get(mn);
        const uint8_t * src = t.data;
        {
            auto _ = lc.scope("repack Q6_K two-stream");
            parallel_rows(N, [&](int64_t r) {
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = src + ((size_t) r * nb + b) * 210;
                    const uint8_t * ql = blk;
                    const uint8_t * qh = blk + 128;
                    const int8_t * sc = (const int8_t *) (blk + 192);
                    uint16_t dbits;
                    memcpy(&dbits, blk + 208, 2);
                    uint8_t vals[256];
                    for (int nn = 0; nn < 2; nn++)
                        for (int l = 0; l < 32; l++) {
                            const uint8_t * L = ql + 64 * nn;
                            const uint8_t * H = qh + 32 * nn;
                            vals[128 * nn + l + 0] = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4);
                            vals[128 * nn + l + 32] = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4);
                            vals[128 * nn + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                            vals[128 * nn + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                        }
                    uint8_t * dst = qd + (size_t) r * m.rb + b * Q6_BLK;
                    for (int p = 0; p < 4; p++)
                        for (int i = 0; i < 32; i++)
                            dst[32 * p + i] = (vals[64 * p + i] & 0xF) | ((vals[64 * p + 32 + i] & 0xF) << 4);
                    for (int p = 0; p < 4; p++)
                        for (int hh = 0; hh < 2; hh++) {
                            uint32_t hA = 0, hB = 0;
                            for (int i = 0; i < 16; i++) {
                                const int sh = 8 * (i % 4) + 2 * (i / 4);
                                hA |= (uint32_t) (vals[64 * p + 16 * hh + i] >> 4) << sh;
                                hB |= (uint32_t) (vals[64 * p + 32 + 16 * hh + i] >> 4) << sh;
                            }
                            memcpy(dst + 128 + 8 * (2 * p + hh), &hA, 4);
                            memcpy(dst + 128 + 8 * (2 * p + hh) + 4, &hB, 4);
                        }
                    uint8_t * mp = md + ((size_t) r * nb + (b & ~1)) * Q6_META;
                    memcpy(mp + 16 * (b & 1), sc, 16);
                    memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
                }
            });
        }
        // main layers: ffn_down's tile-major values and meta pairs, every Q6_K fragment-ordered with the meta inline
        // (one stream: no meta, m.d stays null)
        if (wv_lay) wv_place(m, qd, qn, md, mn, S);
        else
        {
            m.w = dput<uint8_t>(qd, qn, &S);
            m.d = reinterpret_cast<const __half *>(dput<uint8_t>(md, mn, &S));
        }
    } else if (t.type == gguf::IQ4_XS) {
        m.type = T_IQ4X;
        if (nb & 1) throw std::runtime_error(name + ": IQ4_XS rows need an even number of 256-blocks");
        m.rb = nb * 136;
        HSet & S = hset();
        const size_t qn = (size_t) N * m.rb;
        uint8_t * qd = S.q.get(qn);
        {
            auto _ = lc.scope("repack IQ4_XS");
            parallel_rows(N, [&](int64_t r) { repack_iq4xs(t.data + (size_t) r * nb * 136, qd + (size_t) r * m.rb, nb); });
        }
        if (wv_lay) wv_place(m, qd, qn, nullptr, 0, S);
        else
        m.w = dput<uint8_t>(qd, qn, &S);
    } else if (t.type == gguf::Q3_K || t.type == gguf::IQ3_S) {
        // native (rows_q3k / rows_iq3s, prefill TQ3K / TIQ3S): values 96 B per block + meta (Q3_K: the Q6_K meta
        // stream, 18 B; IQ3_S: 16 B), checked bit for bit against ggml's dequantization of every row
        const bool q3 = t.type == gguf::Q3_K;
        m.type = q3 ? T_Q3K : T_IQ3S;
        if (nb & 1) throw std::runtime_error(name + ": Q3_K / IQ3_S rows need an even number of 256-blocks");
        const int mb = q3 ? Q6_META : IQ3S_META;
        m.rb = nb * Q3_BLK;
        m.nd = nb * mb / 2; // halves
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint32_t> grid;
        if (!q3) {
            grid = iq3s_grid_host();
            if (!iq3s_grid_up) {
                CK(mkl::set_g_iq3s_grid(grid.data(), grid.size() * 4));
                iq3s_grid_up = true;
            }
        }
        HSet & S = hset();
        const size_t qn = (size_t) N * m.rb, mn = (size_t) N * nb * mb;
        uint8_t * qd = S.q.get(qn);
        uint8_t * md = S.m.get(mn);
        const auto * tr = ggml_get_type_traits((ggml_type) t.type);
        std::atomic<bool> exact{true};
        {
            auto _ = lc.scope("repack Q3_K / IQ3_S native (+ check)");
            parallel_rows(N, [&](int64_t r) {
                thread_local std::vector<float> f;
                f.resize((size_t) K);
                const uint8_t * src = t.data + (size_t) r * nb * 110;
                tr->to_float(src, f.data(), K);
                uint8_t * qo = qd + (size_t) r * m.rb;
                uint8_t * mo = md + (size_t) r * nb * mb;
                if (!(q3 ? repack_q3k(src, qo, mo, nb, f.data()) : repack_iq3s(src, qo, mo, nb, f.data(), grid))) exact = false;
            });
        }
        if (!exact) throw std::runtime_error(name + ": native Q3_K / IQ3_S repack is not exact");
        if (wv_lay && wv_fd) wv_place(m, qd, qn, md, mn, S); // main-layer ffn_down: tile-major values and meta
        else
        {
            m.w = dput<uint8_t>(qd, qn, &S);
            m.d = reinterpret_cast<const __half *>(dput<uint8_t>(md, mn, &S));
        }
        n_nat++;
        nat_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } else if (t.type == gguf::IQ4_NL) {
        m.type = T_IQ4NL;
        m.rb = nb * 144;
        HSet & S = hset();
        const size_t qn = (size_t) N * m.rb;
        uint8_t * qd = S.q.get(qn);
        {
            auto _ = lc.scope("repack IQ4_NL");
            parallel_rows(N, [&](int64_t r) { repack_iq4nl(t.data + (size_t) r * nb * 144, qd + (size_t) r * m.rb, nb); });
        }
        if (wv_lay) wv_place(m, qd, qn, nullptr, 0, S);
        else
        m.w = dput<uint8_t>(qd, qn, &S);
    } else if (t.type == gguf::Q8_0) {
        const int nb32 = K / 32;
        if (K % 256) throw std::runtime_error(name + ": Q8_0 rows need a multiple of 256 values");
        m.rb = K;
        m.nd = nb32;
        HSet & S = hset();
        const size_t qn = (size_t) N * K, mn = (size_t) N * nb32 * 2;
        uint8_t * qd = S.q.get(qn);
        uint8_t * md = S.m.get(mn); // fp16 scales, one per natural 32-block
        const uint8_t * src = t.data;
        {
            auto _ = lc.scope("repack Q8_0");
            parallel_rows(N, [&](int64_t r) {
                const uint8_t * row = src + (size_t) r * nb32 * 34;
                int8_t vals[256];
                uint8_t * dst = qd + (size_t) r * K;
                for (int j = 0; j < nb32; j++) memcpy(md + ((size_t) r * nb32 + j) * 2, row + 34 * j, 2);
                for (int b = 0; b < K / 256; b++) { // (K % 256 == 0: the 32-unit permutation stays inside a 256-block)
                    for (int j = 0; j < 8; j++) memcpy(&vals[32 * j], row + 34 * (8 * b + j) + 2, 32);
                    for (int u = 8 * b; u < 8 * b + 8; u++) {
                        const int p = (u >> 1) & 3, hh = u & 1;
                        memcpy(dst + 32 * u, &vals[64 * p + 16 * hh], 16);
                        memcpy(dst + 32 * u + 16, &vals[64 * p + 32 + 16 * hh], 16);
                    }
                }
            });
        }
        if (wv_lay) wv_place(m, qd, qn, md, mn, S);
        else
        {
            m.w = dput<uint8_t>(qd, qn, &S);
            m.d = (const __half *) dput<uint16_t>(md, mn / 2, &S);
        }
    } else {
        throw std::runtime_error(name + ": unsupported quant type " + std::to_string(t.type));
    }
    return m;
}

size_t MKEngine::Impl::wv_relay(Mat & m, const uint8_t * w, const uint8_t * meta, size_t meta_n, uint8_t * out, uint8_t * out_meta) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t r = wv_relayout(m.type, m.N, m.K, m.rb, w, meta, meta_n, out, out_meta);
    m.fm = 1;
    wv_n++;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    wv_s += s;
    lc.add("wv re-layout", s);
    return r;
}

void MKEngine::Impl::wv_place(Mat & m, const uint8_t * w, size_t wn, const uint8_t * meta, size_t meta_n, HSet & S) {
    const bool q6f = m.type == T_Q6K;
    const size_t on = q6f ? wn + meta_n : wn, omn = q6f ? 0 : meta_n;
    const size_t moff = (wn + 255) & ~(size_t) 255;
    bool dev = gpurl && wv_dev_ok(m.type, m.N, m.K, m.rb, meta_n, wv_probes);
    if (dev && !stg_d) {
        stg_cap = (size_t) 192 << 20;
        if (cudaMalloc(&stg_d, stg_cap) != cudaSuccess) {
            cudaGetLastError();
            stg_d = nullptr;
            stg_cap = 0;
        }
    }
    dev = dev && moff + meta_n <= stg_cap;
    if (dev && wv_fm(m.type)) {
        WvProbe & p = wv_probes.at(m.type);
        if (!p.dtab) {
            void * d = nullptr;
            CK(cudaMalloc(&d, p.tab.size() * 4));
            CK(cudaMemcpy(d, p.tab.data(), p.tab.size() * 4, cudaMemcpyHostToDevice));
            wv_tabs.push_back(d);
            p.dtab = static_cast<const uint32_t *>(d);
        }
    }
    if (!dev) {
        uint8_t * o = S.o.get(on);
        uint8_t * om = meta_n ? S.om.get(meta_n) : nullptr;
        wv_relay(m, w, meta, meta_n, o, om);
        m.w = dput<uint8_t>(o, on, &S);
        if (omn) m.d = reinterpret_cast<const __half *>(dput<uint8_t>(om, omn, &S));
        return;
    }
    uint8_t * dw = dalloc<uint8_t>(on, false);
    uint8_t * dm = omn ? dalloc<uint8_t>(omn, false) : nullptr;
    wbytes += on + omn;
    const int type = m.type, N = m.N, K = m.K, rb = m.rb;
    uint8_t * sw = stg_d, * sm = stg_d + moff;
    const WvProbe * pp = wv_fm(m.type) ? &wv_probes.at(m.type) : nullptr; // (a map node: stable while others are added)
    const uint64_t q = upl.put2(sw, w, wn, sm, meta, meta_n, [=](cudaStream_t st) {
        return mkl::wv_dev_launch(type, N, K, rb, sw, sm, meta_n, dw, dm, st, pp ? pp->dtab : nullptr, pp ? (int) pp->tab.size() : 0,
                                  pp ? pp->G : 0, pp ? pp->seg : 0);
    });
    if (q) S.seq = relay_seq = q;
    m.w = dw;
    if (omn) m.d = reinterpret_cast<const __half *>(dm);
    m.fm = 1;
    wv_n++;
    wv_dev_n++;
}

void MKEngine::Impl::wv_dev_done() {
    if (!stg_d && wv_tabs.empty()) return;
    auto _ = lc.scope("wv device re-layout tail (wait)");
    upl.wait(relay_seq);
    if (stg_d) cudaFree(stg_d);
    for (void * t : wv_tabs) cudaFree(t);
    stg_d = nullptr;
    stg_cap = 0;
    wv_tabs.clear();
    for (auto & kv : wv_probes) kv.second.dtab = nullptr;
}

QBuf MKEngine::Impl::qbuf(int U) {
    QBuf q;
    q.U = U;
    q.xa = dalloc<int8_t>((size_t) RMAX * U * 16);
    q.xb = dalloc<int8_t>((size_t) RMAX * U * 16);
    q.meta = dalloc<uint2>((size_t) RMAX * U);
    q.ssq = dalloc<float>((size_t) RMAX * U);
    return q;
}

void MKEngine::Impl::fault_inject() {
    fault_now = false;
    fprintf(stderr, "[mk] MK_FAULT: injecting a %s CUDA error\n", fault_kind == 1 ? "non-sticky" : "sticky");
    if (fault_kind == 1) CK(cudaMemcpy(hp.h, hp.h, 4, (cudaMemcpyKind) 99));
    else mkl::pk_fault();
}

void MKEngine::Impl::check_mtp_file(const gguf::File & m, const std::string & mp) const {
    auto bad = [&](const std::string & what) {
        throw std::runtime_error("--mtp " + mp + ": " + what + " (an MTP layer file of this model is needed)");
    };
    auto gi = [](const gguf::File & f, const std::string & k) -> long long {
        const auto it = f.ints.find(k);
        return it == f.ints.end() ? -1 : (long long) it->second;
    };
    const auto arch = m.strs.find("general.architecture");
    if (arch == m.strs.end() || arch->second != "qwen35")
        bad("architecture " + (arch == m.strs.end() ? std::string("missing") : "'" + arch->second + "'") + ", expected qwen35");
    for (const char * k : {"embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv",
                           "attention.key_length", "attention.value_length"}) {
        const std::string key = std::string("qwen35.") + k;
        if (gi(m, key) != gi(file, key))
            bad(key + " " + std::to_string(gi(m, key)) + ", the model has " + std::to_string(gi(file, key)));
    }
    if (gi(m, "qwen35.block_count") != NLA) bad("block_count " + std::to_string(gi(m, "qwen35.block_count")) + ", expected " +
                                               std::to_string(NLA) + " (the MTP block at layer " + std::to_string(NL) + ")");
    if (gi(m, "qwen35.nextn_predict_layers") == 0) bad("nextn_predict_layers 0");
    const auto vm = m.arrs.find("tokenizer.ggml.tokens"), vf = file.arrs.find("tokenizer.ggml.tokens");
    if (vm != m.arrs.end() && vf != file.arrs.end() && vm->second != vf->second)
        bad("vocabulary of " + std::to_string(vm->second) + " tokens, the model has " + std::to_string(vf->second));
    for (const char * k : {"tokenizer.ggml.model", "tokenizer.ggml.pre"}) {
        const auto a = m.strs.find(k), b = file.strs.find(k);
        if (a != m.strs.end() && b != file.strs.end() && a->second != b->second) bad(std::string(k) + " '" + a->second + "', the model has '" + b->second + "'");
    }
    for (const char * k : {"token_embd.weight", "output.weight"}) { // (copies the engine does not use: their shape only)
        const auto it = m.tensors.find(k);
        if (it != m.tensors.end() && !shape2(it->second.ne, D, VOCAB)) bad(std::string(k) + " is not [" + std::to_string(D) + ", " + std::to_string(VOCAB) + "]");
    }
    for (const char * r : {"attn_norm.weight", "post_attention_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                           "ffn_down.weight", "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                           "nextn.shared_head_norm.weight"}) {
        const std::string n = "blk." + std::to_string(NL) + "." + r;
        if (!m.tensors.count(n)) bad("missing tensor " + n);
    }
}

MKEngine::Impl::Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_, const std::string & mtp_)
    : file(path), ctx(ctx_), ndraft(ndraft_), nd_avail(ndraft_), n_draft_max(ndraft_), n_draft_cap(ndraft_), kv_mode(kv_mode_) {
    // a throw anywhere below: ~Impl does not run for a half-built engine, so this frees what it allocated (destroyed
    // after upl_halt: no queued upload still reads a buffer). A caller can then retry, e.g. with a smaller context.
    struct FreeOnThrow {
        Impl * p;
        int n0;
        ~FreeOnThrow() {
            if (std::uncaught_exceptions() <= n0) return;
#if MK_SPEC_LOG
            if (p->slog) fclose(p->slog);
#endif
            p->release();
        }
    } free_on_throw{this, std::uncaught_exceptions()};
    // a refused file (a throw anywhere below) unwinds this body before the members go: the upload worker stops first,
    // so no queued copy reads a buffer a destroyed member owned (own_keep, wv_probes: declared after upl)
    struct UplHalt {
        Uploader & u;
        ~UplHalt() { u.halt(); }
    } upl_halt{upl};
    model_path = path;
    if (file.strs["general.architecture"] != "qwen35") throw std::runtime_error("expected qwen35 architecture");
    // block_count NLA: the 64 layers + the MTP layer; NL: a conversion without the MTP layer
    if ((file.ints["qwen35.block_count"] != NLA && file.ints["qwen35.block_count"] != NL) || file.ints["qwen35.embedding_length"] != D)
        throw std::runtime_error("unexpected model dimensions (this kernel is specialised for the 27B)");
    // compiled in (the RoPE tables, the RMS norms): a model of these shapes with other values would load and answer wrong
    for (const auto & [key, want] : {std::pair<const char *, float>{"qwen35.rope.freq_base", ROPE_BASE},
                                     std::pair<const char *, float>{"qwen35.attention.layer_norm_rms_epsilon", EPS}}) {
        const auto it = file.floats.find(key);
        if (it == file.floats.end()) {
            fprintf(stderr, "[mk] warning: the model has no %s: this kernel takes it as %g\n", key, want);
        } else if (it->second != want) {
            char msg[160];
            snprintf(msg, sizeof msg, "the model's %s is %g, this kernel is built for %g", key, it->second, want);
            throw std::runtime_error(msg);
        }
    }
    // also compiled in: 64 rotated dims (the RoPE table), the IMROPE sections 11 / 11 / 10 (rope_sync: pair i takes t, h,
    // w for i % 3), an attention layer every 4th (L.attn); read as llama.cpp does (a scalar for the sections: all four)
    const std::pair<const char *, std::vector<int64_t>> ihp[] = {{"qwen35.rope.dimension_count", {64}},
                                                                 {"qwen35.rope.dimension_sections", {11, 11, 10, 0}},
                                                                 {"qwen35.full_attention_interval", {4}}};
    const auto vstr = [](const std::vector<int64_t> & a) {
        std::string s;
        for (size_t i = 0; i < a.size(); i++) s += (i ? ", " : "") + std::to_string(a[i]);
        return a.size() == 1 ? s : "[" + s + "]";
    };
    for (const auto & [key, want] : ihp) {
        std::vector<int64_t> v;
        std::string got = "not an integer"; // (a float, a string, a float or long array)
        if (const auto ia = file.iarrs.find(key); ia != file.iarrs.end()) {
            v = ia->second;
            got = vstr(v);
        } else if (const auto in = file.ints.find(key); in != file.ints.end()) {
            v.assign(want.size(), (int64_t) in->second);
            got = vstr(v);
        } else if (!file.arrs.count(key) && !file.floats.count(key) && !file.strs.count(key)) {
            fprintf(stderr, "[mk] warning: the model has no %s: this kernel takes it as %s\n", key, vstr(want).c_str());
            continue;
        }
        if (v != want) throw std::runtime_error(std::string("the model's ") + key + " is " + got + ", this kernel is built for " + vstr(want));
    }
    if (ctx > ctx_limit())
        throw std::runtime_error("context of " + std::to_string(ctx) + " positions exceeds the model's context length (" +
                                 std::to_string(ctx_limit()) + ", qwen35.context_length)");
    {
        // MTP source: the mtp argument (CLI --mtp FILE / --no-mtp), else MK_NO_MTP=1 / MK_MTP=FILE, else the model file's
        std::string mtp = mtp_;
        if (mtp.empty()) {
            if (getenv("MK_NO_MTP") && atoi(getenv("MK_NO_MTP")) != 0) mtp = "none";
            else if (getenv("MK_MTP") && getenv("MK_MTP")[0]) mtp = getenv("MK_MTP");
        }
        const bool file_mtp = file.tensors.count("blk." + std::to_string(NL) + ".nextn.eh_proj.weight") != 0;
        if (mtp == "none") {
            has_mtp = false;
            fprintf(stderr, "[mk] MTP layer off (--no-mtp / MK_NO_MTP=1): plain decoding, depth 0\n");
        } else if (!mtp.empty()) {
            mtpf.reset(new gguf::File(mtp));
            check_mtp_file(*mtpf, mtp);
            mtp_desc = file_ident(mtp, *mtpf);
            fprintf(stderr, "[mk] MTP layer from %s%s\n", mtp.c_str(), file_mtp ? " (instead of the model file's own)" : "");
        } else if (!file_mtp) {
            has_mtp = false;
            fprintf(stderr, "[mk] the model file has no MTP layer (blk.%d): plain decoding, depth 0 (--mtp FILE loads one from another GGUF)\n", NL);
        }
        if (!has_mtp) {
            mtp_desc = "none";
            kv_slots = N_ATT_SLOTS - 1;
            if (ndraft > 0) fprintf(stderr, "[mk] n_draft %d -> 0 (no MTP layer)\n", ndraft);
            ndraft = nd_avail = n_draft_max = n_draft_cap = 0;
        }
    }
    // read-ahead of the tensor data from here on (loader.cuh Prefetch; MK_LOAD_IO threads, 0: off), 3 GiB ahead at most
    // (the model file without an MTP block it does not supply; a --mtp file: its MTP block only, a few threads)
    if (const int nio = getenv("MK_LOAD_IO") ? atoi(getenv("MK_LOAD_IO")) : 8; nio > 0) {
        if (has_mtp && !mtpf) pfx.reset(new Prefetch(file, nio, (size_t) 3 << 30));
        else pfx.reset(new Prefetch(file, nio, (size_t) 3 << 30, [](const std::string & n) { return !is_mtp_name(n); }));
        if (mtpf) pfx_mtp.reset(new Prefetch(*mtpf, std::max(1, nio / 4), (size_t) 3 << 30, is_mtp_name));
    }
    cudaDeviceProp prop;
    {
        auto _ = lc.scope("CUDA context");
        CK(cudaGetDevice(&cuda_dev));
        CK(cudaGetDeviceProperties(&prop, cuda_dev));
        CK(cudaFree(0));
    }
    { // the persistent kernel runs one block per SM (mk_min_grid: launches of up to MAXT rows when n_draft > 4)
        const int rows = ndraft + 1 > MAXT_PF ? MAXT : MAXT_PF, need = mk_min_grid(rows);
        if (prop.multiProcessorCount < need)
            throw std::runtime_error(std::string(prop.name) + " has " + std::to_string(prop.multiProcessorCount) +
                                     " SMs; the megakernel runs one block per SM and needs at least " + std::to_string(need) +
                                     " (launches of " + std::to_string(rows) + " rows at n_draft " + std::to_string(ndraft) +
                                     "; n_draft <= 4: " + std::to_string(mk_min_grid(MAXT_PF)) + ")");
    }
#ifdef __CUDA_ARCH_LIST__
    { // (without machine code for the GPU the driver JIT-compiles the megakernel from PTX, or finds no kernel image)
        bool ok = false;
        std::string have;
        for (int a : {__CUDA_ARCH_LIST__}) {
            ok |= a / 100 == prop.major && a / 10 % 10 <= prop.minor;
            have += (have.empty() ? "sm_" : ", sm_") + std::to_string(a / 10);
        }
        if (!ok)
            throw std::runtime_error(std::string(prop.name) + " is sm_" + std::to_string(10 * prop.major + prop.minor) +
                                     "; this build has GPU code for " + have +
                                     " only: rebuild for it (make ARCH=..., build.ps1 -Arch ...; docs/BUILDING.md)");
    }
#endif
    // MK_RESERVE_EARLY (default 1; 0 = 1ecb507's order): the driver reserves local memory for every thread slot by the stack
    // limit and grows it at the first launch whose frame needs more; set to mk_cycle's frame (the largest; prefill kernels'
    // <= 40 B) here, before the weights, and kept, it no longer comes last, where a full device leaves it to system memory
    // (Windows / WSL2 sysmem fallback)
    reserve_early = !getenv("MK_RESERVE_EARLY") || atoi(getenv("MK_RESERVE_EARLY")) != 0;
    if (reserve_early) {
        cudaFuncAttributes fa{};
        CK(cudaFuncGetAttributes(&fa, mkl::k_mk_cycle));
        const char * sl = getenv("MK_STACK_LIMIT");
        stack_res = std::max((size_t) fa.localSizeBytes, sl ? (size_t) atol(sl) : (size_t) 0);
        CK(cudaDeviceSetLimit(cudaLimitStackSize, stack_res));
        fprintf(stderr, "[mk] MK_RESERVE_EARLY: stack limit %zu B per thread (mk_cycle's frame %zu B), kernel loads before the KV cache\n",
                stack_res, (size_t) fa.localSizeBytes);
    }
    upl.start(!(getenv("MK_LOAD_ASYNC") && atoi(getenv("MK_LOAD_ASYNC")) == 0));
    // token_embd's mapped pinned table (0.7 GB: ~0.8 s under WSL2) allocated by a helper thread while the weights load
    // (MK_LOAD_PINBG=0: at its turn)
    const bool embd_dev = getenv("MK_EMBD_DEVICE") && getenv("MK_EMBD_DEVICE")[0] == '1';
    Joiner embd_th;
    cudaError_t embd_err = cudaSuccess;
    double embd_pin_s = 0;
    const size_t embd_nb = file.get("token_embd.weight", {(uint64_t) D, (uint64_t) VOCAB}).nbytes; // (rows are read by token id)
    auto embd_alloc = [&] {
        const auto t0 = lclk::now();
        embd_err = cudaHostAlloc(&embd_host, embd_nb, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        embd_pin_s = lsec(t0);
    };
    if (!embd_dev && !(getenv("MK_LOAD_PINBG") && atoi(getenv("MK_LOAD_PINBG")) == 0)) embd_th.t = std::thread(embd_alloc);
    nsm = std::min(prop.multiProcessorCount, MK_MAX_GRID); // (blocks of the persistent kernel)
    if (nsm < prop.multiProcessorCount)
        fprintf(stderr, "[mk] %d SMs: the persistent kernel runs %d blocks (MK_MAX_GRID)\n", prop.multiProcessorCount, nsm);
    CK(cudaEventCreateWithFlags(&wd_ev, cudaEventDisableTiming));
    if (const char * w = getenv("MK_WATCHDOG")) wd_limit = atof(w);
    if (const char * w = getenv("MK_WATCHDOG_SPIN")) wd_spin_s = atof(w);
    upl.wd_s = wd_limit > 0.0 ? std::max(wd_limit, 120.0) : 0.0; // (the weight upload: as the prefill chunks)
    if (const char * f = getenv("MK_FAULT")) {
        const std::string s = f;
        const size_t c = s.find(':');
        fault_kind = s.compare(0, c, "cuda") == 0 ? 1 : s.compare(0, c, "sticky") == 0 ? 2 : 0;
        fault_at = c == std::string::npos ? 1 : atoi(s.c_str() + c + 1);
        if (!fault_kind || fault_at < 1) throw std::runtime_error("MK_FAULT: cuda:<n> or sticky:<n>");
    }
    if (memlog) {
        CK(cudaFree(0));
        mem_mark("context");
    }

    auto blk = [](int i, const char * s) { return "blk." + std::to_string(i) + "." + s; };
    // The kernels take the weight type as a template argument and dispatch on each matrix's
    // type at run time (gemv_pairs_any, resid_any, ffn_gu), for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS (plus what want()
    // re-encodes into those). Check every role so that an unsupported type is refused at load.
    auto tname = [](int t) -> std::string {
        switch (t) {
            case gguf::F32: return "F32";
            case gguf::F16: return "F16";
            case gguf::Q4_0: return "Q4_0";
            case gguf::Q8_0: return "Q8_0";
            case gguf::Q3_K: return "Q3_K";
            case gguf::Q4_K: return "Q4_K";
            case gguf::Q5_K: return "Q5_K";
            case gguf::Q6_K: return "Q6_K";
            case gguf::IQ4_NL: return "IQ4_NL";
            case gguf::IQ3_S: return "IQ3_S";
            case gguf::IQ4_XS: return "IQ4_XS";
            default: return "type " + std::to_string(t);
        }
    };
    auto need = [&](const std::string & name, int got, std::initializer_list<int> ok) {
        for (int t : ok)
            if (got == t) return;
        std::string want;
        for (int t : ok) want += (want.empty() ? "" : " or ") + tname(t);
        throw std::runtime_error(name + " is " + tname(got) + ", the kernel needs " + want +
                                 " (see README: the GGUF)");
    };
    auto needq = [&](const std::string & name, int got) {
        need(name, got, {T_Q4K, T_Q5K, T_Q6K, T_Q8_0, T_IQ4X, T_IQ4NL, T_Q3K, T_IQ3S});
    };
    if (const char * e = getenv("MK_MTP_Q4")) {
        if (strcmp(e, "1"))
            throw std::runtime_error(std::string("MK_MTP_Q4=") + e + " was removed: the MTP block is always re-quantised to Q4_K");
    }
    if (const char * e = getenv("MK_DRAFT_Q4")) {
        if (strcmp(e, "0") && strcmp(e, "1")) throw std::runtime_error("MK_DRAFT_Q4: 0 or 1");
        draft_q4 = e[0] == '1';
        draft_q4_env = true;
    }
    rq_start();
    wrec_on = true; // (MK_WDIGEST: the weight allocations from here to token_embd, without the draft ids)
    for (int i = 0; i < (has_mtp ? NLA : NL); i++) { // (no MTP block: hw.L[NL] stays empty)
        LayerW & L = hw.L[i];
        const bool mtp = i == NL;
        req_q4 = mtp;
        wv_lay = !mtp;
        L.attn = mtp || ((i + 1) % 4 == 0);
        L.attn_norm = up_f32(blk(i, "attn_norm.weight"), D);
        L.post_norm = up_f32(blk(i, "post_attention_norm.weight"), D);
        if (L.attn) {
            L.q = up_mat(blk(i, "attn_q.weight"), D, QG);
            L.k = up_mat(blk(i, "attn_k.weight"), D, KVD);
            L.v = up_mat(blk(i, "attn_v.weight"), D, KVD);
            L.o = up_mat(blk(i, "attn_output.weight"), VD, D);
            needq(blk(i, "attn_q.weight"), L.q.type);
            needq(blk(i, "attn_k.weight"), L.k.type);
            needq(blk(i, "attn_v.weight"), L.v.type);
            needq(blk(i, "attn_output.weight"), L.o.type);
            L.qn = up_f32(blk(i, "attn_q_norm.weight"), HD);
            L.kn = up_f32(blk(i, "attn_k_norm.weight"), HD);
        } else {
            L.qkv = up_mat(blk(i, "attn_qkv.weight"), D, QKV);
            L.gate = up_mat(blk(i, "attn_gate.weight"), D, VD);
            L.sout = up_mat(blk(i, "ssm_out.weight"), VD, D);
            needq(blk(i, "attn_qkv.weight"), L.qkv.type);
            needq(blk(i, "attn_gate.weight"), L.gate.type);
            needq(blk(i, "ssm_out.weight"), L.sout.type);
            L.conv = up_f32(blk(i, "ssm_conv1d.weight"), 4 * QKV);
            L.dt = up_f32(blk(i, "ssm_dt.bias"), NVH);
            L.A = up_f32(blk(i, "ssm_a"), NVH);
            L.snorm = up_f32(blk(i, "ssm_norm.weight"), HK);
            const auto & raw_a = tget(blk(i, "ssm_alpha.weight"));
            const auto & raw_b = tget(blk(i, "ssm_beta.weight"));
            if (raw_a.type == gguf::Q8_0 && raw_b.type == gguf::Q8_0 && shape2(raw_a.ne, D, NVH) && shape2(raw_b.ne, D, NVH) &&
                !getenv("MK_AB_F32")) {
                // Q8_0 alpha / beta (UD) kept as int8 rows + fp16 block scales; the kernels form
                // (q * d) * attn_norm on the fly, the same fp32 values as the folded rows below (MK_AB_F32=1)
                HSet & S = hset();
                const size_t nq = (size_t) 2 * NVH * D, nd = (size_t) 2 * NVH * (D / 32);
                int8_t * q = reinterpret_cast<int8_t *>(S.q.get(nq));
                uint16_t * ds = reinterpret_cast<uint16_t *>(S.m.get(nd * 2));
                for (int w = 0; w < 2; w++) {
                    const uint8_t * src = (w ? raw_b : raw_a).data;
                    for (size_t blk_ = 0; blk_ < (size_t) NVH * (D / 32); blk_++) {
                        memcpy(&ds[(size_t) w * NVH * (D / 32) + blk_], src + blk_ * 34, 2);
                        memcpy(&q[((size_t) w * NVH * (D / 32) + blk_) * 32], src + blk_ * 34 + 2, 32);
                    }
                }
                L.abq = dput<int8_t>(q, nq, &S);
                L.abd = reinterpret_cast<const __half *>(dput<uint16_t>(ds, nd, &S));
                L.ab = nullptr;
            } else {
            // alpha/beta projections in fp32 with the attn_norm weight folded in
            const Src ta = want(blk(i, "ssm_alpha.weight"), T_F32);
            const Src tb = want(blk(i, "ssm_beta.weight"), T_F32);
            const auto & tn = tget(blk(i, "attn_norm.weight"));
            // read below as raw floats, so anything but F32 would be reinterpreted silently
            need(blk(i, "ssm_alpha.weight"), (int) ta.type, {T_F32});
            need(blk(i, "ssm_beta.weight"), (int) tb.type, {T_F32});
            if (!shape2(ta.ne, D, NVH) || !shape2(tb.ne, D, NVH))
                throw std::runtime_error(blk(i, "ssm_alpha/beta.weight") + " bad shape");
            HSet & S = hset();
            const size_t nab = (size_t) 2 * NVH * D;
            float * ab = reinterpret_cast<float *>(S.q.get(nab * 4));
            const float * fa = (const float *) ta.data;
            const float * fb = (const float *) tb.data;
            const float * fn = (const float *) tn.data;
            for (int h = 0; h < NVH; h++)
                for (int e = 0; e < D; e++) {
                    ab[(size_t) h * D + e] = fa[(size_t) h * D + e] * fn[e];
                    ab[(size_t) (NVH + h) * D + e] = fb[(size_t) h * D + e] * fn[e];
                }
            L.ab = dput<float>(ab, nab, &S);
            L.abq = nullptr;
            L.abd = nullptr;
            }
        }
        L.fg = up_mat(blk(i, "ffn_gate.weight"), D, FF);
        L.fu = up_mat(blk(i, "ffn_up.weight"), D, FF);
        wv_fd = true;
        L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D);
        wv_fd = false;
        if (!mtp && !L.fd.fm) throw std::runtime_error(blk(i, "ffn_down.weight") + ": the wide pass needs every ffn_down in the wv layout");
        needq(blk(i, "ffn_gate.weight"), L.fg.type);
        needq(blk(i, "ffn_up.weight"), L.fu.type);
        needq(blk(i, "ffn_down.weight"), L.fd.type);
    }
    mem_mark("weights: 64 layers + MTP");
    wv_lay = false;
    fprintf(stderr, "[mk] wv: %d main-pass matrices in the wv layout (%d on the device, the others re-laid-out on the host in %.1f s; "
                    "Q6_K everywhere (FM), ffn_down every type; elsewhere Q3_K / IQ3_S stay on dp4a), MAXT %d\n", wv_n, wv_dev_n, wv_s, MAXT);
    if (has_mtp) {
        req_q4 = true;
        hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D);
        req_q4 = false;
        needq(blk(NL, "nextn.eh_proj.weight"), hw.eh.type);
        hw.enorm = up_f32(blk(NL, "nextn.enorm.weight"), D);
        hw.hnorm = up_f32(blk(NL, "nextn.hnorm.weight"), D);
        hw.shnorm = up_f32(blk(NL, "nextn.shared_head_norm.weight"), D);
    }
    // Every matrix outside the wv layout must be of a type this build's dp4a dispatch keeps: main layers a hybrid type
    // (Q3_K / IQ3_S: wv.cuh gemv_pairs_dp), the MTP block (solo and batched passes: gemv_pairs_any,
    // resid_any, rows16_any) after its re-quantisation also Q4_K / IQ4_XS. The loader above guarantees it for every file
    // it accepts; anything else is refused here rather than mis-read.
    {
        auto dp4a_need = [&](const std::string & name, const Mat & m, bool mtp) {
            const int t = m.type;
            if (m.fm || t == T_Q3K || t == T_IQ3S || (mtp && (t == T_Q4K || t == T_IQ4X))) return;
            throw std::runtime_error(name + " is " + tname(t) + (mtp ? " in the MTP block" : " outside the wv layout") +
                                     ": this build's dp4a kernels do not read it");
        };
        for (int i = 0; i < (has_mtp ? NLA : NL); i++) {
            const LayerW & L = hw.L[i];
            auto chk = [&](const char * r, const Mat & m) { dp4a_need(blk(i, r), m, i == NL); };
            if (L.attn) {
                chk("attn_q.weight", L.q);
                chk("attn_k.weight", L.k);
                chk("attn_v.weight", L.v);
                chk("attn_output.weight", L.o);
            } else {
                chk("attn_qkv.weight", L.qkv);
                chk("attn_gate.weight", L.gate);
                chk("ssm_out.weight", L.sout);
            }
            chk("ffn_gate.weight", L.fg);
            chk("ffn_up.weight", L.fu);
            chk("ffn_down.weight", L.fd);
        }
        if (has_mtp) dp4a_need(blk(NL, "nextn.eh_proj.weight"), hw.eh, true);
    }
    hw.onorm = up_f32("output_norm.weight", D);
    hw.head = up_mat("output.weight", D, VOCAB);
    // Q8_0: some fine-tunes (~0.3 GB more than Q6_K); lm_head reads either type
    if (hw.head.type != T_Q6K && hw.head.type != T_Q8_0)
        throw std::runtime_error("output.weight is " + tname(hw.head.type) + ", the kernel needs Q6_K or Q8_0 (see README: the GGUF)");
    mem_mark("weights: eh, norms, lm_head");
    hw.head_d = hw.head;
    hw.draft_ids = nullptr;
    wrec_on = false;
    if (!draft_vocab.empty() && has_mtp) { // (no MTP block: no draft head)
        FILE * f = ggml_fopen(draft_vocab.c_str(), "rb"); // (a UTF-8 path on Windows too)
        if (!f) throw std::runtime_error("cannot open draft vocab " + draft_vocab);
        std::vector<int> ids;
        int id;
        while (fread(&id, 4, 1, f) == 1) {
            if (id < 0 || id >= VOCAB) throw std::runtime_error("bad id in draft vocab");
            ids.push_back(id);
        }
        fclose(f);
        if (ids.size() & 1) ids.push_back(ids.back());
        // room for DV_EXTRA more rows: tokens of the current sequence that are not in the list
        // are added on the fly (identifiers of pasted code, see dv_add)
        // the draft head reads lm_head rows through the id list (lm_head), so no copy of the rows is kept
        const int cap = (int) ids.size() + DV_EXTRA;
        int * did = dalloc<int>(cap, false);
        CK(cudaMemcpy(did, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        Mat m = hw.head;
        m.N = (int) ids.size();
        hw.head_d = m;
        hw.draft_ids = did;
        dv_ids = ids;
        dv_base = (int) ids.size();
        if (cudaHostAlloc(reinterpret_cast<void **>(&dv_pin), ((size_t) cap + 1) * 4, cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            dv_pin = nullptr; // (synchronous copies)
        }
        dv_in.assign(VOCAB, 0);
        for (int t : ids) dv_in[t] = 1;
        fprintf(stderr, "[mk] MTP draft head restricted to %d tokens (%s)\n", m.N, draft_vocab.c_str());
        if (draft_q4) { // MK_DRAFT_Q4: the rows' Q4_K copy (row r = lm_head row ids[r], from the file's Q6_K or Q8_0)
            auto _ = lc.scope("draft head Q4_K copy (MK_DRAFT_Q4)");
            const auto t0 = lclk::now();
            const auto & t = tget("output.weight");
            const size_t rs_from = ggml_row_size((ggml_type) t.type, D), rs_to = ggml_row_size(GGML_TYPE_Q4_K, D);
            if ((t.type != gguf::Q6_K && t.type != gguf::Q8_0) || rs_to != (size_t) DQ4_RB) throw std::runtime_error("MK_DRAFT_Q4: unexpected lm_head");
            std::vector<uint8_t> q(rs_to * ids.size());
            const auto * tr = ggml_get_type_traits((ggml_type) t.type);
            parallel_rows((int64_t) ids.size(), [&](int64_t r) {
                thread_local std::vector<float> f;
                f.resize((size_t) D);
                tr->to_float(t.data + (size_t) ids[r] * rs_from, f.data(), D);
                ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), q.data() + (size_t) r * rs_to, 0, 1, D, nullptr); // lossy by design
            });
            hw.dq4 = dput<uint8_t>(q.data(), q.size());
            own_keep.push_back(std::move(q)); // (until its upload is done)
            hw.dq4_n = dq4_rows = (int) ids.size();
            fprintf(stderr, "[mk] draft head: %d rows re-quantised to Q4_K in %.2f s (%.1f MiB; drafts only, MK_DRAFT_Q4=0: off)\n",
                    dq4_rows, lsec(t0), (double) rs_to * dq4_rows / 1048576.0);
        }
    } else if (draft_q4 && draft_q4_env) {
        fprintf(stderr, "[mk] MK_DRAFT_Q4 ignored: it needs a draft vocabulary (--draft-vocab)\n");
    }
    mem_mark("draft head copy");
    wrec_on = true;
    {
        const auto & t = tget("token_embd.weight");
        // the embedding lookups dequantise each type (embed_prologue, the MTP M0 phases, pk_embed*, pk_mtp_in)
        if (t.type != gguf::Q4_K && t.type != gguf::Q6_K && t.type != gguf::Q8_0)
            throw std::runtime_error("token_embd.weight is " + tname((int) t.type) + ", the engine reads Q4_K, Q6_K or Q8_0 (see README: the GGUF)");
        if (!shape2(t.ne, D, VOCAB)) throw std::runtime_error("token_embd.weight bad shape");
        hw.embd_t = (int) t.type; // (gguf::Q4_K / Q6_K / Q8_0 == T_Q4K / T_Q6K / T_Q8_0)
        // mapped pinned host memory: a cycle reads only a few 2.8 KB rows over PCIe, saves 0.7 GB of VRAM
        auto _ = lc.scope("token_embd (pinned table)");
        if (!embd_dev) {
            if (embd_th.t.joinable()) embd_th.t.join();
            else embd_alloc();
        }
        if (!embd_dev && embd_err == cudaSuccess) {
            // (write-combined: several threads, 64 MiB pieces)
            const size_t PC = (size_t) 64 << 20, np = (t.nbytes + PC - 1) / PC;
            parallel_rows((int64_t) np, [&](int64_t i) {
                const size_t o = (size_t) i * PC;
                memcpy(static_cast<uint8_t *>(embd_host) + o, t.data + o, std::min(PC, (size_t) t.nbytes - o));
            });
            void * d = nullptr;
            CK(cudaHostGetDevicePointer(&d, embd_host, 0));
            hw.embd = (const uint8_t *) d;
            fprintf(stderr, "[mk] token_embd in mapped host memory (%.2f GB, pinned in %.2f s)\n", t.nbytes / 1e9, embd_pin_s);
        } else {
            cudaGetLastError();
            embd_host = nullptr;
            hw.embd = dput<uint8_t>(t.data, t.nbytes);
            wbytes -= t.nbytes; // (not counted as weights)
        }
    }
    wrec_on = false;
    mem_mark("token_embd");
    wv_dev_done();
    CK(mkl::set_cW(&hw, sizeof(ModelW)));
    mem_mark("cW symbol (module load)");
    if (n_conv)
        fprintf(stderr, "[mk] %d tensors re-encoded in %.1f s (exact: Q3_K / IQ3_S -> Q6_K with MK_REENC3, IQ4_NL -> Q8_0 with MK_IQ4NL_Q8; ssm_alpha / beta -> F32)\n",
                n_conv, conv_s);
    if (n_nat)
        fprintf(stderr, "[mk] %d Q3_K / IQ3_S tensors repacked to their native layouts in %.1f s (exact, checked per row)\n", n_nat, nat_s);
    if (n_req)
        fprintf(stderr, "[mk] MTP block: %d matrices re-quantised to Q4_K in %.1f s (drafts only)\n", n_req, req_s);

    hb.x = dalloc<float>(RMAX * D);
    hb.xm = dalloc<float>(RMAX * D);
    hb.qn = qbuf(U_D);
    hb.qo = qbuf(U_VD);
    hb.qa = qbuf(U_FF);
    hb.qe = qbuf(U_EH);
    hb.z = dalloc<float>(RMAX * VD);
    hb.ab = dalloc<float>(AB_SPLIT * RMAX * 2 * NVH);
    hb.qbuf = dalloc<float>(RMAX * QG);
    hb.kbuf = dalloc<float>(RMAX * KVD);
    hb.vbuf = dalloc<float>(RMAX * KVD);
    hb.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
    hb.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
    mem_mark("decode act bufs, conv, kvgb");
    hb.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
    mem_mark("GDN recurrent state");
    // stockfix switches (bit-exact): K1 MK_K1, K2 MK_K2 (both default on)
    hb.k1 = !(getenv("MK_K1") && atoi(getenv("MK_K1")) == 0);
    hb.k2 = !(getenv("MK_K2") && atoi(getenv("MK_K2")) == 0);
    fprintf(stderr, "[mk] stockfix: K1 ab dealing %d, K2 combine %d (tb %d)\n", hb.k1, hb.k2, MK_K2_TB);
    // stock2 switches: K6 MK_K6 (bits: 2 dealing across blocks, exact; 4 8-row halves of the Q8_0
    // k / v tiles, exact), K8 MK_K8 (bits, prefill, exact: 1 pk_ab on a second stream, 2 pk_norm_quant reads x once,
    // 4 pk_dn_prep MK_K8_NT tokens per block); compile-time K7 (state layout), K6_LD (Q8_0 tile loads)
    hb.k6 = getenv("MK_K6") ? atoi(getenv("MK_K6")) : MK_K6_DEF;
    if (hb.k6 & 1)
        throw std::runtime_error("MK_K6=" + std::to_string(hb.k6) + ": bit 1 (the K-split of the Q8_0 k / v tiles) was removed; bits 2 and 4 remain (default 6)");
    k8 = getenv("MK_K8") ? atoi(getenv("MK_K8")) : MK_K8_DEF;
    if (const char * e = getenv("MK_K8_NT")) k8_nt = atoi(e) == 8 ? 8 : 4;
    fprintf(stderr, "[mk] stock2: K6 %d, K8 %d (nt %d)\n", hb.k6, k8, k8_nt);
    {
        // [name, default, largest]: past what a matrix holds the loops stop at its end, so the largest values mean "all"
        struct L2Env { const char * name; int * v; int def, hi; };
        L2Tune l2{};
        const L2Env l2env[] = {
            {"MK_L2_DN", &l2.dn, DN_PF_ITEM, 65536},
            {"MK_L2_ATTN", &l2.attn, ATT_PFO_L, 256},
            {"MK_L2_ATTN_CTX", &l2.attn_ctx, 2 * ATT_BUCKET, 1 << 30},
            {"MK_L2_TOP", &l2.top, 1, 20},
            {"MK_L2_CK", &l2.ck, MK_CK_PFL, 32},
            {"MK_L2_MTP", &l2.mtp, MTP_PFL, 64},
            {"MK_L2_MTP_R", &l2.mtpr, MTP_PFR, 128},
            {"MK_L2_MTP_KEEP", &l2.keep, 0, 1},
        };
#if MK_L2_TUNE
        std::string l2s;
        for (const L2Env & e : l2env) {
            *e.v = e.def;
            if (const char * s = getenv(e.name); s && *s) {
                char * end;
                const long x = strtol(s, &end, 10);
                if (*end || x < 0 || x > e.hi)
                    throw std::runtime_error(std::string(e.name) + "=" + s + ": an integer from 0 to " + std::to_string(e.hi));
                *e.v = (int) x;
            }
            l2s += std::string(" ") + e.name + "=" + std::to_string(*e.v);
        }
        CK(mkl::set_cL2(&l2, sizeof(l2)));
        fprintf(stderr, "[mk] MK_L2_TUNE:%s\n", l2s.c_str());
#else
        for (const L2Env & e : l2env)
            if (getenv(e.name)) fprintf(stderr, "[mk] %s ignored: the L2 budgets are constants in this build (-DMK_L2_TUNE=1: make MKFLAGS=..., build.ps1 -MkFlags ...)\n", e.name);
#endif
    }
    fprintf(stderr, "[mk] K5: split dn_proj -> dn_core barrier, delta-net heads in dn_proj arrival order\n");
    fprintf(stderr, "[mk] C1: ffn_gu -> ffn_down per-chunk waits instead of a grid barrier\n");
    CK(mkl::wv_deal_init(nsm)); // (MK_FFNA builds; every mk_cycle launch that runs a wv phase has nsm blocks)
    hb.ctx = ctx + KV_PAD; // rows per kv head (KV_PAD: the MTP draft rows written past the last admitted position)
    // the device buffers after the KV cache (MK_RESERVE_EARLY: before it, so that its fit check sees them)
    auto dev_bufs = [&] {
        hb.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
        hb.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
#if MK_MAXD > 8
        if (n_draft_cap > MAXT - 1) {
            apart_hi = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
            aml_hi = dalloc<float2>((size_t) NKV * MAX_CH * AR);
        }
#endif
        hb.h_hist = dalloc<float>((size_t) H_RING * D);
        mem_mark("attn partials, h ring");
        {
            std::vector<float2> rt((size_t) hb.ctx * 32); // (rows past ctx: the MTP draft rows, KV_PAD)
            for (int p = 0; p < hb.ctx; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            float2 * d = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(d, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
            hb.rope = d;
            rope_base = std::move(rt); // (images: rope_sync gathers rewritten rows from it)
        }
        mem_mark("RoPE table");
        hb.hm = dalloc<float>(D);
        hb.blk_best = dalloc<float2>((size_t) nsm * LM_RB * TOPK_MAX);
        // lm_head<16, true>'s logit lines, line k * nwarps + warp for a warp's k-th row pair (~16 MB; builds with a 16-row head)
        hb.lmscr = (MAXD > MAXT || RMAX > MAXT) ? dalloc<float>((size_t) (VOCAB / 2 + nsm * NW) * 32) : nullptr;
        hb.blk_lse = dalloc<float2>((size_t) nsm);
        hb.dlog = dalloc<float2>((size_t) MAXD * (TOPK_MAX + 1));
        hb.dq = dalloc<float2>((size_t) MAXD * TOPK_MAX);
        hb.cand = dalloc<float2>((size_t) RMAX * TOPK_MAX);
        hb.logits = nullptr; // 5 MB used by scoring launches only (score_logits): logits_alloc()
        hb.pcnt = nullptr;   // 0.5 MB for the presence / frequency penalty, allocated by the first set_penalty
        hb.bar = dalloc<unsigned>(C1_WORDS); // count word | go word | C1's per-chunk counters, each on its own line
        hb.cnt = dalloc<unsigned>(4);
        hb.ctl = dalloc<Ctl>(1);
        hb.wvpart = dalloc<float>((size_t) WV_S * RMAX * D);
        hb.wvcnt = dalloc<unsigned>(2 * (D / 32), DA_INIT0); // (starts at 0; the last arrival of each use resets it)
        profiling = getenv("MK_PROFILE") != nullptr;
        hb.prof = profiling ? dalloc<unsigned long long>(4096 + 1024 * 128) : nullptr;
        mem_mark("logits, sampling, ctl");
    };
    size_t free_pre = 0; // (MK_RESERVE_EARLY: the free memory where the default order sizes the cache, for its auto type)
    if (reserve_early) {
        size_t t = 0;
        CK(cudaMemGetInfo(&free_pre, &t));
        // lazy module loading: a kernel's code loads at its first use; mk_cycle's and the prefill kernels' now
        CK(cudaFuncSetAttribute(mkl::k_mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
        pf_kernel_attrs();
        mkl::pk_attn_attr();
        if (getenv("MK_EXACT") ? atoi(getenv("MK_EXACT")) != 0 : exact) mkl::pk_attn8_exa_attr();
        mem_mark("kernel loads (early)");
        dev_bufs();
    }
    {
        const size_t rows = (size_t) kv_slots * NKV * hb.ctx; // (no MTP block: no slot N_ATT_SLOTS - 1)
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        // fp16 needs 2 x rows x HD x 2 bytes; leave ~1.5 GB for the prefill buffers, checkpoint and runtime.
        // A non-empty MK_KV picks the quantized cache in auto mode too.
        const char * kv_spec = getenv("MK_KV");
        const bool want_kv = kv_spec && kv_spec[0];
        const bool q8 = kv_mode == 1 || (kv_mode < 0 && (want_kv || rows * HD * 4 + (size_t) 1536 * 1024 * 1024 > (reserve_early ? free_pre : free_b)));
        hb.kvq8 = q8;
        hb.dattn = !(getenv("MK_DATTN") && atoi(getenv("MK_DATTN")) == 0);
        // MK_ATT_I8=1 (q8 cache, -DMK_ATT_I8=1 builds): int8 Q.K^T in the decode attention of the all-q8 slots, in every
        // kernel (aq_i8_s): exact across draft depths and lanes, not bit-identical to the default fp16 Q (KLD-gated). (2: the
        // same; it selected early K / V copies, which measured slower and were removed)
        {
            const int ai8 = getenv("MK_ATT_I8") ? atoi(getenv("MK_ATT_I8")) : 0;
#if MK_ATT_I8
            hb.att_i8 = q8 && (ai8 == 1 || ai8 == 2) ? 1 : 0;
            if (hb.att_i8)
                fprintf(stderr, "[mk] MK_ATT_I8=%d: int8 Q.K^T in the decode attention of all-q8 slots (every kernel and draft depth; not "
                                "bit-identical to fp16 Q)\n", ai8);
            else if (ai8)
                fprintf(stderr, "[mk] MK_ATT_I8=%d ignored (%s)\n", ai8, q8 ? "values: 1, 2" : "fp16 KV cache");
#else
            if (ai8) fprintf(stderr, "[mk] MK_ATT_I8=%d ignored: int8 attention is not in this build (-DMK_ATT_I8=1: make MKFLAGS=..., build.ps1 -MkFlags ...)\n", ai8);
#endif
        }
        // MK_MTP_WIN=N: the MTP block's decode attention sees the last N positions only (its prefill attention and
        // the KV layout are unchanged). Drafts are verified, so the output is exact; acceptance can change.
        hb.mtp_win = getenv("MK_MTP_WIN") ? atoi(getenv("MK_MTP_WIN")) : 0;
        if (hb.mtp_win > 0) fprintf(stderr, "[mk] MTP decode attention window %d positions (MK_MTP_WIN)\n", hb.mtp_win);
        // the catch-up rows before the last 5 run K / V only (same drafts; not in MK_KV_EMU builds); env MK_MTP_KVC=0: off
        hb.mtp_kvc = !(getenv("MK_MTP_KVC") && atoi(getenv("MK_MTP_KVC")) == 0);
        // MK_KV (q8 cache): per attention slot and K / V, q8_0 (default), q4 (kv_store_q4) or q5 / q6 / q7
        // (kv_store_qn), e.g. "V*=6r,K16=4r": comma-separated <K|V|*><slot>=<4..8>[r], slot 0..15 = attention layer
        // 4 slot + 3, 16 = MTP block, * or empty = all slots; r: stored Hadamard-rotated (K: the queries get the same
        // rotation; V: the head's attention output is rotated back)
        const size_t srows = (size_t) NKV * hb.ctx; // rows of one slot
        for (int s = 0; s < N_ATT_SLOTS; s++) {
            hb.kvb[0][s] = hb.kvb[1][s] = 8;
            hb.kvr[s] = hb.kvrv[s] = 0;
        }
        if (want_kv) {
            if (!q8) throw std::runtime_error("MK_KV needs the q8 KV cache (--kv q8 or auto, not f16)");
            const std::string str = kv_spec;
            size_t i = 0;
            while (i < str.size()) {
                size_t j = str.find(',', i);
                if (j == std::string::npos) j = str.size();
                const std::string tok = str.substr(i, j - i);
                i = j + 1;
                if (tok.empty()) continue;
                const size_t e = tok.find('=');
                if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV: bad item " + tok);
                const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
                if (md.empty() || md.size() > 2 || md[0] < '4' || md[0] > '8' || (md.size() == 2 && md[1] != 'r'))
                    throw std::runtime_error("MK_KV: expected 4..8 or 4r..8r in " + tok);
                const bool rot = md.size() == 2;
                const bool all = sl.empty() || sl == "*";
                char * end = nullptr;
                const long slot = all ? -1 : strtol(sl.c_str(), &end, 10);
                if (!all && (*end || slot < 0 || slot >= N_ATT_SLOTS))
                    throw std::runtime_error("MK_KV: bad slot in " + tok + " (0.." + std::to_string(N_ATT_SLOTS - 1) + " or *)");
                for (int w = 0; w < 2; w++)
                    if (tok[0] == '*' || tok[0] == "KV"[w])
                        for (int s = 0; s < N_ATT_SLOTS; s++)
                            if (all || s == slot) {
                                hb.kvb[w][s] = md[0] - '0';
                                (w == 0 ? hb.kvr : hb.kvrv)[s] = rot;
                            }
            }
        }
        for (int s = kv_slots; s < N_ATT_SLOTS; s++) { // (not allocated)
            hb.kvb[0][s] = hb.kvb[1][s] = 8;
            hb.kvr[s] = hb.kvrv[s] = 0;
        }
        size_t vbytes[2] = {0, 0};
        int n4 = 0;
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < kv_slots; s++) {
                vbytes[w] += srows * HD * hb.kvb[w][s] / 8;
                n4 += hb.kvb[w][s] != 8 || (w == 0 ? hb.kvr[s] : hb.kvrv[s]);
            }
        size_t fit_b = 0; // (MK_RESERVE_EARLY: the cache's device bytes, rings included, as its fit check counts them)
        if (reserve_early) {
            // MK_RESERVE_EARLY: the cache, its MK_KV_WIN rings and what is allocated after them (the prefill scratch, a device
            // checkpoint) plus MK_VRAM_MARGIN MiB for what loads later (kernels at the first prefill / launch, the driver)
            // against the device memory left now: under it, refuse with the -c that fits (MK_VRAM_OVERCOMMIT=1: warn, load)
            size_t ring_b = 0, rc = 0;
            if (want_kv && getenv("MK_KV_WIN") && atoi(getenv("MK_KV_WIN")) > 0) {
                const int win = atoi(getenv("MK_KV_WIN"));
                rc = (size_t) ((win + ATT_BUCKET - 1) / ATT_BUCKET + 1) * ATT_BUCKET + ATP; // (hb.kvrc below)
                const bool all = getenv("MK_KV_WIN_ALL") && atoi(getenv("MK_KV_WIN_ALL")) != 0;
                for (int w = 0; w < 2; w++)
                    for (int s = 0; s < kv_slots; s++)
                        if (hb.kvb[w][s] != 8 || all) ring_b += (size_t) NKV * rc * (HD + HD / 16) + 32;
            }
            // bytes per position: the cache's rows (every slot's K and V, the q8 scales) and the RoPE table's (32 float2)
            const size_t row_b = (q8 ? (vbytes[0] + vbytes[1]) / hb.ctx + (size_t) kv_slots * NKV * (HD / 32) * 4
                                     : (size_t) kv_slots * NKV * HD * 4) + 32 * sizeof(float2);
            fit_b = (q8 ? vbytes[0] + vbytes[1] + (size_t) 2 * rows * (HD / 32) * 2 + 64 : rows * HD * 4 + 32) + ring_b;
            const bool ex = getenv("MK_EXACT") ? atoi(getenv("MK_EXACT")) != 0 : exact;
            const char * cd = getenv("MK_CK_DEVICE");
            // (exact checkpoints keep the rings when there are any: ck_pool's ring_bytes_ck > 0, which pf_init's cap reads)
            const size_t scr = pf_scratch_bytes(pf_cap_eff(ex && ring_b > 0, (int) rc), ex);
            const size_t post = scr + (cd && cd[0] == '1' ? ck_block_bytes() + 16 : 0);
            size_t fr = 0, tt = 0;
            CK(cudaDeviceSynchronize());
            CK(cudaMemGetInfo(&fr, &tt));
            unsigned long long res = 0, used = 0; // (the pool's reserved but unused bytes: the next allocations take them first)
            if (use_pool) {
                cudaMemPool_t pool;
                CK(cudaDeviceGetDefaultMemPool(&pool, cuda_dev));
                CK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res));
                CK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
            }
            const double MiB = 1048576.0;
            const char * mg = getenv("MK_VRAM_MARGIN");
            // (384: the pool reserves ~100 MiB over the cache's bytes, and the free figure moves by up to ~150 MiB between processes
            // with the desktop's use; D at the -c a 256 MiB margin printed dropped to 4.7 MiB free in one of four processes)
            const long long margin = (long long) ((mg ? std::max(0.0, atof(mg)) : 384.0) * MiB);
            const long long avail = (long long) fr + (long long) (res > used ? res - used : 0);
            const long long left = avail - (long long) (fit_b + post);
            fprintf(stderr, "[mk] KV fit (MK_RESERVE_EARLY): %.1f MiB free (+ %.1f in the pool), the cache %.1f MiB (rings %.1f), after it "
                            "%.1f (prefill scratch %.1f): %.1f MiB left, margin %.1f (MK_VRAM_MARGIN)\n",
                    fr / MiB, (avail - (long long) fr) / MiB, fit_b / MiB, ring_b / MiB, post / MiB, scr / MiB, left / MiB, margin / MiB);
            if (left < margin) {
                // the -c whose cache and RoPE table leave the margin (the table is allocated already, at this -c)
                const long long room = avail + (long long) ((size_t) hb.ctx * 32 * sizeof(float2)) - (long long) (ring_b + post + 64) - margin;
                const long long cf = std::max(0LL, (room > 0 ? room / (long long) row_b : 0) - KV_PAD) / 1024 * 1024;
                const std::string fits = cf >= 1024 ? "-c " + std::to_string(cf) + " fits" : "no context fits next to the weights";
                char msg[1024];
                snprintf(msg, sizeof msg,
                         "%.0f MiB of device memory are free for the KV cache of %d positions (%.0f MiB) and the prefill scratch (%.0f MiB): "
                         "%.0f MiB short of the %.0f MiB margin for what loads later (MK_VRAM_MARGIN). The CUDA sysmem fallback (Windows, "
                         "WSL2) would put the overflow in system memory (slower decode, varying per process); other drivers fail later. "
                         "%s (MK_VRAM_OVERCOMMIT=1: load anyway)",
                         avail / MiB, ctx, fit_b / MiB, post / MiB, (margin - left) / MiB, margin / MiB, fits.c_str());
                if (!(getenv("MK_VRAM_OVERCOMMIT") && atoi(getenv("MK_VRAM_OVERCOMMIT")) != 0)) throw std::runtime_error(msg);
                fprintf(stderr, "[mk] warning: %s\n", msg);
            }
        }
        const size_t ab0 = alloc_bytes; // (alloc_bytes - ab0: the cache's device bytes, rings included)
        // a driver without the CUDA sysmem fallback (Linux) fails the allocation where others get the warning below
        const double kv_gb = q8 ? (vbytes[0] + vbytes[1] + 2.0 * rows * (HD / 32) * 2) / 1e9 : rows * HD * 4.0 / 1e9;
        const auto kv_oom = [&](const MKCudaError & e, const char * rings) {
            char msg[1280];
            snprintf(msg, sizeof msg, "the KV cache (%.2f GB for %d positions%s) does not fit in the free device memory (%.2f GB): %s fits (%s)",
                     kv_gb, ctx, rings, free_b / 1e9,
                     q8 ? "a smaller context (-c) or a low-bit preset (MK_KV, docs/CONFIGURATION.md)" : "a smaller context (-c) or --kv q8",
                     e.what());
            return MKCudaError(msg);
        };
        try {
            hb.kc = q8 ? (void *) dalloc<int8_t>(vbytes[0]) : (void *) dalloc<__half>(rows * HD);
            hb.vc = q8 ? (void *) dalloc<int8_t>(vbytes[1]) : (void *) dalloc<__half>(rows * HD);
            hb.ksc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            hb.vsc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
        } catch (const MKCudaError & e) {
            throw kv_oom(e, "");
        }
        size_t off[2] = {0, 0};
        for (int s = 0; s < N_ATT_SLOTS; s++)
            for (int w = 0; w < 2; w++) { // slots packed in order: all-q8 gives the same addresses as [slot][head][ctx][HD]
                if (s >= kv_slots) {
                    hb.kvp[w][s] = nullptr;
                    hb.kvs[w][s] = nullptr;
                    continue;
                }
                hb.kvp[w][s] = static_cast<int8_t *>(w ? hb.vc : hb.kc) + off[w];
                hb.kvs[w][s] = q8 ? (w ? hb.vsc : hb.ksc) + s * srows * (HD / 32) : nullptr;
                off[w] += srows * HD * hb.kvb[w][s] / 8;
            }
        // MK_KV_WIN=W: every low-bit side keeps its last >= W positions as q8 in a ring (kv_ws / kv_src; the ring holds
        // (ceil(W / ATT_BUCKET) + 1) ATT_BUCKET rows per kv head so a bucket-aligned window always fits)
        hb.kvwin = 0;
        hb.kvrc = 0;
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < N_ATT_SLOTS; s++) {
                hb.kvrp[w][s] = nullptr;
                hb.kvrs[w][s] = nullptr;
            }
        size_t ring_bytes = 0;
        if (want_kv && getenv("MK_KV_WIN") && atoi(getenv("MK_KV_WIN")) > 0) {
            hb.kvwin = atoi(getenv("MK_KV_WIN"));
            // + ATP slack rows: rows written past the attention partition of the launch that reads the window (the MTP
            // block's draft rows and catch-up row, up to n_draft past it, the wide pass at -d 7 included) then land in the
            // ring slots of positions before the window, not of its oldest rows (multiple of ATP: a tile never wraps)
            const int kvwr = ((hb.kvwin + ATT_BUCKET - 1) / ATT_BUCKET + 1) * ATT_BUCKET; // the most rows a window spans
            static_assert(MAXD + MAXD <= ATP, "ring slack");
            hb.kvrc = kvwr + ATP;
            // MK_KV_WIN_ALL=1 (test knob): rings for the q8 sides too. Their ring rows are the same q8 rows as the main
            // buffer (and k_ring_refill re-stores a q8 row exactly), so output must stay byte-identical to no window:
            // exercises the ring write / read / refill paths with an exact expectation.
            const bool win_all = getenv("MK_KV_WIN_ALL") && atoi(getenv("MK_KV_WIN_ALL")) != 0;
            int nr = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < kv_slots; s++)
                    if (hb.kvb[w][s] != 8 || win_all) {
                        try {
                            hb.kvrp[w][s] = dalloc<int8_t>((size_t) NKV * hb.kvrc * HD);
                            hb.kvrs[w][s] = dalloc<__half>((size_t) NKV * hb.kvrc * (HD / 32));
                        } catch (const MKCudaError & e) {
                            throw kv_oom(e, " and its MK_KV_WIN rings");
                        }
                        ring_bytes += (size_t) NKV * hb.kvrc * (HD + HD / 16);
                        nr++;
                    }
            fprintf(stderr, "[mk] KV recent window: the last %d..%d positions of %d low-bit sides stay q8 (MK_KV_WIN=%d; rings of %d rows per head, %.1f MiB)\n",
                    kvwr - ATT_BUCKET + 1, kvwr, nr, hb.kvwin, hb.kvrc, ring_bytes / 1048576.0);
        }
        const double gb = q8 ? (vbytes[0] + vbytes[1] + 2.0 * rows * (HD / 32) * 2 + ring_bytes) / 1e9 : rows * HD * 4.0 / 1e9;
        if (reserve_early && alloc_bytes - ab0 != fit_b)
            fprintf(stderr, "[mk] warning: MK_RESERVE_EARLY's fit check counted %zu bytes for the KV cache, it took %zu: update the check\n",
                    fit_b, alloc_bytes - ab0);
        fprintf(stderr, "[mk] KV cache %s, %.2f GB for %d positions\n", q8 ? (n4 ? "q8_0 + q4 (MK_KV)" : "q8_0") : "fp16", gb, ctx);
        // a sysmem-fallback driver puts a cache past the free memory in system memory instead of failing
        if (alloc_bytes - ab0 > free_b)
            fprintf(stderr,
                    "[mk] warning: the KV cache takes %.2f GB of device memory, %.2f GB was free: with the CUDA "
                    "sysmem fallback (Windows, WSL2) the rest sits in system memory, read over PCIe (much slower; "
                    "a smaller context fits)\n",
                    (alloc_bytes - ab0) / 1e9, free_b / 1e9);
        if (n4) {
            std::string k, v, kr, vr;
            for (int s = 0; s < N_ATT_SLOTS; s++) {
                k += (char) ('0' + hb.kvb[0][s]);
                v += (char) ('0' + hb.kvb[1][s]);
                kr += hb.kvr[s] ? 'r' : '.';
                vr += hb.kvrv[s] ? 'r' : '.';
            }
            fprintf(stderr, "[mk] MK_KV bits per slot (layers 3, 7, .., 63, MTP): K %s  V %s; rotated: K %s  V %s\n", k.c_str(), v.c_str(), kr.c_str(), vr.c_str());
        }
    }
    mem_mark("KV cache");
    if (!reserve_early) dev_bufs();
    if (const char * spec = getenv("MK_KV_EMU_SPEC")) kv_emu_parse(spec);
    {
        int * eh = nullptr;
        CK(cudaHostAlloc(reinterpret_cast<void **>(&eh), 2 * sizeof(int), cudaHostAllocMapped));
        eh[0] = eh[1] = 0;
        err_host = eh;
        CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&hb.err), eh, 0));
    }
    {
        Ctl * ce = nullptr;
        CK(cudaHostAlloc(reinterpret_cast<void **>(&ce), sizeof(Ctl), cudaHostAllocMapped));
        memset(ce, 0, sizeof(Ctl));
        ctl_eg_host = ce;
        CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&hb.ctl_eg), ce, 0));
    }
    CK(mkl::set_cBs(&hb, sizeof(Bufs)));

    CK(cudaFuncSetAttribute(mkl::k_mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mkl::k_mk_cycle, NT, SM_TOTAL));
    if (nb < 1) throw std::runtime_error("megakernel does not fit on an SM");
    // The driver reserves local memory for every thread slot of the device (1,536 per SM x 82 SMs), sized by
    // the larger of the stack-size limit (default 1 KB per thread = 123 MiB) and the launched kernels' own
    // (static) frames. No kernel here needs a dynamic stack, so the limit only adds a floor: drop it. (MK_RESERVE_EARLY: set to
    // mk_cycle's frame at the context and kept.)
    if (!reserve_early) {
        const char * sl = getenv("MK_STACK_LIMIT");
        CK(cudaDeviceSetLimit(cudaLimitStackSize, sl ? (size_t) atol(sl) : 0));
    }
    if (memlog) {
        mem_mark("mk_cycle attributes");
        cudaFuncAttributes fa{};
        CK(cudaFuncGetAttributes(&fa, mkl::k_mk_cycle));
        size_t stk = 0, heap = 0, fifo = 0;
        CK(cudaDeviceGetLimit(&stk, cudaLimitStackSize));
        CK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
        CK(cudaDeviceGetLimit(&fifo, cudaLimitPrintfFifoSize));
        fprintf(stderr, "[mk-mem] mk_cycle: %d regs, local %zu B/thread, const %zu B, static smem %zu B; limits: stack %zu B/thread, "
                        "malloc heap %zu B, printf fifo %zu B; %d SMs x %d threads/SM\n",
                fa.numRegs, (size_t) fa.localSizeBytes, (size_t) fa.constSizeBytes, (size_t) fa.sharedSizeBytes, stk, heap, fifo, nsm,
                prop.maxThreadsPerMultiProcessor);
    }
    // the DeltaNet checkpoint and the batched-prefill scratch are allocated here, not inside the first prompt's time
    {
        auto _ = lc.scope("checkpoint pool (pinned host)");
        ck_alloc();
    }
    {
        auto _ = lc.scope("prefill scratch");
        const size_t ab = alloc_bytes;
        pf_init();
        if (reserve_early && alloc_bytes - ab != pf_scratch_bytes(hp.cap, exact))
            fprintf(stderr, "[mk] warning: MK_RESERVE_EARLY's fit check counted %zu bytes for the prefill scratch, pf_init took %zu: update "
                            "pf_scratch_bytes\n", pf_scratch_bytes(hp.cap, exact), alloc_bytes - ab);
    }
    ck_make_plan();
    if (exact && stg_n < 2)
        fprintf(stderr, "[mk] warning: MK_EXACT with MK_PF_CAP %d: no staging slots for checkpoints inside a prefill, so "
                        "none are saved (a reuse prefills from position 0); MK_PF_CAP >= 384 keeps them\n", hp.cap);
    {
        auto _ = lc.scope("upload tail (wait)");
        upl.finish();
    }
    own_keep.clear();
    own_keep.shrink_to_fit();
    for (HSet & S : hs) S = HSet{};
    const double pf_wait = pfx ? pfx->wait_s : 0.0;
    const size_t pf_bytes = pfx ? pfx->bytes() : 0;
    pfx.reset();
    load_report(pf_wait, pf_bytes);
    if (getenv("MK_WDIGEST")) weights_digest();
    if (reserve_early) vram_check("load end");
    fprintf(stderr, "[mk] weights %.2f GB on device, %d SMs, ctx %d, n_draft %d\n", wbytes / 1e9, nsm, ctx, ndraft);
}

void MKEngine::Impl::load_report(double pf_wait, size_t pf_bytes) {
    if (rq_th.t.joinable()) rq_th.t.join(); // (rq_s is the thread's until it ends)
    const double tot = lsec(t_load);
    double known = pf_wait;
    double rep = 0;
    std::string rest;
    for (const auto & p : lc.ph) {
        known += p.second;
        if (p.first.compare(0, 6, "repack") == 0) rep += p.second;
    }
    auto g = [&](const char * k) { return lc.get(k); };
    fprintf(stderr, "[mk] load %.2f s: context %.2f, read-ahead wait %.2f (%.1f GB read), repack %.2f, wv re-layout %.2f, "
                    "MTP requant wait %.2f (background %.2f), token_embd %.2f, ckpt pool %.2f, prefill scratch %.2f, upload tail %.2f, "
                    "other %.2f (device allocs %.2f of it), upload stalls %.2f | uploads %.2f GB, %.2f s in copies (%s)\n",
            tot, g("CUDA context"), pf_wait, pf_bytes / 1e9, rep, g("wv re-layout"), g("MTP Q4_K requant (wait for the background thread)"),
            rq_s, g("token_embd (pinned table)"), g("checkpoint pool (pinned host)"), g("prefill scratch"), g("upload tail (wait)"),
            tot - known, dalloc_s, g("upload stall (host buffer set busy)"), upl.bytes / 1e9, upl.busy_s, getenv("MK_LOAD_ASYNC") && atoi(getenv("MK_LOAD_ASYNC")) == 0 ? "synchronous" : "worker thread");
    if (getenv("MK_LOADLOG"))
        for (const auto & p : lc.ph) fprintf(stderr, "[mk-load] %-52s %7.3f s\n", p.first.c_str(), p.second);
}

void MKEngine::Impl::weights_digest() {
    CK(cudaDeviceSynchronize());
    const auto t0 = lclk::now();
    const size_t PC = (size_t) 64 << 20, NB = 32;
    struct Pc { const uint8_t * p; size_t n; bool host; };
    std::vector<Pc> pcs;
    size_t tot = 0;
    for (const auto & r : wrec) {
        for (size_t o = 0; o < r.second; o += PC) pcs.push_back({static_cast<const uint8_t *>(r.first) + o, std::min(PC, r.second - o), false});
        tot += r.second;
    }
    if (embd_host) {
        const size_t n = file.get("token_embd.weight").nbytes;
        for (size_t o = 0; o < n; o += PC) pcs.push_back({static_cast<const uint8_t *>(embd_host) + o, std::min(PC, n - o), true});
        tot += n;
    }
    std::vector<unsigned long long> ph(pcs.size());
    std::unique_ptr<uint8_t[]> buf(new uint8_t[NB * PC]);
    for (size_t b = 0; b < pcs.size(); b += NB) {
        const size_t k = std::min(NB, pcs.size() - b);
        for (size_t j = 0; j < k; j++)
            if (!pcs[b + j].host) CK(cudaMemcpy(buf.get() + j * PC, pcs[b + j].p, pcs[b + j].n, cudaMemcpyDeviceToHost));
        parallel_rows((int64_t) k, [&](int64_t j) {
            const Pc & q = pcs[b + j];
            if (q.host) memcpy(buf.get() + j * PC, q.p, q.n); // (write-combined memory: copy first)
            ph[b + j] = wd_hash(buf.get() + j * PC, q.n);
        });
    }
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (const auto & r : wrec) h = wd_hash(reinterpret_cast<const uint8_t *>(&r.second), sizeof(size_t), h);
    h = wd_hash(reinterpret_cast<const uint8_t *>(ph.data()), ph.size() * 8, h);
    fprintf(stderr, "[mk] weights digest %016llx (%zu buffers + token_embd, %.2f GB, %.1f s)\n", h, wrec.size(), tot / 1e9, lsec(t0));
}

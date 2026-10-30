// MK_KV / MK_KV_WIN parsing (host), shared by the engines. See kv_lowbit.cuh for the formats.
#pragma once
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// Per attention slot (layers 3, 7, .., 63 = slots 0..15, MTP block = 16) and K (0) / V (1): bits 4..8 (8 = q8_0) and
// whether the side is stored Hadamard-rotated. win = MK_KV_WIN (0: no recent-window ring). low = MK_KV was given.
template <int NS>
struct KvSpec {
    int bits[2][NS];
    int rot[2][NS];
    int win = 0;
    bool low = false;
    bool win_all = false; // MK_KV_WIN_ALL=1 (test knob): rings for the q8 sides too, output must equal no window
};

// MK_KV: comma-separated <K|V|*><slot>=<4..8>[r], e.g. "K*=5r,V*=5": slot 0..NS-1, * or empty = all slots; r: stored
// Hadamard-rotated (K: the queries get the same rotation; V: the head's attention output is rotated back)
template <int NS>
KvSpec<NS> kv_spec_from_env() {
    KvSpec<NS> k;
    for (int w = 0; w < 2; w++)
        for (int s = 0; s < NS; s++) {
            k.bits[w][s] = 8;
            k.rot[w][s] = 0;
        }
    const char * spec = getenv("MK_KV");
    if (!spec || !spec[0]) return k;
    k.low = true;
    const std::string str = spec;
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
        const bool all = sl.empty() || sl == "*";
        int slot = -1;
        if (!all) {
            if (sl.find_first_not_of("0123456789") != std::string::npos || sl.size() > 3) throw std::runtime_error("MK_KV: bad slot in " + tok);
            slot = atoi(sl.c_str());
            if (slot >= NS) throw std::runtime_error("MK_KV: bad slot in " + tok);
        }
        for (int w = 0; w < 2; w++)
            if (tok[0] == '*' || tok[0] == "KV"[w])
                for (int s = 0; s < NS; s++)
                    if (all || s == slot) {
                        k.bits[w][s] = md[0] - '0';
                        k.rot[w][s] = md.size() == 2;
                    }
    }
    if (const char * e = getenv("MK_KV_WIN")) k.win = atoi(e) > 0 ? atoi(e) : 0;
    k.win_all = getenv("MK_KV_WIN_ALL") && atoi(getenv("MK_KV_WIN_ALL")) != 0;
    return k;
}

// The engine's own (single-stream) cache in the MK_KV formats: slots packed one after the other in kc / vc (values of
// [kv head][ctx][hd bits / 8] each), scales in ksc / vsc as for q8, plus the MK_KV_WIN q8 rings. srows = rows of one
// slot (kv heads x ctx). alloc(bytes) returns zeroed device memory. Returns the bytes allocated.
template <int NS, class B, class Alloc>
size_t kv_alloc_lowbit(B & hb, const KvSpec<NS> & k, size_t srows, int hd, int nkv, int bucket, int atp, Alloc && alloc) {
    size_t vbytes[2] = {0, 0};
    for (int w = 0; w < 2; w++)
        for (int s = 0; s < NS; s++) vbytes[w] += srows * hd * k.bits[w][s] / 8;
    const size_t sbytes = (size_t) NS * srows * (hd / 32) * sizeof(__half);
    hb.kc = alloc(vbytes[0]);
    hb.vc = alloc(vbytes[1]);
    hb.ksc = static_cast<__half *>(alloc(sbytes));
    hb.vsc = static_cast<__half *>(alloc(sbytes));
    size_t total = vbytes[0] + vbytes[1] + 2 * sbytes, off[2] = {0, 0};
    for (int s = 0; s < NS; s++) {
        hb.kvr[s] = k.rot[0][s];
        hb.kvrv[s] = k.rot[1][s];
        for (int w = 0; w < 2; w++) {
            hb.kvb[w][s] = k.bits[w][s];
            hb.kvp[w][s] = static_cast<int8_t *>(w ? hb.vc : hb.kc) + off[w];
            hb.kvs[w][s] = (w ? hb.vsc : hb.ksc) + s * srows * (hd / 32);
            off[w] += srows * hd * k.bits[w][s] / 8;
            hb.kvrp[w][s] = nullptr;
            hb.kvrs[w][s] = nullptr;
        }
    }
    hb.kvwin = 0;
    hb.kvrc = 0;
    if (k.win > 0) {
        // + atp slack rows: rows written past the attention partition of the launch that reads the window (the MTP
        // block's draft and catch-up rows) land in the ring slots of positions before the window, not of its oldest rows
        const int span = ((k.win + bucket - 1) / bucket + 1) * bucket; // the most rows a window covers
        hb.kvwin = k.win;
        hb.kvrc = span + atp;
        size_t ring = 0;
        int nr = 0;
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < NS; s++)
                if (k.bits[w][s] != 8 || k.win_all) {
                    hb.kvrp[w][s] = static_cast<int8_t *>(alloc((size_t) nkv * hb.kvrc * hd));
                    hb.kvrs[w][s] = static_cast<__half *>(alloc((size_t) nkv * hb.kvrc * (hd / 32) * sizeof(__half)));
                    ring += (size_t) nkv * hb.kvrc * (hd + hd / 16);
                    nr++;
                }
        total += ring;
        fprintf(stderr, "[mk] KV recent window: the last %d..%d positions of %d low-bit sides stay q8 (MK_KV_WIN=%d; rings of %d rows per head, %.1f MiB)\n",
                span - bucket + 1, span, nr, hb.kvwin, hb.kvrc, ring / 1048576.0);
    }
    return total;
}

// The plain q8 cache (no MK_KV): the per-slot pointers in its [slot][kv head][ctx][hd] layout, for the kernels that
// address slots through them (the pipelined decode attention).
template <int NS, class B>
void kv_bind_q8(B & hb, size_t srows, int hd) {
    for (int s = 0; s < NS; s++) {
        hb.kvr[s] = hb.kvrv[s] = 0;
        for (int w = 0; w < 2; w++) {
            hb.kvb[w][s] = 8;
            hb.kvp[w][s] = static_cast<int8_t *>(w ? hb.vc : hb.kc) + s * srows * hd;
            hb.kvs[w][s] = (w ? hb.vsc : hb.ksc) + s * srows * (hd / 32);
            hb.kvrp[w][s] = nullptr;
            hb.kvrs[w][s] = nullptr;
        }
    }
    hb.kvwin = hb.kvrc = 0;
}

template <int NS>
void kv_spec_log(const KvSpec<NS> & k) {
    std::string kb, vb, kr, vr;
    for (int s = 0; s < NS; s++) {
        kb += (char) ('0' + k.bits[0][s]);
        vb += (char) ('0' + k.bits[1][s]);
        kr += k.rot[0][s] ? 'r' : '.';
        vr += k.rot[1][s] ? 'r' : '.';
    }
    fprintf(stderr, "[mk] MK_KV bits per slot (layers 3, 7, .., 63, MTP): K %s  V %s; rotated: K %s  V %s\n", kb.c_str(), vb.c_str(),
            kr.c_str(), vr.c_str());
}

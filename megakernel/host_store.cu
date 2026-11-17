// MKEngine::Impl: conversation state in host memory (side requests, parked conversations) and its MKEngine API.

#include "engine_impl.h"

char * MKEngine::Impl::hs_take() { // a chunk: from the free list, else a new one within the budget (nullptr: none)
    if (!hs_free.empty()) {
        char * c = hs_free.back();
        hs_free.pop_back();
        return c;
    }
    return hs_new();
}

char * MKEngine::Impl::hs_new() {
    if (hs_alloc + hs_chunk > hs_budget) return nullptr;
    char * c = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void **>(&c), hs_chunk, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    hs_alloc += hs_chunk;
    return c;
}

void MKEngine::Impl::hs_release(Blob & b) {
    for (char * c : b.chunks) hs_free.push_back(c);
    b = Blob();
    hs_trim();
}

void MKEngine::Impl::hs_trim() { // (free chunks above the budget: after a budget cut)
    while (hs_alloc > hs_budget && !hs_free.empty()) {
        cudaFreeHost(hs_free.back());
        hs_free.pop_back();
        hs_alloc -= hs_chunk;
    }
}

bool MKEngine::Impl::blob_put(Blob & b, const void * src, size_t n, bool host) {
    const char * s = static_cast<const char *>(src);
    b.items.push_back({const_cast<char *>(s), n, host});
    for (size_t done = 0; done < n;) {
        const size_t ci = b.used / hs_chunk, off = b.used % hs_chunk;
        if (ci == b.chunks.size()) {
            char * c = hs_take();
            if (!c) return false;
            b.chunks.push_back(c);
        }
        const size_t k = std::min(n - done, hs_chunk - off);
        if (host) memcpy(b.chunks[ci] + off, s + done, k);
        else CK(cudaMemcpyAsync(b.chunks[ci] + off, s + done, k, cudaMemcpyDeviceToHost, 0));
        done += k;
        b.used += k;
    }
    return true;
}

void MKEngine::Impl::blob_get(const Blob & b) {
    for (int pass = 0; pass < 2; pass++) {
        size_t pos = 0;
        for (const auto & it : b.items) {
            if (it.host == (pass == 0)) {
                pos += it.n;
                continue;
            }
            for (size_t done = 0; done < it.n;) {
                const size_t ci = pos / hs_chunk, off = pos % hs_chunk, k = std::min(it.n - done, hs_chunk - off);
                if (it.host) memcpy(it.at + done, b.chunks[ci] + off, k);
                else CK(cudaMemcpyAsync(it.at + done, b.chunks[ci] + off, k, cudaMemcpyHostToDevice, 0));
                done += k;
                pos += k;
            }
        }
    }
}

bool MKEngine::Impl::kv_put(Blob & bl, int a, int b) {
    if (b <= a) return true;
    const size_t cx = (size_t) hb.ctx, n = (size_t) (b - a);
    for (int s = 0; s < kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                if (hb.kvq8) {
                    const size_t rb = (size_t) HD * hb.kvb[w][s] / 8;
                    if (!blob_put(bl, reinterpret_cast<char *>(hb.kvp[w][s]) + ((size_t) g * cx + a) * rb, n * rb, false) ||
                        !blob_put(bl, hb.kvs[w][s] + ((size_t) g * cx + a) * (HD / 32), n * (HD / 32) * sizeof(__half), false))
                        return false;
                } else {
                    __half * r = static_cast<__half *>(w ? hb.vc : hb.kc) + (((size_t) s * NKV + g) * cx + a) * HD;
                    if (!blob_put(bl, r, n * HD * sizeof(__half), false)) return false;
                }
            }
    return true;
}

bool MKEngine::Impl::ring_put(Blob & bl, int a, int b) {
    if (hb.kvwin <= 0 || b <= a) return true;
    const size_t rc = hb.kvrc, n = (size_t) (b - a);
    for (int s = 0; s < kv_slots; s++)
        for (int w = 0; w < 2; w++)
            if (hb.kvrp[w][s])
                for (int g = 0; g < NKV; g++)
                    if (!blob_put(bl, hb.kvrp[w][s] + ((size_t) g * rc + a) * HD, n * HD, false) ||
                        !blob_put(bl, hb.kvrs[w][s] + ((size_t) g * rc + a) * (HD / 32), n * (HD / 32) * sizeof(__half), false))
                        return false;
    return true;
}

bool MKEngine::Impl::live_put(Blob & bl) {
    return blob_put(bl, hb.dstate, CK_NS * 4, false) && blob_put(bl, hb.conv, CK_NC * 4, false) &&
           blob_put(bl, hb.h_hist, CK_NH * 4, false);
}

size_t MKEngine::Impl::kv_token_bytes() const {
    size_t t = 0;
    for (int s = 0; s < kv_slots; s++)
        for (int w = 0; w < 2; w++) t += NKV * (hb.kvq8 ? (size_t) HD * hb.kvb[w][s] / 8 + (HD / 32) * sizeof(__half) : HD * sizeof(__half));
    return t;
}

size_t MKEngine::Impl::ring_bytes() const {
    size_t t = 0;
    if (hb.kvwin > 0)
        for (int s = 0; s < kv_slots; s++)
            for (int w = 0; w < 2; w++)
                if (hb.kvrp[w][s]) t += ring_vb() + ring_sb();
    return t;
}

auto MKEngine::Impl::book_get() const -> Book {
    Book k;
    k.seq = seq;
    k.dv = dv_ids;
    k.dv_n = dv_n;
    k.ctl = hctl;
    k.spans = spans;
    k.kv_top = kv_top;
    k.nd_avail = nd_avail;
    k.ndraft = ndraft;
    k.last_acc = last_acc;
    k.demoted = demoted;
    k.all_acc = all_acc;
    return k;
}

void MKEngine::Impl::book_set(const Book & k) {
    seq = k.seq;
    dv_set(k.dv);
    dv_n = k.dv_n;
    hctl = k.ctl;
    CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
    spans = k.spans;
    rope_sync();
    kv_top = k.kv_top;
    nd_avail = k.nd_avail;
    ndraft = k.ndraft;
    last_acc = k.last_acc;
    demoted = k.demoted;
    all_acc = k.all_acc;
    stale_replay = stale_pcnt = true; // (kvgb, dq, pcnt are not part of a Book: the next prefill rebuilds them)
}

void MKEngine::Impl::dv_set(const std::vector<int> & ids) {
    if (dv_in.empty()) return;
    dv_truncate(dv_base);
    for (size_t i = dv_base; i < ids.size(); i++) {
        dv_in[ids[i]] = 1;
        dv_ids.push_back(ids[i]);
    }
    const int k = (int) dv_ids.size() - dv_base;
    if (k > 0) CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + dv_base, dv_ids.data() + dv_base, k * 4, cudaMemcpyHostToDevice));
    hw.head_d.N = (int) dv_ids.size();
    CK(mkl::set_cW(&hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
}

void MKEngine::Impl::hs_quiesce() {
    ck_wait_restore();
    ring_snap_guard();
    ck_guard_state();
    CK(cudaDeviceSynchronize());
    for (auto & k : ck) k.pending = false;
    ck_scr_busy = false;
    ck_rs_src = -1;
}

void MKEngine::Impl::sw_need(int X, bool rings_all) {
    if (!sw_on || sw_broken) return;
    const int up = (X + MAXD + SW_BLK - 1) / SW_BLK * SW_BLK;
    const int want = std::max(sw_hi, std::min(sw_top, up)),
              rwant = hb.kvwin > 0 ? std::max(sw_rhi, rings_all ? hb.kvrc : std::min(hb.kvrc, up)) : 0;
    if (want <= sw_hi && rwant <= sw_rhi) return;
    const auto t0 = std::chrono::steady_clock::now();
    CK(cudaDeviceSynchronize()); // (nothing still writing those rows; the copies then run before the next kernel)
    const size_t u0 = sw_blob.used;
    const bool ok = kv_put(sw_blob, sw_hi, want) && ring_put(sw_blob, sw_rhi, rwant);
    wait_device("side request copies", std::max(wd_limit, 120.0));
    sw_hi = want;
    sw_rhi = rwant;
    sw_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        sw_broken = true;
        fprintf(stderr, "[mk] side request: the host store is full at row %d: the kept sequence is dropped (a full prefill when it comes back)\n", X);
    } else if (memlog) {
        fprintf(stderr, "[mk-mem] side request: kept rows < %d, ring slots < %d (+%.1f MB)\n", sw_hi, sw_rhi, (sw_blob.used - u0) / 1e6);
    }
}

bool MKEngine::Impl::sw_ck_take(int i) {
    if (!sw_ck_held(i)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t items = sw_blob.items.size(), used = sw_blob.used;
    const bool ok = blob_put(sw_blob, ck[i].mem, ck_block_bytes(), !ck_dev) &&
                    (!sw_ck[i].ring_ok || blob_put(sw_blob, ck[i].ring, ring_bytes_ck, true));
    if (ck_dev) wait_device("side request copies", std::max(wd_limit, 120.0)); // (device block: copies on stream 0)
    sw_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) { // (the partial item goes; its chunks stay with the blob)
        sw_blob.items.resize(items);
        sw_blob.used = used;
        return false;
    }
    sw_ck[i].saved = true;
    return true;
}

size_t MKEngine::Impl::hs_used() const { // bytes of the store in use (parked entries, the side request's copies)
    size_t n = sw_blob.chunks.size();
    for (const auto & p : parked) n += p.second.blob.chunks.size();
    return n * hs_chunk;
}

std::vector<int> MKEngine::Impl::park_cks(int max_cks) const { // slots of the newest checkpoints, newest first, then the pinned ones
    std::vector<int> v;
    for (int i = 0; i < (int) ck.size(); i++)
        if (ck[i].pos >= 0) v.push_back(i);
    std::sort(v.begin(), v.end(), [&](int a, int b) { return ck[a].pos > ck[b].pos; });
    const size_t n = std::min(v.size(), (size_t) std::max(0, max_cks));
    std::vector<int> out(v.begin(), v.begin() + n);
    for (size_t k = n; k < v.size() && max_cks > 0; k++)
        if (ck[v[k]].pin) out.push_back(v[k]);
    return out;
}

int MKEngine::Impl::park_entry(int max_cks) {
    const auto t0 = std::chrono::steady_clock::now();
    Parked e;
    struct Rel { // (the entry's chunks go back unless it is kept, also on an exception)
        Impl & I;
        Blob & b;
        bool keep = false;
        ~Rel() {
            if (keep) return;
            if (cudaStreamSynchronize(0) != cudaSuccess) cudaGetLastError(); // (no copy into its chunks still in flight)
            I.hs_release(b);
        }
    } rel{*this, e.blob};
    e.book = book_get();
    // device state first (copies on stream 0), then the checkpoint blocks (pinned host: memcpy while those run)
    bool ok = kv_put(e.blob, 0, seq_top()) && ring_put(e.blob, 0, hb.kvwin > 0 ? hb.kvrc : 0) && live_put(e.blob);
    for (int si : park_cks(max_cks)) {
        if (!ok) break;
        const CkSlot & s = ck[si];
        ok = blob_put(e.blob, s.mem, ck_block_bytes(), !ck_dev) && (!s.ring_ok || blob_put(e.blob, s.ring, ring_bytes_ck, true));
        e.cks.push_back({si, s.pos, s.ctl, s.ring_ok, s.pin, s.pin_t});
    }
    wait_device("park copies", std::max(wd_limit, 120.0));
    if (!ok) return -1;
    const int id = park_next++;
    hs_bytes = e.blob.used;
    hs_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    rel.keep = true;
    parked.emplace(id, std::move(e));
    return id;
}

void MKEngine::Impl::side_restore() {
    blob_get(sw_blob);
    wait_device("side request copies", std::max(wd_limit, 120.0));
    book_set(sw_book);
    for (size_t k = 0; k < ck.size(); k++) {
        ck[k].pos = sw_ck[k].pos;
        ck[k].ctl = sw_ck[k].ctl;
        ck[k].ring_ok = sw_ck[k].ring_ok;
        ck[k].pin = sw_ck[k].pin;
        ck[k].pin_t = sw_ck[k].pin_t;
        ck[k].pending = false;
    }
    sw_on = false;
    hs_bytes = sw_blob.used;
    hs_release(sw_blob);
}

// ---- conversation state in host memory: side requests, parked conversations; Impl "host store" ----
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void MKEngine::store_budget(size_t bytes) {
    p_->hs_budget = bytes;
    p_->hs_trim();
}

void MKEngine::store_reserve(size_t bytes) {
    Impl & I = *p_;
    while (I.hs_free.size() * I.hs_chunk < bytes) {
        char * c = I.hs_new();
        if (!c) break;
        I.hs_free.push_back(c);
    }
}

MKEngine::StoreInfo MKEngine::store_info() const {
    StoreInfo s;
    s.budget = p_->hs_budget;
    s.allocated = p_->hs_alloc;
    s.used = p_->hs_used();
    s.parked = (int) p_->parked.size();
    s.last_ms = p_->hs_ms;
    s.last_bytes = p_->hs_bytes;
    s.side_ms = p_->sw_ms_out;
    s.side_park_ms = p_->hs_park_ms;
    s.side_park_bytes = p_->hs_park_bytes;
    return s;
}

void MKEngine::kv_sizes(size_t * per_token, size_t * rings, size_t * state, size_t * checkpoint) const {
    *per_token = p_->kv_token_bytes();
    *rings = p_->ring_bytes();
    *state = p_->ck_block_bytes();
    *checkpoint = p_->ck_block_bytes() + p_->ring_bytes_ck;
}

bool MKEngine::side_begin() {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("side_begin: a side request is already running");
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    I.sw_book = I.book_get();
    I.sw_ck.clear();
    for (const auto & k : I.ck) I.sw_ck.push_back({k.pos, k.ctl, k.ring_ok, false, k.pin, k.pin_t});
    I.sw_top = I.seq_top();
    I.sw_hi = I.sw_rhi = 0;
    I.sw_broken = false;
    I.sw_on = true; // (from here a CUDA error drops the kept sequence: the C API calls side_drop)
    const bool ok = I.live_put(I.sw_blob);
    I.wait_device("side request copies", std::max(I.wd_limit, 120.0));
    if (!ok) { // (the store is full: nothing changed)
        I.sw_on = false;
        I.hs_release(I.sw_blob);
        return false;
    }
    for (auto & k : I.ck) k.pos = -1;
    reset();
    I.sw_ms_out = ms_since(t0);
    return true;
}

int MKEngine::side_end() {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_end: no side request is running");
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    if (I.sw_broken) {
        side_drop();
        return 0;
    }
    I.side_restore();
    I.hs_ms = ms_since(t0);
    return 1;
}

int MKEngine::side_end_park(int max_checkpoints, int * id) {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_end_park: no side request is running");
    *id = 0;
    I.hs_park_ms = 0;
    I.hs_park_bytes = 0;
    auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    if (I.sw_broken) { // (the kept sequence is lost: the side request's conversation stays, with its checkpoints)
        I.sw_on = false;
        I.hs_release(I.sw_blob);
        return 0;
    }
    if (!I.seq.empty()) {
        const int pid = I.park_entry(max_checkpoints); // (before side_restore writes the kept sequence's rows and blocks back)
        if (pid > 0) {
            *id = pid;
            I.hs_park_bytes = I.hs_bytes;
            I.hs_park_ms = ms_since(t0);
        }
        t0 = std::chrono::steady_clock::now();
    }
    I.side_restore();
    I.hs_ms = ms_since(t0);
    return 1;
}

int MKEngine::side_unpark(int id) {
    Impl & I = *p_;
    if (!I.sw_on) throw std::runtime_error("side_unpark: no side request is running");
    if (!I.seq.empty()) throw std::runtime_error("side_unpark: the side request has a sequence already");
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return -1;
    const auto t0 = std::chrono::steady_clock::now();
    I.hs_quiesce();
    Impl::Parked & e = it->second;
    const int top = std::max({e.book.kv_top, (int) e.book.seq.size(), e.book.ctl.dn_c});
    if (!I.sw_broken) { // room for what the kept sequence moves out first (else nothing changes)
        const int rows = std::min(I.sw_top, (top + MAXD + Impl::SW_BLK - 1) / Impl::SW_BLK * Impl::SW_BLK) - I.sw_hi;
        size_t need = I.kv_token_bytes() * (size_t) std::max(0, rows) + 2 * I.hs_chunk; // (+ the last chunks' slack)
        if (I.hb.kvwin > 0) need += I.ring_bytes() / I.hb.kvrc * (size_t) (I.hb.kvrc - I.sw_rhi);
        for (const auto & c : e.cks)
            if (I.sw_ck_held(c.slot)) need += I.ck_block_bytes() + (I.sw_ck[c.slot].ring_ok ? I.ring_bytes_ck : 0);
        if (need > I.hs_room()) return 0;
    }
    I.sw_need(top, true); // (the entry's rows, every ring slot)
    for (const auto & c : e.cks)
        if (!I.sw_ck_take(c.slot)) I.sw_ck[c.slot].pos = -1; // (no room after all: the kept sequence loses that checkpoint)
    I.blob_get(e.blob);
    I.wait_device("unpark copies", std::max(I.wd_limit, 120.0));
    I.book_set(e.book);
    for (const auto & c : e.cks) {
        Impl::CkSlot & s = I.ck[c.slot];
        s.pos = c.pos;
        s.ctl = c.ctl;
        s.ring_ok = c.ring_ok;
        s.pin = c.pin;
        s.pin_t = c.pin_t;
        s.pending = false;
    }
    I.hs_bytes = e.blob.used;
    I.hs_ms = ms_since(t0);
    I.hs_release(e.blob);
    I.parked.erase(it);
    return 1;
}

void MKEngine::side_drop() {
    Impl & I = *p_;
    if (!I.sw_on) return;
    I.sw_on = false;
    if (cudaStreamSynchronize(0) != cudaSuccess) cudaGetLastError(); // (a failed call's copies into sw_blob land first)
    I.hs_release(I.sw_blob);
    reset();
}

bool MKEngine::side_active() const { return p_->sw_on; }

int MKEngine::park(int max_checkpoints) {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("park: a side request is running");
    if (I.seq.empty()) return 0;
    I.hs_quiesce();
    return I.park_entry(max_checkpoints);
}

bool MKEngine::unpark(int id) {
    Impl & I = *p_;
    if (I.sw_on) throw std::runtime_error("unpark: a side request is running");
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return false;
    const auto t0 = std::chrono::steady_clock::now();
    reset(); // (the live sequence and its checkpoints go)
    I.hs_quiesce();
    Impl::Parked & e = it->second;
    I.blob_get(e.blob);
    I.wait_device("unpark copies", std::max(I.wd_limit, 120.0));
    I.book_set(e.book);
    for (const auto & c : e.cks) {
        Impl::CkSlot & s = I.ck[c.slot];
        s.pos = c.pos;
        s.ctl = c.ctl;
        s.ring_ok = c.ring_ok;
        s.pin = c.pin;
        s.pin_t = c.pin_t;
        s.pending = false;
    }
    I.hs_bytes = e.blob.used;
    I.hs_ms = ms_since(t0);
    I.hs_release(e.blob);
    I.parked.erase(it);
    return true;
}

bool MKEngine::park_drop(int id) {
    Impl & I = *p_;
    auto it = I.parked.find(id);
    if (it == I.parked.end()) return false;
    I.hs_release(it->second.blob);
    I.parked.erase(it);
    return true;
}

size_t MKEngine::park_need(int max_checkpoints) const {
    const Impl & I = *p_;
    if (I.seq.empty()) return 0;
    size_t n = I.kv_token_bytes() * (size_t) I.seq_top() + I.ring_bytes() + I.ck_block_bytes();
    for (int si : I.park_cks(max_checkpoints)) n += I.ck_block_bytes() + (I.ck[si].ring_ok ? I.ring_bytes_ck : 0);
    return (n + I.hs_chunk - 1) / I.hs_chunk * I.hs_chunk;
}

size_t MKEngine::park_bytes(int id) const {
    auto it = p_->parked.find(id);
    return it == p_->parked.end() ? 0 : it->second.blob.chunks.size() * p_->hs_chunk;
}

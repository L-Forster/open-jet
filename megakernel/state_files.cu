// MKEngine::Impl: post-prefill state files (MKEngine::prefill_state).

#include "engine_impl.h"

unsigned long long MKEngine::Impl::file_hash(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    std::vector<uint8_t> b((size_t) 8 << 20);
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (ssize_t r; (r = read(fd, b.data(), b.size())) > 0;) h = wd_hash(b.data(), (size_t) r, h);
    close(fd);
    return h;
}

unsigned long long MKEngine::Impl::self_hash() {
    static unsigned long long h = 0;
    if (h) return h;
#ifdef _WIN32
    char mp[PATH_MAX];
    std::string path = mk_win_module_path(reinterpret_cast<const void *>(&self_hash), mp, sizeof mp) ? mp : "(module)";
#else
    const uintptr_t a = reinterpret_cast<uintptr_t>(&self_hash);
    std::string path = "/proc/self/exe";
    if (FILE * m = fopen("/proc/self/maps", "r")) {
        char line[4096];
        while (fgets(line, sizeof line, m)) {
            unsigned long lo, hi;
            if (sscanf(line, "%lx-%lx", &lo, &hi) != 2 || a < lo || a >= hi) continue;
            const char * p = strchr(line, '/');
            if (p) {
                path = p;
                while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
            }
            break;
        }
        fclose(m);
    }
#endif
    h = file_hash(path);
    if (!h) throw std::runtime_error("state files: cannot read " + path + " to identify the build");
    return h;
}

std::string MKEngine::Impl::hex64(unsigned long long v) {
    char b[24];
    snprintf(b, sizeof b, "%016llx", v);
    return b;
}

void MKEngine::Impl::st_name(const std::vector<int> & head, std::string & path, std::string & key) {
    std::ostringstream k;
    k << "mkstate 1 bin " << hex64(self_hash());
    cudaDeviceProp prop; // (a build for several architectures runs on other GPUs too)
    CK(cudaGetDeviceProperties(&prop, cuda_dev));
    k << " gpu " << prop.name << " sm_" << prop.major << prop.minor << " " << prop.multiProcessorCount;
    struct stat st{};
    char rp[PATH_MAX];
    if (!realpath(model_path.c_str(), rp)) snprintf(rp, sizeof rp, "%s", model_path.c_str());
    stat(rp, &st);
    const size_t edge = std::min<size_t>(file.size, (size_t) 4 << 20);
    unsigned long long mh = wd_hash(file.base, edge);
    mh = wd_hash(file.base + file.size - edge, edge, mh);
#ifdef _WIN32
    k << " model " << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtime << ".0"
#else
    k << " model " << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtim.tv_sec << "." << (long long) st.st_mtim.tv_nsec
#endif
      << " " << hex64(mh);
    k << " mtp " << mtp_desc; // (the MTP block's weights decide its KV slot; none: no slot)
    k << " ctx " << ctx << " kvq8 " << hb.kvq8 << " kvb";
    for (int w = 0; w < 2; w++)
        for (int s = 0; s < N_ATT_SLOTS; s++) k << " " << hb.kvb[w][s] << (w ? hb.kvrv[s] : hb.kvr[s]);
    k << " win " << hb.kvwin << " " << hb.kvrc << " exact " << exact << " cap " << hp.cap << " slots " << N_ATT_SLOTS << " nkv " << NKV
      << " hring " << H_RING << " conv " << CONV_RING;
    // environment: every MK_* setting except those that only change decode, logging, timing or the load path
    static const char * const skip[] = {"MK_STATE", "MK_POLICY", "MK_PROFILE", "MK_SPEC_LOG", "MK_MEMLOG", "MK_LOADLOG", "MK_LOAD_",
                                        "MK_WDIGEST", "MK_WATCHDOG", "MK_CK_SLOTS", "MK_BENCH", "MK_STACK_LIMIT",
                                        "MK_RESERVE_EARLY", "MK_VRAM_", // (device memory placement / load checks)
                                        "MK_K1=", "MK_K2=", "MK_GSB", // (K1 / K2 / gsync bench: decode only)
                                        "MK_K6=",                     // (K6: decode only; MK_K8 = prefill: kept)
                                        "MK_PF_PROFILE", "MK_PF_GAPS", "MK_PF_SYNC", // (prefill timing / launch flow)
                                        "MK_CK_DEVICE", "MK_STORE_", "MK_SLOT_RESERVE", "MK_CUDAMALLOC"}; // (placement)
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        if (strncmp(*e, "MK_", 3)) continue;
        bool sk = false;
        for (const char * s : skip) sk = sk || !strncmp(*e, s, strlen(s));
        if (!sk) env.push_back(*e);
    }
    std::sort(env.begin(), env.end());
    for (const auto & e : env) k << " env " << e;
    k << " tokens " << head.size() << " " << hex64(wd_hash(reinterpret_cast<const uint8_t *>(head.data()), head.size() * sizeof(int)));
    key = k.str();
    path = st_dir + "/" + hex64(wd_hash(reinterpret_cast<const uint8_t *>(key.data()), key.size())) + ".mks";
}

std::vector<std::pair<char *, size_t>> MKEngine::Impl::st_regions(int hend) {
    std::vector<std::pair<char *, size_t>> r;
    const size_t cx = (size_t) hb.ctx, n = (size_t) hend; // (row stride per kv head)
    for (int s = 0; s < kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                if (hb.kvq8) {
                    const size_t rb = (size_t) HD * hb.kvb[w][s] / 8;
                    r.push_back({reinterpret_cast<char *>(hb.kvp[w][s]) + (size_t) g * cx * rb, n * rb});
                    r.push_back({reinterpret_cast<char *>(hb.kvs[w][s] + (size_t) g * cx * (HD / 32)), n * (HD / 32) * sizeof(__half)});
                } else {
                    __half * b = static_cast<__half *>(w ? hb.vc : hb.kc) + ((size_t) s * NKV + g) * cx * HD;
                    r.push_back({reinterpret_cast<char *>(b), n * HD * sizeof(__half)});
                }
            }
    for (int s = 0; s < kv_slots; s++)
        for (int w = 0; w < 2; w++)
            if (hb.kvrp[w][s]) {
                r.push_back({reinterpret_cast<char *>(hb.kvrp[w][s]), (size_t) NKV * hb.kvrc * HD});
                r.push_back({reinterpret_cast<char *>(hb.kvrs[w][s]), (size_t) NKV * hb.kvrc * (HD / 32) * sizeof(__half)});
            }
    r.push_back({reinterpret_cast<char *>(hb.dstate), CK_NS * 4});
    r.push_back({reinterpret_cast<char *>(hb.conv), CK_NC * 4});
    r.push_back({reinterpret_cast<char *>(hb.h_hist), CK_NH * 4});
    return r;
}

void MKEngine::Impl::st_save(const std::string & path, const std::string & key, int hend) {
    static const char magic[9] = "MKSTATE1";
    const auto t0 = lclk::now();
    CK(cudaDeviceSynchronize());
    mkdir(st_dir.c_str(), 0755);
    const auto R = st_regions(hend);
    const std::string tmp = path + ".tmp" + std::to_string(getpid());
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[mk] state file: cannot write %s (not saved)\n", tmp.c_str());
        return;
    }
    bool ok = fwrite(magic, 8, 1, f) == 1;
    auto w64 = [&](uint64_t v) { ok = ok && fwrite(&v, 8, 1, f) == 1; };
    w64(key.size());
    ok = ok && fwrite(key.data(), 1, key.size(), f) == key.size();
    w64((uint64_t) hend);
    w64(R.size());
    size_t tot = 0;
    for (const auto & r : R) w64(r.second), tot += r.second;
    const size_t PC = (size_t) 64 << 20;
    std::unique_ptr<uint8_t[]> buf(new uint8_t[PC]);
    try {
        for (const auto & r : R)
            for (size_t o = 0; ok && o < r.second; o += PC) {
                const size_t n = std::min(PC, r.second - o);
                CK(cudaMemcpy(buf.get(), r.first + o, n, cudaMemcpyDeviceToHost));
                ok = fwrite(buf.get(), 1, n, f) == n;
            }
    } catch (...) { // (a CUDA error: no half-written file is left behind)
        fclose(f);
        unlink(tmp.c_str());
        throw;
    }
    ok = (fclose(f) == 0) && ok;
#ifdef _WIN32
    if (ok) remove(path.c_str()); // (rename does not replace an existing file on Windows)
#endif
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        fprintf(stderr, "[mk] state file: writing %s failed (not saved)\n", path.c_str());
        return;
    }
    st_s = lsec(t0);
    fprintf(stderr, "[mk] state file saved: %s (%.2f GB, %d positions) in %.2f s\n", path.c_str(), tot / 1e9, hend, st_s);
}

bool MKEngine::Impl::st_load(const std::string & path, const std::string & key, int hend) {
    static const char magic[9] = "MKSTATE1";
    const auto t0 = lclk::now();
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    fstat(fd, &st);
    const size_t fsz = (size_t) st.st_size;
    void * mp = fsz ? mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    close(fd);
    if (mp == MAP_FAILED) return false;
    const uint8_t * b = static_cast<const uint8_t *>(mp);
    size_t pos = 0;
    auto r64 = [&](uint64_t & v) {
        if (pos + 8 > fsz) return false;
        memcpy(&v, b + pos, 8);
        pos += 8;
        return true;
    };
    const auto R = st_regions(hend);
    uint64_t kn = 0, h = 0, nr = 0, v = 0;
    bool ok = fsz >= 8 && !memcmp(b, magic, 8);
    pos = 8;
    ok = ok && r64(kn) && pos + kn <= fsz && key.size() == kn && !memcmp(b + pos, key.data(), kn);
    pos += kn;
    ok = ok && r64(h) && h == (uint64_t) hend && r64(nr) && nr == R.size();
    size_t tot = 0;
    for (size_t i = 0; ok && i < R.size(); i++) ok = r64(v) && v == R[i].second, tot += R[i].second;
    ok = ok && pos + tot == fsz;
    if (!ok) {
        munmap(mp, fsz);
        fprintf(stderr, "[mk] state file %s does not match (rewriting it)\n", path.c_str());
        return false;
    }
    CK(cudaDeviceSynchronize()); // (no copy of an earlier sequence still reading the state)
    { // read the file in with several threads (cold: one thread is bound by its page faults)
        const size_t PC = (size_t) 4 << 20, np = (fsz + PC - 1) / PC;
        std::atomic<size_t> next{0};
        std::atomic<unsigned> sink{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 8; t++)
            th.emplace_back([&] {
                unsigned s = 0;
                for (size_t i; (i = next++) < np;)
                    for (size_t o = i * PC; o < std::min(fsz, (i + 1) * PC); o += 4096) s += b[o];
                sink += s;
            });
        for (auto & t : th) t.join();
    }
    sw_need(hend); // (side request)
    for (const auto & r : R) {
        CK(cudaMemcpy(r.first, b + pos, r.second, cudaMemcpyHostToDevice));
        pos += r.second;
    }
    munmap(mp, fsz);
    kv_top = std::max(kv_top, hend); // (as pf_run of 0 .. hend - 1)
    // MK_STATE_WARM_MS (default 1000; 0: off): GPU warm-up before the decode (st_warm)
    const double wms = getenv("MK_STATE_WARM_MS") ? atof(getenv("MK_STATE_WARM_MS")) : 1000.0;
    if (wms > 0) {
        mkl::st_warm(nsm * 4, (unsigned long long) (wms * 1e6));
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
    st_s = lsec(t0);
    fprintf(stderr, "[mk] state file restored: %s (%.2f GB, %d positions) in %.2f s (incl. %.1f s GPU warm-up, MK_STATE_WARM_MS)\n",
            path.c_str(), tot / 1e9, hend, st_s, wms / 1000.0);
    return true;
}

std::string MKEngine::Impl::file_ident(const std::string & p, const gguf::File & f) {
    char rp[PATH_MAX];
    if (!realpath(p.c_str(), rp)) snprintf(rp, sizeof rp, "%s", p.c_str());
    struct stat st{};
    stat(rp, &st);
    const size_t edge = std::min<size_t>(f.size, (size_t) 4 << 20);
    unsigned long long mh = wd_hash(f.base, edge);
    mh = wd_hash(f.base + f.size - edge, edge, mh);
    std::ostringstream k;
#ifdef _WIN32
    k << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtime << ".0";
#else
    k << rp << " " << (long long) st.st_size << " " << (long long) st.st_mtim.tv_sec << "." << (long long) st.st_mtim.tv_nsec;
#endif
    k << " " << hex64(mh);
    return k.str();
}

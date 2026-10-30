// Minimal read-only GGUF parser (mmap based). Only what the megakernel loader needs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gguf {

enum GType : uint32_t {
    F32 = 0, F16 = 1, Q8_0 = 8, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_NL = 20, IQ3_S = 21, IQ4_XS = 23
};

struct Tensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    uint64_t offset = 0;
    const uint8_t * data = nullptr;
    uint64_t nbytes = 0;
};

inline uint64_t type_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case F32: return n * 4;
        case F16: return n * 2;
        case Q8_0: return n / 32 * 34;
        case Q4_K: return n / 256 * 144;
        case Q5_K: return n / 256 * 176;
        case Q6_K: return n / 256 * 210;
        case Q3_K: return n / 256 * 110;
        case IQ3_S: return n / 256 * 110;
        case IQ4_XS: return n / 256 * 136;
        case IQ4_NL: return n / 32 * 18;
        default: throw std::runtime_error("unsupported ggml type " + std::to_string(type));
    }
}

struct File {
    const uint8_t * base = nullptr;
    size_t size = 0;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, uint64_t> ints;
    std::map<std::string, std::string> strs;

#ifdef _WIN32
    explicit File(const std::string & path) {
        // the path is UTF-8 (from Python); the wide API takes any user's file name
        const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wpath(n > 0 ? n : 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], n);
        HANDLE fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fh == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz;
        GetFileSizeEx(fh, &sz);
        size = (size_t) sz.QuadPart;
        HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(fh);
        if (!mh) throw std::runtime_error("CreateFileMapping failed");
        void * p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mh);  // the view keeps the mapping alive
        if (!p) throw std::runtime_error("MapViewOfFile failed");
        base = (const uint8_t *) p;
        parse();
    }
#else
    explicit File(const std::string & path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        fstat(fd, &st);
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base = (const uint8_t *) p;
        parse();
    }
#endif
    ~File() { release(); }

    // unmaps the file once its tensors are copied out (keeps the metadata): a mapping held for the engine's life keeps
    // the whole model in the page cache, which under WSL stays counted against Windows
    void release() {
#ifdef _WIN32
        if (base) UnmapViewOfFile(base);
#else
        if (base) munmap((void *) base, size);
#endif
        base = nullptr;
        tensors.clear();
    }

    const Tensor & get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

  private:
    size_t pos = 0;
    template <class T> T rd() { T v; memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v; }
    std::string rds() { uint64_t n = rd<uint64_t>(); std::string s((const char *) base + pos, n); pos += n; return s; }

    void skip_val(uint32_t t, const std::string & key, bool top) {
        switch (t) {
            case 0: case 1: case 7: { uint8_t v = rd<uint8_t>(); if (top) ints[key] = v; break; }
            case 2: case 3: pos += 2; break;
            case 4: { uint32_t v = rd<uint32_t>(); if (top) ints[key] = v; break; }
            case 5: { int32_t v = rd<int32_t>(); if (top) ints[key] = (uint64_t) v; break; }
            case 6: pos += 4; break;
            case 8: { std::string s = rds(); if (top) strs[key] = s; break; }
            case 9: {
                uint32_t et = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                for (uint64_t i = 0; i < n; i++) skip_val(et, key, false);
                break;
            }
            case 10: { uint64_t v = rd<uint64_t>(); if (top) ints[key] = v; break; }
            case 11: { int64_t v = rd<int64_t>(); if (top) ints[key] = (uint64_t) v; break; }
            case 12: pos += 8; break;
            default: throw std::runtime_error("bad gguf kv type");
        }
    }

    void parse() {
        if (memcmp(base, "GGUF", 4) != 0) throw std::runtime_error("not a gguf file");
        pos = 4;
        rd<uint32_t>();
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rds();
            uint32_t t = rd<uint32_t>();
            skip_val(t, key, true);
        }
        std::vector<Tensor> infos;
        for (uint64_t i = 0; i < n_tensors; i++) {
            Tensor t;
            t.name = rds();
            uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd; d++) t.ne.push_back(rd<uint64_t>());
            t.type = rd<uint32_t>();
            t.offset = rd<uint64_t>();
            infos.push_back(t);
        }
        uint64_t align = ints.count("general.alignment") ? ints["general.alignment"] : 32;
        size_t data_start = (pos + align - 1) / align * align;
        for (auto & t : infos) {
            t.data = base + data_start + t.offset;
            uint64_t rows = 1;
            for (size_t d = 1; d < t.ne.size(); d++) rows *= t.ne[d];
            t.nbytes = type_row_bytes(t.type, t.ne[0]) * rows;
            tensors[t.name] = t;
        }
    }
};

} // namespace gguf

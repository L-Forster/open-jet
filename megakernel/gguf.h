// Minimal read-only GGUF parser (mmap based). Only what the megakernel loader needs.
// Every read is checked against the file size and every size is computed without overflow: a truncated or malformed
// file throws std::runtime_error (never a read outside the mapping or a division by zero). Every length and count is
// checked against the bytes left before anything is allocated for it: memory grows with the file's size (a string up to
// the rest of the file), never with a number the file declares.
// A tensor's byte size comes from its type and dimensions (never from anything else the file says); its type must be one
// the engine reads, every dimension at least 1, and offset + size inside the data section. Readers that expect a shape
// ask for it (File::get(name, dims), shape_is): a tensor with the right name but another shape (fewer rows, an extra
// dimension) is refused with its name, so no reader trusts a size the file declares. tests/gguf_fuzz.cpp feeds it
// malformed files.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include "compat_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
inline int mk_open(const char * path, int flags) { return open(path, flags); }
#endif

namespace gguf {

enum GType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_NL = 20, IQ3_S = 21, IQ4_XS = 23
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
        case Q4_0: return n / 32 * 18;
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

// weights per block of a type (a row holds whole blocks); 0: a type the engine does not read
inline uint64_t type_block(uint32_t type) {
    switch (type) {
        case F32: case F16: return 1;
        case Q4_0: case Q8_0: case IQ4_NL: return 32;
        case Q4_K: case Q5_K: case Q6_K: case Q3_K: case IQ3_S: case IQ4_XS: return 256;
        default: return 0;
    }
}

inline std::string shape_str(const std::vector<uint64_t> & ne) {
    std::string s = "[";
    for (size_t d = 0; d < ne.size(); d++) s += (d ? ", " : "") + std::to_string(ne[d]);
    return s + "]";
}

// ne is exactly `dims` (GGUF order: ne[0] = the row length), missing or further dimensions being 1
inline bool shape_is(const std::vector<uint64_t> & ne, const std::vector<uint64_t> & dims) {
    for (size_t d = 0; d < std::max(ne.size(), dims.size()); d++)
        if ((d < ne.size() ? ne[d] : 1) != (d < dims.size() ? dims[d] : 1)) return false;
    return true;
}

struct File {
    const uint8_t * base = nullptr;
    size_t size = 0;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, uint64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, float> floats; // float32 kvs (rope.freq_base, layer_norm_rms_epsilon)
    std::map<std::string, uint64_t> arrs; // element count of each array kv (e.g. tokenizer.ggml.tokens: the vocabulary)
    std::map<std::string, std::vector<int64_t>> iarrs; // integer arrays of up to 16 elements (rope.dimension_sections)

    static constexpr uint64_t MAX_ALIGNMENT = 1u << 20; // general.alignment: a power of two up to this

    explicit File(const std::string & path) {
        int fd = mk_open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("cannot stat " + path);
        }
        if (st.st_size < 24) { // magic, version, tensor and kv counts
            close(fd);
            throw std::runtime_error("not a gguf file (" + std::to_string((long long) st.st_size) + " bytes): " + path);
        }
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base = (const uint8_t *) p;
        try {
            parse();
        } catch (const std::exception & e) {
            munmap((void *) base, size);
            base = nullptr;
            throw std::runtime_error(path + ": " + e.what());
        }
    }
    ~File() { if (base) munmap((void *) base, size); }

    const Tensor & get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    // the tensor, which must have the shape `dims` (shape_is): its nbytes then cover exactly what a reader of that shape
    // reads (the type's row bytes x the rows)
    const Tensor & get(const std::string & name, const std::vector<uint64_t> & dims) const {
        const Tensor & t = get(name);
        if (!shape_is(t.ne, dims))
            throw std::runtime_error("tensor " + name + " has shape " + shape_str(t.ne) + ", expected " + shape_str(dims));
        return t;
    }

  private:
    size_t pos = 0; // invariant: pos <= size

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("bad gguf: " + what + " (at byte " + std::to_string(pos) + " of " + std::to_string(size) + ")");
    }
    void need(uint64_t n) const {
        if (n > size - pos) fail("truncated file");
    }
    template <class T> T rd() { need(sizeof(T)); T v; memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v; }
    void skip(uint64_t n) { need(n); pos += (size_t) n; }
    std::string rds() {
        const uint64_t n = rd<uint64_t>();
        need(n);
        std::string s((const char *) base + pos, (size_t) n);
        pos += (size_t) n;
        return s;
    }
    // bytes of a fixed-size value type (0: string, array or invalid)
    static uint64_t scalar_bytes(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: return 1;
            case 2: case 3: return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: return 0;
        }
    }
    // an integer element of type t at p (signed types sign-extended, a u64 as its bits)
    static int64_t int_at(uint32_t t, const uint8_t * p) {
        switch (t) {
            case 0: case 7: return *p;
            case 1: return (int8_t) *p;
            case 2: { uint16_t v; memcpy(&v, p, 2); return v; }
            case 3: { int16_t v; memcpy(&v, p, 2); return v; }
            case 4: { uint32_t v; memcpy(&v, p, 4); return v; }
            case 5: { int32_t v; memcpy(&v, p, 4); return v; }
            default: { int64_t v; memcpy(&v, p, 8); return v; } // 10, 11
        }
    }

    void skip_val(uint32_t t, const std::string & key) {
        switch (t) {
            case 0: case 1: case 7: ints[key] = rd<uint8_t>(); break;
            case 2: case 3: skip(2); break;
            case 4: ints[key] = rd<uint32_t>(); break;
            case 5: ints[key] = (uint64_t) rd<int32_t>(); break;
            case 6: floats[key] = rd<float>(); break;
            case 8: strs[key] = rds(); break;
            case 9: { // arrays are skipped but short integer ones; their elements are scalars or strings (ggml rejects nested arrays too)
                const uint32_t et = rd<uint32_t>();
                const uint64_t n = rd<uint64_t>();
                arrs[key] = n;
                if (const uint64_t eb = scalar_bytes(et)) {
                    if (n > (size - pos) / eb) fail("array " + key + " of " + std::to_string(n) + " elements runs past the end");
                    if (n <= 16 && et != 6 && et != 12) { // (not f32 / f64)
                        std::vector<int64_t> & v = iarrs[key];
                        for (uint64_t i = 0; i < n; i++) v.push_back(int_at(et, base + pos + i * eb));
                    }
                    pos += (size_t) (n * eb);
                } else if (et == 8) {
                    if (n > (size - pos) / 8) fail("array " + key + " of " + std::to_string(n) + " strings runs past the end");
                    for (uint64_t i = 0; i < n; i++) skip(rd<uint64_t>());
                } else {
                    fail("array " + key + " of element type " + std::to_string(et));
                }
                break;
            }
            case 10: ints[key] = rd<uint64_t>(); break;
            case 11: ints[key] = (uint64_t) rd<int64_t>(); break;
            case 12: skip(8); break;
            default: fail("kv " + key + " of type " + std::to_string(t));
        }
    }

    void parse() {
        if (memcmp(base, "GGUF", 4) != 0) throw std::runtime_error("not a gguf file");
        pos = 4;
        const uint32_t version = rd<uint32_t>();
        if (version != 2 && version != 3) fail("unsupported version " + std::to_string(version));
        const uint64_t n_tensors = rd<uint64_t>();
        const uint64_t n_kv = rd<uint64_t>();
        // a kv takes at least 13 bytes (key length, type, a 1-byte value), a tensor info at least 32
        if (n_kv > (size - pos) / 13) fail(std::to_string(n_kv) + " kvs do not fit the file");
        if (n_tensors > (size - pos) / 32) fail(std::to_string(n_tensors) + " tensors do not fit the file");
        std::set<std::string> keys;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rds();
            if (!keys.insert(key).second) fail("duplicate key " + key);
            const uint32_t t = rd<uint32_t>();
            skip_val(t, key);
        }
        std::vector<Tensor> infos;
        for (uint64_t i = 0; i < n_tensors; i++) {
            Tensor t;
            t.name = rds();
            const uint32_t nd = rd<uint32_t>();
            if (nd < 1 || nd > 4) fail("tensor " + t.name + " has " + std::to_string(nd) + " dimensions");
            uint64_t nel = 1;
            for (uint32_t d = 0; d < nd; d++) {
                const uint64_t ne = rd<uint64_t>();
                if (ne == 0) fail("tensor " + t.name + " has a dimension of 0"); // (an empty tensor: nothing to read)
                if (ne > (uint64_t) INT64_MAX || nel > (uint64_t) INT64_MAX / ne)
                    fail("tensor " + t.name + " has too many elements");
                nel *= ne;
                t.ne.push_back(ne);
            }
            t.type = rd<uint32_t>();
            t.offset = rd<uint64_t>();
            infos.push_back(std::move(t));
        }
        const uint64_t align = ints.count("general.alignment") ? ints["general.alignment"] : 32;
        if (align == 0 || (align & (align - 1)) != 0 || align > MAX_ALIGNMENT)
            fail("general.alignment " + std::to_string(align) + " is not a power of two up to " + std::to_string(MAX_ALIGNMENT));
        const uint64_t data_start = (pos + align - 1) / align * align;
        if (data_start > size) fail("the data section starts past the end of the file");
        const uint64_t data_size = size - data_start;
        for (auto & t : infos) {
            const uint64_t bl = type_block(t.type);
            if (!bl) fail("tensor " + t.name + " has unsupported ggml type " + std::to_string(t.type));
            if (t.ne[0] % bl != 0) fail("tensor " + t.name + ": a row of " + std::to_string(t.ne[0]) + " is not whole blocks");
            uint64_t rows = 1;
            for (size_t d = 1; d < t.ne.size(); d++) rows *= t.ne[d]; // (< 2^63: checked above)
            const uint64_t bb = type_row_bytes(t.type, bl), nb = t.ne[0] / bl; // bytes per block, blocks per row
            if (nb > data_size / bb) fail("tensor " + t.name + " is larger than the file");
            const uint64_t rb = nb * bb; // = type_row_bytes(t.type, t.ne[0])
            if (rows && rb > data_size / rows) fail("tensor " + t.name + " is larger than the file");
            t.nbytes = rb * rows;
            if (t.offset % align != 0) fail("tensor " + t.name + " offset " + std::to_string(t.offset) + " is not aligned");
            if (t.offset > data_size || t.nbytes > data_size - t.offset) fail("tensor " + t.name + " runs past the end of the file");
            t.data = base + data_start + t.offset;
            const std::string name = t.name;
            if (!tensors.emplace(name, std::move(t)).second) fail("duplicate tensor " + name);
        }
    }
};

} // namespace gguf

// Malformed-file test of gguf.h (CPU only, no CUDA): every file below must either parse or throw std::exception; a
// crash, a sanitizer report or a hang is a failure. Every file that parses has each tensor's bytes read in full and its
// shape-checked reads (File::get(name, dims)) tried, as the loader does. Build and run (Linux):
//   make tests/gguf_fuzz && tests/gguf_fuzz             (built with -fsanitize=address,undefined)
//   tests/gguf_fuzz --dump model.gguf                   (the parsed kvs and tensors, to compare two builds of gguf.h)
//   GGUF_FUZZ_ONLY="alignment 0" tests/gguf_fuzz      (one check)
#include "../gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<uint8_t>;

template <class T> void put(Bytes & b, T v) {
    const auto * p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(Bytes & b, const std::string & s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Spec { // a small, valid file; the tests change one field at a time
    uint32_t version = 3;
    uint64_t n_tensors_hdr = 2, n_kv_hdr = 5;
    uint32_t align = 32;
    uint64_t arr_n = 3, str_arr_n = 2, str_len = 5;
    uint32_t arr_type = 4;
    std::vector<uint32_t> arr_vals = {7, 8, 9}; // test.array's bytes
    uint32_t a_dims = 1, a_type = gguf::F32;
    uint64_t a_ne0 = 8, a_off = 0;
    uint64_t b_ne0 = 256, b_ne1 = 2, b_off = 32;
    std::vector<uint64_t> b_more; // further dimensions of b
    uint32_t b_type = gguf::Q4_K;
    std::string b_name = "b";
    bool dup_key = false;
};

Bytes build(const Spec & s) {
    Bytes b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, s.version);
    put<uint64_t>(b, s.n_tensors_hdr);
    put<uint64_t>(b, s.n_kv_hdr);
    put_str(b, "general.architecture");
    put<uint32_t>(b, 8);
    put_str(b, "qwen35");
    put_str(b, s.dup_key ? "general.architecture" : "general.alignment");
    put<uint32_t>(b, 4);
    put<uint32_t>(b, s.align);
    put_str(b, "test.array");
    put<uint32_t>(b, 9);
    put<uint32_t>(b, s.arr_type);
    put<uint64_t>(b, s.arr_n);
    for (uint32_t v : s.arr_vals) put<uint32_t>(b, v);
    put_str(b, "test.strings");
    put<uint32_t>(b, 9);
    put<uint32_t>(b, 8);
    put<uint64_t>(b, s.str_arr_n);
    put<uint64_t>(b, s.str_len);
    b.insert(b.end(), {'h', 'e', 'l', 'l', 'o'});
    put_str(b, "x");
    put_str(b, "test.float");
    put<uint32_t>(b, 6);
    put<float>(b, 2.5f);
    // tensor infos
    put_str(b, "a");
    put<uint32_t>(b, s.a_dims);
    put<uint64_t>(b, s.a_ne0);
    for (uint32_t d = 1; d < s.a_dims && d < 8; d++) put<uint64_t>(b, 1);
    put<uint32_t>(b, s.a_type);
    put<uint64_t>(b, s.a_off);
    put_str(b, s.b_name);
    put<uint32_t>(b, 2 + (uint32_t) s.b_more.size());
    put<uint64_t>(b, s.b_ne0);
    put<uint64_t>(b, s.b_ne1);
    for (uint64_t n : s.b_more) put<uint64_t>(b, n);
    put<uint32_t>(b, s.b_type);
    put<uint64_t>(b, s.b_off);
    while (b.size() % 32) b.push_back(0);
    for (int i = 0; i < 8; i++) put<float>(b, 1.5f * i); // a: 8 f32 = 32 bytes at offset 0
    for (int i = 0; i < 288; i++) b.push_back((uint8_t) i); // b: 2 rows of one Q4_K block at offset 32
    return b;
}

std::string g_path;
int g_fail = 0, g_n = 0;
unsigned g_sink = 0;

// 1: parsed, 0: threw; anything else (a crash) ends the process
int parse(const Bytes & b, std::string * err = nullptr) {
    FILE * f = fopen(g_path.c_str(), "wb");
    if (!f || (b.size() && fwrite(b.data(), 1, b.size(), f) != b.size())) {
        fprintf(stderr, "cannot write %s\n", g_path.c_str());
        exit(2);
    }
    fclose(f);
    try {
        gguf::File g(g_path);
        // the loader's reads: every byte of every tensor (inside the mapping, or a SIGSEGV / SIGBUS here), and the
        // shape-checked lookups (an exception is fine)
        unsigned x = 0;
        for (const auto & [name, t] : g.tensors)
            for (uint64_t i = 0; i < t.nbytes; i++) x += t.data[i];
        g_sink += x;
        for (const auto & [name, dims] : std::vector<std::pair<std::string, std::vector<uint64_t>>>{{"a", {8}}, {"b", {256, 2}}}) {
            try {
                const auto & t = g.get(name, dims);
                if (t.nbytes != gguf::type_row_bytes(t.type, dims[0]) * (dims.size() > 1 ? dims[1] : 1)) {
                    fprintf(stderr, "shape-checked %s: %llu bytes\n", name.c_str(), (unsigned long long) t.nbytes);
                    abort();
                }
            } catch (const std::exception &) {
            }
        }
        return 1;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return 0;
    }
}

bool selected(const char * what) {
    const char * only = getenv("GGUF_FUZZ_ONLY");
    return !only || std::string(only) == what;
}

// a shape-checked read of a file that parses: File::get(name, dims) returns (1) or throws (0)
void expect_get(const char * what, const Bytes & b, const std::string & name, const std::vector<uint64_t> & dims, int want) {
    if (!selected(what)) return;
    g_n++;
    if (parse(b) != 1) {
        g_fail++;
        printf("FAIL %s: the file does not parse\n", what);
        return;
    }
    std::string err;
    int got = 0;
    try {
        gguf::File g(g_path);
        g.get(name, dims);
        got = 1;
    } catch (const std::exception & e) {
        err = e.what();
    }
    if (got != want) {
        g_fail++;
        printf("FAIL %s: %s\n", what, got ? "read" : ("threw: " + err).c_str());
    } else {
        printf("ok   %s%s%s\n", what, got ? "" : ": ", err.c_str());
    }
}

void expect(const char * what, const Bytes & b, int want) {
    if (!selected(what)) return;
    std::string err;
    const int got = parse(b, &err);
    g_n++;
    if (got != want) {
        g_fail++;
        printf("FAIL %s: %s\n", what, got ? "parsed" : ("threw: " + err).c_str());
    } else {
        printf("ok   %s%s%s\n", what, got ? "" : ": ", got ? "" : err.substr(err.find("bad gguf") == std::string::npos ? 0 : err.find("bad gguf")).c_str());
    }
}

int dump(const char * path) {
    gguf::File g(path);
    printf("size %zu, %zu tensors, %zu ints, %zu strs\n", g.size, g.tensors.size(), g.ints.size(), g.strs.size());
    for (const auto & [k, v] : g.ints) printf("int %s = %" PRIu64 "\n", k.c_str(), v);
    for (const auto & [k, v] : g.strs) printf("str %s = %zu bytes, hash %zu\n", k.c_str(), v.size(), std::hash<std::string>()(v));
    for (const auto & [k, v] : g.floats) printf("float %s = %.9g\n", k.c_str(), v);
    for (const auto & [k, v] : g.iarrs) {
        printf("int array %s =", k.c_str());
        for (int64_t x : v) printf(" %" PRId64, x);
        printf("\n");
    }
    for (const auto & [k, t] : g.tensors) {
        printf("tensor %s type %u ne", k.c_str(), t.type);
        for (auto n : t.ne) printf(" %" PRIu64, n);
        printf(" offset %" PRIu64 " data %td nbytes %" PRIu64 "\n", t.offset, t.data - g.base, t.nbytes);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--dump") return dump(argv[2]);
    g_path = (std::filesystem::temp_directory_path() / ("gguf_fuzz_" + std::to_string((long long) std::random_device()()) + ".gguf")).string();

    const Spec ok;
    const Bytes good = build(ok);
    expect("valid file", good, 1);
    if (selected("valid file contents")) {
        parse(good);
        gguf::File g(g_path);
        const auto & a = g.get("a");
        const auto & b = g.get("b");
        const bool same = g.strs["general.architecture"] == "qwen35" && g.ints["general.alignment"] == 32 &&
                          g.floats["test.float"] == 2.5f && g.iarrs["test.array"] == std::vector<int64_t>{7, 8, 9} && a.nbytes == 32 &&
                          b.nbytes == 288 && ((const float *) a.data)[3] == 4.5f && b.data[287] == (uint8_t) 287 &&
                          b.data == a.data + 32 && (size_t) (b.data + b.nbytes - g.base) == g.size;
        g_n++;
        if (!same) g_fail++;
        printf("%s valid file contents\n", same ? "ok  " : "FAIL");
    }
    // every truncation: the last tensor's data ends at the end of the file
    if (selected("every truncation")) {
        int trunc_bad = 0;
        for (size_t n = 0; n < good.size(); n++)
            if (parse(Bytes(good.begin(), good.begin() + n)) != 0) trunc_bad++;
        g_n++;
        if (trunc_bad) g_fail++;
        printf("%s every truncation (%zu lengths) throws%s\n", trunc_bad ? "FAIL" : "ok  ", good.size(), trunc_bad ? " (some parsed)" : "");
    }

    auto with = [&](auto f) { Spec s = ok; f(s); return build(s); };
    // short integer arrays (File::iarrs): read at their element width, signed types sign-extended; f32 / f64 arrays and
    // arrays of more than 16 elements are not kept
    if (selected("integer arrays")) {
        struct IA { uint32_t type; uint64_t n; std::vector<uint32_t> vals; std::vector<int64_t> want; bool kept; };
        const std::vector<uint32_t> w3 = {0xffffffffu, 11, 10}, w4 = {0xffffffffu, 0xffffffffu, 11, 0};
        const IA cases[] = {{4, 3, w3, {4294967295ll, 11, 10}, true}, {5, 3, w3, {-1, 11, 10}, true},
                            {2, 6, w3, {65535, 65535, 11, 0, 10, 0}, true}, {3, 6, w3, {-1, -1, 11, 0, 10, 0}, true},
                            {0, 12, w3, {255, 255, 255, 255, 11, 0, 0, 0, 10, 0, 0, 0}, true},
                            {1, 12, w3, {-1, -1, -1, -1, 11, 0, 0, 0, 10, 0, 0, 0}, true}, {7, 12, w3, {255, 255, 255, 255, 11, 0, 0, 0, 10, 0, 0, 0}, true},
                            {10, 2, w4, {-1, 11}, true}, {11, 2, w4, {-1, 11}, true}, {4, 0, {}, {}, true},
                            {6, 3, w3, {}, false}, {12, 2, w4, {}, false}, {4, 17, std::vector<uint32_t>(17, 1), {}, false}};
        int bad = 0;
        for (const IA & c : cases) {
            if (parse(with([&](Spec & s) { s.arr_type = c.type; s.arr_n = c.n; s.arr_vals = c.vals; })) != 1) {
                bad++;
                continue;
            }
            gguf::File g(g_path);
            const auto it = g.iarrs.find("test.array");
            if ((it != g.iarrs.end()) != c.kept || (c.kept && it->second != c.want)) bad++;
        }
        g_n++;
        if (bad) g_fail++;
        printf("%s integer arrays (%d of %zu cases wrong)\n", bad ? "FAIL" : "ok  ", bad, sizeof cases / sizeof cases[0]);
    }
    expect("alignment 0", with([](Spec & s) { s.align = 0; }), 0);
    expect("alignment 3", with([](Spec & s) { s.align = 3; }), 0);
    expect("alignment 2^31", with([](Spec & s) { s.align = 1u << 31; }), 0);
    expect("alignment 16 (offsets still aligned)", with([](Spec & s) { s.align = 16; }), 1);
    expect("version 1", with([](Spec & s) { s.version = 1; }), 0);
    expect("big-endian version", with([](Spec & s) { s.version = 3u << 24; }), 0);
    expect("2^62 kvs", with([](Spec & s) { s.n_kv_hdr = 1ull << 62; }), 0);
    expect("2^62 tensors", with([](Spec & s) { s.n_tensors_hdr = 1ull << 62; }), 0);
    expect("more kvs than written", with([](Spec & s) { s.n_kv_hdr = 6; }), 0);
    expect("array of 2^62 u32", with([](Spec & s) { s.arr_n = 1ull << 62; }), 0);
    expect("array of 2^61 + 1 u64 (byte count wraps)", with([](Spec & s) { s.arr_type = 10; s.arr_n = (1ull << 61) + 1; }), 0);
    expect("nested array", with([](Spec & s) { s.arr_type = 9; }), 0);
    expect("array of type 13", with([](Spec & s) { s.arr_type = 13; }), 0);
    expect("2^62 strings", with([](Spec & s) { s.str_arr_n = 1ull << 62; }), 0);
    expect("string of 2^64 - 1 bytes", with([](Spec & s) { s.str_len = ~0ull; }), 0);
    expect("string past the end", with([](Spec & s) { s.str_len = 1ull << 40; }), 0);
    expect("duplicate key", with([](Spec & s) { s.dup_key = true; }), 0);
    expect("duplicate tensor", with([](Spec & s) { s.b_name = "a"; }), 0);
    expect("0 dimensions", with([](Spec & s) { s.a_dims = 0; }), 0);
    expect("5 dimensions", with([](Spec & s) { s.a_dims = 5; }), 0);
    expect("4 dimensions", with([](Spec & s) { s.a_dims = 4; }), 1);
    expect("dimension 2^63", with([](Spec & s) { s.b_ne1 = 1ull << 63; }), 0);
    expect("elements overflow int64", with([](Spec & s) { s.b_ne0 = 1ull << 40; s.b_ne1 = 1ull << 30; }), 0);
    expect("row bytes overflow", with([](Spec & s) { s.a_ne0 = (1ull << 62) + 8; }), 0);
    expect("unknown type", with([](Spec & s) { s.b_type = 99; }), 0);
    expect("row not whole blocks", with([](Spec & s) { s.b_ne0 = 300; }), 0);
    expect("tensor past the end", with([](Spec & s) { s.b_ne1 = 3; }), 0);
    expect("offset past the end", with([](Spec & s) { s.b_off = 1ull << 40; }), 0);
    expect("offset + size wraps", with([](Spec & s) { s.b_off = ~0ull - 31; }), 0);
    expect("unaligned offset", with([](Spec & s) { s.a_off = 4; }), 0);
    expect("not a gguf file", Bytes(64, 'x'), 0);
    expect("empty file", Bytes(), 0);
    // dimensions: none may be 0 (an empty tensor, nbytes 0, would pass as any shape that starts with its other dims)
    expect("dimension 0", with([](Spec & s) { s.b_ne1 = 0; }), 0);
    expect("dimension 0 in a third dimension", with([](Spec & s) { s.b_more = {0}; }), 0);
    expect("row of 0", with([](Spec & s) { s.a_ne0 = 0; }), 0);
    expect("third dimension 1", with([](Spec & s) { s.b_more = {1}; }), 1);
    // size lies: the bytes a tensor's dims declare must be in the file (the size is never taken from elsewhere)
    expect("size lie: one more row than the data", with([](Spec & s) { s.b_ne1 = 3; }), 0);
    expect("size lie: a wider row", with([](Spec & s) { s.b_ne0 = 512; }), 0);
    expect("size lie: a third dimension of 2", with([](Spec & s) { s.b_more = {2}; }), 0);
    // (a size lie that stays inside the file parses, like an overlap; the shape-checked read below refuses it)
    expect("size lie inside the file: a as 80 values over b", with([](Spec & s) { s.a_ne0 = 80; }), 1);
    // offset lies: offset + size inside the data section (an overlap stays inside the file: allowed here, harmless on a
    // read-only mapping; ggml's gguf.cpp refuses it, as it wants the tensors contiguous and in order)
    expect("offset lie: b one block later", with([](Spec & s) { s.b_off = 64; }), 0);
    expect("offset lie: b at 2^63", with([](Spec & s) { s.b_off = 1ull << 63; }), 0);
    expect("offset lie: b over a (overlap)", with([](Spec & s) { s.b_off = 0; }), 1);
    expect("offset lie: a at the end of the data", with([](Spec & s) { s.a_off = 320; }), 0);
    // the loader's shape-checked reads (File::get(name, dims)): a tensor of the right name and another shape is refused
    expect_get("get: the expected shape", good, "b", {256, 2}, 1);
    expect_get("get: 1-D a as [8]", good, "a", {8}, 1);
    expect_get("get: 1-D a as [8, 1]", good, "a", {8, 1}, 1);
    expect_get("get: fewer rows than expected", good, "b", {256, 3}, 0);
    expect_get("get: more rows than expected", good, "b", {256, 1}, 0);
    expect_get("get: same bytes, other row length", good, "b", {512, 1}, 0);
    expect_get("get: a as [4, 2]", good, "a", {4, 2}, 0);
    expect_get("get: [256, 2, 1] as [256, 2]", with([](Spec & s) { s.b_more = {1}; }), "b", {256, 2}, 1);
    expect_get("get: [256, 1, 2] as [256, 2]", with([](Spec & s) { s.b_ne1 = 1; s.b_more = {2}; }), "b", {256, 2}, 0);
    expect_get("get: a missing tensor", good, "c", {8}, 0);
    expect_get("get: a as 80 values over b, read as [8]", with([](Spec & s) { s.a_ne0 = 80; }), "a", {8}, 0);

    // random corruption of the valid file: anything may happen except a crash (sanitizers catch out-of-bounds reads)
    std::mt19937_64 rng(1234);
    const int iters = !selected("random") ? 0 : argc > 1 ? atoi(argv[1]) : 20000;
    int parsed = 0;
    for (int it = 0; it < iters; it++) {
        Bytes b = good;
        const int k = 1 + (int) (rng() % 4);
        for (int j = 0; j < k; j++) {
            const size_t p = rng() % b.size();
            switch (rng() % 4) {
                case 0: b[p] ^= (uint8_t) (1u << (rng() % 8)); break;
                case 1: b[p] = (uint8_t) rng(); break;
                case 2: { const uint64_t v = (rng() % 2) ? rng() : (1ull << (rng() % 64)) - (rng() % 3);
                          memcpy(&b[p], &v, std::min<size_t>(8, b.size() - p)); break; }
                default: b.resize(p); break;
            }
            if (b.empty()) break;
        }
        parsed += parse(b);
    }
    if (iters) printf("ok   %d random corruptions: %d parsed, %d threw, no crash\n", iters, parsed, iters - parsed);
    std::remove(g_path.c_str());
    printf("%s: %d of %d checks failed\n", g_fail ? "FAILED" : "PASSED", g_fail, g_n);
    return g_fail ? 1 : 0;
}

// mk_vision command line: encode images on the CPU, report the grid and time, optionally save the rows.
//   mkv_cli -m model.gguf --mmproj mmproj.gguf [-t threads] [--min-tokens N] [--max-tokens N] [--fa -1|0|1] [-r reps]
//           [--backend auto|cpu|mtmd] [-o out_prefix] image...
// -o: rows of image i -> <out_prefix>.<i>.f32 ([nx * ny][n_embd] little-endian fp32, raster order)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mk_vision.h"

static std::vector<unsigned char> read_file(const char * path) {
    std::vector<unsigned char> b;
    FILE * f = fopen(path, "rb");
    if (!f) return b;
    fseek(f, 0, SEEK_END);
    b.resize((size_t) ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), f) != b.size()) b.clear();
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    mkv_params p = mkv_params_default();
    const char * model = nullptr, * mmproj = nullptr, * out = nullptr;
    int reps = 1;
    std::vector<const char *> imgs;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "-t") p.n_threads = atoi(next());
        else if (a == "--min-tokens") p.min_tokens = atoi(next());
        else if (a == "--max-tokens") p.max_tokens = atoi(next());
        else if (a == "--fa") p.flash_attn = atoi(next());
        else if (a == "-r") reps = atoi(next());
        else if (a == "-o") out = next();
        else if (a == "-v") p.verbose = 1;
        else if (a == "--backend") {
            const std::string b = next();
            p.backend = b == "cpu" ? MKV_BACKEND_CPU : b == "mtmd" ? MKV_BACKEND_MTMD : MKV_BACKEND_AUTO;
        }
        else imgs.push_back(argv[i]);
    }
    if (!model || !mmproj || imgs.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --mmproj mmproj.gguf [-t N] [--min-tokens N] [--max-tokens N] [--fa -1|0|1] "
                        "[--backend auto|cpu|mtmd] [-r reps] [-o prefix] image...\n", argv[0]);
        return 1;
    }
    mkv_ctx * c = mkv_new(mmproj, model, p);
    if (!c) {
        fprintf(stderr, "error: %s\n", mkv_last_error());
        return 2;
    }
    const int ne = mkv_n_embd(c);
    printf("mmproj %s: %s backend, %d threads, rows of %d\n", mmproj, mkv_backend(c) == MKV_BACKEND_CPU ? "cpu" : "mtmd",
           mkv_n_threads(c), ne);
    for (size_t k = 0; k < imgs.size(); k++) {
        const std::vector<unsigned char> b = read_file(imgs[k]);
        if (b.empty()) {
            fprintf(stderr, "%s: cannot read\n", imgs[k]);
            return 3;
        }
        double best = 1e30, sum = 0;
        int nx = 0, ny = 0, n = 0;
        std::vector<float> rows;
        for (int r = 0; r < reps; r++) {
            mkv_image * im = mkv_image_load(c, b.data(), b.size());
            if (!im || mkv_image_encode(c, im) != 0) {
                fprintf(stderr, "%s: %s\n", imgs[k], mkv_last_error());
                return 4;
            }
            n = mkv_image_grid(im, &nx, &ny);
            const double ms = mkv_image_encode_ms(im);
            best = std::min(best, ms);
            sum += ms;
            if (r == reps - 1) rows.assign(mkv_image_rows(im), mkv_image_rows(im) + (size_t) n * ne);
            mkv_image_free(im);
        }
        double l1 = 0;
        for (float v : rows) l1 += std::fabs(v);
        printf("%s: %d x %d = %d tokens, encode best %.1f ms mean %.1f ms, mean |row| %.6f\n", imgs[k], nx, ny, n, best, sum / reps,
               l1 / rows.size());
        if (out) {
            const std::string path = std::string(out) + "." + std::to_string(k) + ".f32";
            FILE * f = fopen(path.c_str(), "wb");
            if (!f || fwrite(rows.data(), sizeof(float), rows.size(), f) != rows.size()) {
                fprintf(stderr, "%s: cannot write\n", path.c_str());
                return 5;
            }
            fclose(f);
        }
        fflush(stdout);
    }
    mkv_free(c);
    return 0;
}

// C wrappers around the per-token kernels, for tests/test_ik_kernels.py (built into libinkling_test.so only).
#include "ik_kernels.cuh"

extern "C" {

void ikt_embed(const uint8_t * row, const float * w, float * out, int n, float eps) { ik::embed(row, w, out, n, eps, 0); }
void ikt_embed_type(int emb_type, const uint8_t * row, const float * w, float * out, int n, float eps) {
    ik::embed(row, w, out, n, eps, 0, emb_type);
}

void ikt_mix(const float * const * states, int n_states, const float * query, int n_q, float * out, int n) {
    ik::MixStates s{};
    for (int i = 0; i < n_states; ++i) s.p[i] = states[i];
    s.n = n_states;
    ik::mix(s, query, n_q, out, n, 0);
}

void ikt_control(const float * cur, const float * msg, const float * down, const float * gain, const float * forget, float * out,
                 int n, int rank) {
    ik::ControlArgs a{};
    a.cur[0] = cur; a.msg[0] = msg; a.down[0] = down; a.gain[0] = gain; a.forget[0] = forget; a.out[0] = out;
    static float * scratch = nullptr;
    if (!scratch) cudaMalloc(&scratch, sizeof(float) * 4 * 64);
    ik::control(a, 1, n, rank, scratch, 0);
}

void ikt_rmsnorm(const float * x, const float * w, float * out, int n, float eps) {
    ik::NormArgs a{};
    a.x[0] = x; a.w[0] = w; a.out[0] = out;
    ik::rmsnorm(a, 1, n, eps, 0);
}

void ikt_matvec_f32(const float * w, const float * x, float * y, int rows, int n) { ik::matvec_f32(w, x, y, rows, n, 0); }

void ikt_attn_prep(ik::AttnPrepArgs * a) { ik::attn_prep(*a, 0); }

void ikt_attention(ik::AttnArgs * a) { ik::attention(*a, 0); }

void ikt_conv_residual(const float * x, float * hist, const float * w, const float * add, float * out, int n) {
    ik::conv_residual(x, hist, w, add, out, n, 0);
}

void ikt_router(const float * w, const float * bias, const float * x, int32_t * ids, float * wts, int n_expert, int n, int k, float scale) {
    static float * scratch = nullptr;
    if (!scratch) cudaMalloc(&scratch, sizeof(float) * 256);
    ik::router(w, bias, x, ids, wts, n_expert, n, k, scale, scratch, 0);
}

int ikt_sizeof_attn_prep() { return (int) sizeof(ik::AttnPrepArgs); }
int ikt_sizeof_attn() { return (int) sizeof(ik::AttnArgs); }

}

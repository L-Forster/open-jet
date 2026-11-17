// wv device re-layout (MK_LOAD_GPURL): the kernels of wv_host.cuh's permutation on the GPU (launched by
// mkl::wv_dev_launch). Included by qwen38_mk.cu inside its anonymous namespace.
#pragma once

__global__ void wv_fm_dev(const uint64_t * __restrict__ src, uint64_t * __restrict__ dst, const uint32_t * __restrict__ tab, int W,
                          int ngrp, int64_t rbw, int segw, int64_t total) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t gi = i / W;
        const int j = (int) (i - gi * W);
        const int64_t tile = gi / ngrp;
        const int grp = (int) (gi - tile * ngrp);
        const uint32_t t = tab[j];
        dst[i] = src[(tile * 16 + (t >> 24)) * rbw + (int64_t) grp * segw + (t & 0xffffff)];
    }
}
// tm(): dst block ((tile nbk + i) 16 + rr) = src row tile 16 + rr, block i (bsw words per block, rowbw words per row)
__global__ void wv_tm_dev(const uint32_t * __restrict__ src, uint32_t * __restrict__ dst, int64_t rowbw, int bsw, int nbk, int64_t total) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        int64_t q = i / bsw;
        const int j = (int) (i - q * bsw);
        const int rr = (int) (q % 16);
        q /= 16;
        const int b = (int) (q % nbk);
        const int64_t tile = q / nbk;
        dst[i] = src[(tile * 16 + rr) * rowbw + (int64_t) b * bsw + j];
    }
}
// one thread per (tile, super-block, row of the tile): that row's bytes of the Q6F_UNIT unit (wv_relayout's Q6_K loop)
__global__ void wv_q6f_dev(const uint8_t * __restrict__ w, const uint8_t * __restrict__ meta, uint8_t * __restrict__ out, int N, int nb, int rb) {
    const int mrow = nb * Q6_META;
    const int64_t items = (int64_t) (N / 16) * nb * 16;
    for (int64_t it = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; it < items; it += (int64_t) gridDim.x * blockDim.x) {
        const int rr = (int) (it % 16);
        const int64_t q = it / 16;
        const int sb = (int) (q % nb);
        const int64_t tile = q / nb;
        const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
        uint8_t * u = out + ((size_t) tile * nb + sb) * Q6F_UNIT;
        const uint8_t * blk = w + (size_t) r * rb + (size_t) sb * Q6_BLK;
        const uint8_t * mp = meta + (size_t) r * mrow + (sb >> 1) * 2 * Q6_META;
        for (int k = 0; k < 16; k++) u[32 * g + 16 * half + k] = mp[16 * (sb & 1) + k];
        for (int k = 0; k < 2; k++) u[Q6F_D + 4 * g + 2 * half + k] = mp[32 + 2 * (sb & 1) + k];
        for (int tq = 0; tq < 4; tq++) {
            const int l = 4 * g + tq;
            for (int p = 0; p < 4; p++) {
                for (int k = 0; k < 4; k++) u[Q6F_LO + 512 * p + 16 * l + 8 * half + k] = blk[32 * p + 4 * tq + k];
                for (int k = 0; k < 4; k++) u[Q6F_LO + 512 * p + 16 * l + 8 * half + 4 + k] = blk[32 * p + 16 + 4 * tq + k];
                uint8_t * hd = u + Q6F_HI + 256 * p + 8 * l + 4 * half;
                for (int mm = 0; mm < 4; mm++) {
                    const uint8_t * hq = blk + 128 + 8 * (2 * p) + mm;
                    const int h00 = (hq[0] >> (2 * tq)) & 3, h10 = (hq[4] >> (2 * tq)) & 3; // (part, c) = (0, 0), (1, 0)
                    const int h01 = (hq[8] >> (2 * tq)) & 3, h11 = (hq[12] >> (2 * tq)) & 3; // (0, 1), (1, 1)
                    hd[mm] = (uint8_t) ((h00 << 4) | (h10 << 6) | h01 | (h11 << 2));
                }
            }
        }
    }
}

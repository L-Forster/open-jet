#pragma once

#include <cuda_runtime.h>

// Resource constants are used by both host launch code and device code. Single-architecture
// sm_75 builds must define this in both passes (the Makefile does so).
#ifndef QWEN_MK_SM75
#define QWEN_MK_SM75 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && !QWEN_MK_SM75
#error "sm_75 builds require -DQWEN_MK_SM75=1 for matching host/device resource layouts"
#endif

namespace mk_cuda {

template <int Bytes>
__device__ __forceinline__ void copy(unsigned dst, const void * src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "copy size");
#if __CUDA_ARCH__ >= 800
    if constexpr (Bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    else if constexpr (Bytes == 8)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src) : "memory");
#else
    // All callers provide size-aligned addresses and use a CTA barrier before cross-thread
    // consumption. These copies complete in-thread; commit/wait need no extra CTA barriers.
    void * p = reinterpret_cast<void *>(__cvta_shared_to_generic(dst));
    if constexpr (Bytes == 16) *static_cast<uint4 *>(p) = *static_cast<const uint4 *>(src);
    else if constexpr (Bytes == 8) *static_cast<uint2 *>(p) = *static_cast<const uint2 *>(src);
    else *static_cast<unsigned *>(p) = *static_cast<const unsigned *>(src);
#endif
}

__device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void wait() {
    static_assert(N >= 0, "async group count");
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

__device__ __forceinline__ void mma_f32(float (&c)[4], unsigned a0, unsigned a1,
                                      unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // m16n8k16's first two A registers and first B register are the k=0..7
    // fragment; the remaining registers are k=8..15. C/D lane layouts are identical.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a2), "r"(a3), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(unsigned (&d)[2], const unsigned (&a)[4],
                                      unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

} // namespace mk_cuda

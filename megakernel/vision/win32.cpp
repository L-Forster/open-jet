// Win32 side of vision/win32.h (Windows build only). Processor group 0 only (up to 64 logical CPUs).
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>

#include <vector>

#include "win32.h"

namespace mkv_win {

bool avx2_fma() {
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!fma || !osxsave || !avx || (_xgetbv(0) & 6) != 6) return false; // (XMM + YMM state saved by the OS)
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
}

namespace {
// the logical-CPU mask of each physical core (group 0)
std::vector<unsigned long long> core_masks() {
    std::vector<unsigned long long> masks;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return masks;
    std::vector<unsigned char> buf(len);
    auto * info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return masks;
    for (DWORD off = 0; off < len;) {
        auto * e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1 && e->Processor.GroupMask[0].Group == 0)
            masks.push_back(static_cast<unsigned long long>(e->Processor.GroupMask[0].Mask));
        off += e->Size;
    }
    return masks;
}
} // namespace

std::vector<int> cpu_order() {
    std::vector<int> first, second;
    for (unsigned long long m : core_masks()) {
        bool lead = true;
        for (int c = 0; c < 64; c++)
            if (m >> c & 1) {
                (lead ? first : second).push_back(c);
                lead = false;
            }
    }
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

int physical_cores() { return static_cast<int>(core_masks().size()); }

void pin_thread(void * native_handle, int cpu) {
    if (cpu >= 0 && cpu < 64) SetThreadAffinityMask(static_cast<HANDLE>(native_handle), DWORD_PTR(1) << cpu);
}

} // namespace mkv_win
#endif // _WIN32

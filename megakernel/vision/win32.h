// Windows build only: the CPU queries of the vision encoder, implemented with Win32 in vision/win32.cpp (no <windows.h>
// here, so its macros never meet the encoder's identifiers). Linux reads the same from sysfs (vit_cpu.cpp).
#pragma once
#ifdef _WIN32
#include <vector>

namespace mkv_win {
bool avx2_fma();                                // CPU and OS support for AVX2 + FMA (CPUID, XGETBV)
std::vector<int> cpu_order();                   // the first SMT sibling of each core, then the other siblings
int physical_cores();                           // 0 if unknown
void pin_thread(void * native_handle, int cpu); // best effort
} // namespace mkv_win
#endif

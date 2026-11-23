// Win32 implementations for inkling/engine/compat_win32.h (Windows build only).
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <io.h>

#include "compat_win32.h"

void * mk_win_mmap_ro(int fd, size_t len, long long off) {
    const HANDLE fh = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (fh == INVALID_HANDLE_VALUE || len == 0) return MAP_FAILED;
    const HANDLE m = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) return MAP_FAILED;
    void * p = MapViewOfFile(m, FILE_MAP_READ, static_cast<DWORD>(static_cast<unsigned long long>(off) >> 32),
                             static_cast<DWORD>(off), len);
    CloseHandle(m);
    return p ? p : MAP_FAILED;
}

int mk_win_munmap(void * p) { return UnmapViewOfFile(p) ? 0 : -1; }

bool mk_win_module_path(const void * addr, char * out, size_t cap) {
    HMODULE hm = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCSTR>(addr), &hm))
        return false;
    const DWORD n = GetModuleFileNameA(hm, out, static_cast<DWORD>(cap));
    return n > 0 && n < cap;
}

ssize_t mk_win_pread(int fd, void * buf, size_t count, long long off) {
    const HANDLE fh = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (fh == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(static_cast<unsigned long long>(off) & 0xffffffffull);
    ov.OffsetHigh = static_cast<DWORD>(static_cast<unsigned long long>(off) >> 32);
    DWORD got = 0;
    const DWORD want = count > 0x7fffffffu ? 0x7fffffffu : static_cast<DWORD>(count);
    if (!ReadFile(fh, buf, want, &got, &ov)) return -1;
    return static_cast<ssize_t>(got);
}
#endif  // _WIN32

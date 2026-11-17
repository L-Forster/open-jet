// Windows (MSVC) stand-ins for the POSIX calls of the host code: open / read / close / dup / unlink / getpid (the CRT's
// POSIX names) on binary-mode file descriptors, a read-only mmap, stat / fstat with 64-bit sizes, realpath, mkdir,
// setenv, PATH_MAX, ssize_t, environ.
// Included instead of <fcntl.h> <sys/mman.h> <sys/stat.h> <unistd.h> when _WIN32 is defined, so the Linux build never
// sees it. No <windows.h> here (its macros would meet the engine's identifiers): the Win32 calls are in
// compat_win32.cpp.
#pragma once
#ifdef _WIN32

#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>

typedef intptr_t ssize_t;

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// the CRT opens in text mode by default: hashed and mapped files must be read as bytes
#undef O_RDONLY
#define O_RDONLY (_O_RDONLY | _O_BINARY)

// the default struct stat has a 32-bit st_size (a GGUF is ~17 GB): struct stat / stat() / fstat() -> the 64-bit forms
#define stat _stat64
#define fstat _fstat64

#ifndef environ
#define environ _environ
#endif

#define PROT_READ 1
#define MAP_SHARED 1
#define MAP_FAILED ((void *) -1)

// read-only view of len bytes of an open file from offset off (MAP_FAILED on error); the view outlives the descriptor
void * mk_win_mmap_ro(int fd, size_t len, long long off);
int mk_win_munmap(void * p);
// the path of the module (DLL or EXE) that contains addr; false if unknown
bool mk_win_module_path(const void * addr, char * out, size_t cap);
// sleep about us microseconds (a high-resolution waitable timer; std::this_thread::sleep_for rounds up to the timer tick)
void mk_win_sleep_us(unsigned us);

// UTF-8 -> UTF-16; false for text that is not UTF-8 (the CLI's argv comes in the ANSI code page)
inline bool mk_win_wide(const char * s, std::wstring & w) {
    w.clear();
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(s); *p;) {
        unsigned c = *p++;
        const unsigned n = c < 0x80 ? 0 : c >= 0xC2 && c < 0xE0 ? 1 : c >= 0xE0 && c < 0xF0 ? 2 : c >= 0xF0 && c < 0xF5 ? 3 : 4;
        if (n == 4) return false;
        if (n) c &= 0x3F >> n;
        for (unsigned i = 0; i < n; i++, p++) {
            if ((*p & 0xC0) != 0x80) return false;
            c = c << 6 | (*p & 0x3F);
        }
        if ((n == 2 && c < 0x800) || (n == 3 && (c < 0x10000 || c > 0x10FFFF)) || (c >= 0xD800 && c < 0xE000)) return false;
        if (c >= 0x10000) {
            w += static_cast<wchar_t>(0xD800 + ((c - 0x10000) >> 10));
            w += static_cast<wchar_t>(0xDC00 + ((c - 0x10000) & 0x3FF));
        } else {
            w += static_cast<wchar_t>(c);
        }
    }
    return true;
}
// the server's paths are UTF-8, which the CRT's narrow open would read in the ANSI code page
inline int mk_open(const char * path, int flags) {
    std::wstring w;
    return mk_win_wide(path, w) ? _wopen(w.c_str(), flags) : _open(path, flags);
}

inline void * mmap(void *, size_t len, int, int, int fd, long long off) { return mk_win_mmap_ro(fd, len, off); }
inline int munmap(void * p, size_t) { return mk_win_munmap(p); }
inline char * realpath(const char * path, char * resolved) { return _fullpath(resolved, path, PATH_MAX); }
inline int mkdir(const char * path, int) { return _mkdir(path); }
inline int setenv(const char * name, const char * value, int overwrite) {
    return !overwrite && getenv(name) ? 0 : _putenv_s(name, value);
}

#endif // _WIN32

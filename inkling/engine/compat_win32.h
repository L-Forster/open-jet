// Windows (MSVC) stand-ins for POSIX calls used by inkling/engine (mirrors megakernel/compat_win32.h).
#pragma once
#ifdef _WIN32

#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
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

#undef O_RDONLY
#define O_RDONLY (_O_RDONLY | _O_BINARY)
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#define stat _stat64
#define fstat _fstat64

#define PROT_READ 1
#define MAP_SHARED 1
#define MAP_FAILED ((void *) -1)

void * mk_win_mmap_ro(int fd, size_t len, long long off);
int mk_win_munmap(void * p);
bool mk_win_module_path(const void * addr, char * out, size_t cap);
ssize_t mk_win_pread(int fd, void * buf, size_t count, long long off);

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

inline int mk_open(const char * path, int flags) {
    std::wstring w;
    return mk_win_wide(path, w) ? _wopen(w.c_str(), flags) : _open(path, flags);
}

inline void * mmap(void *, size_t len, int, int, int fd, long long off) { return mk_win_mmap_ro(fd, len, off); }
inline int munmap(void * p, size_t) { return mk_win_munmap(p); }
inline ssize_t pread(int fd, void * buf, size_t count, long long off) { return mk_win_pread(fd, buf, count, off); }

#endif  // _WIN32

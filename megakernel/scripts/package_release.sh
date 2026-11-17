#!/usr/bin/env bash
# Package a Linux release (tarball + sha256) from a finished `make` of this tree.
#   scripts/package_release.sh --version v2026.10.22 [--llama ../llama.cpp] [--out dist]
# The llama.cpp build should be CPU-only and portable (docs/BUILDING.md, "Prebuilt packages"). Needs patchelf
# (`pip install patchelf`): the binaries are re-pointed at their own directory ($ORIGIN) instead of the build tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/../llama.cpp" OUT="$ROOT/dist" VERSION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --llama) LLAMA="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
cd "$ROOT"
LLAMA="$(cd "$LLAMA" && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
[ -n "$VERSION" ] || VERSION="$(git describe --tags --exact-match 2>/dev/null || echo "dev-$(git rev-parse --short=12 HEAD)")"
command -v patchelf > /dev/null || { echo "patchelf not found (pip install patchelf)" >&2; exit 1; }

ELF_ROOT=(qwen38_mk libqwen38mk.so libqwen38vision.so)
ELF_TOOLS=(tools/llama_ref tools/vocab_freq tools/mkv_cli tools/vision_ref)
for f in "${ELF_ROOT[@]}" "${ELF_TOOLS[@]}"; do [ -f "$f" ] || { echo "missing $f: run make (with the vision library) first" >&2; exit 1; }; done
COMMIT="$(sed -n 's/^#define MK_BUILD_COMMIT "\(.*\)"/\1/p' build/mk_build_info.h)"
FLAGS="$(sed -n 's/^#define MK_BUILD_FLAGS "\(.*\)"/\1/p' build/mk_build_info.h)"
[ "$COMMIT" = "$(git rev-parse --short=12 HEAD)" ] || { echo "the build is of '$COMMIT', HEAD is $(git rev-parse --short=12 HEAD): run make again" >&2; exit 1; }

# the GPU code the build holds (Makefile ARCH: the first GPU's, unless given)
ARCHS="$("$(dirname "$NVCC")/cuobjdump" --list-elf libqwen38mk.so | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | paste -sd- - || true)"
[ -n "$ARCHS" ] || { echo "no GPU code found in libqwen38mk.so (cuobjdump --list-elf)" >&2; exit 1; }
GPU_NOTE=""
[ "$ARCHS" = sm_86 ] && GPU_NOTE=" (compute capability 8.6; 8.9 runs it, untested)"

NAME="qwen38-megakernel-$VERSION-linux-x64-${ARCHS//_/}"
STAGE="$OUT/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE/tools" "$STAGE/licenses" "$STAGE/server" "$STAGE/scripts"
cp "${ELF_ROOT[@]}" "$STAGE/"
cp "${ELF_TOOLS[@]}" "$STAGE/tools/"
# llama.cpp's libraries under the names the binaries ask for (their SONAMEs)
LIBS=()
for l in libllama libggml libggml-base libggml-cpu libmtmd; do
    f=$(ls "$LLAMA/build/bin/$l.so."[0-9] 2>/dev/null | head -n1)
    [ -n "$f" ] || { echo "missing $LLAMA/build/bin/$l.so.N" >&2; exit 1; }
    cp -L "$f" "$STAGE/"; LIBS+=("$(basename "$f")")
done
for f in "${ELF_ROOT[@]}" "${LIBS[@]}"; do patchelf --set-rpath '$ORIGIN' "$STAGE/$f"; done
for f in "${ELF_TOOLS[@]}"; do patchelf --set-rpath '$ORIGIN/..' "$STAGE/$f"; done

cp server/*.py "$STAGE/server/"
cp scripts/serve.sh scripts/exactness.py scripts/l2_sweep.py "$STAGE/scripts/"
cp -r docs "$STAGE/"
cp README.md LICENSE THIRD_PARTY_NOTICES.md requirements.txt draft_vocab_32k.bin "$STAGE/"
cp "$LLAMA/LICENSE" "$STAGE/licenses/llama.cpp-LICENSE"
for f in "$LLAMA"/licenses/*; do [ -f "$f" ] && cp "$f" "$STAGE/licenses/llama.cpp-$(basename "$f")"; done
sed -n '/^This software is available under 2 licenses/,$p' "$LLAMA/vendor/stb/stb_image.h" > "$STAGE/licenses/stb_image-LICENSE"
sed -n '/^This software is available as a choice of the following licenses/,$p' "$LLAMA/vendor/miniaudio/miniaudio.h" > "$STAGE/licenses/miniaudio-LICENSE"
for f in "$STAGE"/licenses/*; do [ -s "$f" ] || { echo "empty license file $f" >&2; exit 1; }; done

# every library a binary needs is in the package or a system library; nothing points back into a build tree
fail=0
while IFS= read -r f; do
    while IFS= read -r line; do
        case "$line" in
            *"not found"*) echo "$f: $line" >&2; fail=1 ;;
            *"=> $STAGE/"*|*"=> /lib/"*|*"=> /usr/lib/"*|*"=> /lib64/"*|*linux-vdso*|*ld-linux*|"") ;;
            *) echo "$f: outside the package: $line" >&2; fail=1 ;;
        esac
    done < <(env -u LD_LIBRARY_PATH ldd "$f" | sed 's/^[[:space:]]*//')
done < <(find "$STAGE" -type f \( -name '*.so*' -o -name qwen38_mk -o -path '*/tools/*' \))
[ $fail = 0 ] || { echo "dependency check failed" >&2; exit 1; }
GLIBC=$(find "$STAGE" -type f \( -name '*.so*' -o -name qwen38_mk -o -path '*/tools/*' \) -exec objdump -T {} \; |
        grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -n1)
GLIBCXX=$(find "$STAGE" -type f \( -name '*.so*' -o -name qwen38_mk -o -path '*/tools/*' \) -exec objdump -T {} \; |
          grep -oE 'GLIBCXX_[0-9.]+' | sort -Vu | tail -n1)

cat > "$STAGE/BUILD_INFO.txt" << EOF
qwen38-megakernel $VERSION, Linux x86-64
commit     $COMMIT (https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/tree/$COMMIT)
MKFLAGS    $FLAGS
GPU code   ${ARCHS//-/, }$GPU_NOTE
CUDA       $("$NVCC" --version | tail -n1)
compiler   $(g++ --version | head -n1)
llama.cpp  $(git -C "$LLAMA" describe --tags 2>/dev/null) (https://github.com/ggml-org/llama.cpp/tree/$(git -C "$LLAMA" describe --tags 2>/dev/null)), $(grep -E '^(GGML_(CUDA|NATIVE|AVX2|AVX512|OPENMP)):' "$LLAMA/build/CMakeCache.txt" | sed 's/:BOOL//' | tr '\n' ' ')
needs      $GLIBC, $GLIBCXX (libstdc++), an NVIDIA driver with CUDA 13.2 support, an x86-64 CPU with AVX2
built      $(date -u +%Y-%m-%dT%H:%MZ)
EOF
cat > "$STAGE/QUICKSTART.txt" << 'EOF'
qwen38-megakernel: Qwen3.8-27B on one 24 GB NVIDIA GPU, with an OpenAI / Anthropic / OpenAI Responses compatible server.

1. Python 3.10+:   pip install -r requirements.txt
2. The model:      pip install huggingface_hub
                   hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
3. Run:            scripts/serve.sh                     # http://127.0.0.1:8080, 120k context
                   scripts/serve.sh 8080 64000          # port, context
                   MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144   # the full 262k context (preset D)
                   MK_MODEL=/path/to/model.gguf scripts/serve.sh

Under WSL2 keep the model on the Linux filesystem (not /mnt/c). Client settings: docs/CLIENTS.md. Every setting:
docs/CONFIGURATION.md. What is in this package and what it needs: BUILD_INFO.txt, THIRD_PARTY_NOTICES.md.
EOF

tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "$OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1)), needs $GLIBC / $GLIBCXX"

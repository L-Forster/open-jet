LLAMA ?= ../llama.cpp
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50); sm_86 if nvidia-smi is missing
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
ARCH := sm_$(patsubst sm_%,%,$(ARCH))
# the engine's extra -D flags: none for the served build; debug / experiment builds, e.g. MKFLAGS=-DMK_SPEC_LOG=1; the
# opt-in builds in development: MKFLAGS=-DMK_MAXD=16 (the deep pass), -DMK_ATT_I8=1 (int8 decode attention),
# -DMK_L2_TUNE=1 (the L2 prefetch budgets from the environment).
# ptxas peaks at ~21 GiB of RAM (2026-11-03; ~26 GiB with -DMK_ATT_I8=1).
MKFLAGS ?=
# where libqwen38mk.so looks for libllama / libggml-base at run time; OpenJet's prebuilt release archives ship them
# next to the engine with RPATH='$$ORIGIN' (.github/workflows/engine.yml)
RPATH ?= $(abspath $(LLAMA)/build/bin)
# ptxas / host options as single tokens (-Xptxas=a,b): ccache does not know that nvcc's -Xptxas takes the next word, so
# with "-Xptxas -v" it moves -v out of its pair and its `nvcc -E` fails ("Unknown option '-fPIC'"): no caching at all.
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-dlcm=cg,-v -Xcompiler=-fPIC $(MKFLAGS)
# With NVCC="ccache nvcc": depend mode (keys = the whole command line + the -MD header list). ccache's default fallback
# hashes `nvcc -E`, which is only the device-side preprocessing, without the -D flags: a -D that changes only host-side
# code (#ifndef __CUDA_ARCH__, e.g. in the CUDA headers) would get the other build's object back. No effect without ccache.
export CCACHE_DEPEND ?= 1
# Most of the build is ptxas on the megakernel (one thread, ~21 GiB heap). It runs ~20% faster when glibc's malloc asks
# for transparent huge pages, which this tunable does (glibc 2.35+; nothing changes where THP is off). Same machine code.
export GLIBC_TUNABLES ?= glibc.malloc.hugetlb=1

all: qwen38_mk libqwen38mk.so tools/vocab_freq tools/llama_ref $(if $(wildcard $(LLAMA)/build/bin/libmtmd.so),vision)

# CPU vision encoder (no CUDA, no GPU memory) for server/mk_server.py --mmproj: vision/mk_vision.cpp (C API, image
# decoding, libmtmd backend) + on x86-64 vision/vit_cpu.cpp (the AVX2 / FMA backend, compiled for x86-64-v3 in its
# own object so that the library still loads on older CPUs, which then use libmtmd; -ffp-contract=off keeps the
# preprocessing's arithmetic exactly llama.cpp's). Built when llama.cpp's build has libmtmd (LLAMA_BUILD_TOOLS, its
# default); `make vision` builds it explicitly.
vision: libqwen38vision.so tools/mkv_cli tools/vision_ref
VISION_INC = -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/tools/mtmd -I$(LLAMA)/vendor
VISION_LIBS = -L$(LLAMA)/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$(abspath $(LLAMA)/build/bin)
ifeq ($(shell uname -m),x86_64)
VIT_OBJ = vision/vit_cpu.o
VIT_DEF =
else
VIT_OBJ =
VIT_DEF = -DMKV_NO_VIT
endif

vision/vit_cpu.o: vision/vit_cpu.cpp vision/vit_cpu.h
	g++ -O3 -std=c++17 -fPIC -march=x86-64-v3 -mtune=znver3 -ffp-contract=off -pthread $(VISION_INC) -c $< -o $@

libqwen38vision.so: vision/mk_vision.cpp vision/mk_vision.h $(VIT_OBJ)
	g++ -O2 -std=c++17 -shared -fPIC -pthread $(VIT_DEF) $(VISION_INC) vision/mk_vision.cpp $(VIT_OBJ) $(VISION_LIBS) -o $@

tools/mkv_cli: vision/mkv_cli.cpp vision/mk_vision.h libqwen38vision.so
	g++ -O2 -std=c++17 -Ivision $< -L. -lqwen38vision -Wl,-rpath,'$$ORIGIN/..' $(VISION_LIBS) -o $@

# test: vit_cpu against libmtmd (pixels, rows, time) on images; uses libmtmd's internal clip API (not shipped)
tools/vit_check: vision/vit_check.cpp vision/vit_cpu.o
	g++ -O2 -std=c++17 -pthread $(VISION_INC) -Ivision $< vision/vit_cpu.o $(VISION_LIBS) -o $@

# llama.cpp's next-token distributions for an image prompt + the image rows it fed (qwen38_mk --kld ... --kld-images)
tools/vision_ref: tools/vision_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread $(VISION_INC) $< $(VISION_LIBS) -o $@

# the engine's compile flags, rewritten only when they change: another MKFLAGS or ARCH rebuilds the engine's objects
build/nvflags: FORCE
	@mkdir -p build
	@echo '$(NVCC) $(NVFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv $@.tmp $@

# the device code and its launchers; ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu mk_shared.cuh mk_launch.h qwen38_mk.h gguf.h prefill.cuh prefill_gemm.cuh prefill_attn.cuh wv.cuh wv_dev.cuh \
		build/nvflags
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -MD -MF $@.d -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

# the host side (no device code: a change here does not rebuild qwen38_mk.o); the same flags, so the shared definitions agree
HOST_SRC = engine.cu ck_pool.cu loader.cu prefill_host.cu state_files.cu host_store.cu lanes.cu
HOST_HDR = engine_impl.h mk_shared.cuh mk_launch.h qwen38_mk.h gguf.h loader.cuh wv_host.cuh
HOST_OBJ = $(HOST_SRC:.cu=.o)
$(HOST_OBJ): %.o: %.cu $(HOST_HDR) build/nvflags
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -MD -MF $@.d -c $< -o $@

qwen38_mk: qwen38_mk.o $(HOST_OBJ) main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o $(HOST_OBJ) \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

# the build's commit and engine flags (capi.cpp mk_build_info, the server's /props); rewritten only when they change
build/mk_build_info.h: FORCE
	@mkdir -p build
	@printf '#define MK_BUILD_COMMIT "%s"\n#define MK_BUILD_FLAGS "%s"\n' \
		"$$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)$$(git diff --quiet HEAD -- 2>/dev/null || echo -dirty)" \
		"$(strip $(MKFLAGS))" > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv $@.tmp $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libqwen38mk.so: qwen38_mk.o $(HOST_OBJ) capi.cpp qwen38_mk.h build/mk_build_info.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -Ibuild -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o $(HOST_OBJ) \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker '$(RPATH)' -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

# llama.cpp's next-token distributions for the accuracy check (see "Correctness" in README.md)
tools/llama_ref: tools/llama_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

# test: the GGUF parser (gguf.h) on malformed files under AddressSanitizer / UBSan, CPU only: tests/gguf_fuzz
tests/gguf_fuzz: tests/gguf_fuzz.cpp gguf.h
	g++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-sanitize-recover=undefined $< -o $@

# README's exact-output claims on this build (scripts/exactness.py; the GPU to itself): make exactness MK_MODEL=<gguf>
exactness: qwen38_mk libqwen38mk.so
	python3 scripts/exactness.py $(if $(MK_MODEL),-m $(MK_MODEL)) $(EXACTNESS_ARGS)

# The benches' GPU self-checks without a model (the GPU to itself). wvgemm_bench check: its own tensor-core verify
# kernel (a counterpart of wv.cuh, not the engine's code) bit-identical at every T, and it and the engine's dp4a GEMVs
# within error bounds of fp64; wv_ref.cuh, its copy of the engine's dp4a code, is regenerated from mk_shared.cuh +
# qwen38_mk.cu every time and rewritten only when it changes. dattn_bench TINV=1: its own copies of the decode
# attention kernels (not regenerated: older than the engine's, up to T 5) bit-identical to each other and T-invariant.
bench/wv_ref.cuh: FORCE
	@python3 bench/wv_extract.py qwen38_mk.cu $@.tmp > /dev/null
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv $@.tmp $@

bench/wvgemm_bench: bench/wvgemm_bench.cu bench/wv_ref.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo $< -o $@

bench/dattn_bench: bench/dattn_bench.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -Xptxas -dlcm=cg $< -o $@

# mmq_bench check: the prefill GEMM kernels (prefill_gemm.cuh) against a scalar reference on the engine's layouts
bench/mmq_bench: bench/mmq_bench.cu prefill_gemm.cuh mk_shared.cuh mk_launch.h wv_host.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-dlcm=cg -Ibench/stub $(MMQFLAGS) $< -o $@

# attn_bench: pk_attn9 (prefill_attn.cuh, the engine's own kernels) bit for bit against pk_attn8 on random data in the engine's KV layouts, past the q8 window
bench/attn_bench: bench/attn_bench.cu prefill_attn.cuh mk_shared.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-dlcm=cg -Ibench/stub $< -o $@

check: bench/wvgemm_bench bench/dattn_bench bench/mmq_bench bench/attn_bench
	bench/wvgemm_bench check
	bench/mmq_bench check
	TINV=1 bench/dattn_bench 4096 4 3 0 1 2
	TINV=1 bench/dattn_bench 101600 5 3 0 1 2
	for f in F D q8 Vl any45; do bench/attn_bench $$f 16384 0 1024:0 17:2040 777:2047 1024:2600 1024:6000 || exit 1; done
	EXA=1 bench/attn_bench F 16384 0 1024:6000
	EXA=2 bench/attn_bench D 16384 0 1024:6000

# every GPU check of a build in one run (scripts/gate.sh; the GPU to itself, ~75 min on an RTX 3090):
#   make gate MK_MODEL=<gguf> GATE_ARGS="--record DIR"   (a reference)   or   GATE_ARGS="--ref DIR"
gate: qwen38_mk libqwen38mk.so
	scripts/gate.sh $(if $(MK_MODEL),-m $(MK_MODEL)) $(GATE_ARGS)

clean:
	rm -f qwen38_mk libqwen38mk.so qwen38_mk.o $(HOST_OBJ) qwen38_mk.o.d $(HOST_OBJ:=.d) ptxas.log tools/vocab_freq tools/llama_ref libqwen38vision.so tools/mkv_cli tools/vision_ref tools/vit_check vision/vit_cpu.o tests/gguf_fuzz \
		bench/wvgemm_bench bench/dattn_bench bench/mmq_bench bench/attn_bench

.PHONY: all vision clean exactness check gate FORCE

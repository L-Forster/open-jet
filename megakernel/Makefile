LLAMA ?= ../llama.cpp
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_75 RTX 8000, sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50)
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
override ARCH := sm_$(patsubst sm_%,%,$(ARCH))
# where the libraries look for libllama / libggml-base at run time; the prebuilt release archives
# ship them next to the engine with RPATH='$$ORIGIN'
RPATH ?= $(abspath $(LLAMA)/build/bin)
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC
ifeq ($(ARCH),sm_75)
NVFLAGS += -DQWEN_MK_SM75=1
endif

all: qwen38_mk libqwen38mk.so tools/vocab_freq tools/llama_ref

# ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu qwen38_mk.h gguf.h kv_paged.cuh kv_lowbit.cuh kv_spec.h prefill.cuh prefill_attn.cuh prefill_gemm.cuh cuda_compat.cuh bmm.cuh
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

qwen38_mk: qwen38_mk.o main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libqwen38mk.so: qwen38_mk.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker '$(RPATH)' -o $@

# The engine for --agents / --parallel with MK_KV: the megakernel also decodes a lone conversation from the slots'
# low-bit pages (KV_SOLO, kv_paged.cuh). libqwen38mk.so leaves that code out of the megakernel, so it doesn't slow the
# single conversation; mk_server.py loads this one when it is next to it. sm_80+ (Turing refuses MK_KV).
qwen38_mk_agents.o: qwen38_mk.cu qwen38_mk.h gguf.h kv_paged.cuh kv_lowbit.cuh kv_spec.h prefill.cuh prefill_attn.cuh prefill_gemm.cuh cuda_compat.cuh bmm.cuh
	$(NVCC) $(NVFLAGS) -DOPENJET_MK_AGENTS=1 -I$(LLAMA)/ggml/include -c qwen38_mk.cu -o $@ 2> ptxas-agents.log || (cat ptxas-agents.log; false)

libqwen38mk_agents.so: qwen38_mk_agents.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk_agents.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker '$(RPATH)' -o $@

# Extended quant support is opt-in; neither `all` nor the native library builds it.
qwen38_mixed.o: qwen38_mixed.cu qwen38_mk.h gguf.h kv_paged.cuh kv_lowbit.cuh kv_spec.h prefill.cuh prefill_attn.cuh prefill_gemm_mixed.cuh
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include -c qwen38_mixed.cu -o $@ 2> ptxas-mixed.log || (cat ptxas-mixed.log; false)

qwen38_mixed: qwen38_mixed.o main.cu logit_stats.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mixed.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

libqwen38mk_mixed.so: qwen38_mixed.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mixed.o \
		-L$(LLAMA)/build/bin -lllama -lggml-base -Xlinker -rpath -Xlinker '$(RPATH)' -o $@

bench/turing_check: bench/turing_check.cu qwen38_mk.cu qwen38_mk.h gguf.h kv_paged.cuh kv_lowbit.cuh kv_spec.h prefill.cuh prefill_attn.cuh prefill_gemm.cuh cuda_compat.cuh
	$(NVCC) $(NVFLAGS) -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lggml-base \
		-Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

bench/mmq_bench: bench/mmq_bench.cu prefill_gemm.cuh cuda_compat.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

# llama.cpp's next-token distributions for the accuracy check (see "Correctness" in README.md)
tools/llama_ref: tools/llama_ref.cpp logit_stats.h
	g++ -O2 -std=c++17 -pthread -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

clean:
	rm -f qwen38_mk qwen38_mixed libqwen38mk.so libqwen38mk_mixed.so libqwen38mk_agents.so qwen38_mk.o qwen38_mixed.o \
		qwen38_mk_agents.o ptxas.log ptxas-mixed.log ptxas-agents.log \
		tools/vocab_freq tools/llama_ref bench/turing_check bench/mmq_bench

LLAMA ?= ../llama.cpp
NVCC ?= /usr/local/cuda/bin/nvcc
# the first GPU's architecture (sm_86 RTX 30, sm_89 RTX 40, sm_120 RTX 50); sm_86 if nvidia-smi is missing
ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. '),86)
ARCH := sm_$(patsubst sm_%,%,$(ARCH))
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC

all: qwen38_mk libqwen38mk.so tools/vocab_freq

# ptxas register / spill report for every phase function goes to ptxas.log
qwen38_mk.o: qwen38_mk.cu qwen38_mk.h gguf.h prefill.cuh prefill_gemm.cuh
	$(NVCC) $(NVFLAGS) -c qwen38_mk.cu -o $@ 2> ptxas.log || (cat ptxas.log; false)

qwen38_mk: qwen38_mk.o main.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -I$(LLAMA)/include -I$(LLAMA)/ggml/include main.cu qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

# engine + tokenizer as a shared library with a C API (capi.cpp), used by server/mk_server.py
libqwen38mk.so: qwen38_mk.o capi.cpp qwen38_mk.h
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include capi.cpp qwen38_mk.o \
		-L$(LLAMA)/build/bin -lllama -Xlinker -rpath -Xlinker $(abspath $(LLAMA)/build/bin) -o $@

tools/vocab_freq: tools/vocab_freq.cpp
	g++ -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -L$(LLAMA)/build/bin -lllama \
		-Wl,-rpath,$(abspath $(LLAMA)/build/bin) -o $@

clean:
	rm -f qwen38_mk libqwen38mk.so qwen38_mk.o ptxas.log tools/vocab_freq

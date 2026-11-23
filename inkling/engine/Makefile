# OpenJet's Inkling engine. ARCH defaults to the RTX 30 series (sm_86).
ARCH ?= sm_86
NVCC ?= nvcc
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fPIC

all: libinkling_kernels.so libinkling.so

# expert kernels alone (tests/test_expert_kernels.py, engine_v0.py)
libinkling_kernels.so: expert_kernels.cu gguf_kernels.cu iq_tables.h
	$(NVCC) $(NVFLAGS) -shared -o $@ expert_kernels.cu gguf_kernels.cu

# the engine: ik_new / ik_decode C API
# the engine plus the mk_* API for megakernel/server/mk_server.py (capi.cpp). LLAMA: a llama.cpp checkout built with
# inkling2_loop support (tokenizer and chat template); its shared libraries are copied into lib/ and found via rpath.
LLAMA ?= /home/louis/bench-engines/llama.cpp-inkling
RPATH ?= $$ORIGIN/lib
libinkling.so: ik_engine.cu ik_kernels.cu ik_prefill.cu ik_prefill_tc.cu ik_dspark.cu expert_kernels.cu gguf_kernels.cu cpu_experts.cpp capi.cpp ik_kernels.cuh ik_prefill.cuh ik_dspark.cuh gguf.h iq_tables.h
	mkdir -p lib && cp -P $(LLAMA)/build/bin/libllama.so* $(LLAMA)/build/bin/libggml*.so* lib/
	$(CXX) $(CPUFLAGS) -c -o cpu_experts.o cpu_experts.cpp
	$(CXX) -O2 -std=c++17 -fPIC -I$(LLAMA)/include -I$(LLAMA)/ggml/include -c -o capi.o capi.cpp
	$(NVCC) $(NVFLAGS) -Xcompiler -mavx2,-mfma,-mf16c -shared -o $@ ik_engine.cu ik_kernels.cu ik_prefill.cu ik_prefill_tc.cu ik_dspark.cu expert_kernels.cu gguf_kernels.cu cpu_experts.o capi.o \
		-Llib -lllama -ldl -Xlinker -rpath,'$(RPATH)'

clean:
	rm -f libinkling_kernels.so libinkling.so libinkling_test.so libinkling_cpu.so *.o

.PHONY: all clean

# per-kernel test wrappers (tests/test_ik_kernels.py)
libinkling_test.so: ik_test_api.cu ik_kernels.cu ik_kernels.cuh
	$(NVCC) $(NVFLAGS) -shared -o $@ ik_test_api.cu ik_kernels.cu

# CPU expert kernels (AVX2 + FMA)
CXX ?= g++
CPUFLAGS = -O3 -std=c++17 -fPIC -mavx2 -mfma -mf16c
libinkling_cpu.so: cpu_experts.cpp iq_tables.h
	$(CXX) $(CPUFLAGS) -shared -o $@ cpu_experts.cpp

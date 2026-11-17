# Third-party components in the release packages

This repository's own code is licensed under the GNU AGPL v3 ([LICENSE](LICENSE)). The prebuilt packages on the
releases page also contain the components below, each under its own license. The packages' `licenses/` folder holds
the full license texts.

| Component | Where it is in a package | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b10246: libllama (tokenizer), libmtmd (image library), ggml | `llama` / `ggml*` / `mtmd` libraries | MIT (`licenses/llama.cpp-LICENSE`, plus the files of its `licenses/` folder) |
| stb_image (Sean Barrett), compiled into libmtmd and the vision library | `mtmd` and `qwen38vision` libraries | MIT or public domain (`licenses/stb_image-LICENSE`) |
| miniaudio (David Reid), compiled into libmtmd | `mtmd` library | public domain or MIT No Attribution (`licenses/miniaudio-LICENSE`) |
| NVIDIA CUDA runtime (`cudart_static`), linked into the engine | `qwen38mk.dll` / `libqwen38mk.so`, `qwen38_mk(.exe)` | NVIDIA CUDA Toolkit End User License Agreement, https://docs.nvidia.com/cuda/eula/ |
| Microsoft Visual C++ runtime (`msvcp140`, `vcruntime140`, `vcruntime140_1`, `vcomp140`), Windows package only | next to the executables | Microsoft Visual Studio license terms (redistributable code) |

The CUDA runtime is NVIDIA's software and the Visual C++ runtime is Microsoft's; neither is covered by this project's
license. The model files are not part of the packages: they come from Unsloth's Hugging Face repository under the
model's own license.

Source code: each release is built from the tagged commit of this repository (the release page links its source
archive) and llama.cpp tag b10246 (https://github.com/ggml-org/llama.cpp/tree/b10246). `BUILD_INFO.txt` in each package
names the commit, the build flags and the toolchain.

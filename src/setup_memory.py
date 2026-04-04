from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .runtime_limits import read_memory_snapshot

_MIN_CTX = 1024

# GGUF metadata value type codes.
_GGUF_TYPES: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 8: "str", 9: "arr",
    10: "<Q", 11: "<q", 12: "<d",
}


def _read_gguf_metadata(path: Path) -> dict[str, object]:
    """Read metadata key-value pairs from a GGUF file header."""
    meta: dict[str, object] = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return meta
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            return meta
        _tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        def read_str() -> str:
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8")

        def read_value(type_code: int) -> object:
            fmt = _GGUF_TYPES.get(type_code)
            if fmt == "str":
                return read_str()
            if fmt == "arr":
                elem_type = struct.unpack("<I", f.read(4))[0]
                count = struct.unpack("<Q", f.read(8))[0]
                return [read_value(elem_type) for _ in range(count)]
            if fmt is None:
                return None
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(kv_count):
            key = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_value(vtype)
    return meta


def _kv_bytes_per_token_from_gguf(path: Path) -> float | None:
    """Compute KV cache bytes per token from GGUF model metadata.

    Formula depends on the model family:
    - standard/GQA models: n_layer * n_head_kv * (key_length + value_length) * bytes_per_element
    - MLA models: llama.cpp stores only the compressed K cache, so use
      n_layer * n_head_kv * key_length_mla * bytes_per_element

    Runtime uses q8_0 for both K and V caches (~1 byte per element).
    """
    meta = _read_gguf_metadata(path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return None
    n_embd = meta.get(f"{arch}.embedding_length")
    n_head = meta.get(f"{arch}.attention.head_count")
    n_head_kv = meta.get(f"{arch}.attention.head_count_kv", n_head)
    n_layer = meta.get(f"{arch}.block_count")
    if not all(isinstance(v, int) and v > 0 for v in (n_head_kv, n_layer)):
        return None

    # Hybrid-attention Qwen3.5 models only allocate context-growing KV cache on
    # their periodic full-attention layers. GGUF exposes that cadence directly.
    full_attention_interval = meta.get(f"{arch}.full_attention_interval")
    if isinstance(full_attention_interval, int) and full_attention_interval > 1:
        n_layer = (n_layer + full_attention_interval - 1) // full_attention_interval

    # q8_0: 32 values stored in 34 bytes (32 quantized + 2-byte scale).
    bytes_per_element = 34 / 32

    key_length_mla = meta.get(f"{arch}.attention.key_length_mla")
    if isinstance(key_length_mla, int) and key_length_mla > 0:
        return n_layer * n_head_kv * key_length_mla * bytes_per_element

    key_length = meta.get(f"{arch}.attention.key_length")
    value_length = meta.get(f"{arch}.attention.value_length")
    if (
        isinstance(key_length, int) and key_length > 0
        and isinstance(value_length, int) and value_length > 0
    ):
        return n_layer * n_head_kv * (key_length + value_length) * bytes_per_element

    if not all(isinstance(v, int) and v > 0 for v in (n_embd, n_head)):
        return None
    head_dim = n_embd // n_head
    return 2 * n_layer * n_head_kv * head_dim * bytes_per_element


def _max_context_tokens_from_gguf(path: Path) -> int | None:
    """Read the model-declared maximum context length from GGUF metadata."""
    meta = _read_gguf_metadata(path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return None
    context_length = meta.get(f"{arch}.context_length")
    if isinstance(context_length, int) and context_length > 0:
        return context_length
    return None


def _model_file_size_mb(refs: Iterable[object]) -> float | None:
    for ref in refs:
        path = Path(str(ref).strip()).expanduser()
        if path.is_file():
            return round(path.stat().st_size / (1024 * 1024), 2)
    return None


def _model_gguf_path(refs: Iterable[object]) -> Path | None:
    for ref in refs:
        path = Path(str(ref).strip()).expanduser()
        if path.is_file() and path.suffix == ".gguf":
            return path
    return None


def _run_tool(name: str, args: list[str]) -> subprocess.CompletedProcess | None:
    tool = shutil.which(name)
    if not tool:
        return None
    try:
        proc = subprocess.run([tool, *args], capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None
    return proc if proc.returncode == 0 else None


def _run_tool_via_login_shell(name: str, args: list[str]) -> subprocess.CompletedProcess | None:
    shell = os.environ.get("SHELL") or "/bin/bash"
    command = " ".join([shutil.which(name) or name, *args])
    try:
        return subprocess.run(
            [shell, "-lc", command],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None


def _nvidia_free_vram_mb() -> float | None:
    proc = _run_tool("nvidia-smi", [
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if not proc:
        return None
    free = []
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) == 4:
            try:
                free.append(float(parts[3]))
            except ValueError:
                pass
    return max(free) if free else None


def _amd_free_vram_mb() -> float | None:
    free = []
    for total_path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        used_path = total_path.with_name("mem_info_vram_used")
        if not used_path.is_file():
            continue
        try:
            total = int(total_path.read_text(encoding="utf-8").strip())
            used = int(used_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if total > 0 and used >= 0:
            free.append(max(0.0, (total - used) / (1024 * 1024)))
    if free:
        return max(free)

    proc = _run_tool("rocm-smi", ["--showmeminfo", "vram", "--json"])
    if not proc:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for v in payload.values():
        if not isinstance(v, dict):
            continue
        total_raw = v.get("VRAM Total Memory (B)") or v.get("vram_total") or v.get("total")
        used_raw = v.get("VRAM Total Used Memory (B)") or v.get("vram_used") or v.get("used")
        try:
            total = float(str(total_raw).strip().replace(",", ""))
            used = float(str(used_raw).strip().replace(",", ""))
        except ValueError:
            continue
        free.append(max(0.0, (total - used) / (1024 * 1024)))
    return max(free) if free else None


def _parse_vulkaninfo_free_vram_mb(text: str) -> float | None:
    free_values: list[float] = []
    current_size: int | None = None
    current_budget: int | None = None
    current_usage: int | None = None
    current_device_local = False

    def flush() -> None:
        nonlocal current_size, current_budget, current_usage, current_device_local
        if current_device_local:
            if current_budget is not None and current_usage is not None:
                free_values.append(max(0.0, (current_budget - current_usage) / (1024 * 1024)))
            elif current_size is not None:
                free_values.append(max(0.0, current_size / (1024 * 1024)))
        current_size = None
        current_budget = None
        current_usage = None
        current_device_local = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"memoryHeaps\[\d+\]:", line):
            flush()
            continue
        if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in line:
            current_device_local = True
            continue
        if line.startswith("size"):
            match = re.search(r"=\s*(\d+)", line)
            if match:
                current_size = int(match.group(1))
            continue
        if line.startswith("budget"):
            match = re.search(r"=\s*(\d+)", line)
            if match:
                current_budget = int(match.group(1))
            continue
        if line.startswith("usage"):
            match = re.search(r"=\s*(\d+)", line)
            if match:
                current_usage = int(match.group(1))
            continue

    flush()
    return max(free_values) if free_values else None


def _vulkan_free_vram_mb() -> float | None:
    if not shutil.which("vulkaninfo"):
        return None
    for args in (["--summary"], []):
        proc = _run_tool_via_login_shell("vulkaninfo", args)
        if proc is None:
            continue
        output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        free = _parse_vulkaninfo_free_vram_mb(output)
        if free is not None:
            return free
    return None


def _detect_free_memory_mb(device: str) -> float | None:
    dev = (device or "").strip().lower()

    if dev in {"cpu", "metal"} and sys.platform == "darwin":
        mem = read_memory_snapshot()
        return float(mem.available_mb) if mem else None

    if dev == "cuda":
        free = _nvidia_free_vram_mb()
        if free is not None:
            return free
        if Path("/dev/nvhost-gpu").exists():
            mem = read_memory_snapshot()
            return float(mem.available_mb) if mem else None
        return None

    if dev == "vulkan":
        free = _amd_free_vram_mb()
        if free is not None:
            return free
        return _vulkan_free_vram_mb()

    if dev == "rocm":
        return _amd_free_vram_mb()

    return None


def _max_tokens_for_memory(
    available_mb: float,
    kv_bytes_per_token: float,
    *,
    max_context_tokens: int | None = None,
    reserve_ratio: float = 0.10,
) -> int:
    if available_mb <= 0 or kv_bytes_per_token <= 0:
        return _MIN_CTX
    usable_mb = max(0.0, float(available_mb) * (1.0 - max(0.0, reserve_ratio)))
    if usable_mb <= 0:
        return _MIN_CTX
    tokens = max(_MIN_CTX, int(usable_mb * 1024 * 1024 / kv_bytes_per_token))
    if max_context_tokens is not None and max_context_tokens > 0:
        tokens = min(tokens, int(max_context_tokens))
    return max(_MIN_CTX, tokens)


def _vulkan_max_alloc_ctx(kv_bpt: float) -> int | None:
    """Cap context so the KV buffer stays within Vulkan's maxMemoryAllocationSize.

    Dozen (Vulkan-over-D3D12 on WSL2) enforces a 2 GB per-allocation limit.
    Allocations that exceed this silently produce broken buffers.
    """
    max_alloc = 2 * 1024 * 1024 * 1024  # 2 GB — typical D3D12 limit
    if kv_bpt <= 0:
        return None
    return max(_MIN_CTX, int(max_alloc * 0.95 / kv_bpt))


def recommend_context_window_for_model(
    *,
    device: str,
    fallback_tokens: int,
    model_size_mb: float | None,
    kv_bytes_per_token: float | None,
    model_max_context: int | None = None,
    total_vram_mb: float | None = None,
    free_memory_mb: float | None = None,
) -> int:
    fallback = max(_MIN_CTX, int(fallback_tokens))
    if model_size_mb is None or model_size_mb <= 0:
        return fallback
    if kv_bytes_per_token is None or kv_bytes_per_token <= 0:
        return fallback

    dev = (device or "").strip().lower()
    effective_model_max_context = model_max_context
    if dev == "vulkan":
        alloc_cap = _vulkan_max_alloc_ctx(kv_bytes_per_token)
        if alloc_cap is not None:
            if effective_model_max_context is not None:
                effective_model_max_context = min(effective_model_max_context, alloc_cap)
            else:
                effective_model_max_context = alloc_cap

    if free_memory_mb is not None:
        return _max_tokens_for_memory(
            free_memory_mb - model_size_mb,
            kv_bytes_per_token,
            max_context_tokens=effective_model_max_context,
        )

    if total_vram_mb is not None and total_vram_mb > 0 and dev in {"cuda", "vulkan", "rocm", "metal"}:
        return _max_tokens_for_memory(
            total_vram_mb - model_size_mb,
            kv_bytes_per_token,
            max_context_tokens=effective_model_max_context,
        )

    return fallback


def recommend_setup_context_window(
    *,
    runtime: str,
    device: str,
    fallback_tokens: int,
    model_refs: Iterable[object],
    total_vram_mb: float | None = None,
) -> int:
    fallback = max(_MIN_CTX, int(fallback_tokens))
    refs = list(model_refs)

    model_mb = _model_file_size_mb(refs)
    if model_mb is None:
        return fallback

    gguf_path = _model_gguf_path(refs)
    kv_bpt = _kv_bytes_per_token_from_gguf(gguf_path) if gguf_path else None
    if kv_bpt is None:
        return fallback
    model_max_context = _max_context_tokens_from_gguf(gguf_path) if gguf_path else None

    return recommend_context_window_for_model(
        device=device,
        fallback_tokens=fallback,
        model_size_mb=model_mb,
        kv_bytes_per_token=kv_bpt,
        model_max_context=model_max_context,
        total_vram_mb=total_vram_mb,
        free_memory_mb=_detect_free_memory_mb(device),
    )

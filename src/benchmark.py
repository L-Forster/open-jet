"""Run llama-bench using the active model profile parameters."""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import load_config
from .hardware import detect_hardware_info, read_device_model, recommended_device
from .setup_memory import _read_gguf_metadata


def _find_llama_binary(cfg: dict, name: str) -> str:
    server_path = cfg.get("llama_server_path")
    if server_path:
        candidate = Path(server_path).parent / name
        if candidate.is_file():
            return str(candidate)
    path = shutil.which(name)
    if path:
        return path
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"{name} not found. Build it with:\n"
        f"  cd ~/llama.cpp && cmake --build build --target {name}\n"
        f"Or ensure {name} is on your PATH."
    )


def _find_llama_bench(cfg: dict) -> str:
    try:
        return _find_llama_binary(cfg, "llama-bench")
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        "llama-bench not found. Build it with:\n"
        "  cd ~/llama.cpp && cmake --build build --target llama-bench\n"
        "Or ensure llama-bench is on your PATH."
    )


def _find_llama_completion(cfg: dict) -> str:
    return _find_llama_binary(cfg, "llama-completion")


def _get_gpu_name() -> str | None:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("vulkaninfo"):
        try:
            out = subprocess.check_output(
                ["vulkaninfo", "--summary"], text=True, timeout=8, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "deviceName" in line:
                    _, _, name = line.partition("=")
                    name = name.strip()
                    if name:
                        return name
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _resolve_device(cfg: dict) -> str:
    device = str(cfg.get("device") or "auto").strip().lower()
    if device in ("cuda", "cpu", "vulkan", "rocm", "metal"):
        return device
    return recommended_device()


def _bench_env(bench_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = os.path.dirname(bench_bin)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env


def _base_bench_cmd(bench_bin: str, model_path: str) -> list[str]:
    return [bench_bin, "-m", model_path, "-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0"]


def _requested_cpu_moe_layers(cfg: dict, gpu_layers: int) -> int:
    if bool(cfg.get("llama_cpu_moe", False)):
        return max(1, gpu_layers)
    try:
        n_cpu_moe = int(cfg.get("llama_n_cpu_moe", 0) or 0)
    except (TypeError, ValueError):
        n_cpu_moe = 0
    return max(0, n_cpu_moe)


def _moe_block_count(model_path: str) -> int | None:
    try:
        meta = _read_gguf_metadata(Path(model_path))
    except (OSError, ValueError, struct.error):
        return None
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return None
    expert_count = meta.get(f"{arch}.expert_count")
    block_count = meta.get(f"{arch}.block_count")
    if (
        isinstance(expert_count, int) and expert_count > 0
        and isinstance(block_count, int) and block_count > 0
    ):
        return block_count
    return None


def _moe_layer_count(cfg: dict, gpu_layers: int, model_path: str) -> int:
    n_layers = _requested_cpu_moe_layers(cfg, gpu_layers)
    block_count = _moe_block_count(model_path)
    if block_count:
        n_layers = min(n_layers, block_count)
    return n_layers


def _is_moe_benchmark(cfg: dict, model_path: str) -> bool:
    if bool(cfg.get("llama_cpu_moe", False)):
        return True
    if _requested_cpu_moe_layers(cfg, 1) > 0:
        return True
    if _moe_block_count(model_path):
        return True
    return bool(cfg.get("unified_memory_only") and cfg.get("active_model_size_mb"))


def _moe_cli_args(cfg: dict, gpu_layers: int, model_path: str) -> list[str]:
    if bool(cfg.get("llama_cpu_moe", False)):
        return ["--cpu-moe"]
    n_layers = _moe_layer_count(cfg, gpu_layers, model_path)
    if n_layers > 0:
        return ["-ncmoe", str(n_layers)]
    return []


def _moe_gpu_residency_args(device: str) -> list[str]:
    if device == "vulkan":
        return ["--no-host"]
    return []


def _moe_label(cfg: dict, gpu_layers: int, model_path: str) -> str:
    n_layers = _moe_layer_count(cfg, gpu_layers, model_path)
    if n_layers <= 0:
        if _is_moe_benchmark(cfg, model_path):
            return "gpu_experts=auto_fit, no_repack=on"
        return "off"
    mode = "cpu_moe" if bool(cfg.get("llama_cpu_moe", False)) else "n_cpu_moe"
    return f"{mode}={n_layers}, gpu_experts=auto_fit, no_repack=on"


def _load_bench_context(cfg: dict) -> tuple[str, str, str, int, dict[str, str], list[str]]:
    """Return (bench_bin, model_path, device, gpu_layers, env, base_cmd)."""
    model_path = str(cfg.get("llama_model") or "")
    if not model_path or not Path(model_path).is_file():
        raise SystemExit(f"Model not found: {model_path or '(none)'}. Run `openjet setup` first.")
    device = _resolve_device(cfg)
    gpu_layers = int(cfg.get("gpu_layers", 99)) if device != "cpu" else 0
    if _is_moe_benchmark(cfg, model_path):
        completion_bin = _find_llama_completion(cfg)
        return completion_bin, model_path, device, gpu_layers, _bench_env(completion_bin), []

    bench_bin = _find_llama_bench(cfg)
    return (
        bench_bin,
        model_path,
        device,
        gpu_layers,
        _bench_env(bench_bin),
        _base_bench_cmd(bench_bin, model_path),
    )


def _print_header(title: str, model_path: str, device: str, gpu_layers: int) -> None:
    gpu_name = _get_gpu_name()
    board = read_device_model()
    hw = detect_hardware_info()
    print(f"{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Model:   {Path(model_path).stem}")
    print(f"Device:  {device} (gpu_layers={gpu_layers})")
    if gpu_name:
        print(f"GPU:     {gpu_name}")
    if board:
        print(f"Board:   {board}")
    print(f"RAM:     {hw.total_ram_gb:.1f} GB")
    if hw.vram_mb > 0:
        print(f"VRAM:    {hw.vram_mb:.0f} MB")


def _run_bench(cmd: list[str], env: dict[str, str]) -> None:
    print(f"{'─' * 60}\n$ {' '.join(cmd)}\n{'─' * 60}")
    sys.stdout.flush()
    subprocess.run(cmd, env=env)


def _run_completion_process(cmd: list[str], env: dict[str, str]) -> None:
    print(f"{'─' * 60}\n$ {' '.join(cmd)}\n{'─' * 60}")
    sys.stdout.flush()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    suppress_generation = False
    cpu_mapped_model_mib: float | None = None
    vulkan_model_mib: float | None = None
    vulkan_total_mib: float | None = None
    saw_perf = False
    interesting_prefixes = (
        "WARNING:",
        "load_backend:",
        "load_tensors:",
        "llama_context:",
        "llama_kv_cache:",
        "llama_memory_recurrent:",
        "sched_reserve:",
        "system_info:",
        "generate:",
        "common_perf_print:",
        "llama_memory_breakdown_print:",
    )
    for line in proc.stdout:
        if "CPU_Mapped model buffer size" in line:
            cpu_mapped_model_mib = _parse_mib_value(line)
        elif "Vulkan0 model buffer size" in line:
            vulkan_model_mib = _parse_mib_value(line)
        elif "llama_memory_breakdown_print: |   - Vulkan0" in line:
            vulkan_total_mib = _parse_vulkan_total_mib(line)
        elif line.startswith("common_perf_print:"):
            saw_perf = True
        if line.startswith("generate: "):
            suppress_generation = True
            print(line, end="")
            continue
        if suppress_generation:
            if line.startswith(("common_perf_print:", "llama_memory_breakdown_print:", "llama_perf")):
                suppress_generation = False
            else:
                continue
        lower = line.lower()
        if (
            line.startswith(interesting_prefixes)
            or "error" in lower
            or "failed" in lower
            or "abort" in lower
            or "assert" in lower
            or "fatal" in lower
            or "signal" in lower
        ):
            print(line, end="")
    return_code = proc.wait()
    if return_code != 0:
        print(f"OpenJet diagnostic: benchmark process exited with code {return_code}.")
    elif not saw_perf:
        print("OpenJet diagnostic: benchmark process exited before printing performance timings.")
    if return_code == 0 and saw_perf:
        diagnostic = _host_resident_moe_diagnostic(
            cpu_mapped_model_mib=cpu_mapped_model_mib,
            vulkan_model_mib=vulkan_model_mib,
            vulkan_total_mib=vulkan_total_mib,
        )
        if diagnostic:
            print(diagnostic)


def _parse_mib_value(line: str) -> float | None:
    match = re.search(r"=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_vulkan_total_mib(line: str) -> float | None:
    match = re.search(r"\|\s+([0-9]+)\s*=", line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _host_resident_moe_diagnostic(
    *,
    cpu_mapped_model_mib: float | None,
    vulkan_model_mib: float | None,
    vulkan_total_mib: float | None,
) -> str:
    if cpu_mapped_model_mib is None or vulkan_model_mib is None:
        return ""
    if cpu_mapped_model_mib < max(2048.0, vulkan_model_mib * 2.0):
        return ""
    total = f"; llama.cpp reports Vulkan0 total {vulkan_total_mib:.0f} MiB" if vulkan_total_mib else ""
    return (
        "OpenJet diagnostic: MoE expert weights stayed host-resident "
        f"({cpu_mapped_model_mib:.0f} MiB CPU_Mapped vs {vulkan_model_mib:.0f} MiB Vulkan0{total}). "
        "This llama.cpp Vulkan path is not dynamically streaming only active experts into VRAM, "
        "so decode is host-memory bound. Use a partial `llama_n_cpu_moe` split plus `--no-host` to "
        "fit some expert layers into the exposed Vulkan heap; it still is not a dynamic active-expert "
        "paging path."
    )


def _synthetic_prompt(n_prompt: int) -> str:
    seed = (
        "OpenJet throughput probe. "
        "This synthetic context mixes short factual clauses, numbers, and punctuation "
        "so completion does not collapse into one repeated token. "
    )
    parts: list[str] = []
    i = 0
    while len(" ".join(parts).split()) < max(1, n_prompt):
        parts.append(f"{seed} Segment {i}: alpha={i % 17}, beta={i % 29}, gamma={i % 43}.")
        i += 1
    return " ".join(parts)


def _run_completion_benchmark(
    cfg: dict,
    *,
    model_path: str,
    device: str,
    gpu_layers: int,
    n_prompt: int,
    n_gen: int,
    extra_args: list[str] | None,
) -> None:
    completion_bin = _find_llama_completion(cfg)
    env = _bench_env(completion_bin)
    ctx = max(n_prompt + n_gen + 32, int(cfg.get("context_window_tokens", 4096) or 4096))
    prompt = _synthetic_prompt(n_prompt)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="openjet-bench-", suffix=".txt") as prompt_file:
        prompt_file.write(prompt)
        prompt_file.flush()
        cmd = [
            completion_bin,
            "-m", model_path,
            "--flash-attn", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--no-repack",
            *_moe_gpu_residency_args(device),
            *_moe_cli_args(cfg, gpu_layers, model_path),
            "-fit", "on",
            "-ngl", str(gpu_layers),
            "-c", str(ctx),
            "-b", str(min(2048, max(512, n_prompt))),
            "-ub", str(min(512, max(128, n_prompt))),
            "-f", prompt_file.name,
            "-n", str(n_gen),
            "--no-display-prompt",
            "--perf",
            "--no-conversation",
            "--simple-io",
            "--color", "off",
            "--log-colors", "off",
        ]
        if extra_args:
            cmd.extend(extra_args)
        _run_completion_process(cmd, env)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    output_format: str = "md",
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 5,
    extra_args: list[str] | None = None,
) -> None:
    cfg = load_config()
    bench_bin, model_path, device, gpu_layers, env, base = _load_bench_context(cfg)

    _print_header("OpenJet Benchmark", model_path, device, gpu_layers)
    print(f"Config:  fa=on, ctk=q8_0, ctv=q8_0, moe={_moe_label(cfg, gpu_layers, model_path)}")
    print(f"Test:    pp{n_prompt} / tg{n_gen} x{repetitions} reps")

    if _is_moe_benchmark(cfg, model_path):
        if repetitions != 1:
            print("Runner:  llama-completion --no-repack (single run; llama-bench cannot disable MoE repack)")
        else:
            print("Runner:  llama-completion --no-repack")
        _run_completion_benchmark(
            cfg,
            model_path=model_path,
            device=device,
            gpu_layers=gpu_layers,
            n_prompt=n_prompt,
            n_gen=n_gen,
            extra_args=extra_args,
        )
        return

    cmd = base + [
        "-ngl", str(gpu_layers),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-r", str(repetitions), "-o", output_format,
    ]
    if extra_args:
        cmd.extend(extra_args)

    _run_bench(cmd, env)


# ---------------------------------------------------------------------------
# Sweep: one variable at a time
# ---------------------------------------------------------------------------

def run_benchmark_sweep(*, repetitions: int = 3) -> None:
    cfg = load_config()
    bench_bin, model_path, device, gpu_layers, env, base = _load_bench_context(cfg)

    _print_header("OpenJet Benchmark Sweep", model_path, device, gpu_layers)
    print(f"Config:  fa=on, ctk=q8_0, ctv=q8_0, moe={_moe_label(cfg, gpu_layers, model_path)}")
    print(f"Reps:    {repetitions} per test point")
    if _is_moe_benchmark(cfg, model_path):
        raise SystemExit(
            "MoE sweep is not supported with this bundled llama-bench because it cannot disable "
            "weight repacking. Run `openjet benchmark` for the llama-completion --no-repack measurement."
        )

    reps = ["-r", str(repetitions), "-o", "md"]

    # GPU layer offloading
    if device != "cpu":
        step = max(1, gpu_layers // 8)
        ngl_vals = list(range(0, gpu_layers, step))
        if ngl_vals[-1] != gpu_layers:
            ngl_vals.append(gpu_layers)
        ngl_csv = ",".join(str(v) for v in ngl_vals)
        print(f"\n>>> GPU layer offloading  (-ngl {ngl_csv})")
        _run_bench(base + ["-ngl", ngl_csv, "-p", "512", "-n", "128"] + reps, env)

    # Batch size
    batch_csv = "64,128,256,512,1024,2048"
    print(f"\n>>> Batch size  (-b {batch_csv})")
    _run_bench(base + ["-ngl", str(gpu_layers), "-n", "0", "-p", "1024", "-b", batch_csv] + reps, env)

    # Thread scaling
    cpus = os.cpu_count() or 4
    t_vals: list[int] = []
    t = 1
    while t <= cpus:
        t_vals.append(t)
        t *= 2
    if t_vals[-1] != cpus:
        t_vals.append(cpus)
    thread_csv = ",".join(str(v) for v in t_vals)
    print(f"\n>>> Thread scaling  (-t {thread_csv})")
    _run_bench(base + ["-ngl", str(gpu_layers), "-p", "512", "-n", "128", "-t", thread_csv] + reps, env)

    print(f"\n{'=' * 60}\nSweep complete.\n{'=' * 60}")

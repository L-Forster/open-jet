"""Run llama-bench using the active model profile parameters."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .hardware import detect_hardware_info, read_device_model, recommended_device


def _find_llama_bench(cfg: dict) -> str:
    server_path = cfg.get("llama_server_path")
    if server_path:
        candidate = Path(server_path).parent / "llama-bench"
        if candidate.is_file():
            return str(candidate)
    path = shutil.which("llama-bench")
    if path:
        return path
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / "llama-bench"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        "llama-bench not found. Build it with:\n"
        "  cd ~/llama.cpp && cmake --build build --target llama-bench\n"
        "Or ensure llama-bench is on your PATH."
    )


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


def _base_cmd(bench_bin: str, model_path: str) -> list[str]:
    return [bench_bin, "-m", model_path, "-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0"]


def _load_bench_context(cfg: dict) -> tuple[str, str, str, int, dict[str, str], list[str]]:
    """Return (bench_bin, model_path, device, gpu_layers, env, base_cmd)."""
    bench_bin = _find_llama_bench(cfg)
    model_path = str(cfg.get("llama_model") or "")
    if not model_path or not Path(model_path).is_file():
        raise SystemExit(f"Model not found: {model_path or '(none)'}. Run `openjet setup` first.")
    device = _resolve_device(cfg)
    gpu_layers = int(cfg.get("gpu_layers", 99)) if device != "cpu" else 0
    return bench_bin, model_path, device, gpu_layers, _bench_env(bench_bin), _base_cmd(bench_bin, model_path)


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
    print(f"Config:  fa=on, ctk=q8_0, ctv=q8_0")
    print(f"Test:    pp{n_prompt} / tg{n_gen} x{repetitions} reps")

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
    print(f"Reps:    {repetitions} per test point")

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

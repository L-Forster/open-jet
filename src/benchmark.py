"""Run llama-bench using the active model profile parameters."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import load_config
from .hardware import detect_hardware_info, read_device_model, recommended_device
from .setup_memory import _read_gguf_metadata


_TURBO_BENCHMARK_PROMPT = (
    "You are benchmarking local agent generation throughput. Write a compact Python implementation "
    "of a file-backed task queue with enqueue, claim, complete, retry, and list operations. Include "
    "the key data structures, locking strategy, and one small usage example. Keep the answer direct."
)
_TURBO_DRAFT_MODEL_NAMES = (
    "dflash-draft-3.6-q8_0.gguf",
    "Qwen3.6-27B-DFlash-Q8_0.gguf",
    "qwen3.6-27b-dflash-q8_0.gguf",
    "dflash-draft-3.6-q4_k_m.gguf",
    "model.safetensors",
)
_LUCEBOX_REPO_URL = "https://github.com/Luce-Org/lucebox-hub.git"
_LUCEBOX_TARGET_FILE = "Qwen3.6-27B-Q4_K_M.gguf"
_LUCEBOX_DRAFT_REPO = "z-lab/Qwen3.6-27B-DFlash"
_LUCEBOX_DRAFT_FILE = "model.safetensors"
_LUCEBOX_PY_DEPS = ("fastapi", "uvicorn", "transformers", "jinja2")


@dataclass(frozen=True)
class TurboBenchmarkSettings:
    target_model: str
    draft_model: str
    backend_path: str
    backend_kind: str
    backend_label: str
    context_size: int
    draft_context_size: int
    gpu_layers: int
    draft_gpu_layers: int
    batch_size: int
    ubatch_size: int
    thinking_enabled: bool
    baseline_tok_s: float | None


@dataclass(frozen=True)
class TurboBenchmarkTimings:
    prompt_eval_tok_s: float | None
    generation_tok_s: float | None
    prompt_tokens: int | None
    generation_tokens: int | None
    raw: dict


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


def _find_llama_server_for_turbo(cfg: dict) -> str:
    turbo_cfg = cfg.get("turbo") if isinstance(cfg.get("turbo"), dict) else {}
    candidates = [
        os.getenv("OPENJET_TURBO_LLAMA_SERVER", "").strip(),
        str(turbo_cfg.get("llama_server_path") or "").strip(),
        str(cfg.get("dflash_llama_server_path") or "").strip(),
        str(cfg.get("llama_server_path") or "").strip(),
        shutil.which("llama-server") or "",
        str(Path.home() / "buun-llama-cpp" / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")),
        str(Path.home() / "llama.cpp" / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")),
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "llama-server for DFlash was not found. Build spiritbuun/buun-llama-cpp and set "
        "`turbo.llama_server_path` in config.yaml or OPENJET_TURBO_LLAMA_SERVER."
    )


def _find_lucebox_backend_for_turbo(cfg: dict) -> str:
    turbo_cfg = _turbo_cfg(cfg)
    root_raw = _lucebox_root_raw(turbo_cfg)
    candidates: list[str] = [
        os.getenv("OPENJET_LUCEBOX_DFLASH_BIN", "").strip(),
        str(turbo_cfg.get("lucebox_bin") or "").strip(),
        str(turbo_cfg.get("lucebox_server_path") or "").strip(),
    ]
    roots = [Path(root_raw).expanduser()] if root_raw else []
    roots.extend([
        Path.cwd() / "lucebox-hub" / "dflash",
        Path.home() / "lucebox-hub" / "dflash",
    ])
    exe_name = "test_dflash.exe" if os.name == "nt" else "test_dflash"
    for root in roots:
        candidates.extend([
            str(root / "build" / exe_name),
            str(root / "scripts" / "server.py"),
        ])
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "Lucebox DFlash backend was not found. Clone Luce-Org/lucebox-hub, build dflash, "
        "then set `turbo.backend_kind: lucebox` and `turbo.lucebox_root`, `turbo.lucebox_bin`, "
        "or OPENJET_LUCEBOX_DFLASH_BIN."
    )


def _lucebox_root_raw(turbo_cfg: dict) -> str:
    return (
        os.getenv("OPENJET_LUCEBOX_ROOT", "").strip()
        or str(turbo_cfg.get("lucebox_root") or "").strip()
    )


def _default_lucebox_root() -> Path:
    return Path(".openjet") / "turbo" / "lucebox-hub" / "dflash"


def _normalize_turbo_backend_kind(value: object, backend_path: str | None = None) -> str:
    raw = str(value or "auto").strip().lower().replace("_", "-")
    if raw in {"llama", "llama-server", "spiritbuun", "buun", "buun-llama-cpp"}:
        return "llama-server"
    if raw in {"luce", "lucebox", "lucebox-hub", "test-dflash"}:
        return "lucebox"
    if raw not in {"", "auto"}:
        raise SystemExit("turbo backend kind must be one of: auto, llama-server, lucebox")
    lowered = (backend_path or "").lower().replace("\\", "/")
    if "lucebox" in lowered or lowered.endswith("/test_dflash") or lowered.endswith("/test_dflash.exe"):
        return "lucebox"
    return "llama-server"


def _backend_supports_dflash(backend_path: str, *, assume_supported: bool = False) -> bool:
    if assume_supported:
        return True
    try:
        proc = subprocess.run(
            [backend_path, "--help"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = f"{proc.stdout}\n{proc.stderr}".lower()
    return "dflash" in help_text and ("--spec-type" in help_text or "spec-type" in help_text)


def _dflash_backend_label(backend_path: str) -> str:
    lowered = backend_path.lower()
    if "buun-llama-cpp" in lowered:
        return "spiritbuun/buun-llama-cpp"
    return "DFlash-capable llama-server"


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


def _get_cuda_driver_summary() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    first = out.splitlines()[0].strip()
    parts = [part.strip() for part in first.split(",")]
    if len(parts) >= 3:
        return f"{parts[0]} ({parts[1]} MiB VRAM, driver {parts[2]})"
    return first


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


def _turbo_cfg(cfg: dict) -> dict:
    raw = cfg.get("turbo")
    return raw if isinstance(raw, dict) else {}


def _cfg_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _cfg_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _run_setup_step(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"Setup:   {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required setup command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        where = f" in {cwd}" if cwd else ""
        raise SystemExit(f"Setup command failed{where}: {' '.join(cmd)}") from exc


def _ensure_lucebox_checkout(root: Path) -> None:
    repo_root = root.parent
    if (root / "scripts" / "server.py").is_file():
        return
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_root.exists() and any(repo_root.iterdir()):
        raise SystemExit(
            f"Lucebox directory exists but does not look complete: {repo_root}. "
            "Remove it or set turbo.lucebox_root to a valid lucebox-hub/dflash checkout."
        )
    _run_setup_step(["git", "clone", "--recurse-submodules", _LUCEBOX_REPO_URL, str(repo_root)])


def _ensure_lucebox_python_deps(root: Path) -> None:
    marker = root / ".openjet-server-deps-installed"
    if marker.is_file():
        return
    _run_setup_step([sys.executable, "-m", "pip", "install", *_LUCEBOX_PY_DEPS], cwd=root)
    marker.write_text("ok\n", encoding="utf-8")


def _ensure_lucebox_binary(root: Path) -> str:
    exe_name = "test_dflash.exe" if os.name == "nt" else "test_dflash"
    binary = root / "build" / exe_name
    if binary.is_file():
        return str(binary)
    if not shutil.which("cmake"):
        raise SystemExit("cmake is required to build Lucebox DFlash. Install cmake, then rerun `openjet turbo benchmark`.")
    _run_setup_step(["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release"], cwd=root)
    _run_setup_step(["cmake", "--build", "build", "--target", "test_dflash", "-j", str(os.cpu_count() or 4)], cwd=root)
    if not binary.is_file():
        raise SystemExit(f"Lucebox build completed but binary was not found: {binary}")
    return str(binary)


def _hf_download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    target = local_dir / filename
    if target.is_file():
        return target
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Setup:   downloading {repo_id}/{filename} to {local_dir}")
    sys.stdout.flush()
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except Exception as exc:
        raise SystemExit(
            f"Failed to download {repo_id}/{filename}. If the repo is gated, set HF_TOKEN or download it manually."
        ) from exc


def _find_existing_lucebox_target_model(root: Path) -> Path | None:
    candidates: list[Path] = []
    for base in (
        Path.cwd(),
        Path.cwd() / "models",
        Path.home() / ".openjet" / "models",
        root / "models",
    ):
        if base.is_file() and base.suffix.lower() == ".gguf":
            candidates.append(base)
            continue
        if not base.is_dir():
            continue
        for pattern in ("*Qwen3.6*27B*.gguf", "*Qwen3.5*27B*.gguf", "*qwen3.6*27b*.gguf", "*qwen3.5*27b*.gguf"):
            candidates.extend(base.glob(pattern))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ensure_lucebox_target_model(cfg: dict, override: str | None, root: Path) -> str:
    if override or os.getenv("OPENJET_TURBO_TARGET_MODEL", "").strip() or _turbo_cfg(cfg).get("target_model") or cfg.get("dflash_target_model") or cfg.get("llama_model"):
        return _resolve_turbo_target_model(cfg, override)
    discovered = _find_existing_lucebox_target_model(root)
    if discovered:
        return str(discovered)
    raise SystemExit(
        "Target GGUF not configured and OpenJet will not download a duplicate target model. "
        "Run `openjet setup` first, or pass `--target-model /path/to/Qwen3.6-27B-Q4_K_M.gguf`."
    )


def _ensure_lucebox_draft_model(cfg: dict, target_model: str, override: str | None, root: Path) -> str:
    if override or os.getenv("OPENJET_TURBO_DRAFT_MODEL", "").strip() or _turbo_cfg(cfg).get("draft_model") or cfg.get("dflash_draft_model"):
        return _resolve_turbo_draft_model(cfg, target_model, override)
    return str(_hf_download_file(_LUCEBOX_DRAFT_REPO, _LUCEBOX_DRAFT_FILE, root / "models" / "draft"))


def _ensure_lucebox_backend(cfg: dict) -> tuple[Path, str]:
    turbo_cfg = _turbo_cfg(cfg)
    root = Path(_lucebox_root_raw(turbo_cfg)).expanduser() if _lucebox_root_raw(turbo_cfg) else _default_lucebox_root()
    _ensure_lucebox_checkout(root)
    _ensure_lucebox_python_deps(root)
    binary = _ensure_lucebox_binary(root)
    return root, binary


def _resolve_turbo_target_model(cfg: dict, override: str | None = None) -> str:
    turbo_cfg = _turbo_cfg(cfg)
    raw = (
        override
        or os.getenv("OPENJET_TURBO_TARGET_MODEL", "").strip()
        or str(turbo_cfg.get("target_model") or "").strip()
        or str(cfg.get("dflash_target_model") or "").strip()
        or str(cfg.get("llama_model") or "").strip()
    )
    path = Path(raw).expanduser() if raw else Path()
    if not raw or not path.is_file():
        raise SystemExit(
            f"Target model not found: {raw or '(none)'}. Configure `turbo.target_model` "
            "or use the active `llama_model` profile."
        )
    return str(path)


def _candidate_draft_model_paths(cfg: dict, target_model: str) -> list[Path]:
    roots: list[Path] = []
    target_parent = Path(target_model).expanduser().parent
    roots.append(target_parent)
    model_download_path = str(cfg.get("model_download_path") or "").strip()
    if model_download_path:
        roots.append(Path(model_download_path).expanduser().parent)
    roots.extend([
        Path.cwd() / "models",
        Path.home() / ".openjet" / "models",
        Path.home() / "open-jet" / "models",
    ])
    candidates: list[Path] = []
    for root in roots:
        for name in _TURBO_DRAFT_MODEL_NAMES:
            candidates.append(root / name)
    return candidates


def _resolve_turbo_draft_model(cfg: dict, target_model: str, override: str | None = None) -> str:
    turbo_cfg = _turbo_cfg(cfg)
    explicit = (
        override
        or os.getenv("OPENJET_TURBO_DRAFT_MODEL", "").strip()
        or str(turbo_cfg.get("draft_model") or "").strip()
        or str(cfg.get("dflash_draft_model") or "").strip()
    )
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        if path.is_dir():
            found = _find_safetensors(path)
            if found:
                return str(found)
        raise SystemExit(
            f"DFlash draft model not found: {explicit}. Configure `turbo.draft_model` "
            "with the spiritbuun DFlash GGUF path or Lucebox/z-lab model.safetensors path."
        )
    for candidate in _candidate_draft_model_paths(cfg, target_model):
        if candidate.is_file():
            return str(candidate)
    searched = ", ".join(str(path) for path in _candidate_draft_model_paths(cfg, target_model)[:4])
    raise SystemExit(
        "DFlash draft model not found. Set `turbo.draft_model` or OPENJET_TURBO_DRAFT_MODEL "
        f"to spiritbuun/Qwen3.6-27B-DFlash-GGUF, preferably dflash-draft-3.6-q8_0.gguf. Searched: {searched}"
    )


def _find_safetensors(root: Path) -> Path | None:
    if root.is_file() and root.name.endswith(".safetensors"):
        return root
    if not root.is_dir():
        return None
    for path in root.rglob("model.safetensors"):
        return path
    return None


def _load_turbo_settings(
    cfg: dict,
    *,
    thinking_enabled: bool,
    target_model: str | None = None,
    draft_model: str | None = None,
    backend_path: str | None = None,
    backend_kind: str | None = None,
    context_size: int | None = None,
    baseline_tok_s: float | None = None,
) -> TurboBenchmarkSettings:
    turbo_cfg = _turbo_cfg(cfg)
    requested_kind = _normalize_turbo_backend_kind(
        backend_kind or turbo_cfg.get("backend_kind") or turbo_cfg.get("backend_type"),
        backend_path,
    )
    if backend_path:
        backend = str(Path(backend_path).expanduser())
        resolved_kind = _normalize_turbo_backend_kind(backend_kind or requested_kind, backend)
        target = _resolve_turbo_target_model(cfg, target_model)
        draft = _resolve_turbo_draft_model(cfg, target, draft_model)
    elif requested_kind == "lucebox":
        lucebox_root, backend = _ensure_lucebox_backend(cfg)
        resolved_kind = "lucebox"
        target = _ensure_lucebox_target_model(cfg, target_model, lucebox_root)
        draft = _ensure_lucebox_draft_model(cfg, target, draft_model, lucebox_root)
    else:
        backend = _find_llama_server_for_turbo(cfg)
        resolved_kind = "llama-server"
        target = _resolve_turbo_target_model(cfg, target_model)
        draft = _resolve_turbo_draft_model(cfg, target, draft_model)
    if not Path(backend).is_file():
        raise SystemExit(f"DFlash backend not found: {backend}")
    assume_supported = bool(turbo_cfg.get("assume_dflash_backend") or os.getenv("OPENJET_ASSUME_DFLASH_BACKEND"))
    if resolved_kind == "llama-server" and not _backend_supports_dflash(backend, assume_supported=assume_supported):
        raise SystemExit(
            "DFlash support was not detected in the configured llama-server. Build "
            "spiritbuun/buun-llama-cpp with CUDA and set `turbo.llama_server_path` "
            "or OPENJET_TURBO_LLAMA_SERVER to that binary."
        )
    baseline = (
        baseline_tok_s
        if baseline_tok_s is not None
        else _cfg_float(os.getenv("OPENJET_TURBO_BASELINE_TOK_S", ""))
        or _cfg_float(turbo_cfg.get("baseline_tok_s"))
        or _cfg_float(cfg.get("benchmark_baseline_tok_s"))
    )
    ctx = context_size or _cfg_int(
        turbo_cfg.get("context_window_tokens", cfg.get("context_window_tokens", 6048)),
        6048,
        minimum=512,
    )
    return TurboBenchmarkSettings(
        target_model=target,
        draft_model=draft,
        backend_path=backend,
        backend_kind=resolved_kind,
        backend_label=str(
            turbo_cfg.get("backend")
            or ("Luce-Org/lucebox-hub dflash" if resolved_kind == "lucebox" else _dflash_backend_label(backend))
        ),
        context_size=ctx,
        draft_context_size=_cfg_int(turbo_cfg.get("draft_context_window_tokens", 256), 256, minimum=16),
        gpu_layers=_cfg_int(turbo_cfg.get("gpu_layers", cfg.get("gpu_layers", 99)), 99, minimum=0),
        draft_gpu_layers=_cfg_int(
            turbo_cfg.get("draft_gpu_layers", turbo_cfg.get("gpu_layers", cfg.get("gpu_layers", 99))),
            99,
            minimum=0,
        ),
        batch_size=_cfg_int(turbo_cfg.get("batch_size", 256), 256, minimum=1),
        ubatch_size=_cfg_int(turbo_cfg.get("ubatch_size", 64), 64, minimum=1),
        thinking_enabled=thinking_enabled,
        baseline_tok_s=baseline,
    )


def _turbo_server_cmd(settings: TurboBenchmarkSettings, *, host: str, port: int) -> list[str]:
    return [
        settings.backend_path,
        "-m",
        settings.target_model,
        "-md",
        settings.draft_model,
        "--spec-type",
        "dflash",
        "-ngl",
        str(settings.gpu_layers),
        "-ngld",
        str(settings.draft_gpu_layers),
        "-np",
        "1",
        "-c",
        str(settings.context_size),
        "-cd",
        str(settings.draft_context_size),
        "-fa",
        "on",
        "-b",
        str(settings.batch_size),
        "-ub",
        str(settings.ubatch_size),
        "--host",
        host,
        "--port",
        str(port),
        "--jinja",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": settings.thinking_enabled}, separators=(",", ":")),
    ]


def _lucebox_paths(settings: TurboBenchmarkSettings) -> tuple[str, str]:
    backend = Path(settings.backend_path).expanduser()
    if backend.name == "server.py":
        root = backend.parent.parent
        exe_name = "test_dflash.exe" if os.name == "nt" else "test_dflash"
        return str(backend), str(root / "build" / exe_name)
    return str(backend.parent.parent / "scripts" / "server.py"), str(backend)


def _lucebox_server_cmd(settings: TurboBenchmarkSettings, *, host: str, port: int) -> list[str]:
    server_script, test_dflash_bin = _lucebox_paths(settings)
    cmd = [
        sys.executable,
        server_script,
        "--host",
        host,
        "--port",
        str(port),
        "--target",
        settings.target_model,
        "--draft",
        settings.draft_model,
        "--bin",
        test_dflash_bin,
        "--budget",
        "22",
        "--max-ctx",
        str(settings.context_size),
    ]
    return cmd


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_log_tail(log_path: Path, *, max_chars: int = 6000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def _wait_for_turbo_server(proc: subprocess.Popen, base_url: str, log_path: Path, *, timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                detail = _server_log_tail(log_path) or "no server output"
                raise RuntimeError(f"llama-server exited before becoming ready: {detail}")
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    detail = _server_log_tail(log_path)
    raise TimeoutError(f"llama-server did not become ready within {timeout_seconds:.0f}s" + (f": {detail}" if detail else ""))


def _wait_for_lucebox_server(proc: subprocess.Popen, base_url: str, log_path: Path, *, timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                detail = _server_log_tail(log_path) or "no server output"
                raise RuntimeError(f"Lucebox server exited before becoming ready: {detail}")
            try:
                response = client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    detail = _server_log_tail(log_path)
    raise TimeoutError(f"Lucebox server did not become ready within {timeout_seconds:.0f}s" + (f": {detail}" if detail else ""))


def _extract_turbo_timings(payload: dict) -> TurboBenchmarkTimings:
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        timings = payload.get("usage", {}).get("timings") if isinstance(payload.get("usage"), dict) else {}
    if not isinstance(timings, dict):
        timings = {}

    def _num(*keys: str) -> float | None:
        for key in keys:
            value = timings.get(key)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            return parsed
        return None

    prompt_tps = _num("prompt_per_second", "prompt_eval_tok_s")
    generation_tps = _num("predicted_per_second", "generation_tok_s", "eval_tok_s")
    prompt_tokens = _num("prompt_n", "prompt_tokens", "prompt_eval_count")
    generation_tokens = _num("predicted_n", "completion_tokens", "eval_count")
    prompt_ms = _num("prompt_ms", "prompt_eval_ms")
    generation_ms = _num("predicted_ms", "generation_ms", "eval_ms")
    if prompt_tps is None and prompt_tokens and prompt_ms and prompt_ms > 0:
        prompt_tps = prompt_tokens / (prompt_ms / 1000.0)
    if generation_tps is None and generation_tokens and generation_ms and generation_ms > 0:
        generation_tps = generation_tokens / (generation_ms / 1000.0)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    if prompt_tokens is None:
        prompt_tokens = _cfg_float(usage.get("prompt_tokens"))
    if generation_tokens is None:
        generation_tokens = _cfg_float(usage.get("completion_tokens"))
    return TurboBenchmarkTimings(
        prompt_eval_tok_s=prompt_tps,
        generation_tok_s=generation_tps,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        generation_tokens=int(generation_tokens) if generation_tokens is not None else None,
        raw=payload,
    )


def _request_turbo_completion(base_url: str, *, n_gen: int, thinking_enabled: bool) -> TurboBenchmarkTimings:
    payload = {
        "model": "local",
        "messages": [
            {
                "role": "system",
                "content": "You are a concise local coding agent benchmarked for generation throughput.",
            },
            {"role": "user", "content": _TURBO_BENCHMARK_PROMPT},
        ],
        "temperature": 0,
        "max_tokens": n_gen,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": thinking_enabled},
        "reasoning_format": "auto" if thinking_enabled else "none",
    }
    started = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)) as client:
        response = client.post(f"{base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    timings = _extract_turbo_timings(data)
    if timings.generation_tok_s is None and timings.generation_tokens:
        elapsed = max(0.001, time.monotonic() - started)
        timings = TurboBenchmarkTimings(
            prompt_eval_tok_s=timings.prompt_eval_tok_s,
            generation_tok_s=timings.generation_tokens / elapsed,
            prompt_tokens=timings.prompt_tokens,
            generation_tokens=timings.generation_tokens,
            raw=timings.raw,
        )
    return timings


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _run_turbo_server_benchmark(settings: TurboBenchmarkSettings, *, n_gen: int) -> TurboBenchmarkTimings:
    host = "127.0.0.1"
    port = _find_free_port()
    base_url = f"http://{host}:{port}"
    if settings.backend_kind == "lucebox":
        cmd = _lucebox_server_cmd(settings, host=host, port=port)
        wait_for_server = _wait_for_lucebox_server
    else:
        cmd = _turbo_server_cmd(settings, host=host, port=port)
        wait_for_server = _wait_for_turbo_server
    env = _bench_env(settings.backend_path)
    log_path = Path(tempfile.gettempdir()) / f"openjet-turbo-llama-server-{os.getpid()}-{port}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        print(f"Server:  {' '.join(cmd)}")
        print(f"Log:     {log_path}")
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            wait_for_server(proc, base_url, log_path)
            return _request_turbo_completion(base_url, n_gen=n_gen, thinking_enabled=settings.thinking_enabled)
        finally:
            _terminate_process(proc)


def _format_tok_s(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f}"


def run_turbo_benchmark(
    *,
    thinking_enabled: bool = False,
    n_gen: int = 400,
    target_model: str | None = None,
    draft_model: str | None = None,
    backend_path: str | None = None,
    backend_kind: str | None = None,
    context_size: int | None = None,
    baseline_tok_s: float | None = None,
) -> None:
    cfg = load_config()
    hw = detect_hardware_info()
    cuda_summary = _get_cuda_driver_summary()
    if not hw.has_cuda:
        raise SystemExit("CUDA was not detected. DFlash turbo benchmark currently targets CUDA/NVIDIA GPUs.")
    settings = _load_turbo_settings(
        cfg,
        thinking_enabled=thinking_enabled,
        target_model=target_model,
        draft_model=draft_model,
        backend_path=backend_path,
        backend_kind=backend_kind,
        context_size=context_size,
        baseline_tok_s=baseline_tok_s,
    )

    print(f"{'=' * 60}\nOpenJet Turbo Benchmark (experimental DFlash)\n{'=' * 60}")
    print("WARNING: DFlash mode is experimental and requires spiritbuun/buun-llama-cpp.")
    print(f"Hardware:      {hw.label}; RAM={hw.total_ram_gb:.1f} GB; VRAM={hw.vram_mb:.0f} MB")
    if cuda_summary:
        print(f"CUDA:          available; {cuda_summary}")
    else:
        print("CUDA:          available")
    board = read_device_model()
    if board:
        print(f"Board:         {board}")
    print(f"Target model:  {Path(settings.target_model).name}")
    print(f"Draft model:   {Path(settings.draft_model).name}")
    print(f"Context size:  {settings.context_size}")
    print(f"Thinking mode: {'on' if settings.thinking_enabled else 'off'}")
    if settings.thinking_enabled:
        print("Warning: Qwen thinking mode can collapse DFlash acceptance for this drafter.")
    print(f"Backend:       {settings.backend_label} ({settings.backend_kind}; {settings.backend_path})")
    if settings.backend_kind == "lucebox":
        print("Timing note:   Lucebox server does not expose prompt timing; generation tok/s uses wall-clock completion tokens.")
    print(f"Generation:    {n_gen} tokens, temperature=0")

    timings = _run_turbo_server_benchmark(settings, n_gen=n_gen)
    print(f"{'=' * 60}\nResult\n{'=' * 60}")
    print(f"prompt eval tok/s: {_format_tok_s(timings.prompt_eval_tok_s)}")
    print(f"generation tok/s:  {_format_tok_s(timings.generation_tok_s)}")
    if settings.baseline_tok_s and timings.generation_tok_s:
        print(f"speedup vs baseline: {timings.generation_tok_s / settings.baseline_tok_s:.2f}x")
    elif settings.baseline_tok_s:
        print("speedup vs baseline: unavailable")
    else:
        print("speedup vs baseline: not provided")


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

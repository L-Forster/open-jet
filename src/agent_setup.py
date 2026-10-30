"""Hardware-aware setup for callers without an interactive terminal."""

from __future__ import annotations

import asyncio
import csv
import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import yaml

from .config import CONFIG_PATH, load_config, load_project_config, setup_direct_model_catalog
from .hardware import detect_hardware_info, recommended_device_for_hardware
from .llama_server import megakernel_long_context_limit, megakernel_long_context_reason
from .model_profiles import sync_active_model_profile
from .provisioning import provision_setup_artifacts, recommend_direct_model
from .runtime_limits import read_memory_snapshot
from .runtime_registry import create_runtime_client
from .setup import _direct_catalog_payload, _recommended_context_for_payload
from .setup_memory import _detect_free_memory_mb


class SetupError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def inspect_hardware():
    detected = detect_hardware_info()
    budget = detected
    memory = read_memory_snapshot()
    facts = {**asdict(detected), "os": platform.system(), "architecture": platform.machine(), "gpus": []}
    if memory:
        facts["available_ram_gb"] = memory.available_mb / 1024
        budget = replace(budget, total_ram_gb=min(detected.total_ram_gb, memory.available_mb / 1024))
    if detected.has_metal:
        budget = replace(budget, vram_mb=budget.total_ram_gb * 1024)
    elif not detected.has_cuda:
        free = _detect_free_memory_mb(recommended_device_for_hardware("auto", detected))
        if free is not None:
            budget = replace(budget, vram_mb=min(detected.vram_mb, free) if detected.vram_mb else free)
    if detected.has_cuda:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            gpus = []
            for row in csv.reader(io.StringIO(output)):
                index, uuid, name, total, free = [part.strip() for part in row]
                gpus.append({"index": index, "uuid": uuid, "name": name, "total_mb": float(total), "free_mb": float(free)})
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                ids = [part.strip() for part in visible.split(",") if part.strip()]
                gpus = [gpu for gpu in gpus if any(gpu["index"] == value or gpu["uuid"].startswith(value) for value in ids)]
            if not gpus:
                raise SetupError("hardware", "No visible NVIDIA GPU. Check CUDA_VISIBLE_DEVICES and the driver.")
            facts["gpus"] = gpus
            # A conservative single-device budget; never assume VRAM can be pooled.
            budget = replace(budget, vram_mb=min(gpu["free_mb"] for gpu in gpus))
        except (OSError, ValueError, subprocess.SubprocessError):
            if "jetson" not in detected.label.lower() and not Path("/dev/nvhost-gpu").exists():
                raise SetupError("hardware", "NVIDIA memory could not be queried. Check that nvidia-smi works.")
    facts["budget"] = asdict(budget)
    return budget, facts


def _config_target() -> Path:
    if load_project_config() is not None:
        raise SetupError("configuration", "A project .openjet/config.yaml is active. Run setup from a directory without a project model overlay.")
    local = Path.cwd() / "config.yaml"
    return local if local.exists() else CONFIG_PATH


def _save_verified_config(cfg: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(cfg, handle, sort_keys=False)
        if target.exists():
            temporary.chmod(target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def verify_inference(cfg: dict) -> dict:
    client = create_runtime_client({**cfg, "setup_verification": True})
    try:
        async def probe():
            await client.start()
            async with httpx.AsyncClient(timeout=60) as http:
                response = await http.post(
                    f"{client.base_url}/v1/chat/completions",
                    json={"model": "local", "messages": [{"role": "user", "content": "Say hello briefly."}],
                          "max_tokens": 64, "stream": False, "chat_template_kwargs": {"enable_thinking": False}},
                )
                response.raise_for_status()
                choices = response.json().get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                if not str(message.get("content") or "").strip():
                    raise RuntimeError("The runtime returned no generated text.")
                return {"passed": True, "backend": client.active_backend,
                        "context_window_tokens": client.context_window_tokens, "gpu_layers": client.gpu_layers}
        return await asyncio.wait_for(probe(), timeout=300)
    finally:
        await client.close()


async def run_setup(*, model: str | None = None, context: int | None = None, dry_run: bool = False, yes: bool = False,
                    long_context: bool = False, log=None) -> dict:
    if log is None:
        class StderrLog:
            def write(self, text):
                print(text, file=sys.stderr)
        log = StderrLog()
    if not dry_run and not yes:
        raise SetupError("confirmation", "Pass --yes to download/build and configure OpenJet, or --dry-run to inspect the recommendation.")
    target = _config_target()
    cfg = load_config()
    if cfg.get("runtime", "llama_cpp") != "llama_cpp" and not model:
        raise SetupError("configuration", "An API profile is active. Pass --model to explicitly select a local model.")
    hardware, facts = inspect_hardware()
    device = recommended_device_for_hardware("auto", hardware)
    direct = recommend_direct_model(hardware, cfg=cfg)
    catalog = setup_direct_model_catalog(cfg)
    existing = str(cfg.get("llama_model") or "")
    selected = next((row for row in catalog if model in (row["filename"], row["label"])), None) if model else None
    local = Path(model or existing).expanduser() if model or existing else None
    if selected:
        payload = _direct_catalog_payload(selected)
        payload["llama_cpp_ref"] = str(selected.get("llama_cpp_ref") or "")
    elif local is not None and local.is_file():
        if local.suffix.lower() != ".gguf":
            raise SetupError("model", "Local models must be .gguf files.")
        with local.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:4] != b"GGUF" or int.from_bytes(header[4:8], "little") not in {2, 3}:
            raise SetupError("model", f"Invalid or incomplete GGUF file: {local}")
        payload = {"model_source": "local", "llama_model": str(local.resolve()), "model_size_mb": local.stat().st_size / (1024 ** 2)}
        # Preserve tuned runtime settings when reusing the configured model.
        if not model or str(local.resolve()) == str(Path(existing).expanduser().resolve()):
            for key in ("llama_cpu_moe", "llama_n_cpu_moe", "llama_mtp", "active_model_size_mb"):
                if key in cfg:
                    payload[key] = cfg[key]
    elif model:
        raise SetupError("model", f"Unknown model {model!r}. Use a catalog filename from --dry-run or an existing GGUF path.")
    else:
        payload = _direct_catalog_payload(direct)
        payload.update({key: direct[key] for key in ("llama_cpu_moe", "llama_n_cpu_moe")})
        payload["llama_cpp_ref"] = str(direct.get("llama_cpp_ref") or "")
    gpu = device in {"cuda", "rocm", "vulkan"} and hardware.vram_mb > 0
    budget_mb = hardware.vram_mb if gpu else hardware.total_ram_gb * 1024
    size_mb = float(payload.get("active_model_size_mb") or payload.get("model_size_mb") or 0)
    if budget_mb <= 0 or size_mb > budget_mb * 0.9:
        raise SetupError("model", "No selected model fits available memory with runtime headroom. Free memory or select a smaller catalog model with --model.")
    recommended_context = _recommended_context_for_payload(
        payload, device=device, fallback_tokens=2048, total_vram_mb=hardware.vram_mb,
    )
    if long_context:
        planned = str(payload.get("llama_model") or payload.get("model_download_path") or "")
        reason = megakernel_long_context_reason(planned)
        if reason:
            raise SetupError("model", f"--long-context needs the megakernel: {reason}.")
        recommended_context = megakernel_long_context_limit(recommended_context, int(payload.get("max_context_tokens") or 0))
        payload["megakernel_long_context"] = True
    if context is not None and (context < 512 or context > recommended_context):
        raise SetupError("model", f"--context-tokens must be between 512 and the recommended memory limit ({recommended_context}).")
    payload.update({"runtime": "llama_cpp", "hardware_profile": "auto", "hardware_override": "", "device": device,
                    "context_window_tokens": context or recommended_context, "gpu_layers": 0 if device == "cpu" else 99,
                    "setup_complete": False, "preserve_model_files": True})
    server_path = str(cfg.get("llama_server_path") or "")
    if server_path and Path(server_path).expanduser().is_file():
        payload["llama_server_path"] = str(Path(server_path).expanduser().resolve())
    if not model and existing and Path(existing).expanduser().is_file():
        payload["context_window_tokens"] = context or min(int(cfg.get("context_window_tokens") or recommended_context), recommended_context)
        payload["gpu_layers"] = int(cfg.get("gpu_layers", payload["gpu_layers"]))
    planned_path = Path(str(payload.get("llama_model") or payload.get("model_download_path")))
    parent = planned_path.parent
    while not parent.exists():
        parent = parent.parent
    download_bytes = 0 if planned_path.is_file() else int(float(payload.get("model_size_mb") or 0) * 1024 ** 2)
    if download_bytes and shutil.disk_usage(parent).free < download_bytes + 2 * 1024 ** 3:
        raise SetupError("disk", "Insufficient free disk space for the model plus 2 GiB of runtime/build headroom.")
    result = {"schema_version": 1, "status": "planned", "hardware": facts, "configuration": str(target),
              "model": str(planned_path), "context_window_tokens": payload["context_window_tokens"], "device": device,
              "download_bytes": download_bytes, "catalog": [{"id": row["filename"], "label": row["label"]} for row in catalog],
              "next_steps": ["cd <your-project>", "openjet", 'openjet chat "Explain this repository"']}
    if dry_run:
        return result
    if log:
        log.write(f"Selected {planned_path.name}; device={device}, context={payload['context_window_tokens']}")
    try:
        resolved = await provision_setup_artifacts(payload, hardware_info=hardware, log=log,
                                                  set_status=lambda text: log.write(text) if log else None,
                                                  clear_status=lambda: None)
    except Exception as exc:
        raise SetupError("provisioning", str(exc)) from exc
    # Clear model-specific tuning from the previous profile when changing models.
    candidate = dict(cfg)
    for key in ("model_download_url", "model_download_path", "model_size_mb", "active_model_size_mb",
                "resident_model_size_mb", "kv_bytes_per_token", "max_context_tokens", "unified_memory_only",
                "megakernel_long_context"):
        candidate.pop(key, None)
    candidate.update({"llama_cpu_moe": False, "llama_n_cpu_moe": 0, "llama_mtp": False, **resolved})
    try:
        if log:
            log.write("Verifying local inference...")
        verification = await verify_inference(candidate)
    except Exception as exc:
        raise SetupError("verification", str(exc) or "Inference verification timed out.") from exc
    candidate.update({"setup_complete": True, "context_window_tokens": verification["context_window_tokens"], "gpu_layers": verification["gpu_layers"]})
    profile_name = candidate.pop("model_profile_name", None)
    candidate.pop("preserve_model_files", None)
    sync_active_model_profile(candidate, preferred_name=profile_name)
    _save_verified_config(candidate, target)
    result.update({"status": "ready", "model": candidate["llama_model"], "verification": verification,
                   "context_window_tokens": candidate["context_window_tokens"], "device": candidate["device"]})
    return result

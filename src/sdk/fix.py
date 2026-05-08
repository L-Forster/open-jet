from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..hardware import HardwareInfo, detect_hardware_info
from .recommendations import HardwareRecommendation, recommend_hardware_config
from .tok_s import (
    MTP_DECODE_SPEEDUP,
    TokenGenerationEstimate,
    estimate_token_generation_speed,
)


_RUNTIME_ALIASES = {
    "llama.cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama-server": "llama.cpp",
    "ollama": "ollama",
    "vllm": "vllm",
    "mlx": "mlx",
    "mlx-lm": "mlx",
}


@dataclass(frozen=True)
class RuntimeProcess:
    backend: str
    pid: int
    argv: tuple[str, ...]
    executable: str = ""
    host: str = "127.0.0.1"
    port: int | None = None

    @property
    def command(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class RuntimeObservation:
    backend: str
    process: RuntimeProcess | None = None
    model_path: str = ""
    device: str = "auto"
    gpu_layers: int | None = None
    context_window_tokens: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    flash_attention: bool | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FixFinding:
    ok: bool | None
    text: str


@dataclass(frozen=True)
class FixEstimate:
    decode_current_tok_s: float | None = None
    decode_target_low_tok_s: float | None = None
    decode_target_high_tok_s: float | None = None
    prefill_current_tok_s: float | None = None
    prefill_target_low_tok_s: float | None = None
    prefill_target_high_tok_s: float | None = None
    host_ram_reduction_mb: float | None = None
    confidence: str = "low"
    note: str = ""


@dataclass(frozen=True)
class FixReport:
    backend: str
    hardware: HardwareInfo
    recommendation: HardwareRecommendation
    observation: RuntimeObservation
    findings: tuple[FixFinding, ...]
    recommended_args: tuple[str, ...]
    estimate: FixEstimate
    detected_processes: tuple[RuntimeProcess, ...] = field(default_factory=tuple)


class RuntimeFixer:
    backend = "unknown"

    def matches(self, argv: Sequence[str], executable: str) -> bool:
        raise NotImplementedError

    def inspect(
        self,
        process: RuntimeProcess | None,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        run_probe: bool,
    ) -> RuntimeObservation:
        raise NotImplementedError

    def diagnose(
        self,
        observation: RuntimeObservation,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        recommendation: HardwareRecommendation,
    ) -> tuple[tuple[FixFinding, ...], tuple[str, ...], FixEstimate]:
        raise NotImplementedError


class LlamaCppFixer(RuntimeFixer):
    backend = "llama.cpp"

    def matches(self, argv: Sequence[str], executable: str) -> bool:
        names = {Path(executable).name.lower()}
        if argv:
            names.add(Path(argv[0]).name.lower())
        if names & {"llama-server", "llama-server.exe"}:
            return True
        command = " ".join([executable, *argv]).lower()
        if "llama.cpp" in command and names & {"server", "server.exe", "main", "main.exe", "llama-cli", "llama-cli.exe"}:
            return True
        if ".gguf" in command and any(flag in command for flag in ("--port", " -m ", "--model", "-ngl", "--n-gpu-layers")):
            return True
        return "llama-server" in command

    def inspect(
        self,
        process: RuntimeProcess | None,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        run_probe: bool,
    ) -> RuntimeObservation:
        if process is None:
            return RuntimeObservation(
                backend=self.backend,
                notes=("No llama.cpp process was found in the local process table.",),
            )
        argv = process.argv if process else ()
        model_path = _flag_value(argv, "-m", "--model")
        device = str(cfg.get("device") or "auto").strip().lower() or "auto"
        gpu_layers = _int_flag(argv, "-ngl", "--n-gpu-layers", "--gpu-layers")
        if gpu_layers is None and argv:
            gpu_layers = _cfg_int(cfg, "gpu_layers")
        ctx = _int_flag(argv, "-c", "--ctx-size", "--ctx_size")
        if ctx is None and argv:
            ctx = _cfg_int(cfg, "context_window_tokens")
        batch = _int_flag(argv, "-b", "--batch-size", "--batch_size")
        ubatch = _int_flag(argv, "-ub", "--ubatch-size", "--ubatch_size")
        flash = _bool_flag(argv, "-fa", "--flash-attn", "--flash_attn")
        draft_engaged = _llama_cpp_draft_engaged(argv)
        model_has_mtp_suffix = _model_has_mtp_suffix(model_path)
        prefill: float | None = None
        decode: float | None = None
        notes: list[str] = []
        metadata: dict[str, object] = {
            "draft_engaged": draft_engaged,
            "model_has_mtp_suffix": model_has_mtp_suffix,
        }
        if run_probe and process.port:
            probed = _probe_llama_cpp_decode(process.host, process.port)
            if probed is not None:
                metadata["raw_decode_tok_s"] = probed
                decode = _effective_llama_cpp_decode_speed(
                    probed,
                    draft_engaged=draft_engaged,
                    model_has_mtp_suffix=model_has_mtp_suffix,
                )
        return RuntimeObservation(
            backend=self.backend,
            process=process,
            model_path=model_path,
            device=device,
            gpu_layers=gpu_layers,
            context_window_tokens=ctx,
            batch_size=batch,
            ubatch_size=ubatch,
            flash_attention=flash,
            prefill_tok_s=prefill,
            decode_tok_s=decode,
            notes=tuple(notes),
            metadata=metadata,
        )

    def diagnose(
        self,
        observation: RuntimeObservation,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        recommendation: HardwareRecommendation,
    ) -> tuple[tuple[FixFinding, ...], tuple[str, ...], FixEstimate]:
        recommended_device = recommendation.llama.device
        recommended_ngl = recommendation.llama.gpu_layers
        recommended_ctx = _recommended_fix_context_tokens(recommendation.llama.context_window_tokens)
        recommended_batch = 2048 if recommended_device != "cpu" else 512
        recommended_ubatch = 512 if recommended_device != "cpu" else 128

        if observation.process is None:
            findings = (
                FixFinding(False, "No running llama.cpp process detected"),
                FixFinding(True if any((hardware.has_cuda, hardware.has_rocm, hardware.has_vulkan, hardware.has_metal)) else False, f"{_accelerator_label(hardware)} device available"),
                FixFinding(None, "Runtime flags not inspected"),
                FixFinding(None, "Throughput not measured"),
            )
            args = _llama_recommended_args(
                device=recommended_device,
                gpu_layers=recommended_ngl,
                context_window_tokens=recommended_ctx,
                batch_size=recommended_batch,
                ubatch_size=recommended_ubatch,
            )
            return findings, tuple(args), FixEstimate(
                confidence="none",
                note="no running runtime was available to compare or benchmark",
            )

        model_path = observation.model_path
        is_gguf = Path(model_path).suffix.lower() == ".gguf" if model_path else False
        has_accelerator = any((hardware.has_cuda, hardware.has_rocm, hardware.has_vulkan, hardware.has_metal))
        backend_label = _accelerator_label(hardware)
        flash_ok = observation.flash_attention
        if flash_ok is None and observation.process is not None:
            flash_ok = False
        batch_ok = _passes_min_int(observation.batch_size, recommended_batch)
        ubatch_ok = _passes_min_int(observation.ubatch_size, recommended_ubatch)
        ctx_ok = _passes_min_int(observation.context_window_tokens, recommended_ctx)
        findings = [
            FixFinding(True if backend_label else None, f"{backend_label or 'Accelerator'} device detected"),
            FixFinding(True if is_gguf else False if model_path else None, "Model loaded from GGUF"),
            FixFinding(
                _passes_min_int(observation.gpu_layers, recommended_ngl) if has_accelerator else True,
                "n_gpu_layers matches OpenJet target" if _passes_min_int(observation.gpu_layers, recommended_ngl) else "n_gpu_layers appears too low",
            ),
            FixFinding(
                flash_ok,
                "Flash attention enabled" if flash_ok else "Flash attention not enabled" if flash_ok is False else "Flash attention not observed",
            ),
            FixFinding(
                batch_ok,
                "batch size sized for prefill" if batch_ok else "batch size too small for prefill" if batch_ok is False else "batch size not observed",
            ),
            FixFinding(
                ubatch_ok,
                "ubatch size sized for GPU kernels" if ubatch_ok else "ubatch size too small for GPU kernels" if ubatch_ok is False else "ubatch size not observed",
            ),
            FixFinding(
                ctx_ok,
                "context window matches OpenJet target" if ctx_ok else "context window below OpenJet target" if ctx_ok is False else "context window not observed",
            ),
            FixFinding(
                True if observation.decode_tok_s is not None else None,
                "Decode speed measured" if observation.decode_tok_s is not None else "Decode speed not measured",
            ),
        ]
        args = _llama_recommended_args(
            device=recommended_device,
            gpu_layers=recommended_ngl,
            context_window_tokens=recommended_ctx,
            batch_size=recommended_batch,
            ubatch_size=recommended_ubatch,
        )
        estimate = _estimate_llama_speedup(
            observation,
            cfg=cfg,
            recommendation=recommendation,
            findings=tuple(findings),
        )
        return tuple(findings), tuple(args), estimate


class OllamaFixer(RuntimeFixer):
    backend = "ollama"
    default_port = 11434

    def matches(self, argv: Sequence[str], executable: str) -> bool:
        names = {Path(executable).name.lower()}
        if argv:
            names.add(Path(argv[0]).name.lower())
        return "ollama" in names

    def inspect(
        self,
        process: RuntimeProcess | None,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        run_probe: bool,
    ) -> RuntimeObservation:
        notes: list[str] = []
        port = process.port if process and process.port else self.default_port
        api_models = _ollama_api_ps("127.0.0.1", port)
        loaded = _ollama_ps_rows()
        installed = _ollama_list_rows()
        if api_models:
            model = api_models[0]
            model_name = str(model.get("name") or model.get("model") or "")
            ctx = _coerce_int(model.get("context_length"))
            processor = _ollama_processor_for_model(loaded, model_name)
            metadata = {
                "parameter_size": _ollama_detail(model, "parameter_size"),
                "quantization_level": _ollama_detail(model, "quantization_level"),
                "size_mb": _bytes_to_mb(model.get("size")),
                "size_vram_mb": _bytes_to_mb(model.get("size_vram")),
                "processor": processor,
            }
            decode = _probe_ollama_decode("127.0.0.1", port, model_name) if run_probe else None
            notes.append(_format_ollama_rows_note("loaded", [{"name": model_name}]))
            return RuntimeObservation(
                backend=self.backend,
                process=process,
                model_path=model_name,
                context_window_tokens=ctx,
                metadata=metadata,
                decode_tok_s=decode,
                notes=tuple(notes),
            )
        if installed:
            notes.append(_format_ollama_rows_note("installed", installed))
        else:
            notes.append("No Ollama models are installed or loaded.")
        return RuntimeObservation(
            backend=self.backend,
            process=process,
            notes=tuple(notes),
        )

    def diagnose(
        self,
        observation: RuntimeObservation,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        recommendation: HardwareRecommendation,
    ) -> tuple[tuple[FixFinding, ...], tuple[str, ...], FixEstimate]:
        findings = (
            FixFinding(True if observation.process else False, "Ollama service detected" if observation.process else "No Ollama service detected"),
            FixFinding(True if observation.model_path else None, "Loaded Ollama model detected" if observation.model_path else "No loaded Ollama model detected"),
        )
        return findings, (), FixEstimate(confidence="none", note="Ollama comparison uses process/model metadata only")


class GenericRuntimeFixer(RuntimeFixer):
    def __init__(self, backend: str, markers: tuple[str, ...], default_port: int | None = None) -> None:
        self.backend = backend
        self.markers = markers
        self.default_port = default_port

    def matches(self, argv: Sequence[str], executable: str) -> bool:
        haystack = " ".join([executable, *argv]).lower()
        return any(marker in haystack for marker in self.markers)

    def inspect(
        self,
        process: RuntimeProcess | None,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        run_probe: bool,
    ) -> RuntimeObservation:
        return RuntimeObservation(
            backend=self.backend,
            process=process,
            model_path=_flag_value(process.argv if process else (), "-m", "--model", "--model-name") or "",
            notes=("Backend-specific optimization rules are not implemented yet.",),
        )

    def diagnose(
        self,
        observation: RuntimeObservation,
        *,
        cfg: Mapping[str, object],
        hardware: HardwareInfo,
        recommendation: HardwareRecommendation,
    ) -> tuple[tuple[FixFinding, ...], tuple[str, ...], FixEstimate]:
        findings = (
            FixFinding(True if observation.process else None, f"{self.backend} process detected"),
            FixFinding(None, "Backend-specific diagnosis is pending"),
            FixFinding(True if any((hardware.has_cuda, hardware.has_rocm, hardware.has_vulkan, hardware.has_metal)) else False, f"{_accelerator_label(hardware) or 'Accelerator'} available"),
        )
        estimate = FixEstimate(
            decode_target_low_tok_s=recommendation.tok_s.estimated_tokens_per_second,
            decode_target_high_tok_s=recommendation.tok_s.estimated_tokens_per_second,
            confidence="low",
            note="OpenJet can estimate the target model/device path, but this runtime fixer needs backend-specific rules.",
        )
        return findings, (), estimate


_FIXERS: tuple[RuntimeFixer, ...] = (
    LlamaCppFixer(),
    OllamaFixer(),
    GenericRuntimeFixer("vllm", ("vllm", "api_server"), default_port=8000),
    GenericRuntimeFixer("mlx", ("mlx_lm", "mlx-lm", "mlx.server"), default_port=8080),
)


def fix(
    backend: str | None = None,
    *,
    cfg: Mapping[str, object] | None = None,
    hardware: HardwareInfo | None = None,
    run_probe: bool = False,
    observed_prefill_tok_s: float | None = None,
    observed_decode_tok_s: float | None = None,
) -> FixReport:
    """Diagnose a running local LLM runtime against OpenJet's computed setup.

    ``backend`` can be omitted for auto-detection or set to aliases like
    ``llama.cpp``, ``ollama``, ``vllm``, or ``mlx``.
    """

    current_cfg = dict(cfg or {})
    detected_hardware = hardware or detect_hardware_info()
    recommendation = recommend_hardware_config(detected_hardware, cfg=current_cfg)
    processes = tuple(detect_runtime_processes())
    fixer = _resolve_fixer(backend, processes)
    process = _select_process(fixer.backend, processes)
    observation = fixer.inspect(
        process,
        cfg=current_cfg,
        hardware=detected_hardware,
        run_probe=run_probe,
    )
    if observed_prefill_tok_s is not None or observed_decode_tok_s is not None:
        decode_tok_s = observed_decode_tok_s if observed_decode_tok_s is not None else observation.decode_tok_s
        metadata = dict(observation.metadata)
        if observed_decode_tok_s is not None and observation.backend == "llama.cpp":
            metadata["raw_decode_tok_s"] = observed_decode_tok_s
            decode_tok_s = _effective_llama_cpp_decode_speed(
                observed_decode_tok_s,
                draft_engaged=bool(observation.metadata.get("draft_engaged")),
                model_has_mtp_suffix=bool(observation.metadata.get("model_has_mtp_suffix")),
            )
        observation = RuntimeObservation(
            backend=observation.backend,
            process=observation.process,
            model_path=observation.model_path,
            device=observation.device,
            gpu_layers=observation.gpu_layers,
            context_window_tokens=observation.context_window_tokens,
            batch_size=observation.batch_size,
            ubatch_size=observation.ubatch_size,
            flash_attention=observation.flash_attention,
            prefill_tok_s=observed_prefill_tok_s if observed_prefill_tok_s is not None else observation.prefill_tok_s,
            decode_tok_s=decode_tok_s,
            notes=observation.notes,
            metadata=metadata,
        )
    findings, args, estimate = fixer.diagnose(
        observation,
        cfg=current_cfg,
        hardware=detected_hardware,
        recommendation=recommendation,
    )
    return FixReport(
        backend=fixer.backend,
        hardware=detected_hardware,
        recommendation=recommendation,
        observation=observation,
        findings=findings,
        recommended_args=args,
        estimate=estimate,
        detected_processes=processes,
    )


def detect_runtime_processes() -> list[RuntimeProcess]:
    processes: list[RuntimeProcess] = []
    for pid, argv, executable in _iter_process_argv():
        fixer = _matching_fixer(argv, executable)
        if fixer is None:
            continue
        port = _int_flag(argv, "--port") or getattr(fixer, "default_port", None)
        host = _flag_value(argv, "--host") or "127.0.0.1"
        processes.append(
            RuntimeProcess(
                backend=fixer.backend,
                pid=pid,
                argv=tuple(argv),
                executable=executable,
                host=host,
                port=port,
            )
        )
    return processes


def format_fix_report(report: FixReport) -> str:
    return _format_unified_fix_report(report)


def _resolve_fixer(backend: str | None, processes: Sequence[RuntimeProcess]) -> RuntimeFixer:
    normalized = _normalize_backend(backend) if backend else ""
    if normalized:
        for fixer in _FIXERS:
            if fixer.backend == normalized:
                return fixer
        return GenericRuntimeFixer(normalized, (normalized,))
    if processes:
        for fixer in _FIXERS:
            if any(process.backend == fixer.backend for process in processes):
                return fixer
    return _FIXERS[0]


def _select_process(backend: str, processes: Sequence[RuntimeProcess]) -> RuntimeProcess | None:
    if backend == "ollama":
        for process in processes:
            if any(arg == "serve" for arg in process.argv[1:]):
                return process
    for process in processes:
        if process.backend == backend and process.port is not None:
            return process
    for process in processes:
        if process.backend == backend:
            return process
    return None


def _matching_fixer(argv: Sequence[str], executable: str) -> RuntimeFixer | None:
    for fixer in _FIXERS:
        if fixer.matches(argv, executable):
            return fixer
    return None


def _normalize_backend(backend: str | None) -> str:
    key = str(backend or "").strip().lower().replace(" ", "")
    return _RUNTIME_ALIASES.get(key, key)


def _iter_process_argv() -> list[tuple[int, list[str], str]]:
    current_pid = os.getpid()
    if os.name == "posix":
        proc_dir = Path("/proc")
        if proc_dir.is_dir():
            rows: list[tuple[int, list[str], str]] = []
            for entry in proc_dir.iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == current_pid:
                    continue
                try:
                    raw = (entry / "cmdline").read_bytes()
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if not raw:
                    continue
                argv = [part for part in raw.decode(errors="ignore").split("\x00") if part]
                if not argv:
                    continue
                executable = argv[0]
                try:
                    executable = str((entry / "exe").readlink())
                except (FileNotFoundError, PermissionError, OSError):
                    pass
                rows.append((pid, argv, executable))
            return rows
    return _iter_process_argv_with_ps()


def _iter_process_argv_with_ps() -> list[tuple[int, list[str], str]]:
    try:
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,command="],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, list[str], str]] = []
    current_pid = os.getpid()
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        argv = command.split()
        if argv:
            rows.append((pid, argv, Path(argv[0]).name))
    return rows


def _ollama_ps_rows() -> list[dict[str, str]]:
    return _ollama_table(["ollama", "ps"])


def _ollama_list_rows() -> list[dict[str, str]]:
    return _ollama_table(["ollama", "list"])


def _ollama_table(cmd: list[str]) -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(cmd, text=True, timeout=5, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    headers = [header.lower() for header in re.split(r"\s{2,}", lines[0].strip())]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = re.split(r"\s{2,}", line.strip())
        if not values:
            continue
        row: dict[str, str] = {}
        for idx, header in enumerate(headers):
            if idx < len(values):
                row[header] = values[idx]
        if row:
            rows.append(row)
    return rows


def _ollama_api_ps(host: str, port: int) -> list[dict[str, object]]:
    try:
        with urlopen(f"http://{host}:{port}/api/ps", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Mapping):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    return [dict(model) for model in models if isinstance(model, Mapping)]


def _ollama_detail(model: Mapping[str, object], key: str) -> str:
    details = model.get("details")
    if not isinstance(details, Mapping):
        return ""
    return str(details.get(key) or "")


def _ollama_processor_for_model(rows: list[dict[str, str]], model_name: str) -> str:
    normalized = model_name.strip().lower()
    for row in rows:
        name = str(row.get("name") or "").strip().lower()
        if name == normalized:
            return str(row.get("processor") or "").strip()
    if rows:
        return str(rows[0].get("processor") or "").strip()
    return ""


def _parse_ollama_context(value: str) -> int | None:
    text = value.strip().lower()
    if not text:
        return None
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return None


def _format_ollama_rows_note(label: str, rows: list[dict[str, str]]) -> str:
    names = [str(row.get("name") or "").strip() for row in rows]
    names = [name for name in names if name]
    if not names:
        return f"Ollama {label} model list is empty."
    return f"Ollama {label}: {', '.join(names[:5])}"


def _coerce_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed


def _bytes_to_mb(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed / (1024.0 * 1024.0)


def _flag_value(argv: Sequence[str], *names: str) -> str:
    values = set(names)
    for idx, arg in enumerate(argv):
        if arg in values and idx + 1 < len(argv):
            return str(argv[idx + 1])
        for name in values:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix):]
    return ""


def _int_flag(argv: Sequence[str], *names: str) -> int | None:
    raw = _flag_value(argv, *names)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _bool_flag(argv: Sequence[str], *names: str) -> bool | None:
    values = set(names)
    for idx, arg in enumerate(argv):
        if arg in values:
            if idx + 1 >= len(argv):
                return True
            nxt = str(argv[idx + 1]).strip().lower()
            if nxt in {"1", "true", "on", "yes"}:
                return True
            if nxt in {"0", "false", "off", "no"}:
                return False
            return True
        for name in values:
            prefix = f"{name}="
            if arg.startswith(prefix):
                value = arg[len(prefix):].strip().lower()
                if value in {"1", "true", "on", "yes"}:
                    return True
                if value in {"0", "false", "off", "no"}:
                    return False
    return None


def _llama_cpp_draft_engaged(argv: Sequence[str]) -> bool:
    return bool(
        _flag_value(
            argv,
            "--model-draft",
            "--draft-model",
            "--draft",
            "-md",
        )
    )


def _model_has_mtp_suffix(model_path: str) -> bool:
    if not model_path:
        return False
    name = Path(model_path).name.lower()
    stem = Path(name).stem
    return bool(re.search(r"(?:^|[-_.])mtp(?:[-_.]|$)", stem))


def _effective_llama_cpp_decode_speed(
    raw_decode_tok_s: float,
    *,
    draft_engaged: bool,
    model_has_mtp_suffix: bool,
) -> float:
    if draft_engaged or model_has_mtp_suffix:
        return raw_decode_tok_s * MTP_DECODE_SPEEDUP
    return raw_decode_tok_s


def _cfg_int(cfg: Mapping[str, object], key: str) -> int | None:
    try:
        return int(cfg.get(key) or 0)
    except (TypeError, ValueError):
        return None


def _passes_min_int(value: int | None, target: int) -> bool | None:
    if value is None:
        return None
    return value >= target


def _accelerator_label(hardware: HardwareInfo) -> str:
    if hardware.has_cuda:
        return "CUDA"
    if hardware.has_rocm:
        return "ROCm"
    if hardware.has_metal:
        return "Metal"
    if hardware.has_vulkan:
        return "Vulkan"
    return "CPU"


def _llama_recommended_args(
    *,
    device: str,
    gpu_layers: int,
    context_window_tokens: int,
    batch_size: int,
    ubatch_size: int,
) -> list[str]:
    args: list[str] = []
    if device != "cpu":
        args.extend(["--n-gpu-layers", "-1" if gpu_layers >= 99 else str(gpu_layers)])
        args.append("--flash-attn")
    args.extend(
        [
            "--ctx-size",
            str(context_window_tokens),
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(ubatch_size),
        ]
    )
    return args


def _recommended_fix_context_tokens(setup_context_tokens: int) -> int:
    return max(512, min(int(setup_context_tokens or 0), 32768))


def _estimate_llama_speedup(
    observation: RuntimeObservation,
    *,
    cfg: Mapping[str, object],
    recommendation: HardwareRecommendation,
    findings: tuple[FixFinding, ...],
) -> FixEstimate:
    multiplier = 1.0
    prefill_multiplier = 1.0
    if any("n_gpu_layers appears too low" in finding.text for finding in findings):
        current = max(0, observation.gpu_layers or 0)
        target = max(1, recommendation.llama.gpu_layers)
        gap = max(0.0, min(1.0, (target - current) / target))
        multiplier *= 1.25 + gap
        prefill_multiplier *= 1.15 + gap * 0.8
    if any("Flash attention not enabled" in finding.text for finding in findings):
        multiplier *= 1.08
        prefill_multiplier *= 1.25
    if any("batch size too small" in finding.text for finding in findings):
        prefill_multiplier *= 1.45

    decode_current = observation.decode_tok_s
    prefill_current = observation.prefill_tok_s
    target_decode = recommendation.tok_s.estimated_tokens_per_second
    if decode_current is not None:
        low = max(decode_current, decode_current * multiplier * 0.9)
        high = max(low, decode_current * multiplier * 1.15)
        confidence = "medium"
    elif target_decode > 0:
        low = target_decode * 0.85
        high = target_decode * 1.15
        confidence = "low"
    else:
        low = high = None
        confidence = "low"
    if prefill_current is not None:
        prefill_low = max(prefill_current, prefill_current * prefill_multiplier * 0.9)
        prefill_high = max(prefill_low, prefill_current * prefill_multiplier * 1.2)
    else:
        prefill_low = prefill_high = None
    host_ram_reduction = _estimate_host_ram_reduction_mb(observation, cfg=cfg, recommendation=recommendation)
    if decode_current is not None and prefill_current is not None:
        note = "requires benchmark after restart"
    else:
        note = "run with --probe or pass observed speeds for a before/after estimate"
    return FixEstimate(
        decode_current_tok_s=decode_current,
        decode_target_low_tok_s=low,
        decode_target_high_tok_s=high,
        prefill_current_tok_s=prefill_current,
        prefill_target_low_tok_s=prefill_low,
        prefill_target_high_tok_s=prefill_high,
        host_ram_reduction_mb=host_ram_reduction,
        confidence=confidence,
        note=note,
    )


def _estimate_host_ram_reduction_mb(
    observation: RuntimeObservation,
    *,
    cfg: Mapping[str, object],
    recommendation: HardwareRecommendation,
) -> float | None:
    if observation.gpu_layers is None or recommendation.llama.gpu_layers <= observation.gpu_layers:
        return None
    if not observation.model_path:
        return None
    model_size_mb = _model_size_mb(observation.model_path, cfg)
    if model_size_mb <= 0:
        return None
    current_fraction = max(0.0, min(1.0, observation.gpu_layers / max(1, recommendation.llama.gpu_layers)))
    return model_size_mb * (1.0 - current_fraction)


def _model_size_mb(model_path: str, cfg: Mapping[str, object]) -> float:
    for key in ("resident_model_size_mb", "model_size_mb", "active_model_size_mb"):
        try:
            value = float(cfg.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    if model_path:
        try:
            return Path(model_path).expanduser().stat().st_size / (1024.0 * 1024.0)
        except OSError:
            return 0.0
    return 0.0


_VERDICT_HEADLINES = {
    "no_runtime": "OpenJet did not detect a running local LLM on this machine.",
    "barely_usable": "OpenJet found why your current local LLM setup is barely usable.",
    "slow": "OpenJet found why your current local LLM setup is slow.",
    "optimal": "Your current local LLM setup is already close to optimal.",
}


def _format_unified_fix_report(report: FixReport) -> str:
    verdict = _classify_verdict(report)
    rows: list[list[tuple[str, str]]] = []

    sections: list[str] = [_VERDICT_HEADLINES[verdict], ""]

    current_rows = _current_setup_rows(report, verdict)
    sections.append("Current setup")
    sections.append(_format_kv_table(current_rows))
    sections.append("")

    if verdict in {"slow", "barely_usable"}:
        sections.append("Problem")
        sections.append(_format_problem(report, verdict))
        sections.append("")

    target_rows = _openjet_target_rows(report, verdict)
    sections.append("OpenJet setup target")
    sections.append(_format_kv_table(target_rows))
    sections.append("")

    if verdict in {"slow", "barely_usable"}:
        sections.append("Difference")
        sections.append(_format_diff_table(current_rows, target_rows))
        sections.append("")
    sections.append("Speed up your model by running `openjet setup`.")

    return "\n".join(sections).rstrip() + "\n"


def _classify_verdict(report: FixReport) -> str:
    obs = report.observation
    if obs.process is None:
        return "no_runtime"
    fit = _gpu_fit(report)
    has_accelerator = any(
        (
            report.hardware.has_cuda,
            report.hardware.has_rocm,
            report.hardware.has_vulkan,
            report.hardware.has_metal,
        )
    )
    if not has_accelerator:
        return "barely_usable"
    if fit == "CPU/RAM":
        return "barely_usable"
    if fit == "Partial":
        return "slow"
    # Full GPU fit. If we have a measured speed close to the MTP-projected
    # target, the user is already running an MTP-style runtime: optimal.
    target = report.recommendation.tok_s.estimated_tokens_per_second
    if obs.decode_tok_s is not None and target > 0:
        if obs.decode_tok_s >= target * 0.7:
            return "optimal"
        return "slow"
    # No measured speed: ollama-default tuning rarely matches MTP llama.cpp;
    # llama.cpp running with full offload + flash-attn is close to optimal.
    if obs.backend == "llama.cpp" and obs.flash_attention is True:
        return "optimal"
    return "slow"


def _gpu_fit(report: FixReport) -> str:
    obs = report.observation
    has_accelerator = any(
        (
            report.hardware.has_cuda,
            report.hardware.has_rocm,
            report.hardware.has_vulkan,
            report.hardware.has_metal,
        )
    )
    if not has_accelerator:
        return "CPU only"
    if obs.backend == "ollama":
        size_mb = _metadata_number(obs, "size_mb")
        vram_mb = _metadata_number(obs, "size_vram_mb")
        if size_mb and vram_mb is not None:
            ratio = vram_mb / max(1.0, size_mb)
            if ratio >= 0.97:
                return "Full"
            if ratio >= 0.4:
                return "Partial"
            return "CPU/RAM"
        processor = str(obs.metadata.get("processor") or "")
        if processor:
            match = re.search(r"(\d+)\s*%\s*GPU", processor, re.IGNORECASE)
            if match:
                pct = int(match.group(1))
                if pct >= 95:
                    return "Full"
                if pct >= 40:
                    return "Partial"
                return "CPU/RAM"
        return "Unknown"
    if obs.backend == "llama.cpp":
        ngl = obs.gpu_layers
        target_ngl = report.recommendation.llama.gpu_layers
        if ngl is None:
            return "Unknown"
        if ngl >= 99 or ngl >= target_ngl:
            return "Full"
        if ngl <= 0:
            return "CPU/RAM"
        return "Partial"
    return "Unknown"


def _current_setup_rows(report: FixReport, verdict: str) -> list[tuple[str, str]]:
    obs = report.observation
    if obs.process is None:
        return [
            ("Backend", "none detected"),
            ("Model", "—"),
            ("Context", "—"),
            ("GPU fit", "—"),
            ("Speed", "—"),
        ]
    backend_label = _backend_display(obs.backend)
    model_label = _current_model_label(report)
    ctx_label = _format_context(obs.context_window_tokens)
    fit_label = _gpu_fit(report)
    speed_label = _format_current_speed(obs)
    return [
        ("Backend", backend_label),
        ("Model", model_label),
        ("Context", ctx_label),
        ("GPU fit", fit_label),
        ("Speed", speed_label),
    ]


def _openjet_target_rows(report: FixReport, verdict: str) -> list[tuple[str, str]]:
    rec = _recommended_values(report)
    ctx_target = rec.get("ctx") or _recommended_fix_context_tokens(
        report.recommendation.llama.context_window_tokens
    )
    target_decode = report.recommendation.tok_s.estimated_tokens_per_second
    speed_label = _format_target_speed(target_decode, report.observation.decode_tok_s)
    fit_label = "Full" if report.recommendation.llama.device != "cpu" else "CPU only"
    model_label = report.recommendation.model.label or "auto"
    if _current_model_matches_recommendation(report):
        model_label = _current_model_label(report)
    return [
        ("Backend", "llama.cpp (auto-tuned)"),
        ("Model", model_label),
        ("Context", _format_context(ctx_target)),
        ("GPU fit", fit_label),
        ("Speed", speed_label),
    ]


def _format_problem(report: FixReport, verdict: str) -> str:
    obs = report.observation
    fit = _gpu_fit(report)
    has_accelerator = any(
        (
            report.hardware.has_cuda,
            report.hardware.has_rocm,
            report.hardware.has_vulkan,
            report.hardware.has_metal,
        )
    )
    lines: list[str] = []
    if not has_accelerator:
        lines.append("  No GPU detected, so generation runs on CPU.")
        lines.append("  Throughput is limited by RAM bandwidth.")
    elif fit == "CPU/RAM":
        lines.append("  The model is too large for your GPU.")
        lines.append("  Most generation is falling back to CPU/RAM.")
    elif fit == "Partial":
        lines.append("  Your context or model is too large for your GPU.")
        lines.append("  Layers spill into normal RAM, so decode slows down.")
    elif obs.backend == "ollama":
        lines.append("  Ollama is running with default tuning.")
        lines.append("  llama.cpp with MTP speculative decoding is faster on this hardware.")
    elif _llama_cpp_mtp_present(obs):
        lines.append("  The current model already appears to be the MTP variant.")
        lines.append("  Measured decode speed is still below OpenJet's target for this hardware.")
    else:
        lines.append("  Runtime flags are below OpenJet's auto-tuned target.")
        lines.append("  MTP speculative decoding is not enabled.")
    return "\n".join(lines)


def _format_kv_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label.ljust(width)}    {value}" for label, value in rows)


def _format_diff_table(
    current: list[tuple[str, str]], target: list[tuple[str, str]]
) -> str:
    by_key = {label: value for label, value in current}
    rows: list[tuple[str, str, str]] = []
    for label, target_value in target:
        current_value = by_key.get(label, "—")
        if current_value == target_value:
            continue
        if current_value == "—":
            continue
        rows.append((label, current_value, target_value))
    if not rows:
        return "  (no material differences)"
    label_w = max(len(r[0]) for r in rows)
    cur_w = max(len(r[1]) for r in rows)
    return "\n".join(
        f"  {label.ljust(label_w)}    {cur.ljust(cur_w)}  →  {tgt}"
        for label, cur, tgt in rows
    )


def _llama_cpp_mtp_present(obs: RuntimeObservation) -> bool:
    return bool(
        obs.backend == "llama.cpp"
        and (
            obs.metadata.get("draft_engaged")
            or obs.metadata.get("model_has_mtp_suffix")
        )
    )


def _current_model_matches_recommendation(report: FixReport) -> bool:
    current = _normalize_model_identity(report.observation.model_path)
    if not current:
        return False
    return current in {
        _normalize_model_identity(report.recommendation.model.filename),
        _normalize_model_identity(report.recommendation.model.label),
    }


def _normalize_model_identity(value: str) -> str:
    text = Path(str(value or "")).name.lower()
    if text.endswith(".gguf"):
        text = text[:-5]
    text = re.sub(r"(?:^|[-_.\s])mtp(?:[-_.\s]|$)", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _backend_display(backend: str) -> str:
    return {
        "llama.cpp": "llama.cpp",
        "ollama": "Ollama",
        "vllm": "vLLM",
        "mlx": "MLX",
    }.get(backend, backend)


def _current_model_label(report: FixReport) -> str:
    obs = report.observation
    if not obs.model_path:
        return "none loaded"
    if obs.backend == "ollama":
        params = str(obs.metadata.get("parameter_size") or "").strip()
        quant = str(obs.metadata.get("quantization_level") or "").strip()
        suffix = " ".join(part for part in (params, quant) if part)
        base = obs.model_path
        return f"{base} ({suffix})" if suffix else base
    return Path(obs.model_path).name or obs.model_path


def _format_context(tokens: int | None) -> str:
    if tokens is None or tokens <= 0:
        return "unknown"
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    if tokens >= 1024:
        return f"{round(tokens / 1024)}k"
    return str(tokens)


def _format_current_speed(obs: RuntimeObservation) -> str:
    if obs.decode_tok_s is None:
        return "not measured"
    return f"{_fmt_number(obs.decode_tok_s)} tok/s"


def _format_target_speed(target_decode: float, current: float | None) -> str:
    if target_decode <= 0:
        return "unknown"
    if current is not None and current > target_decode:
        # Current already faster than projected; show conservative tune-up.
        low = current * 1.05
        high = current * 1.15
    else:
        low = target_decode * 0.85
        high = target_decode * 1.15
    return f"{_fmt_number(low)}-{_fmt_number(high)} tok/s"


def _recommended_values(report: FixReport) -> dict[str, int | None]:
    values: dict[str, int | None] = {"ctx": None, "batch": None, "ubatch": None, "ngl": None}
    args = list(report.recommended_args)
    for idx, arg in enumerate(args):
        if idx + 1 >= len(args):
            continue
        value = args[idx + 1]
        try:
            parsed = int(value)
        except ValueError:
            continue
        if arg == "--ctx-size":
            values["ctx"] = parsed
        elif arg == "--batch-size":
            values["batch"] = parsed
        elif arg == "--ubatch-size":
            values["ubatch"] = parsed
        elif arg == "--n-gpu-layers":
            values["ngl"] = parsed
    return values


def _metadata_number(obs: RuntimeObservation, key: str) -> float | None:
    try:
        value = float(obs.metadata.get(key) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _fmt_number(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    rounded = round(float(value))
    if abs(float(value) - rounded) < 0.05:
        return str(int(rounded))
    return f"{float(value):.1f}"


def _probe_llama_cpp_decode(host: str, port: int, *, timeout: float = 60.0) -> float | None:
    prompt = (
        "OpenJet runtime throughput probe. "
        "Write concise technical notes about local model performance, batching, "
        "GPU memory residency, and decode latency. "
    ) * 8
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": 160,
            "stream": False,
            "cache_prompt": False,
        }
    ).encode("utf-8")
    results: list[float] = []
    for _ in range(2):
        req = Request(
            f"http://{host}:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        timings = payload.get("timings") if isinstance(payload, Mapping) else None
        if not isinstance(timings, Mapping):
            continue
        value = timings.get("predicted_per_second")
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if result > 0:
            results.append(result)
    return max(results) if results else None


def _probe_ollama_decode(host: str, port: int, model_name: str, *, timeout: float = 60.0) -> float | None:
    if not model_name:
        return None
    body = json.dumps(
        {
            "model": model_name,
            "prompt": "Write one short sentence about open source.",
            "stream": False,
            "options": {"num_predict": 32},
        }
    ).encode("utf-8")
    req = Request(
        f"http://{host}:{port}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    eval_count = payload.get("eval_count")
    eval_duration = payload.get("eval_duration")  # nanoseconds
    try:
        count = int(eval_count)
        duration_ns = int(eval_duration)
    except (TypeError, ValueError):
        return None
    if count <= 0 or duration_ns <= 0:
        return None
    return count / (duration_ns / 1_000_000_000.0)

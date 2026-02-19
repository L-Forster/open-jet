"""Runtime helpers for memory and token budgeting on edge devices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import tiktoken
from typing import Any


MIN_TOKEN_BUDGET = 128


@dataclass(frozen=True)
class MemorySnapshot:
    total_mb: float
    available_mb: float
    used_percent: float


@dataclass(frozen=True)
class CpuSample:
    total: int
    idle: int


@dataclass(frozen=True)
class ContextBudget:
    window_tokens: int
    reserve_tokens: int
    prompt_tokens: int


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode_ordinary(text))


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    encoding_name = os.getenv("OPEN_JET_TOKENIZER", "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


def read_memory_snapshot() -> MemorySnapshot | None:
    mem_total_kb: int | None = None
    mem_available_kb: int | None = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])
    except OSError:
        return None

    if not mem_total_kb or mem_available_kb is None:
        return None

    total_mb = mem_total_kb / 1024.0
    available_mb = mem_available_kb / 1024.0
    used_percent = ((mem_total_kb - mem_available_kb) / mem_total_kb) * 100.0
    return MemorySnapshot(total_mb=total_mb, available_mb=available_mb, used_percent=used_percent)


def read_memory_info() -> dict[str, Any]:
    snapshot = read_memory_snapshot()
    if snapshot is None:
        return {
            "mem_total_mb": None,
            "mem_used_mb": None,
            "mem_available_mb": None,
            "mem_used_percent": None,
        }
    used_mb = snapshot.total_mb - snapshot.available_mb
    return {
        "mem_total_mb": round(snapshot.total_mb, 2),
        "mem_used_mb": round(used_mb, 2),
        "mem_available_mb": round(snapshot.available_mb, 2),
        "mem_used_percent": round(snapshot.used_percent, 2),
    }


def read_cpu_percent(
    previous: CpuSample | None,
    *,
    precision: int = 2,
) -> tuple[float | None, CpuSample | None]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline().strip()
    except OSError:
        return None, previous
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None, previous
    try:
        nums = [int(v) for v in parts[1:]]
    except ValueError:
        return None, previous
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    sample = CpuSample(total=total, idle=idle)
    if previous is None:
        return None, sample

    total_delta = sample.total - previous.total
    idle_delta = sample.idle - previous.idle
    if total_delta <= 0:
        return None, sample
    busy = total_delta - idle_delta
    return round((busy / total_delta) * 100.0, precision), sample


def derive_file_token_budget(mem_available_mb: float | None) -> int:
    if mem_available_mb is None:
        return 512
    # Conservative dynamic budget based on currently available RAM.
    return max(MIN_TOKEN_BUDGET, int(mem_available_mb * 4.0))


def derive_context_budget(
    window_tokens: int,
    *,
    reserve_tokens: int | None = None,
    reserve_ratio: float = 0.20,
    min_reserve_tokens: int = 256,
    min_prompt_tokens: int = 256,
) -> ContextBudget:
    window = max(512, int(window_tokens))
    auto_reserve = max(min_reserve_tokens, int(window * reserve_ratio))
    reserve = int(reserve_tokens) if reserve_tokens is not None else auto_reserve
    reserve = max(64, reserve)
    reserve = min(reserve, window - 64)

    prompt_tokens = window - reserve
    if prompt_tokens < min_prompt_tokens:
        prompt_tokens = min(min_prompt_tokens, window - 64)
        reserve = window - prompt_tokens

    return ContextBudget(
        window_tokens=window,
        reserve_tokens=reserve,
        prompt_tokens=prompt_tokens,
    )

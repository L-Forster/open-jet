"""Shared air-gapped mode policy and hard network guard."""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Mapping, MutableMapping
from urllib.parse import urlsplit


class AirgapViolationError(RuntimeError):
    """Raised when air-gapped mode blocks a network operation."""


_STATE_LOCK = threading.RLock()
_AIRGAPPED = False
_PATCH_INSTALLED = False

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo

_LOOPBACK_HOSTS = {"localhost", "localhost.localdomain"}


def airgapped_from_cfg(cfg: Mapping[str, object] | None, *, override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    if not isinstance(cfg, Mapping):
        return False
    return bool(cfg.get("airgapped", False))


def is_airgapped() -> bool:
    with _STATE_LOCK:
        return _AIRGAPPED


def set_airgapped(enabled: bool) -> None:
    global _AIRGAPPED
    _install_network_guard()
    with _STATE_LOCK:
        _AIRGAPPED = bool(enabled)


def apply_airgap_env(env: MutableMapping[str, str], *, enabled: bool) -> MutableMapping[str, str]:
    if not enabled:
        return env
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def assert_endpoint_allowed(target: str | None, *, label: str) -> None:
    if not is_airgapped():
        return
    host = _extract_host(target)
    if host and _is_loopback_host(host):
        return
    detail = target.strip() if isinstance(target, str) else "<unspecified>"
    raise AirgapViolationError(f"Air-gapped mode blocks {label}: {detail}")


def _install_network_guard() -> None:
    global _PATCH_INSTALLED
    with _STATE_LOCK:
        if _PATCH_INSTALLED:
            return
        socket.socket.connect = _guarded_connect
        socket.socket.connect_ex = _guarded_connect_ex
        socket.create_connection = _guarded_create_connection
        socket.getaddrinfo = _guarded_getaddrinfo
        _PATCH_INSTALLED = True


def _guarded_connect(sock: socket.socket, address) -> object:
    _ensure_address_allowed(address)
    return _ORIGINAL_CONNECT(sock, address)


def _guarded_connect_ex(sock: socket.socket, address) -> object:
    _ensure_address_allowed(address)
    return _ORIGINAL_CONNECT_EX(sock, address)


def _guarded_create_connection(address, *args, **kwargs) -> socket.socket:
    _ensure_address_allowed(address)
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if is_airgapped():
        normalized = _normalize_host(host)
        if normalized and not _is_loopback_host(normalized):
            raise AirgapViolationError(
                f"Air-gapped mode blocks network name resolution: {normalized}:{port}"
            )
    return _ORIGINAL_GETADDRINFO(host, port, *args, **kwargs)


def _ensure_address_allowed(address) -> None:
    if not is_airgapped():
        return
    host = _extract_host(address)
    if host and _is_loopback_host(host):
        return
    raise AirgapViolationError(
        f"Air-gapped mode blocks network access to {_format_address(address)}"
    )


def _extract_host(target) -> str | None:
    if isinstance(target, tuple) and target:
        return _normalize_host(target[0])
    if isinstance(target, bytes):
        return _normalize_host(target.decode("utf-8", errors="ignore"))
    if not isinstance(target, str):
        return None
    text = target.strip()
    if not text:
        return None
    if "://" in text:
        parsed = urlsplit(text)
        return _normalize_host(parsed.hostname)
    if text.startswith("[") and "]" in text:
        return _normalize_host(text[1:text.index("]")])
    if "/" in text and not text.startswith("/"):
        return None
    host = text.split(":", 1)[0]
    return _normalize_host(host)


def _normalize_host(host) -> str | None:
    if host is None:
        return None
    if isinstance(host, bytes):
        host = host.decode("utf-8", errors="ignore")
    normalized = str(host).strip()
    if not normalized:
        return None
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return normalized.lower()


def _is_loopback_host(host: str) -> bool:
    normalized = _normalize_host(host)
    if not normalized:
        return False
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _format_address(address) -> str:
    if isinstance(address, tuple) and address:
        host = address[0]
        port = address[1] if len(address) > 1 else "?"
        return f"{host}:{port}"
    return str(address)

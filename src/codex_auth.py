from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .airgap import assert_endpoint_allowed


CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_AUTH_PROVIDER = "openai-codex"


class CodexAuthError(RuntimeError):
    """Raised when Codex OAuth authentication is unavailable or invalid."""


@dataclass(frozen=True)
class CodexCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    id_token: str = ""
    account_id: str = ""
    email: str = ""
    plan_type: str = ""

    @classmethod
    def from_auth_json(cls, payload: dict[str, Any]) -> "CodexCredentials":
        if str(payload.get("auth_mode") or "").strip().lower() not in {"chatgpt", "chat_gpt"}:
            raise CodexAuthError("Codex is not logged in with ChatGPT OAuth.")
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            raise CodexAuthError("Codex auth.json does not contain ChatGPT OAuth tokens.")
        access_token = str(tokens.get("access_token") or "").strip()
        refresh_token = str(tokens.get("refresh_token") or "").strip()
        id_token = _raw_id_token(tokens.get("id_token"))
        if not access_token or not refresh_token:
            raise CodexAuthError("Codex OAuth tokens are incomplete.")
        claims = _jwt_claims(id_token) or _jwt_claims(access_token) or {}
        auth_claims = _auth_claims(claims)
        expires_at = _jwt_expires_at(access_token)
        if expires_at <= 0:
            expires_at = _coerce_expires_at(tokens.get("expires_at") or payload.get("expires_at"))
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            id_token=id_token,
            account_id=str(
                tokens.get("account_id")
                or auth_claims.get("chatgpt_account_id")
                or claims.get("chatgpt_account_id")
                or _first_org_id(claims)
                or ""
            ).strip(),
            email=str(claims.get("email") or _profile_claims(claims).get("email") or "").strip(),
            plan_type=str(auth_claims.get("chatgpt_plan_type") or "").strip(),
        )

    @classmethod
    def from_refresh_response(
        cls,
        payload: dict[str, Any],
        *,
        previous: "CodexCredentials",
    ) -> "CodexCredentials":
        access_token = str(payload.get("access_token") or "").strip()
        refresh_token = str(payload.get("refresh_token") or previous.refresh_token).strip()
        id_token = str(payload.get("id_token") or previous.id_token).strip()
        if not access_token or not refresh_token:
            raise CodexAuthError("Codex OAuth refresh response did not include usable tokens.")
        claims = _jwt_claims(id_token) or _jwt_claims(access_token) or {}
        auth_claims = _auth_claims(claims)
        expires_at = _jwt_expires_at(access_token)
        if expires_at <= 0:
            expires_at = _expires_at_from_expires_in(payload.get("expires_in"))
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            id_token=id_token,
            account_id=str(
                payload.get("account_id")
                or previous.account_id
                or auth_claims.get("chatgpt_account_id")
                or _first_org_id(claims)
                or ""
            ).strip(),
            email=str(claims.get("email") or _profile_claims(claims).get("email") or previous.email or "").strip(),
            plan_type=str(auth_claims.get("chatgpt_plan_type") or previous.plan_type or "").strip(),
        )

    def is_expired(self, *, skew_seconds: int = 120) -> bool:
        return time.time() >= self.expires_at - skew_seconds


class CodexAuthStore:
    """Reads and updates the official Codex CLI ChatGPT OAuth auth file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or default_codex_auth_path()).expanduser()

    def load(self) -> CodexCredentials | None:
        payload = self._load_payload()
        if payload is None:
            return None
        try:
            return CodexCredentials.from_auth_json(payload)
        except CodexAuthError:
            return None

    def save(self, credentials: CodexCredentials) -> None:
        payload = self._load_payload() or {}
        payload["auth_mode"] = "chatgpt"
        payload["tokens"] = {
            "id_token": credentials.id_token,
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "account_id": credentials.account_id or None,
            "expires_at": credentials.expires_at,
        }
        payload["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _write_json_private(self.path, payload)

    def clear(self) -> None:
        codex = _codex_binary()
        if codex:
            try:
                result = subprocess.run([codex, "logout"], check=False, timeout=60)
                if result.returncode == 0:
                    return
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def status(self) -> dict[str, object]:
        credentials = self.load()
        if credentials is None:
            return {"logged_in": False, "storage": str(self.path)}
        return {
            "logged_in": True,
            "expires_at": credentials.expires_at,
            "expired": credentials.is_expired(),
            "account_id": credentials.account_id,
            "email": credentials.email,
            "plan_type": credentials.plan_type,
            "storage": str(self.path),
        }

    def _load_payload(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


class CodexOAuthProvider:
    def __init__(
        self,
        *,
        store: CodexAuthStore | None = None,
        http: httpx.AsyncClient | None = None,
        client_id: str = CODEX_CLIENT_ID,
        token_url: str = CODEX_TOKEN_URL,
    ) -> None:
        self.store = store or CodexAuthStore()
        self.client_id = client_id
        self.token_url = token_url
        self._http = http

    async def credentials(self) -> CodexCredentials:
        credentials = self.store.load()
        if credentials is None:
            raise CodexAuthError("Not logged in to Codex OAuth. Run /connect openai-codex first.")
        if credentials.is_expired():
            credentials = await self.refresh(credentials)
        return credentials

    async def access_token(self) -> str:
        return (await self.credentials()).access_token

    async def refresh(self, credentials: CodexCredentials) -> CodexCredentials:
        assert_endpoint_allowed(self.token_url, label="Codex OAuth token refresh")
        async with self._client() as http:
            response = await http.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": credentials.refresh_token,
                    "client_id": self.client_id,
                },
                headers={"Accept": "application/json"},
            )
        if response.status_code >= 400:
            raise CodexAuthError("Codex OAuth refresh failed. Run /connect openai-codex again.")
        refreshed = CodexCredentials.from_refresh_response(response.json(), previous=credentials)
        self.store.save(refreshed)
        return refreshed

    async def login_browser(self, *, device_auth: bool = False) -> CodexCredentials:
        if not _codex_binary():
            raise CodexAuthError("Official Codex CLI is not installed. Install `@openai/codex` first.")
        await asyncio.to_thread(_run_codex_login, device_auth=device_auth)
        credentials = self.store.load()
        if credentials is None:
            raise CodexAuthError("Codex login completed but no ChatGPT OAuth session was found.")
        return credentials

    def logout(self) -> None:
        self.store.clear()

    def status(self) -> dict[str, object]:
        return self.store.status()

    def _client(self):
        if self._http is not None:
            return _BorrowedAsyncClient(self._http)
        return httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0))


class _BorrowedAsyncClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def default_codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    return Path(codex_home).expanduser() / "auth.json" if codex_home else Path.home() / ".codex" / "auth.json"


def _codex_binary() -> str | None:
    return shutil.which("codex")


def _run_codex_login(*, device_auth: bool = False) -> None:
    codex = _codex_binary()
    if not codex:
        raise CodexAuthError("Official Codex CLI is not installed. Install `@openai/codex` first.")
    command = [codex, "login"]
    if device_auth:
        command.append("--device-auth")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise CodexAuthError("Codex OAuth login failed.")


def _raw_id_token(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("raw_jwt") or "").strip()
    return ""


def _jwt_expires_at(token: str) -> float:
    claims = _jwt_claims(token) or {}
    try:
        return float(claims.get("exp") or 0)
    except (TypeError, ValueError):
        return 0.0


def _expires_at_from_expires_in(value: object) -> float:
    try:
        seconds = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return time.time() + seconds if seconds > 0 else 0.0


def _coerce_expires_at(value: object) -> float:
    try:
        expires_at = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return expires_at if expires_at > 0 else 0.0


def _jwt_claims(token: str) -> dict[str, Any] | None:
    parts = str(token or "").split(".")
    if len(parts) != 3 or not parts[1]:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return claims if isinstance(claims, dict) else None


def _auth_claims(claims: dict[str, Any]) -> dict[str, Any]:
    value = claims.get("https://api.openai.com/auth")
    return value if isinstance(value, dict) else {}


def _profile_claims(claims: dict[str, Any]) -> dict[str, Any]:
    value = claims.get("https://api.openai.com/profile")
    return value if isinstance(value, dict) else {}


def _first_org_id(claims: dict[str, Any]) -> str:
    orgs = claims.get("organizations")
    if isinstance(orgs, list) and orgs and isinstance(orgs[0], dict):
        return str(orgs[0].get("id") or "").strip()
    auth_orgs = _auth_claims(claims).get("organizations")
    if isinstance(auth_orgs, list) and auth_orgs and isinstance(auth_orgs[0], dict):
        return str(auth_orgs[0].get("id") or "").strip()
    return ""


def _write_json_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    tmp_path = Path(tmp_name)
    try:
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.write("\n")
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

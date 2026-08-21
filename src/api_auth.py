from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai-compatible": "",
}


def normalize_provider_id(provider: str) -> str:
    return str(provider or "").strip().lower().replace("_", "-")


def default_api_key_env(provider: str) -> str:
    return DEFAULT_API_KEY_ENV.get(normalize_provider_id(provider), "")


def pi_auth_json_path() -> Path:
    env_dir = str(os.environ.get("PI_CODING_AGENT_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser() / "auth.json"
    return Path.home() / ".pi" / "agent" / "auth.json"


def key_from_pi_credential(row: object) -> str | None:
    if not isinstance(row, dict):
        return None
    cred_type = str(row.get("type") or "").strip()
    if cred_type == "api_key":
        value = str(row.get("key") or "").strip()
        return value or None
    if cred_type == "oauth":
        value = str(row.get("access") or "").strip()
        return value or None
    return None


class ApiKeyStore:
    """API keys from env, OS keyring, or credentials Pi wrote to auth.json.

    OpenJet never stores plaintext keys in auth.json itself; that file is
    owned by Pi's login flows. Our own saves go to the OS keyring only.
    Logout may remove this provider's entry from auth.json, but never adds
    or modifies credential contents there.
    """

    def __init__(self, *, service_name: str = "openjet.api-keys") -> None:
        self.service_name = service_name

    def load_key(self, provider: str) -> str | None:
        provider_id = normalize_provider_id(provider)
        if not provider_id:
            return None
        return self._load_keyring(provider_id) or self._load_pi_auth(provider_id)

    def save_key(self, provider: str, api_key: str) -> None:
        provider_id = normalize_provider_id(provider)
        value = str(api_key or "").strip()
        if not provider_id or not value:
            raise ValueError("provider and API key are required")
        if self._save_keyring(provider_id, value):
            return
        raise ValueError("OS keyring is unavailable. Set the provider API key environment variable instead.")

    def clear_key(self, provider: str) -> bool:
        provider_id = normalize_provider_id(provider)
        if not provider_id:
            return True
        keyring_ok = self._clear_keyring(provider_id)
        pi_ok = self._clear_pi_auth(provider_id)
        return keyring_ok and pi_ok

    def providers(self) -> list[str]:
        return []

    def status(self, providers: list[str] | tuple[str, ...]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for provider in providers:
            provider_id = normalize_provider_id(provider)
            env_name = default_api_key_env(provider_id)
            env_present = bool(env_name and os.environ.get(env_name))
            stored_present = bool(self.load_key(provider_id))
            result[provider_id] = {
                "env": env_name,
                "env_present": env_present,
                "stored": stored_present,
                "storage": "env" if env_present else self._storage_label(provider_id),
            }
        return result

    def resolve_key(self, provider: str, *, env_name: str = "") -> str | None:
        provider_id = normalize_provider_id(provider)
        candidates = [str(env_name or "").strip(), default_api_key_env(provider_id)]
        for candidate in candidates:
            if candidate and os.environ.get(candidate):
                return str(os.environ[candidate]).strip()
        return self.load_key(provider_id)

    def _storage_label(self, provider_id: str) -> str:
        if self._load_keyring(provider_id):
            return "keyring"
        if self._load_pi_auth(provider_id):
            return "pi-auth"
        return "missing"

    def _load_pi_auth(self, provider_id: str) -> str | None:
        path = pi_auth_json_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return key_from_pi_credential(data.get(provider_id))

    def _clear_pi_auth(self, provider_id: str) -> bool:
        path = pi_auth_json_path()
        if not path.is_file():
            return True
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or provider_id not in data:
                return True
            data.pop(provider_id, None)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            path.chmod(0o600)
            return True
        except Exception:
            return False

    def _load_keyring(self, provider_id: str) -> str | None:
        try:
            import keyring  # type: ignore

            value = keyring.get_password(self.service_name, provider_id)
            return str(value).strip() if value else None
        except Exception:
            return None

    def _save_keyring(self, provider_id: str, value: str) -> bool:
        try:
            import keyring  # type: ignore

            keyring.set_password(self.service_name, provider_id, value)
            return True
        except Exception:
            return False

    def _clear_keyring(self, provider_id: str) -> bool:
        try:
            import keyring  # type: ignore

            if not keyring.get_password(self.service_name, provider_id):
                return True
            keyring.delete_password(self.service_name, provider_id)
            return True
        except Exception:
            return False

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
LOCAL_BIN_DIR="${HOME}/.local/bin"
SYSTEM_BIN_DIR="/usr/local/bin"
INSTALL_MODE="${OPENJET_INSTALL_MODE:-install}"
UPDATE_REINSTALL="${OPENJET_UPDATE_REINSTALL:-0}"
INSTALL_STATE_FILE="${VENV_DIR}/.openjet-install-hash"
BUILD_SETUPTOOLS_SPEC="setuptools>=68,<80"

link_launchers() {
  local target_dir="$1"
  if [ ! -d "${target_dir}" ] || [ ! -w "${target_dir}" ]; then
    return 1
  fi
  ln -sf "${VENV_DIR}/bin/open-jet" "${target_dir}/open-jet"
  ln -sf "${VENV_DIR}/bin/openjet" "${target_dir}/openjet"
}

install_fingerprint() {
  "${VENV_DIR}/bin/python" - <<'PY'
import hashlib
from pathlib import Path

root = Path.cwd()
digest = hashlib.sha256()
for relative_path in ("pyproject.toml", "setup.py"):
    path = root / relative_path
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    if path.is_file():
        digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

path_has_dir() {
  local target_dir="$1"
  case ":${PATH}:" in
    *":${target_dir}:"*) return 0 ;;
    *) return 1 ;;
  esac
}

clean_inplace_extensions() {
  local built_exts=()
  local artifact
  while IFS= read -r artifact; do
    built_exts+=("${artifact}")
  done < <(find "${ROOT_DIR}/src" -maxdepth 1 \( -name '*.so' -o -name '*.pyd' \) -type f | sort)

  if ((${#built_exts[@]} == 0)); then
    return 0
  fi

  echo "Removing stale in-place extension artifacts so the editable Python sources stay active:"
  for artifact in "${built_exts[@]}"; do
    echo "  - ${artifact#${ROOT_DIR}/}"
    rm -f "${artifact}"
  done
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "error: OpenJet requires Python 3.10 or newer; ${PYTHON_BIN} is too old" >&2
  echo "set PYTHON to a newer interpreter, for example: PYTHON=python3.11 ./install.sh" >&2
  exit 1
fi

create_virtualenv() {
  if "${PYTHON_BIN}" -c "import ensurepip" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    return 0
  fi

  echo "stdlib venv support is unavailable; falling back to virtualenv" >&2
  "${PYTHON_BIN}" -m pip install --user virtualenv
  "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
}

build_backend_ready() {
  "${VENV_DIR}/bin/python" - <<'PY'
import importlib.util
import re
import sys

try:
    import setuptools
except Exception:
    sys.exit(1)

def version_tuple(value):
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))

setuptools_version = version_tuple(getattr(setuptools, "__version__", ""))
has_supported_setuptools = (68,) <= setuptools_version < (80,)
has_wheel = importlib.util.find_spec("wheel") is not None
sys.exit(0 if has_supported_setuptools and has_wheel else 1)
PY
}

ensure_build_backend() {
  if build_backend_ready; then
    return 0
  fi

  echo "Installing Python build requirements for editable metadata"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip "${BUILD_SETUPTOOLS_SPEC}" wheel
}

clean_inplace_extensions
if [ ! -f "${VENV_DIR}/bin/python" ]; then
  create_virtualenv
  "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
ensure_build_backend

CURRENT_INSTALL_HASH="$(install_fingerprint)"

RUN_PIP_INSTALL=1
INSTALL_REASON="fresh install mode"
RUN_PROVISION=1

if [ -x "${VENV_DIR}/bin/openjet" ]; then
  RUN_PIP_INSTALL=0
  INSTALL_REASON="reusing existing editable install"
  if [ -f "${INSTALL_STATE_FILE}" ]; then
    if [ "$(tr -d '\n' < "${INSTALL_STATE_FILE}")" != "${CURRENT_INSTALL_HASH}" ]; then
      RUN_PIP_INSTALL=1
      INSTALL_REASON="install metadata changed"
    fi
  else
    RUN_PIP_INSTALL=1
    INSTALL_REASON="install state missing"
  fi
fi

if [ "${INSTALL_MODE}" = "update" ]; then
  if [ "${UPDATE_REINSTALL}" = "1" ]; then
    RUN_PIP_INSTALL=1
    INSTALL_REASON="update changed install requirements"
  fi
  RUN_PROVISION=0
fi

if [ "${RUN_PIP_INSTALL}" -eq 0 ]; then
  RUN_PROVISION=0
fi

if [ "${RUN_PIP_INSTALL}" -eq 1 ]; then
  echo "Refreshing editable install: ${INSTALL_REASON}"
  OPENJET_BUILD_EXTENSIONS=0 "${VENV_DIR}/bin/python" -m pip install --no-build-isolation -e "${ROOT_DIR}"
  printf '%s\n' "${CURRENT_INSTALL_HASH}" > "${INSTALL_STATE_FILE}"
else
  echo "Skipping pip install: ${INSTALL_REASON}"
fi

"${VENV_DIR}/bin/python" -m pip install --quiet --disable-pip-version-check hf_transfer 'huggingface_hub>=0.25' || \
  echo "warning: could not install hf_transfer; downloads will fall back to single-stream"

if [ "${RUN_PROVISION}" -eq 1 ] && "${VENV_DIR}/bin/python" - <<'PY'
from src.observation.processors import provision_default_faster_whisper_model
raise SystemExit(0 if provision_default_faster_whisper_model() else 1)
PY
then
  echo "Provisioned bundled local transcription assets"
elif [ "${RUN_PROVISION}" -eq 0 ]; then
  echo "Skipping transcription asset provisioning"
else
  echo "warning: could not pre-provision local transcription assets; OpenJet will retry on first microphone use"
fi

INSTALLED_LAUNCHER_DIR=""
if [ -d "${SYSTEM_BIN_DIR}" ] && [ -w "${SYSTEM_BIN_DIR}" ] && path_has_dir "${SYSTEM_BIN_DIR}" && link_launchers "${SYSTEM_BIN_DIR}"; then
  INSTALLED_LAUNCHER_DIR="${SYSTEM_BIN_DIR}"
elif mkdir -p "${LOCAL_BIN_DIR}" 2>/dev/null && path_has_dir "${LOCAL_BIN_DIR}" && link_launchers "${LOCAL_BIN_DIR}"; then
  INSTALLED_LAUNCHER_DIR="${LOCAL_BIN_DIR}"
elif mkdir -p "${LOCAL_BIN_DIR}" 2>/dev/null && link_launchers "${LOCAL_BIN_DIR}"; then
  INSTALLED_LAUNCHER_DIR="${LOCAL_BIN_DIR}"
elif link_launchers "${SYSTEM_BIN_DIR}"; then
  INSTALLED_LAUNCHER_DIR="${SYSTEM_BIN_DIR}"
fi

echo "Installed open-jet from ${ROOT_DIR}"
echo "Install mode: editable Python source"
if [ -n "${INSTALLED_LAUNCHER_DIR}" ]; then
  echo "Launch with: openjet --setup"
else
  echo "warning: could not install launchers into ${LOCAL_BIN_DIR} or ${SYSTEM_BIN_DIR}"
  echo "run with: ${VENV_DIR}/bin/openjet --setup"
fi

if [ -n "${INSTALLED_LAUNCHER_DIR}" ]; then
  case ":${PATH}:" in
    *":${INSTALLED_LAUNCHER_DIR}:"*) ;;
    *)
      echo "warning: ${INSTALLED_LAUNCHER_DIR} is not on PATH"
      echo "run with: ${INSTALLED_LAUNCHER_DIR}/openjet --setup"
      ;;
  esac
fi

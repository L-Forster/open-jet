#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
LOCAL_BIN_DIR="${HOME}/.local/bin"
SYSTEM_BIN_DIR="/usr/local/bin"
INSTALL_MODE="${OPENJET_INSTALL_MODE:-install}"
UPDATE_REINSTALL="${OPENJET_UPDATE_REINSTALL:-0}"

link_launchers() {
  local target_dir="$1"
  if [ ! -d "${target_dir}" ] || [ ! -w "${target_dir}" ]; then
    return 1
  fi
  ln -sf "${VENV_DIR}/bin/open-jet" "${target_dir}/open-jet"
  ln -sf "${VENV_DIR}/bin/openjet" "${target_dir}/openjet"
}

clean_inplace_extensions() {
  mapfile -t built_exts < <(find "${ROOT_DIR}/src" -maxdepth 1 \( -name '*.so' -o -name '*.pyd' \) -type f | sort)
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

create_virtualenv() {
  if "${PYTHON_BIN}" -c "import ensurepip" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    return 0
  fi

  echo "stdlib venv support is unavailable; falling back to virtualenv" >&2
  "${PYTHON_BIN}" -m pip install --user virtualenv
  "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
}

clean_inplace_extensions
if [ ! -f "${VENV_DIR}/bin/python" ]; then
  create_virtualenv
  "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
fi

RUN_PIP_INSTALL=1
INSTALL_REASON="fresh install mode"

if [ "${INSTALL_MODE}" = "update" ]; then
  INSTALL_REASON="editable package missing from virtualenv"
  if [ -x "${VENV_DIR}/bin/openjet" ]; then
    RUN_PIP_INSTALL=0
    INSTALL_REASON="reusing existing editable install"
  fi
  if [ "${UPDATE_REINSTALL}" = "1" ]; then
    RUN_PIP_INSTALL=1
    INSTALL_REASON="update changed install requirements"
  fi
fi

if [ "${RUN_PIP_INSTALL}" -eq 1 ]; then
  echo "Refreshing editable install: ${INSTALL_REASON}"
  OPENJET_BUILD_EXTENSIONS=0 "${VENV_DIR}/bin/python" -m pip install --no-build-isolation -e "${ROOT_DIR}"
else
  echo "Skipping pip install: ${INSTALL_REASON}"
fi

if [ "${INSTALL_MODE}" != "update" ] && "${VENV_DIR}/bin/python" - <<'PY'
from src.observation.processors import provision_default_faster_whisper_model
raise SystemExit(0 if provision_default_faster_whisper_model() else 1)
PY
then
  echo "Provisioned bundled local transcription assets"
elif [ "${INSTALL_MODE}" = "update" ]; then
  echo "Skipping transcription asset provisioning during update"
else
  echo "warning: could not pre-provision local transcription assets; OpenJet will retry on first microphone use"
fi

INSTALLED_LAUNCHER_DIR=""
if mkdir -p "${LOCAL_BIN_DIR}" 2>/dev/null && link_launchers "${LOCAL_BIN_DIR}"; then
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

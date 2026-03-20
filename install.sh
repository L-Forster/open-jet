#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
LOCAL_BIN_DIR="${HOME}/.local/bin"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}"

mkdir -p "${LOCAL_BIN_DIR}"
ln -sf "${VENV_DIR}/bin/open-jet" "${LOCAL_BIN_DIR}/open-jet"
ln -sf "${VENV_DIR}/bin/openjet" "${LOCAL_BIN_DIR}/openjet"

echo "Installed open-jet from ${ROOT_DIR}"
echo "Launch with: open-jet --setup"

case ":${PATH}:" in
  *":${LOCAL_BIN_DIR}:"*) ;;
  *)
    echo "warning: ${LOCAL_BIN_DIR} is not on PATH"
    echo "run with: ${LOCAL_BIN_DIR}/open-jet --setup"
    ;;
esac

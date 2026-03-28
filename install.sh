#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
LOCAL_BIN_DIR="${HOME}/.local/bin"

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
  rm -rf "${VENV_DIR}"

  if "${PYTHON_BIN}" -c "import ensurepip" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    return 0
  fi

  echo "stdlib venv support is unavailable; falling back to virtualenv" >&2
  "${PYTHON_BIN}" -m pip install --user virtualenv
  "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
}

clean_inplace_extensions
create_virtualenv
"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
OPENJET_BUILD_EXTENSIONS=0 "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}"

if "${VENV_DIR}/bin/python" - <<'PY'
from src.observation.processors import provision_default_faster_whisper_model
raise SystemExit(0 if provision_default_faster_whisper_model() else 1)
PY
then
  echo "Provisioned bundled local transcription assets"
else
  echo "warning: could not pre-provision local transcription assets; OpenJet will retry on first microphone use"
fi

mkdir -p "${LOCAL_BIN_DIR}"
ln -sf "${VENV_DIR}/bin/open-jet" "${LOCAL_BIN_DIR}/open-jet"
ln -sf "${VENV_DIR}/bin/openjet" "${LOCAL_BIN_DIR}/openjet"

echo "Installed open-jet from ${ROOT_DIR}"
echo "Install mode: editable Python source"
echo "Launch with: open-jet --setup"

case ":${PATH}:" in
  *":${LOCAL_BIN_DIR}:"*) ;;
  *)
    echo "warning: ${LOCAL_BIN_DIR} is not on PATH"
    echo "run with: ${LOCAL_BIN_DIR}/open-jet --setup"
    ;;
esac

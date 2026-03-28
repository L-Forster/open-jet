#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DIST_DIR="${ROOT_DIR}/dist"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("Cython", "wheel") if importlib.util.find_spec(name) is None]
if missing:
    names = ", ".join(missing)
    raise SystemExit(
        "error: missing build dependency/dependencies: "
        f"{names}. Install them in the current environment, then retry."
    )
PY

rm -rf "${ROOT_DIR}/build"
mkdir -p "${DIST_DIR}"

echo "Building compiled wheel into ${DIST_DIR}"
OPENJET_BUILD_EXTENSIONS=1 "${PYTHON_BIN}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${DIST_DIR}" \
  "${ROOT_DIR}"

echo "Built compiled wheel(s):"
find "${DIST_DIR}" -maxdepth 1 -type f -name '*.whl' -printf '  - %f\n' | sort

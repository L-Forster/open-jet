#!/usr/bin/env bash
set -euo pipefail

# Build and upload compiled wheel distributions.
# Usage:
#   scripts/release_pypi.sh testpypi
#   scripts/release_pypi.sh pypi

TARGET="${1:-testpypi}"
if [[ "$TARGET" != "testpypi" && "$TARGET" != "pypi" ]]; then
  echo "Usage: $0 [testpypi|pypi]" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required" >&2
  exit 1
fi

if ! command -v auditwheel >/dev/null 2>&1; then
  echo "auditwheel is required. Install with: python -m pip install auditwheel" >&2
  exit 1
fi

python scripts/sync_pypi_readme.py

python -m pip install --upgrade "build" "twine" "Cython>=3.0" "setuptools>=68,<80" "wheel"

rm -rf build dist ./*.egg-info ./*/*.egg-info

# stdlib distutils removed in Python 3.12+; let setuptools provide it.
unset SETUPTOOLS_USE_DISTUTILS 2>/dev/null || true
python -m build --wheel --no-isolation
RAW_WHEEL="$(find dist -maxdepth 1 -name '*.whl' -printf '%T@ %p\n' | sort -rn | head -n1 | cut -d' ' -f2)"
if [[ -z "${RAW_WHEEL}" ]]; then
  echo "No wheel produced in dist/." >&2
  exit 1
fi

mkdir -p dist/repaired
if auditwheel show "${RAW_WHEEL}" >/dev/null 2>&1; then
  auditwheel repair "${RAW_WHEEL}" -w dist/repaired
else
  cp "${RAW_WHEEL}" dist/repaired/
fi

python -m twine check dist/repaired/*

if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "Set TWINE_PASSWORD to your PyPI API token first." >&2
  exit 1
fi

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

if [[ "$TARGET" == "testpypi" ]]; then
  python -m twine upload --repository testpypi dist/repaired/*
  echo "Uploaded to TestPyPI."
else
  python -m twine upload dist/repaired/*
  echo "Uploaded to PyPI."
fi

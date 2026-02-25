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

python -m pip install --upgrade "build" "twine" "Cython>=3.0" "setuptools>=68,<80" "wheel"

rm -rf build dist ./*.egg-info ./*/*.egg-info

# Required in some Debian/Ubuntu Python setups where numpy/distutils is injected.
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"
python -m build --wheel --no-isolation
RAW_WHEEL="$(ls -t dist/open_jet-*.whl | head -n1)"
if [[ -z "${RAW_WHEEL}" ]]; then
  echo "No wheel produced in dist/." >&2
  exit 1
fi

mkdir -p dist/repaired
auditwheel repair "${RAW_WHEEL}" -w dist/repaired

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

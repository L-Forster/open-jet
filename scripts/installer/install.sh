#!/bin/sh
# Installs OpenJet without a system Python or pipx (Linux and macOS):
#
#   curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
#
# uv downloads its own Python for OpenJet, separate from the system's. OPENJET_VERSION pins a
# release (for example OPENJET_VERSION=0.5.4).
set -eu

PYTHON_VERSION=3.12
BIN_DIR="${XDG_BIN_HOME:-${HOME}/.local/bin}"  # where uv puts itself and openjet

if command -v uv >/dev/null 2>&1; then
  UV=uv
elif [ -x "${BIN_DIR}/uv" ]; then
  UV="${BIN_DIR}/uv"
else
  echo "Installing uv, which provides OpenJet's Python..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="${BIN_DIR}/uv"
fi

SPEC=open-jet
if [ -n "${OPENJET_VERSION:-}" ]; then
  SPEC="open-jet==${OPENJET_VERSION}"
fi

echo "Installing ${SPEC} with Python ${PYTHON_VERSION}..."
"${UV}" tool install --force --python "${PYTHON_VERSION}" "${SPEC}"
"${UV}" tool update-shell >/dev/null 2>&1 || true

echo
echo "OpenJet is installed. Open a new terminal, then run:"
echo
echo "  openjet setup"
echo
echo "If openjet is not found, run it as ${BIN_DIR}/openjet or add ${BIN_DIR} to your PATH."

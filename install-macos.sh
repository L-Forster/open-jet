#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [ -n "${SCRIPT_SOURCE}" ]; then
  ROOT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
else
  ROOT_DIR="$(pwd -P)"
fi

if [ "$(uname -s)" != "Darwin" ]; then
  echo "warning: install-macos.sh is intended for macOS; continuing with install.sh" >&2
fi

exec "${ROOT_DIR}/install.sh" "$@"

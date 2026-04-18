#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "warning: install-macos.sh is intended for macOS; continuing with install.sh" >&2
fi

exec "${ROOT_DIR}/install.sh" "$@"

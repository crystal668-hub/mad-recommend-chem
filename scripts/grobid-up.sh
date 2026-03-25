#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v docker >/dev/null 2>&1; then
  exec docker compose -f compose.yaml up -d
fi

if command -v powershell.exe >/dev/null 2>&1; then
  exec powershell.exe -NoProfile -Command "docker compose -f compose.yaml up -d"
fi

echo "Neither docker nor powershell.exe is available; cannot start GROBID with Docker Compose." >&2
exit 1

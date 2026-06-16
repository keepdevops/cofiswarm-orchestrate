#!/usr/bin/env bash
set -euo pipefail
ROLE="${1:?usage: assert-layout.sh <role>}"
ROOT="$(cd "$(dirname "$0")/../standalone" && pwd)"
for dir in \
  "${ROOT}/opt/cofiswarm/${ROLE}" \
  "${ROOT}/etc/cofiswarm/${ROLE}" \
  "${ROOT}/var/lib/cofiswarm/${ROLE}" \
  "${ROOT}/var/log/cofiswarm/${ROLE}" \
  "${ROOT}/run/cofiswarm"; do
  [[ -d "$dir" ]] || { echo "missing: $dir"; exit 1; }
done
echo "ok: standalone layout for ${ROLE}"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
python3 -c "from cofiswarm_orchestrate.modes.map_reduce import MapReduceMode; print('ok:', MapReduceMode.__name__)"
PORT="${ORCH_TEST_PORT:-18003}"
export COFISWARM_CONFIG_ROOT="${COFISWARM_CONFIG_ROOT:-$HOME/cofiswarm/fhs/etc/cofiswarm/config}"
export ORCH_SIDECAR_HOST=127.0.0.1 ORCH_SIDECAR_PORT="$PORT"
python3 "${ROOT}/scripts/run-sidecar.py" &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do
  lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
  sleep 0.25
done
lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null
echo "ok: orchestrate sidecar"

#!/usr/bin/env bash
# Regression guard: RAG context is injected exactly once on the MLX orchestrate path
# (modes inject params["rag_context"] via rag_xml(); service_orchestrate must NOT also
# prepend its own block). Skips cleanly if the backend-sdk import isn't resolvable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if ! python3 -c "import backends.base" >/dev/null 2>&1; then
  echo "skip: backends.base not importable — rag inject-once guard"
  exit 0
fi

python3 "${ROOT}/test/scripts/rag_inject_once_check.py"

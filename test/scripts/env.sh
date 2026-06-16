#!/usr/bin/env bash
# Source from integration tests and init-standalone.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../standalone" && pwd)"
export COFISWARM_TEST_ROOT="$ROOT"
export COFISWARM_OPT_ROOT="${ROOT}/opt/cofiswarm"
export COFISWARM_ETC_ROOT="${ROOT}/etc/cofiswarm"
export COFISWARM_VAR_LIB="${ROOT}/var/lib/cofiswarm"
export COFISWARM_VAR_LOG="${ROOT}/var/log/cofiswarm"
export COFISWARM_RUN_ROOT="${ROOT}/run/cofiswarm"

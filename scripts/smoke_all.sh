#!/usr/bin/env bash
# End-to-end smoke for the EchoMap stack. Verifies — in one go — that:
#   1. The release `echomap_server` binary builds and binds a port.
#   2. The plugin loader runs and the CLI lists registered groups.
#   3. The hardware bridge can drive a MockArm through the agent loop.
#   4. A live heuristic boxing match connects two agents and reaches
#      "match_end" naturally (1 short round, no LLM, no GUI).
#
# Each phase must exit 0. Server is torn down on success, failure, or
# interrupt. Use this as the pre-ship sanity check; the per-subsystem
# tests (cargo test, pytest) cover correctness in detail.
#
# Usage:
#   bash scripts/smoke_all.sh                 # full smoke
#   SMOKE_PORT=9200 bash scripts/smoke_all.sh # override port
#
# Exit code: 0 on success, non-zero on first failing phase.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${REPO_ROOT}/target/release/echomap_server"
SMOKE_PORT="${SMOKE_PORT:-9117}"
ROUND_DURATION="${SMOKE_ROUND_DURATION:-8}"  # short so heuristic match ends ~8s
LOGDIR="$(mktemp -d -t echomap-smoke-XXXXXX)"
SERVER_PID=""

cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "==> logs left in ${LOGDIR}"
}
trap cleanup EXIT INT TERM

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

# ---- Phase 1: build server binary -----------------------------------------
echo "==> [1/4] Building release server binary"
if [ ! -x "${SERVER_BIN}" ]; then
    cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" --bin echomap_server
fi
[ -x "${SERVER_BIN}" ] || { echo "FAIL: server binary missing at ${SERVER_BIN}"; exit 1; }

# ---- Phase 2: plugin loader -----------------------------------------------
echo "==> [2/4] CLI list-plugins"
python3 -m echomap_client.cli list-plugins >"${LOGDIR}/plugins.log" 2>&1 || {
    echo "FAIL: cli list-plugins"; cat "${LOGDIR}/plugins.log"; exit 1
}
tail -3 "${LOGDIR}/plugins.log"

# ---- Phase 3: hardware bridge ---------------------------------------------
echo "==> [3/4] Hardware bridge — MockArm"
python3 "${REPO_ROOT}/demos/connect_real_arm.py" --backend mock --steps 20 \
    >"${LOGDIR}/hardware.log" 2>&1 || {
    echo "FAIL: connect_real_arm.py --backend mock"; cat "${LOGDIR}/hardware.log"; exit 1
}
tail -3 "${LOGDIR}/hardware.log"

# ---- Phase 4: live heuristic boxing match ---------------------------------
echo "==> [4/4] Live boxing — heuristic on port ${SMOKE_PORT}"
WS_PORT="${SMOKE_PORT}" \
TCP_PORT="$((SMOKE_PORT - 1))" \
ROUND_DURATION="${ROUND_DURATION}" \
NUM_ROUNDS=1 \
    "${SERVER_BIN}" >"${LOGDIR}/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server to bind (15s ceiling).
for _ in $(seq 1 75); do
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.1); \
        sys.exit(0 if s.connect_ex(('127.0.0.1', ${SMOKE_PORT}))==0 else 1)" 2>/dev/null; then
        break
    fi
    sleep 0.2
done

if ! python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.1); \
    sys.exit(0 if s.connect_ex(('127.0.0.1', ${SMOKE_PORT}))==0 else 1)" 2>/dev/null; then
    echo "FAIL: server did not bind on ${SMOKE_PORT}"
    cat "${LOGDIR}/server.log"
    exit 1
fi

# Hard ceiling to avoid hangs even if the match never reaches match_end.
# Round duration + countdowns + buffer < 30s for 1 round of 8s.
if ! timeout 45 python3 "${REPO_ROOT}/demos/connect_boxing_agents.py" \
    --mode heuristic --port "${SMOKE_PORT}" \
    >"${LOGDIR}/boxing.log" 2>&1; then
    echo "FAIL: heuristic boxing match"
    tail -30 "${LOGDIR}/boxing.log"
    exit 1
fi
tail -5 "${LOGDIR}/boxing.log"

# Sanity: scoreboard line must be present.
if ! grep -q "Final Score:" "${LOGDIR}/boxing.log"; then
    echo "FAIL: no Final Score line in boxing output"
    tail -30 "${LOGDIR}/boxing.log"
    exit 1
fi

echo "==> smoke OK"
exit 0

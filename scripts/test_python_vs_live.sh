#!/usr/bin/env bash
# Live-server Python integration gate (goal/008 deliverable 6).
#
# Builds the release echomap_server binary, spawns a single long-lived
# instance on port 9002, runs the full Python test suite against it, and
# tears the server back down on exit (success or failure). Exit code is
# pytest's.
#
# Usage:
#   bash scripts/test_python_vs_live.sh
#   bash scripts/test_python_vs_live.sh -k bind   # forwarded to pytest
#
# Env overrides:
#   ECHOMAP_LIVE_PORT  — port for the long-lived server (default 9002)
#   ECHOMAP_LIVE_HOST  — bind host (default 127.0.0.1)
#   ECHOMAP_LIVE_TIMEOUT — seconds to wait for the server to bind (default 30)
#
# Tests in python/tests/ that prefer to spawn their own short-lived server
# (test_agent_bind, test_agent_platform_e2e) honor the WS_PORT env var, so
# pointing them at our long-lived instance via WS_PORT keeps the surface
# uniform across the suite.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${REPO_ROOT}/target/release/echomap_server"
PORT="${ECHOMAP_LIVE_PORT:-9002}"
HOST="${ECHOMAP_LIVE_HOST:-127.0.0.1}"
TIMEOUT="${ECHOMAP_LIVE_TIMEOUT:-30}"

log() { printf '==> %s\n' "$*"; }

log "Building release echomap_server"
cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" --bin echomap_server >/dev/null

if [ ! -x "$SERVER_BIN" ]; then
    echo "FATAL: build succeeded but $SERVER_BIN is missing" >&2
    exit 2
fi

LOG_FILE="$(mktemp -t echomap_live_server.XXXXXX.log)"
log "Spawning echomap_server on ${HOST}:${PORT} (log: ${LOG_FILE})"

# Subshell so we can cleanly trap and kill on exit.
WS_PORT="$PORT" ROUND_DURATION=30 NUM_ROUNDS=1 \
    "$SERVER_BIN" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
    local rc=$?
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (pid=${SERVER_PID})"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        # Wait briefly then SIGKILL if needed.
        for _ in 1 2 3 4 5; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$rc" -ne 0 ]; then
        echo "---- last 60 lines of server log ----" >&2
        tail -n 60 "$LOG_FILE" >&2 || true
    fi
    rm -f "$LOG_FILE"
    exit "$rc"
}
trap cleanup EXIT INT TERM

log "Waiting for port ${PORT} to accept connections (timeout ${TIMEOUT}s)"
deadline=$(( $(date +%s) + TIMEOUT ))
until python3 - "$HOST" "$PORT" <<'PY' 2>/dev/null
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
s.connect((host, port))
s.close()
PY
do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FATAL: server died before binding port ${PORT}" >&2
        exit 3
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "FATAL: server did not bind port ${PORT} within ${TIMEOUT}s" >&2
        exit 4
    fi
    sleep 0.25
done
log "Server is accepting connections"

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export WS_PORT="$PORT"

log "Running pytest python/ -v"
cd "$REPO_ROOT"
python3 -m pytest python/ -v "$@"

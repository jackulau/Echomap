#!/usr/bin/env bash
# Live-server Python integration gate (goal/008 deliverable 6).
#
# Two-phase gate:
#   1. Build the release `echomap_server` binary and prove it binds the
#      goal's reference port (WS:9002). Smoke-tests connect + observe +
#      stop so a broken build fails fast before pytest starts.
#   2. Run the full Python suite (`pytest python/ -v`). The live-server
#      test classes (test_agent_bind, test_agent_platform_e2e,
#      test_bridge_parity) spawn their own short-lived servers on
#      per-class ports — we explicitly UNSET WS_PORT/TCP_PORT so those
#      spawns get clean defaults instead of inheriting ours.
#
# Exit code mirrors pytest's, modulo build/smoke failures (which return
# distinct non-zero codes).
#
# Usage:
#   bash scripts/test_python_vs_live.sh
#   bash scripts/test_python_vs_live.sh -k bind   # forwarded to pytest
#
# Env overrides:
#   ECHOMAP_LIVE_PORT  — WS port for the smoke phase (default 9002)
#   ECHOMAP_LIVE_HOST  — bind host (default 127.0.0.1)
#   ECHOMAP_LIVE_TIMEOUT — seconds to wait for smoke server to bind (default 30)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${REPO_ROOT}/target/release/echomap_server"
PORT="${ECHOMAP_LIVE_PORT:-9002}"
HOST="${ECHOMAP_LIVE_HOST:-127.0.0.1}"
TIMEOUT="${ECHOMAP_LIVE_TIMEOUT:-30}"

log() { printf '==> %s\n' "$*"; }

log "Phase 1/2 — build release echomap_server"
cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" --bin echomap_server >/dev/null

if [ ! -x "$SERVER_BIN" ]; then
    echo "FATAL: build succeeded but $SERVER_BIN is missing" >&2
    exit 2
fi

log "Phase 1/2 — smoke-spawning server on ${HOST}:${PORT}"
LOG_FILE="$(mktemp -t echomap_live_server.XXXXXX.log)"
WS_PORT="$PORT" TCP_PORT="$((PORT - 1))" ROUND_DURATION=30 NUM_ROUNDS=1 \
    "$SERVER_BIN" >"$LOG_FILE" 2>&1 &
SMOKE_PID=$!

cleanup_smoke() {
    if kill -0 "$SMOKE_PID" 2>/dev/null; then
        kill -TERM "$SMOKE_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$SMOKE_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$SMOKE_PID" 2>/dev/null || true
        wait "$SMOKE_PID" 2>/dev/null || true
    fi
}
trap 'rc=$?; cleanup_smoke; rm -f "$LOG_FILE"; exit $rc' EXIT INT TERM

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
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then
        echo "FATAL: server died before binding port ${PORT}" >&2
        tail -n 60 "$LOG_FILE" >&2 || true
        exit 3
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "FATAL: server did not bind port ${PORT} within ${TIMEOUT}s" >&2
        exit 4
    fi
    sleep 0.25
done

log "Phase 1/2 — server bound port ${PORT}; tearing down smoke instance"
cleanup_smoke
rm -f "$LOG_FILE"
trap - EXIT INT TERM

log "Phase 2/2 — running pytest python/ -v"
# Unset port env so each live test class gets to use its own port without
# inheriting our smoke-phase values. PYTHONPATH wires up the in-tree
# echomap_client without requiring `pip install -e`.
unset WS_PORT TCP_PORT
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
cd "$REPO_ROOT"
python3 -m pytest python/ -v "$@"

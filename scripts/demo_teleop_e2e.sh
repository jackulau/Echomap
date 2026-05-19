#!/usr/bin/env bash
# End-to-end tele-op demo (goal/011 D7).
#
# Reuses the headless server harness from D1: builds echomap_server +
# teleop_e2e_demo, spawns the server on TCP:19001 / WS:19002, runs the demo
# which records 100 sinusoidal steps then immediately replays the trace
# via Player::replay. Exits 0 only on full success.

set -euo pipefail

TCP_PORT="${TCP_PORT:-19001}"
WS_PORT="${WS_PORT:-19002}"
WAIT_SECS="${WAIT_SECS:-10}"
STEPS="${STEPS:-100}"
TOLERANCE="${TOLERANCE:-1e-2}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRACE_DIR="$REPO_ROOT/tasks/011-robot-render-teleop-readiness"
TRACE_PATH="$TRACE_DIR/teleop_trace.jsonl"
mkdir -p "$TRACE_DIR"

SERVER_PID=""
SERVER_LOG="$(mktemp -t echomap_teleop_demo.XXXXXX.log)"

cleanup() {
    local rc=$?
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in 1 2 3 4; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
            sleep 0.5
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ $rc -ne 0 ]]; then
        echo "---- server log (tail) ----" >&2
        tail -50 "$SERVER_LOG" >&2 || true
    fi
    rm -f "$SERVER_LOG"
    exit $rc
}
trap cleanup EXIT INT TERM

echo "[demo] cargo build --release --bin echomap_server --example teleop_e2e_demo"
cargo build --release --bin echomap_server --example teleop_e2e_demo

echo "[demo] starting headless server on TCP=$TCP_PORT WS=$WS_PORT"
TCP_PORT="$TCP_PORT" WS_PORT="$WS_PORT" \
    ./target/release/echomap_server >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$(( $(date +%s) + WAIT_SECS ))
ready=0
while [[ $(date +%s) -lt $deadline ]]; do
    if nc -z 127.0.0.1 "$TCP_PORT" 2>/dev/null && \
       nc -z 127.0.0.1 "$WS_PORT" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[demo] server exited before becoming ready" >&2
        exit 1
    fi
    sleep 0.2
done
if [[ $ready -ne 1 ]]; then
    echo "[demo] server did not bind within ${WAIT_SECS}s" >&2
    exit 1
fi
echo "[demo] server ready"

echo "[demo] running teleop_e2e_demo (steps=$STEPS, tolerance=$TOLERANCE)"
./target/release/examples/teleop_e2e_demo \
    --addr "ws://127.0.0.1:$WS_PORT" \
    --robot-id 0 \
    --steps "$STEPS" \
    --tolerance "$TOLERANCE" \
    --record "$TRACE_PATH"

if [[ ! -s "$TRACE_PATH" ]]; then
    echo "[demo] trace artifact missing or empty: $TRACE_PATH" >&2
    exit 1
fi

echo "[demo] OK — trace artifact at $TRACE_PATH"

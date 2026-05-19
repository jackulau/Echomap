#!/usr/bin/env bash
# Headless-server end-to-end smoke (goal/011 D1).
#
# Builds echomap_server, spawns it on TCP_PORT=19001 / WS_PORT=19002, waits
# for the listener, then runs the `--ignored` integration test
# `tests/teleop_e2e.rs` which drives 100 sinusoidal steps and asserts the
# step_count is monotonic. Server is torn down on EXIT.
#
# Exits 0 only on full success. Any failure (build, server crash, client
# assertion) propagates as non-zero.

set -euo pipefail

TCP_PORT="${TCP_PORT:-19001}"
WS_PORT="${WS_PORT:-19002}"
WAIT_SECS="${WAIT_SECS:-10}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SERVER_PID=""
SERVER_LOG="$(mktemp -t echomap_smoke.XXXXXX.log)"

cleanup() {
    local rc=$?
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        # Give it 2s to flush, then force-kill.
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

echo "[smoke] cargo build --release --bin echomap_server"
cargo build --release --bin echomap_server

echo "[smoke] starting headless server on TCP=$TCP_PORT WS=$WS_PORT"
TCP_PORT="$TCP_PORT" WS_PORT="$WS_PORT" \
    ./target/release/echomap_server >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for both listeners.
deadline=$(( $(date +%s) + WAIT_SECS ))
ready=0
while [[ $(date +%s) -lt $deadline ]]; do
    if nc -z 127.0.0.1 "$TCP_PORT" 2>/dev/null && \
       nc -z 127.0.0.1 "$WS_PORT" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[smoke] server exited before becoming ready" >&2
        exit 1
    fi
    sleep 0.2
done
if [[ $ready -ne 1 ]]; then
    echo "[smoke] server did not bind within ${WAIT_SECS}s" >&2
    exit 1
fi
echo "[smoke] server ready"

echo "[smoke] running 100-step client (tests/teleop_e2e.rs)"
RUST_E2E_PORT="$WS_PORT" \
    cargo test --release --test teleop_e2e \
    -- --ignored --nocapture --test-threads=1

echo "[smoke] OK"

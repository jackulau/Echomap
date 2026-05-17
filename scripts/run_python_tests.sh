#!/usr/bin/env bash
# Run the full Python test suite against a live release headless server.
#
# Test files in python/tests/ either:
#   - run pure-Python (test_hardware, test_plugins, test_client*, test_runner,
#     test_agents, test_commentary), or
#   - manage their own short-lived `echomap_server` subprocess on a per-class
#     port (test_agent_bind, test_agent_platform_e2e).
#
# This script therefore just ensures the release server binary is built
# (the live-server tests skip themselves otherwise) and invokes pytest.
#
# Usage:
#   bash scripts/run_python_tests.sh             # full suite
#   bash scripts/run_python_tests.sh -k bind     # filtered
#
# Exit code mirrors pytest's.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${REPO_ROOT}/target/release/echomap_server"

echo "==> Ensuring release server binary is built ($SERVER_BIN)"
if [ ! -x "$SERVER_BIN" ]; then
    cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" --bin echomap_server
fi

# Make in-tree echomap_client importable without `pip install -e`.
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

echo "==> Running pytest"
cd "$REPO_ROOT"
exec python3 -m pytest python/tests/ -q "$@"

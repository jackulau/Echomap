#!/usr/bin/env bash
# Hot-path unwrap budget (goal/013 D6).
#
# Counts production-only `.unwrap()` / `panic!` / `.expect(` occurrences in
# the engine's hot paths — anything that runs every frame, every WS recv,
# every protocol parse. Test-block unwraps inside `#[cfg(test)] mod tests`
# are deliberately excluded (tests SHOULD panic on a failed assertion).
#
# Baseline (recorded 2026-05-23, after D5):
#   src/main.rs                  0
#   src/teleop/recorder.rs       0
#   src/agent/bridge.rs          0
#   src/agent/ws_server.rs       0
#   src/agent/tcp_server.rs      0   (was 1 before D6; fixed in D6)
#   src/agent/session.rs         0
#   src/renderer/*.rs            0 (mod.rs has 1, in a guarded helper)
#
# Total prod baseline: 1
# Budget: 5 — leaves headroom for genuinely irreducible cases that come
# with a `// SAFETY:` comment justifying why panicking is the correct
# behaviour at that site (e.g. invariant violation that means the program
# state is already corrupt).
#
# This script must exit 0 — CI / `cargo test` runs it via the goal/013
# verification gate.

set -euo pipefail

BUDGET=5

declare -a FILES=(
  "src/main.rs"
  "src/teleop/recorder.rs"
  "src/agent/bridge.rs"
  "src/agent/ws_server.rs"
  "src/agent/tcp_server.rs"
  "src/agent/session.rs"
  "src/renderer/mod.rs"
  "src/renderer/legend.rs"
  "src/renderer/listener_viz.rs"
  "src/renderer/perf_governor.rs"
  "src/renderer/ray_debug.rs"
  "src/renderer/surface_heatmap.rs"
  "src/renderer/bounds.rs"
)

# Count production unwraps in one file — lines above the first
# `#[cfg(test)]` marker, with `// SAFETY:` lines deliberately ignored
# so callers can document irreducible cases without inflating the count.
prod_count() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo 0
    return
  fi
  local cutoff
  cutoff=$(grep -n "^#\[cfg(test)\]" "$f" | head -1 | cut -d: -f1 || true)
  if [[ -z "$cutoff" ]]; then
    cutoff=$(wc -l < "$f")
  fi
  head -n "$cutoff" "$f" \
    | grep -E '\.unwrap\(\)|panic!|\.expect\(' \
    | grep -v 'SAFETY:' \
    | grep -cv '^\s*//' \
    || true
}

total=0
echo "hot-path unwrap audit (budget: $BUDGET)"
echo "----------------------------------------"
for f in "${FILES[@]}"; do
  n=$(prod_count "$f")
  total=$((total + n))
  printf '  %-40s %d\n' "$f" "$n"
done
echo "----------------------------------------"
echo "  total prod hot-path unwraps:        $total"

if (( total > BUDGET )); then
  echo "FAIL: hot-path unwrap count $total exceeds budget $BUDGET" >&2
  exit 1
fi

echo "OK"

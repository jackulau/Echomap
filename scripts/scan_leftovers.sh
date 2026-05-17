#!/usr/bin/env bash
# Scan production code (not tests, not CLI/main) for leftover issues that a
# release-grade application should not ship with:
#
#   1. `todo!(` / `unimplemented!(` anywhere in src/
#   2. `.unwrap()` in solver hot paths (acoustics, fluids, gas, surface,
#      robot collision/dynamics/kinematics) — excluding code inside any
#      `mod tests { ... }` block. Tests are allowed to unwrap.
#   3. `println!` outside CLI/main/bin entry points and outside test modules.
#      Solver/runtime code should use `log::*` macros, not raw println.
#   4. Top-level `print(` in python/echomap_client/, excluding cli.py
#      (legitimate CLI output) and runner.py (live match commentary that
#      users explicitly enable).
#
# Exit 0 if every scan returns zero hits; exit 1 on the first hit. Optional
# `--report <path>` writes a markdown report with the exact scan commands
# and result.
#
# Tests are detected by the awk filter `intest=1` once a `#[cfg(test)]` or
# `mod tests` line is seen — this matches the convention used across the
# repo (test modules always appear at the bottom of a file).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT=""
if [[ "${1:-}" == "--report" ]]; then
    REPORT="$2"
    mkdir -p "$(dirname "$REPORT")"
    : >"$REPORT"
fi

cd "$REPO_ROOT"

emit() {
    if [[ -n "$REPORT" ]]; then
        echo "$@" >>"$REPORT"
    fi
    echo "$@"
}

fail=0

prod_only() {
    # Strip lines that fall inside a `#[cfg(test)]` block or below an
    # explicit `mod tests {` for each input file.
    awk '
        FNR == 1 { intest = 0 }
        /^#\[cfg\(test\)\]/ { intest = 1 }
        /^mod tests[[:space:]]/ { intest = 1 }
        { if (!intest) print FILENAME ":" FNR ":" $0 }
    ' "$@"
}

# ---- scan 1: todo!/unimplemented! anywhere in src/ ------------------------
emit "## Scan 1 — todo!/unimplemented! in src/"
emit ""
emit "    grep -RnE 'todo!\\(|unimplemented!\\(' src/"
emit ""
hits1="$(grep -RnE 'todo!\(|unimplemented!\(' src/ 2>/dev/null || true)"
if [[ -n "$hits1" ]]; then
    emit '```'
    emit "$hits1"
    emit '```'
    emit "**FAIL** — $(printf '%s\n' "$hits1" | wc -l | tr -d ' ') hit(s)."
    fail=1
else
    emit "_zero hits._"
fi
emit ""

# ---- scan 2: .unwrap() in solver hot paths (production only) --------------
emit "## Scan 2 — .unwrap() in solver hot paths (test-mode excluded)"
emit ""
emit "    prod_only src/acoustics/ src/fluids/ src/gas/ src/surface/ \\"
emit "              src/robot/collision.rs src/robot/dynamics.rs src/robot/kinematics.rs \\"
emit "        | grep -E '\\.unwrap\\(\\)'"
emit ""
hot_files=()
while IFS= read -r f; do
    hot_files+=("$f")
done < <(find src/acoustics src/fluids src/gas src/surface -type f -name '*.rs' \
         ; echo src/robot/collision.rs ; echo src/robot/dynamics.rs ; echo src/robot/kinematics.rs)
hits2="$(prod_only "${hot_files[@]}" | grep -E '\.unwrap\(\)' || true)"
if [[ -n "$hits2" ]]; then
    emit '```'
    emit "$hits2"
    emit '```'
    emit "**FAIL** — $(printf '%s\n' "$hits2" | wc -l | tr -d ' ') hit(s)."
    fail=1
else
    emit "_zero hits in production code (test-mode unwraps are allowed)._"
fi
emit ""

# ---- scan 3: println! outside CLI/main/bin and outside tests --------------
emit "## Scan 3 — println! outside CLI/main/bin/tests"
emit ""
emit "    prod_only \$(non-cli rust files) | grep -E 'println!'"
emit ""
nonbin=()
while IFS= read -r f; do
    nonbin+=("$f")
done < <(find src -type f -name '*.rs' \
         ! -path 'src/main.rs' \
         ! -path 'src/bin/*' \
         | sort)
hits3="$(prod_only "${nonbin[@]}" | grep -E 'println!' || true)"
if [[ -n "$hits3" ]]; then
    emit '```'
    emit "$hits3"
    emit '```'
    emit "**FAIL** — $(printf '%s\n' "$hits3" | wc -l | tr -d ' ') hit(s)."
    fail=1
else
    emit "_zero hits._"
fi
emit ""

# ---- scan 4: print( in python/echomap_client/ outside cli.py + runner.py --
emit "## Scan 4 — top-level print( in python/echomap_client/ (cli.py + runner.py exempt)"
emit ""
emit "    grep -RnE '^[[:space:]]*print\\(' python/echomap_client/ \\"
emit "        | grep -vE '/(cli|runner)\\.py:'"
emit ""
hits4="$(grep -RnE '^[[:space:]]*print\(' python/echomap_client/ 2>/dev/null \
         | grep -vE '/(cli|runner)\.py:' || true)"
if [[ -n "$hits4" ]]; then
    emit '```'
    emit "$hits4"
    emit '```'
    emit "**FAIL** — $(printf '%s\n' "$hits4" | wc -l | tr -d ' ') hit(s)."
    fail=1
else
    emit "_zero hits (cli.py + runner.py are intentionally allowed)._"
fi
emit ""

if [[ $fail -eq 0 ]]; then
    emit "## Summary"
    emit ""
    emit "All four scans returned zero hits. Production code is free of"
    emit "leftover \`todo!\`/\`unimplemented!\`, solver-hot-path unwraps,"
    emit "stray \`println!\`, and unexpected Python prints."
    exit 0
else
    emit "## Summary"
    emit ""
    emit "One or more scans surfaced production-code hits. See sections above."
    exit 1
fi

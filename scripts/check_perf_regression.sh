#!/usr/bin/env bash
# Compare current `cargo bench --bench physics` results against the baselines
# recorded in benches/baselines.md. Fail (exit 1) if any bench is more than
# REGRESSION_THRESHOLD percent slower than its baseline.
#
# Usage:
#   bash scripts/check_perf_regression.sh                # full run
#   bash scripts/check_perf_regression.sh --dry-run      # parse baselines only
#
# Output: one line per bench: "<name>  baseline=<X> current=<Y> delta=<Z>%  PASS|FAIL"
# Exit code: 0 if every bench within tolerance, 1 if any regressed beyond it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES="${REPO_ROOT}/benches/baselines.md"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-15}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

if [[ ! -f "${BASELINES}" ]]; then
    echo "error: missing ${BASELINES}" >&2
    exit 1
fi

# Convert a duration like "4.10 ms" / "135 µs" / "1.12 ns" to seconds.
to_seconds() {
    local value="$1"
    local unit="$2"
    case "${unit}" in
        s)   echo "${value}" ;;
        ms)  awk "BEGIN{printf \"%.12g\", ${value}/1000}" ;;
        us|µs) awk "BEGIN{printf \"%.12g\", ${value}/1000000}" ;;
        ns)  awk "BEGIN{printf \"%.12g\", ${value}/1000000000}" ;;
        *)   echo "error: unknown unit '${unit}'" >&2; return 1 ;;
    esac
}

# Parse the baselines table into "bench_name|seconds" pairs.
parse_baselines() {
    awk -F '|' '
        /^\| [a-z]/ {
            # Strip leading/trailing whitespace from each cell.
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            # $4 looks like "4.10 ms" or "135 µs".
            if ($2 ~ /\//) {
                print $2 "|" $4
            }
        }' "${BASELINES}"
}

BASELINE_ROWS=()
while IFS= read -r line; do
    BASELINE_ROWS+=("${line}")
done < <(parse_baselines)

if [[ ${#BASELINE_ROWS[@]} -eq 0 ]]; then
    echo "error: no baselines parsed from ${BASELINES}" >&2
    exit 1
fi

echo "loaded ${#BASELINE_ROWS[@]} baseline entries from ${BASELINES}"
for row in "${BASELINE_ROWS[@]}"; do
    bench="${row%%|*}"
    duration="${row##*|}"
    value="${duration%% *}"
    unit="${duration##* }"
    seconds="$(to_seconds "${value}" "${unit}")"
    echo "  baseline: ${bench} = ${duration} (${seconds} s)"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "dry-run OK — baselines file parses cleanly"
    exit 0
fi

# Run benches.
echo "running cargo bench --bench physics --quick ..."
BENCH_OUT="$(mktemp)"
trap 'rm -f "${BENCH_OUT}"' EXIT

if ! (cd "${REPO_ROOT}" && cargo bench --bench physics -- --quick) >"${BENCH_OUT}" 2>&1; then
    cat "${BENCH_OUT}" >&2
    echo "error: cargo bench failed" >&2
    exit 1
fi

# Parse Criterion bench output. Each bench produces either:
#
#   <group>/<name>      time:   [low med high unit]
# or
#   <group>/<name>
#                       time:   [low med high unit]
#
# followed later by "change: time: ..." lines which we MUST ignore. Strategy:
# track the most recent line that looks like a bench name; on the first
# matching "time: [...]" we see after it, emit "<name>|<med> <unit>", then
# clear the name so subsequent change: lines do not re-match.
parse_current() {
    awk '
        # A line whose first whitespace-separated token is "<group>/<bench>"
        # registers the current bench name.
        $1 ~ /^[a-z][a-z_]*\/[a-z][a-z0-9_]*$/ {
            current_name = $1
        }
        # The actual measurement line: "time:   [low med high unit]" (no
        # leading "change:"). Skip lines that contain "change:" which carry
        # delta percentages, not absolute durations.
        /time:/ && !/change:/ && current_name != "" {
            for (i = 1; i <= NF; i++) {
                if (index($i, "[") > 0) {
                    gsub(/\[/, "", $i)
                    mid_val = $(i + 2)
                    mid_unit = $(i + 3)
                    gsub(/\]/, "", mid_unit)
                    print current_name "|" mid_val " " mid_unit
                    current_name = ""
                    break
                }
            }
        }
    ' "${BENCH_OUT}"
}

CURRENT_FILE="$(mktemp)"
trap 'rm -f "${BENCH_OUT}" "${CURRENT_FILE}"' EXIT
parse_current >"${CURRENT_FILE}"

lookup_current() {
    local name="$1"
    awk -F '|' -v target="${name}" '$1==target { print $2; exit }' "${CURRENT_FILE}"
}

fail=0
current_count="$(wc -l <"${CURRENT_FILE}" | tr -d ' ')"
echo "comparing ${current_count} current benches against baselines (threshold=${REGRESSION_THRESHOLD}%):"
for row in "${BASELINE_ROWS[@]}"; do
    bench="${row%%|*}"
    duration="${row##*|}"
    value="${duration%% *}"
    unit="${duration##* }"
    base_s="$(to_seconds "${value}" "${unit}")"
    cur_duration="$(lookup_current "${bench}")"
    if [[ -z "${cur_duration}" ]]; then
        echo "  ${bench}  baseline=${duration} current=MISSING  SKIP"
        continue
    fi
    cur_value="${cur_duration%% *}"
    cur_unit="${cur_duration##* }"
    cur_s="$(to_seconds "${cur_value}" "${cur_unit}")"
    delta_pct="$(awk "BEGIN{printf \"%.1f\", (${cur_s}-${base_s})/${base_s}*100}")"
    status="PASS"
    over="$(awk "BEGIN{print (${delta_pct} > ${REGRESSION_THRESHOLD}) ? 1 : 0}")"
    if [[ "${over}" -eq 1 ]]; then
        status="FAIL"
        fail=1
    fi
    echo "  ${bench}  baseline=${duration} current=${cur_duration} delta=${delta_pct}%  ${status}"
done

if [[ "${fail}" -ne 0 ]]; then
    echo "regression threshold exceeded" >&2
    exit 1
fi

echo "all benches within ${REGRESSION_THRESHOLD}% of baseline"
exit 0

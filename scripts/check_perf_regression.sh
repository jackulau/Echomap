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

# Parse the baselines table into "bench_name|seconds" pairs. Only rows whose
# baseline cell is a plain "<number> <unit>" median are comparable here;
# threshold/qualitative rows (e.g. "≤50 ms", "record†", "≥5x speedup") are
# validated by their own bench harness (benches/acoustics.rs), not this gate.
parse_baselines() {
    awk -F '|' '
        /^\| [a-z]/ {
            # Strip leading/trailing whitespace from each cell.
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            # $4 looks like "4.10 ms" or "135 µs".
            if ($2 ~ /\// && $4 ~ /^[0-9]+(\.[0-9]+)?[[:space:]](s|ms|us|µs|ns)$/) {
                print $2 "|" $4
            }
        }' "${BASELINES}"
}

# Threshold/qualitative rows skipped by parse_baselines, for transparency.
parse_skipped() {
    awk -F '|' '
        /^\| [a-z]/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            if ($2 ~ /\// && $4 !~ /^[0-9]+(\.[0-9]+)?[[:space:]](s|ms|us|µs|ns)$/) {
                print $2 "|" $4
            }
        }' "${BASELINES}"
}

BASELINE_ROWS=()
while IFS= read -r line; do
    BASELINE_ROWS+=("${line}")
done < <(parse_baselines)

SKIPPED_ROWS=()
while IFS= read -r line; do
    SKIPPED_ROWS+=("${line}")
done < <(parse_skipped)

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

# Run benches BENCH_RUNS times and take the per-bench MINIMUM. A single
# `--quick` run swings ±25% with machine noise (thermal state, background
# load), which makes a 15% threshold flaky. Noise only ever ADDS time, so
# the min across runs is the robust estimator of true cost.
BENCH_RUNS="${BENCH_RUNS:-3}"
BENCH_OUT="$(mktemp)"
trap 'rm -f "${BENCH_OUT}"' EXIT

for run in $(seq 1 "${BENCH_RUNS}"); do
    echo "running cargo bench --bench physics --quick (run ${run}/${BENCH_RUNS}) ..."
    RUN_OUT="$(mktemp)"
    if ! (cd "${REPO_ROOT}" && cargo bench --bench physics -- --quick) >"${RUN_OUT}" 2>&1; then
        cat "${RUN_OUT}" >&2
        rm -f "${RUN_OUT}"
        echo "error: cargo bench failed" >&2
        exit 1
    fi
    cat "${RUN_OUT}" >>"${BENCH_OUT}"
    rm -f "${RUN_OUT}"
done

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

# All measurements for a bench (one line per run).
lookup_all() {
    local name="$1"
    awk -F '|' -v target="${name}" '$1==target { print $2 }' "${CURRENT_FILE}"
}

# Minimum across runs, echoed as "<value> <unit>|<seconds>".
lookup_min() {
    local name="$1"
    local best_s="" best_human=""
    while IFS= read -r cand; do
        [[ -z "${cand}" ]] && continue
        local v="${cand%% *}" u="${cand##* }" s
        s="$(to_seconds "${v}" "${u}")" || continue
        if [[ -z "${best_s}" ]] || awk "BEGIN{exit !(${s} < ${best_s})}"; then
            best_s="${s}"
            best_human="${cand}"
        fi
    done < <(lookup_all "${name}")
    [[ -n "${best_s}" ]] && echo "${best_human}|${best_s}"
}

fail=0
current_count="$(wc -l <"${CURRENT_FILE}" | tr -d ' ')"
echo "comparing ${current_count} measurements (min of ${BENCH_RUNS} runs per bench) against baselines (threshold=${REGRESSION_THRESHOLD}%):"
for row in "${BASELINE_ROWS[@]}"; do
    bench="${row%%|*}"
    duration="${row##*|}"
    value="${duration%% *}"
    unit="${duration##* }"
    base_s="$(to_seconds "${value}" "${unit}")"
    cur_min="$(lookup_min "${bench}")"
    if [[ -z "${cur_min}" ]]; then
        echo "  ${bench}  baseline=${duration} current=MISSING  SKIP"
        continue
    fi
    cur_duration="${cur_min%%|*}"
    cur_s="${cur_min##*|}"
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

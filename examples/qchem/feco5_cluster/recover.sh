#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
records=("$ROOT"/results*/submitted.tsv)
[[ -f "${records[0]}" ]] || { echo 'No campaign submission records found.' >&2; exit 1; }
recorded=$(awk -F '\t' 'FNR>1 {if ($4 !~ /^[0-9]+$/) exit 1; print $4}' "${records[@]}" | sort -u)
# Query current jobs once; expired IDs in older records need not exist in Slurm.
live=$(squeue --noheader --user="$(id -un)" --format='%F' | sort -u)
to_cancel=$(comm -12 <(printf '%s\n' "$recorded" | sort -u) <(printf '%s\n' "$live" | sort -u))
if [[ -n "$to_cancel" ]]; then
    ids=()
    while IFS= read -r job; do ids+=("$job"); done <<< "$to_cancel"
    scancel "${ids[@]}"
fi
export OUTPUT_ROOT="$ROOT/results_retry_$(date +%Y%m%d_%H%M%S)"
printf 'New output directory: %s\nWaiting for the build/smoke job before submitting chemistry jobs.\n' "$OUTPUT_ROOT"
bash "$ROOT/submit.sh" verify-and-submit

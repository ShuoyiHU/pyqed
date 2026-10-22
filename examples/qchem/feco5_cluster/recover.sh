#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
record="$ROOT/results/submitted.tsv"
[[ -f "$record" ]] || { echo "Missing original campaign submission record: $record" >&2; exit 1; }
mapfile -t ids < <(awk -F '\t' 'NR>1 {print $4}' "$record")
[[ ${#ids[@]} == 46 ]] || { echo 'Unexpected job count; inspect the submission record.' >&2; exit 1; }
for id in "${ids[@]}"; do [[ "$id" =~ ^[0-9]+$ ]] || exit 1; done
# These are only this campaign's old jobs, including their array tasks.
job_list=$(IFS=,; printf '%s' "${ids[*]}")
live=$(squeue --noheader --jobs="$job_list" --format='%F' | sort -u)
if [[ -n "$live" ]]; then
    mapfile -t live_ids <<< "$live"
    scancel "${live_ids[@]}"
fi
export OUTPUT_ROOT="$ROOT/results_retry_$(date +%Y%m%d_%H%M%S)"
printf 'New output directory: %s\n' "$OUTPUT_ROOT"
bash "$ROOT/submit.sh" submit

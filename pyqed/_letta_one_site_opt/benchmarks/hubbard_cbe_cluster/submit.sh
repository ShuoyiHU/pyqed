#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == worker ]]; then
 run=$2
 exec bash "$run/python.sh" -u "$run/run.py" --run "$run" --index "$SLURM_ARRAY_TASK_ID"
fi
base=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-plan}
case $mode in plan|check|submit) ;; *) echo 'Usage: bash submit.sh [plan|check|submit]' >&2; exit 2;; esac
bash "$base/python.sh" "$base/check_plan.py" "$base"
[[ $mode != plan ]] || exit 0
repo=$(cat "$base/SOURCE_PATH")
bash "$base/python.sh" "$base/preflight.py" ours "$repo"
bash "$base/python.sh" -m pytest -q -p no:cacheprovider  "$repo/tests/test_letta_cbe_stable_metric.py" "$repo/tests/test_letta_factor_coordinates.py"  "$repo/tests/test_letta_reduced_cbe.py"
[[ $mode != check ]] || exit 0
run="$base/runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run/logs"
cp "$base/"*.py "$base/"*.sh "$base/plan.json" "$base/SOURCE_PATH" "$run/"
bash "$run/python.sh" "$run/check_plan.py" "$run" "$run"
while IFS=$'\t' read -r memory indices; do
 job=$(sbatch --parsable --partition="${PARTITIONS:-gubing}" --qos="${QOS:-huge}" \
   --array="$indices" --cpus-per-task=1 --mem="$memory" \
   --job-name=letta_cbe_qr --output="$run/logs/%A_%a.out" --error="$run/logs/%A_%a.err" \
   "$run/submit.sh" worker "$run")
 printf '%s\t%s\t%s\n' "$job" "$memory" "$indices" | tee -a "$run/jobs.tsv"
done < "$run/submission_groups.tsv"
printf '%s\n' "$run" > "$base/latest_run.txt"
printf 'Results: %s\n' "$run"

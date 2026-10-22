#!/usr/bin/env bash
set -euo pipefail
mode=${1:-dry-run}
if [[ "$mode" == worker ]]; then
    : "${FECO_ROOT:?Worker needs original bundle directory}"
else
    FECO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$FECO_ROOT/results}
export FECO_ROOT OUTPUT_ROOT
if [[ "$mode" == worker || "$mode" == preflight || "$mode" == collect ]]; then
    source "$FECO_ROOT/environment.sh"
fi
case "$mode" in
 worker)
    stage=${2:?};basis=${3:-0};space=${4:-0}
    if [[ "$stage" == build ]]; then
        "$LETTA_PYTHON" "$FECO_ROOT/restore_runtime.py"
        "$LETTA_PYTHON" "$FECO_ROOT/campaign.py" preflight
        "$LETTA_PYTHON" -c 'import Cython, setuptools; print("Compiler dependencies available")'
        "$LETTA_PYTHON" -m pyqed.mps.nonabelian.sweep_engine
        exec "$LETTA_PYTHON" "$FECO_ROOT/smoke.py"
    fi
    if [[ "$stage" == prepare ]]; then
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
    fi
    exec "$LETTA_PYTHON" -u "$FECO_ROOT/campaign.py" "$stage" --basis "$basis" --space "$space" --cap "${SLURM_ARRAY_TASK_ID:-0}"
    ;;
 preflight) exec "$LETTA_PYTHON" "$FECO_ROOT/campaign.py" preflight ;;
 collect) exec "$LETTA_PYTHON" "$FECO_ROOT/campaign.py" collect ;;
 dry-run|submit) ;;
 *) echo 'Usage: bash submit.sh [dry-run|preflight|submit|collect]' >&2; exit 2 ;;
esac
# Refuse duplicate submission; dry-run never creates a submission record.
if [[ "$mode" == submit ]]; then
    mkdir -p "$OUTPUT_ROOT/logs"
    (set -o noclobber; printf 'stage\tbasis\tspace\tjob_id\tdependency\n' > "$OUTPUT_ROOT/submitted.tsv") || {
        echo 'Submission record exists; use targeted resubmission, not a duplicate campaign.' >&2; exit 1;
    }
fi
common=(--parsable --partition="${PARTITION:-gubing}" --qos="${QOS:-huge}" --time=0 --nodes=1 --ntasks=1 --export=ALL --chdir="$FECO_ROOT")
dispatch() {
    local stage=$1 basis=$2 space=$3 dep=$4 array=$5 cpus=$6 memory=$7
    local opts=("${common[@]}" --job-name="qc_${stage}_${basis}_${space}" --cpus-per-task="$cpus" --mem="$memory"
        --output="$OUTPUT_ROOT/logs/${stage}_${basis}_${space}_%A_%a.out" --error="$OUTPUT_ROOT/logs/${stage}_${basis}_${space}_%A_%a.err")
    [[ "$dep" == - ]] || opts+=(--dependency="$dep")
    [[ "$array" == - ]] || opts+=(--array="$array")
    if [[ "$mode" == dry-run ]]; then
        printf '%q ' sbatch "${opts[@]}" "$FECO_ROOT/submit.sh" worker "$stage" "$basis" "$space"; printf '\n'
        job_id="${stage}_${basis}_${space}_ID"
    else
        local reply
        reply=$(sbatch "${opts[@]}" "$FECO_ROOT/submit.sh" worker "$stage" "$basis" "$space")
        job_id=${reply%%;*};[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch reply: $reply" >&2; exit 1; }
        printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$basis" "$space" "$job_id" "$dep" >> "$OUTPUT_ROOT/submitted.tsv"
        echo "Submitted $stage basis=$basis CAS=$space job=$job_id dependency=$dep"
    fi
}
dispatch build 0 0 - - 1 8G; build_id=$job_id
for basis in 0 1 2 3 4; do
    dispatch prepare "$basis" 0 "afterok:$build_id" - "${PREP_CPUS:-8}" "${PREP_MEMORY:-64G}";prep_id=$job_id
    for space in 0 1 2 3; do
        dispatch mps "$basis" "$space" "afterok:$prep_id" 0-3 1 "${SOLVER_MEMORY:-64G}";mps_id=$job_id
        # Array element 0 is D_mul=30. Do not wait for D=50,70,90.
        dispatch letta "$basis" "$space" "afterok:${mps_id}_0" - 1 "${SOLVER_MEMORY:-64G}"
    done
done
echo 'Prepared grid: 80 MPS tasks + 20 LETTA tasks; additionally 5 preparation jobs and 1 build/smoke job.'

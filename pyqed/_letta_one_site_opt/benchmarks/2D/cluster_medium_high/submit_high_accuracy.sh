#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-submit}
if [[ "$MODE" == worker ]]; then
    : "${BENCHMARK_ROOT:?Slurm worker requires the exported original bundle path}"
fi
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
ROOT=$BENCHMARK_ROOT
SOURCE_ROOT="$ROOT/source"
CONDA_ROOT=${CONDA_ROOT:-/share/home/gubingLab/hushuoyi/miniconda3}
CONDA_ENV=${CONDA_ENV:-pyqed_letta_cbe}
CPUS=${CPUS:-1}
MEMORY_GIB=${MEMORY_GIB:-64}
WORKSPACE_MB=${WORKSPACE_MB:-1024}
SWEEPS=${SWEEPS:-2000}
TWO_SITE_SWEEPS=${TWO_SITE_SWEEPS:-20}
export TWO_SITE_SWEEPS
NONLINEAR_ITERATIONS=${NONLINEAR_ITERATIONS:-100}
ENERGY_REFINEMENT_ITERATIONS=${ENERGY_REFINEMENT_ITERATIONS:-256}
SEEDS=${SEEDS:-731}
PROFILES=${PROFILES:-als100-lsmr2000}
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$CPUS}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS=$OPENBLAS_NUM_THREADS
export MPLCONFIGDIR=${TMPDIR:-/tmp}/letta_compression_${USER:-user}_${SLURM_JOB_ID:-preflight}
mkdir -p "$MPLCONFIGDIR"
if [[ -z ${LETTA_PYTHON:-} ]]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    LETTA_PYTHON=$(command -v python)
fi
cd "$SOURCE_ROOT"
case "$MODE" in
    preflight)
        exec "$LETTA_PYTHON" "$ROOT/run_high_accuracy.py" preflight --root "$ROOT"
        ;;
    worker)
        : "${RUN_DIR:?Missing RUN_DIR}" "${SLURM_ARRAY_TASK_ID:?Missing array index}"
        exec "$LETTA_PYTHON" -u "$ROOT/run_high_accuracy.py" run --root "$ROOT" \
            --plan "$RUN_DIR/plan.json" --task-index "$SLURM_ARRAY_TASK_ID"
        ;;
    collect)
        RUN_DIR=${RUN_DIR:-${2:-}}
        if [[ -z "$RUN_DIR" ]]; then RUN_DIR=$(cat "$ROOT/latest_run.txt"); fi
        exec "$LETTA_PYTHON" "$ROOT/run_high_accuracy.py" collect --plan "$RUN_DIR/plan.json"
        ;;
    plan|submit) ;;
    *) echo 'Usage: bash submit_high_accuracy.sh [preflight|plan|submit|collect [RUN_DIR]|worker]' >&2; exit 2 ;;
esac
for value in "$CPUS" "$MEMORY_GIB" "$WORKSPACE_MB" "$SWEEPS" "$TWO_SITE_SWEEPS" "$NONLINEAR_ITERATIONS" "$ENERGY_REFINEMENT_ITERATIONS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Resource and iteration budgets must be positive integers' >&2; exit 2; }
done
"$LETTA_PYTHON" "$ROOT/run_high_accuracy.py" preflight --root "$ROOT"
RUN_DIR=${RUN_DIR:-${ROOT}/runs/$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$RUN_DIR/logs"
plan_args=(--root "$ROOT")
read -r -a seeds_array <<< "$SEEDS"
for selection in MODELS CASES ALGORITHMS PROFILES; do
    value=${!selection:-}
    if [[ -n "$value" ]]; then
        read -r -a values <<< "$value"
        case "$selection" in
            MODELS) plan_args+=(--models "${values[@]}");;
            CASES) plan_args+=(--cases "${values[@]}");;
            ALGORITHMS) plan_args+=(--algorithms "${values[@]}");;
            PROFILES) plan_args+=(--profiles "${values[@]}");;
        esac
    fi
done
"$LETTA_PYTHON" "$ROOT/run_high_accuracy.py" plan --output "$RUN_DIR/plan.json" \
    --seeds "${seeds_array[@]}" --sweeps "$SWEEPS" --nonlinear-iterations "$NONLINEAR_ITERATIONS" \
    --energy-refinement-iterations "$ENERGY_REFINEMENT_ITERATIONS" \
    --workspace-mb "$WORKSPACE_MB" --memory-gib "$MEMORY_GIB" --cpus "$CPUS" "${plan_args[@]}"
printf 'Plan: %s\n' "$RUN_DIR/plan.json"
if [[ "$MODE" == plan ]]; then exit 0; fi
last=$("$LETTA_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"])-1)' "$RUN_DIR/plan.json")
export BENCHMARK_ROOT RUN_DIR CONDA_ROOT CONDA_ENV LETTA_PYTHON CPUS
# Preserve the original bundle path: Slurm executes a copied batch script.
# --time=0 requests unlimited walltime. There is no % concurrency cap.
JOB_ID=$(sbatch --parsable -p gubing -q huge --time=0 --job-name=letta_medium_high \
    --array="0-$last" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="${MEMORY_GIB}G" \
    --export=ALL --output="$RUN_DIR/logs/%A_%a.out" --error="$RUN_DIR/logs/%A_%a.err" \
    "$ROOT/submit_high_accuracy.sh" worker)
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job_id.txt"
printf '%s\n' "$RUN_DIR" > "$ROOT/latest_run.txt"
printf 'Submitted %s\nResults: %s\n' "$JOB_ID" "$RUN_DIR"

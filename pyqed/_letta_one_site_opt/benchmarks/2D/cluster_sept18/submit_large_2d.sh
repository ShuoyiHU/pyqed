#!/usr/bin/env bash
# Invoke with bash on the login node; this script submits its worker mode.
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYQED_REPO=${PYQED_REPO:-/share/home/gubingLab/hushuoyi/software/pyqed_bg_letta_cbe}
CONDA_ROOT=${CONDA_ROOT:-/share/home/gubingLab/hushuoyi/miniconda3}
CONDA_ENV=${CONDA_ENV:-pyqed_letta_cbe}
PARTITION=${PARTITION:-gubing}
# Empty or 'default' omits --qos and lets Slurm use the account default.
QOS=${QOS-huge}
CPUS=${CPUS:-8}
MEMORY_GIB=${MEMORY_GIB:-256}
MAX_SWEEPS=${MAX_SWEEPS:-100}
TWO_SITE_MAX_SWEEPS=${TWO_SITE_MAX_SWEEPS:-${MAX_SWEEPS}}
TOLERANCE=${TOLERANCE:-1e-9}
MODELS=${MODELS:-ising heisenberg bose_hubbard fermi_hubbard}
SHAPES=${SHAPES:-3x3 3x6 3x9 4x4 4x8 4x12 5x5 5x10 6x6 6x12 7x7 8x8 9x9}
BOND_DIMS=${BOND_DIMS:-4 8}
SEEDS=${SEEDS:-731 732}
SOLVERS=${SOLVERS:-one_site cbe two_site}
ALLOW_LARGE_MEMORY=${ALLOW_LARGE_MEMORY:-0}
FORCE_RERUN=${FORCE_RERUN:-0}
DRIVER=${RUN_ROOT}/run_large_2d.py
MODE=${1:-submit}

[[ "$CPUS" =~ ^[1-9][0-9]*$ && "$MEMORY_GIB" =~ ^[1-9][0-9]*$ ]] || {
    echo "CPUS and MEMORY_GIB must be positive integers" >&2; exit 2;
}
export PYTHONPATH="$PYQED_REPO"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NUMBA_DISABLE_JIT=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS=$OPENBLAS_NUM_THREADS
export MPLCONFIGDIR=${TMPDIR:-/tmp}/letta_mpl_${USER:-user}_${SLURM_JOB_ID:-local}
mkdir -p "$MPLCONFIGDIR"

if [[ -z ${LETTA_PYTHON:-} ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    LETTA_PYTHON=$(command -v python)
fi

case "$MODE" in
    preflight)
        exec "$LETTA_PYTHON" "$DRIVER" preflight --repo "$PYQED_REPO"
        ;;
    collect)
        : "${PLAN:?Set PLAN to the run plan.json}"
        exec "$LETTA_PYTHON" "$DRIVER" collect --plan "$PLAN"
        ;;
    worker)
        : "${PLAN:?Missing PLAN}" "${SLURM_ARRAY_TASK_ID:?Missing array index}"
        flags=()
        [[ "$ALLOW_LARGE_MEMORY" != 1 ]] || flags+=(--allow-large-memory)
        [[ "$FORCE_RERUN" != 1 ]] || flags+=(--force)
        exec "$LETTA_PYTHON" "$DRIVER" run --plan "$PLAN" \
            --task-index "$SLURM_ARRAY_TASK_ID" --memory-gib "$MEMORY_GIB" "${flags[@]}"
        ;;
    plan|submit) ;;
    *) echo "Usage: bash submit_large_2d.sh [preflight|plan|submit|collect]" >&2; exit 2 ;;
esac

"$LETTA_PYTHON" "$DRIVER" preflight --repo "$PYQED_REPO"
if [[ -z ${PLAN:-} ]]; then
    RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
    [[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
    RUN_DIR=${RUN_ROOT}/runs/${RUN_ID}
    PLAN=${RUN_DIR}/plan.json
    read -r -a model_args <<< "$MODELS"
    read -r -a shape_args <<< "$SHAPES"
    read -r -a bond_args <<< "$BOND_DIMS"
    read -r -a seed_args <<< "$SEEDS"
    read -r -a solver_args <<< "$SOLVERS"
    "$LETTA_PYTHON" "$DRIVER" plan --output "$PLAN" --repo "$PYQED_REPO" \
        --models "${model_args[@]}" --shapes "${shape_args[@]}" \
        --bond-dims "${bond_args[@]}" --seeds "${seed_args[@]}" \
        --solvers "${solver_args[@]}" --max-sweeps "$MAX_SWEEPS" \
        --two-site-max-sweeps "$TWO_SITE_MAX_SWEEPS" --tolerance "$TOLERANCE" \
        --memory-gib "$MEMORY_GIB"
else
    RUN_DIR=$(dirname -- "$PLAN")
fi
if [[ "$MODE" == plan ]]; then
    echo "Submit this plan: PLAN='$PLAN' bash '$RUN_ROOT/submit_large_2d.sh' submit"
    exit 0
fi

mkdir -p "$RUN_DIR/logs"
LAST_TASK=$("$LETTA_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"])-1)' "$PLAN")
export RUN_ROOT PLAN PYQED_REPO CONDA_ROOT CONDA_ENV CPUS MEMORY_GIB
export ALLOW_LARGE_MEMORY FORCE_RERUN
scheduler_args=(-p "$PARTITION")
qos_label="account default"
if [[ -n "$QOS" && "$QOS" != default ]]; then
    scheduler_args+=(-q "$QOS")
    qos_label=$QOS
fi
if ! JOB_ID=$(sbatch --parsable "${scheduler_args[@]}" --job-name=letta_2d_sept18 \
    --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="${MEMORY_GIB}G" \
    --time=0 --array="0-${LAST_TASK}" --chdir="$RUN_ROOT" --export=ALL \
    --output="$RUN_DIR/logs/%A_%a.out" --error="$RUN_DIR/logs/%A_%a.err" \
    "$RUN_ROOT/submit_large_2d.sh" worker); then
    echo "Submission failed; the plan is preserved at $PLAN" >&2
    echo 'Check allowed QoS: sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS' >&2
    echo "Retry with QOS=<allowed-name> PLAN='$PLAN' bash '$RUN_ROOT/submit_large_2d.sh' submit" >&2
    echo "QOS=default omits the explicit QoS request and uses the account default." >&2
    exit 1
fi
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID: $JOB_ID" >&2; exit 1; }
echo "$JOB_ID" > "$RUN_DIR/job_id.txt"
echo "Submitted $((LAST_TASK+1)) independent tasks as array $JOB_ID."
echo "Partition $PARTITION; QoS $qos_label; --time=0; no array concurrency cap."
echo "Plan: $PLAN"
echo "Monitor: squeue -j $JOB_ID"
echo "Collect: PLAN='$PLAN' bash '$RUN_ROOT/submit_large_2d.sh' collect"

#!/bin/bash
#SBATCH --job-name=h10_gto_density
#SBATCH --output=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_gto_density_%A_%a_out.txt
#SBATCH --error=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_gto_density_%A_%a_err.txt
#SBATCH --partition=gubing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-8

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc

CONDA_ENV=${CONDA_ENV:-pyqed-bg}
PYQED_REPO=${PYQED_REPO:-/share/home/gubingLab/hushuoyi/software/pyqed_bg}
RUN_ROOT=${RUN_ROOT:-/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10}

conda activate "${CONDA_ENV}"

if [ ! -d "${PYQED_REPO}" ]; then
    echo "PYQED_REPO=${PYQED_REPO} does not exist."
    exit 1
fi

export PYTHONPATH="${PYQED_REPO}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mplconfig_${USER}"
mkdir -p "${MPLCONFIGDIR}" "${RUN_ROOT}/logs"

GEOMETRIES=(afm afm afm bonding bonding bonding edge_localized edge_localized edge_localized)
INTENSITIES=(off I1e13 I5e14 off I1e13 I5e14 off I1e13 I5e14)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_TASKS=9

if [ -n "${RESULT_DIR:-}" ]; then
    OUTPUT_DIR="${RESULT_DIR}"
else
    if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${N_TASKS}" ]; then
        echo "TASK_ID=${TASK_ID} outside valid range [0, $((N_TASKS - 1))]."
        exit 1
    fi

    GEOMETRY=${GEOMETRIES[$TASK_ID]}
    INTENSITY=${INTENSITIES[$TASK_ID]}
    BASIS=${BASIS:-cc-pvdz}
    NCAS=${NCAS:-14}
    NELECAS=${NELECAS:-10}
    D=${D:-40}
    OUTPUT_DIR="${RUN_ROOT}/benchmark_results/h10_${GEOMETRY}_gto_tdvp_${BASIS}_CAS${NELECAS}e_${NCAS}o_D${D}_${INTENSITY}"
fi

echo "============================================================"
echo "H10 GTO density plot task"
echo "SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID:-none}"
echo "CONDA_ENV = ${CONDA_ENV}"
echo "PYQED_REPO = ${PYQED_REPO}"
echo "RUN_ROOT = ${RUN_ROOT}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "============================================================"

python - <<'PY'
import pyqed
print("Using pyqed from:", pyqed.__file__, flush=True)
PY

python "${RUN_ROOT}/plot_h10_gto_density.py" "${OUTPUT_DIR}"

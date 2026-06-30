#!/bin/bash
#SBATCH --job-name=h10_bg_cap
#SBATCH --output=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_bg_cap_%A_%a_out.txt
#SBATCH --error=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_bg_cap_%A_%a_err.txt
#SBATCH --partition=gubing
#SBATCH -q huge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=0-8%3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

source ~/.bashrc

CONDA_ENV=${CONDA_ENV:-pyqed-bg}
PYQED_REPO=${PYQED_REPO:-/share/home/gubingLab/hushuoyi/software/pyqed_bg}
RUN_ROOT=${RUN_ROOT:-/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10}

conda activate "${CONDA_ENV}"

if [ ! -d "${PYQED_REPO}" ]; then
    echo "PYQED_REPO=${PYQED_REPO} does not exist."
    echo "Set PYQED_REPO to the BG checkout, for example:"
    echo "  export PYQED_REPO=/share/home/gubingLab/hushuoyi/software/pyqed_bg"
    exit 1
fi

export PYTHONPATH="${PYQED_REPO}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mplconfig_${USER}"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/benchmark_results"

python - <<'PY'
import pyqed
print("Using pyqed from:", pyqed.__file__, flush=True)
PY

GEOMETRIES=(afm afm afm bonding bonding bonding edge_localized edge_localized edge_localized)
INTENSITIES=(off I1e13 I5e14 off I1e13 I5e14 off I1e13 I5e14)
AMPLITUDES=(0.0 0.016880323915389028 0.11936191509197033 0.0 0.016880323915389028 0.11936191509197033 0.0 0.016880323915389028 0.11936191509197033)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_TASKS=9

if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${N_TASKS}" ]; then
    echo "TASK_ID=${TASK_ID} outside valid range [0, $((N_TASKS - 1))]."
    exit 1
fi

GEOMETRY=${GEOMETRIES[$TASK_ID]}
INTENSITY=${INTENSITIES[$TASK_ID]}
AMPLITUDE=${AMPLITUDES[$TASK_ID]}

ORIGINAL_NZ=63
EXPANDED_NZ=71
ORIGINAL_LZ=18.0
EXPANDED_LZ=20.25
D=40
NEWTON_CYCLES=200
OMEGA=0.05841455452769231
CYCLES=2
DT=0.1
SAVE_EVERY=20
CAP_WIDTH=2.0
CAP_STRENGTH=0.01
CAP_ORDER=2

OUTPUT_DIR="${RUN_ROOT}/benchmark_results/h10_${GEOMETRY}_bg_tdvp_cap_Nz${EXPANDED_NZ}_D${D}_${INTENSITY}"

echo "============================================================"
echo "H10 BG GDVR-TDDMRG CAP task"
echo "SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "geometry = ${GEOMETRY}"
echo "intensity = ${INTENSITY}"
echo "drive amplitude = ${AMPLITUDE}"
echo "original grid: Lz=${ORIGINAL_LZ}, Nz=${ORIGINAL_NZ}"
echo "expanded CAP grid: Lz=${EXPANDED_LZ}, Nz=${EXPANDED_NZ}"
echo "D = ${D}"
echo "newton_cycles = ${NEWTON_CYCLES}"
echo "omega = ${OMEGA}"
echo "cycles = ${CYCLES}"
echo "dt = ${DT}"
echo "save_every = ${SAVE_EVERY}"
echo "CAP width/strength/order = ${CAP_WIDTH}/${CAP_STRENGTH}/${CAP_ORDER}"
echo "output directory = ${OUTPUT_DIR}"
echo "============================================================"

python "${RUN_ROOT}/h10_bg_tdvp_cap.py" \
    --geometry "${GEOMETRY}" \
    --intensity "${INTENSITY}" \
    --bond-dim "${D}" \
    --newton-cycles "${NEWTON_CYCLES}" \
    --dt "${DT}" \
    --omega "${OMEGA}" \
    --cycles "${CYCLES}" \
    --drive-amplitude "${AMPLITUDE}" \
    --save-every "${SAVE_EVERY}" \
    --cap-width "${CAP_WIDTH}" \
    --cap-strength "${CAP_STRENGTH}" \
    --cap-order "${CAP_ORDER}" \
    --output-dir "${OUTPUT_DIR}"

echo "Finished H10 BG CAP geometry=${GEOMETRY}, intensity=${INTENSITY}"

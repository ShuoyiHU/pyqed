#!/bin/bash
#SBATCH --job-name=h10_memprobe
#SBATCH --output=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_29th_test_memory/logs/h10_memprobe_%A_%a_out.txt
#SBATCH --error=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_29th_test_memory/logs/h10_memprobe_%A_%a_err.txt
#SBATCH --partition=gubing
#SBATCH -q huge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=0-2%1
#SBATCH --chdir=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_29th_test_memory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_RUN_ROOT=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_29th_test_memory

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

source ~/.bashrc

CONDA_ENV=${CONDA_ENV:-pyqed-bg}
PYQED_REPO=${PYQED_REPO:-/share/home/gubingLab/hushuoyi/software/pyqed_bg}
RUN_ROOT=${RUN_ROOT:-${DEFAULT_RUN_ROOT}}

conda activate "${CONDA_ENV}"

if [ ! -d "${PYQED_REPO}" ]; then
    echo "PYQED_REPO=${PYQED_REPO} does not exist."
    exit 1
fi

export PYTHONPATH="${PYQED_REPO}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mplconfig_${USER}"
mkdir -p "${MPLCONFIGDIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/memory_probe"

python - <<'PY'
import pyqed
print("Using pyqed from:", pyqed.__file__, flush=True)
PY

GEOMETRIES=(bonding edge_localized afm)
INTENSITIES=(I5e14 I5e14 I5e14)
AMPLITUDES=(0.11936191509197033 0.11936191509197033 0.11936191509197033)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
GEOMETRY=${GEOMETRIES[$TASK_ID]}
INTENSITY=${INTENSITIES[$TASK_ID]}
AMPLITUDE=${AMPLITUDES[$TASK_ID]}

D=${D:-40}
STEPS=${STEPS:-260}
DT=${DT:-0.1}
NEWTON_CYCLES=${NEWTON_CYCLES:-200}
DMRG_SWEEPS=${DMRG_SWEEPS:-20}
DIAG_EVERY=${DIAG_EVERY:-1}
STOP_RSS_GB=${STOP_RSS_GB:-185}

OUTPUT_DIR="${RUN_ROOT}/memory_probe/h10_${GEOMETRY}_cap_D${D}_${INTENSITY}_steps${STEPS}"

echo "============================================================"
echo "H10 BG TDVP CAP memory probe"
echo "SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID:-none}"
echo "geometry = ${GEOMETRY}"
echo "intensity = ${INTENSITY}"
echo "drive amplitude = ${AMPLITUDE}"
echo "D = ${D}"
echo "steps = ${STEPS}"
echo "dt = ${DT}"
echo "newton_cycles = ${NEWTON_CYCLES}"
echo "dmrg_sweeps = ${DMRG_SWEEPS}"
echo "diag_every = ${DIAG_EVERY}"
echo "stop_rss_gb = ${STOP_RSS_GB}"
echo "output directory = ${OUTPUT_DIR}"
echo "============================================================"

python "${RUN_ROOT}/memory_probe_h10_bg_tdvp_cap.py" \
    --geometry "${GEOMETRY}" \
    --intensity "${INTENSITY}" \
    --bond-dim "${D}" \
    --steps "${STEPS}" \
    --dt "${DT}" \
    --drive-amplitude "${AMPLITUDE}" \
    --newton-cycles "${NEWTON_CYCLES}" \
    --dmrg-sweeps "${DMRG_SWEEPS}" \
    --diag-every "${DIAG_EVERY}" \
    --stop-rss-gb "${STOP_RSS_GB}" \
    --output-dir "${OUTPUT_DIR}"

echo "Memory probe finished: ${OUTPUT_DIR}"
echo "Main table: ${OUTPUT_DIR}/memory_step_diagnostics.csv"

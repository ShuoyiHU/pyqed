#!/bin/bash
#SBATCH --job-name=h10_gto_tdvp
#SBATCH --output=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_gto_tdvp_%A_%a_out.txt
#SBATCH --error=/storage/gubingLab/hushuoyi/gdvr_tddmrg_bg_branch/June_28th_H10/logs/h10_gto_tdvp_%A_%a_err.txt
#SBATCH --partition=gubing
#SBATCH -q huge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=0-8

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

BASIS=${BASIS:-cc-pvdz}
NCAS=${NCAS:-14}
NELECAS=${NELECAS:-10}
D=${D:-40}
DMRG_SWEEPS=${DMRG_SWEEPS:-20}
OMEGA=0.05841455452769231
CYCLES=2
DT=0.1
SAVE_EVERY=20
TDVP_DYNAMIC_MODE=midpoint
TRACK_ENERGY=${TRACK_ENERGY:-1}
SAVE_DENSITY=${SAVE_DENSITY:-1}
DENSITY_NZ=${DENSITY_NZ:-401}
DENSITY_PADDING=${DENSITY_PADDING:-4.0}

TRACK_ENERGY_ARG=()
if [ "${TRACK_ENERGY}" = "1" ] || [ "${TRACK_ENERGY}" = "true" ] || [ "${TRACK_ENERGY}" = "yes" ]; then
    TRACK_ENERGY_ARG=(--track-energy)
fi

SAVE_DENSITY_ARG=()
if [ "${SAVE_DENSITY}" = "1" ] || [ "${SAVE_DENSITY}" = "true" ] || [ "${SAVE_DENSITY}" = "yes" ]; then
    SAVE_DENSITY_ARG=(--save-density --density-nz "${DENSITY_NZ}" --density-padding "${DENSITY_PADDING}")
fi

OUTPUT_DIR="${RUN_ROOT}/benchmark_results/h10_${GEOMETRY}_gto_tdvp_${BASIS}_CAS${NELECAS}e_${NCAS}o_D${D}_${INTENSITY}"

echo "============================================================"
echo "H10 GTO TDDMRG TDVP task"
echo "SLURM_ARRAY_JOB_ID = ${SLURM_ARRAY_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "geometry = ${GEOMETRY}"
echo "intensity = ${INTENSITY}"
echo "drive amplitude = ${AMPLITUDE}"
echo "basis = ${BASIS}"
echo "active orbitals ncas = ${NCAS}"
echo "active electrons nelecas = ${NELECAS}"
echo "active space = CAS(${NELECAS}e, ${NCAS}o)"
echo "D = ${D}"
echo "dmrg_sweeps = ${DMRG_SWEEPS}"
echo "omega = ${OMEGA}"
echo "cycles = ${CYCLES}"
echo "dt = ${DT}"
echo "save_every = ${SAVE_EVERY}"
echo "tdvp_dynamic_mode = ${TDVP_DYNAMIC_MODE}"
echo "track_energy = ${TRACK_ENERGY}"
echo "save_density = ${SAVE_DENSITY}"
echo "density_nz/padding = ${DENSITY_NZ}/${DENSITY_PADDING}"
echo "CAP = none for finite GTO active-space propagation"
echo "output directory = ${OUTPUT_DIR}"
echo "============================================================"

python "${RUN_ROOT}/h10_gto_tdvp.py" \
    --geometry "${GEOMETRY}" \
    --intensity "${INTENSITY}" \
    --basis "${BASIS}" \
    --ncas "${NCAS}" \
    --nelecas "${NELECAS}" \
    --bond-dim "${D}" \
    --dmrg-sweeps "${DMRG_SWEEPS}" \
    --dt "${DT}" \
    --omega "${OMEGA}" \
    --cycles "${CYCLES}" \
    --drive-amplitude "${AMPLITUDE}" \
    --save-every "${SAVE_EVERY}" \
    --tdvp-dynamic-mode "${TDVP_DYNAMIC_MODE}" \
    "${TRACK_ENERGY_ARG[@]}" \
    "${SAVE_DENSITY_ARG[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo "Generating quick-look figures for ${OUTPUT_DIR}"
python "${RUN_ROOT}/plot_h10_gto_tdvp.py" "${OUTPUT_DIR}" || echo "Warning: plot generation failed for ${OUTPUT_DIR}"

echo "Finished H10 GTO TDVP geometry=${GEOMETRY}, intensity=${INTENSITY}"

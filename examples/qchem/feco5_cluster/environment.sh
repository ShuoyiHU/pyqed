#!/usr/bin/env bash
# Source after FECO_ROOT is set. No block2 runtime or MKL preloading is required.
if [[ -z ${LETTA_PYTHON:-} ]]; then
    LETTA_PYTHON="${PYQED_SU2_ENV:-/share/home/gubingLab/hushuoyi/miniconda3/envs/pyqed_letta_su2}/bin/python"
fi
[[ -x "$LETTA_PYTHON" ]] || { echo "Set LETTA_PYTHON to a Python with PySCF, NumPy, SciPy, Cython, setuptools." >&2; return 1; }
export LETTA_PYTHON
export PYTHONPATH="$FECO_ROOT/source:$FECO_ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYQED_SWEEP_CACHE="$FECO_ROOT/build/sweeps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/feco5_mpl_${USER:-user}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$MPLCONFIGDIR"

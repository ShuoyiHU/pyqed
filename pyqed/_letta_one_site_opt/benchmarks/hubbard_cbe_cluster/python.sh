#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cat "$base/SOURCE_PATH")
[[ -d "$repo/pyqed" ]] || { echo "Frozen source missing: $repo" >&2; exit 1; }
runtime=${PYQED_SU2_ENV:-/share/home/gubingLab/hushuoyi/miniconda3/envs/pyqed_letta_su2}
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/pyqed-su2-$USER/$(basename "$repo")}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/pyqed-mpl-$USER}
exec "$runtime/bin/python" -P "$@"

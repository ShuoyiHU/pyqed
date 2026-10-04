#!/usr/bin/env bash
set -euo pipefail
# Invoke from the repository root. All optimization runs use a single CPU thread.
PYTHON=${PYTHON:-.venv-1/bin/python}
OUTPUT=${1:-/private/tmp/letta_periodic_hubbard}
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/private/tmp/letta-periodic-mpl}
"$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard reference --output "$OUTPUT"
"$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard run --output "$OUTPUT"
for LENGTH in 5 10; do
    WARM="$OUTPUT/warm_L${LENGTH}"
    "$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard run \
        --output "$WARM" --lengths "$LENGTH" --bonds 6 \
        --warm-start "$OUTPUT/L${LENGTH}_D4_mps_seed731.json"
    "$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard_validation --output "$WARM"
done
"$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard_validation --output "$OUTPUT"
"$PYTHON" -m pyqed._letta_one_site_opt.benchmarks.periodic_hubbard plot --output "$OUTPUT"

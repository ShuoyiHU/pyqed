#!/bin/bash
# Same-start periodic MPS / NN-tied LETTA, with independent reference validation.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/letta-mpl}"
output="${1:-/private/tmp/periodic_bose_hubbard_fresh}"
python="${PYTHON:-.venv-1/bin/python}"
runner=pyqed._letta_one_site_opt.benchmarks.periodic_hubbard
"$python" -m "$runner" reference --model bose --output "$output"
"$python" -m "$runner" run --model bose --gauge-floor 1e-3 --output "$output"
for length in 5 10; do
  "$python" -m "$runner" run --model bose --gauge-floor 1e-3 --output "$output/warm_L$length" --lengths "$length" --bonds 6 --warm-start "$output/L${length}_D4_mps_seed731.json"
done
"$python" -m pyqed._letta_one_site_opt.benchmarks.periodic_bose_validation --output "$output"
"$python" -m "$runner" plot --model bose --main-only --output "$output"

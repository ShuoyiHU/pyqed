#!/usr/bin/env bash
set -euo pipefail

LETTA_AUDIT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LETTA_AUDIT_PYTHON="${LETTA_AUDIT_PYTHON:-/Users/shuoyihu/miniforge3/envs/block2-pyscf/bin/python}"
LETTA_AUDIT_PURE_SITE="${LETTA_AUDIT_PURE_SITE:-/Users/shuoyihu/miniforge3/envs/pyqed-plot/lib/python3.9/site-packages}"

if [[ ! -x "${LETTA_AUDIT_PYTHON}" ]]; then
    echo "Python executable not found: ${LETTA_AUDIT_PYTHON}" >&2
    echo "Set LETTA_AUDIT_PYTHON to a Python 3.10+ executable with NumPy and SciPy." >&2
    exit 2
fi

# The numerical Python has the desired NumPy/SciPy versions, while pytest and
# opt_einsum are pure-Python packages in pyqed-plot. Assemble those packages in
# an isolated temporary import directory without modifying either environment.
LETTA_AUDIT_TMP="$(mktemp -d /private/tmp/letta-cbe-audit.XXXXXX)"

cleanup_letta_audit() {
    case "${LETTA_AUDIT_TMP}" in
        /private/tmp/letta-cbe-audit.*)
            rm -rf -- "${LETTA_AUDIT_TMP}"
            ;;
    esac
}
trap cleanup_letta_audit EXIT

for LETTA_AUDIT_PACKAGE in \
    pytest _pytest pluggy iniconfig packaging pygments opt_einsum py.py
do
    LETTA_AUDIT_SOURCE="${LETTA_AUDIT_PURE_SITE}/${LETTA_AUDIT_PACKAGE}"
    if [[ ! -e "${LETTA_AUDIT_SOURCE}" ]]; then
        echo "Required pure-Python package not found: ${LETTA_AUDIT_SOURCE}" >&2
        exit 2
    fi
    ln -s "${LETTA_AUDIT_SOURCE}" \
        "${LETTA_AUDIT_TMP}/${LETTA_AUDIT_PACKAGE}"
done

mkdir -p "${LETTA_AUDIT_TMP}/mplconfig"
export PYTHONPATH="${LETTA_AUDIT_ROOT}:${LETTA_AUDIT_TMP}"
export MPLCONFIGDIR="${LETTA_AUDIT_TMP}/mplconfig"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${LETTA_AUDIT_ROOT}"

echo "Python and numerical-library versions"
"${LETTA_AUDIT_PYTHON}" - <<'PY'
import sys

import numpy
import opt_einsum
import pytest
import scipy

print("python     ", sys.version.split()[0])
print("numpy      ", numpy.__version__)
print("scipy      ", scipy.__version__)
print("pytest     ", pytest.__version__)
print("opt_einsum ", opt_einsum.__version__)
PY

echo
echo "1/3: correctness, forbidden-pair-operation, and scaling assertions"
"${LETTA_AUDIT_PYTHON}" -m pytest \
    tests/test_letta_cbe.py \
    tests/test_letta_cbe_scaling.py \
    -q

echo
echo "2/3: deterministic contraction-graph scaling report"
"${LETTA_AUDIT_PYTHON}" - <<'PY'
from pyqed._letta_one_site_opt.benchmarks.cbe_scaling import (
    run_scaling_profile,
)

print(
    "direction  axis       one-site   strict+CBE/SVD   pair-reference"
)
for direction in ("lr", "rl"):
    report = run_scaling_profile(direction=direction)
    for axis in ("bond", "physical", "mpo"):
        exponents = report["exponents"][axis]
        print(
            f"{direction:>9}  {axis:<8}  "
            f"{exponents['one_site_action']:9.4f}  "
            f"{exponents['strict_selector_with_svd']:15.4f}  "
            f"{exponents['pair_action']:14.4f}"
        )
    print("proof counters:", report["proof"])
PY

echo
echo "3/3: convergence comparison from the same initial state"
"${LETTA_AUDIT_PYTHON}" -m \
    pyqed._letta_one_site_opt.benchmarks.cbe_convergence \
    --shape 2x3 \
    --bond-dim 2 \
    --expansion-dimension 1 \
    --max-sweeps 4 \
    --seed 732

echo
echo "LETTA strict-CBE audit completed successfully."

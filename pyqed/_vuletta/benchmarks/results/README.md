# Conditional-gauge VULETTA benchmark snapshot

This snapshot compares the direct conditional-canonical VULETTA solver with
the repository's one-site VUMPS implementation. The full row-by-row results
are available as [Markdown](one_dimensional_models.md) and
[CSV](one_dimensional_models.csv).

## Configuration

- models: TFIM at $J=1$, $g=1.5$ and the antiferromagnetic spin-$1/2$
  Heisenberg chain;
- LETTA bond dimensions: $D=1,2,3$ with bond-dimension continuation;
- MPS bond dimensions: $D=1,2,3,4,6$ with deterministic seed 3;
- tolerance: $10^{-7}$; maximum iterations: 300;
- runtime: median of three measured single-threaded solves after one untimed
  warm-up;
- platform: macOS 15.7.4 arm64, Python 3.12.12, NumPy 2.4.2, SciPy 1.17.1.

The command was

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=. \
python -m pyqed._vuletta.benchmarks.one_dimensional_models \
  --letta-bond-dimensions 1 2 3 \
  --mps-bond-dimensions 1 2 3 4 6 \
  --tolerance 1e-7 --max-iterations 300 --repeats 3 \
  --csv pyqed/_vuletta/benchmarks/results/one_dimensional_models.csv \
  --markdown pyqed/_vuletta/benchmarks/results/one_dimensional_models.md
```

## Main observations

For TFIM, VULETTA reaches energy errors $5.26\times10^{-3}$,
$6.94\times10^{-6}$, and $3.34\times10^{-8}$ at $D=1,2,3$. The corresponding
median runtimes are 0.147, 0.549, and 1.365 seconds. The $D=3$ observables are
also within $5.2\times10^{-7}$ of the exact values, but this row is deliberately
marked nonconverged because its tangent residual $5.57\times10^{-7}$ remains
above the requested $10^{-7}$ threshold.

TFIM VUMPS converges at every tested $D$. Its energy error falls from
$1.09\times10^{-1}$ at $D=1$ to $3.75\times10^{-10}$ at $D=6$, with median
runtimes from 0.022 to 0.112 seconds. On this small dense implementation VUMPS
is therefore faster, while VULETTA obtains substantially more accuracy than an
MPS with the same nominal $D$ because LETTA has twice the transfer bond and
twice the stored tensor entries for physical dimension two.

For the critical Heisenberg chain, VULETTA improves from a poor stationary
$D=1$ solution to energy errors $3.54\times10^{-2}$ and $7.61\times10^{-3}$ at
$D=2$ and $D=3$. Those two rows stop at the injectivity-safe Armijo boundary
with residuals near $4\times10^{-5}$, so they are useful variational states but
not converged solutions at the requested tolerance.

## VUMPS Heisenberg warning

The current one-site VUMPS implementation does not converge on the
antiferromagnetic Heisenberg chain in this setup. It enters an undamped
two-cycle and reaches the 300-iteration limit for every tested $D$. The $D=4$
row even produces energy and observables outside the physical operator bounds
because the nonconverged transfer fixed point is not positive. Those
Heisenberg VUMPS numbers are retained as an explicit failure diagnostic and
must not be interpreted as valid MPS estimates. A two-site unit cell or a
damped/mixed VUMPS update is needed for a meaningful Heisenberg baseline.


# Finite LETTA two-site optimization

This package implements adjacent-pair variational sweeps for the finite
`LatticeLETTA` representation in `_letta_one_site_opt`.

## Why the split is not an MPS split

Adjacent LETTA tensors can depend on the same physical spin. If their
neighborhoods are `N_i` and `N_j`, the pair is divided into

$$
L=N_i\setminus S,\qquad
S=N_i\cap N_j,\qquad
R=N_j\setminus S.
$$

The merged tensor is

$$
\Theta_{l,\sigma_L,\sigma_S,\sigma_R,r}
=
\sum_a
A_{l,\sigma_L,\sigma_S,a}
B_{a,\sigma_S,\sigma_R,r}.
$$

The shared physical configuration occurs once in the merged tensor. Splitting
is performed independently for every fixed value of that configuration. The
bond-rank constraint therefore applies to every shared sector separately.

## Optimization sequence

For every adjacent chain pair, the solver:

1. Builds the exact block-diagonal overlap metric.
2. Applies the effective Hamiltonian through compatible sparse MPO transition
   pairs.
3. Solves the whitened generalized eigenproblem, using a matrix-free solver
   above the configured dense threshold.
4. Initializes the fixed-bond factorization with a conditional SVD.
5. By default, refines both factors with alternating least squares in the full
   LETTA wavefunction norm.
6. Evaluates the exact post-truncation Rayleigh quotient and rejects any
   energy-increasing proposal.
7. Shifts a direction-dependent QR gauge without shrinking the requested
   virtual-bond allocation.

The Hamiltonian is never materialized in Hilbert space by the production MPO
path. Dense pair frames and dense pair Hamiltonians are used only in small
tests or when `matrix_free=False` is explicitly selected.

## Basic usage

```python
from pyqed._letta_two_site_opt import (
    LETTATwoSiteOptions,
    letta_two_site_dmrg,
)
from pyqed._letta_two_site_opt._letta_for_3d import (
    snake_letta_state,
    transverse_field_ising_mpo,
)

shape = (2, 2, 2)
state = snake_letta_state(shape, bond_dim=2, seed=7)
hamiltonian = transverse_field_ising_mpo(
    shape,
    coupling=1.0,
    field=1.5,
)
result = letta_two_site_dmrg(
    hamiltonian,
    state=state,
    bond_dim=2,
    options=LETTATwoSiteOptions(
        max_sweeps=4,
        split_method="metric-als",
        one_site_polish_sweeps=2,
    ),
)
```

## Symmetry-adapted two-site optimization

The two-site solver accepts the same `AbelianSymmetry` and optional
`bond_charges` arguments as the one-site solver. It restricts the merged-pair
eigenproblem to the target charge, splits each shared-physical configuration
independently inside each virtual charge block, constrains metric-ALS and
energy refinement to allowed factor entries, and performs blockwise QR gauges.

```python
from pyqed._letta_one_site_opt import AbelianSymmetry
from pyqed._letta_one_site_opt._letta_for_3d import snake_coordinates

z2_even = AbelianSymmetry((0, 1), sector=0, moduli=2, name="ising-z2")
hamiltonian = transverse_field_ising_mpo(
    (2, 2, 2), coupling=1.0, field=1.5, basis="x"
)
result = letta_two_site_dmrg(
    hamiltonian,
    lattice_shape=(2, 2, 2),
    coordinates=snake_coordinates((2, 2, 2)),
    bond_dim=4,
    symmetry=z2_even,
    options=LETTATwoSiteOptions(max_sweeps=8, matrix_free=True),
)
```

`LETTAPairUpdate.local_dimension` reports the charge-reduced pair dimension;
`full_local_dimension` reports the unrestricted value.

Run the reproducible with/without-symmetry comparison with

```bash
PYTHONPATH=. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pyqed._letta_two_site_opt.benchmarks.ising_symmetry \
  --repeats 5 --bond-dim 4 \
  --json-output /private/tmp/letta_z2_benchmark.json
```

The JSON output includes energies, convergence, sweep counts, wall times,
parity leakage, allowed/dense parameter counts, local dimensions, a dense
operator-memory proxy, and a cubic local-solver work proxy for every raw run.

`conditional-svd` remains available as a cheaper diagnostic split method.
`metric-als` is the default because it minimizes truncation error in the
represented wavefunction norm rather than the raw tensor Frobenius norm.

## Fixed-rank energy refinement

The alternative `energy-refined` split avoids metric-ALS. It starts from the
shared-sector-aware conditional SVD and alternately minimizes the effective
pair energy over the left and right fixed-rank factors:

$$
H_A(B)a = E N_A(B)a,
$$

followed by

$$
H_B(A)b = E N_B(A)b.
$$

The initializer enforces the requested bond dimension before refinement, so
the refinement optimizes the actual retained LETTA state rather than fitting
the unrestricted merged tensor. Every substep is checked with the pair
Rayleigh quotient. A factor-norm growth limit rejects numerically unstable
near-null-metric directions.

```python
result = letta_two_site_dmrg(
    hamiltonian,
    state=state,
    bond_dim=2,
    options=LETTATwoSiteOptions(
        max_sweeps=20,
        split_method="energy-refined",
        energy_refinement_max_iterations=8,
        energy_refinement_tolerance=1.0e-10,
        energy_refinement_max_factor_norm_growth=100.0,
    ),
)
```

The energy-refinement iterations reuse batched Hamiltonian contractions for
the complete left or right factor frame. This avoids applying the pair
Hamiltonian separately to every frame column.

## One-site versus two-site benchmark

The benchmark runner compares three optimizers from exactly the same random
initial LETTA state for each lattice:

- `one_site`;
- `two_site_metric_als`, the original environment-distance fit;
- `two_site_energy_refined`, the new fixed-rank energy refinement.

It reports:

- whether each run converged and how many sweeps it executed;
- the number of local tensor or tensor-pair updates;
- accepted updates, final energy, and exact-energy error when feasible;
- solver wall time, excluding model construction, exact diagonalization, and
  the optional one-sweep contraction warmup.

From the repository root, run the default 2D `2x3` and 3D `2x2x2` cases with:

```bash
PYTHONPATH=. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pyqed._letta_two_site_opt.benchmarks.ising_convergence
```

For more reliable timings, use at least three repeats and save the complete
per-repeat and per-sweep data:

```bash
PYTHONPATH=. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pyqed._letta_two_site_opt.benchmarks.ising_convergence \
  --repeats 3 \
  --max-sweeps 12 \
  --tolerance 1e-9 \
  --energy-refinement-max-iterations 8 \
  --json-output /private/tmp/letta_ising_benchmark.json
```

Run only one geometry or change its size with, for example,
`--dimension 2d --shape-2d 3x3` or
`--dimension 3d --shape-3d 2x2x3`. Use `--exact-max-sites 0` to disable the
exponentially scaling exact reference for larger lattices. Run `--help` for
all controls. The original two-site comparison uses `metric-als` by default;
`--two-site-split conditional-svd` changes only that baseline, not the
energy-refined method.

A one-site sweep contains one update per lattice site, while a two-site sweep
contains one update per adjacent chain pair. Sweep counts are therefore not
equal units of computational work; compare the convergence flag, local-update
count, time, and final-energy error together. The JSON file keeps raw runs and
the complete energy-density-change history for plotting or further analysis.

## Diagnostics

Every pair update records:

- old, unrestricted, and accepted post-split energies;
- generalized-eigenproblem residual and metric rank;
- the shared physical-site labels;
- per-sector retained ranks and conditional discarded weight;
- environment-weighted truncation loss and ALS iteration count;
- whether the exact energy safeguard accepted the proposal.

## Current scope

- Only adjacent tensors in the one-dimensional LETTA chain are optimized.
- The virtual-bond shape is fixed inside a sweep stage.
- Exact dense frontier environments or the existing sparse-MPO frontier
  contractions are used. Approximate boundary-MPS compression is intentionally
  not exposed here because it would weaken the exact per-pair energy safeguard.
- Two- and three-dimensional model builders are re-exported through the case
  subpackages without duplicating geometry or Hamiltonian code.

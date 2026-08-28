# Finite LETTA one-site optimization

This package separates the reusable one-site optimization machinery from
dimension-specific lattice models.

## Shared core

- `state.py`: finite `LatticeLETTA` representation.
- `operators.py`: generic `LatticeMPO` and small exact-reference helpers.
- `contractions.py`: overlap, expectation, effective-operator, and boundary
  environment contractions.
- `solver.py`: alternating one-site generalized-eigenvalue sweeps and bond
  continuation.

The core must not import either case package.

## User-defined Abelian symmetry sectors

`AbelianSymmetry` describes diagonal on-site charges for any product of
cyclic groups and additive integer charges. A finite modulus defines a
$\mathbb{Z}_n$ factor; `None` defines an unrestricted additive component.
For example, the even Ising spin-flip sector in the $X$ basis is

```python
from pyqed._letta_one_site_opt import (
    AbelianSymmetry,
    LETTADMROptions,
    letta_dmrg,
)
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo,
)

symmetry = AbelianSymmetry(
    physical_charges=(0, 1),
    sector=0,
    moduli=2,
    name="ising-z2",
)
hamiltonian = transverse_field_ising_mpo(
    (2, 3), coupling=1.0, field=1.5, basis="x"
)
result = letta_dmrg(
    hamiltonian,
    lattice_shape=(2, 3),
    bond_dim=4,
    seed=7,
    symmetry=symmetry,
    options=LETTADMROptions(matrix_free=True),
)
```

The first physical axis of each LETTA tensor owns that lattice site's charge.
Positive-neighbor axes are dependency indices and do not count the same charge
again. Virtual charges are allocated automatically, or can be supplied through
`bond_charges`. The local eigensolver, matrix-free action, QR gauge shift, bond
expansion, and bond schedule all preserve the selected sector. The result's
`LETTASiteUpdate.local_dimension` is the symmetry-reduced dimension and
`full_local_dimension` records the corresponding dense dimension.

The physical basis must diagonalize the requested symmetry. For the transverse
field Ising Hamiltonian, `basis="x"` rotates $X\leftrightarrow Z$, exposing the
global spin-flip parity as charges `(0, 1)` without changing the spectrum.

## Exact reduced SU(2)

`ReducedLatticeLETTA` stores SU(2) irreps and outer multiplicities, but never
stores magnetic quantum numbers as independent variational parameters.
Clebsch--Gordan coefficients restore magnetic components only inside local,
polynomial-size contractions. This is a genuine reduced representation rather
than a dense tensor with forbidden entries masked out.

```python
from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedSymmetry,
    letta_dmrg,
    su2_heisenberg_mpo,
)

n = 8
basis = ReducedPhysicalBasis.spin_half()
symmetry = ReducedSymmetry.su2(basis, target_two_j=0)
state = ReducedLatticeLETTA.random(
    (1, n),
    symmetry=symmetry,
    multiplets_per_sector=3,
    seed=7,
)
hamiltonian = su2_heisenberg_mpo(n, physical_basis=basis)
result = letta_dmrg(
    hamiltonian,
    state=state,
    options=LETTADMROptions(
        max_sweeps=8,
        tolerance=1.0e-9,
        gauge_mode="scalar",
        matrix_free=True,
        dense_solver_threshold=64,
    ),
)
print(result.energy, result.state.parameter_count)
```

When `state` is omitted and `symmetry` is a `ReducedSymmetry`, the public
`letta_dmrg` dispatcher constructs this state automatically; in that form,
`bond_dim` denotes the number of outer-multiplicity copies allocated for each
reachable bond sector. Supplying an explicit state is preferable when each
sector needs a different capacity.

The same entry point supports a user-defined local representation. Construct a
`ReducedPhysicalBasis` from sector labels, irreps, and outer multiplicities,
select the desired total charge and spin with `ReducedSymmetry.su2`, and supply
a `ReducedMPOHamiltonian(reduced_factors, canonical_factors)`. The compact
factors describe the reduced operator; the canonical factors are its exact
local magnetic-component view. Raw rank-coupled factor lists are rejected so
that an ambiguous recoupling convention cannot silently produce a wrong
answer on chains longer than two sites.

`ReducedMPOHamiltonian` is a contract for an SU(2)-scalar Hamiltonian: custom
builders are responsible for making the reduced and canonical factor chains
represent the same operator. Small-system tests should compare the canonical
chain with an independently built dense reference before production runs.
The optimizer validates ordered physical legs, physical dimensions, boundary
dimensions, and every adjacent MPO virtual dimension before contraction.

`su2_spin_operator` requires `physical_basis` and an explicit
`fully_reduced=True/False` argument. This prevents a sector dimension from
being misread as either magnetic degeneracy or outer multiplicity. The helper
deliberately rejects outer multiplicities greater than one; define a custom
`ReducedTensorOperator` when the operator acts nontrivially in flavor/copy
space.

For a spin-1/2-only local basis there is one scalar reduced physical label, so
LETTA dependency axes have dimension one and the state reduces to its exact
frontier-MPS backbone. Multi-irrep bases, such as `spatial_orbital()`, retain
nontrivial scalar conditioning axes. In both cases bond storage is organized by
complete multiplets.

Use U(1) instead when only total $2S_z$ is conserved:

```python
u1_zero = AbelianSymmetry(
    physical_charges=(1, -1),
    sector=0,
    moduli=None,
    name="U(1) total 2Sz=0",
)
```

The exact SU(2) path currently requires a `ReducedMPOHamiltonian` whose
canonical local-factor representation is available. Its contractions remain
polynomial in the explicit frontier-MPS dimensions and do not build a full
Hilbert-space state or Hamiltonian. For higher-dimensional orderings, exact
cost can still grow exponentially with graph frontier width. The canonical
local-CG expansion is exact, although it is not yet
a fully recoupled 6j-only contraction engine. Accepted updates rebalance the
state's scalar gauge, and final reported energies use a temporary QR-conditioned
frontier MPS plus an extended-precision boundary contraction. This conditioning
does not change or densely expand the stored variational state.
For local reduced dimensions above `dense_solver_threshold`, `matrix_free=True`
uses exact projected Hamiltonian and norm actions in an N-orthonormal Davidson
solver. Below the threshold, the dense projected generalized eigensolve is
usually faster. Thus a tiny test can show little timing benefit even while its
stored parameter count is already reduced.

## Case packages

- `_letta_for_2d`: compact-coordinate lattice bonds, TFIM builders, and the
  2D comparison example.
- `_letta_for_3d`: 3D orderings, TFIM builders, LETTA entry points, the MPS
  comparison solver, and 3D examples.

Future two-site optimization should be implemented as a sibling package so
it can share state/MPO concepts deliberately without coupling its truncation
rules to the one-site solver.

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

## Case packages

- `_letta_for_2d`: compact-coordinate lattice bonds, TFIM builders, and the
  2D comparison example.
- `_letta_for_3d`: 3D orderings, TFIM builders, LETTA entry points, the MPS
  comparison solver, and 3D examples.

Future two-site optimization should be implemented as a sibling package so
it can share state/MPO concepts deliberately without coupling its truncation
rules to the one-site solver.

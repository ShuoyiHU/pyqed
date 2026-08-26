# VUMPS

This package implements the one-site variational uniform matrix product state
algorithm for an infinite chain with a dense nearest-neighbor Hamiltonian.

## Conventions

Every MPS site tensor has index order

$$
A_{\alpha s \beta}
\quad\longleftrightarrow\quad
\texttt{A[left, physical, right]}.
$$

The mixed-canonical tensors have shapes

$$
A_L,\ A_C,\ A_R \in \mathbb{C}^{D\times d\times D},
\qquad
C \in \mathbb{C}^{D\times D}.
$$

At convergence they satisfy

$$
A_C^s=A_L^s C=C A_R^s,
$$

with the canonical conditions

$$
\sum_s (A_L^s)^\dagger A_L^s=I,
\qquad
\sum_s A_R^s(A_R^s)^\dagger=I.
$$

A rank-four bond Hamiltonian uses the order

$$
h_{s t k l}
=
\langle s,t|h|k,l\rangle,
$$

or it can be supplied as the matrix of shape

$$
(d^2,d^2).
$$

## Example

```python
import numpy as np

from pyqed._vumps import VUMPSOptions, vumps

identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.array([[1.0, 0.0], [0.0, -1.0]])

# H = sum_n (-Z_n Z_{n+1} - 1.5 X_n).
# Each on-site term is split equally between the two adjacent bonds.
h = -np.kron(z, z) - 0.75 * (
    np.kron(x, identity) + np.kron(identity, x)
)

result = vumps(
    h,
    bond_dim=2,
    seed=3,
    options=VUMPSOptions(
        max_iterations=100,
        tolerance=1.0e-10,
        verbosity=1,
    ),
)

print(result.energy)
print(result.residual_norm)
print(result.converged)

AL = result.state.AL
AC = result.state.AC
C = result.state.C
AR = result.state.AR
```

For this example, bond dimension two gives an energy density close to
`-1.6717366239`.

## Iteration

Each iteration performs these operations:

1. Construct the left and right infinite Hamiltonian environments by solving
   regularized transfer equations.
2. Construct matrix-free effective Hamiltonians for the center tensor and
   center matrix.
3. Find their lowest eigenvectors.
4. Use polar decompositions to recover new left- and right-canonical tensors.
5. Measure the canonical-consistency residual

$$
\epsilon_{\mathrm{canonical}}
=
\max\left(
\left\|A_C-A_LC\right\|,
\left\|A_C-CA_R\right\|
\right).
$$

6. Measure the change of the independently optimized center tensors,

$$
\epsilon_{\mathrm{fixed\ point}}
=
\max\left(
\left\|A_C^{\mathrm{new}}-A_C^{\mathrm{old}}\right\|,
\left\|C^{\mathrm{new}}-C^{\mathrm{old}}\right\|
\right).
$$

The reported residual is the maximum of these two values. Requiring the
fixed-point change is important at bond dimension one, where the canonical
factorization residual alone is identically zero. The result is converged when
the combined residual is no larger than the requested tolerance.

## Modules

- `state.py` contains canonical tensors, transfer actions, and
  canonicalization.
- `operators.py` contains Hamiltonian environments and the two effective
  Hamiltonian actions.
- `solver.py` contains eigensolvers, polar gauge matching, convergence
  tracking, and the public `vumps` driver.

## Current scope

The implementation currently assumes:

- one tensor per unit cell;
- a dense two-site nearest-neighbor Hamiltonian;
- a dense MPS bond space without Abelian or non-Abelian block structure.
- an injective MPS with a full-rank transfer fixed point and center matrix.

A one-site unit cell enforces one-site translation symmetry. Models whose
optimal representation has a larger unit cell should use a future multi-site
extension; increasing only the bond dimension does not remove that ansatz
restriction. MPO Hamiltonians and LETTA tensors are not converted implicitly.
Rank-deficient or noninjective states currently raise an error instead of
automatically reducing their bond support.

## Examples

The transverse-field Ising comparison in
[`examples/tfim_comparison.py`](examples/tfim_comparison.py) demonstrates
VUMPS, finite periodic exact diagonalization, exact thermodynamic-limit
energies, and local observables. Its derivation and expected output are
described in [`examples/README.md`](examples/README.md).

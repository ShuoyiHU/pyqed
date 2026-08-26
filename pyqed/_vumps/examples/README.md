# Transverse-Field Ising Example

The example in `tfim_comparison.py` solves the one-dimensional
transverse-field Ising model in three independent ways:

1. one-site VUMPS directly in the thermodynamic limit;
2. sparse exact diagonalization of a finite periodic ring;
3. exact thermodynamic-limit Jordan-Wigner integrals.

Run it from the repository root:

```bash
PYTHONPATH=. python -m pyqed._vumps.examples.tfim_comparison
```

## 1. Hamiltonian

The model is

$$
H
=
-J\sum_n Z_nZ_{n+1}
-g\sum_n X_n.
$$

The VUMPS solver accepts a two-site term. The on-site field is therefore split
equally between the two adjacent bonds:

$$
h_{n,n+1}
=
-JZ_nZ_{n+1}
-\frac{g}{2}
\left(
X_n\otimes I_{n+1}
+I_n\otimes X_{n+1}
\right).
$$

Summing this term over every bond counts each field operator twice with
coefficient one half, reproducing the original Hamiltonian.

The function implementing this construction is
`tfim_bond_hamiltonian`.

## 2. VUMPS calculation

For each requested bond dimension, `vumps_tfim_ground_state` calls
`vumps` with the two-site Hamiltonian. The returned state contains the
left-canonical tensor, center tensor, center matrix, and right-canonical
tensor.

The energy density is evaluated from the physical uniform state represented
by the left-canonical tensor. Its right transfer fixed point satisfies

$$
\rho_R
=
\sum_s A_L^s\rho_R(A_L^s)^\dagger,
\qquad
\operatorname{Tr}(\rho_R)=1.
$$

For a one-site operator, the expectation value is

$$
\langle O\rangle
=
\sum_{s,t}
O_{st}
\operatorname{Tr}
\left[
\rho_R
(A_L^s)^\dagger
A_L^t
\right].
$$

This contraction is implemented by the public
`one_site_expectation` function. The example uses it to calculate the
transverse magnetization:

$$
m_x
=
\langle X_n\rangle.
$$

It also calculates the nearest-neighbor correlation:

$$
C_{zz}
=
\langle Z_nZ_{n+1}\rangle.
$$

## 3. Exact infinite-chain reference

After a Jordan-Wigner and Bogoliubov transformation, the exact ground-state
energy density is

$$
e_0
=
-\frac{1}{2\pi}
\int_0^{2\pi}
\sqrt{
J^2+g^2-2Jg\cos k
}
\,dk.
$$

The transverse magnetization follows either from the transformed ground state
or from the Hellmann-Feynman relation:

$$
m_x
=
-\frac{\partial e_0}{\partial g}
=
\frac{1}{2\pi}
\int_0^{2\pi}
\frac{g-J\cos k}
{\sqrt{J^2+g^2-2Jg\cos k}}
\,dk.
$$

The nearest-neighbor correlation follows directly from the
Hellmann-Feynman derivative with respect to the coupling:

$$
C_{zz}
=
\frac{1}{2\pi}
\int_0^{2\pi}
\frac{J-g\cos k}
{\sqrt{J^2+g^2-2Jg\cos k}}
\,dk.
$$

It also obeys the energy identity

$$
e_0
=
-JC_{zz}
-gm_x,
$$

which provides a useful conceptual cross-check. The example evaluates the
direct integrals with adaptive quadrature. For the correlation integral, it
pairs momenta before quadrature to avoid cancellation when the coupling is
much smaller than the field.

At the critical point,

$$
J=g=1,
$$

the implementation is tested against the known values

$$
e_0
=
-\frac{4}{\pi},
\qquad
m_x
=
\frac{2}{\pi}.
$$

## 4. Finite exact diagonalization

For a ring of length

$$
L,
$$

the exact-diagonalization Hilbert space contains

$$
2^L
$$

computational basis states. In that basis:

- every Ising interaction contributes to the diagonal;
- every transverse-field operator flips one bit.

The example applies this sparse action through a matrix-free
`scipy.sparse.linalg.LinearOperator` and obtains its lowest eigenpair with
`scipy.sparse.linalg.eigsh`. It never stores the exponentially large
Hamiltonian matrix, although it must still store vectors of length

$$
2^L.
$$

For that reason, the example limits exact diagonalization to 20 sites. This
method is exact for the finite ring but differs from the infinite reference
because of finite-size effects.

## 5. Default comparison

The default parameters are

$$
J=1,
\qquad
g=1.5,
\qquad
L=12.
$$

The observed values are:

| Method | Energy/site | Energy error | Transverse magnetization | Magnetization error |
|---|---:|---:|---:|---:|
| Exact infinite | -1.6719262215 | 0 | 0.8773282152 | 0 |
| Periodic ED, 12 sites | -1.6720520571 | 1.258e-4 | 0.8764626452 | 8.656e-4 |
| VUMPS, bond 1 | -1.5625000000 | 1.094e-1 | 0.7500000001 | 1.273e-1 |
| VUMPS, bond 2 | -1.6717366239 | 1.896e-4 | 0.8780762124 | 7.480e-4 |
| VUMPS, bond 4 | -1.6719259669 | 2.547e-7 | 0.8773301392 | 1.924e-6 |

The finite-ring energy can lie slightly below the infinite-chain energy
density because these are different systems. VUMPS instead approximates the
infinite system directly, and its error decreases rapidly as the bond
dimension increases.

Command-line parameters can be changed, for example:

```bash
PYTHONPATH=. python -m pyqed._vumps.examples.tfim_comparison \
  --field 1.0 \
  --sites 14 \
  --bond-dimensions 2 4 8
```

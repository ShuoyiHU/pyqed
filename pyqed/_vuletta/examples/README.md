# Ising Comparison

The example solves the infinite transverse-field Ising chain

$$
H
=
-J\sum_n Z_nZ_{n+1}
-g\sum_n X_n
$$

with three descriptions:

1. the exact Jordan-Wigner thermodynamic solution;
2. direct variational uniform LETTA;
3. ordinary VUMPS.

Run it from the repository root:

```bash
PYTHONPATH=. python -m pyqed._vuletta.examples.tfim_comparison
```

## Hamiltonian passed to the solvers

Both variational solvers accept a two-site term. The field is divided equally
between adjacent bonds:

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

## Analytical reference

The exact energy density is

$$
e_0
=
-\frac{1}{2\pi}
\int_0^{2\pi}
\sqrt{J^2+g^2-2Jg\cos k}
\,dk.
$$

The exact transverse magnetization is

$$
\langle X\rangle
=
\frac{1}{2\pi}
\int_0^{2\pi}
\frac{g-J\cos k}
{\sqrt{J^2+g^2-2Jg\cos k}}
\,dk.
$$

The exact nearest-neighbor correlation is

$$
\langle ZZ\rangle
=
\frac{1}{2\pi}
\int_0^{2\pi}
\frac{J-g\cos k}
{\sqrt{J^2+g^2-2Jg\cos k}}
\,dk.
$$

## Default result

The default parameters are

$$
J=1,
\qquad
g=1.5.
$$

The observed comparison is:

| Method | Stored tensor entries | Transfer bond | Energy/site | Energy error | Transverse magnetization | Magnetization error |
|---|---:|---:|---:|---:|---:|---:|
| Exact infinite | - | - | -1.6719262215 | 0 | 0.8773282152 | 0 |
| VULETTA, bond 1 | 4 | 2 | -1.6666666667 | 5.260e-3 | 0.8888888890 | 1.156e-2 |
| VULETTA, bond 2 | 16 | 4 | -1.6719192766 | 6.945e-6 | 0.8773681144 | 3.990e-5 |
| VULETTA, bond 3 | 36 | 6 | -1.6719261882 | 3.338e-8 | 0.8773285384 | 3.232e-7 |
| VUMPS, bond 2 | 8 | 2 | -1.6717366239 | 1.896e-4 | 0.8780762124 | 7.480e-4 |
| VUMPS, bond 4 | 32 | 4 | -1.6719259669 | 2.547e-7 | 0.8773301393 | 1.924e-6 |

The corresponding nearest-neighbor values are:

| Method | Nearest-neighbor correlation | Error |
|---|---:|---:|
| Exact infinite | 0.3559338987 | 0 |
| VULETTA, bond 1 | 0.3333333331 | 2.260e-2 |
| VULETTA, bond 2 | 0.3558671051 | 6.679e-5 |
| VULETTA, bond 3 | 0.3559333805 | 5.182e-7 |
| VUMPS, bond 2 | 0.3546223053 | 1.312e-3 |
| VUMPS, bond 4 | 0.3559307580 | 3.141e-6 |

## How to read the comparison

LETTA bond dimension and MPS bond dimension are not directly equivalent.
For the spin-half chain,

$$
d=2.
$$

A LETTA tensor with bond dimension

$$
D
$$

has transfer bond

$$
2D
$$

and stored-entry count

$$
4D^2.
$$

An unrestricted MPS with bond dimension

$$
\chi
$$

has stored-entry count

$$
2\chi^2.
$$

At equal transfer bond, the unrestricted MPS is more expressive and gives the
lower variational energy. LETTA stores half as many tensor entries at that
transfer bond. In this example, LETTA bond two is already highly accurate, but
it does not beat unrestricted MPS bond four in absolute accuracy.

These raw counts include normalization and gauge redundancies. The comparison
therefore demonstrates a storage-and-structure tradeoff, not a universal
advantage of LETTA over MPS.

## Bond-dimension continuation

The LETTA objective is nonconvex. A fresh random tensor at a larger bond
dimension can converge to a worse basin even though the variational space is
larger. The default example therefore solves dimensions in order and uses

$$
T_{D'}^{(0)}
=
\operatorname{normalize}
\left[
\operatorname{embed}_{D\rightarrow D'}(T_D)
+\epsilon R
\right],
$$

where

$$
\lVert R\rVert_F=1,
\qquad
\epsilon=0.03.
$$

The embedded tensor preserves the converged lower-dimensional state. The
perturbation activates the added virtual sector and restores a generic
injective transfer problem. To diagnose the raw seed dependence instead, run

```bash
PYTHONPATH=. python -m pyqed._vuletta.examples.tfim_comparison \
    --independent-letta-initializations
```

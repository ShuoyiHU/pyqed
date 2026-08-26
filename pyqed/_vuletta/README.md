# Variational Uniform LETTA

This package implements a direct variational uniform version of the
nearest-neighbor leg-tied tensor ansatz.

Read [vuletta_theory.tex](theory/vuletta_theory.tex) for the complete
derivation. It covers the structured transfer contraction, physical-dependent
gauge freedom, Gram matrix, tangent projector, VU-like stationarity equation,
analytic transfer response, and the implemented numerical steps.
[THEORY.md](THEORY.md) remains as a compact summary.

## Public API

- `UniformLETTA` stores one repeated pair tensor.
- `transfer_data` constructs the thermodynamic transfer fixed points.
- `one_site_expectation` evaluates a local observable.
- `two_site_expectation` evaluates a nearest-neighbor observable.
- `energy_density` evaluates a nearest-neighbor Hamiltonian.
- `energy_and_gradient` evaluates the exact transfer-response derivative.
- `tangent_gram_matrix` constructs the connected uniform-state metric.
- `expand_uniform_letta` embeds a converged state at a larger bond dimension.
- `vuletta` minimizes the energy directly over the LETTA entries.

The tensor order is

$$
(\text{left},\text{previous physical},\text{current physical},\text{right}).
$$

For physical dimension

$$
d
$$

and LETTA bond dimension

$$
D,
$$

the tensor stores

$$
d^2D^2
$$

scalar entries and has structured transfer bond dimension

$$
\chi_{\mathrm{transfer}}=dD.
$$

## Minimal use

```python
import numpy as np

from pyqed._vuletta import VULETTAOptions, vuletta

identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.array([[1.0, 0.0], [0.0, -1.0]])

h = -np.kron(z, z) - 0.75 * (
    np.kron(x, identity) + np.kron(identity, x)
)

result = vuletta(
    h,
    bond_dim=2,
    seed=3,
    options=VULETTAOptions(
        max_iterations=150,
        tolerance=1.0e-8,
    ),
)

print(result.energy)
print(result.residual_norm)
print(result.converged)
```

The default update is a tangent-metric natural-gradient step. The result is
marked converged only when the gauge-invariant tangent residual is below

$$
\texttt{stationarity_tolerance}.
$$

The default LETTA bond dimension is two when neither `bond_dim` nor an initial
state is supplied. If an initial state is supplied, its bond dimension is used
unless an explicit, matching `bond_dim` is also given.

## Convergence diagnostics

`max_iterations` is an upper bound. The default natural-gradient solver can
stop earlier because:

- the tangent-metric residual reaches `stationarity_tolerance`;
- the Armijo line search reaches the floating-point energy floor;
- the function-evaluation budget is exhausted.

The derivative is analytic. It differentiates the transfer fixed points with
the reduced resolvent

$$
\mathcal R
=
Q(1-\mathcal E)^{-1}Q.
$$

This needs one transfer eigensystem per accepted iteration, instead of one
eigensystem for every perturbed coordinate. `gradient_method="finite_difference"`
with `update_method="lbfgs"` remains available as a reference check.

`residual_norm` is

$$
\epsilon_{\mathrm{tan}}
=
\sqrt{g_{\mathrm{coord}}^{\mathsf T}
\mathsf G^+g_{\mathrm{coord}}},
$$

where the pseudoinverse removes normalization, projective phase, and LETTA
gauge directions. `gradient_norm` is retained as the Euclidean
parameter-sphere norm for diagnostics, but it is coordinate and gauge
dependent. `metric_rank` reports the retained physical tangent dimension.

## Ising comparison

Run

```bash
PYTHONPATH=. python -m pyqed._vuletta.examples.tfim_comparison
```

The example compares VULETTA with exact infinite-chain values and ordinary
VUMPS. See [`examples/README.md`](examples/README.md) for the formulas,
observables, output table, and interpretation.

The comparison uses bond-dimension continuation by default. After solving at
bond dimension `D`, it embeds that tensor into the next larger virtual space
and adds a normalized 3% perturbation. The perturbation activates the new
sector and avoids an exactly rank-deficient padded tensor. This substantially
reduces seed-dependent trapping at larger bond dimension.

Independent random initializations can still be requested with

```bash
PYTHONPATH=. python -m pyqed._vuletta.examples.tfim_comparison \
    --independent-letta-initializations
```

## Important distinction

The structured MPS tensor is a contraction identity:

$$
B^s_{(\alpha,p),(\beta,q)}
=
\delta_{q,s}T^{p,s}_{\alpha,\beta}.
$$

The solver never optimizes

$$
B.
$$

It optimizes only the LETTA tensor

$$
T.
$$

Therefore this implementation does not optimize an MPS and project it back to
LETTA. The LETTA constraint is present throughout every objective evaluation.

## Current limitations

- one repeated nearest-neighbor pair tensor;
- one-site translation invariance;
- dense transfer eigensolvers;
- dense tangent Gram matrices and reduced resolvents;
- injective transfer operators only;
- no long-range leg sharing or symmetry blocks;
- nonconvex optimization; continuation reduces but cannot mathematically
  eliminate seed dependence.

The paper introducing LETTA does not derive a direct variational optimizer.
This package should therefore be treated as a tested research prototype, not
as a reproduction of a published LETTA optimization algorithm.

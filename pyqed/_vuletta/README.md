# Variational Uniform LETTA

This package implements a direct variational uniform version of the
nearest-neighbor leg-tied tensor ansatz.

The implementation uses the exact structured-MPS contraction identity while
optimizing only the tied LETTA entries. For the nearest-neighbor chain, the
shared physical state supports an exact conditional mixed-canonical gauge.

## Public API

- `UniformLETTA` stores one repeated pair tensor.
- `ConditionalCanonicalLETTA` stores `TL`, `TC`, `TR`, and conditioned centers.
- `conditional_canonicalize` constructs the exact NN conditional gauge.
- `transfer_data` constructs the thermodynamic transfer fixed points.
- `one_site_expectation` evaluates a local observable.
- `two_site_expectation` evaluates a nearest-neighbor observable.
- `energy_density` evaluates a nearest-neighbor Hamiltonian.
- `energy_and_gradient` evaluates the exact transfer-response derivative.
- `tangent_gram_matrix` constructs the connected uniform-state metric.
- `conditional_tangent_direction` constructs horizontal whitened coordinates.
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

The default update is a conditional-canonical tangent step. The result is
marked converged only when both the horizontal tangent residual and the
conditional-canonical residual are below

$$
\texttt{stationarity_tolerance}.
$$

The default LETTA bond dimension is two when neither `bond_dim` nor an initial
state is supplied. If an initial state is supplied, its bond dimension is used
unless an explicit, matching `bond_dim` is also given.

## Convergence diagnostics

`max_iterations` is an upper bound. The default conditional-canonical solver can
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
eigensystem for every perturbed coordinate. The old dense connected-Gram path
remains available with `update_method="natural_gradient"`, and
`gradient_method="finite_difference"` with `update_method="lbfgs"` remains a
finite-difference reference.

For the default path, let $N_q$ span the orthogonal complement of the
left-canonical conditioned block and let $\rho_q$ be the corresponding
right-metric block. `residual_norm` is

$$
\epsilon_{\mathrm{tan}}^2
=
\sum_q
\left\|N_q^\dagger g_q\rho_q^{-1/2}\right\|_F^2,
$$

which equals the dense connected-Gram pseudoinverse residual but never forms
that singular matrix. `gradient_norm` is retained as a Euclidean diagnostic,
but it is coordinate and gauge dependent. `metric_rank` and
`reduced_dimension` report the active real tangent dimension.

## Conditional mixed-canonical gauge

Write the pair tensor as $T^{p,q}_{\alpha\beta}$, where $q$ is the shared
physical state on the right edge. Its exact gauge freedom is

$$
T^{p,q}\longrightarrow G_p^{-1}T^{p,q}G_q.
$$

The nearest-neighbor copy structure makes the canonical conditions sectorwise:

$$
\sum_p (T_L^{p,q})^\dagger T_L^{p,q}=I_D,
\qquad
\sum_q T_R^{p,q}(T_R^{p,q})^\dagger=I_D.
$$

The conditioned centers obey

$$
T_C^{p,q}=T_L^{p,q}C_q=C_pT_R^{p,q}.
$$

Every active variation satisfies

$$
\sum_p (T_L^{p,q})^\dagger X^{p,q}=0
\qquad\text{for every }q.
$$

Thus normalization, projective phase, and all physical-dependent virtual gauge
directions are absent from the active coordinates. Full conditioned transfer
support is required by the optimizer; `canonical_rcond` controls its numerical
rank threshold.

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
- dense transfer eigensolvers and analytic-gradient reduced resolvents;
- injective transfer operators only;
- no long-range leg sharing or symmetry blocks;
- nonconvex optimization; continuation reduces but cannot mathematically
  eliminate seed dependence.

The paper introducing LETTA does not derive a direct variational optimizer.
This package should therefore be treated as a tested research prototype, not
as a reproduction of a published LETTA optimization algorithm.

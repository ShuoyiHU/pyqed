# Selectable physical-norm compression

Both `LETTADMROptions` and `LETTATwoSiteOptions` accept the same immutable
`MetricCompressionOptions` object. The implementation is in
`pyqed/_letta_compression.py` and does not import benchmark solvers.

```python
from pyqed._letta_one_site_opt import (
    MetricCompressionOptions, LETTADMROptions, letta_dmrg,
)
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg

compression = MetricCompressionOptions(
    solver="variable-projection",  # "als", "joint-ls", "grassmann-newton"
    max_iterations=100,
    tolerance=1e-10,
    max_workspace_mb=128,
)

cbe_options = LETTADMROptions(
    cbe_enabled=True, cbe_selector="shrewd", compression=compression,
)
two_site_options = LETTATwoSiteOptions(
    split_method="metric-energy", compression=compression,
)
# Given an MPO and a compatible initial LETTA state:
# cbe = letta_dmrg(mpo, state=initial.copy(), options=cbe_options)
# two = letta_two_site_dmrg(
#     mpo, state=initial.copy(), bond_dim=max(initial.bond_dimensions),
#     options=two_site_options,
# )
```

The default solver remains ALS. Existing `metric-als` and `metric-als-energy`
split names are aliases for `metric` and `metric-energy`: the `compression`
option determines the norm solver. `conditional-svd` bypasses weighted fitting;
`energy-refined` bypasses weighted fitting and performs energy refinement only.

## What is minimized?

Let the target be the tensor before compression, and let the bilinear merge
map combine the two restricted-bond factors. The environment is fixed during
this subproblem. Its overlap metric is positive semidefinite in exact arithmetic.

$$
\mathcal L(X,T)
=\operatorname{vec}(\mathcal M(X,T)-A)^\dagger
 N\operatorname{vec}(\mathcal M(X,T)-A).
$$

The numerical optimizer uses the supported square root, with small metric
eigenvalues removed according to the calling solver's `metric_tolerance`.
It applies this root blockwise; it never constructs a dense full square root.

$$
N_{\rm supported}=S^\dagger S,\qquad
r(X,T)=S\operatorname{vec}(\mathcal M(X,T)-A).
$$

The best candidate is retained using the **original physical metric**. Both
that loss and the supported loss are reported for nonlinear solves, because
near-null metric directions can make the two numerically different.

For an ordinary matrix the merge is matrix multiplication. For a LETTA pair,
the same physical indices can appear in both tensors and are identified by the
merge, rather than summed twice. Factor charts and balancing respect these
shared indices and Abelian charge masks. The full pair metric is retained,
even if it couples different shared-physical configurations.

In shrewd CBE, compression acts on the expanded active-site tensor in its
one-site environment metric; its transfer factor is absorbed into the neighbor.
Shared-physical sectors are split as in the existing conditional CBE trim.
Exact/separable metric SVD and negligible-loss shortcuts remain in use.
In exact CBE and two-site optimization, the target is the merged pair tensor.
These options do **not** change CBE residual selection, the local energy
eigensolver, subsequent energy refinement, or the final energy acceptance rule.
Reduced SU(2) two-site optimization has its own reduced truncation algorithm;
nondefault compression options are rejected there. Reduced SU(2) CBE was
already unsupported.

## ALS

With one factor fixed, the other solves a linear weighted least-squares problem:

$$
X_{k+1}=\arg\min_X\mathcal L(X,T_k),\qquad
T_{k+1}=\arg\min_T\mathcal L(X_{k+1},T).
$$

The existing matrix-free LSMR implementation remains the default and fallback.
Its outer budget normally comes from `cbe_refinement_max_iterations` (CBE) or
`truncation_max_iterations` (two-site). Explicit common overrides are available:

```python
compression = MetricCompressionOptions(
    solver="als", als_max_iterations=40, lsmr_max_iterations=400,
)
```

The same overrides apply when a nonlinear method falls back to ALS. Without
an override, CBE uses 40 LSMR iterations per linear subproblem; two-site uses
its existing size-dependent 50–1000 iteration cap. ALS can stop at a stationary
point where improving both factors together would help.

## Joint-factor least squares: `joint-ls`

Optimize both factors simultaneously in real coordinates. Complex factors
are represented by separate real and imaginary coordinates. The exact
residual Jacobian follows directly from bilinearity:

$$
\delta r=S\operatorname{vec}\bigl(
\mathcal M(\delta X,T)+\mathcal M(X,\delta T)\bigr).
$$

A trust-region reflective least-squares solver approximately minimizes the
local linear-residual model and tests the resulting step against the nonlinear
objective:

$$
\min_{\delta z}\frac12\|r+J\delta z\|^2,
\qquad \|\delta z\|\ \text{restricted by the trust region}.
$$

This is a Gauss–Newton method, not full Newton. It can move both factors at
once, but retains factor-basis redundancy. Balancing the input and output
reduces bad scaling; it does not remove all redundant coordinates during the
solve. Charge-forbidden entries are excluded from its coordinates.

## Variable projection: `variable-projection`

Eliminate one factor by solving its linear least-squares problem at every
trial value of the other. Build a weighted design matrix for the linear factor:

$$
K(X)t=S\operatorname{vec}\mathcal M(X,T),\qquad
b=S\operatorname{vec}(A),\qquad t_*(X)=K(X)^+b.
$$

The remaining objective is

$$
\min_X\frac12\|r_*(X)\|^2,\qquad
r_*(X)=K(X)t_*(X)-b.
$$

An SVD solves the eliminated linear problem, with relative singular-value
cutoff `pinv_tolerance`. The implementation optimizes the smaller of the left
and right **subspace coordinate counts**. In each admissible shared-index/charge
block it removes basis redundancy using a local Grassmann chart:

$$
X(Z)=Q_0+Q_\perp Z.
$$

The columns of the two Q blocks are orthonormal, and their cross inner product
is zero. Each trial subspace gets its own optimally fitted other factor. A
trust-region reflective least-squares method uses the exact reduced residual
Jacobian. In the real coordinates used internally, at locally constant rank:

$$
\delta r_*=(I-KK^+)\,\delta K\,t_*
 -(K^+)^T(\delta K)^T r_*.
$$

The second term matters when the residual is nonzero. Eliminating the linear
factor avoids repeatedly optimizing a factor that can already be solved
exactly, but constructing and factorizing the design matrix costs more than
one cheap ALS step.

## Balanced Grassmann Newton: `grassmann-newton`

This uses the same factor elimination and smaller-side subspace chart, but
uses the **full reduced objective Hessian**, including residual curvature,
in a trust-region Newton method. For half the squared residual norm, define
all the following in real coordinates:

$$
J_X[:,i]=K_i t_*,\qquad K_i=\frac{\partial K}{\partial z_i},
\qquad B_i=J_X[:,i]^T K+r_*^T K_i.
$$

$$
g=J_X^T r_*,\qquad
H=J_X^TJ_X-B(K^TK)^+B^T.
$$

The implementation forms the Schur-complement term from the SVD of the design
matrix, avoiding an explicit inverse of the normal equations. `trust-exact`
uses this Hessian in a trust-region subproblem, which can handle indefinite
curvature. It is not an unconstrained Newton step that always accepts an
inverse-Hessian direction. Residual-Jacobian and Hessian formulas are checked
against finite differences for real and complex problems.

This is a **local fixed-chart Grassmann method**. It does not recenter the
chart during a solve, and subspaces orthogonal to the chart origin are not
represented at finite coordinates. Changing pseudoinverse rank also breaks
the smooth constant-rank assumptions at that transition; observed inner ranks
are reported. Neither method guarantees the global rank-constrained minimum.

## Balancing and numerical safeguards

A factor basis can become enormous while its partner becomes tiny without
changing the represented product. Within each admissible block, QR followed
by a bond-sized SVD gives balanced factors:

$$
XT=U\Sigma V^\dagger,\qquad
X_{\rm bal}=U\Sigma^{1/2},\qquad
T_{\rm bal}=\Sigma^{1/2}V^\dagger.
$$

The implementation obtains this factorization through small QR/SVD cores,
without forming a new full pair tensor for balancing. Zero padding preserves
the requested bond size. Balancing is kept only when the physical loss remains
within roundoff of the best fit.

- The initial fit and every finite valid trial compete on original-metric loss.
  The optimizer's last iterate is not automatically the returned fit.
- Trials with invalid physical norm or excessive factor-norm product are not
  retained. `max_factor_norm_growth=100` limits that product to its initial
  scale times the square of this setting. This is a numerical safeguard, not
  a bound on the final energy error.
- Dense Jacobians, derivative design matrices, SVDs and Newton work are
  conservatively estimated before chart allocation. Exceeding
  `max_workspace_mb` runs matrix-free ALS and records the reason. This bounds
  the estimated nonlinear workspace, not the total Python process memory or
  existing environment-contraction storage.
- Nonfinite trials or linear-algebra failures trigger ALS fallback, while
  retaining any better valid fit already found. Invalid user options raise
  errors. Exhausting the nonlinear budget returns the best fit, without
  silently claiming convergence.
- `max_iterations` limits residual evaluations for the two least-squares
  methods and trust-region iterations for Newton. These are different units.
  `tolerance` controls nonlinear gradient, step and/or objective stopping;
  it is not a guaranteed distance from the global optimum.

Inspect per-update `compression_diagnostics` in two-site histories or
`cbe_compression_diagnostics` in CBE histories. They include requested/used
solver, initial/final loss, fallback reason, workspace estimate and nonlinear
optimizer status. Optimizer success/optimality describes its terminating
iterate; the retained and balanced best iterate may differ. CBE's exact
separable shortcuts report `used_solver="separable-svd"`; negligible-loss
sectors require no nonlinear solve.

A better physical-norm approximation does not imply lower energy than another
compression method. All methods still require the existing energy refinement
and acceptance safeguards. Long-sweep energy and runtime comparisons must use
identical initial states and budgets before selecting a new default.

## Verification (2026-09-26)

163 focused tests pass: 42 production-solver tests plus 121 existing tests.
Coverage includes real/complex residual Jacobians and reduced Hessians, known
weighted-SVD minima, singular metrics, full coupled shared-index metrics,
Abelian masks, rank padding, numerical/workspace fallbacks, and live updates
through two-site, exact CBE and shrewd CBE.

Short Bose–Hubbard smoke comparisons used the model defaults (hopping 1,
interaction 4, chemical potential 2, local occupations 0–2), seed 731,
identical initial states, default nonlinear budgets and one BLAS thread.
Sweeps here are directional passes. These runs are validation, not convergence
or repeated timing benchmarks.

| Case | Algorithm | Compression | Final energy | Seconds | ALS fallbacks |
|---|---|---|---:|---:|---:|
| 2×2, D=2, 5 sweeps | cbe | als | -11.6956784155 | 0.535 | 0 |
| 2×2, D=2, 5 sweeps | cbe | variable-projection | -11.6956983123 | 0.505 | 0 |
| 2×2, D=2, 5 sweeps | cbe | joint-ls | -11.6956983123 | 0.583 | 0 |
| 2×2, D=2, 5 sweeps | cbe | grassmann-newton | -11.6956983123 | 0.492 | 0 |
| 2×2, D=2, 5 sweeps | two-site | als | -11.6970448642 | 0.650 | 0 |
| 2×2, D=2, 5 sweeps | two-site | variable-projection | -11.6970437974 | 0.719 | 0 |
| 2×2, D=2, 5 sweeps | two-site | joint-ls | -11.6970293248 | 1.109 | 0 |
| 2×2, D=2, 5 sweeps | two-site | grassmann-newton | -11.6970437974 | 0.762 | 0 |
| 3×2, D=3, 2 sweeps | cbe | als | -18.7726897947 | 2.455 | 0 |
| 3×2, D=3, 2 sweeps | cbe | variable-projection | -18.7724893227 | 2.389 | 0 |
| 3×2, D=3, 2 sweeps | cbe | joint-ls | -18.7712794030 | 2.554 | 0 |
| 3×2, D=3, 2 sweeps | cbe | grassmann-newton | -18.7724893227 | 2.510 | 0 |
| 3×2, D=3, 2 sweeps | two-site | als | -18.7663185565 | 3.685 | 0 |
| 3×2, D=3, 2 sweeps | two-site | variable-projection | -18.7673046980 | 3.771 | 4 |
| 3×2, D=3, 2 sweeps | two-site | joint-ls | -18.7754750956 | 6.631 | 0 |
| 3×2, D=3, 2 sweeps | two-site | grassmann-newton | -18.7673046980 | 3.784 | 4 |

On the 3×2 two-site run, variable projection and Newton used ALS on four
of ten fits because of the default 128 MiB workspace estimate. Their results
therefore describe a mixed nonlinear/ALS run, not ten nonlinear fits. All
recorded sweep energies decreased. The ordering of the methods already
differs between cases and algorithms; no universal energy/runtime winner is
claimed.

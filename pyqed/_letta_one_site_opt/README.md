# Finite LETTA one-site optimization

This package separates the reusable one-site optimization machinery from
dimension-specific lattice models.

## Controlled bond expansion

LETTA-CBE is an optional per-site optimization mode.  Its strict `shrewd` path
first streams cheap Hamiltonian-weighted left and right half environments to
preselect a small parent space.  Inside that space, the final contraction forms
the physical residual covector `(H - E N) psi`, raises it with the temporary
expanded one-site overlap metric, and removes the current one-site tangent in
that same metric.  The selected basis enlarges the active bond only for the
next one-site generalized eigensolve.  This expanded eigensolve is the central
operation: selection changes the variational subspace but does not perturb the
pre-optimization state.  The optimized bond is returned to its original width
by fixed-rank ALS in the active one-site LETTA norm, with a streamed scalar
energy safeguard and ordinary one-site fallback.  Each attempted expansion is
also compared with an ordinary one-site candidate from the same pre-update
state. By default (`cbe_baseline_guard_fraction=0.0`), the refined CBE candidate
must have strictly lower energy than both that ordinary candidate and the
pre-update state. Ties select the ordinary update; the general energy-increase
tolerance does not permit a higher-energy CBE candidate in this mode.
Set `cbe_baseline_guard_fraction=1.0` to require only strict descent from the
pre-update state. Intermediate fractions retain the exploratory allowance;
`0.2` reproduces the previous 20%-loss acceptance rule for an ablation.

    options = LETTADMROptions(
        matrix_free=True,
        cbe_enabled=True,
        cbe_selector="shrewd",
        cbe_expansion_dimension=1,
        # Optional; automatic is min(D + 2 * deltaD, parent sizes).
        cbe_preselection_dimension=2,
    )
    result = letta_dmrg(hamiltonian, state=state, options=options)

After trimming, CBE relaxes both the trimmed factors and the incumbent by
alternating energy minimization, then keeps the lower-energy result. When this
local relaxation changes the energy by at most
`cbe_coupled_energy_threshold * max(1, abs(E))` (default threshold `1e-6`), a
bounded coupled correction can address directions missed by separate A/B
updates. It whitens the supported metric of each factor before solving the
joint tangent metric, and activates only when the joint residual exceeds the
independent residual by `cbe_coupled_activation_ratio` (default `5`). Every
accepted backtracking step lowers the streamed physical energy at fixed rank.
Early coupled steps can change the basin of attraction, so the energy gate is
part of the algorithm, not just a performance shortcut.

The correction uses factor overlaps and fresh one-site Hamiltonian actions;
it does not construct a merged pair, pair Hamiltonian, or physical Jacobian.
Its dense parameter-space solve is an additional cost, separate from the
streamed selector scaling below. It is skipped when the two factors together
have more than `cbe_coupled_max_parameters=512` entries, bounding its quadratic
storage and cubic solve cost. Such bonds continue with ordinary CBE relaxation.
Use `cbe_coupled_max_iterations=0` to disable it for an ablation (default `8`);
`cbe_coupled_metric_tolerance=1e-12` controls supported metric directions.
Update diagnostics report `cbe_coupled_iterations`,
`cbe_coupled_accepted_steps`, and `cbe_coupled_hamiltonian_applications`,
including work on the discarded candidate. The last count is included in the
total Hamiltonian applications; scalar energy contractions remain timing work.

The `exact` selector explicitly constructs the pair metric and tangent
Jacobian and remains the default correctness oracle.  The active `shrewd`
selector performs opposite-half canonicalization, weighted preselection, and
metric-projected physical-residual selection using sparse MPO transitions and
one-site overlap blocks.  Its complete update invokes no pair action, pair
metric, merged pair tensor, pair Rayleigh quotient, or pair trim.  Random
bond-expansion noise is not part of either selector.

The deterministic `cbe_scaling` audit profiles the sparse-Hamiltonian
contraction graphs with `opt_einsum`.  For both sweep directions, those streamed
contractions have no larger asymptotic exponent than the ordinary one-site
action in bond dimension, have the same local-physical exponent while the pair
action has one additional power, and are linear in the number of sparse MPO
paths.  The physical-residual correction additionally constructs and solves a
temporary expanded one-site block metric of width `D + p`; it adds one-site
metric work, but no two-site vector space or physical-dimension exponent.
Timing is reported only as supporting evidence; contraction shapes and
operation counts are the proof.
It currently supports nonsymmetric MPO runs with exact, site-granularity
boundary environments.  The derivation, diagnostics, and cost comparison are
in `letta_cbe_theory.tex`.  A reproducible four-way comparison of one-site,
exact CBE, shrewd CBE, and two-site LETTA is available as the
`cbe_convergence` benchmark module.

Eight independently click-runnable condensed-model comparisons (Ising, XXZ
Heisenberg, truncated Bose--Hubbard, and spinful Fermi--Hubbard, each in 1D and
2D) and a consolidated launcher are documented in
[`benchmarks/README.md`](benchmarks/README.md).  Every comparison starts all
five solvers from the same physical state and reports an exact reference when
the requested Hilbert-space cutoff permits it.

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

Two-site optimization lives in the sibling _letta_two_site_opt package.  The
exact CBE oracle reuses its shared-physical-axis pair algebra.  The strict
shrewd path stays in one-site parent spaces, and enabling or disabling CBE does
not change the legacy one-site path.

## Selectable compression solvers

Both CBE trimming and two-site norm compression accept `compression=MetricCompressionOptions(solver="variable-projection")`. Other choices are `"als"`, `"joint-ls"`, and `"grassmann-newton"`. ALS remains the default. See the [compression solver guide](../_letta_two_site_opt/COMPRESSION.md) for examples, equations, budgets, diagnostics, and limitations.
## CBE numerical recovery

CBE steps are transactional, including the gauge transformation. After each
step, a fresh whole-network contraction checks the norm and energy. A numerical
failure restores the pre-step tensors, clears trial canonical certificates,
rebuilds the environments, and retries the active site with ordinary one-site
optimization. The next bond attempts CBE again. This adds fresh contractions
to prioritize correctness over sweep time.

If a gauge alone fails, the validated one-site update is retained without that
gauge. If the ordinary update also fails validation, the last valid tensors
are retained. `cbe_recovery_reason` records the event and
`cbe_recovery_rejected` distinguishes an unresolved ordinary retry from a
successful recovery. A sweep containing recovery cannot establish convergence.
Invalid options and incompatible tensor shapes still raise their original
errors rather than being disguised as numerical failures.

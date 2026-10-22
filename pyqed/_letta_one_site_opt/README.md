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

## Spin-sensitive SU(2) ties

See [SPIN_FUSION.md](SPIN_FUSION.md) for exact adjacent-pair blocking that
exposes singlet/triplet and multiplicity labels to the ties, its matched
blocked-MPS control, and restrictions. This is an opt-in rung-blocked
extension; it does not change the orbital ansatz.

### Reduced CBE sector-support safeguard

The exact-residual LETTA adaptation now retains at least one multiplet in every
incumbent bond sector during tied CBE compression. Remaining multiplets are
allocated by the existing weighted singular-value rule, followed by the unchanged
physical-metric compression and alternating energy refinement. The total bond
budget and strict one-site energy acceptance guard remain enforced. An infeasible
support requirement rejects the candidate through the existing transactional fallback.
This additional constraint addresses sector extinction observed for the doped 2x4
Hubbard U/t=4 benchmark. It is an implementation-specific safeguard, not a claim
from Gleis et al., Phys. Rev. Lett. 130, 246402 (2023),
https://doi.org/10.1103/PhysRevLett.130.246402. It can exclude a better sector
allocation and does not guarantee global fixed-D convergence. Ordinary two-site
MPS truncation and the separate MPS-CBE selector are unchanged.

### Reduced CBE residual allocation

Open-boundary SU(2) LETTA CBE and two-site LETTA use the same
`compress_reduced_pair` physical-metric factor solver. CBE uses this solver
first to fit expansion directions to the tangent-orthogonal Hamiltonian
residual, then to compress the state obtained by its expanded one-site solve.
Two-site LETTA compresses its optimized pair eigenvector. Both state-compression
steps therefore solve the same kind of constrained metric fit, with different
target states. CBE now caches residual fits by intermediate irrep and rank, because
those blocks are orthogonal in the open-chain physical metric. Increasing one
sector's rank does not require refitting every already-selected sector. Each
rank profile uses two independent starts and, when available, a continuation
from the previous rank. The unchanged outer metric root is reused across
fits. Ring metrics retain their separate allocation path; sector separation
is not assumed there.

This is a LETTA adaptation of the residual expansion idea in Gleis, Li, and
von Delft, Phys. Rev. Lett. **130**, 246402 (2023),
https://doi.org/10.1103/PhysRevLett.130.246402. It does not implement their
shrewd MPS selector or inherit its single-site cost claim. QR/SVD factor solves
follow Trefethen and Bau, *Numerical Linear Algebra* (SIAM, 1997),
https://doi.org/10.1137/1.9780898719574. Greedy allocation uses measured metric
loss gains; nonlinear rank fits can still be nonstationary or locally optimal.
Their status is reported per sector. Neither caching nor an energy guard
certifies global fixed-D convergence.


The shared tied-pair compression now ranks retained multiplets in the physical
QR environment metric, including whole-spin-multiplet weights, before fitting
legal LETTA factors. Raw-coordinate singular weights are gauge dependent and
must not determine this allocation. This uses the Schmidt construction in the
unrestricted frontier space; it is a heuristic allocation for LETTA, not a
proof of optimal tied-factor compression. The subsequent metric fit, energy
refinement, and energy acceptance guard remain necessary. The gauge-invariance
regression explicitly rescales an external virtual bond without changing the
physical state. This is the same QR/SVD formulation cited above, not a new
single-site-cost guarantee for CBE.


Open-boundary tangent projection also factors independent middle-irrep blocks
separately. All blocks use one global singular-value cutoff, so this computes
the same truncated SVD projector as the full tangent matrix, with reduced
workspace. Stored singular vectors and the largest SVD workspace count toward
the configured limit. If that bound is exceeded, matrix-free projection remains
the fallback. Neither the projection iteration cap nor its tolerance is reduced.


### Compiled frontier routing

Reduced LETTA frontier index construction uses `frontier_indices.c`, a C11
integer implementation shared by one-site, CBE, and two-site environments.
This replaces Python assignment enumeration and per-entry indexing without
changing tensor ordering, sector selection, numerical tolerances, or precision.
It is an implementation of the existing exact frontier map, not a new
approximation or a replacement for the CBE selector/compression algorithm.

A C compiler (`CC`, default `cc`) is required on first use. The loader builds
with `-O3` and no fast-math flags in the system temporary directory, keyed by
source contents and compiler command. macOS and Linux are supported. Compilation
failures are reported; they do not silently change the engine. The source is
included in package data. Integer maps use 64-bit indices and validate size
limits before entering C. These changes do not yet move all contraction and
compression operations out of Python.


### Compiled reduced Hamiltonian actions

`ReducedEnvironmentChain.prepare_local_action` and `pair_action` prepare
`SectorOperator` schedules and execute the complete transfer sum in C-generated
BLAS loops. The existing reduced transfer kernels supply all recoupling data.
Only exactly zero kernel entries are omitted. The pair kernel contracts the
shared MPO index once during preparation and retains the existing intermediate
irrep normalization. This does not construct a determinant-space Hamiltonian or
change the variational ansatz. It is an exact algebraic contraction rewrite,
with possible floating-point differences from changed contraction order.

Schedules copy fixed boundary matrices. Pair schedules are invalidated by
`replace_sites`; a prepared local action is immutable and must be rebuilt after
site/environment changes, as before. `_pair_action_reference` is retained for
validation only. Tests compare real/complex actions, all three tested spin
sectors, and replacement-induced invalidation. Cython 3 and a C compiler are
required on first use, as documented for the shared metric engine.

### Exact overlap-factor assembly

Local QR overlap factors now assemble their scattered tensor products in a
C-generated loop (`overlap_factor.py`, shared contraction extension). Repeated
scatter routes accumulate in their original order, including complex-valued
factors. The existing QR, SVD, column equilibration, singular-value cutoffs, and
local eigensolver settings are unchanged. This is an implementation rewrite of
the existing overlap map, not a new approximation or a different variational
space. The reference contraction and its adjoint are checked in
`tests/test_letta_overlap_factor.py`.

Frontier embeddings retain one index map per site, keyed by ordered block
shapes, physical labels, and the neighborhood. Tensor values are never cached.
A layout change replaces that site's map. Each embedding also retains its
physical-configuration route geometry; state-dependent QR/SVD factors remain
fresh. Both caches are owned by the current frontier/embedding objects rather
than a global table. Tests cover changed values, changed layouts, and agreement
between cached and freshly constructed metric factors.

The frontier-owned cache has an estimated 64 MiB budget (including an allowance
for coordinate-route arrays). Least-recently-used maps are evicted; an oversized
map is used without retaining it in the cache. Eviction only triggers exact
recomputation and never truncation. Active coordinate objects may still own
maps, so this is a cache bound, not a bound on total process memory.


### Compact frontier storage and streamed environments

Wide lattices can have small LETTA tensors but a very large auxiliary MPS:
its frontier axes contain repeated physical-label assignments and structural
zeros. `ReducedFrontier.to_mps(..., compact=True)` stores the exact scatter
entries instead. QR and raw environment contractions materialize the current
block or site, then release it. Canonical Hamiltonian environments consume
outer QR cores immediately instead of retaining a second complete chain.
No bond truncation, extra singular-value cutoff, reduced precision, or smaller
inner iteration budget is introduced. Small pair problems retain the ordinary
path below a 64 MiB estimated dense-chain payload to avoid sparse bookkeeping.

This adapts sector QR to the LETTA scatter representation; see Trefethen and
Bau, *Numerical Linear Algebra* (SIAM, 1997), DOI:10.1137/1.9780898719574.
It does not change the variational ansatz. Exactly constant tie axes may be
removed from an expectation-value view, never from the optimized tensor or its
tangent space. Rank-deficient optimization can be sensitive to floating-point
contraction order: direct action checks and full-sweep comparisons are both
required. Compact storage is not a bound on total memory: pair vectors, QR
workspaces, and boundary environments can still be large on wide lattices.

### Moving QR boundaries for gauge conditioning

`GaugeQRBoundary` contracts only the side required by a conditional gauge.
Initial conditioning carries that boundary to the next bond instead of
reconstructing both sides of the auxiliary chain at every cut. The gauge changes
the last processed tensor, so its QR factor is recomputed from the unchanged
incoming boundary before reuse. This retains the full-rebuild contraction order
and uses the same SVD support threshold and invertible gauge transformation.
Only R is computed; unused orthonormal Q tensors are not formed.

For N sites, initial conditioning performs at most 2(N−1) site contractions,
compared with (N−1)² in the former two-sided reconstruction. One-site, two-site,
and CBE drivers also carry a directional boundary between gauge shifts.
A value/layout fingerprint detects changed tensors within the cached side,
including in-place changes; such changes and reverse traversal trigger a rebuild.
Local updates outside that side do not invalidate it. Failed gauge updates restore
both tensors and reset the boundary.

This is an exact-storage/reuse adaptation of sector QR, following Trefethen and
Bau, *Numerical Linear Algebra* (SIAM, 1997), DOI:10.1137/1.9780898719574.
It adds no truncation, approximation, or reduced iteration budget. It does not
remove the exponential width dependence of a single dense frontier QR, and its
gauge-stage speedup should not be presented as a whole-sweep speedup.

### Moving canonical boundaries

The one-site sweep context reuses QR factors and canonical Hamiltonian boundaries
for unchanged prefixes and suffixes. The contractions and QR convention match
fresh construction; no additional truncation or precision reduction is applied.
The boundary cache is limited to 256 MiB of numerical arrays and evicts entries
when necessary. This is a cache budget, not a total process memory limit.
Value signatures detect in-place tensor changes, and sector/layout changes are
part of the signatures. The Hamiltonian is fixed during a sweep context.

Independent QR-conditioned energy checks remain active. Repeated checks of an
identical tensor state reuse their previously computed scalar value; a gauge,
normalization, or tensor-value change requires a fresh check. This does not
replace independent acceptance checks with an unconditioned local metric.
The numerical QR formulation follows Trefethen and Bau, *Numerical Linear
Algebra*, SIAM (1997), doi:10.1137/1.9780898719574. Reuse is an implementation
adaptation and adds no convergence or global-optimality guarantee.

### Local coordinate reuse

Least-squares correction factors cache their fixed QR decompositions for the
lifetime of the local coordinate object; rebuilding overlap factors invalidates
the cache. Orthogonal expansion and adjoint retain their established matrix
product sequence. A compiled replacement was benchmarked but not adopted:
it did not provide a reliable CBE speedup and changed nonlinear trajectories.

### Value-checked energy reuse during CBE

CBE's ordinary and expanded updates, and fixed-bond energy refinement, reuse
an independently QR-conditioned energy only when the tensor-value signature
is unchanged. The cache contains one signature and scalar, is scoped to a
fixed Hamiltonian, and is discarded after the update/refinement. In-place
changes, normalization, and changed bond layouts invalidate it. Physical
acceptance checks still contract changed states independently; local Ritz
values do not replace those checks. This optimization does not change the
normalization path, expansion budget, compression tolerance, or inner solver
iterations, and does not unify the CBE and two-site pair implementations.

### Local action construction and raw residual contractions

One-site local-action setup uses QR boundary values and active block shapes;
it does not compute the current canonical active tensor when that value is
unused. Callers requiring canonical tensors retain the full construction.
The forward and adjoint raw-coordinate Hamiltonian transforms use staged
matrix contractions instead of a direct three-operand einsum. This changes
floating-point summation order, not the operator, retained support, or solver
controls. Complex/gauge-scaled direct-reference and Hermiticity tests cover
these transforms. Gauge QR contractions retain their existing order. A local
kernel timing is not an end-to-end sweep speedup or convergence guarantee.

Repeated first matrix products in the reduced-sector Hamiltonian schedule are
computed once per action invocation when they fit a 64 MiB complex-workspace
budget. The second products and output accumulation retain the original route
order. Reuse never crosses vector applications, and workspace is invocation
local. The original uncached kernel remains a direct validation reference.


Independent stable-energy prefix reuse keeps the QR factor, Hamiltonian boundary,
norm boundary, and their logarithmic scales for unchanged prefixes (64 MiB of
numerical buffers per energy cache). Value signatures and the full frontier
structure invalidate changed prefixes; a different MPO object clears the cache.
This applies to the compact-frontier stable-energy path only. It preserves the
fresh streamed QR contraction order, including the norm denominator, and does
not replace the acceptance energy with the local eigensolver estimate. This is
memoization of the QR algorithm of Trefethen and Bau, *Numerical Linear Algebra*
(SIAM, 1997), doi:10.1137/1.9780898719574, with no rank cutoff or approximation.

### Compact right-gauge contraction

For large compact frontier blocks (at least 200,000 entries) with at most half
of the output rows occupied, the right-gauge contraction materializes only
those rows. It retains the entire contracted axis and the original NumPy
`einsum` reduction; it does not use sparse BLAS, prune small values, or change
QR rank thresholds. The full output shape is restored before QR. The path is
restricted to float64/complex128 blocks and finite right boundaries; other
inputs retain the existing contraction. This reduces temporary input storage,
not the output or QR workspace. Regression tests check exact equality and
forbid materializing the full input block. Full-solver performance remains
workload dependent; isolated contraction timings are not sweep speedups.

## Quantum chemistry workflow

See [the QC example and checkpoint guide](../../examples/qchem/README_letta_qc.md)
for singlet CAS two-site MPS preparation, mutual-information ties, and one-site
LETTA with U(1) + SU(2). The reusable driver is `qchem_workflow.QCLETTA`.

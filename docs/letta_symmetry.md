# LETTA symmetry integration

Development branch: `letta_oct_4_sym`. The complete required support matrix and current unfinished work are in [the implementation plan](plans/2026-10-04-letta-symmetry.md).

## Shared metric compression

The nonsymmetric solver and shared native U(1)/SU(2) CBE and tied two-site solvers accept `MetricCompressionOptions`. The solver names are:

- `als`: alternate linear least-squares solves for the two factors.
- `variable-projection`: solve the second factor at each nonlinear first-factor iterate.
- `joint-ls`: optimize both factors together with a least-squares trust-region method.
- `grassmann-newton`: variable projection in a subspace chart, with the reduced objective Hessian.

All four minimize the same physical norm error with the environment metric. Gauge blocks preserve the middle charge/spin sector and shared invariant physical labels. SU(2) compression works in multiplicity coordinates; it does not reconstruct a magnetic wavefunction or determinant Hamiltonian.

```python
from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg

compression = MetricCompressionOptions(
    solver="variable-projection",  # also als, joint-ls, grassmann-newton
    als_max_iterations=100,       # complete left + right fitting rounds
    lsmr_max_iterations=300,      # iterations within each linear solve
    max_iterations=100,           # nonlinear solver budget
    tolerance=1e-10,
    max_workspace_mb=128,
)
options = LETTATwoSiteOptions(
    max_sweeps=20,
    split_method="metric-als-energy",
    energy_refinement_max_iterations=32,
    energy_refinement_tolerance=1e-10,
    reduced_sector_growth=True,
    compression=compression,
)
# result = letta_two_site_dmrg(hamiltonian, state=state, bond_dim=16, options=options)
```

The example initializes a retained multiplet allocation, fits it with the selected metric compressor, and then alternates energy minimization on A and B. One energy-refinement round updates both sites; its budget is independent of the ALS and inner LSMR budgets. `metric-als` disables this final energy alternation; the older `conditional-svd` entry point retains its projected metric-fit behavior. Untied reduced MPS uses its physical-metric Schmidt split instead of iterative fitting, followed by energy alternation when requested.

`als_max_iterations=None` uses the caller's `truncation_max_iterations` for reduced two-site fitting. `lsmr_max_iterations=None` uses an automatic cap between 50 and 1000 depending on the factor size. Nonlinear `max_iterations` counts residual evaluations for the least-squares methods and trust-region iterations for Newton; it does not cap the dense pseudoinverse used by variable projection. A nonlinear workspace limit can trigger an explicitly reported ALS fallback.

## Accuracy diagnostics

A reduced two-site update exposes `compression_diagnostics`, including requested/used solver, initial/final physical loss and nonlinear fallback information. ALS adds:

- `als_max_iterations`, actual outer `truncation_iterations`;
- `linear_solves`: side, LSMR stop code, requested cap, actual iterations, residual and normal-residual norms, and convergence status;
- relative factor-gradient `stationarity`;
- `optimizer_success` and `status`, distinguishing convergence, budget exhaustion and stagnation.

A capped fit retains its best non-increasing-loss iterate. It is not described as converged. Compression quality and final energy acceptance are separate: improving norm error alone cannot guarantee an energy decrease.

For open virtual boundaries, the reduced metric square root factors boundary multiplicity Grams. The experimental cyclic pair adapter instead uses the full local reduced coefficient-space Gram; it does not assume a separable ring environment. Tiny coordinate scales are equilibrated before rank decisions. Neither path constructs a global determinant-space frame.

## Representation and current boundary scope

QC supports U(1) number, U(1) number × U(1) spin projection, and U(1) number × SU(2). SU(2) physical dependencies use invariant irrep/multiplicity labels. Spatial orbitals have empty, single and double labels; magnetic components are structural Clebsch–Gordan coordinates. D in the reduced solver counts complete multiplets, not magnetic states or total stored parameters.

The native reduced backend supports open virtual boundaries for one-site, CBE and two-site updates. Native closed-ring one-site, residual-CBE and two-site updates use `ReducedRingLETTA`, an explicit covariant target closure and the full cyclic metric. The unified model/topology API and broader feature-matrix verification remain unfinished. The inherited periodic benchmark backend is separate and does not establish support for these unfinished native adapters.

## Reduced energy updates and recovery

Reduced one-site updates check the physical energy again after normalization. All tensor changes and cached environments are rolled back if normalization or another numerical operation fails. Invalid norms, significant imaginary energies and nonfinite energies raise explicit numerical errors.

Reduced two-site candidates—including temporary sector allocations—are built on copies. Numerical failure discards the candidate and attempts an ordinary one-site update for the current sweep site. If that also fails, the incumbent is retained and the failure is recorded. The next pair is still attempted. An update records `fallback` and `recovery_reason`.

When the incumbent bond fits the requested cap, every two-site candidate is compared to an ordinary one-site update from the same incumbent. The lower energy is retained, with `baseline_energy` and `baseline_selected` identifying this decision. This comparison is skipped when the requested bond cap is smaller than the incumbent allocation, because that baseline would be infeasible at the new cap.

Energy refinement records its initial/final energies, actual A/B round count, accepted substeps and fresh final local residuals in `energy_refinement_diagnostics`. Local eigensolves record `relative_residual` and `local_converged`. Sweep convergence requires a small energy change and a fresh local stationarity audit; rejected updates, unresolved local solves, failed baselines and numerical fallback are not treated as successful convergence. These are local stationarity checks, not proofs of a global variational minimum. The residual scaling uses the norms of Hx and E Nx in the local coordinates.


## Native reduced CBE

```python
from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg

options = LETTADMROptions(
    cbe_enabled=True,
    cbe_selector="exact",
    cbe_expansion_dimension=1,      # complete multiplets added per bond
    cbe_projection_max_iterations=300,  # tangent-space projection LSMR
    cbe_energy_refinement_max_iterations=32,  # A/B energy rounds after trim
    compression=compression,      # same four solvers and budgets shown above
    max_sweeps=100,
)
# result = letta_dmrg(hamiltonian, state=reduced_state, options=options)
```

This implementation evaluates the native reduced pair residual, removes variations already accessible to the original two one-site parameter spaces in the physical metric, and fits the remaining residual with legal symmetry-preserving factor blocks. It greedily allocates whole multiplets by the achieved physical-metric fitting loss. Two deterministic starts include every legal tie label. `exact` refers to the pair actions; it does not certify a globally optimal nonconvex direction fit. Pair coefficient arrays are stored, but a dense pair metric and dense tangent Jacobian are not.

For a left-to-right step the selected right-factor rows are appended with zero left-factor columns; right-to-left reverses this. Padding exactly preserves the wavefunction. The energy eigensolve updates only the expanded single core. The supplied resulting pair vector is then compressed, optionally energy-refined, and compared to an ordinary one-site update from the same incumbent. The nominal cap is the largest initial bond allocation in complete multiplets and remains fixed across gauge rank reductions and later sweeps. A candidate is kept only if it strictly beats that baseline at the cap.

Both residual-factor fitting and post-expansion fitting use `compression.als_max_iterations` when specified; otherwise they use `cbe_refinement_max_iterations`. Inner compression LSMR uses `compression.lsmr_max_iterations`. The separate `cbe_projection_max_iterations` bounds tangent-space projection, not ALS. Untied reduced MPS uses its physical-metric Schmidt truncation; this exact split does not need a nonlinear compression solver.

Diagnostics expose the missing residual norm, captured physical weight, tangent overlap, actual allocation, per-fit solver/budget/status, raw compressed energy, post-refinement energy and baseline decision. Numerical selection, expanded solve or compression failures discard the candidate and retain the independently computed ordinary step. If that ordinary step itself fails, the incumbent is unchanged and the update is rejected. Subsequent sites still attempt CBE. Unresolved projection, rejected baseline and failed/capped final fitting are not reported as successful sweep convergence.

Currently only a native `ReducedMPOHamiltonian` with the `exact` selector is accepted. A streamed/shrewd selector, nonzero baseline allowance and separate preselection controls are not implemented. Reduced energy refinement uses alternating one-site solves; the nonsymmetric coupled-factor energy-refinement controls do not apply here. General forward/backward ties are covered by the full native metric. The default reduced frontier gauge uses an exact shared-frontier gauge where available and a legal marginal gauge elsewhere; `gauge_mode="none"` remains available. This does not complete the general closed-ring support matrix.


## U(1) and products of U(1)

The shared block backend also accepts ordinary Abelian LETTA states. `AbelianReducedMap` groups equal physical charges, preserves their multiplicity and local basis permutation, and attaches a trivial one-dimensional spin representation. This is an exact coordinate conversion. It neither imposes physical spin SU(2) nor projects the wavefunction. Signed charges, repeated/noncontiguous equal-charge labels, and multiple independent U(1) factors are supported. Finite cyclic groups are explicitly rejected by this adapter.

Native MPO compilation now keeps **every** conserved charge component. Thus a spin-flip term is valid with number-only symmetry but rejected when Nalpha and Nbeta are separately fixed. All conversion, compilation and optimization use local tensors and virtual spaces. Dense global states/operators appear only in independent tests.

```python
import numpy as np
from pyqed._letta_one_site_opt import (
    LETTADMROptions, MetricCompressionOptions, abelian_dmrg,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, initial_state
from pyqed._letta_two_site_opt import LETTATwoSiteOptions

n = 3
h1 = -np.eye(n, k=1) - np.eye(n, k=-1)
eri = np.zeros((n, n, n, n))
for i in range(n):
    eri[i, i, i, i] = 4.0
problem = ElectronicProblem(h1, eri, (2, 1))
state = initial_state(problem, max_bond_dim=8, symmetry="nalpha_nbeta")
compression = MetricCompressionOptions(als_max_iterations=100, lsmr_max_iterations=300)

one = abelian_dmrg(problem.mpo(), state=state,
    options=LETTADMROptions(max_sweeps=100))
cbe = abelian_dmrg(problem.mpo(), state=state,
    options=LETTADMROptions(max_sweeps=100, cbe_enabled=True, compression=compression))
two = abelian_dmrg(problem.mpo(), state=state, bond_dim=8,
    options=LETTATwoSiteOptions(max_sweeps=20, reduced_sector_growth=True,
                               compression=compression))
```

The returned state is a `LatticeLETTA` with the original physical basis, coordinates, symmetry and ties, plus the retained charge allocation. D counts ordinary virtual states here, since all irreps are one dimensional. Existing `letta_dmrg(..., cbe_enabled=True)` calls with U(1) states automatically use this adapter. `abelian_dmrg` explicitly selects the shared backend for ordinary one-site and two-site runs as well; it validates the corresponding options through their existing public entry points. Bond schedules and finite cyclic groups are not yet implemented in this shared adapter. Open virtual boundaries are required; a periodic Hamiltonian is distinct from a closed virtual ring.


## Gauge conditioning with arbitrary ties

For a cut whose entire physical frontier is present on both neighboring tensors, the reduced solver uses the existing conditional boundary gauge. Otherwise it conditions on the subset of labels available on **both** tensors. It sums diagonal environment blocks over the unavailable labels, without altering the dependency graph.

Write the boundary environment in a fixed symmetry sector as a matrix over memory labels and virtual multiplicities. If the memory labels split into shared labels and other labels, the gauge uses

$$
\overline G_{s;ab}=\sum_u G_{(s,u,a),(s,u,b)}.
$$

The eigenvectors and positive supported eigenvalues of this marginal define an invertible multiplicity transformation. Its inverse is absorbed into the neighboring tensor using the same shared labels. Small eigenvalues receive an invertible unit gauge rather than being deleted. Thus the physical state, all target magnetic components, bond allocation, and ties are preserved. On a cut with no shared labels this reduces to an unconditional sector-wise virtual gauge.

Only the supported marginal becomes identity. Correlations between different frontier-memory assignments remain in the full environment, and every local solve still uses its full overlap metric. There is no claim that the generalized eigenproblem becomes Euclidean.

`reduced_gauge_variables(state, cut)` reports the actual conditioning labels. `reduced_frontier_grams` reports the corresponding marginals. The gauge functions accept `strict=True` to require a complete shared frontier and reject before changing tensors. Low-level shifts and whole canonicalization passes retain independent tensor snapshots and restore them if an operation raises; their standalone API propagates that error. The sweep wrapper additionally checks that the physical energy is unchanged, restores tensors and sector metadata on numerical/resource failure, rebuilds moving environments, and continues at the next update. An initial-gauge failure continues from the original state. Generic `recovery_reason` diagnostics prevent a recovered sweep from being reported as converged.

This behavior is available to native SU(2) and the shared U(1) backend for ordinary one-site, CBE and two-site updates. Validation covers cyclic physical dependencies on an **open virtual chain**; this is separate from the closed-ring cyclic contraction described below.


## Native closed-ring one-site calculation

`ReducedRingLETTA` stores physical conditional cores and a separate covariant target core. Both virtual legs at this closure remain ordinary multiplet spaces: this is a closed virtual loop, not an open chain with a periodic Hamiltonian. The target core carries the dual total charge/spin and closes the physical state into a scalar. The Hamiltonian acts as the identity on this auxiliary representation.

```python
import numpy as np
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed._letta_one_site_opt import (
    ReducedPhysicalBasis, ReducedRingLETTA, LETTADMROptions, letta_dmrg,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem

# Three-site periodic fermionic Hubbard model, N=3 and S=1/2.
n = 3
h1 = np.eye(n) - np.ones((n, n))
eri = np.zeros((n,)*4)
for i in range(n):
    eri[i, i, i, i] = 4.0
problem = ElectronicProblem(h1, eri, (2, 1))
state = ReducedRingLETTA.random(
    n, ReducedPhysicalBasis.spatial_orbital(),
    SpinChargeSector(3, SU2Irrep(1)),
    multiplets_per_sector=2,
    neighborhoods=((0, 1), (1, 2), (2, 0)),
    seed=71,
)
result = letta_dmrg(problem.su2_mpo(), state=state,
    options=LETTADMROptions(max_sweeps=20, gauge_mode="frontier"))
```

`multiplets_per_sector` initializes that many copies of EACH reachable sector; it is not a total D cap. `state.bond_dimensions` reports the actual multiplet counts, including both closure-adjacent bonds. A constructor taking explicit `bond_sectors`, `tensors` and `closure` is available when a particular allocation is required. The optional `anchor_sector` chooses the virtual irrep at the distinguished ring cut; it need not be the scalar irrep. U(1) and product-U(1) use generic `Sector` labels with a trivial SU(2) factor, as in the shared Abelian backend.

Every dependency list starts with the core's own physical site and may contain forward, backward, nonadjacent or last–first ties. Physical labels cross the sequential embedding as neutral multiplicity memory. The original virtual bond still closes through the target core. No target magnetic component is copied as a physical tie label.

A sweep updates all physical cores and then the target closure (reverse order on the next sweep). The closure is variational in its multiplicity coefficients, so its coefficients are included in `parameter_count`; it is not counted as an extra physical site when reporting energy per site. `RingSiteUpdate.is_target_closure` identifies its local update.

Local solves retain the complete cyclic overlap matrix, including correlated environments. Dense local-parameter H/N matrices or the shared matrix-free generalized eigensolver are used; neither path constructs a global determinant-space Hamiltonian. `gauge_mode="frontier"` currently applies unconditional sector-multiplicity Gram balancing on each ring edge. It is legal for every tie pattern but does not make the cyclic metric the identity. `"scalar"` balances only a scalar core scale; `"none"` skips gauge moves. The open-chain QR/frontier canonicalization is not reused on a ring.

The norm convention is the invariant physical-plus-target scalar norm. Each raw auxiliary magnetic slice has 1/(2S+1) of that squared norm. Multiplying a slice by sqrt(2S+1) and its CG phase gives the corresponding physical target component. Scalar energy ratios agree in all components.

Accepted local updates pass a fresh physical energy check after normalization. A failed solve restores its incumbent. A failed gauge restores the state after the accepted local update and continues with subsequent cores. Recovery is recorded and prevents that sweep from being called converged. Energy plateaus require a fresh all-core local-residual audit; they are not guarantees of a global minimum.

The ring state supports public one-site, residual-CBE and two-site paths. Enable ring CBE through `LETTADMROptions(cbe_enabled=True)` with the exact selector. The existing open-chain boundary-Gram compression root cannot be used for a general cyclic metric.


## Cyclic pair compression: internal adapter

The native ring pair and compression adapters are independently verified and connected to public ring two-site sweeps. The supplied-target adapter also compresses the result of the genuine expanded-one-site ring CBE update. `CyclicPairProblem` forms the full correlated H/N pair response on a physical/physical or closure-adjacent graph edge. Its fusion layout includes missing middle irreps, independent of the incumbent allocation.

`compress_ring_pair` accepts the same `MetricCompressionOptions` and explicit ALS/LSMR budgets as the open reduced backend. All four solvers operate on the same cyclic physical-norm loss. The caller remains responsible for changing sector allocations, choosing starting factors, alternating energy minimization, comparing against an ordinary one-site baseline, and committing or restoring a candidate. Calling the adapter alone does not guarantee an energy improvement.

The current cyclic metric root uses dense LOCAL pair matrices with a workspace guard. This is distinct from a forbidden global determinant projection, but can still be expensive for large pair spaces. Long-ring numerical scaling and the full model/topology verification matrix remain unfinished.


## Two-site optimization on the covariant ring

Pass a `ReducedRingLETTA` state to `letta_two_site_dmrg`. The ordinary
`LETTATwoSiteOptions` and `MetricCompressionOptions` apply, including separate
ALS, inner LSMR, nonlinear optimizer and A/B energy-refinement budgets. Set
`reduced_sector_growth=True` to discover locally reachable middle sectors;
otherwise the existing allocation is used. `bond_dim` caps the total retained
multiplets at each visited graph bond, rather than copies per sector.

Each step works on a copy. New left columns are zero and complementary right
rows are seeded: their product remains exactly the incumbent. The full cyclic
pair H/N is solved, and a coefficient SVD supplies only initial ranks/factors.
Both incumbent-based and SVD-based factor starts are fitted in the correlated
physical metric using the requested compressor. Failed starts are recorded;
finite fitted states are compared by physical fitting loss. The retained
allocation is physically installed on both cores, including closure metadata.
For energy-refined split modes, full-metric one-site solves alternate between
the two graph vertices. A fresh norm and energy are evaluated on the resulting
state before any candidate is committed.

An ordinary one-site update is also computed from the untouched incumbent. If
its current bond meets the requested cap, it is a feasible baseline and the
lower-energy candidate is selected. Numerical/resource failure records a
reason and falls back to that baseline; failure of both candidates leaves the
original state and allocation intact. When reducing a bond cap below the
incumbent allocation, an oversized baseline cannot satisfy the new cap and is
not silently accepted as a successful compression. A rejected step can retain
the old allocation; the sweep is not called converged while any bond exceeds
the requested cap.

The graph vertices are physical cores 0 through L-1 followed by target closure
L. A forward sweep visits (0,1), ..., (L-1,L), (L,0), and the next sweep reverses
that order. Thus both closure-adjacent bonds participate in growth and
compression. This schedule does not call a three-core update through the
closure a two-physical-site update: physical (L-1,0) are not adjacent graph
vertices. The physical Hamiltonian may contain the periodic last-first term;
that term remains present in every exact environment contraction.

`conditional_discarded_weight` records the initialization SVD statistic, which
is NOT a cyclic physical discarded weight. `metric_truncation_loss` is the
physical-norm fitting loss before subsequent normalization/energy refinement.
Compression diagnostics retain the requested/used solver and linear/nonlinear
stopping reports. Energy plateaus require successful compression reports and a
fresh all-core local residual audit before convergence is declared. This is a
stationarity condition, not proof of a global minimum. Gauge failures restore
the state after the accepted update and are marked in the sweep record.

Independent tests cover all four compressors, exact Hubbard dimer energy,
three-site periodic Hubbard doublets and Bose-Hubbard with arbitrary ties,
missing spin-sector growth, both closure edges, matrix-free pair actions,
state-preserving growth/shrinkage, same-start recovery and failed partial gauge
writes. Larger systems and the complete topology/model/method matrix remain
separate validation gates.


## Residual CBE on the covariant ring

`letta_dmrg(..., state=ring_state, options=LETTADMROptions(cbe_enabled=True))`
uses the exact residual selector with the same shared compression controls as
open reduced CBE. The fixed total multiplet cap is the largest bond allocation
in the supplied state at the start of the sweep run. Sector allocation may
change while that cap is respected. A separate bond schedule is not supported.
Every physical core and the target closure participate; reverse sweeps select
the preceding graph edge modulo L+1, so neither closure edge is skipped.

The ordinary one-site baseline is computed first. Selection is performed on a
copy of that baseline. Locally reachable sectors provide a temporary source
layout, preserving the baseline's coefficients by exact zero padding. Let x
be its merged pair coefficients, and let H and N be the full cyclic pair
Hamiltonian and overlap. The physical Rayleigh quotient and coefficient
residual are

$$
E=\frac{x^\dagger Hx}{x^\dagger Nx},\qquad r=Hx-E Nx.
$$

Let S and W be the supported metric root and whitening maps described above,
with S†S=N and SW=I on retained support. The residual in orthonormal physical
coordinates is

$$
g=W^\dagger r.
$$

For the actual conditional source factors a,b, denote their exact bilinear
merge by M(a,b). Restrict variations to the incumbent bond allocation. The
weighted tangent map is

$$
J(\delta a,\delta b)=S\bigl[M(\delta a,b)+M(a,\delta b)\bigr].
$$

Its adjoint includes the sparse tie-copy adjoints; copying physical labels is
not assumed to be isometric. LSMR solves the least-squares tangent projection,
without materializing J. The missing residual and its coefficient target are

$$
z_* = \arg\min_z\|Jz-g\|_2^2,\qquad
 g_\perp=g-Jz_*,\qquad t=Wg_\perp.
$$

The selector reports both the projection stopping status and the remaining
tangent overlap. Failure to converge triggers ordinary-step recovery. A greedy
allocation fits t with complete multiplets, testing each available sector and
two deterministic source-factor starts using the selected common compressor.
The selection budget counts new complete multiplets. It does not count magnetic
components, and the greedy nonconvex fit is not claimed to be globally optimal.

On a forward update, selected right-factor rows are appended and the matching
left columns are zero. A reverse update appends selected left columns and zero
right rows. In both cases the starting physical wavefunction is unchanged.
Only the active **one-site** Hamiltonian is then diagonalized. Its resulting
pair vector is passed to `fit_ring_pair_target`, which never calls the pair
eigensolver, for full-metric compression and optional A/B energy alternation.
The trial is committed only when its freshly contracted energy is strictly
lower than the ordinary baseline and its bond allocation meets the fixed cap.
Otherwise the ordinary baseline is retained.

Failures in selection, expansion, the expanded solve or trimming discard the
trial and preserve the independently computed baseline. Failure of the baseline
itself preserves the incumbent and reports a rejected recovery. Gauge recovery
continues to restore the post-update state. Errors and incomplete compression
prevent a sweep from being declared converged; an energy plateau still requires
fresh all-core local stationarity checks. These checks are not global optimality
certificates.

Unlike the current OBC selector, this initial cyclic accuracy backend does
materialize the **local reduced pair Gram** for its correlated root. Diagnostics
set `cbe_materialized_pair_metric=True`; neither a global physical-state frame
nor a determinant Hamiltonian is built in production. The workspace limit can
trigger ordinary-step fallback. Streamed/shrewd selection, separate preselection
controls and relaxed baseline allowances are explicitly rejected. The dense
cyclic root is not a claim of scalable large-ring CBE.

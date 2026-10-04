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

The reduced metric square root factors only boundary multiplicity Grams. No dense square root of the full pair metric is constructed. Tiny coordinate scales are equilibrated before rank decisions, sharing the bg metric-conditioning implementation.

## Representation and current boundary scope

QC supports U(1) number, U(1) number × U(1) spin projection, and U(1) number × SU(2). SU(2) physical dependencies use invariant irrep/multiplicity labels. Spatial orbitals have empty, single and double labels; magnetic components are structural Clebsch–Gordan coordinates. D in the reduced solver counts complete multiplets, not magnetic states or total stored parameters.

The integrated native reduced backend currently uses open virtual boundaries. The inherited periodic benchmark backend is a distinct closed-ring implementation with narrower model/tie/symmetry support. Its presence does not complete general SU(2) periodic support. Reduced SU(2) and Abelian CBE now support these open virtual boundaries. General ring environments and the unified public API remain implementation tasks; see the full plan rather than inferring support from a file name.

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

`reduced_gauge_variables(state, cut)` reports the actual conditioning labels. `reduced_frontier_grams` reports the corresponding marginals. The gauge functions accept `strict=True` to require a complete shared frontier and reject before changing tensors. Individual shifts and whole canonicalization passes retain independent tensor snapshots and restore them if an operation raises; the error is propagated after restoration rather than hidden.

This behavior is available to native SU(2) and the shared U(1) backend for ordinary one-site, CBE and two-site updates. Validation covers cyclic physical dependencies on an **open virtual chain**; this is separate from closed-ring virtual contraction, which remains pending.

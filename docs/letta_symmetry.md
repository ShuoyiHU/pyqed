# LETTA symmetry integration

Development branch: `letta_oct_4_sym`. The complete required support matrix and current unfinished work are in [the implementation plan](plans/2026-10-04-letta-symmetry.md).

## Shared metric compression

Both the nonsymmetric solver and native reduced SU(2) tied two-site solver accept `MetricCompressionOptions`. The solver names are:

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

The integrated native reduced backend currently uses open virtual boundaries. The inherited periodic benchmark backend is a distinct closed-ring implementation with narrower model/tie/symmetry support. Its presence does not complete general SU(2) periodic support. Symmetry-enabled one-site CBE, general ring environments and the unified public API remain implementation tasks; see the full plan rather than inferring support from a file name.

## Reduced energy updates and recovery

Reduced one-site updates check the physical energy again after normalization. All tensor changes and cached environments are rolled back if normalization or another numerical operation fails. Invalid norms, significant imaginary energies and nonfinite energies raise explicit numerical errors.

Reduced two-site candidates—including temporary sector allocations—are built on copies. Numerical failure discards the candidate and attempts an ordinary one-site update for the current sweep site. If that also fails, the incumbent is retained and the failure is recorded. The next pair is still attempted. An update records `fallback` and `recovery_reason`.

When the incumbent bond fits the requested cap, every two-site candidate is compared to an ordinary one-site update from the same incumbent. The lower energy is retained, with `baseline_energy` and `baseline_selected` identifying this decision. This comparison is skipped when the requested bond cap is smaller than the incumbent allocation, because that baseline would be infeasible at the new cap.

Energy refinement records its initial/final energies, actual A/B round count, accepted substeps and fresh final local residuals in `energy_refinement_diagnostics`. Local eigensolves record `relative_residual` and `local_converged`. Sweep convergence requires a small energy change and a fresh local stationarity audit; rejected updates, unresolved local solves, failed baselines and numerical fallback are not treated as successful convergence. These are local stationarity checks, not proofs of a global variational minimum. The residual scaling uses the norms of Hx and E Nx in the local coordinates.

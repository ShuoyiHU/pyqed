# Proposed reduced one-site CBE implementation

This is the next implementation design, not completed support. Preserve the full parent plan: U(1)/SU(2), QC/condensed, OBC/closed PBC, arbitrary ties, all three optimization methods and all four compressors remain required.

## Existing reusable pieces

- Native `reduced_pair_problem` supplies H/N actions in polynomial-size reduced pair coordinates, with new fusion sectors supplied by `_expand_reduced_pair_space`.
- `ReducedPairMetricRoot` supplies a boundary-Gram square root without constructing full N.
- `_pair_vector_from_sources`, `_left_source_adjoint`, `_right_source_adjoint` provide the bilinear tangent map and its Euclidean adjoint.
- `compress_reduced_pair` supports all four solvers and honest ALS/LSMR limits in admissible factor gauge blocks.
- `optimize_reduced_site`, `refine_reduced_pair_energy` and fresh energy checks now provide the variational update/recovery building blocks.

## Exact residual selector first

1. Construct a candidate space containing reachable complete multiplets. Retain the incumbent exactly: never count random seeded partner rows as an already-active tangent direction.
2. Form the covector residual `(H - E N) x` using native reduced actions. Raise it with the supported metric inverse, respecting the boundary Gram nullspace and irrep dimension weights.
3. Build a matrix-free tangent operator for variations of the ORIGINAL left and right source parameters. In weighted coordinates, project the raised residual off this tangent with LSMR; check the projection status and remaining tangent overlap.
4. Fit the missing direction with a small legal reduced factor pair. Use a physical-metric sector SVD as an initializer and the shared compressor to enforce arbitrary tying. Weighted sector ranking must include complete-multiplet norm weights. An ordinary unweighted frontier SVD alone is not a metric-optimal selector.
5. For a left-to-right step, append selected right-factor rows and zero left-factor columns; reverse for right-to-left. Introduce only the selected multiplet multiplicities. Verify exact state preservation before any optimization.
6. Solve the EXPANDED ONE-SITE problem. No two-site energy eigensolve is allowed in this CBE path.
7. Refactor pair truncation into a reusable target-vector fitting helper so it can compress the expanded one-site state WITHOUT first optimizing a merged pair. Allow energy acceptance against the original one-site baseline, not against the lower uncompressed expanded energy.
8. Compress to the chosen total multiplet cap, energy-refine, and accept only if it beats the ordinary one-site baseline from the same pre-update state. Restore and return that baseline on any numerical failure.

This first native selector may materialize polynomial-size reduced pair coefficient arrays for correctness. Label its diagnostics `exact`; do not claim the streamed/shrewd no-pair guarantee. The existing nonsymmetric shrewd path must remain intact. A later streamed reduced selector can use this native selector as its reference.

## Metric factor details to verify

For boundary multiplicity Grams, equilibrated factors give S and W with S W = I on support. The pair metric is the boundary Kronecker action with the right irrep dimension and accumulated environment scale. Its supported inverse and square-root adjoints must be validated against an independently assembled small local metric, including complex rank-deficient and strongly rescaled cases.

A sector SVD can use padded square boundary factors, so template dimensions remain unchanged even when the support is smaller. Divide the whitened sector block by sqrt(dim J_middle), rank using dim J_middle times singular-value squared, then undo the boundary factors. After projecting factors back to legal LETTA dependencies, recompute the actual full-metric captured weight; do not treat the unconstrained SVD weight as the achieved value.

## U(1) sharing opportunity

Investigate representing U(1) through the same reduced backend with trivial spin irreps and physical multiplicities (e.g. the two singly occupied spatial-orbital states). This could reuse all native contraction/compression/update code without duplicating CBE. Validate fermionic signs, physical basis order, sector mapping and exact conversion from Abelian LatticeLETTA. General signed charges/multiple U(1)s may need generic Sector metadata rather than a nonnegative particle-number-specific sector class. This is a proposal, not an assumption that the current interfaces already support it.

## Required tests before exposing the new dispatch

- Metric inverse/root identities and tangent adjoint/orthogonality tests.
- New-sector discovery from a deliberately incomplete allocation.
- Exact padding in both sweep directions; arbitrary forward/backward/shared ties.
- Native execution with magnetic/determinant expansion monkeypatched to fail.
- Guard that no pair energy eigensolver is called during CBE.
- Same-start one-site energy baseline, nominal cap, complete multiplets and numerical rollback including sector/cache restoration.
- Hubbard and molecular small-reference comparisons, all compression solvers, U(1) and SU(2). Do not equate fewer sweeps with correctness or assert global convergence from energy plateaus.

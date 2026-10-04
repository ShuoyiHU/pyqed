# Public LETTA API integration plan

The public entry point is now implemented but broader validation is still running. This is not a completed feature claim or the final implementation note. Keep numerical algorithms in the already tested shared backends rather than making a second solver implementation.

## Independent choices

A model defines its Hamiltonian, physical basis, target symmetry, number of physical sites and physical ordering. Hamiltonian edges (including PBC edges), virtual topology, and LETTA ties are independent inputs. A periodic Hubbard Hamiltonian must work with either open or closed virtual bonds and with or without the last-first physical tie.

Use a small `pyqed.letta` package with model construction, state initialization and optimization entry points. The model object should carry the native Hamiltonian, ReducedSymmetry, physical basis/permutation and provenance needed to validate supplied states. Reuse ElectronicProblem for molecular integrals and fermionic Hubbard signs; reuse AbelianReducedMap for U(1) products. Do not route production model building through benchmark modules or dense many-body references.

## Models

- Molecular ElectronicProblem: exact particle number alone, number plus Sz, or number plus SU(2); explicit target spin for open shells. Retain core energy exactly once.
- Fermionic Hubbard: construct h1 and local ERIs and reuse ElectronicProblem. Accept an explicit undirected edge list as well as a chain with open/periodic Hamiltonian boundaries, so arbitrary lattice geometries do not require separate optimizers.
- Bose-Hubbard: finite occupation cutoff, U(1) particle number, hopping, interaction and chemical potential. Physical spin SU(2) is not present for this scalar boson model; trivial SU(2) coordinates in the shared backend are representation plumbing, not an additional physical constraint.
- Heisenberg: U(1) Sz or SU(2) total spin. SU(2) invariant physical ties have one label for a spin-half site and therefore do not add expressive power. Any spin-coupled tie extension is a different ansatz and must not be silently substituted.
- Custom native reduced Hamiltonians and physical bases remain usable without an ElectronicProblem.

Validate site count, Hermiticity/conservation using existing builders, target reachability, duplicate/self/out-of-range edges, and incompatible symmetry options before entering numerical recovery. Preserve full complex coefficients where allowed.

## State creation

Expose `topology='open'|'ring'`, explicit neighborhoods or named untied/NN/periodic-NN patterns, `multiplets_per_sector`, seed, real/complex tensors, and optional ring anchor sector. `nn` means an open physical dependency chain independently of virtual topology; a wrap tie is explicit. Arbitrary forward/backward/crossing ties use the existing exact frontier embedding. Do not silently carry extra ties or change user dependencies for a gauge.

Reuse ReducedLatticeLETTA.random and ReducedRingLETTA.random; keep the existing exact conversion of MPS/Abelian states available. `multiplets_per_sector` is a copy count PER allowed sector, not total D. Report actual allocations. A ring with nonunit endpoint multiplicities and an explicit target closure must remain a real virtual ring. Validate the supplied state's basis, target, physical ordering and site count against the model without modifying it.

## Optimization

Expose one method selector: `one-site`, `cbe`, `two-site`. Provide a small common options object for sweep budget, energy/residual/metric tolerances, eigensolver budget, matrix-free threshold, gauge mode/direction, compression options, A/B energy rounds and CBE expansion/tangent-projection budget. Translate into existing backend options; never duplicate update logic.

Reuse MetricCompressionOptions directly. Defaults should give explicit independent ALS and LSMR budgets, with no ambiguity between a complete A/B compression cycle and one inner linear iteration. The same options must reach OBC/ring, U(1)/SU(2), CBE/two-site. Keep nonlinear least-squares evaluation and Newton iteration budget semantics explicit. Selection fitting and final compression may share this object; do not expose a misleading second ALS field that is overridden by the common option.

Two-site execution should enable symmetry-sector growth by default and enforce the requested retained total multiplet cap. One-site keeps its starting allocation. Current CBE preserves the maximum initial allocation as its nominal cap; reject a conflicting requested cap instead of ignoring it. Any future independently requested CBE cap needs explicit state-preserving allocation semantics and tests first.

Return existing detailed result/history objects, preserving convergence, recovery, inner-solver diagnostics and actual allocations. Energy plateaus, small local residuals, and global variational optimality are different statements.

## Required independent verification

Exercise the public API, not just option conversion mocks. Cross methods/topologies/symmetries on small molecular/fermionic, Bose and Heisenberg models; include multiple seeds and copy counts, periodic Hamiltonians with open virtual chains, closed nontrivial loops, and backward/wrap ties. Compare operators/actions and final expectation values to independent exact references, test energy bounds and symmetry leakage, inspect diagnostic budgets and allocation caps, and verify input states are unchanged. Verify model rejection errors are not swallowed as numerical recovery. Then run the combined implementation suite and write the full code-grounded implementation note.


## Implementation checkpoint

Added pyqed/letta/models.py, solver.py and __init__.py with native model wrappers, input/conservation/Hermiticity validation, explicit open/ring and tie construction, common optimization controls, and public method dispatch. Added examples/letta_symmetry.py and a quickstart in docs/letta_symmetry.md. The underlying numerical solvers are reused.

Fresh final-source checks: 26 model/input tests passed; 16 corrected compressor tests passed with actual missing-sector CBE expansion, strict independent molecular energy gain, no numerical recovery, all four solvers and explicit ALS/LSMR budget checks. The first compressor fixture was too expressive: ordinary one-site solved the dimer and correctly skipped expansion, so expecting compression reports was an invalid test assumption. It was replaced with an existing missing-sector fixture, not weakened to accept a no-op.

The public three-site periodic Hubbard example (U=4, t=1, mu=0, N=3, S=1/2), with virtual OBC and CBE, gave -1.274917217635371 versus independent fixed-sector exact -1.2749172176353756; error 4.66e-15. It correctly reports a one-sweep cap and zero numerical recoveries.

Original full test run remains live: tool session 41861, process 91584, log /private/tmp/letta-public-api.log. It was collected before the six extra input tests and before correction of the eight CBE compressor fixtures. Do not restart it just because it is slow: its broader method/topology cases still provide required evidence. Expect the old eight fixture assertions to fail; evaluate all other results and combine with the separately passed corrected compressor run. The original run imported the pre-Hermiticity-check model wrapper; the new check has separately passed the fresh model and compressor runs.

A redundant secondary run (session 24348, process 91936) was intentionally stopped with SIGTERM after its known fixture failures when concurrent cyclic cases caused memory pressure. This is not a numerical validation result. A fresh Python attempt also ended at startup with InterruptedError during that pressure episode, before collection. The corrected compressor run was then successfully executed after memory recovered. Keep expensive cyclic cases sequential.

Detailed test commands/results are in 2026-10-05-letta-public-api-tests.txt. The full feature matrix and combined final validation remain pending; no performance phase or usage check has begun.


## Eigensolver correction follow-up

The public model/input and real molecular compressor selections have been rerun after the local-metric equilibration correction (bce395a): 42 passed, 37 deselected in 41.07 s, log /private/tmp/letta-public-api-post-equilibration.log. The original full process still runs the older solver. It remains useful earlier-source evidence but does not replace final-source matrix verification. The public API is committed as an implementation checkpoint, not declared fully validated across every combination.

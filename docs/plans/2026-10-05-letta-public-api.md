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


## Current-source matrix checkpoint after lowest-root correction

On numerical commit 126bc0b, the public checks completed in separate processes:
64 model/input/compressor/open/spin-ring tests (52.75 s), six Bose/SU(2)-fermionic
ring method cases (555.84 s), four Bose/Heisenberg ring seed/copy allocations
(24.66 s), and one fermionic U(1) one-copy ring allocation (51.54 s). These
selections are disjoint: 75 of the current 79 public tests passed. Logs are
/private/tmp/letta-root-exploration-public.log,
/private/tmp/letta-public-matrix-bose-su2-final.log,
/private/tmp/letta-public-allocations-bose-spin-final.log and
/private/tmp/letta-public-allocation-fermion-one-final.log.

The four remaining current-source cases run sequentially in session 67996,
log /private/tmp/letta-public-remaining-fermion-final.log: three methods for the
three-site fermionic U(1) ring with backward/wrap ties, and the two-copy fermionic
ring allocation. Do not launch another copy of these large cases in parallel.

The original process 91584 / session 41861 was intentionally terminated after
verifying its command, exit 143. Its 15 completed dots remain in
/private/tmp/letta-public-api.log. It was superseded because it loaded code
before two reproduced eigensolver corrections and obsolete compressor fixtures;
its remaining output could not close the current-source validation gate.
Stopping it before the current-source large ring run avoids competing memory
use. This is not a numerical failure or a completed test result, and the old
partial output is not counted in the 75 current passes. The decision was based
on changed numerical source, not an observation timeout.

A combined implementation suite is also active in session 96127, log
/private/tmp/letta-combined-final.log, with the complex three-orbital molecular
backward-tie CBE stress case explicitly deselected. That stress case previously
passed on an earlier solver and still requires final-source accounting; neither
an exclusion nor a live process is a completed verification gate.


The combined run (session 96127) completed with exit 0: **400 passed, 1 deselected
in 402.35 s**, on numerical commit 126bc0b. It covers reduced symmetry/state,
frontier/norm/H, all compressors, eigensolver scaling/exploration, update/gauge/
resource recovery, Abelian conversion, native ring/target/pair/allocation/scaling,
all three methods, Schmidt and molecular symmetry tests. Counts overlap earlier
focused runs and must not be added as unique tests.

The sole excluded stress test now runs separately in session 24558, log
/private/tmp/letta-qc-ring-stress-final.log. It uses a complex three-orbital
molecular doublet, a nontrivial spin anchor and backward/wrap ties, and checks
independent physical energy, charge and total spin with no numerical recovery.
The remaining public test session is still 67996. These five cases are the
outstanding current-source numerical checks; the implementation note must
receive their results and final audit before the performance-phase usage gate.


The final-source molecular stress case completed successfully: session 24558
exited 0, **1 passed in 1741.47 s** (1739.18 s test call). Its log is
/private/tmp/letta-qc-ring-stress-final.log. The test independently checks the
complex three-orbital covariant-ring CBE energy against the fermionic determinant
Hamiltonian, fixed N=3 and S=1/2, unchanged backward/wrap ties and no numerical
recovery. This supplies the previously excluded current-source stress evidence;
it does not stand in for the four public fermionic U(1) cases still running in
session 67996.


## Live fermionic-ring resource observation

The same current-source session 67996 / PID 61659 remained live after 50:13
elapsed, with 45:40.70 CPU time and 90.1% CPU. No completed result or failure
was reported by the first selected one-site case. A one-second macOS sample
at 2026-10-05 04:53:55 +0800 placed the main thread in NumPy complex matrix
multiplication / BLAS and reported 32.2 GB physical footprint, 49.8 GB peak.
Raw diagnostic: /private/tmp/letta-public-ring-stack.txt. This is liveness and
resource evidence, not a completed test or a Python-level bottleneck profile.
The guide now records the exact test configuration and limits of this
observation. No numerical source changed, no process was restarted, and the
four-case correctness gate and subsequent usage/performance gate remain open.


## Interrupted handle and observed replacement run

The app lost exec cell 731 and process session 67996. Direct OS inspection then
confirmed that PID 61659 was absent and that no other copy of the pytest command
survived. The original log ends at the first selected case without a traceback
or pytest summary. Its cause of termination is unknown, and it supplies neither
a passing result nor a demonstrated numerical failure. This restart was based
on a missing OS process, not an elapsed observation window.

The exact same four selected tests and numerical commit 126bc0b are now running
under PID 93298, started in an independent process session. Diagnostic runner:
/private/tmp/letta_ring_validation_observed.py. Durable artifacts share prefix
/private/tmp/letta-public-ring-observed: .log (pytest), .status.json (PID and
terminal exit code when available), .progress.jsonl (function entry/return),
and .stacks.log (periodic Python stacks). The status file alone is not a
liveness signal; verify the OS process before interpreting unfinished status.
Function wrappers record timings and delegate unchanged arguments and results.
No solver budget, tolerance, state, Hamiltonian or production code changed.

The first two-minute stack identifies _transfer_product at
reduced_ring_contraction.py:111, called by CyclicReducedOperator.overlap from
the initial ring_energy in ring_dmrg. Progress records entered ring_energy at
7.66 seconds and had not returned by that sample. This is a measured initial
energy contraction bottleneck, not evidence of failed eigensolver convergence.
The stack log's 'Timeout' text is faulthandler's recurring dump timer; it does
not terminate the test or impose a solver time limit. All four accuracy gates
remain open; no completion or performance-phase claim follows from this sample.


## Smaller routine fermionic-ring fixtures (2026-10-05)

At the user's request, PID 93298 was terminated with SIGTERM after nearly seven
hours; subsequent OS inspection confirmed it was absent. No case had completed.
Its status is recorded as cancelled, not passed or numerically failed.

The three fermionic U(1) ring method tests now use the same three-site periodic
Hubbard Hamiltonian, complex initial states and full reachable charge sectors,
but one last-to-first tie instead of two simultaneous opposite wrap ties.
The doubled-multiplicity allocation test uses an untied virtual ring, retaining
both closure dimensions equal to two. The copies=1 allocation test still checks
both physical-index dependencies. Independent forward/backward/wrap embedding
and molecular ring CBE tests remain unchanged. This separates coverage rather
than claiming the full combinations were verified. No production code, solver
budget, or numerical assertion tolerance changed.

Set LETTA_FULL_RING_STRESS=1 to reproduce the original four configurations:

```bash
LETTA_FULL_RING_STRESS=1 PYTHONPATH=. python -m pytest -p no:cacheprovider -v tests/test_letta_public_api.py -k '(method_symmetry_topology_matrix and ring and fermion-u1) or (initial_allocations and 19-2-fermion-u1)'
```

The four revised routine cases **passed in 54.37 s**:
- one-site: 8.41 s
- CBE: 11.49 s
- two-site: 31.78 s
- doubled-multiplicity closure: 0.08 s

Log: /private/tmp/letta-public-ring-smaller.log. Together with the disjoint
75 previously passed, unchanged cases, this supplies passing evidence for all
79 current routine public cases. It is not a pass of the original four stress
configurations, nor a ground-state convergence or large-system scaling claim.
The separate implementation evidence remains 401 tests (400 combined plus the
independently completed molecular ring CBE stress case).

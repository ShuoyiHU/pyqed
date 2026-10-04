# LETTA symmetry integration implementation plan

> Execute this plan in the isolated `letta_oct_4_sym` worktree. Use the Executing Plans skill for task-by-task verification; retain this full scope across continuation turns.

**Goal:** Deliver one-site, controlled bond expansion, and two-site LETTA for quantum chemistry and condensed matter, with U(1)/SU(2), open/closed virtual boundaries, arbitrary physical dependency graphs, and selectable metric compression with explicit iteration controls.

**Architecture:** Reuse the native reduced QC backend and the newer bg metric conditioning, compression and transactional recovery. Separate model Hamiltonians, physical dependencies, virtual boundary topology, symmetry representation, and optimization method. Add a small public API over shared contraction/metric/update components; do not disguise a two-site eigensolve as one-site CBE or use determinant-space projections in active SU(2) solvers.

**Tech stack:** Python, NumPy, SciPy LinearOperator/LSMR, native reduced MPO environments, pytest; PySCF only as an independent molecular reference in tests/examples.

## Source preservation

- Original bg: `/Users/shuoyihu/Documents/GitHub/pyqed` (left untouched).
- Original QC: `/Users/shuoyihu/Documents/ChatGPT/LETTA/pyqed-letta-qchem` (left untouched).
- Integration: `/Users/shuoyihu/.codex/worktrees/letta-oct-4-sym/pyqed`.
- File snapshots, tracked diffs and status: `/private/tmp/letta-oct-4-source-snapshot`.
- Committed provenance: `2026-10-04-letta-source-manifest.json` in this directory.
- Actual shared source snapshot: QC commit e0dff43, which captured dirty bg. Both current working copies are three-way integrated against it.
- Include QC's uncommitted stable energy acceptance, physical-norm Schmidt split and independent tests. Include bg recovery, calibrated metric support and generalized ties.
- Preserve all other source changes in the original worktrees; historical generated benchmark output need not enter this implementation branch.

## Completion matrix (all cells are required; inherited code is not new verification)

| Requirement | Evidence required | Status |
|---|---|---|
| Preserve source work and requested branch | Source hashes unchanged; integration commit with manifest | Source hashes verified; integration checkpoint tested |
| Shared ALS/inner least-squares limits | Limits reach each backend, diagnostics distinguish cap from convergence | Pending |
| ALS, variable projection, joint LS, Grassmann Newton | Same physical-metric objective, derivative and fit tests, both CBE/two-site | Reduced two-site and SU(2) CBE integrated; remaining backend adapters pending |
| U(1) OBC one-site/CBE/two-site | Symmetry leakage zero, adaptive allocation, exact small-reference comparisons | Pending |
| SU(2) OBC one-site/CBE/two-site | Native reduced actions, complete multiplets, numerical-reference gates | Implemented, small references pass; broad model/tie/API validation remains |
| U(1) closed-ring all three methods | Cyclic contractions and wrap bond, nonidentity metric, independent references | Pending |
| SU(2) closed-ring all three methods | Reduced cyclic recoupling, explicit target representation, no magnetic/determinant solver fallback | Pending |
| Arbitrary tying | Forward/backward/nonadjacent/cross-cut/wrap dependencies, exact embedding and metric tests | Pending symmetry/ring extensions |
| QC and condensed models | Molecular integrals and Hubbard/Bose/Heisenberg examples with allowed symmetries | Pending integrated validation |
| Recovery and variational acceptance | Restored state/sectors/caches, same-start one-site baseline, strict fresh energy check | Reduced one-/two-site/CBE transactions and strict baseline implemented; ring integration pending |
| Accuracy diagnostics | Inner residuals/status, truncation stationarity, rejected-update handling; plateau != global minimum | Pending |
| Public API/documentation | Clear boundary/model distinction, D multiplets vs magnetic dimension, runnable examples | Pending |
| Final verification and commit | Focused and combined numerical tests; clean scoped diff and accurate commit description | Pending |

## Task 1: Integrate authoritative sources and establish a baseline

Files: native reduced modules and qchem adapter under `pyqed/_letta_one_site_opt/`, `pyqed/_letta_two_site_opt/reduced_solver.py`, bg CBE/compression/gauge modules, corresponding tests.

1. Snapshot source hashes and diffs without mutating sources.
2. Three-way merge working files; inspect every conflict rather than prefer an entire source branch.
3. Parse all integrated Python files and check no conflict markers remain.
4. Run native symmetry/norm/gauge/Schmidt/acceptance and bg CBE recovery/compression tests; repair integration errors.
5. Commit only integrated LETTA code, tests, fixtures and provenance; exclude unrelated graphics and large generated results.

## Task 2: Shared compression contract and convergence diagnostics

Files: `pyqed/_letta_compression.py`, both truncation adapters, reduced two-site solver; new `tests/test_letta_reduced_compression.py`.

1. Add failing tests for user ALS rounds and inner linear-solver caps in reduced fitting.
2. Introduce explicit per-linear-solve status/residuals, enforce finite iterates and physical loss improvement, preserve best candidate.
3. Feed common compression options into the reduced adapter; remove silent solver-option ignoring.
4. Adapt reduced factor layouts into the existing bilinear compression interface with admissible charge/spin gauge blocks and a full physical metric action.
5. Test all four compression methods, complex/semidefinite metrics, gauge invariance, capped solves and no determinant-space projection.
6. Add reduced alternating energy refinement, accuracy-based stopping and transactional one-site fallback tests.

## Task 3: Symmetry-preserving CBE on open networks

Files: new `pyqed/_letta_one_site_opt/reduced_cbe.py`, shared CBE update protocol, solver dispatch, tests for reduced/Abelian CBE.

1. Test whole-sector residual directions outside the represented tangent space, including initially absent sectors.
2. Implement reduced residual actions and metric-supported projection; use reduced two-site action only as a selection validation oracle, not an expanded two-site eigensolve.
3. Add exact zero-padding embedding, direction-dependent nonzero partner factors, sector-aware cache invalidation.
4. Optimize the expanded single core, compress to a whole-multiplet budget, energy-refine and compare against an ordinary one-site update from the same initial state.
5. On numerical failures restore tensors, sectors and caches, execute the one-site fallback and continue CBE.
6. Reuse the protocol for Abelian U(1) with charge masks/allocation; test molecular and Hubbard/Heisenberg cases.

## Task 4: Boundary topology and arbitrary ties

Files: dependency/frontier descriptions, native environment builders, periodic/open-boundary modules, new topology tests.

1. Represent Hamiltonian boundary conditions separately from virtual open/closed topology and physical ties.
2. Test every dependency routing category including backward and crossing ties; preserve physical charge ownership once.
3. Supply valid generic conditioning when a shared-frontier gauge is unavailable; never silently delete ties or pretend N=I.
4. Build U(1) cyclic H/N contractions and local/pair response maps with nonuniform sector multiplicities and wrap updates.
5. Build native SU(2) cyclic reduced contractions and target closure (singlet and explicit non-singlet target leg), validating recoupling against small external references.
6. Connect each boundary backend to the same one-site/CBE/two-site protocol and compression contract.

## Task 5: Model API, examples and end-to-end accuracy

Files: `pyqed/letta/` public entry points as justified by existing interfaces; molecular and condensed adapters; `tests/test_letta_symmetry_integration.py`; user guide/examples.

1. Add validated options for method, symmetry, topology, tying and compression with unambiguous iteration units.
2. Keep molecular integrals/fermionic signs and condensed model terms independent from optimization.
3. Run small independent reference cases for each method/symmetry/topology and supported tie layout, with multiple seeds/bond dimensions.
4. Check symmetry quantum numbers, energy bounds, raw residuals, exact embeddings, gauge invariance, growth/truncation losses and fallback behavior.
5. Verify enhanced variational spaces can retain their starting states; report capped/stationary runs accurately, not as proven global minima.
6. Run combined focused regression suite, inspect source preservation, document actual support and limits, and prepare final commit description with exact validation results.

## Validation command pattern

Use the existing main-repo `.venv-1/bin/python` with `PYTHONPATH=.` from this worktree; set `PYTHONDONTWRITEBYTECODE=1`, `NUMBA_CACHE_DIR=/private/tmp/letta-oct4-numba`, `MPLCONFIGDIR=/private/tmp/letta-mpl`, and all BLAS/OpenMP thread limits to 1. Run pytest with `-p no:cacheprovider`. Keep heavy outputs under `/private/tmp`. Dense FCI and magnetic reconstruction are independent validation only, never production fixes.

## Progress log

- 2026-10-04: created isolated branch, snapshotted relevant code from both current working copies and integrated against dirty-bg snapshot. No source worktree mutations. Full feature matrix remains open.

- Integration baseline: 121 passed, 3 deliberate unsupported correlated-diagonal metric combinations skipped. All 308 imported Python sources parsed.
- Added native boundary-factor metric root and shared reduced compression adapter; 85 focused reduced/QC/common-compression tests pass, including all four public two-site compression selections and independent ALS/LSMR budgets. Deleted the duplicate reduced fitting code with hidden linear limits.
- Full matrix remains incomplete: reduced post-compression energy alternation, stronger outer stopping/recovery, symmetry CBE, general cyclic reduced environments, public model/topology API, and full cross-product accuracy tests remain required.

- Final checkpoint validation: **204 passed, 3 skipped in 208.09 s**. Skips are deliberately impossible correlated-diagonal metric representations. Includes native SU(2), molecular acceptance, reduced norm/gauge/Schmidt, all four reduced/common compressors, reduced two-site, CBE rollback, open/periodic legacy backends and diagnostic reconstruction. Exact output: `2026-10-04-letta-checkpoint-tests.txt`. This is not verification of the still-pending full feature matrix.

- 2026-10-04 follow-up: native reduced two-site default now includes A/B energy alternation; one-site updates are transactional across normalization and caches; pair failures discard temporary allocations and fall back to an ordinary one-site update. Added same-start baseline selection when the incumbent allocation fits the requested cap.
- Added explicit invalid-energy checks and local residual diagnostics; flat rejected or unresolved updates no longer satisfy convergence. Final-state local residuals are rebuilt when checking sweep stationarity.
- Validation for this follow-up: 96 regression tests passed before the final finite-energy and baseline refinements; 40 finite-energy/one-site/QC tests passed; 17 native/Schmidt/molecular-acceptance checks passed after finite-energy validation; 63 focused final pair/refinement/compression/QC/Schmidt tests passed after adding the baseline guard. Logs are retained in `2026-10-04-letta-updates-tests.txt`. Tests overlap and are not summed into a unique-test count.
- Next implementation priority remains actual symmetry-aware one-site CBE. Proposed reduced selection/compression design is recorded in `2026-10-04-reduced-cbe-design.md`; it is not yet implemented or validated.


- 2026-10-04 CBE checkpoint: added native reduced residual projection, greedy whole-multiplet direction fitting, exact state-preserving padding, expanded one-site optimization, shared metric compression and strict same-start one-site energy acceptance. No pair energy eigensolve is called in CBE. Rebuild environments after allocation changes.
- All four shared compressors work in both reduced selection fitting and tied post-expansion trimming. ALS/inner LSMR/tangent projection/A-B energy budgets are separate and reported. Unsupported reduced shrewd/preselection/nonzero-baseline-allowance settings raise explicit errors.
- Added 18 focused CBE tests: missing-sector recovery, both directions, complex arbitrary ties, dense tangent projection, independent molecular energies, four-site Hubbard FCI and Heisenberg references, highly rescaled singular metrics, projection/baseline failure and retry. Earlier expanded test sets are superseded by the combined checkpoint log.
- Next work: shared Abelian representation/adapter, generic gauge fallback for arbitrary dependencies, closed-ring native symmetry contractions, model/topology public API, and full cross-product reference validation. The overall goal remains incomplete.

- Combined CBE checkpoint validation: **131 passed in 213.09 s**, including native reduced regressions, all four compressors, independent QC references, molecular roundoff acceptance and nonsymmetric CBE recovery. Exact output: `2026-10-04-letta-cbe-tests.txt`. Manifest recheck confirms 302 bg and 300 QC source files unchanged.

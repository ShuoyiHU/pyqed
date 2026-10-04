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
| Shared ALS/inner least-squares limits | Limits reach each backend, diagnostics distinguish cap from convergence | Shared controls implemented and tested in open and cyclic reduced adapters; unified API still pending |
| ALS, variable projection, joint LS, Grassmann Newton | Same physical-metric objective, derivative and fit tests, both CBE/two-site | Open reduced and cyclic two-site/CBE adapters implemented; broader combination verification pending |
| U(1) OBC one-site/CBE/two-site | Symmetry leakage zero, adaptive allocation, exact small-reference comparisons | Shared adapter implemented; independent small-model and conversion tests pass; broader API/topology gates remain |
| SU(2) OBC one-site/CBE/two-site | Native reduced actions, complete multiplets, numerical-reference gates | Implemented, small references pass; broad model/tie/API validation remains |
| U(1) closed-ring all three methods | Cyclic contractions and wrap bond, nonidentity metric, independent references | All three ring methods implemented; independent Bose/fermionic small references pass; broader matrix pending |
| SU(2) closed-ring all three methods | Reduced cyclic recoupling, explicit target representation, no magnetic/determinant solver fallback | All three ring methods implemented; reduced cyclic reference tests pass; molecular backward-tie stress test passed |
| Arbitrary tying | Forward/backward/nonadjacent/cross-cut/wrap dependencies, exact embedding and metric tests | Open/ring embeddings and all three methods implemented; complex molecular ring CBE validation passed |
| QC and condensed models | Molecular integrals and Hubbard/Bose/Heisenberg examples with allowed symmetries | Pending integrated validation |
| Recovery and variational acceptance | Restored state/sectors/caches, same-start one-site baseline, strict fresh energy check | Open/ring update and gauge recovery implemented; strict baselines, cache rebuilds and injected partial failures tested |
| Accuracy diagnostics | Inner residuals/status, truncation stationarity, rejected-update handling; plateau != global minimum | Pending |
| Public API/documentation | Clear boundary/model distinction, D multiplets vs magnetic dimension, runnable examples | Pending |
| Full implementation note | Complete code-grounded description, especially symmetry representations, contractions, gauges, expansion, compression, recovery and limitations | Required after implementation; pending |
| Conditional precision-preserving speedups | After correctness and note, query weekly remaining capacity; if >10%, profile and validate each speedup until approximately 5% remains or worthwhile options are exhausted | Not started; correctness gates take priority |
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


- 2026-10-04 Abelian checkpoint: added lossless U(1)/U(1)-product coordinate conversion into the native reduced block backend, preserving physical basis order, duplicate charges, virtual allocation, coordinates and arbitrary ties. Native MPO compilation now carries all charge differences rather than extracting only one particle-number label. No physical SU(2) constraint is added to Abelian models.
- Added public `abelian_dmrg` for shared one-site/CBE/two-site execution and automatic dispatch from ordinary U(1) `letta_dmrg` CBE. Existing ordinary Abelian paths remain available; the unified method/model/topology API is still pending. Added optional `normalize=False` state construction to make coordinate round trips coefficient-exact.
- Added 23 tests for arbitrary complex ties and repeated/signed/multiple charges, independent fermionic MPO action, all three methods versus Hubbard FCI and Bose/Heisenberg sector references, absent-charge discovery without pair diagonalization, all four compression choices/budgets, and conservation of the second U(1) component.
- Remaining work is unchanged in scope: generic gauge fallback on arbitrary dependency layouts, closed-ring native symmetry contractions and target closure, unified model/topology API, and full cross-product verification. Current checkpoints do not establish those requirements.

- Abelian regression checkpoint: **135 passed, 1 deselected in 180.68 s**. Excluded the six-orbital stress sweep; native compiler/reduced CBE and ordinary Abelian/QC/state/two-site regressions passed. Exact command/output in `2026-10-04-letta-abelian-tests.txt`. Reverified 302 bg and 300 QC source hashes unchanged.


- 2026-10-04 general-gauge checkpoint: reduced frontier conditioning now uses the largest shared physical-label subset. Partial tracing unavailable frontier labels yields a legal multiplicity gauge while retaining the full correlated local metric. Both U(1) and SU(2), all three methods, use this fallback automatically in frontier mode; strict full-frontier validation remains explicitly selectable.
- Gauge shifts and whole initialization passes restore independent tensor snapshots on exceptions. Added target-multiplet preservation, marginal whitening, nonidentity full metric, cache consistency, arbitrary-tie energy references and partial-write failure tests.
- Verification: **100 passed in 51.48 s**, including molecular roundoff acceptance and native U(1)/SU(2) CBE/Schmidt/update regressions. Exact command/output in `2026-10-04-letta-general-gauge-tests.txt`.
- Next priority is native closed-ring contraction and target closure; the existing OBC scalar environment cannot simply be traced to obtain an SU(2) ring norm. A proposed transfer-channel derivation and its validation gates are recorded in `2026-10-04-native-ring-design.md`. The full goal remains active and incomplete.


## User objective extension, 2026-10-05

After the complete implementation/support matrix is working and verified, write a full implementation note, with special attention to symmetry. Ground it in the final code and explain tensor/sector conventions, irrep multiplicities and target sectors, physical index ownership and arbitrary ties, actual virtual OBC/PBC topology, native H/N contractions and recoupling, gauges and metric support, one-site/CBE/two-site update steps, all compression methods and independent iteration units, acceptance and numerical recovery, independent references and remaining numerical limitations. Include derivations where needed rather than skipping steps. Intermediate design notes do not satisfy this deliverable.

Only after that correctness-and-documentation gate, read the account's current weekly usage using the usage-limits tool. Interpret the thresholds as **remaining** capacity. If more than 10% remains, profile the completed implementation and investigate worthwhile speedups, including reuse of existing DMRG backends or carefully scoped compiled kernels. Implement one improvement at a time and verify numerical accuracy before moving to the next. Changes may alter results only at machine-precision levels; do not weaken truncation, inner convergence, symmetry or acceptance criteria to improve timing. Continue until approximately 5% weekly capacity remains or worthwhile verified improvement opportunities are exhausted. Recheck actual usage between completed improvements, not by guessing from token counts. Do not start this optional performance phase while required correctness work is unfinished.

The requested scope remains all original model/symmetry/topology/method combinations plus this note and conditional performance phase. No goal completion claim is justified by the current checkpoints.


- 2026-10-05 ring foundation: native cyclic reduced norm/Hamiltonian contractions retain every total-spin channel and Hamiltonian intermediate fusion channel. Full local metric/H actions, nontrivial covariant target closure, signed charge labels and heterogeneous target-leg MPO metadata are implemented. Independent references cover complex/unequal-multiplicity rings, electronic singlet/doublet/triplet targets, U(1) Bose-Hubbard and product-U(1) fermions.
- Independent periodic Heisenberg reference exposed an AutoMPO family-prefix duplication. Fixed prefix sharing and removed the obsolete fully reduced exchange compensation; preserved and extended reduced-vs-independent operator tests. Production LETTA still uses native reduced contractions; no determinant projection was introduced.
- Checkpoint validation: **187 passed in 52.06 s**. Exact command/output in `2026-10-05-letta-ring-tests.txt`. Source manifest verifies 302 bg and 300 QC files unchanged.
- This is a contraction/closure foundation, not completed ring optimization: arbitrary-tie cyclic state embedding, sweep/closure update policy, ring pair metrics/compressors, CBE/two-site integration, sweep-level gauge recovery, full API and full implementation note remain required. Long cyclic product scaling also needs validation. Performance phase has not started.


- 2026-10-05 ring one-site checkpoint: added `ReducedRingLETTA` conditional storage, L+1 explicit virtual allocations and variational target closure, signed-sector seeded construction, and reuse of the exact sparse physical-label frontier without imposing unit virtual endpoints. The public `letta_dmrg` dispatches this state type explicitly.
- Native cyclic local H/N are composed with the exact tie embedding; dense local and matrix-free generalized solves retain the full correlated metric. Sweeps include physical cores and the target closure. Legal unconditional sector gauges and scalar scaling cover every cycle edge, with transactional failure recovery and a fresh all-core residual convergence audit.
- Independent tests cover forward/backward/wrap ties, unchanged imported amplitudes, complex non-unit virtual loops, local H/N versus an independently enumerated frame, sector-gauge invariance, dimer singlet/doublet/triplet energies, three-site periodic fermionic Hubbard, public U(1) Bose-Hubbard and injected solve/gauge failures. No production magnetic/determinant state expansion is used.
- Combined regression: **147 passed in 65.29 s**. Then strengthened the matrix-free check to a singular metric with non-unit closure bonds: **1 passed in 2.06 s**. These counts overlap; exact output/commands are in `2026-10-05-letta-ring-sweep-tests.txt`. Manifest again verifies 302 bg and 300 QC source files unchanged.
- Ring CBE remains an explicit unsupported error rather than silently using ordinary one-site sweeps. Next: cyclic pair response/metric root and all four compressors, true expanded-one-site CBE and two-site integration including the target-closure edges, open-chain sweep-level gauge recovery, long-chain product scaling, full model/topology API and final implementation note. Performance phase remains gated on full correctness and documentation.


- 2026-10-05 cyclic pair/compression checkpoint: native full cyclic pair H/N, including missing intermediate sectors and both target-closure edges; supported correlated metric square root; analytic factor adjoints and legal gauge blocks; all four common compressors. Public ring two-site/CBE updates remain pending.
- Validation: **174 passed in 64.12 s**, recorded in `2026-10-05-letta-cyclic-compression-tests.txt`. Original 302 bg and 300 QC source hashes verified unchanged. Next: exact allocation changes, multistart supplied-target fitting, A/B energy refinement and transactional update integration.


- 2026-10-05 ring two-site checkpoint: exact whole-sector allocation growth/shrinkage on every cyclic graph edge, multistart supplied-target fitting with all four compressors, optional A/B full-metric energy alternation, same-start ordinary one-site baseline, transactional numerical/resource recovery, and public two-site dispatch for `ReducedRingLETTA`. The target closure is explicitly included in the update schedule.
- Validation: **126 passed in 183.93 s**, plus **1 optional-polish check passed in 2.58 s** after that test was added. Exact commands/output are in `2026-10-05-letta-ring-two-site-tests.txt`. Independent checks include exact Hubbard dimer energy for all compressors, a periodic Hubbard doublet without magnetic expansion, and three-site periodic Bose-Hubbard with arbitrary ties.
- Ring CBE remains unimplemented; its next step is residual projection with the full cyclic root, whole-sector direction selection and exact expanded-one-site insertion, then the supplied-target fitter verified here. Larger cyclic scaling, public model/topology API, complete feature-matrix validation, OBC sweep gauge recovery and the final implementation note remain outstanding. Goal stays active; performance/usage gate has not been reached.


- 2026-10-05 ring CBE checkpoint: shared residual/tangent selector, native full-cyclic metric, whole-sector directional insertion, expanded one-site solve, all four physical-metric compressors, strict same-start baseline and transactional recovery including closure-adjacent updates. No pair eigensolve or global/magnetic state expansion enters CBE.
- Validation: 116 passed, 1 deselected in 96.88 s; the original separate ring run completed with 44 passed in 1181.73 s, including the complex three-orbital molecular backward-tie case. Counts overlap; commands/output in 2026-10-05-letta-ring-cbe-tests.txt. OBC sweep recovery, broader API/feature matrix, cyclic scaling and final implementation note remain outstanding.


- 2026-10-05 OBC gauge recovery: sweep-level physical-energy checks, transactional tensor/sector restore, moving-environment rebuild after partial cache mutation, and continuation without false convergence. Initial and subsequent shifts are covered in both directions for one-site, CBE and two-site. Low-level standalone gauge errors still propagate after restoration.
- Verification: 17 new focused recovery tests passed, then 113 combined reduced/gauge/update/CBE/two-site/molecular-acceptance/ring-sweep/Abelian tests passed in 87.15 s. Exact command/output in 2026-10-05-letta-sweep-recovery-tests.txt.
- Long-ring scale stress tests now reproduce intermediate overflow/underflow in native single-site and pair complement products, despite representable final contractions. Fixing this numerical issue is the next correctness task; this is not the optional performance phase.


- 2026-10-05 cyclic scaling checkpoint: reproduced six failures from intermediate overflow/underflow under cancelling binary gauges. Shared exact power-of-two transfer multiplication fixes both one-site and pair complements without rank/channel truncation or independent H/N rescaling. Truly unrepresentable absolute environments fail explicitly.
- Verification: 8 numerical-range tests passed; 134 combined cyclic norm/operator/target/pair/allocation/compression/two-site/CBE/sweep tests passed, with the 20-minute molecular CBE case excluded from this rerun after its previous successful run. Counts overlap. Exact outputs in 2026-10-05-letta-ring-scaling-tests.txt.
- Next implementation task: public model/state/method API and its independent cross-product tests, detailed in 2026-10-05-letta-public-api.md. Existing detailed diagnostics require a public-API audit; complete feature-matrix verification and final implementation note remain required before the usage/performance gate.

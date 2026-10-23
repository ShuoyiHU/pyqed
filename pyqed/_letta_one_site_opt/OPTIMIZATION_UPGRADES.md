# Exact optimization upgrades, 23 October 2026

These changes operate on reduced SU(2) tensors and existing allowed-path
coordinates. No determinant-space Hamiltonian, magnetic-state expansion,
new physical truncation, or looser accepted residual tolerance is introduced.
This document distinguishes implemented execution strategies from measured
speedups; an available strategy is not necessarily the fastest one.

## Local Hamiltonian

Let $F$ map the active source parameters to the physical norm coordinates.
The existing equilibrated SVD constructs an orthonormal image $Q$ and source
map $B$, with $FB=Q$. The local solver acts on

$$K=B^\dagger H_{\mathrm{source}}B=Q^\dagger H_{\mathrm{aux}}Q.$$

The old exact action expands a vector, applies the reduced operator, and
pulls the answer back every iteration. `compact_hamiltonian.py` instead
assembles $K$ directly from the same sector routes. For one route,

$$K_{uv}\mathrel{+}=c\,U_u^\dagger
[(Q_{Lu}^\dagger LQ_{Lv})\otimes(Q_{Ru}^\dagger RQ_{Rv})]U_v.$$

Physical slices, complex conjugation, SU(2) multiplicity weights and all
off-diagonal image couplings are retained. Boundary products are shared
within a bounded LRU cache and rank columns are tiled. The code does not
probe the operator with every coordinate unit vector.

If the retained solver rank is $r$, the complex matrix occupies $16r^2$
bytes, independently of the much larger auxiliary frontier vector. This
does not make assembly cheap. Its arithmetic includes boundary transforms,
contractions with each image, and output-block accumulation. The work screen
models boundary-product cache reuse and compares assembly MACs with one
factored application, including coordinate transforms. This is an arithmetic
heuristic, not a predicted wall time. The unchanged factored action is the
fallback when the configured work or storage budget is exceeded.

Full compact assembly is optional. Diagonal image blocks can instead supply
a block preconditioner without requiring QR absorption or a full matrix.
They are never substituted for the full physical Hamiltonian.

## Search and certification

The Davidson space retains $V$, $NV$ and $HV$. Restarts rotate all three;
old Hamiltonian products are not recomputed merely because of a restart.
Search-space size and retained Ritz-vector count are configurable, with an
optional memory budget. Defaults retain the original 48/4 policy and do
not impose a new memory cap on existing jobs. Reports count search
applications, iterations, restarts and reused vectors.

Optional recycling stores at most four source-coordinate vectors per local
problem, within a 16 MiB context cache. Their source layout and support are
checked before reuse. Every recycled vector receives fresh norm and H
applications for the new problem; stale eigenvalues or H products are never
reused. The full-support probes and excited-incumbent safeguards remain.

Optional physical-residual stopping removes excessive raw-coordinate
conditioning factors from the internal stopping target. A new source H
application still certifies the same requested physical residual. Failed
certification retries the stricter solve and remains unconverged if that
also fails. This may add work and is not enabled by default.

This is an adaptation of E. R. Davidson, *J. Comput. Phys.* **17**, 87–94
(1975), [doi:10.1016/0021-9991(75)90065-0](https://doi.org/10.1016/0021-9991(75)90065-0),
with metric coordinates, restart retention, block inverse approximations and
search safeguards. Finite exploration does not prove that an arbitrary
stationary eigenvector is the globally lowest eigenvector.

## Pair preparation, CBE and compression

Pair problems now defer raw H/N environment construction until a raw action
is actually requested. The canonical pair solve often never needs it.
Canonical boundary caches are passed through pair sweeps and CBE
transactions. Value/layout signatures invalidate changed boundaries.

The pair metric root is prepared once per state signature and tolerance.
Target physical coordinates and their metric adjoint are fixed during a
compression fit and are evaluated once. During ALS, the exterior QR
boundaries do not change: only the current partner tensor is absorbed to
construct the next factor coordinates. CBE tangent coordinates use the same
fixed exterior boundaries. Pair Hamiltonian routes combine identical fixed
channels before compiling their action.

Energy refinement and CBE one-site candidates can use transactional moving
contexts. Normalization then changes the active tensor, instead of balancing
unchanged exterior tensors after every inner update. Every final candidate
still passes an independent energy check. Candidate states own their tensors;
only value-validated immutable boundaries are shared.

Stationarity checks reuse that same moving context, rebuilding their local
actions from the final tensors. The independent energy certificate also uses
the context's QR contraction mode. Previously it always selected the
structured mode, while some pair/CBE contexts selected the ordinary mode;
alternating between them cleared the entire boundary cache after each local
update. Matching modes removes this invalidation cycle without replacing the
energy certificate with a solver eigenvalue. The scalar energy cache is shared
between serial CBE candidates and checks tensor-value signatures before reuse.

The existing sector-wise tangent SVD and lazy sector/rank-profile fitting
were already implemented. They are retained, rather than replaced with a
second selector. Sector-loss bounds prune impossible winning allocations;
incumbent zeros never prohibit opening new symmetry sectors.

The new `joint-ls-operator` compression option minimizes the same full
physical loss as the other compressors:

$$f(A,B)=\|F(AB-T)\|^2,\qquad
J(\delta A,\delta B)=F(\delta A B+A\delta B).$$

Its adjoint uses the existing reduced factor contractions and physical-root
adjoint. Real/imaginary components are packed explicitly for complex data.
A trust-region least-squares method uses LSMR on these actions, without
forming $J$ or $J^\dagger J$. The original factor coordinates, iteration
budgets, physical loss, factor-growth safeguards and best-iterate guard are
preserved. It is optional; default ALS remains unchanged. Nonconvex factor
fitting has no global optimum guarantee, and finite-budget trajectories can
differ between algorithms.

The linear solver component adapts Fong and Saunders, *SIAM J. Sci. Comput.*
**33**, 2950–2971 (2011), [doi:10.1137/10079687X](https://doi.org/10.1137/10079687X).
The LETTA residual selector remains an adaptation, not a reproduction of the
shrewd CBE algorithm of Gleis, Li and von Delft, *Phys. Rev. Lett.* **130**,
246402 (2023), [doi:10.1103/PhysRevLett.130.246402](https://doi.org/10.1103/PhysRevLett.130.246402).

## Tall norm factors

For tall overlap blocks, row tiles feed a TSQR tree. Only the small final
$R$ undergoes SVD. Leaf QR factors are recomputed when assembling the final
left singular vectors, avoiding storage of the full input factor and never
forming a squared-condition-number Gram matrix. The largest singular value
of each block gives a lower bound on the global cutoff. Columns below that
bound are omitted before storing or emitting left singular vectors; the
original global cutoff then acts on the remaining columns. No vector that
passes the global cutoff is removed early. The retained image itself still
requires storage. Wide blocks or blocks whose small QR core exceeds the
row-workspace policy retain the existing dense SVD route.

This serial recomputed-leaf adaptation follows the QR structure of Demmel,
Grigori, Hoemmen and Langou, *SIAM J. Sci. Comput.* **34**, A206–A239 (2012),
[doi:10.1137/080731992](https://doi.org/10.1137/080731992).

## Operator-aware paths

The first graph follows `(bra sector, ket sector, MPO sector/copy)` from
both open boundaries. A route without a complete path is removed before
environment contraction. Its graph depends on allocation, not coefficients.
A changed allocation can change globally pruned routes, even in an otherwise
unchanged prefix. Boundary keys therefore include the actual filtered local
transfer kernels, as well as tensor values and quantum-number layouts. Only
boundaries whose inputs changed are invalidated. A global graph fingerprint
alone is not a reason to discard a boundary.
Coarse-graph cache keys contain sector/channel transitions and boundary
sectors; they exclude pending-label fingerprints, which cannot affect this
graph. The fine graph includes those fingerprints. Per-boundary value/layout
signatures remain in force for both, so this avoids unnecessary global
invalidation without retaining an invalid changed prefix or suffix.

An optional finer graph additionally carries both pending physical-label
assignments. Its transitions use the exact allowed-path graph and the
existing nonzero reduced MPO kernels, preserving fermionic coefficients and
recoupling factors. The free pair gets a Cartesian bridge across both active
sites; imposing the legal factor graph there would incorrectly shrink the
two-site variational problem. Exceeding the graph budget returns the coarser
exact operator unchanged.

Both graphs remove complete impossible transfer/kernel routes. Surviving
auxiliary multiplicity matrices are still stored in the current matrix
format: this does **not** implement a fully ragged joint bra/ket boundary
representation. Physical pending-label masks are never applied to
QR-rotated multiplicity indices.

## Cache lifetime correction

The existing shared-product schedule held a strong reference back to its
owning Hamiltonian operator. That cycle could retain large arrays from old
sites until Python cyclic garbage collection ran. The schedule now holds a
weak owner reference; discarded operators release their arrays immediately.
A regression test disables cyclic GC and verifies immediate destruction.
Products are still recomputed for every new trial vector.

## Controls

| Control | Default | Meaning |
|---|---:|---|
| `PYQED_COMPACT_H_MB` | 0 | Full compact matrix storage budget; zero disables |
| `PYQED_COMPACT_WORKSPACE_MB` | 64 | Assembly tile and boundary reuse policy |
| `PYQED_COMPACT_H_MAX_MAC` | 2000000000 | Assembly work cap; zero disables this cap |
| `PYQED_COMPACT_MAX_ACTIONS` | 0 | Assembly/action MAC ratio cap; zero disables |
| `PYQED_LOCAL_PRECONDITIONER` | none | `blocks` enables the independent block inverse |
| `PYQED_PRECONDITIONER_MB` | 64 | Diagonal-block storage budget |
| `PYQED_PRECONDITIONER_MAX_MAC` | 1000000000 | Block assembly work cap; zero disables |
| `PYQED_DAVIDSON_SPACE` / `RETAIN` | 48 / 4 | Search dimension and retained Ritz vectors |
| `PYQED_DAVIDSON_MB` | 0 | Optional search-storage budget; zero keeps the original policy |
| `PYQED_DAVIDSON_RECYCLE` | 0 | Recycle layout-checked source vectors in a moving context |
| `PYQED_PHYSICAL_RESIDUAL_SOLVE` | 0 | Try physical stopping, with strict independent certification |
| `PYQED_PAIR_CONTEXT` | 1 | Reuse local preparation in native pair/CBE inner updates |
| `PYQED_OVERLAP_STREAM_MB` | 64 | Input-factor size threshold; negative disables TSQR |
| `PYQED_OPERATOR_PATHS` | 1 | Coarse bra/ket/MPO reachability |
| `PYQED_OPERATOR_PENDING_EDGES` | 0 | Fine graph edge budget; zero disables |

Cache and workspace controls are not total-process RSS limits. Benchmark
scripts impose a separate 4 GiB local guard. All local production-size tests
are 4×4; no 6×6 local test is part of this update.

## Completed validation

The focused regression suite passed **329 tests**. Coverage includes real and
complex actions, singlet and non-singlet sectors, newly opened CBE sectors,
free pair support, changed-H recycling, failed residual certificates,
ill-conditioned overlap factors, nonlinear Jacobian adjoints, cyclic full
metrics, and cache invalidation after tensor or allocation changes.

The performance input was one saved, developed 4×4 Hubbard LETTA state:
$U/t=4$, $t'=0$, open boundaries, half filling, spin singlet, $D_{\rm mul}=30$,
and forward physical nearest-neighbor ties. It was not an MPS initializer.
All measurements used one CPU thread and unchanged numerical controls.

| Method and measured scope | Frozen reference, seconds | Updated, seconds | Largest energy difference, in units of t | Peak GiB, before → after |
|---|---:|---:|---:|---:|
| One-site, 16 directional updates, default settings | 76.3946 | 77.4604 | 5.33e-14 | 3.075 → 3.089 |
| One-site, same sweep, selective compact H | 76.3946 | 72.1799 | 3.91e-14 | 3.075 → 3.088 |
| Two-site, one boundary update at sites 0–1 | 386.9617 | 126.3054 | 2.49e-14 | 1.666 → 1.394 |
| CBE, one boundary update at sites 0–1 | 945.6571 | 136.5059 | 3.55e-15 | 1.389 → 1.786 |

One-site errors are maxima across the 16 independently checked updated
states; pair/CBE errors compare independently contracted final energies.
The largest one-site relative physical residual was 8.63e-11. Explicit
between-site cyclic garbage collection was applied to both sides of the
one-site comparison because the old ownership cycle otherwise reached the
local memory guard. Sweep times sum the updates, excluding checkpoint
writes and between-site collection. These are individual timings, not a
statistical estimate over repeated runs.

Default one-site execution did not become faster on this input. Selective
compact assembly saved 5.5% over the complete directional sweep; its
256 MiB matrix budget and estimated 16-action work cap remain opt-in.
Unconditional compact assembly regressed in a separate matched test.
Forced TSQR preserved energy but showed no time or memory benefit here.
Physical-residual stopping alone did not reduce Hamiltonian applications.
The nonlinear compressor has correctness tests, not a demonstrated 4×4
full-sweep speedup.

The CBE comparison performed the same 36 local solves and 112 factor-map
builds. Factor-map time fell from 274.39 to 0.86 seconds; local-problem
preparation from 371.87 to 36.11 seconds; energy checking from 208.74 to
51.01 seconds. These are disjoint instrumented stages. Reusing the fixed
exterior and preventing the QR-mode cache flush removed repeated work;
fewer optimization iterations did not cause the improvement. Keeping useful
boundaries increased CBE peak memory by about 0.40 GiB.

Pair and CBE were validated on boundary updates, not complete 4×4 sweeps.
Their inner refinement caps differ between methods (8 for pair, 32 for CBE)
but are unchanged within each before/after comparison. Consequently this
table does not establish a speed ranking between the methods, or predict
6×6 runtime. The fine operator graph is still bounded route pruning, not
fully ragged joint-boundary packing.

Raw results, checkpoints, source checksums, plotting scripts and figures are
saved in the task artifact directory `letta_upgrades_oct23`. Its README
provides commands and records stopped experiments separately.

## Follow-up: reuse across complete pair transactions

The next round compares against commit `4b4ed6fd`, which already contains the
upgrades above. An ALS half-step solves a physical least-squares problem with
one factor fixed. Its norm geometry requires the two exterior QR factors and
the fixed partner. Rebuilding the full Hamiltonian is unnecessary for this
step. `reduced_pair_problem(..., norm_only=True)` now prepares that geometry;
an explicit subsequent H action lazily builds the unchanged exact operator.

Ordinary baseline, expanded candidate, compressed candidate and energy
refinement now share value-validated boundary preparations. They retain
separate mutable states. Pair acceptance normalizes the active center only,
so it no longer changes the entire exterior just to rescale one state.

For each inner one-site solve, source residual certification already computes
the physical Rayleigh quotient of the actual returned source vector:

$$E(x)=\frac{x^\dagger Hx}{x^\dagger Nx}.$$

Energy refinement reuses this value, keyed to the normalized candidate's
tensor values. It does not substitute the eigensolver's Ritz eigenvalue.
The final candidate receives a separate centered energy contraction; a
disagreement with the source certificate rejects the refinement. Ordinary
one-site transactions retain their existing independent checks. Setting
`reuse_local_energy=False` on the refinement helper retains all separate
checks for numerical comparisons. Iteration limits, acceptance thresholds,
stationarity requirements and SVD cutoffs are unchanged.

An interior-pair stack trace also exposed an unused active-center contraction
inside factor-coordinate construction. For active input shape $(l,p,r)$ and
boundary shapes $(a,l)$ and $(b,r)$, the old direct three-input contraction
scheduled approximately $aplrb$ multiply-accumulates and an $apb$ output.
The factor-coordinate map does not read that output: it uses the exterior
factors and the canonical partner. `factor_only=True` omits the active-center
transformation entirely, while performing the same exterior QR and overlap
SVD. It applies to both CBE tangent coordinates and compression coordinates.
The omitted center is explicitly represented as absent; requesting a full
Hamiltonian later rebuilds the canonical data. Tests compare both forward
and adjoint factor maps, including complex inputs.

For matrix-free CBE tangent projection, the previous map first built a full
pair vector in the exterior QR basis, then projected it into the existing
pair SVD support. The new map composes those operations before expanding the
pair. With left active factor $A$, fixed partner $B$, and existing support
isometries $U_L,U_R$, its block action is

$$J_L(z)=(U_L^\dagger A(z))(B U_R^*).$$

Physical legs are kept in their original order; SU(2) multiplicity weights
are unchanged. The projected fixed partner is prepared once. The adjoint
reverses these same products. The right-factor map is constructed in the
same way. This is exactly the previous supported tangent operator, not a
smaller tangent space. It avoids allocating complete QR pair vectors for
each LSMR application. ALS still measures the full physical compression loss;
its full-QR factor map now writes blocks directly into one output vector
instead of retaining a block dictionary and a second packed copy.

Packed pair-metric maps also share exactly equal fixed matrices across
routes. Shape and value hashes find candidates; full equality verifies reuse.
The contraction order and route coefficients remain unchanged. An interior
4×4 projection map with 84 routes needs 18,303,088 bytes of fixed matrices,
instead of 239,622,768 bytes when each route stores a separate copy. This
does not include tangent singular-vector storage, trial vectors or scratch.

This is exact data-flow reuse, not approximate contraction or a restriction
of the variational space. The original CBE and compression fidelity limits
described above still apply.

### Follow-up validation and limits

The expanded focused suite passed **402 tests**. Fresh serial measurements
against `4b4ed6fd`, on the same developed 4×4 checkpoint and sites 0–1, gave:

| Method | Before, seconds | After, seconds | Peak GiB before → after | Separate energy contractions before → after |
|---|---:|---:|---:|---:|
| Two-site LETTA | 63.4262 | 12.1168 | 1.293 → 1.217 | 24 → 6 |
| CBE LETTA | 77.1223 | 17.4807 | 1.533 → 1.234 | 42 → 7 |

Two-site retains 17 local solves and 8 energy-refinement rounds; CBE retains
36 local solves and 17 refinement rounds. These counts and all numerical
budgets match their controls. The separate energy counter now instruments
the pair module's imported energy helper as well as the one-site helper;
the earlier CBE count of 40 omitted two calls through the former alias.
Source residual/energy contractions inside a solve are counted within that
solve, rather than as separate whole-state energy evaluations.

Independent final energies differ by 0 and 7.11e-15 t respectively. Comparing
the normalized physical pair states after removing exterior scalar gauges
gives distances 1.72e-16 and 2.27e-16. No determinant expansion is used for
that comparison: both pair vectors are contracted through the same full
physical QR norm map. Fresh controls are used instead of mixing their
timings with the earlier measurements above.

At the interior pair 7–8, the omitted active-center transformations would
schedule 13,806,549,132,498 MACs across the two tangent maps. This is an
arithmetic count, not a measured speedup. The interior CBE replay still
crosses the 4 GiB local RSS guard during the coordinate/tangent workflow;
there is no completed interior energy or whole-sweep timing comparison.
The remaining large-basis storage needs further investigation. The boundary
measurements do not establish 6×6 feasibility or make that limit acceptable.

Reproducible scripts, checkpoints, source hashes, complete timing logs,
interrupted-run records and figures are in `letta_pair_transactions_oct23`.

# Local Hamiltonian runtime

Fixed-channel precontraction uses exact distributivity to combine terms before repeated local Hamiltonian applications. It selects which MPO channel to sum using an arithmetic estimate, subject to a 2 GiB prepared-operator budget. If neither grouped representation fits, the original contraction schedule is retained. The bound covers packed matrices and route tables, not total RSS: temporary matrices, the original operator, complex conversion caches, and application workspace can coexist. No numerical coefficient threshold or physical-rank approximation is used. Floating-point summation order changes.

The reduced one-site Hamiltonian preparation uses this optimization. It does not imply that all separate pair/CBE contraction paths have been optimized.

The eigensolver remains an unpreconditioned residual-subspace solver, with the same tolerance, minimum explored space, restart size, maximum iteration budget, and initial full-support probes. Auxiliary Ritz roots remain active while the initial Rayleigh energy has not decreased by a tolerance-resolved amount. After a decrease, only the lowest root is required to converge. This avoids spending applications converging excited roots after the requested root is accurate. The excited-incumbent safeguard is retained; finite search does not certify the global lowest eigenvalue in all possible problems. No diagonal preconditioner was added.

Validation: complex/real random operator actions, repeated vectors and budget fallback; stationary-excited-start regressions; a frozen 4x4 U/t=4 D=30 central local problem; matched four-sweep 2x4 half-filled and one-hole calculations. Benchmark results are in the task artifact h_action_audit_oct19. Component timing gains must not be presented as measured full-sweep cluster gains.


## Sector workers

`PYQED_SECTOR_THREADS=4` enables four threads over disjoint output physical slices. Each output's route order is preserved; input matrices are shared read-only and each worker owns its temporary and first-product reuse buffers. BLAS must be limited to one thread per worker to avoid nested oversubscription. The default is one sector worker. This changes execution scheduling, not the variational space, thresholds, or eigensolver settings. Extra memory is up to one existing workspace and reuse buffer per worker; the fixed-channel 2 GiB budget does not include these buffers. The pool is reused across applications. Small operators may not speed up; tested cluster central-site actions scaled approximately fourfold on four workers.

## Initial gauge preparation (October 20)

`GaugeQRBoundary` now contracts sector blocks with BLAS matrix multiplication.
The right contraction retains the compact-row allocation: absent scatter rows
are omitted from the multiplication and restored as zeros afterward. This change
is restricted to gauge preparation; the compression contraction routines are unchanged.
After a conditional gauge, the cached square-root factor is multiplied by the
known gauge on its conditional column blocks. For a contracted side `Q R`, the
new side is `Q (R G)`; the cached factor need not remain triangular. This avoids
a second site contraction and QR. This is an exact algebraic adaptation of QR
factor propagation (Trefethen and Bau, *Numerical Linear Algebra*, SIAM, 1997,
doi:10.1137/1.9780898719574), with no rank cutoff or solver tolerance change.
Floating-point rounding and the internal row basis may differ; comparisons must
use boundary Gram matrices and physical observables, not elementwise QR factors.

Local validation used the actual 6x6 D=30-multiplet MPS checkpoint through cuts
35 to 29, plus four-sweep 2x4 D=10 singlet/doublet trajectory comparisons. Cut 29
preparation decreased from 17.66 s to 1.16 s on one local CPU thread. Final small
system energy differences were below 8e-12 t. These are not full 6x6 sweep timings
or a completed 6x6 accuracy certification.

## Boundary-cache budget

Canonical QR/H environments now default to a 2 GiB total cache budget.
Set `PYQED_BOUNDARY_CACHE_MB` (in MiB) for larger models, e.g. 8192 for 8 GiB.
An explicit `CanonicalBoundaryCache(max_bytes=...)` takes precedence. Cache
eviction changes reuse only, not truncation or accuracy. This budget excludes
contraction temporaries, solver workspace and other caches; increasing it does
not accelerate first-time boundary construction. Running processes retain their
existing cache objects and are not changed by updating source or environment.

### Environment construction (October 20)

`advance_left` and `advance_right` group transfers by output symmetry sector.
`PYQED_ENVIRONMENT_THREADS` controls independent sector workers; if unset it
inherits `PYQED_SECTOR_THREADS` (otherwise one). Each output retains its original
term order. Keep BLAS at one thread when using sector workers. Parallel workers
need concurrent intermediate arrays; the boundary-cache limit does not limit
these workspaces.

The final contraction uses BLAS GEMM alpha/beta to combine spin-dimension
scaling and accumulation. Transpose flags avoid unnecessary operand layout
copies. This preserves the contraction tree and introduces no truncation, but
floating-point rounding need not be bitwise identical to separate scale/add.
Canonical H-environment boundaries use matrix products with compact occupied
rows where available. Other canonicalization call sites retain their existing
path. No local eigenproblem settings or variational space are changed.

Validation scripts and saved data are in the `environment_update_oct20` artifact
folder. They compare four complete directional sweeps on singlet and doublet
2x4, the actual 4x4 checkpoint's local H/N actions, and a seven-boundary 6x6
prefix. Prefix measurements exclude the initial gauge and eigensolver, and do
not establish a complete 6x6 sweep speedup.


## Exact fixed-tie dependency contraction (opt-in one-site SU(2))

Use `LETTADMROptions(gauge_mode="none", simplify_fixed_ties=True)` with the
reduced MPO backend. Each local problem removes a fixed tensor's tie axis only
when every slice is exactly equal. The active tensor retains every tie and all
its variational parameters. Views are temporary; the stored LETTA ansatz is
unchanged. Dependencies are checked again after each tensor update. This is an
exact algebraic contraction simplification, not a rank approximation.

Local QR boundaries and metric whitening remain enabled. The global conditional
gauge pass is skipped because it hides otherwise constant dependencies. This
can change the numerical trajectory relative to the default global-gauge run;
compare simplification on/off with the same gauge setting to isolate its effect.
CBE is unsupported by this option. Defaults remain unchanged.

MPS starts have many constant ties. As optimized tensors acquire genuine tie
dependence, this saving can shrink; early-site timings do not predict full-sweep
cost. Convergence tolerances and solver iteration limits are unchanged. Stable
energy evaluation uses the same exact simplification with every tensor fixed.


### Structured boundaries within the dependency path

The opt-in dependency path now factors exact disconnected row/column support
components separately during boundary QR. It removes only structural zero
rows/columns; no singular-value threshold is introduced. Rectangular groups
retain their complete QR images. The active tensor's parameterization is unchanged.

Canonical scalar-physical core matrices with at most 5% nonzero entries use
CSR products during H absorption when both cores qualify and the operator kernel
has at most eight nonzeros. Other transfers use the existing dense path.
These thresholds select execution, not truncation: every nonzero is retained.
Conversions are cached for one absorption, shared read-only between workers,
and discarded before the tensors can change. One core/environment/core product
is reused across operator output channels sharing an input channel. Outputs
remain dense under the current environment API; this limits memory savings.

Boundary caches are invalidated on a change of QR strategy. Existing value and
layout signatures still invalidate changed tensor dependencies. The previous
path remains available internally as DependencySweepContext(...,structured=False)
for matched diagnostics. This is an exact algebraic optimization of the reduced
representation, not a new physical approximation or published solver guarantee.


Oversized boundaries in the structured path are packed losslessly for the
canonical cache: sparse arrays use CSR storage only if it saves bytes, and
materialize their exact values on a cache hit. The existing byte budget remains
in force; no entries larger than that packed budget are retained. Working
Hamiltonian environments remain dense, so the cache budget is not a process
memory limit. Real canonical cores are no longer copied for conjugation;
complex conjugates are shared across transfers within one absorption.


## Solve reuse and single-update audit (October 20)

The local solver evaluates final source-coordinate H, energy and norm from one
shared Hamiltonian action. Its residual report is reused by the update caller
only for an exactly matching coefficient vector; the independently contracted
whole-state acceptance energy remains required. No tolerance, iteration cap,
retained norm support or acceptance criterion is relaxed.

Coordinate applications expand directly into the sector operator's packed input
and use block views for the output. They still materialize the expanded working
vector: this is allocation/copy elimination, not a fully projected small-space
Hamiltonian. Each returned result owns independent storage. Worker scratch and
first-product buffers are thread-local and reused; readiness flags reset on each
application. First-product reuse is still worker-local and capped at 64 MiB per
worker. Cross-worker sharing and full coordinate/operator precontraction are
not implemented pending the representative audit.

Overlap assembly skips exactly zero factors, and column normalization updates
the assembled factor in place. SVD and the existing rank threshold are unchanged.
`SectorOperator.timing_callback` optionally records input conversion/allocation,
scheduling, kernel wall time, per-worker setup/kernel time, scaling, route counts
and workspace bytes. Kernel time still includes gather and accumulation inside
the C loop; it is not a measurement of BLAS alone. Scheduled route arithmetic is
reported separately in the replay harness and is not evidence of necessary cost.

Validation: 98 focused tests and matched four-sweep 2x4 (7/8 electrons), 3x3
(8 electrons), U/t=4, t'=0, D=30 multiplets. Largest trajectory energy difference
was below 7e-11 t, with identical accepted-update counts. The `solve_reuse_oct20`
artifact stores timings and reproducible scripts. These small calculations do
not establish a 6x6 speedup. The cluster replay reconstructs the original state
after site 5, saves it, and runs baseline/updated site 6 in independent jobs from
the same file. It saves each reconstructed-site checkpoint and detailed timings.

### Shared first products (October 20)

`PYQED_SHARED_PRODUCTS=1` enables exact action-local sharing across sector
workers. It computes repeated `L @ X` or `X @ R.T` products once, then assigns
output slices to workers. The mathematical action and retained directions are
unchanged. This is optional pending complete-update performance validation.

`PYQED_PRODUCT_CACHE_MB` defaults to 16384 (maximum 16 GiB). Optional
`PYQED_PRODUCT_DISK_MB` defaults to zero and is bounded by 65536 (64 GiB).
`PYQED_PRODUCT_SCRATCH` selects the scratch directory; otherwise the system temp
directory is used. Products beyond these budgets are recomputed, never truncated.
The budget is conservatively planned for complex128. It is additional to the
boundary cache, operator storage, per-worker matrix products, and solver vectors.
Disk-backed products are rewritten on every action; disk spill is not enabled
by default because its benefit depends on measured I/O speed. Scratch files are
closed and removed on exit, including failures. The fixed route plan may be
reused, but no product survives between input vectors.

Timing reports separate first-product and second-product work and report RAM
and disk product bytes. Output tiling is used only when every participating
first product is cached, avoiding extra recomputation on uncached routes.

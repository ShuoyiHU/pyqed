# Chemistry MPO construction and native sweep preparation (2026-10-27)

These changes preserve the input integrals, physical sectors, multiplet cap,
solver residual thresholds, and variational space. Fixed-pass diagnostics below
are not converged accuracy comparisons.

## Work removed

For K spatial orbitals the two-electron input contains up to K^4 coefficients.
That does not require dense symbolic virtual-channel matrices. Chemistry now
uses sparse symbolic transitions, exact prefix/suffix sharing with coefficients
on a middle bridge, and build-local caches for fixed fermion algebra. The reduced
compiler still performs its original rank decisions; economic overwrite QR avoids
an unnecessary temporary copy. Reachability uses Boolean channel masks instead
of Python objects for every individual edge. Transfer compilation indexes allowed
physical-sector pairs once rather than filtering channels for every virtual-sector
pair. All contributing contractions retain their accumulation order.

Reference construction remains available through
`PYQED_QC_SYMBOLIC_STORAGE=dense` and `PYQED_QC_AUTOMPO_LAYOUT=prefix`.

## Native preparation

A failed next-bond refresh formerly cleared the previous active execution arenas
before storing them in their cached plan. Stashing them before clearing prevents
cascading empty-cache rebuilds.

A separate remaining rebuild source was the cached boundary pivot solve. If
P contains r pivot rows, the old refresh formed P P^T and inverted that Gram
matrix. This squares the condition number. In the observed CAS(16,16) run,
three boundary components with two pivots repeatedly failed its numerical pivot
check even though the pivot span remained usable.

On Gram failure, a reorthogonalized QR fallback now solves using the same pivots.
It reconstructs every component and rejects reuse unless the summed squared
residual is at most `1e-28 * max(1, total_squared_norm)`, matching the original
boundary decomposition scale. Rank-deficient, nonfinite, or inaccurate refreshes
still rebuild. For c components of m entries, the fallback takes
O(r^2 m + c r m) arithmetic and O(r m + c r) temporary storage. No local root
residual, sweep stopping rule, or physical direction is relaxed.

## Measurements

Local Apple host, Python 3.13, one BLAS/OpenMP thread. Peak memory is process
maximum RSS in decimal GB. Runs include the same initial state and settings
within each comparison. Native timings exclude symbolic MPO construction and
independent Python energy verification.

| Case and scope | Reference | Updated |
| --- | ---: | ---: |
| H8 D40, four passes, total | 9.411 s | 2.512 s |
| H8 D40, setup | 8.251 s | 1.824 s |
| H8 D40, peak memory | 2.207 GB | 0.490 GB |
| H8 D40, native sweep time | 0.402 s | 0.403 s |
| FeCO5 def2-TZVP CAS(12,12) D30, prefix vs balanced setup | 76.835 s | 11.761 s |
| Same CAS(12,12), total four passes | 83.130 s | 17.962 s |
| Same CAS(12,12), peak memory | 4.804 GB | 1.480 GB |

The CAS12 comparison isolates path layout with the newer algebra caches in both
runs. Maximum directional energy difference is 1.35e-13 Ha. H8 energy differs
by 5.3e-15 Ha. These are single-run measurements.

CAS(16,16) D30 completes four full adapter passes in 66.608 s, with 50.869 s
setup and 3.716 GB peak memory. There is no matching old full-adapter CAS16
measurement, so this is a capacity check, not a measured speedup.

For the native QR fallback, independent reduced-MPO expectations of the twelve-
pass CAS16 final states differ by 7.1e-15 Ha. All local residuals remain below
1e-11. The last four passes had 18 rebuilds before and zero with the fallback.

Two additional runs per version, alternating before/after order, measured mean
last-four-pass time 3.473 s before and 2.572 s after (25.9% reduction).
Total twelve-pass timings were 19.071/18.945 s before and 17.804/17.821 s after:
initial preparation dominates this short run, limiting the whole-run reduction
to about 6%.
These timings exclude independent energy verification. Fifteen native solver
regression tests passed with the fallback.

## Limits

In the matched H8 fixed-pass comparison, Block2 native sweeps took 0.216 s.
PyQED's setup gains do not establish parity in native sweeps. Block2's final
state expectation differs by about 3 microhartree after four passes; its local
pre-truncation energy is not the final-state energy. Python/BLAS runtimes also
differ. A converged equal-accuracy timing study remains outstanding.

Larger active spaces, reduced mutual-information diagnostics, and end-to-end
LETTA performance with representative chemistry ties still need broader study.
Small tied-system validation confirms numerical consistency but does not prove
large-system speedups.

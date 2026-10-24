# Fe(CO)5: larger-basis SU(2) MPS/LETTA campaign

80 MPS calculations and 20 LETTA calculations: cc-pVDZ, cc-pVTZ, cc-pVQZ,
def2-TZVP, def2-QZVP; CAS(12,12), (14,14), (16,16), (18,18);
MPS total multiplet caps 30, 50, 70, 90. LETTA uses the matching D=30 state,
its source charge/spin sectors, and all pairs with either conditional entropy
strictly below zero. No one-site MPS polishing stage precedes tie selection.

`submit.sh dry-run` prints the graph without submitting. `submit.sh submit`
submits one compilation/H2-smoke job, five RHF/integral preparation jobs,
twenty four-element MPS arrays, and twenty LETTA jobs. Each LETTA depends on
array element zero (D=30), not the whole array. These are 106 task executions
including setup/preparation. This bundle is prepared, not automatically submitted.

Cluster use, from the deployed bundle directory:

```bash
bash submit.sh dry-run
bash submit.sh submit
# Later, without starting any new calculations:
bash submit.sh collect
```

Default partition `gubing`, QOS `huge`. Override PARTITION/QOS if needed.
Preparations request eight CPUs/64 GB; solvers one CPU/64 GB, with single-thread
BLAS for reproducible resource use. SCF uses exact direct integrals and a
disk-backed AO-to-active-MO transformation; no AO four-index array is requested.
Set LETTA_PYTHON if the existing pyqed_letta_su2 environment is elsewhere. It must
provide PySCF, NumPy, SciPy, matplotlib, Cython and setuptools; the bundle vendors
its existing pure-Python sympy/mpmath/opt_einsum/tensorly dependencies. Linux needs
a C++17 compiler, OpenMP and LP64 BLAS/LAPACK libraries. The build gate checks the
real cluster runtime before admitting any preparation/calculation jobs.
If compiler Python packages are absent, install them in the selected environment
before resubmission: `"$LETTA_PYTHON" -m pip install 'Cython>=3,<4' setuptools`.
No block2 engine or block2-specific LD_PRELOAD is used.

MPS is the new pyqed compiled **plain two-site** solver, not the previous solver
that also tested a one-site competitor per bond. The absolute local residual tolerance is 1e-11 and the MPS
energy-plateau tolerance is 1e-6 Ha; cap 200 directional sweeps, requiring two
successive small energy changes. LETTA uses the existing reduced one-site
solver's convergence checks, with cap 1000. Caps are not convergence claims.
Unconverged capped jobs exit nonzero and preserve checkpoints; they must be
reviewed before extending caps. Extending config changes the campaign signature:
use a deliberate restart/migration rather than silently mixing configurations.
Checkpointing occurs every completed sweep. MPS restarts alternate directions.

Canonical active MOs are chosen as complete energy groups with spread <=1e-4 Eh,
separately for occupied and virtual orbitals, selecting the exact requested
count closest to the frontier. This avoids the previous 53/54 split in the
6-31G CAS, but is a **near-degeneracy heuristic**, not an orbital-character or
point-group-complete valence CAS guarantee. It may skip nearer MOs and need not
produce nested CAS spaces. Each preparation records actual MO indices, groups,
energies and leading Löwdin AO weights. Review these before chemical conclusions.
If no exact-size union exists, preparation fails instead of splitting a group.
No Fiedler ordering, orbital optimization, scalar relativity or ECP is applied.
The RHF determinant reconstructed from each CAS must match full RHF within 1e-8 Eh.
RHF internal stability is checked; no assertion of external spin stability is made.

Both methods use identical saved exact PySCF integrals. No FCI is scheduled for
these larger spaces. Results contain energies, convergence status, multiplet and
magnetic dimensions, stored coefficients, timings, ties and progress plots.
Compare different D within a fixed basis/CAS; do not treat the best D=90 energy
as FCI. Timings include separate setup and per-sweep solve/verification scopes;
they are not a controlled speed benchmark across concurrently scheduled jobs.

Basis references: Weigend & Ahlrichs, PCCP 7, 3297 (2005),
https://doi.org/10.1039/B508541A; correlation-consistent transition-metal sets:
Balabanov & Peterson, JCP 123, 064107 (2005), https://doi.org/10.1063/1.1998907.
The def2 choices provide an independent triple/quadruple-zeta family; increasing
CAS to 16/18 expands correlated occupied and virtual spaces, not an automatic
claim of a chemically complete active space.

## Exact-case Linux deployment

Runtime files are shipped in `runtime.tar.gz` directly from Git object paths,
not copied through a macOS directory tree. This preserves distinct `HEOM/heom`
and `DFT/dft` directories. The build job checks the archive hash, validates every
member, extracts it on Linux, and checks actual directory-entry capitalization
before installing source/vendor. Existing runtime directories are retained in
a timestamped backup. The normal source verification remains enabled.

Run `bash recover.sh` on the cluster after repairing the selected environment.
It cancels only live jobs recorded in this campaign's `results*/submitted.tsv`,
retains all logs, and uses a new `results_retry_DATE_TIME` directory.
Recovery waits for the build and tiny H2 validation to succeed before submitting
chemistry jobs, so a setup failure does not create another blocked job graph.

Use the established `pyqed_letta_su2` environment, as in the Hubbard campaign.
`PYQED_SU2_ENV` can override its location; `LETTA_PYTHON` can explicitly select
another interpreter. Preflight verifies that imported pyqed is the bundled,
source-verified snapshot, rather than another installation in the environment.
Check dependencies in this environment before recovery:

```bash
/share/home/gubingLab/hushuoyi/miniconda3/envs/pyqed_letta_su2/bin/python -c 'import Cython, setuptools, numpy, scipy, pyscf; print("Dependencies available")'
```

Install only missing dependencies in this environment. No separate Python
package directory or environment is needed.


## October 23 memory-sized retry

Use `bash retry_cases.sh --source results_retry_20261023_002723 --spaces 12 --submit`
for fresh CAS(12,12) MPS runs at 1e-6 Ha, followed by their D30-dependent LETTA jobs.
Use `--spaces 12,14,16,18` to retry all cases. Saved integrals are hash-checked and
copied into a fresh output directory; no SCF or integral transformation repeats.
The migration verifies that the old input signature differs only by the explicitly
requested energy tolerance. Old checkpoints and results are retained unchanged.
Only old selected LETTA jobs whose queue reason is DependencyNeverSatisfied are
cancelled. Every new worker records peak child RSS and timing through Python resource accounting;
no external `time` executable is required.

MPS memory requests: CAS12 48G, CAS14 128G, CAS16 320G, CAS18 768G. CAS12 LETTA
retains 64G because tied-state peak memory is not measured; other LETTA jobs use
the corresponding MPS request, provisionally. Override with `--memory` if needed.
Default partitions are gubing,test,testf,test-intel. CAS18 needs an eligible node
with at least 768G allocatable memory; partition capacity has not been verified.

Basis for sizing: the 20 CAS12 jobs used 30.00–30.11G MaxRSS, almost independent
of Dmul. CAS14 jobs were killed at 64G. A topology-only ERI AutoMPO probe on saved
cc-pVDZ integrals found 55,804,306 adjacent-channel pairs at n=12 and 151,051,878
at n=14 (2.707x), before dense allocation. Peak channel counts were 5,128 and
7,820. This is the current dense channel representation's overhead, not an
inherent memory requirement of DMRG. The physical integral array costs O(n^4)
and is below 1 MiB through n=18; MPS coefficients scale approximately as
O(n d Dmul^2), with d the local physical dimension and symmetry reducing storage.
The current builder's approximately cubic channel growth gives a dense storage
estimate approaching O(n^7). Calibrating against CAS12 gives an estimated
82–89G for CAS14, 193–226G for CAS16, and 413–515G for CAS18; only CAS12 has
measured completed-job peak memory. These ranges are extrapolations, not bounds.
The requests add roughly 40–60% headroom to the conservative estimates. No
operator terms, sectors, bond dimensions, or local solver tolerances are removed.

### Compiled chemistry Hamiltonian reuse

`environment.sh` enables `PYQED_QC_MPO_CACHE=$FECO_ROOT/build/qc_mpo` by
default. Jobs using the same integrals, orbital order, electron sector, scalar
offset, cutoff and implementation share one compiled SU(2) Hamiltonian across
MPS bond dimensions and LETTA restarts. The campaign already passes zero core
offset to both solvers, adding it once in the reported energies.

The Python library creates **no disk cache by default locally**. Opt in with
`problem.su2_mpo(cache_dir='/path/to/trusted/cache')` or set the environment
variable. Pass `cache_dir=False` to override the environment; set the variable
to an empty string to disable it in the cluster launcher. Cache entries are
trusted Python pickles, written atomically under a per-key `flock`; waiting
jobs load the completed file instead of duplicating construction. Source and
integral hashes prevent reuse after relevant Python implementation or input
changes. Keep the cache on a shared filesystem supporting `flock`. Obsolete
entries may be deleted when no jobs are using that cache directory.

Compilation contracts sparse magnetic channel blocks with the right suffix
before expanding physical operator coordinates. It keeps the same QR/SVD
thresholds and full intended operator. Reference component cores are expanded
lazily; reduced sweeps use the compiled representation directly. This path
also operates locally, with no disk storage required.

A six-orbital synthetic test (seed 431, three symmetric integral factors,
float64, one BLAS thread, macOS, Python 3.13) measured:

| Stage | Before | Streamed sparse contraction |
| --- | ---: | ---: |
| Builder | 2.281 s | 2.298 s |
| Native compilation | 0.555 s | 0.213 s |
| Peak process RSS | 719 MB | 564 MB |
| Stored native coefficients | 5372 | 5372 |

The same test with disk caching took 2.529 s on the cold call and 0.0045 s
on a fresh-process warm call, with a 66,238-byte cache file and 252 MB warm
peak process RSS (including Python imports).

These are single small-case measurements, not predictions for CAS16. Original
channel dimensions were 1, 57, 131, 277, 505, 705, 1. The builder still allocates
some dense virtual coefficient arrays; this change does not eliminate its
first-build peak memory. Cache hits skip both the builder and compilation,
but do not cache the C++ solver's moving environments or local solves.

Reproduce preparation measurements with `examples/qchem/benchmark_mpo_preparation.py
--orbitals 6` under `PYTHONPATH=.` and single-thread BLAS settings. Add
`--cache-dir /path/to/cache` and run twice in separate processes to measure
cold and warm preparation. `su2_mpo_seconds` includes compilation on a cache
miss; `native_mpo_seconds` is then a memory lookup. No solver time is included.
Existing deployed, signature-checked campaigns must use a new source bundle
to adopt these changes; do not overwrite a bundle serving running jobs.

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
that also tested a one-site competitor per bond. Absolute local residual and
energy-plateau tolerances are 1e-11; cap 200 directional sweeps, requiring two
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

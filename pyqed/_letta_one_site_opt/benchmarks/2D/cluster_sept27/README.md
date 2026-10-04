# September 27 compression comparison

Run on the cluster:

```bash
cd /storage/gubingLab/hushuoyi/letta/letta_cbe_sept_27th
bash submit_compression.sh preflight
bash submit_compression.sh submit
```

The submit command also runs preflight. It creates a fresh frozen plan, then
submits 408 independent tasks on `gubing / huge`, one CPU and 64 GiB each,
using the existing `pyqed_letta_cbe` environment. `--time=0` requests unlimited
walltime; the array has no concurrency cap. Slurm/account policy still applies.

Every case and seed has 17 jobs: one-site control, eight CBE profiles and the
same eight two-site profiles. CBE uses strict comparison against the ordinary
one-site step (`cbe_baseline_guard_fraction=0`). Two-site uses compression
followed by energy refinement. There is no dense Hamiltonian construction.

| Compression profile | ALS outer iterations | LSMR iteration cap |
|---|---:|---:|
| als-default | CBE 4; two-site 8 | CBE 40; two-site size-dependent 50–1000 |
| als40-lsmr40 | 40 | 40 |
| als4-lsmr400 | 4 | 400 |
| als40-lsmr400 | 40 | 400 |
| als100-lsmr2000 | 100 | 2000 |
| variable-projection | 40 on fallback | 400 on fallback |
| joint-ls | 40 on fallback | 400 on fallback |
| grassmann-newton | 40 on fallback | 400 on fallback |

The fixed-cap ALS profiles are the same in both algorithms; some are not
strict increases relative to the adaptive two-site default. Their explicit
budgets allow comparison of outer versus inner work. All nonlinear profiles
use 100 evaluations/iterations, nonlinear tolerance 1e-10 and a 1024 MiB
estimated workspace cap. Variable projection/joint LS count residual
evaluations; Newton counts trust-region iterations. Larger fits can fall
back to ALS40/LSMR400. Exact separable SVD shortcuts remain enabled. Requested
and actual solver usage are recorded: a run with fallbacks is a mixed run.

| Model | Lattices and bond dimensions |
|---|---|
| Bose–Hubbard | 2×2 D2; 3×2 D3/D4; 3×3 D4/D5; 4×3 D4 |
| Ising | 3×3 D4; 4×3 D4 |
| Heisenberg | 3×3 D4; 4×3 D4 |
| Fermi–Hubbard | 3×2 D3; 3×3 D3 |

Both seeds 731 and 1735 are used for each case. All methods start from the
same normalized random MPS embedded in LETTA; fingerprints are saved for
verification. The smaller side of each rectangle is the traversal width.
Model parameters come from `condensed_models.py` and are recorded per job.
Each job runs up to 100 directional sweeps, with early stopping at energy
change per site <= 1e-12 (the existing solver criterion).

Each finished sweep atomically updates its JSON result with energy, elapsed
solver time, solver/fallback counts, nonlinear termination counts and CBE
phase times. Observer I/O time is excluded from reported solver time. Only
final tensor factors are saved; no large dense states or repeated checkpoints
are produced. Final energies are checked by a fresh expectation contraction.

Collect completed and partial results:

```bash
bash submit_compression.sh collect
# Or choose a specific run:
bash submit_compression.sh collect /storage/gubingLab/hushuoyi/letta/letta_cbe_sept_27th/runs/RUN_ID
```

This writes `summary.csv` and `summary.json`, including energy relative to a
completed one-site control at the same D, seed and model, and initial-state
fingerprint agreement. Partial energies can come from different sweep counts.
A `running` result after a kill does not establish that Slurm still runs the
job; inspect `sacct` and logs. Existing result files are never overwritten.

Optional smaller submission, or higher nonlinear memory budget:

```bash
MODELS=bose_hubbard SEEDS=731 bash submit_compression.sh submit
CASES='bose_hubbard:3x3:D4 bose_hubbard:3x3:D5' SEEDS=731 bash submit_compression.sh submit
WORKSPACE_MB=4096 bash submit_compression.sh submit
```

Other controls: `SWEEPS`, `NONLINEAR_ITERATIONS`, `CPUS`, `MEMORY_GIB`,
`ALGORITHMS` (space-separated `one-site cbe two-site`), and `PROFILES`
(space-separated profile names). `bash submit_compression.sh plan` generates
a plan without submitting. Every submission receives a new run directory.

The bundle includes a small frozen source tree and SHA256 manifest. Preflight
checks every bundled file and imported pyqed path, including the newly added
`pyqed/_letta_compression.py` and `pyqed/davidson.py`. Worker paths are passed
through the environment so Slurm's copied-script location cannot redirect
imports. No cluster installation or unrelated old run is modified.

## Local validation

Validated before staging: 56 existing launcher/compression tests and three
new frozen-bundle tests passed. The latter execute all 17 method variants for
two directional sweeps on 2×2 Bose–Hubbard D2 through a copied Slurm-style
worker script, check actual nonlinear dispatch and matching initial tensors,
collect results, reject modified source, and prevent result overwrites.
Slurm submission arguments were captured with a test executable; no cluster
job was submitted during these checks. Cluster preflight verifies its own
Python environment when you run the submit command.

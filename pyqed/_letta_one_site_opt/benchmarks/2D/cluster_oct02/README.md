# 3×12 and 4×12 LETTA comparison

Default: 80 independent jobs, four models × two rectangles × two seeds ×
(one-site, CBE with two compression budgets, two-site with those same budgets),
up to 500 directional sweeps each. Compression budgets are ALS4/LSMR400 and
ALS40/LSMR400. The ALS labels are maximum iteration budgets, not forced counts.
All methods for a case load exactly the same serialized starting tensors.

| Model | D | Parameters |
| --- | --- | --- |
| Ising | 4 | J=1, transverse field h=1, Pauli convention |
| Heisenberg | 4 | J=1, delta=1, h=0, spin-1/2 convention |
| Bose–Hubbard | 4 | t=1, U=4, mu=2, local occupation 0,1,2 |
| Fermi–Hubbard | 3 | t=1, U=4, mu=2, four local states |

Open spatial boundaries, unconstrained particle-number/spin sectors as in the
September27 benchmarks. D is matched between methods within every model/case.
These are 36/48-site models, not three/four-site chains. Internally rectangles
are stored as `(12,3)` / `(12,4)` to keep the C-order frontier short. This is a
rotation of the requested isotropic 3×12 / 4×12 geometry. Result names use the
internal orientation.

## Accuracy revision

This revision includes row-preserving gauges, equilibrated metric support and
compression factors, scale-correct coupled directions, conditioning guards, and
fresh whole-network sweep energies. Normalization is projected out after
whitening, and candidate projection does not promote cancellation noise. ENERGY_REFINEMENT_ITERATIONS defaults to32 and
is recorded in the plan; it controls alternating energy relaxation, separately
from ALS/LSMR compression budgets. This is an accuracy experiment, not a claim
that any 500-sweep endpoint is the correct variational minimum. Multiple starts
and the recorded convergence/energy-check results must be considered.

The earlier `letta_cbe_oct_02` directory remains a frozen, superseded baseline.
Use the separately verified `letta_cbe_oct_02_accuracy_v2` bundle below.

## Submit

```bash
bash /storage/gubingLab/hushuoyi/letta/letta_cbe_oct_02_accuracy_v2/submit_rectangles.sh preflight
bash /storage/gubingLab/hushuoyi/letta/letta_cbe_oct_02_accuracy_v2/submit_rectangles.sh submit
```

Uses the existing `pyqed_letta_cbe` environment, partition `gubing`, QoS `huge`,
one CPU per task, `--time=0`, no array concurrency cap. Default memory is 256 GiB
per task, chosen to include the Fermi–Hubbard 4×12 case, whose heuristic working
memory estimate is 215.6 GiB including compression workspace. This is not a
peak-memory guarantee. All other cases have estimates below 31 GiB. To avoid
reserving 256 GiB for small cases, submit these two disjoint arrays instead of
the default command:

```bash
MODELS='ising heisenberg bose_hubbard' MEMORY_GIB=64 bash /storage/gubingLab/hushuoyi/letta/letta_cbe_oct_02_accuracy_v2/submit_rectangles.sh submit
MODELS='fermi_hubbard' MEMORY_GIB=256 bash /storage/gubingLab/hushuoyi/letta/letta_cbe_oct_02_accuracy_v2/submit_rectangles.sh submit
```

This creates 60 and 20 tasks respectively. Record both printed run directories;
`latest_run.txt` points only to the most recent submission.

## Optional comparisons

All eight September27 compression profiles remain selectable through `PROFILES`:
`als-default`, `als40-lsmr40`, `als4-lsmr400`, `als40-lsmr400`,
`als100-lsmr2000`, `variable-projection`, `joint-ls`, `grassmann-newton`.
The defaults focus on ALS because the completed small cases do not establish a
nonlinear solver advantage and dense nonlinear workspaces may trigger fallback.
Actual solver/fallback counts are recorded at each sweep.

All three algorithms are included by default with the same D and starting tensors.
Use `ALGORITHMS='one-site cbe'` to select only those methods, or
`ALGORITHMS='two-site'` for a separate two-site comparison. There is no proven
convergence or walltime estimate for these larger rectangles. No transfer-map
compression approximation is enabled by this benchmark.

## Results and provenance

```bash
bash /storage/gubingLab/hushuoyi/letta/letta_cbe_oct_02_accuracy_v2/submit_rectangles.sh collect /path/printed/after/submission
```

Each run has an immutable plan, Slurm logs, per-sweep JSON diagnostics and final
tensor NPZ files. Completed energy is independently contracted before marking
completion; nonconvergence at the sweep cap is reported. A source manifest and an
initial-state catalog are verified before every worker. Existing result files
are never overwritten by a retry. A Slurm-copied script uses the exported bundle
path rather than its temporary spool directory.

To stage locally, use `PYTHONPATH=.` with one BLAS/OpenMP thread and run
`stage_bundle.py --repo /path/to/pyqed --output /empty/directory`. Only source,
small documentation/tests, and starting tensors are staged; no old run data,
Git history, caches, environments, or large output files are included.

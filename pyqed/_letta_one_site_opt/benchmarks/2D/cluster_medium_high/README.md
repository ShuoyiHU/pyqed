# Medium LETTA cases with larger inner budgets

24 jobs: Bose–Hubbard and Heisenberg; 3×3 and 3×4 (stored/traversed as 4×3);
D=2,3; seed 731; one-site, strict shrewd CBE, and two-site.
The source and initial tensors are copied from the preceding frozen local batch.
With `stage_bundle.py --recovery-checkout /path/to/pyqed`, the CBE and one-site
driver modules are updated to include numerical step recovery. Other numerical
source and initial tensors stay identical to the parent bundle. The bundle
manifest records whether recovery was included. Always use a new output folder;
do not replace source underneath an existing Slurm run.

| Budget | Old local batch | This batch |
|---|---:|---:|
| ALS rounds per compression subproblem | 4 | 100 |
| LSMR iterations per linear solve | 400 | 2000 |
| Alternating A/B rounds per start | 32 | 256 |
| One-site/CBE outer sweeps | 500 | 2000 |
| Two-site outer sweeps | 20 | 20 |

Each method can stop earlier at its existing tolerance. CBE and two-site both
compare independently relaxed compressed and incumbent starts, so the combined
alternation budget can reach 512 rounds per bond. Progress records include
combined round counts, both-start cap hits, and the selected start. This does
not expose individual LSMR termination codes; the larger caps alone are not
proof of inner convergence.

Other algorithm settings remain the same, including strict CBE baseline
acceptance, expansion dimension 1, selector refinement cap 4, optional coupled
correction cap 8, metric tolerance 1e-10, inner energy tolerance 1e-10, and outer
energy-density-change tolerance 1e-12. Only ALS compression is requested here.
The existing fresh-energy failure guard remains active. Higher budgets are an
accuracy experiment, not a claimed fix for the previously observed energy-check
failure or a guarantee of a global minimum.

The recovery revision checks fresh whole-network energy after every complete
CBE step (including its gauge transformation). A numerical exception, invalid
norm, nonfinite energy, or energy increase beyond `energy_increase_tolerance`
restores all pre-step tensors, clears trial canonical certificates, rebuilds
environments, and retries that active site with ordinary one-site optimization.
CBE resumes at the next bond. If only the gauge fails, the validated one-site
state is kept without applying that gauge. If the one-site retry also fails,
the previous state is kept and the step is marked unresolved. A recovery sweep
cannot trigger convergence. Per-sweep `compression.numerical_recovery` records
the sites, reasons, and rejected one-site retries. Fresh contractions add cost;
this revision prioritizes correctness over sweep time.

To rerun only the two previously failed Heisenberg D3 cases in a recovery bundle:

```bash
ALGORITHMS=cbe CASES='heisenberg:3x3:D3 heisenberg:4x3:D3' bash submit_high_accuracy.sh submit
```

```bash
bash submit_high_accuracy.sh preflight
bash submit_high_accuracy.sh submit
bash submit_high_accuracy.sh collect
```

Submission uses `-p gubing -q huge --time=0`, one CPU and 64 GiB per job,
array 0–23 without a concurrency cap, and environment `pyqed_letta_cbe`.
Slurm partition/QoS policies may still impose limits. The launcher exports the
original bundle path so copied Slurm scripts work. Every run gets a new folder;
completed results cannot be overwritten by repeating a worker index.

Environment overrides include SWEEPS, TWO_SITE_SWEEPS,
ENERGY_REFINEMENT_ITERATIONS, PROFILES, CPUS, and MEMORY_GIB. Save the defaults
for the intended comparison. Each task records its actual options and hashes.

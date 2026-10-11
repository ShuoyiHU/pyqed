Matched Hubbard CBE jobs

Eight random-start jobs matching the earlier one-site models:
0/1: 2x6, D=200/1000, one hole, spin 1/2.
2/3: 2x12, D=200/1000, two holes, spin 0.
4/5: 2x18, D=200/1000, three holes, spin 1/2.
6: 4x6, D=200, two holes, spin 0.
7: 4x12, D=200, four holes, spin 0.
D counts SU(2) multiplets. U/t=12, rung=leg=t, diagonal hopping=-0.25t,
V=0, open boundaries, column-major order, seed 4. Initializer and graph
match the previous one-site bundles; initial tensor hashes are recorded.

Maximum 400 directional sweeps, energy-change tolerance 1e-6 per site.
Inner stationarity tolerance remains 1e-10 for the two-leg comparisons and
1e-6 for the four-leg comparisons, matching their previous settings.
CBE expansion=1; ALS and energy-refinement caps=32; projection cap=1000.
Direct projection workspace defaults to 256 MiB; larger maps use LSMR.
Every CBE acceptance is compared with the ordinary one-site candidate.

This directory is a template until stage_bundle.py freezes a committed source
and fills SOURCE_PATH and the plan's revision. Do not submit the template.
After staging: bash submit.sh check, then bash submit.sh submit.
Optional subset: INDICES=0,2,4 bash submit.sh submit.
Defaults: gubing / huge; override PARTITIONS or QOS if needed.
No array concurrency cap and no requested time limit.
Memory requests: 64G for two-leg D=200, 256G otherwise. These are reservations,
not measured peak predictions. memory.jsonl records actual process use.

The shared pyqed_letta_su2 conda environment is reused; python.sh selects the
immutable source snapshot explicitly. There is no environment cloning or
replacement of the advisor package. Submission verifies source checksums,
the Hubbard dimer energy, and focused CBE regression tests before any sbatch.
The worker receives the frozen run directory explicitly, including when
Slurm copies the launcher. Source, plan, scripts, and logs remain attributable.

Plot finished and ongoing jobs:
  bash python.sh plot_results.py RUN_DIRECTORY

# September 18 large 2D LETTA runs

Run on the cluster login node:

```bash
cd /storage/gubingLab/hushuoyi/letta/letta_cbe_sept_18th
bash submit_large_2d.sh submit
```

This activates `pyqed_letta_cbe` from
`/share/home/gubingLab/hushuoyi/miniconda3` and loads the synchronized code at
`/share/home/gubingLab/hushuoyi/software/pyqed_bg_letta_cbe` via `PYTHONPATH`.
No installation or environment modification is required. `preflight` imports
the required libraries and prints the actual Python, package versions and source.

Defaults: 4 models × 13 shapes × 2 bond dimensions × 2 seeds × 3 solvers =
**624 independent array tasks**. Each solver has its own process, result and
log, so a slow two-site job cannot delay or erase one-site/CBE results.

- Models: Ising, Heisenberg, Bose-Hubbard (`max_occupancy=2`) and Fermi-Hubbard.
  Physical parameters are the existing `condensed_models.build_model` defaults
  and are saved in each result. The Hubbard runs are grand canonical, without
  a fixed particle-number constraint.
- Shapes: 3×3, 3×6, 3×9, 4×4, 4×8, 4×12, 5×5, 5×10, 6×6, 6×12, 7×7, 8×8, 9×9.
- D=4,8; seeds=731,732; up to 100 directional sweeps (50 LR/RL cycles),
  energy-density tolerance 1e-9; CBE expansion dimension=1.
- One-site, strict CBE, and two-site with metric ALS plus alternating A/B energy
  refinement. No exact-CBE oracle or dense exact diagonalization is run.
- Rectangles are rotated to put the shorter side along the C-order chain width:
  requested 3×9 is contracted as 9×3, and 4×12 as 12×4. For these isotropic open
  models this preserves the physical lattice, while reducing frontier cost.
  Both requested and contraction shapes are recorded. Ties are positive nearest
  neighbors; this is not a diagonal/bidirectional-tie study.
- Deterministic MPS initialization embedded in LETTA; all three methods use the
  same factors per model/shape/D/seed. The collector checks their fingerprints.
- 8 CPUs and 256 GiB per task; `-p gubing -q huge`, `--time=0` (unlimited
  requested walltime), and `--array=0-623` with **no `%` concurrency cap**.
  Slurm partition/QoS/account limits still apply; scripts do not override them.

## Choose the run size

### Scheduler QoS

The initially requested `large` QoS was rejected by this cluster. The account
query on September 19 returned `gpu,gpu-huge,huge,normal`, with default `normal`.
The launcher now defaults to `huge`, matching the previous large CPU runs.
Override with `QOS=<allowed-name>`; `QOS=default` (or an empty value) omits `-q`
and uses the account default. Check current allowed values with:

```bash
sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS
scontrol show partition gubing
```

After a rejected submission, reuse the existing plan, for example:

```bash
QOS=huge PLAN=/storage/gubingLab/hushuoyi/letta/letta_cbe_sept_18th/runs/20260919-100025-1836280/plan.json \
  bash submit_large_2d.sh submit
```

`PARTITION` defaults to `gubing`. Explicit QoS overrides are passed unchanged.
The unlimited walltime request and uncapped array remain unchanged.

Inspect the manifest without submitting:

```bash
bash submit_large_2d.sh plan
```

Examples (environment settings apply when creating the plan):

```bash
# Omit two-site entirely.
SOLVERS='one_site cbe' bash submit_large_2d.sh submit

# Same large study, with a separate two-site pass cap.
TWO_SITE_MAX_SWEEPS=20 bash submit_large_2d.sh submit

# Start with requested narrow strips; one seed and bond dimension.
SHAPES='3x3 3x6 3x9 4x4 4x8 4x12' BOND_DIMS=4 SEEDS=731 \
  bash submit_large_2d.sh submit

# Request more memory. CPUS, MAX_SWEEPS, MODELS, SEEDS are also configurable.
MEMORY_GIB=512 CPUS=16 bash submit_large_2d.sh submit
```

Each invocation creates `runs/<timestamp>-<pid>/plan.json`; it is never silently
overwritten. `PLAN=/absolute/path/to/plan.json` reuses an existing plan rather
than generating another. To retry incomplete results with that plan use
`FORCE_RERUN=1`; completed cases are skipped by default. Use a new run directory
when changing the synchronized source to avoid mixing code versions.

## Memory and incomplete cases

Large lattice size is not the only cost: exact Hamiltonian environments scale
with the square of the open physical frontier dimension. Wide 9×9 Hubbard
cases are generally beyond these allocations with the current exact solver.
The full requested grid stays in the manifest; no method is replaced with an
approximate algorithm to make those cases appear feasible.

Before allocating a Hamiltonian, every job counts physical frontier incidences
and estimates the saved dense environments and product-term MPO. The guard
uses `3 × saved storage + 8 GiB` for working space. This is a heuristic, not a
guarantee: CBE/ALS intermediates and native solver workspace may be larger.
A case exceeding the allocation records `resource_blocked` and exits nonzero.
It is **not** counted as completed or converged. To attempt all cases regardless
of this estimate, set `ALLOW_LARGE_MEMORY=1`; Slurm may then terminate them for
exceeding the actual allocation. Increasing MEMORY_GIB also raises the guard.

## Results

Each case writes `runs/<id>/results/<case>/<solver>.json` atomically before the
solve and at completion/error. Sweep energies are printed live to
`runs/<id>/logs/<job>_<task>.out`; completed JSON includes sweep energies/times,
convergence, CBE diagnostics, peak RSS, package versions and a source-manifest
hash. Final wavefunctions and dense vectors are not stored.

Use the exact PLAN path printed at submission to collect completed and partial
results at any time:

```bash
PLAN=/storage/gubingLab/hushuoyi/letta/letta_cbe_sept_18th/runs/RUN_ID/plan.json \
  bash submit_large_2d.sh collect
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

`summary.csv` compares energy against one-site and reports missing, incomplete,
failed and resource-blocked jobs. A native crash, external cancellation or OOM
can leave `starting`/`running` in JSON; consult Slurm accounting for the terminal
status. An energy-change convergence flag can mean stagnation, especially for
two-site; compare energies and histories before claiming a common minimum.

# 3×6 comparisons, October 4

40 independent jobs: four models × two seeds × five method/profile combinations.
Ising, Heisenberg and Bose–Hubbard use D=4; Fermi–Hubbard uses D=3.
Seeds are 731 and 1735. The internal shape is (6,3), retaining the same
three-site frontier orientation as the previous (12,3) calculations.

Methods: one-site, CBE/ALS4, CBE/ALS40, two-site/ALS4, two-site/ALS40.
Each case/seed starts from identical serialized tensors across methods.
The solver source is copied byte-for-byte from the October 2 accuracy-v2 bundle,
not from subsequently modified local solvers. The parent manifest is retained.

Unchanged numerical settings: 500 directional outer sweeps; energy tolerance
1e-12; metric/compression tolerance 1e-10; ALS caps 4 or 40; LSMR cap 400;
alternating energy-refinement cap 32; nonlinear cap 100; workspace budget 1024 MiB.
These are maximum iterations, not a guarantee of inner convergence. Existing
convergence criteria, energy acceptance rules, and fresh energy checks are preserved.

Submit with `bash submit_rectangles.sh submit`. The launcher first verifies the
frozen files and asks Slurm to validate the resource request with `--test-only`.
`bash submit_rectangles.sh check` performs validation without submitting jobs.
`bash submit_rectangles.sh plan` only writes a plan and needs no scheduler.

The single array is eligible for `gubing,test,testf,test-intel`; Slurm chooses
one eligible partition per job. This does not duplicate jobs or guarantee equal
distribution. Defaults: QoS huge, 1 CPU, 256 GiB, unlimited requested walltime,
no array concurrency cap. Cluster partition/QoS limits still apply. PARTITIONS,
QOS, MEMORY_GIB, CPUS and numerical budgets are explicit environment overrides.
Submission stops if Slurm rejects the request; it does not silently drop partitions.

Results are written after each sweep in `runs/<submission>/results`.
The last attempted history entry can be rejected by the outer energy check;
collection/plotting must distinguish accepted states from rejected trials.

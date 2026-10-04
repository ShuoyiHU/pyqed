# Stationary excited starts in the matrix-free local solve

The integrated generalized Davidson solver could return an exact excited eigenvector as its lowest local root. A 32-coordinate diagonal problem with incumbent e0 and a lower eigenvalue at the last coordinate returned 0 instead of -2. The residual was exactly zero; deterministic coordinate fallback explored only the first 16 coordinates before stopping.

An independent regression uses 64 physical coordinates, an exact excited incumbent of energy 0.25, and a ground energy -2 outside the initial coordinate prefix. It tests a diagonal Hamiltonian, a disconnected complex rotated subspace, and an embedding with redundant/zero coordinates and large diagonal scaling. Before correction, all three matrix-free cases returned 0.25; their dense controls passed. This is a local root-search defect, not a limitation of the LETTA ansatz or a global optimization comparison.

The solver now adds two reproducible full-support complex probes to the metric-orthonormalized initial Ritz space. Adding probes alone did not pass the regression: their initial Rayleigh quotients were above the stationary incumbent, leaving the incumbent residual zero. The iteration must also follow residuals of the next two lowest Ritz roots. It now expands along the first independent unconverged residual among those roots, preserving the existing coordinate fallback and restart with up to four low Ritz vectors. Probes do not perturb the Hamiltonian or represented incumbent, and the same local metric/support handling is retained.

The reference problems now reach the independent lowest eigenvalue and eigenvector. An additional regression verifies that a warm start already inside a degenerate ground space is preserved. Random exploration is not an unconditional certificate for a finite capped iterative lowest-root solve; this limitation is stated explicitly in the mathematical companion. Production acceptance still uses a fresh conditioned physical energy, and convergence retains its local residual checks.

Validation on the final numerical source for this correction:

- Before: 3 matrix-free failures, 3 dense controls passed.
- After: all 15 scaling/exploration/support/degeneracy tests passed as part of the combined set below.
- 110 reduced regressions passed in 54.79 s: scaling/exploration, one-site/norm, shared updates, sweep/resource recovery, CBE, physical Schmidt truncation, two-site and real molecular roundoff acceptance.
- 3 native checks passed (17 deselected) in 5.02 s: one-/two-site sweeps with magnetic/global expansion forbidden and the six-orbital metric-nullspace stress case.
- 64 public API checks passed (15 deselected) in 52.75 s: independent model/input checks, all molecular compressors, every open method/model combination, spin rings and topology/tie independence.
- 27 cyclic matrix-free/compressor/failure/workspace checks passed (35 deselected) in 17.88 s.
- Original-source hashes rechecked unchanged: 302 bg files and 300 QC files.
- git diff --check passed.

Logs: /private/tmp/letta-root-exploration-{before,after,regression,native,public,ring}.log. Python was the main checkout's .venv-1 with PYTHONPATH=., bytecode/cache output outside the worktree and single-thread BLAS/OpenMP.

The six final-source Bose/SU(2)-fermionic ring method cases are still running in session 22749, with verbose output in /private/tmp/letta-public-matrix-bose-su2-final.log. The older original full public test process remains session 41861 / PID 91584, log /private/tmp/letta-public-api.log; it predates both this correction and metric equilibration and is not final-source validation. Preserve its outcome, including the documented obsolete CBE fixtures. Remaining matrix/allocation verification and the final implementation-note audit still gate the performance phase.

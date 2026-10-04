# Local generalized-eigensolver equilibration — 2026-10-05

Reviewing the implementation derivation exposed inconsistent rank decisions: compression equilibrated Gram columns, but local dense solves truncated eigenvalues of the raw Gram and matrix-free Davidson used raw-coordinate support/scale tests.

An independent two-dimensional physical Hamiltonian with eigenvalue -1.276208734813001 was embedded in three redundant coordinates. Invertible diagonal rescaling made the old dense solver return 0.25 or -1.25, and one old matrix-free case returned 0.25. The unscaled problem passed. This is a lost physical direction, not a changed ansatz or an iteration-cap comparison.

The dense solver now uses the existing equilibrated Gram factorization. The matrix-free solver obtains only the Gram diagonal through norm actions, applies inverse diagonal scaling to both H and N, and performs Davidson iterations in those coordinates. The normalized Gram trace bounds its PSD spectral norm in numerical dependence tests. Native conditional one-/two-site support projectors are built from equilibrated boundary Grams. Raw-coordinate projectors cannot be applied unchanged after this coordinate transformation. Obsolete raw scale/projector fields were removed from the local problem adapters.

The Hamiltonian, physical metric and retained symmetry sectors are not approximated to implement this change. Numerical support thresholds remain explicit. Matrix-free diagonal construction currently costs one norm action per source coordinate; no runtime improvement is claimed. Arbitrary tied/closed-ring layouts without an analytic projector continue to use the iterative metric-orthogonality checks.

New tests cover three coordinate scalings, dense and matrix-free paths, redundant columns, comparison with an independent physical eigenvector, normalized metric norm, and native conditional one-/two-site projectors after large legal virtual gauges. Native projector tests check that the physical metric action is preserved, the projector is idempotent, and it does not increase Euclidean norm in equilibrated coordinates.

Final evidence (overlapping sets; counts must not be added as unique tests):

- 103 passed in 51.39 s: eigen scaling, real molecular roundoff acceptance, reduced one-site/norm/updates/gauge recovery/resource recovery/CBE/Schmidt/two-site.
- 26 passed, 36 deselected in 17.21 s: ring matrix-free solve, all compressors, failure/recovery/workspace checks.
- 42 passed, 37 deselected in 41.07 s: current public model/input and all four molecular CBE/two-site compressor selections.
- 11 passed, 17 deselected in 4.33 s before removal of unused adapter fields: new scaling/projector cases plus native no-expansion one-/two-site sweeps and six-orbital matrix-free nullspace stress regression.
- 7 mathematical derivative/root/gauge checks passed before this eigensolver correction; those test the derivation's existing compression routines rather than the correction itself.
- git diff --check passed. The 302 bg and 300 QC source hashes remain unchanged.

Test logs remain under /private/tmp/letta-eigen-scaling-{final,ring-final,native}.log and /private/tmp/letta-public-api-post-equilibration.log. Tests ran with the main checkout's .venv-1 Python, PYTHONPATH=., bytecode/cache output outside the worktree, and single BLAS/OpenMP threads.

The original full public API process (session 41861, PID 91584) remains live and predates this eigensolver change. Its eventual result is earlier-source evidence, not final-source verification. Do not restart it merely because it is slow; preserve its outcome and account for its older compressor fixtures as documented in the public API plan. The broader final support matrix and the full implementation-note audit are still required. The optional performance phase has not started.

# Reduced MPS sweep engine port

Source: upstream/bg `8204ae6a0e4deec3cc5c6c4e4c0d43da5e1d7c18`, snapshot
`f0a36f8612909c5d1063effdb23ba00dd4ff371a`. Five engine source files are
copied from that snapshot; the Davidson header has the local correction described below; `upstream_sweep_sources.json` records their provenance.
Exact reduced QR routines are adapted to the existing NonabelianTensor type.

Build explicitly, outside calculation timing:

```
PYTHONPATH=. python -m pyqed.mps.nonabelian.sweep_engine
```

Then, with an ElectronicProblem and fully reduced initial MPS:

```
from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG
solver = ReducedDMRG(problem, sites, max_bond=30, threads=1)
solver.run(max_sweeps=100, residual_tolerance=1e-10)
# solver.sites is directly importable into ReducedLatticeLETTA.from_mps.
```

The cap is the SUM of reduced multiplets across sectors. One sweep means
one direction. No singular-value cutoff is introduced. Energy convergence
uses two consecutive directional changes in total energy; local roots must
meet the absolute residual tolerance or fail. Each exported state is evaluated
independently by the existing reduced Hamiltonian contraction. Truncation
can raise energy, so monotonicity is not assumed. A plateau is not proof of
global optimality. Inputs are copied; no determinant-space projection is used.

This is an opt-in ordinary two-site MPS driver. Existing LETTA, CBE, cluster
scripts, and the older generic MPS driver are not silently redirected. It
supports real spatial-orbital Hamiltonians and spin-charge SU(2) sectors;
complex states and point-group sectors are not part of this adapter's scope.
Linux builds use OpenMP; the macOS build uses Accelerate and is tested with
one thread. New builds are keyed by source and Python ABI. Do not run two
builds into the same cache simultaneously.

Algorithm references: White, PRL 69, 2863 (1992),
https://doi.org/10.1103/PhysRevLett.69.2863; Schollwoeck, Ann. Phys. 326,
96 (2011), https://doi.org/10.1016/j.aop.2010.09.012. This is a reduced SU(2)
implementation of two-site DMRG, with upstream persistent environments and
contraction plans plus our independent energy verification; it does not add
CBE or transfer the entire upstream LETTA implementation.

Local accuracy fix: normalize each preconditioned correction before the
linear-dependence test in the real and complex single-root standard Davidson
solvers. The upstream absolute correction-norm threshold could reject an
independent direction while the residual was still above tolerance (reproduced
on 2x3 U=4, half filling, seed 7). The residual tolerance itself is unchanged.
The provenance manifest contains original upstream hashes, not local hashes.

## Comparison with the existing LETTA-based MPS driver

The existing `letta_two_site_dmrg` also solves a one-site competitor at every
bond and may keep it instead of the truncated two-site candidate. This port
implements upstream plain two-site DMRG; these are different update policies.
The safeguarded driver remains unchanged. In truncated cases the safeguard
can attain a lower energy in the same sweep count. Do not replace that driver
silently or describe plain and safeguarded runs as identical algorithms.
The matched benchmark additionally runs a plain two-site version of the
Python reference, only in the benchmark, to separate engine accuracy from
this policy difference. Production LETTA safeguards are preserved.

Cluster restart: `run(start_direction="rl")` continues after a completed left-to-right pass, with the corresponding exact initial gauge.
Linux links LP64 BLAS/LAPACK explicitly (default `blas,lapack`, searching the Python environment's `lib` directory). Override `PYQED_SWEEP_LIBRARIES=openblas` only for an LP64 OpenBLAS installation. Build settings and the build script are included in the cache fingerprint.

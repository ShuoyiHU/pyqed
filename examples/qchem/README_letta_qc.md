# Singlet quantum chemistry with reduced SU(2) LETTA

This example packages the Fe(CO)5 workflow used in the local CAS(10,10)
experiments. The reusable code lives in:

- `pyqed/_letta_one_site_opt/qchem.py`: spatial CAS integrals and Hamiltonian.
- `qchem_workflow.py`: `QCLETTA`, source-sector-preserving embedding, checkpoints,
  reduced coefficient counts and removal of construction-only MPO storage.
- `orbital_information.py`: fermionic one-/two-orbital entropies from MPS
  transfers, mutual information, conditional entropy and negative-conditional-entropy ties (default) or greedy disjoint MI ties.
- `reduced_solver.py`: existing one-site solver, now with an optional completed-
  sweep callback. The two-site algorithm remains in `_letta_two_site_opt`.

The driver supports real spatial integrals in chemists' notation, even-electron
singlets, fixed orbital ordering, and U(1) particle number plus SU(2) spin.
It uses the existing reduced solvers; it does not introduce a determinant-space
projection into their variational path. No orbital optimization, automatic
Fiedler reordering, or dynamic discarded-weight target is performed here.

## Quick verification example

From the repository root, with NumPy, SciPy, opt_einsum, SymPy, Cython,
matplotlib and PySCF available (plus pyqed's usual dependencies):

```sh
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python examples/qchem/letta_qc.py --molecule h2 --bond-dim 4 \
  --two-site-sweeps 2 --polish-sweeps 2 --letta-sweeps 2 \
  --reference --output /tmp/letta-h2
```

The example saves integrals, an atomic checkpoint, a JSON report, energy-versus-
sweep/time figures, and orbital-information plots. FCI and full wavefunction
reconstruction are optional small-system validation only, restricted to at most
12 orbitals in this example. They do not select the ties or optimize the state.
A finite sweep limit is reported honestly: `success` is true only when the
native one-site energy and fresh local residual tests pass. A small energy
change alone is not labelled convergence.

## Fe(CO)5 CAS(10,10), 6-31G

The exact geometry is included in `letta_qc.py`. Generate canonical RHF active-
space integrals and run:

```sh
python examples/qchem/letta_qc.py --molecule feco5 --basis 6-31g \
  --ncas 10 --nelecas 10 --bond-dim 30 --two-site-sweeps 39 \
  --polish-sweeps 600 --letta-sweeps 600 \
  --reference --output /tmp/letta-feco5
```

This freezes 43 doubly occupied orbitals and uses MO44 through MO53 (one-based).
RHF is checked for convergence; this example does not perform an RHF stability
analysis. New RHF calculations can rotate degenerate orbitals, so use saved
integrals for exact comparisons to a previous experiment:

```sh
python examples/qchem/letta_qc.py --integrals /path/to/integrals.npz \
  --bond-dim 30 --two-site-sweeps 39 --polish-sweeps 600 \
  --letta-sweeps 600 --edges 1-5,2-6,3-9 \
  --reference --output /tmp/letta-feco5-three-ties
```

The NPZ keys are `h1`, `eri`, `nelec` (alpha,beta), `ecore`, and optionally
`orbital_order`. Explicit edges are **zero-based positions in the loaded chain**:
`1-5,2-6,3-9` means MO45–49, MO46–50, MO47–53 in the original Fe(CO)5 window.
Each pair becomes one forward dependency; `3-9` conditions tensor 3 on orbital
9's reduced occupancy label. Reversing the textual pair does not create a
backward tie. There are no carried dependencies or extra nearest-neighbor ties.

By default, the example stops before tying if MPS polishing has not passed its
stationarity audit. `--allow-unconverged-mps` explicitly permits an approximate
pilot MPS, and reports its unconverged status. Default tie selection includes **every pair with S(i|j)<0 or S(j|i)<0**,
using a strict zero threshold. This can include overlapping pairs; no disjointness
or frontier-width cap is silently imposed. Edges are represented in forward
chain order, regardless of which conditional direction was negative. Explicit
edges override selection. Use `--tie-selection mi --ties 2` to reproduce the
previous strongest-positive-MI disjoint-pair heuristic.
Three or more long-range ties can substantially increase frontier contraction
cost even when their stored parameter count is modest.

## Checkpoints and continuation

A checkpoint stores the tensor state and matching history after every completed
sweep, using an atomic file replacement. No introspection or monkey-patching of
the solver is required. Continue **trusted local checkpoints only** (pickle is
not safe for untrusted inputs):

```python
import numpy as np
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA

with np.load('/tmp/letta-feco5/integrals.npz') as data:
    problem = ElectronicProblem(data['h1'], data['eri'], tuple(data['nelec']),
                                float(data['ecore']), tuple(data['orbital_order']))
job = QCLETTA.load(problem, '/tmp/letta-feco5/checkpoint.pkl')
# Choose the unfinished stage; counts below are ADDITIONAL sweeps.
job.optimize_letta(max_sweeps=200)
# For an untied checkpoint, use job.polish(200), or job.two_site(remaining).
print(job.energy, job.success, job.message)
```

Input integrals, electron sector, core offset and orbital order are hashed and
checked on load. Sweep direction is retained across continuation. Keep the same
pyqed revision when resuming: arbitrary cross-version pickle compatibility is
not guaranteed. The example CLI starts a new workflow; use the API above to
resume and call the example's `plot(job.report(), output_path, reference_energy)`
to regenerate figures. A checkpoint made before the final convergence audit
conservatively records `success=False`.

## Dimensions, diagnostics and comparison limits

`bond_dim` is the maximum total number of SU(2) multiplets on each source bond:

\[
D_{\mathrm{mul},k}=\sum_{N,S}m_{k,N,S},\qquad
D_{\mathrm{mag},k}=\sum_{N,S}(2S+1)m_{k,N,S}.
\]

Two-site sweeps can redistribute sectors under that cap. One-site polishing
and LETTA retain the prepared source sectors and multiplicities. Direct ties
add reduced physical conditioning axes, each of dimension 3 (empty, single,
double), not independent spin-projection coefficients. `parameter_counts`
counts stored scalar coefficients, potentially complex, before gauge reduction.
It does not count CG coefficients or claim independent real manifold dimensions.

The solver removes the constant `ecore` from local eigenproblems and restores it
once in reported total energies. Fixed-sweep two-site preparation uses the
existing conditional-SVD split, including its possible truncation energy
increases; it is not presented as a monotone fully converged solver.

For diagnostics, the untied reduced MPS is expanded into *local* magnetic tensors
with physical dimension 4. Fermionic parity strings are included for nonadjacent
orbital pairs. The routine assumes definite total parity (guaranteed here by
fixed N). It forms n one-orbital 4x4 RDMs and n(n-1)/2 two-orbital 16x16 RDMs,
not a full 4^n vector. With fixed physical dimension, transfers scale cubically
in the magnetic bond dimension M; open intermediates scale quadratically in M.
Explicit sequential contractions avoid the previous unnecessary quartic transfer.

Using natural logarithms and **no factor 1/2**:

\[
I_{ij}=S_i+S_j-S_{ij},\qquad S(i|j)=S_i-I_{ij}.
\]

Conditional-entropy diagonals are undefined here and masked. Negative values
are allowed. Both selection rules are heuristics, not optimal graph searches or proven
energy-gain criteria. Strict negativity near floating-point zero should be
interpreted with numerical care; inspect explicit edges for expensive cases. Ordering and tie orientation affect cost and accuracy.
The old experiment compared canonical-order networks; no superiority over
optimally reordered DMRG is implied.

Reported sweep timings exclude checkpoint writes; one-site initialization is
included in the first sweep, and fresh convergence audits between sweeps are
included in the next interval. A terminal audit is not included in a subsequent
sweep interval. Hamiltonian preparation is reported separately. These timings
are not a substitute for isolated, matched-scope performance benchmarks.
Compacting the MPO removes construction-only retained data but does not eliminate
initial Hamiltonian-builder peak memory.

## References and fidelity

- Ö. Legeza and J. Sólyom, *Optimizing the density-matrix renormalization group
  method using quantum information entropy*, Phys. Rev. B **68**, 195116 (2003),
  https://doi.org/10.1103/PhysRevB.68.195116. Orbital entropy/MI diagnostic;
  the present transfer implementation and tie-selection rules are adaptations,
  not a reproduction of the paper's complete optimization procedure.
- M. Horodecki, J. Oppenheim and A. Winter, *Quantum state merging and negative
  information*, Commun. Math. Phys. **269**, 107–136 (2007),
  https://arxiv.org/abs/quant-ph/0512247. Meaning of quantum conditional entropy;
  this workflow computes that quantity and does not implement state merging.

Run the focused regression suite with:

```sh
python -m pytest tests/test_letta_qc_workflow.py tests/test_letta_qchem_symmetry.py -q
```

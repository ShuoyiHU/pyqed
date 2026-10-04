# 2D LETTA comparisons

These scripts compare fixed-bond one-site LETTA, general strict CBE, and
ALS-plus-energy two-site LETTA from **the same initial physical state**.

- `ising.py`: transverse-field Ising.
- `heisenberg.py`: XXZ Heisenberg.
- `bose_hubbard.py`: truncated bosons, with `--max-occupancy`.
- `fermi_hubbard.py`: spinful fermions with Jordan–Wigner strings.
- `compare.py --model MODEL`: common entry point.

Default: open 2×2 lattice, D=2, 10 directional sweeps, seed 731. The exact
reference is computed only for Hilbert dimension at most 256. Full norm
initialization contracts MPS environments without expanding the physical state.

From the repository root, with the environment's Python:

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python pyqed/_letta_one_site_opt/benchmarks/2D/ising.py \
  --rows 2 --columns 3 --bond-dim 2 --max-sweeps 30 \
  --output /private/tmp/letta-ising-2d.json

python pyqed/_letta_one_site_opt/benchmarks/2D/fermi_hubbard.py \
  --rows 2 --columns 2 --bond-dim 2 --skip-two-site \
  --output /private/tmp/letta-hubbard-2d.json

python pyqed/_letta_one_site_opt/benchmarks/2D/heisenberg.py \
  --rows 2 --columns 3 --max-sweeps 30 --two-site-max-sweeps 4 \
  --tie-pattern diagonal --output /private/tmp/letta-diagonal.json
```

`--skip-two-site` prevents the two-site solver from being called. In an IDE,
set `INCLUDE_TWO_SITE = False` in the model file for the same default selection.
`--two-site-max-sweeps` sets a separate cap while retaining the one-site/CBE
budget. A cap is reported as non-convergence when the tolerance was not met;
compare converged energies before drawing conclusions about accuracy.
The convergence flag measures energy change only. It can also report
stagnation after rejected/truncated updates, so it does not certify a common
variational minimum. Check the reported energy differences and histories;
the small-D 2D Hubbard examples can stop above the one-site energy.

Use `--help` for model parameters and tolerances. `--solvers` permits an
explicit selection; exact CBE and ordinary MPS are optional, not defaults.
`--output` saves per-sweep energies/timestamps, failure information, actual
physical dependencies, common initial-state fingerprint, and both sweep caps.
`--json` prints the same report. Errors raise by default rather than silently
omitting a failed optimizer.

## Physical ties

Hamiltonian interaction bonds and LETTA tensor dependencies are independent.
All model Hamiltonians remain the requested 2D nearest-neighbor models.

- `lattice`: existing positive-neighbor ties (default).
- `diagonal`: also tie each tensor to its positive diagonal neighbor.
- `bidirectional`: tie to both positive and negative nearest neighbors.
- `--ties-json FILE`: explicit list of physical arguments per tensor. Each
  entry starts with its home site; additional entries may precede or follow it
  in the chain. All entries must be distinct and in range. Site numbering is
  C-order: `site = row * columns + column`.

For a 2×2 lattice, this is a valid nonlocal/backward example:

```json
[[0, 2, 3], [1, 3], [2, 0], [3, 1]]
```

The same interface is available directly:

```python
from pyqed._letta_one_site_opt import LatticeLETTA
state = LatticeLETTA.random(
    (2, 2), bond_dim=2, seed=731,
    neighborhoods=((0, 2, 3), (1, 3), (2, 0), (3, 1)),
)
```

Pass this state to any of the three solvers. Copies and bond expansion retain
its dependencies. General ties enlarge physical frontiers; exact contraction
cost depends on those frontier dimensions, not just D. CBE still uses the
physical metric and shared active labels as in the category theory note. It
has no universal one-site-cost guarantee for arbitrary ties.

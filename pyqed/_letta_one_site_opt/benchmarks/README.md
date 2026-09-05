# LETTA condensed-model benchmarks

This directory contains four model families in both one and two dimensions:

| Model | Hamiltonian conventions | Local dimension |
|---|---|---:|
| Ising | \(-J\sum_{\langle ij\rangle}Z_iZ_j-h\sum_iX_i\) | 2 |
| XXZ Heisenberg | \(J\sum_{\langle ij\rangle}[\tfrac12(S_i^+S_j^-+S_i^-S_j^+)+\Delta S_i^zS_j^z]-h\sum_iS_i^z\) | 2 |
| Bose--Hubbard | \(-t\sum_{\langle ij\rangle}(b_i^\dagger b_j+\mathrm{h.c.})+\tfrac U2\sum_i n_i(n_i-1)-\mu\sum_i n_i\) | `max_occupancy + 1` |
| spinful Fermi--Hubbard | \(-t\sum_{\langle ij\rangle,\sigma}(c_{i\sigma}^\dagger c_{j\sigma}+\mathrm{h.c.})+U\sum_i n_{i\uparrow}n_{i\downarrow}-\mu\sum_i(n_{i\uparrow}+n_{i\downarrow})\) | 4 |

Lattices have open boundaries and C-order site numbering.  A 1D chain is the
native LETTA shape `(1, length)`; 2D is `(rows, columns)`.  Fermionic hopping
uses exact Jordan--Wigner strings in that ordering.  These are grand-canonical
benchmarks: particle-number sectors are not fixed.

Every script compares:

1. fixed-bond one-site LETTA;
2. exact-selector LETTA-CBE (small-system oracle);
3. strict streamed LETTA-CBE;
4. two-site LETTA; and
5. conventional two-site MPS DMRG.

All five receive copies of one normalized random MPS.  The LETTA copy is an
exact embedding of the same physical wavefunction, so the initial-state hash
and initial energy agree.  Equal nominal bond dimension does **not** mean equal
parameter count: LETTA retains positive-neighbor physical dependency axes and
is more expressive than an ordinary MPS, including for shape `(1, length)`.

## Click-run entry points

Run any file directly from an IDE or terminal; repository-root `PYTHONPATH` is
inserted automatically.

```text
ising_1d.py                 ising_2d.py
heisenberg_1d.py            heisenberg_2d.py
bose_hubbard_1d.py          bose_hubbard_2d.py
fermi_hubbard_1d.py         fermi_hubbard_2d.py
```

For example:

```bash
python pyqed/_letta_one_site_opt/benchmarks/heisenberg_2d.py \
  --rows 3 --columns 4 --bond-dim 4 --expansion-dimension 1 \
  --max-sweeps 8 --J 1.0 --delta 1.2 --h 0.1

python pyqed/_letta_one_site_opt/benchmarks/bose_hubbard_1d.py \
  --length 8 --bond-dim 6 --max-sweeps 10 \
  --t 1.0 --U 6.0 --mu 2.5 --max-occupancy 3
```

Use `--help` on a file for its size and model parameters.  Common controls are
`--bond-dim`, `--expansion-dimension`, `--max-sweeps`, `--seed`, `--tolerance`,
`--exact-max-dimension`, `--cbe-baseline-guard-fraction`, and a comma-separated
`--solvers` subset.  `--json` prints the complete machine-readable report.
Exact diagonalization is skipped when the Hilbert dimension exceeds
`--exact-max-dimension`.

Run all eight current defaults (`D=4`, 50 sweeps) with:

```bash
python pyqed/_letta_one_site_opt/benchmarks/run_condensed_suite.py
```

The Hamiltonians use an exact direct-sum product-term MPO.  It is deliberately
simple and independently testable rather than minimally compressed; the strict
CBE path enumerates its sparse transitions.  The automatic strict-CBE
preselection width is
`min(D + 2*deltaD, left parent size, right parent size)`, so it stays moderate
and is not inflated by a redundant MPO bond representation.  Larger production
calculations should still use compressed model-specific MPO builders to reduce
the number of sparse paths and the Hamiltonian-contraction prefactor.

## Reading the diagnostics

`cbe-ok/fallback` counts accepted trimmed expansions and ordinary one-site
fallbacks.  The JSON report also includes the ordinary-candidate selection
count, mean CBE-minus-baseline energy, guard allowance, missing norm, retained
weight, trim loss, parameter counts, and pair-operation/materialization flags.
For strict CBE, `missing` is the norm of the raised, tangent-projected physical
residual in the temporary expanded one-site metric.  All pair-action,
pair-metric, and merged-pair counters must be zero.  Exact and strict
retained-weight diagnostics use different restricted spaces and should not be
compared as if they were identical.

Small `max_sweeps` values test execution and expose trajectories; they are not
convergence studies.  Increase the sweep limit and compare energy histories
before drawing physics conclusions.  Keep the expansion modest (usually
`deltaD=1` or a small fraction of `D`): a large temporary expansion can gain
energy before trim but lose it again when compressed back to `D`.

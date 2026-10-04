# Open virtual boundaries with periodic Hubbard Hamiltonians

`open_boundary.py` supplies `OpenState`, `OpenContractions`,
`OpenOneSiteOptions`, and `open_one_site`. The physical Hamiltonian is supplied
as product terms and may have periodic interactions. The comparison benchmark
reuses exactly the fermionic/bosonic periodic Hamiltonian builders from
`periodic.py`, including the full boundary Jordan–Wigner string for fermions.

## Three ansätze

Every virtual boundary has dimension one; internal bonds have dimension at most
D. There is no virtual trace closure.

- MPS: tensors `A_i[s_i, a, b]` throughout.
- LETTA without wrap tie: tensors `A_i[s_i, s_(i+1), a, b]` for all but the
  last site; the last tensor is `A_last[s_last, a, 0]`.
- LETTA with wrap tie: the last tensor retains `A_last[s_last, s_first, a, 0]`.
  This leaves a loop through physical indices despite open virtual boundaries.

The one-site active solver contracts exact double-layer transfers. It never
constructs or projects into the physical configuration basis. Such constructions
are used only in independent benchmark validation and support-bound ED.

## Gauge and local solve

For an open MPS, the norm matrix has components

$$N_{s a b, s' c d}=\delta_{s s'} L_{a c}R_{b d}.$$

For open NN-tied LETTA without the wrap tie, an interior norm block has

$$N_{s t a b, s' t' c d}=\delta_{s s'}\delta_{t t'}L^{(s)}_{a c}R^{(t)}_{b d}.$$

Left/right norm marginals are whitened with charge-preserving Hermitian square
roots. Their inverses are applied to neighboring tensors, preserving every
physical amplitude. For LETTA these gauges depend on the shared physical index.
Open MPS obtains a scalar identity on its supported norm space; local diagonal
equilibration removes the scalar. Singular directions remain explicitly in the
tensor parameterization and receive finite gauge scales, rather than rank
truncation or inversion of tiny eigenvalues. LETTA with a wrap tie may retain
correlated norm blocks, and all variants use the full supported generalized
problem, not an assumed identity matrix.

After the gauge, tensor Frobenius norms are balanced to their geometric mean by
cancelling scalar factors with unit product. This avoids huge mutually cancelling
tensor scales and meaningless absolute coordinate residuals. The physical state
and variational family do not change under this operation.

Each step solves the lowest supported generalized eigenpair, independently
recontracts the physical energy, and rolls back an energy-increasing candidate.
A sweep is one directional pass through every site; direction alternates.
No bond expansion, two-site update, or truncation is used.

## Benchmark protocol and limitations

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/letta-mpl PYTHONPATH=. \
.venv-1/bin/python -m pyqed._letta_one_site_opt.benchmarks.open_boundary_hubbard \
run --output /private/tmp/letta_open_comparison --model fermi
```

Repeat with `--model bose`. Defaults: L=5,10, D=2,3,4,6; up to 500 sweeps;
t=1, U=4, mu=0, N=L, and bosonic occupancy cutoff 2. All three variants share
the same starting physical MPS for each model/L/D/seed. Closed and open virtual
networks are different variational families and have different random starts;
the closed-ring curves are historical paper-gauge benchmarks, not a same-state
gauge ablation.

The inherited charge sequence is [0,1,-1,0,1,-1,...]. At each cut, multiplicities
are capped by the available MPS prefix/suffix capacities; actual per-bond charge
lists and dimensions are recorded. This deliberately keeps the same bond layouts
for the three open variants. In particular, it is not an unrestricted optimization
over every possible LETTA charge allocation at maximum D.

The boundary charge is fixed to zero, and the mask obeys
`n(s_i) + q_left - q_right = 1`. Thus each prefix particle-number deviation must
belong to the prescribed charge list on that cut. D=3,4,6 use only -1,0,1.
These lists exclude physical configurations, even when one-site convergence is
excellent. Repeated charge degeneracies do not restore missing charge sectors.

The reference helper explicitly diagonalizes the physical Hamiltonian restricted
to this configuration support. Its energy is a lower bound for every ansatz with
these charge lists, not a fixed-D optimum. For fermions, the reference uses the
minimal-|Sz| sector; the number-only support projector preserves SU(2), so a
member of each allowed spin multiplet lies there.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=. .venv-1/bin/python \
-m pyqed._letta_one_site_opt.benchmarks.open_boundary_support \
--output /private/tmp/letta_open_comparison

MPLCONFIGDIR=/private/tmp/letta-mpl PYTHONPATH=. .venv-1/bin/python \
-m pyqed._letta_one_site_opt.benchmarks.open_boundary_hubbard plot \
--output /private/tmp/letta_open_comparison
```

Energy plateaus mean three successive changes below 1e-11 per site, with no
rejected local updates. They do not certify a global variational minimum.
Positive E_k-E_last values are plotted logarithmically; zero/negative values
are omitted rather than replaced with an artificial floor. Capped runs are
explicitly labeled. Final energies are independently checked by occupation-basis
Hamiltonian action with an absolute discrepancy tolerance of 1e-8.

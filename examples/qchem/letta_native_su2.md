# Native SU(2) LETTA: implementation and verification record

This is the continuation of `letta_symmetry.md`. Baseline implementation:
`2f92835`; explicit checkpoint before the native numerical work: `7e15d86`, on
`codex/letta-qchem`. No remote publication is needed to trace these commits.

## Scope and current status

The implementation uses U(1) particle number and SU(2) total spin. Spatial
orbitals have empty `(0,0)`, single `(1,1/2)`, and double `(2,0)` multiplets.
LETTA ties repeat invariant local multiplet labels, never magnetic components.
The Hamiltonian and norm matvecs, one-site and two-site optimization, conditional
gauges, and whole-multiplet growth now have native reduced paths.

One-site sweeps with frontier gauges maintain moving boundary environments.
Two-site problems use native reduced pair contractions but still rebuild their
boundary chains between pair optimizations. This is a remaining performance
opportunity, not an approximation to the wavefunction or Hamiltonian.

Native contractions are validated against independent operators and component
references. End-to-end molecular timings now include equilibrium LiF and
stretched N2; stretched LiF also tests adaptive multiplet allocation. The
measurements support a speed benefit on these small molecular examples, not a
universal speedup over optimized external DMRG. Spin-coupled links for additional
pure-spin expressivity are a separate ansatz extension; the current links remain
invariant-label ties.

## 1. Reduced wavefunction and norm convention

For charge alone, use `problem.symmetry('n')`. For charge and spin projection,
use `problem.symmetry('n_sz')` with charges $(N,2S_z)$. A tensor counts only its
owned orbital in $q_R=q_L+q(s_i)$; tied copies condition amplitudes and never
add extra particles. Those Abelian modes keep the original four physical
labels. `problem.symmetry('su2', two_s=2*S)` instead fixes $(N,S)$ and ties the
three invariant occupancy multiplets. The default $2S$ is
$|N_\alpha-N_\beta|$; set it explicitly when targeting a higher-spin state.

For each virtual charge-spin sector $q=(N,S)$, store multiplicities $r_q$:

$$V=\bigoplus_q\mathbb C^{r_q}\otimes V_S,\qquad
A^{m_lm_pm_r}_{a p b}=B_{a p b} C^{S_rm_r}_{S_lm_l,S_pm_p}.$$

All independent numbers are in $B$. Repeated LETTA labels are absorbed into
multiplicity coordinates by the sparse `ReducedFrontier` embedding $P_i$.
Local source-coordinate actions are $P_i^\dagger H_i P_i$ and
$P_i^\dagger N_i P_i$; the adjoint is essential because the embedding need not
be an isometry.

A scalar norm boundary has the form $G_q\otimes I_{2S_q+1}$. CG orthogonality
gives the left and right recursions

$$G_R(q_r)=\sum_{q_l,q_p,p}B_p^\dagger G_L(q_l)B_p,$$
$$G_L(q_l)_{a l}=\sum_{q_p,q_r,p,b,r}
\frac{2S_r+1}{2S_l+1}\overline{B_{a p b}}G_R(q_r)_{b r}B_{l p r}.$$

The local Euclidean adjoint metric action carries an additional factor
$2S_r+1$. Advancing a boundary and applying a local quadratic form therefore
have different dimension factors. The invariant contraction sums all target
$M$ components; the public `state.norm()` divides by $2S_{\rm target}+1$.

`reduced_norm.py` implements these recursions directly. `reduced_gauge.py`
extracts each conditional memory-sector Gram and whitens only its supported
multiplicity space. Small eigenvalues retain an invertible unit gauge. The
right gauge uses a transpose in bra/ket ordering. Auxiliary sector QR is used
for stable energy reporting without changing the LETTA parametrization.

## 2. Why the original reduced DMRG adapter was insufficient

A direct probe of `_target_local_matrix` with the old block-environment helpers
gave incorrect identity metrics and non-Hermitian local Hamiltonians, even on
short Heisenberg chains. Chemistry operator cores also have internal channels
whose labels do not consistently identify a single irreducible spin tensor.
Their complete component MPO is nevertheless independently correct.

Rather than alter energies by a many-body projection or patch spin-dependent
constants, the new path first puts that MPO into a verified irreducible basis.
The original component implementation remains an explicit test oracle.

## 3. Compile the operator into spin multiplets

Vectorize a local operator using the physical output and dual input irreps.
An orthonormal operator basis is

$$T^{k\mu}_m[m_o,m_i]=(-1)^{S_i-m_i}
 C^{km}_{S_om_o,S_i,-m_i}.$$

$\mu$ identifies the physical input/output sectors and their multiplicity
copies; its charge is $N_o-N_i$, which can be negative. Operator sectors use
the generic signed-charge `Sector`, rather than the nonnegative particle-count
`SpinChargeSector` used for states. A spatial orbital has 16 operator components
organized into 10 multiplets.

`SpinTensorMPO.compile` performs these steps:

1. Apply the local unitary operator-basis change to every MPO core.
2. Right-orthonormalize the operator chain with local QR factorizations.
3. At each cut, couple the known left irrep and local operator irrep with CG
   coefficients, then factor each charge-spin sector separately.
4. Retain complete multiplets, removing only singular values below the explicit
   compilation tolerance (default $2\times10^{-13}$ times the local norm).
5. Require the final operator boundary to have charge zero and spin zero.
   Reject a material residual instead of silently projecting a non-scalar MPO.
6. Check local reconstruction residuals. Their reported maximum is a local
   diagnostic, not a rigorous global operator-error bound; independent small
   Hamiltonian comparisons supply the end-to-end correctness check.

The resulting cores contain only reduced coefficients $W$:

$$\mathcal W^{m_lm_r}_{o i}
=\sum_{k\mu m} W_{l,\mu,r}
C^{J_rm_r}_{J_lm_l,km}\,T^{k\mu}_m[o,i].$$

Only local MPO tensors and polynomial virtual dimensions are processed. No
$4^n$ determinant Hamiltonian or wavefunction is constructed by the compiler.
The chemistry assembly still reuses the existing AutoMPO and spin-free ERI
builder, with one-body and two-body chains assembled separately before summing.

## 4. Reduced Hamiltonian contractions

Use the structural environment basis

$$\mathcal B^{J}_{m_w,m_b,m_k}
=C^{S_bm_b}_{S_km_k,Jm_w},\qquad
\|\mathcal B^{J}\|_F^2=2S_b+1.$$

Contract this basis on both sides of the local operator with the bra and ket
CG tensors. The resulting scalar coefficient depends only on nine spins and
is cached by `spin_transfer`. It is equivalent to a recoupling coefficient;
the implementation evaluates its small structural magnetic sum once. It does
not expand any variational tensor or environment into magnetic components.

For boundary propagation, divide by the squared norm of the outgoing
environment basis. For a local Hamiltonian action, use the coefficient without
that division. A two-site transfer joins the two local coefficients and divides
once by the intermediate bra-irrep dimension. This distinction was missing in
the earlier attempted adapter.

`ReducedEnvironmentChain` stores only multiplicity arrays. Compiled transfer
plans are reused for unchanged sector layouts. Changed tensors invalidate only
boundaries that cross them; lazy advancement restores the boundaries needed at
the next center. In one-site frontier sweeps, local Rayleigh quotients and
norms reuse these exact environments, avoiding full-chain energy contractions
after every update.

### Numerical support of the local metric

A six-orbital, three-multiplets-per-sector Hubbard test exposed a failure not
seen in the smaller correctness suite. The matrix-free Davidson basis retained
coordinates in the norm nullspace. A subsequent gauge change amplified these
unobservable coordinates to about $10^8$, after which the local Rayleigh
quotient could disagree with a freshly conditioned full contraction and even
fall below FCI. The exact dense local solver did not have this problem because
its whitening explicitly discards the metric nullspace.

Adding a spectral-scale rejection threshold alone did not fix that mechanism.
The implemented fix removes invisible coordinates before forming the Davidson
basis. On a shared frontier, the norm is diagonal in the owned and conditioned
physical labels. Each remaining multiplicity block is

$$N_{\sigma,q_l,q_r}=(2S_r+1)G_L(\sigma,q_l)\otimes G_R(\sigma,q_r).$$

Its support projector is the product of the two small Gram support projectors.
Apply it to the initial vector and new search directions. Tests verify
$N\Pi x=Nx$, $\Pi^2x=\Pi x$, and non-increasing Euclidean norm. This removes
gauge redundancy; it is independent of the Hamiltonian and never constructs a
many-body projected Hamiltonian. Native two-site problems use the analogous
outer-boundary projectors. A Kronecker spectral bound additionally controls
near-null directions with the same metric tolerance as the dense solver.

This analytic one-site construction applies when the sparse embedding assigns
every source coordinate once, including untied, NN, and carried ties. General
pass-through embeddings explicitly decline this factorization and retain the
generic iterative treatment; they need further large-system conditioning tests.

After the fix, the six-orbital regression completes two sweeps, respects the
FCI variational bound, decreases energy at every update, and agrees with the
independent component contraction. The 52 norm/one-site/two-site/chemistry
symmetry tests and four targeted native regression tests passed after the fix.

## 5. Reproduction and validation

Use the isolated worktree root. Set `PYTHONPATH=.` and limit local thread fanout:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.
python -m pytest -q tests/test_letta_native_su2.py tests/test_letta_reduced_norm.py
python examples/qchem/benchmark_letta_native_su2.py --norb 4 --multiplicity 2 --repeats 3 --output examples/qchem/letta_native_su2_microbenchmark.json
```

The development Python needs NumPy, SciPy, PySCF, pytest, opt_einsum, SymPy, and
TensorLy through the existing package imports. In this session only missing
test dependencies were supplied through `/private/tmp/letta-symmetry-test-deps`;
the user's Python environment was not modified. Append that temporary directory
to `PYTHONPATH` to reproduce the session commands until dependencies are installed
normally. It is not a portable project dependency location.

The tests check independent fermionic Hamiltonians on one through four orbitals,
operator-basis unitarity, rejection of spin/charge-breaking inputs, complex
singlet/doublet/triplet actions, NN and carried ties, exact norm/gauge recursions,
moving-boundary invalidation, and one-/two-site sweeps with magnetic/global state
expansion disabled. The no-expansion tests compare converged energies with PySCF
FCI. Existing tests additionally cover zero Hamiltonians, one-orbital open shells,
whole-multiplet growth, rejected-update rollback, and badly conditioned gauges.

The microbenchmark uses identical states and the same compressed MPO in both
representations. It reports assembly and compilation separately, boundary-build
time, complete local-action time, component-kernel-only time, and storage.
These timings are not a claim about speed versus Abelian DMRG, nor about the
time required to converge at a fixed energy accuracy.

### Initial timing evidence

The following measurements predate the final contraction-path cache. They
remain archived to show the development history; the final rerun is below.

The isolated four-orbital microbenchmark (`letta_native_su2_microbenchmark.json`)
measured a 4.96 ms native local action versus 698 ms for the component action
with the same compiled MPO and state. Boundary storage was 96,800 versus
2,944,528 bytes. Compilation took 67.9 ms. These are comparisons with this
repository's component implementation, not universal SU(2) speed factors.

The more relevant end-to-end comparisons deliberately report the less favorable
results as well:

| Model / target error | SU(2) NN solver time | Abelian NN solver time | Result |
|---|---:|---:|---|
| Hubbard 4, $10^{-7}$ Ha | 0.209 s | 0.053 s | Both reached FCI in one pass |
| Hubbard 6, $10^{-6}$ Ha | 2.153 s | 0.766 s | SU(2): two passes; Abelian: one |

These figures exclude separately recorded Hamiltonian setup and are individual
runs, not statistical estimates. The four-site timing predates the subsequent
norm-support fix; the six-site timing includes it. Both methods start from the
same wavefunction and magnetic virtual dimensions. Their tied ansatz spaces
still differ: Abelian ties resolve spin components, while invariant SU(2) ties
do not. The Abelian implementation is masked dense. The untied reference in
this script is fixed-sector one-site MPS optimization, not adaptive two-site
DMRG. Neither small-system timing establishes the requested speed advantage
over the Abelian solver. Python overhead, convergence, and molecular Hamiltonians
are the next performance investigations.

The first archived **molecular** comparison, equilibrium LiF CAS(6e,6o), does
show an advantage. A fresh Python process timed SU(2) NN first, before the untied
control could warm its transfer caches. At $10^{-6}$ Ha target error:

| LiF CAS(6e,6o) | SU(2) NN | Abelian NN |
|---|---:|---:|
| Final error (Ha) | $7.60\times10^{-7}$ | $2.84\times10^{-14}$ |
| Solver time | 2.643 s | 30.909 s |
| Hamiltonian assembly/compilation | 3.808 s | 0.414 s |
| Setup plus solver | 6.451 s | 31.323 s |

This is about 11.7x in solver time and 4.9x including setup for this case and
this implementation. Both start from the same wavefunction (difference
$4.32\times10^{-16}$), with internal multiplet dimensions `(9,18,30,18,9)` and
magnetic dimensions `(12,30,60,30,12)`. The Abelian run uses the compact graph
AutoMPO because the CP-SVD builder exceeded its temporary storage budget for
these integrals. Exact integrals, orbital order, and core energy come from the
previously archived `lif_eq.npz`. These are single-run timings with unequal final
errors but the same stopping threshold; broader conclusions require additional
molecules and tighter targets. The fixed-sector untied control did not reach
the target and is not used in the speed ratio.

Reproduce the matched-state benchmark with:

```sh
python examples/qchem/benchmark_letta_symmetry_speed.py --norb 6 --multiplicity 3 --max-passes 6 --accuracy 1e-6 --output examples/qchem/letta_symmetry_speed_hubbard6.json
```

Use `--integrals examples/qchem/letta_active_space_results/integrals/lif_eq.npz`
to load the archived molecular integrals instead of the Hubbard model.

### Stretched molecules and allocation failure

The same cold-process protocol on stretched N2 CAS(6e,6o), with the same
`(9,18,30,18,9)` multiplet and `(12,30,60,30,12)` magnetic allocations, gives:

| Stretched N2 CAS(6e,6o) | SU(2) NN | Abelian NN |
|---|---:|---:|
| Final absolute error (Ha) | $1.28\times10^{-13}$ | zero at reported precision |
| Solver time | 2.813 s | 43.924 s |
| Hamiltonian assembly/compilation | 3.841 s | 0.421 s |
| Setup plus solver | 6.654 s | 44.345 s |

The $-1.28\times10^{-13}$ Ha signed error is numerical roundoff, not a
violation of the variational bound at meaningful precision. This run gives
15.6x in solver time and 6.7x including setup. All NN cases have one invariant
label crossing each cut: three scalar assignments for SU(2), versus four
magnetic/occupation assignments for the Abelian ties. This difference is part
of the ansatz definition and is disclosed rather than counted as a pure
representation-only speedup. The microbenchmark above isolates the latter.

The fixed allocation **failed** the $10^{-6}$ Ha target on stretched LiF:
after six passes it remained $4.605\times10^{-6}$ Ha above FCI (14.321 s of
solving). The Abelian run reached FCI in one pass (47.351 s). No matched-error
speed ratio is reported for this failed run. It motivated a separate adaptive
MPS control using two-site whole-multiplet growth, followed by exact embedding
in NN LETTA with the resulting sector allocation held fixed.

With a total cap of 30 multiplets, adaptive stretched-LiF MPS reached
$4.354\times10^{-8}$ Ha error in three sweeps. Its actual internal dimensions
were `(3,10,30,10,3)` multiplets, or `(4,16,56,16,4)` magnetic states. NN
refinement reduced the error to $2.704\times10^{-9}$ Ha in two further passes,
with embedding distance zero at reported precision and spin residual
$1.05\times10^{-17}$. Thus equal maximum $D$ did not specify equivalent
allocations: distributing multiplicities among sectors mattered. This was an
allocation/convergence limitation, with no evidence of a spin-contraction
failure. The archived report is `letta_su2_adaptive_lif_stretched_cap30.json`.
Its original `mps_seconds` includes lazy operator compilation; it excludes
component MPO assembly. Later script output separates assembly/compilation
as `setup_seconds`. Do not combine that old MPS time with a separately timed
setup to make a speed claim.

A direct retry with four copies per reachable sector also resolves the
stretched-LiF plateau without the adaptive-MPS warm start. Both methods begin
from the same random wavefunction (difference $5.24\times10^{-16}$), with
`(12,24,40,24,12)` multiplets and `(16,40,80,40,16)` magnetic dimensions:

| Stretched LiF, four copies/sector | SU(2) NN | Abelian NN |
|---|---:|---:|
| Final error (Ha) | $1.38\times10^{-12}$ | $4.26\times10^{-14}$ |
| Solver time | 3.668 s | 108.492 s |
| Hamiltonian assembly/compilation | 3.844 s | 0.410 s |
| Setup plus solver | 7.512 s | 108.902 s |

Both reach the $10^{-6}$ Ha threshold in one pass. This is 29.6x in solver time
and 14.5x including setup against this masked-dense Abelian implementation.
It uses a larger allocation than the failed three-copy run, so the two LiF
trials must not be presented as having the same $D$. The original failure and
successful retry are both retained in the archive.

The adaptive control is the repository's own two-site solver, using a reduced
sector split and physical-metric factor refinement. It still rebuilds
environments and performs iterative factor refinement, so these runs verify
allocation, inclusion, and energies; their timings are not comparisons with
an optimized external SU(2) DMRG package.

On stretched N2, the adaptive control removes the apparent large advantage
over the poorly allocated fixed-sector MPS:

| Multiplet cap | Actual center $D_{\rm mult}/D_{\rm mag}$ | MPS error (Ha) | NN error after exact embedding (Ha) |
|---|---:|---:|---:|
| 12 | 12 / 14 | $3.487\times10^{-2}$ | $3.591\times10^{-3}$ |
| 30 | 20 / 32 | $-1.28\times10^{-13}$ | $-2.27\times10^{-13}$ |

The cap-12 MPS was still improving after its six-sweep budget, so that row is
only a finite-iteration comparison, not an estimate of the best MPS energy.
The cap-30 MPS converged in four sweeps (47.07 s of solver time); NN refinement
took one pass (1.00 s). Both describe FCI within roundoff. Their exact sector
allocation is recorded in the JSON reports, the embedding distance is zero,
and the final spin residual is $3.47\times10^{-16}$ for cap 30. A cap is an
upper bound, not the number of multiplets ultimately retained. In particular,
the adaptive MPS needs only 20 center multiplets here, while a badly allocated
30-multiplet MPS was inaccurate. The inclusion principle predicts that LETTA
can do at least as well when the embedded state is retained; it does not
predict a strict improvement when MPS is already exact.

## 6. Recommended workflow and commands

1. Supply real, spin-independent, orthonormal spatial-orbital integrals in the
   existing `ElectronicProblem` convention. Core and nuclear energies belong
   in `ecore` and are included once. Spin-orbit terms require another symmetry.
2. Choose a common orbital order for all methods. `problem.reordered(order)`
   permutes all integral indices together; tie edges use positions in that
   reordered chain. The archived benchmarks preserve the NPZ orbital order.
3. Run adaptive SU(2) MPS first when the sector allocation is unknown. A
   one-site sweep cannot create missing sectors. `bond_dim` is a total
   **multiplet** cap, whereas `multiplets_per_sector` allocates that many copies
   of **each** reachable sector; they are not interchangeable.
4. Import the MPS with `ReducedLatticeLETTA.from_mps` and NN neighborhoods.
   This embeds the incumbent exactly. Use `gauge_mode='frontier'` and retain
   the generalized local metric. A matrix-free threshold of 32 exercises the
   native iterative solver on the examples here.
5. Optimize, check the variational energy, and report both multiplet and
   magnetic dimensions. Energy-change convergence alone does not certify an
   FCI error. For small examples, independently check $S^2$, norm, and FCI.
6. For distant invariant-label ties, `carry=True` restores shared frontiers
   but enlarges the tensor dependencies and ansatz; it is not just a gauge
   transform. Bound frontier width explicitly before adding ties. General
   direct ties and spin-coupled links need the additional work described below.

Reproduce the molecular experiments from the worktree root with the thread
limits and Python path above:

```sh
python examples/qchem/benchmark_letta_symmetry_speed.py --integrals examples/qchem/letta_active_space_results/integrals/lif_eq.npz --multiplicity 3 --max-passes 6 --accuracy 1e-6 --output /private/tmp/lif_eq_speed.json
python examples/qchem/benchmark_letta_symmetry_speed.py --integrals examples/qchem/letta_active_space_results/integrals/lif_stretched.npz --multiplicity 3 --max-passes 6 --accuracy 1e-6 --output /private/tmp/lif_stretched_speed.json
python examples/qchem/benchmark_letta_symmetry_speed.py --integrals examples/qchem/letta_active_space_results/integrals/lif_stretched.npz --multiplicity 4 --max-passes 4 --accuracy 1e-6 --output /private/tmp/lif_stretched_r4_speed.json
python examples/qchem/benchmark_letta_symmetry_speed.py --integrals examples/qchem/letta_active_space_results/integrals/n2_stretched.npz --multiplicity 3 --max-passes 6 --accuracy 1e-6 --output /private/tmp/n2_stretched_speed.json
python examples/qchem/letta_symmetry.py --integrals examples/qchem/letta_active_space_results/integrals/lif_stretched.npz --cap 30 --sweeps 6 --output /private/tmp/lif_adaptive.json
python examples/qchem/letta_symmetry.py --integrals examples/qchem/letta_active_space_results/integrals/n2_stretched.npz --cap 30 --sweeps 8 --output /private/tmp/n2_adaptive.json
```

FCI and expanded state vectors in the example scripts are validation oracles,
outside the production optimization path. The native solver does not call
them. Timings use one BLAS/OpenMP thread, Python 3.12, NumPy 2.4.2, SciPy 1.17.1,
and PySCF 2.12.1 on the local development machine. Each speed report is a single
run and has no statistical confidence interval.

The implementation is organized as follows:

| File in `pyqed/_letta_one_site_opt` | Responsibility |
|---|---|
| `reduced_symmetry.py`, `reduced_state.py` | Physical multiplets, charge-spin sectors, reduced cores, exact MPS import |
| `reduced_frontier.py` | Sparse embedding of tied labels into a sequential contraction frontier |
| `reduced_mpo_compile.py` | Verified local compilation of the chemistry MPO into spin multiplets |
| `reduced_norm.py`, `reduced_gauge.py` | Multiplicity Grams, metric support, equivariant conditional gauges |
| `reduced_environment.py` | Cached recoupling coefficients and reduced Hamiltonian boundaries/actions |
| `reduced_solver.py` | Local generalized eigensolves and moving one-site environments |
| `qchem.py` | Integral conventions, target symmetry, one-/two-body operator assembly |

`pyqed/_letta_two_site_opt/reduced_solver.py` supplies pair problems, whole-
multiplet growth, factor refinement, truncation, and rejected-update rollback.
`examples/qchem/letta_symmetry.py` supplies the adaptive MPS/NN checks; the two
scripts `benchmark_letta_native_su2.py` and `benchmark_letta_symmetry_speed.py`
isolate kernel measurements from convergence timings.

Conceptual references are [the LETTA paper](https://arxiv.org/abs/2609.30101),
[Singh and Vidal's invariant-tensor construction](https://arxiv.org/abs/1208.3919),
and [Sharma and Chan's spin-adapted chemistry DMRG](https://arxiv.org/abs/1408.5039).
The NN invariant-label construction and its limitations are explained in
`letta_symmetry.md`; they should not be confused with a proof that this
parametrizes every spin-pure state of the original magnetic-label LETTA.

## 7. Final contraction-path optimization

Profiling one four-orbital sweep found 5,669 NumPy `einsum_path` calls: 0.291 s
of the 0.532 s profiled work was inside that path-planning routine. The profile
adds overhead and is not a solver timing. It identified repeated planning of
small sector contractions as a concrete bottleneck.

`numpy_contractions.cached_einsum` now reuses the existing prepared NumPy
kernels with a bounded cache keyed by equation and operand shapes. It caches
paths and axis orders only, never tensor values or environments. Reduced norm
and Hamiltonian boundary propagation and local/pair actions use this helper.
Gauge changes, new states, and dtype changes always supply fresh arrays.
The numerical representation, local metric, and Hamiltonian are unchanged.

Fresh-process reruns with the final cache give:

| Model / target error | SU(2) solve | Abelian solve | SU(2) setup + solve | Abelian setup + solve |
|---|---:|---:|---:|---:|
| Hubbard 4 / $10^{-7}$ Ha | 0.108 s | 0.053 s | 0.123 s | 0.054 s |
| Equilibrium LiF / $10^{-6}$ Ha | 1.056 s | 30.811 s | 4.899 s | 31.227 s |

The LiF result preserves the earlier $7.5955\times10^{-7}$ Ha SU(2) error;
the Abelian error is $2.84\times10^{-14}$ Ha. That is 29.2x solver speed and
6.4x including setup at the common threshold, with the same initialization
and allocations described above. The small Hubbard case still favors Abelian.
Reports are `letta_symmetry_speed_{hubbard4,lif_eq}_cached.json`; earlier
stretched-molecule and adaptive timings should not be mistaken for reruns of
this final cache. Python overhead, compilation, and convergence remain
important: there is no size-independent speed factor.

## 8. Validation and scope limits

After the metric-nullspace fix, the complete 12-module suite passed **129
tests** with four pre-existing NumPy deprecation warnings in 208.22 seconds.
After the final path-cache optimization, 66 affected contraction/norm/solver
tests passed, followed by **132 passed** across the full 13-module suite in
202.23 seconds on 2026-10-03, with the same four deprecation warnings.
The final small demonstration (`letta_symmetry_native_results.json`) verifies
a three-orbital doublet at FCI, zero-distance MPS embedding, unit norm, and
spin residual $1.57\times10^{-16}$; the four-orbital singlet also preserves
spin, norm, and the variational improvement at fixed imported sectors.

Reproduce the full focused regression with:

```sh
python -m pytest -q tests/test_letta_reduced_two_site.py tests/test_letta_reduced_frontier.py tests/test_letta_reduced_symmetry.py tests/test_letta_qchem.py tests/test_letta_qchem_active_space.py tests/test_letta_reduced_gauge.py tests/test_letta_qchem_symmetry.py tests/test_letta_reduced_state.py tests/test_letta_reduced_mpo.py tests/test_letta_reduced_one_site.py tests/test_letta_reduced_norm.py tests/test_letta_native_su2.py tests/test_letta_contraction_paths.py
```

The implemented and measured scope is U(1) number, U(1) number/spin projection,
and native U(1) x SU(2) **invariant-label** LETTA for small active spaces. The
native path has independent action/operator checks, exact-spin non-singlet
tests, one-/two-site solvers, NN/shared-frontier gauges, adaptive whole-multiplet
growth, molecular references, and measured computational savings.

The implementation, tests, molecular retries, profiling, and reproduction notes
complete the six evidence gates in `docs/plans/2026-10-02-letta-su2-native.md`
for this scope. Git preserves the pre-implementation checkpoint `7e15d86`,
native backend `c6d8ab4`, cold LiF timing protocol `05963a4`, and the subsequent
validation/cache commit. The original checkout and stash archives are described
in `letta_ground_state.md`; no remote publication is required for this local
commit trace.

The following remain limits or separate extensions:

- Spin-coupled links require their own intertwiners, recoupling, and gauge
  derivation. At strictly fixed single occupancy, current invariant-label ties
  add no spin expressivity. They do not parametrize the full spin-pure subset
  of original magnetic-label LETTA at fixed $D$.
- General direct ties without shared frontiers need additional conditioning
  tests; use carried ties for the validated shared-frontier construction.
- Large active spaces, memory peaks, and scaling against an optimized external
  block-sparse Abelian/SU(2) DMRG code have not been established. The current
  chemistry assembly has startup cost and the internal Abelian baseline is
  masked dense. The reported memory comparison is allocated environment-array
  storage, not process peak RSS.
- Two-site boundary reuse and improved reduced pair factorization could reduce
  adaptive-solver cost further. A fixed cap or a small sweep-energy change is
  not a proof of global convergence; retain the best incumbent and inspect the
  actual sector distribution.


## H₂O dimension comparison follow-up (2026-10-03)

The follow-up [water comparison](water_su2_comparison_final/REPORT.md) uses charge U(1) and spin SU(2) for four active spaces, multiple multiplet caps, independent CAS-FCI references, and additional block2 controls. The untied native MPS now uses a physical-metric Schmidt split; the general tied factorization remains unchanged. Earlier accuracy comparisons are historical and should not substitute for the stronger controls in this study. Low-D minima can depend on initialization in both native and block2 solvers. The final tables retain the best validated candidate per method and cap, with all source records preserved.


## Stable one-site local coordinates

The reduced one-site solver constructs norm factors with sector QR and splits
those factors into disjoint physical-label blocks. SVD of column-equilibrated
factors defines orthonormal coordinates. The Hamiltonian acts through a
mixed-canonical auxiliary reduced MPS in those coordinates. It does not use a
determinant-space projection or expand magnetic components.

For a local factor $F$, the overlap is $N=F^\dagger F$. The implementation
factors $F$ directly, avoiding loss of null-space accuracy from diagonalizing
the explicitly contracted Gram matrix. If $D$ contains column norms and
$F D^{-1}=U\Sigma V^\dagger$, the retained coordinate map is
$W=D^{-1}V_r\Sigma_r^{-1}$. The local eigenproblem uses $W^\dagger H W$.
QR and the block decomposition preserve the state in exact arithmetic; the
relative `metric_tolerance` cutoff discards small singular directions.

This is an adaptation of standard QR/SVD numerical linear algebra to LETTA's
reduced multiplicity and physical-label structure, not a reproduction of a
published LETTA algorithm. Reference: L. N. Trefethen and D. Bau III,
*Numerical Linear Algebra*, SIAM (1997),
[DOI: 10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574).
The finite overlap cutoff restricts the tested local subspace; neither local
eigensolver convergence nor sweep convergence guarantees a global variational
minimum. The implementation supports the existing open-chain reduced frontier
representation, not an arbitrary periodic tensor network.

For these QR local problems, `residual_norm` and `relative_residual` measure
stationarity in retained physical coordinates. `raw_residual_norm` and
`raw_relative_residual` retain the unprojected tensor-coordinate diagnostics.
The independent energy acceptance check remains in force. A sweep cap must
still be distinguished from `converged=True`.


The legal frontier gauge also uses reduced QR square-root factors. For a
marginal over unshared labels it stacks the corresponding factor column slices,
then takes an SVD of that factor, instead of diagonalizing a contracted Gram.
This avoids spurious negative Gram eigenvalues near null directions. Singular
values below the square root of the metric threshold times the largest singular
value across all conditional blocks retain a unit gauge; no
bond directions are truncated by this gauge. This is an adaptation of standard
QR/SVD conditioning (Trefethen and Bau, *Numerical Linear Algebra*, SIAM 1997,
https://doi.org/10.1137/1.9780898719574), not an exact canonicalization of a
correlated LETTA metric. The independent physical-energy preservation check
and transactional recovery remain in force.


## Exact runtime reuse in reduced one-site sweeps

Local Hamiltonian environments can be initialized lazily: only the left and
right boundaries requested by an action are contracted. Stable energy checks
reuse one expanded, left-canonical reduced state for the Hamiltonian and norm.
A local eigensolve binds its fixed environment operands once and reuses the
existing contraction plan, including the original operation order. Such a
prepared action is scoped to that immutable solve and must be rebuilt after
any change to its environments or tensors. Frontier embedding indices are
constructed with batched integer indexing in the same ordering as before.

These changes do not alter the metric cutoff, sector allocation, eigensolver
settings, floating-point dtype, or energy acceptance checks. They introduce no
new approximation or replacement numerical algorithm. Tests compare lazy and
prepared actions exactly with the eager path, including complex inputs and
boundary invalidation. The runtime benefit depends on the problem and cannot
be inferred from a profiled local update alone.


### Memory-bounded overlap construction

The one-site SU(2) coordinates (also used by CBE) store the exact overlap map
as scatter indices and left/right QR factors. A dense overlap block is built
only while computing that block's full SVD. Column equilibration and the global
relative singular-value cutoff are unchanged. The whitening maps are applied
through singular vectors, singular values, and column scales, rather than
retaining two additional dense matrices. This is a storage/data-flow adaptation
of the QR/SVD formulation documented above, not a new low-rank approximation.
Arithmetic order changes; energies and residuals must agree to numerical
precision, rather than bitwise. Singular vectors and one-block SVD workspace
remain potentially large, so this does not guarantee a fixed total memory cap.

Structural MPO transfer caches retain at most two layouts per site. Evicted
kernels are rebuilt exactly when needed; live environments retain their own
references. Unreachable declared sectors have exact zero-row QR factors, so
CBE's sparse sector changes do not cause missing-factor lookup errors. No
nonzero state component is removed by this handling.

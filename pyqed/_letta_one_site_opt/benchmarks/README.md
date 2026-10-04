# LETTA condensed-model benchmarks

This directory contains four model families in both one and two dimensions,
plus the 1D Hubbard--Holstein model:

| Model | Hamiltonian conventions | Local dimension |
|---|---|---:|
| Ising | \(-J\sum_{\langle ij\rangle}Z_iZ_j-h\sum_iX_i\) | 2 |
| XXZ Heisenberg | \(J\sum_{\langle ij\rangle}[\tfrac12(S_i^+S_j^-+S_i^-S_j^+)+\Delta S_i^zS_j^z]-h\sum_iS_i^z\) | 2 |
| Bose--Hubbard | \(-t\sum_{\langle ij\rangle}(b_i^\dagger b_j+\mathrm{h.c.})+\tfrac U2\sum_i n_i(n_i-1)-\mu\sum_i n_i\) | `max_occupancy + 1` |
| spinful Fermi--Hubbard | \(-t\sum_{\langle ij\rangle,\sigma}(c_{i\sigma}^\dagger c_{j\sigma}+\mathrm{h.c.})+U\sum_i n_{i\uparrow}n_{i\downarrow}-\mu\sum_i(n_{i\uparrow}+n_{i\downarrow})\) | 4 |
| Hubbard--Holstein (1D) | Fermi--Hubbard plus $\omega\sum_i b_i^\dagger b_i+g\sum_i(n_i-1)(b_i+b_i^\dagger)$ | `4 * (max_phonons + 1)` |

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

Local runtime note (2026-09-10): the 280-test LETTA validation passed twice
with Python 3.13.2, NumPy 2.4.6 and an isolated SciPy 1.17.1 installation.
The held-out four-method chain benchmarks also completed with that setup.
SciPy 1.18.0 and 1.18.1 intermittently crashed natively in combined geometry
and eigensolver tests on this Mac; the underlying cause is not established.
The repository's installed environment has not been changed. See
`docs/benchmarks/2026-09-10-general-cbe-validation.md` for the evidence and
algorithmic limitations; this is not a cross-platform dependency constraint.

Run any file directly from an IDE or terminal; repository-root `PYTHONPATH` is
inserted automatically.

```text
ising_1d.py                 ising_2d.py
heisenberg_1d.py            heisenberg_2d.py
bose_hubbard_1d.py          bose_hubbard_2d.py
fermi_hubbard_1d.py         fermi_hubbard_2d.py
hubbard_holstein_1d.py
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
The model scripts and suite also accept `--eigensolver-tolerance` (default
`1e-10`) and `--eigensolver-max-iterations` (default `300`). These control
the LETTA iterative local eigensolves; the iteration limit counts ARPACK
restart iterations. The one-site and CBE solvers start from the current
tensor in supported metric coordinates. An unconverged iterative solve is
reported as a failure, rather than silently accepting an incomplete Ritz vector.
Exact diagonalization is skipped when the Hilbert dimension exceeds
`--exact-max-dimension`.

Run all 18 registered comparisons (`D=4`, 50 sweeps) with:

```bash
python pyqed/_letta_one_site_opt/benchmarks/run_condensed_suite.py
```

To display only the four LETTA methods, add:

```text
--solvers letta_one_site,letta_cbe_exact,letta_cbe_strict,letta_two_site
```

The Hamiltonians use an exact direct-sum product-term MPO.  It is deliberately
simple and independently testable rather than minimally compressed. General
CBE currently contracts dense local MPO factors against exact labelled
frontiers; the ordinary solver still supports sparse transitions. The automatic strict-CBE
preselection width is
`min(D + 2*deltaD, left parent size, right parent size)`, so it stays moderate
and is not inflated by a redundant MPO bond representation.  Larger production
calculations should still use compressed model-specific MPO builders to reduce
the number of sparse paths and the Hamiltonian-contraction prefactor.

## Reading the diagnostics

Both the model tables and the suite table show energy, error against the
available exact reference, time, `swp`, convergence, and the last `dE/site`.
For LETTA, **one sweep here is one directional pass (LR or RL)**. Two passes
make a full LR+RL cycle; the separate cluster driver uses full cycles instead.
`conv=True` means the energy-change stopping condition was met, not that the
ground-state error is zero. JSON retains the entire `sweep_energies` list.

The individual model tables also show `local-Hmv`: local Hamiltonian action
calls, including both CBE candidate and baseline solves even when one is
discarded. Dense one-site solves and CBE selector/energy-check contractions
are outside this counter; zero is possible for entirely dense one-site runs.
One-site and two-site actions have different costs, so compare time as well.
`cbe_phase_seconds` separates selection (including initial energy), expanded
solve (including embedding/environments), trim (including candidate energy
checks), and baseline. These are disjoint stage timings, not a complete
accounting of environment construction and sweep overhead.

`cbe-ok/fallback` counts accepted trimmed expansions and ordinary one-site
fallbacks.  The JSON report also includes the ordinary-candidate selection
count, mean CBE-minus-baseline energy, guard allowance, missing norm, retained
weight, trim loss, parameter counts, and pair-operation/materialization flags.
For strict CBE, `missing` is the norm of the supported target in the candidate
space after both old one-site tangents are removed, measured in the Schur
metric that eliminates the old active coefficients. All pair-action,
pair-metric, and merged-pair counters must be zero.  Exact and strict
retained-weight diagnostics use different restricted spaces and should not be
compared as if they were identical.

The standalone `cbe_scaling.py` instruments the actual general selector,
including physical metric operations and one-site tangent cross-Gram SVD.
It reports contraction-path costs and observed decomposition work proxies.
The proxy excludes uninstrumented NumPy matrix products. This is a comparison
of one selector call with single local Hamiltonian actions, not complete
updates: eigensolves, baseline, trim and sweep environment construction need
the separate end-to-end timings. No universal single-site-cost bound is asserted.

The dependency-aware selector classifies each physical index's home and
incidence, preserves bra/ket connectors and COPY equalities, and generates
candidates independently only for physical arguments shared by both active
tensors. It uses supported SVD only after checking separability of the
restricted metric; other metrics use a monotone, locally optimized ALS fit.
Environment-only LR labels never become new arguments of an active tensor.

Small `max_sweeps` values test execution and expose trajectories; they are not
convergence studies.  Increase the sweep limit and compare energy histories
before drawing physics conclusions.  Keep the expansion modest (usually
`deltaD=1` or a small fraction of `D`): a large temporary expansion can gain
energy before trim but lose it again when compressed back to `D`.

## 1D Hubbard--Holstein

`hubbard_holstein_1d.py` uses the Hamiltonian in Eq. (11) of
[Gleis, Li and von Delft, PRL 130, 246402 (2023)](https://arxiv.org/html/2207.14712v2),
with an optional chemical potential:

$$
H=-t\sum_{i=1}^{L-1}\sum_\sigma(c_{i\sigma}^\dagger c_{i+1,\sigma}+\mathrm{h.c.})
  +U\sum_i n_{i\uparrow}n_{i\downarrow}-\mu\sum_i n_i
  +\omega\sum_i b_i^\dagger b_i
  +g\sum_i(n_i-1)(b_i+b_i^\dagger).
$$

Defaults match the paper's couplings: `t=1`, `U=0.8`, `omega=0.5`,
`g=sqrt(0.2)`, `mu=0`. The electron--phonon coupling is centered at one
electron per site, not at zero. No phonon zero-point energy is added.
Each site groups the electron basis `(empty, up, down, double)` with phonon
occupations `0..max_phonons`; the phonon index varies fastest. Fermionic
Jordan--Wigner parity acts only on the electron factor.

For manageable click-run validation, the standalone defaults are **L=3,
D=4, max_phonons=1 (d=8), max-sweeps=50**. It prints all four LETTA methods
plus the ordinary MPS two-site reference. All methods use the same initial
physical state and report energies, directional passes, convergence and time.

```bash
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/hubbard_holstein_1d.py
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/hubbard_holstein_1d.py --length 4 --bond-dim 6 --max-phonons 1 --exact-max-dimension 0
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/hubbard_holstein_1d.py --length 2 --max-phonons 3 --omega 0.5 --g 0.4472135954999579
```

`--max-phonons 3` gives d=16; this is separate from bond dimension D.
`--max-phonons 0` projects out phonon excitations and reduces to the ordinary
Hubbard Hamiltonian, not a phonon-integrated effective interaction. Increase
the cutoff to check phonon convergence. The tiny default is not a converged
phonon approximation to the paper's benchmark. Exact diagonalization, when
enabled, uses the same cutoff; its error does not measure cutoff error.

**Sector distinction:** the paper studies L=50 at fixed N=L and total spin
S=0 with symmetry-reduced bonds. These runners do not impose those sectors
and count ordinary bond states, not symmetry multiplets. The default is
therefore the same Hamiltonian but not a reproduction of the paper's state
or timings. `--mu 0.4` adds a chemical potential; it does not enforce fixed
particle number. Larger L, phonon cutoffs and exact-selector CBE can be costly.

`cbe_harder_cases.py` enables `hubbard_holstein_3_4` for click-run, using
`("hubbard_holstein", "1d", 3, 4)` in the same `(model, dimension, size, D)`
format and model defaults. Use the standalone script to vary phonon and
coupling parameters. It runs first, followed by the previous six default
Fermi cases. To click-run only this model, set
`DEFAULT_CASES = ("hubbard_holstein_3_4",)`.

## Harder cases and conditional-compression ablation

Click-run `cbe_harder_cases.py` using the editable `HARD_CASES` dictionary.
After each case completes, it saves a separate signed-energy-difference figure
for **one-site LETTA, two-site LETTA, and strict CBE LETTA**. The header shows
the Hamiltonian equation, couplings, actual bond dimension D, and chain length
L (or lattice shape and site count for higher dimensions). One iteration is
one directional pass, LR or RL. Curves show
$\Delta E(k)=E_{\mathrm{method}}(k)-E_{\mathrm{two-site,final}}$, starting at pass 1,
and stop when that method stops; no padding or smoothing is applied. This is a
common final-energy reference, not a same-iteration subtraction. Negative
values indicate energy below the final two-site result. The symmetric-log
(`symlog`) axis shows positive, negative, and zero values; its linear region is
$|\Delta E|<10^{-10}$, configurable through `PLOT_LINTHRESH` in the script.
The figure labels the reference energy and whether that two-site run converged.
No exact-ground-state interpretation is implied. If two-site was not run or
failed, the figure explicitly reports the missing reference instead of
fabricating differences. Source JSON retains the raw energies and records
the reference and plot scale separately.

Exact CBE is **not calculated by default**. Add `--include-cbe-exact`, set
`INCLUDE_CBE_EXACT = True` near the case list for click-run, or pass
`include_cbe_exact=True` to `run_harder_suite` to include it in the numerical
comparison. An explicit `--solvers` list containing `letta_cbe_exact` also
enables it. Exact CBE is still omitted from the three-method figure.

Click-run exports PNG, PDF, and per-case source JSON files into
`output/benchmarks/cbe_harder_cases/` under the repository root, independent of
the working directory. Filenames end in `_energy_difference` and contain the
case name and actual L and D;
rerunning the same case at the same L and D overwrites those files. Change
`DEFAULT_FIGURE_DIR` near the case list or pass `--figure-dir /path/to/figures`.
Use `--no-plots` to disable exports. Programmatic `run_harder_suite` calls enable
plots by supplying `figure_dir=...`. Missing methods are explicitly noted in
the figure, and plot errors are recorded separately without dropping completed
solver results. To run only the three plotted methods, use
`--solvers letta_one_site,letta_two_site,letta_cbe_strict`.

Each tuple is **`(model, dimension, size, D)`**: for example,
`("fermi_hubbard", "1d", 10, 8)` means a ten-site chain with **D=8**.
The fourth value is no longer a seed offset. Each selected case uses its own
D unless `--bond-dim` explicitly overrides them all (or `bond_dim=...` in
`run_harder_suite`). Defaults remain **deltaD=1, max-sweeps=50**, with the three
plotted LETTA methods (no exact CBE or MPS reference by default). `DEFAULT_CASES` currently selects
every uncommented entry in `HARD_CASES`; comment/uncomment rows or use `--cases` to select others.
Every completed case is printed immediately. `--output` checkpoints JSON
after each case; it does not save an unfinished solver's state.

From the repository root:

```bash
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py --output harder.json
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py --cases fermi_hubbard_10_4 --global-trim --output global-trim.json
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py --cases fermi_hubbard_10_4 --bond-dim 8 --max-sweeps 25
```

Strict CBE now compresses independently for each physical configuration
shared by the adjacent LETTA factors. Previously it imposed a single global
transfer matrix, which could discard even a conditionally rank-one state.
The new transfer depends on those shared indices and is absorbed into the
neighbor at the same indices. All work remains in the active one-site metric;
no merged pair, pair metric, or pair action is introduced. Exact CBE's
pair-metric trimming is unchanged. Mixed real/complex metric factors are
promoted to a common dtype before iterative least-squares solves.

Compression now also checks metric separability: a verified product metric
uses supported weighted SVD; a general metric retains the local ALS fit.
This dispatch applies to both conditional and global compression. The norm
loss is evaluated from the actual reconstruction residual, not a difference
of nearly equal objective values.

`--global-trim` disables the shared-configuration split for a matched ablation.
It retains the current metric-aware SVD/ALS dispatch, so it does not reproduce
the entire historical implementation.
This switch also works in the individual model scripts and
`run_condensed_suite.py`. The solver option is `cbe_conditional_trim=False`.
For `cbe_harder_cases.py`, `--seed` (default 731) sets the seed directly for
every case, independently of D and case order. Filtering cases does not change
their initial states. JSON records the trim mode, seed, actual per-case
`bond_dim` and initial fingerprint; the suite-level `bond_dim` is the optional
override (`null` when using the case values).

The conditional fix is not a guarantee of better convergence for every seed.
In local checks it reduced eight-site Heisenberg D=4 from 34 passes to 5;
three additional seeds changed from 50, 50, 6 passes to 5, 5, 5. Six-site
Fermi Hubbard D=4 instead changed from 18 to 20 passes, and Bose Hubbard
still reached the 50-pass cap. Compare final energy and convergence flags,
not just runtime. Those measurements predate the general-selector rewrite.
The current strict selector removes both old one-site tangents from its
physical target, but evaluates that target only in its proposed candidate
space. This restricted space need not capture the exact selector's optimum.

## Complete-solver scaling

Click-run `cbe_update_scaling.py` to compare ordinary one-site, strict CBE,
and two-site solver costs on 1D, 2D, and 3D dependency patterns. It varies
D, physical dimension, and preselection width one axis at a time. These are
short cost probes with Hermitian product-sum MPOs, not convergence evidence
for the condensed-matter models above.

```bash
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_update_scaling.py --output update-scaling.json
```

Each point uses one identical initial LETTA state for every method. A separate
instrumented warm-up records contraction-path work and SVD/eigh proxies;
three uninstrumented repetitions measure complete solver time, including
environment setup, projection, eigensolves, trim and baseline fallback.
Use `--passes`, `--repeats`, `--geometries`, `--bond-dimensions`,
`--physical-dimensions`, and `--preselection-dimensions` to adjust the study.
Memory numbers estimate the largest instrumented array, not peak RSS or the
sum of concurrently live arrays. Work proxies omit uninstrumented matrix
products and iterative vector algebra. Output is checkpointed per point.

The condensed-model JSON now also records strict selector substage times,
metric-kind counts, maximum supported tangent residual, largest tangent
cross-Gram block, connector dimension, and overlap/fit iteration counts.
Selector substage times are contained within `cbe_phase_seconds.selection`;
do not add them to the outer phase totals a second time.

LETTA JSON records also contain `sweep_elapsed_seconds`, aligned with
`sweep_energies`. These cumulative timestamps include solver initialization
and permit comparison of time to a common energy target. They are not per-pass
durations. `cbe_trim_metric_kinds` counts the compression metric dispatches.

## QR versus frontier gauge: click-run convergence comparison

Open `run_gauge_convergence.py` and click Run. Edit its `DEFAULTS` block to
change the system or tolerances; no command-line arguments or working-directory
setup are required. By default it compares an eight-site nearest-tied Ising
chain at `D=4,8,16`, field `0.9`, seed `73`, with three timing repetitions and
up to 12 directional sweeps. Numerical libraries are set to one thread before
imports; restart an existing IDE kernel if it already loaded those libraries.

```bash
python pyqed/_letta_one_site_opt/benchmarks/run_gauge_convergence.py
python pyqed/_letta_one_site_opt/benchmarks/run_gauge_convergence.py \
  --columns 10 --bond-dims 8 16 --max-sweeps 20 --repeats 3
```

Use `--rows 2 --columns 3 --bond-dims 3` for a small lattice with obstructed
cuts. `--exact-max-dimension 0` skips the dense reference; `--no-plot` skips
Matplotlib. `--help` lists all overrides. Outputs go to
`output/benchmarks/gauge_convergence/` by default (overwritten on rerun); use
`--output /your/path` to keep separate experiments:

- `results.json`: all repetitions, exact energy when available, stopping
  messages, physical residuals, time to the requested total-energy error,
  per-site metric ranks and types, and cache reuse counts.
- `sweeps.csv`: energy, exact-energy error, energy-density change, cumulative
  solver time and pass time for every sweep. The first pass time includes
  initialization.
- `convergence.png`: energy error versus sweep and time, plus the stopping
  statistic. Each trajectory is the actual median-runtime repetition for its
  gauge and bond dimension. Display floors are labelled; raw data are unchanged.

Both gauges start from exactly the same tensors within each bond dimension;
repetitions reuse seed 73 to measure timing variability, not robustness across
initializations. Change `seed` separately to test that. Method order alternates
between repetitions. Solver timings include initialization and exclude exact
references, fresh physical-energy validation, and plotting. For larger systems,
exact diagonalization is automatically skipped above the configured Hilbert
dimension limit.

The energy is the normalized Rayleigh quotient
$E=\langle\Psi|H|\Psi\rangle/\langle\Psi|\Psi\rangle$. A one-site step solves
$H_i a=E N_i a$ on the retained metric support. An identity norm simplifies that
problem without changing its exact physical minimum. Finite precision, rank
cutoffs and iterative tolerances can nevertheless give different numerical
trajectories. The local update is accepted only if it does not increase energy
beyond `energy_increase_tolerance` (default `1e-9`).

One sweep means **one directional pass**, LR or RL. The outer solver stops when
$|E_k-E_{k-1}|/n\leq\texttt{tolerance}$; it does not stop on error relative to
exact diagonalization or a global residual. Consequently `converged=True`
means energy stabilization, and a fixed-bond local minimum or rejected updates
can also satisfy it. Check exact-energy error and final physical residual
$\|H\psi-E\psi\|$ when available. A small residual alone does not identify the
ground state. Local residuals are coordinate-dependent and should not be used
as gauge-independent accuracy comparisons. `target_error` is a benchmark
measurement of total-energy accuracy, independent of the solver stopping rule.

The outer `tolerance`, local `eigensolver_tolerance`, and support
`metric_tolerance` have different roles. Tightening only the outer threshold
cannot repair inadequate bond dimension or an inaccurate local solve. The
one-site and two-site option defaults now select `gauge_mode="frontier"`, so
the condensed-model and harder-case LETTA comparisons use it automatically.
Set `gauge_mode="qr"` in `LETTADMROptions` or `LETTATwoSiteOptions` for an
explicit QR comparison. Full $N_i=I$ requires full conditional rank; otherwise
the metric is identity on the supported coordinates, with tiny unresolved
modes retained by the invertible gauge.

Two-site sweeps canonicalize toward the starting center and maintain the
outgoing complete norm environment after each pair update. This also works
with repeated sweeps in one direction and optional one-site polishing.
Structurally inadmissible cuts (possible for wider dependency patterns) use
the existing fixed-size QR fallback; they are not assumed to be identities.
Frontier gauges do not support reduced SU(2) states; those runs must explicitly
select `gauge_mode="qr"`. Gauge transformations preserve the physical state,
but finite-tolerance solves and fixed-bond truncation can produce different
optimization trajectories. See the two-site frontier validation report in
`docs/benchmarks/2026-09-14-letta-two-site-frontier.md`.

## Additional 1D nearest-neighbor chains

`nn_chain_1d.py` runs any registered model on the nearest-tied LETTA shape
`(1, length)`. Its default is SSH and the three methods one-site, two-site,
and strict CBE; exact CBE is off. Edit `MODEL` for IDE click-run or use
`--model`. All interactions are on-site or nearest-neighbor, with open ends.
SSH here is **spinless electronic hopping with prescribed alternating bonds**;
there are no dynamical SSH phonons. Fermion benchmarks do not fix particle
number or parity. Spin-1/2 operators are $S^a=\sigma^a/2$; the existing
`ising` model uses Pauli matrices instead.

| Model name | Hamiltonian / convention | Local dimension |
|---|---|---:|
| `ssh` | $-\sum_i t_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})-\mu\sum_i n_i$ | 2 |
| `rice_mele` | SSH plus $m\sum_i(-1)^i n_i$, with `stagger` = $m$ | 2 |
| `spinless_tv` | $-t\sum_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})+V\sum_i q_iq_{i+1}-\mu\sum_i q_i$, $q_i=n_i-1/2$ | 2 |
| `kitaev` | $-t\sum_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})+\Delta_p\sum_i(c_ic_{i+1}+\mathrm{h.c.})-\mu\sum_i(n_i-1/2)$; `pairing` = $\Delta_p$ | 2 |
| `xy` | $J\sum_i[(1+\gamma)S_i^xS_{i+1}^x+(1-\gamma)S_i^yS_{i+1}^y]-h\sum_i S_i^z$ | 2 |
| `xyz` | $\sum_{i,a}J_a S_i^aS_{i+1}^a-h\sum_i S_i^z$ | 2 |
| `dimerized_heisenberg` | XXZ with $J_i=J[1+(-1)^i\delta_d]$; `dimer` = $\delta_d$ | 2 |
| `spin1_heisenberg` | Spin-1 XXZ plus $A\sum_i(S_i^z)^2$, with `single_ion` = $A$ | 3 |
| `blume_capel` | $-J\sum_i S_i^zS_{i+1}^z-h\sum_i S_i^x+A\sum_i(S_i^z)^2$ for spin 1 | 3 |

SSH and Rice–Mele use `t1` on bonds starting at even site indices and `t2`
on odd indices, counting from zero. For even lengths, `t1 < t2` gives weak
end bonds. The t–V interaction is explicitly centered; its `mu=0` convention
includes the corresponding boundary fields and constant energy offset.
All other parameter names and defaults are available with model-specific help:

```bash
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/nn_chain_1d.py --model ssh --help
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/nn_chain_1d.py \
  --model ssh --length 8 --bond-dim 4 --t1 0.6 --t2 1.4 --max-sweeps 20
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/nn_chain_1d.py \
  --model ising --length 8 --bond-dim 4 --h 1.0
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/nn_chain_1d.py \
  --model spin1_heisenberg --length 6 --bond-dim 3 --single-ion 1.0
```

### 29 ready-to-run CBE presets with figures

`nn_chain_cases.py` contains editable lengths, bond dimensions and parameter
overrides. It includes three Ising field strengths; Heisenberg/XX/easy-plane/
easy-axis XXZ; XY/XYZ; weak and stronger dimerization; weak-end/strong-end/
uniform SSH; Rice–Mele; three t–V interactions; three Kitaev chemical
potentials; three spin-1 XXZ settings; and two Blume–Capel settings. Additional
12-site Ising, Heisenberg and SSH cases probe larger chains at the same D.
Preset regime names identify input choices, not measured phases of the finite
chain or a guarantee of improved CBE convergence.

The presets are registered in `cbe_harder_cases.py` without changing its
existing click-run case selection. Use `--list-cases` to inspect them,
`--cases` to select a few, or `--nn-chains` to run all 29. For IDE click-run,
set `DEFAULT_CASES = tuple(NN_CHAIN_CASES)` after the registration block, or
set it to a tuple of selected preset names. `CASE_PARAMETERS` holds overrides
for the four-element entries in `HARD_CASES`.

```bash
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py --list-cases
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py \
  --cases ssh_weak_end_8_4,ising_critical_8_4,heisenberg_chain_8_4 \
  --max-sweeps 20 --output /private/tmp/nn-chains.json
.venv-1/bin/python pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py \
  --nn-chains --max-sweeps 20 --exact-max-dimension 512 \
  --output /private/tmp/all-nn-chains.json
```

The full preset collection is a benchmark workload, not a quick smoke test.
`--exact-max-dimension 512` skips dense diagonalization for larger Hilbert
spaces; this reference cutoff is independent of the exact-CBE selector flag.
The harder-case figures retain Hamiltonian labels, signed-log energy
comparisons, and filled magenta strict-CBE markers wherever
`sweep_cbe_accepted` is positive. Exact CBE remains off by default in this
runner and the new single-chain entry point. The older model-specific scripts
and `run_condensed_suite.py` retain their existing five-solver default.

## General ties and focused 2D comparisons

The `2D/` directory provides three-method comparisons (one-site, strict CBE,
optional two-site), a `--skip-two-site` switch, a separate
`--two-site-max-sweeps` cap, JSON histories, and custom dependency patterns.
See [2D/README.md](2D/README.md) for examples including diagonal and backward
ties. These scripts use the same Hamiltonian builders and common-state
initialization as the benchmarks above.

## CBE energy refinement (22 September 2026)

After reducing the expanded bond back to its fixed dimension, CBE now
alternates one-site energy solves over A and B. It independently relaxes both
the metric-trimmed pair and the pre-expansion incumbent, then compares their
final energies. This prevents an attractive pre-refinement energy from
selecting an inferior fixed-rank route. The strict selector retains its
streamed implementation: refinement does not build a merged pair tensor,
pair Hamiltonian/metric, or full-system projection.

`LETTADMROptions.cbe_energy_refinement_max_iterations` defaults to three
alternating passes **per start**, stopping earlier at an absolute local energy
improvement of `cbe_energy_refinement_tolerance` (default `1e-10`). Set the
iteration count to zero for the previous norm-only trim and its lower cost.
The benchmark `run_benchmark` accepts the same iteration-count keyword. The
historical `cbe_energy_trim.py` ablation explicitly disables the production
refinement to avoid applying it twice.

`cbe_trimmed_energy` and `cbe_trim_loss` still describe the metric-compression
initializer. They do not describe the energy-relaxed candidate. The new
`cbe_refined_energy`, `cbe_incumbent_refined_energy`, and
`cbe_energy_refinement_start` distinguish the two relaxed routes. Accepted
CBE updates may select the relaxed incumbent; their acceptance alone is not
evidence of useful bond expansion. Reports count such selections separately.
`energy_refinement` phase timing and local Hamiltonian-application totals
include both routes, including the discarded one. The usual finite-energy,
non-increase, and ordinary one-site baseline guards still apply.

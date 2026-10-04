# LETTA active-space chemistry benchmarks

This extends the H2/H4 checks to LiF and N2 at equilibrium and stretched bonds,
plus water, using 6-31G and CAS(6,6)/CAS(8,8).

**Correction to the original interpretation:** the cold-start table below
compared DMRG with adaptive charge-sector multiplicities against LETTA with
fixed initial multiplicities. Equal total D did not establish nested
variational spaces. It therefore cannot establish that LETTA has worse
variational accuracy. A follow-up using exact embeddings of the optimized
DMRG states confirms that NN LETTA matches or improves every tested energy.

The implementation and results belong to `codex/letta-qchem` in the isolated
`pyqed-letta-qchem` checkout. The original `bg` checkout and its stashes are
preserved as described in [the initial report](letta_ground_state.md).

## Exact DMRG embedding control

For compatible bond sectors, MPS is a subset of LETTA: a tied tensor can
ignore its extra physical labels. Thus the variational optimum satisfies
$E^\star_{\mathrm{LETTA}}(D)\le E^\star_{\mathrm{MPS}}(D)$. With the actual DMRG state embedded and
energy-nonincreasing updates, the achieved LETTA energy should also not exceed
its DMRG starting energy. Strict improvement is not guaranteed.

Canonical orbitals, CAS(6,6), D=32, seed 731; errors in mEh:

| System | DMRG error | NN LETTA from DMRG error |
|---|---:|---:|
| LiF, 1.56 Å | 0.001958419 | 0.001296446 |
| LiF, 3.00 Å | 0.018980552 | 0.0081968631 |
| N2, 1.10 Å | 0 | 0 |
| N2, 2.00 Å | 2.6022235e-05 | 0 |

All four embeddings preserve the normalized vector to within 9e-16 in norm
and the energy to within 1e-14 Eh. The audit records both original and DMRG
bond-sector multiplicities and independent PySCF final-state diagnostics.

The restricted-space issue is measurable in LiF: at the middle cut, with
left charge (2,2) and the next orbital fixed to empty, the DMRG vector has
conditional Schmidt rank 4, with its fourth singular value about 0.00790.
The original NN LETTA allocation allowed only 3 virtual states in that sector,
so its conditional rank is at most 3. It cannot represent that DMRG state.
This is a restriction of the allocation used in the benchmark, not of the
general LETTA ansatz. It coexists with the observed initialization sensitivity.

Reproduce the inclusion-preserving comparison with `--methods warm-nn`.
The benchmark now verifies exact embedding and rejects a final energy above
its DMRG starting energy by more than 1e-9 Eh. Raw audit data are in
`letta_active_space_results/inclusion_audit.json` and `sector_rank_audit.json`.

## Original cold-start comparison

These are optimizer outcomes with different sector-allocation protocols,
not a comparison of nested variational spaces.

Canonical index order, seed 731. Errors are relative to CASCI, in mEh.

| System | CAS | Cap | NN LETTA error | DMRG error | NN / DMRG coefficients |
|---|---|---:|---:|---:|---:|
| LiF, 1.56 Å | (6,6) | 32 | 0.0933952 | 0.00195842 | 1732 / 494 |
| LiF, 3.00 Å | (6,6) | 32 | 0.0941805 | 0.0189806 | 1732 / 494 |
| N2, 1.10 Å | (6,6) | 32 | 0.46081 | 0 | 1732 / 480 |
| N2, 2.00 Å | (6,6) | 32 | 18.6649 | 2.60222e-05 | 1732 / 480 |
| Water | (8,8) | 25 | 18.8338 | 0.874148 | 2220 / 916 |

Water uses a five-pass LETTA run versus up to 30 DMRG passes; its LETTA state
is not converged. The CAS(6,6) rows use up to 30 passes for both solvers.
Stretched-N2 LETTA is strongly seed dependent: seed 947 at cap 32 reduces its
natural-order error to 0.3457 mEh. All comparisons retain the residual and spin
diagnostics below; energy agreement alone does not establish ground-state convergence.

## Protocol

`pyqed/_letta_one_site_opt/benchmarks/qchem_active_space.py` builds RHF canonical
orbitals with PySCF, freezes the lowest occupied orbitals, and takes the next
`ncas` orbitals as a contiguous active window. There is no CASSCF orbital
optimization. Exact canonical-space `h1`, `eri`, electron counts and core
energies are also archived as NPZ files in `letta_active_space_results/integrals`. The JSON
specifies the geometry, active MO indices, orbital energies, frozen-core energy,
integral hash, versions, seed, and ordering.
CASCI supplies the exact reference within each selected active space; it is
not the all-electron full-basis FCI energy.

Both optimizers use the same electronic Hamiltonian MPO, with the frozen-core
and nuclear constant added back to final energies. The new `symbolic` MPO
backend uses pyqed's existing Hopcroft–Karp AutoMPO builder on Jordan–Wigner
operator products. This avoids the large intermediate CP-SVD arrays of the
initial adapter. No integral screening is requested. The solver path never
forms a full determinant-space Hamiltonian or a full state vector. Full
vectors are used only by these small-CAS reference diagnostics and the MI
ordering oracle.

The baseline is the actual `pyqed.mps.dmrg.DMRG` two-site Abelian solver, using
its symmetry-block tensors. LETTA uses the existing one-site optimizer,
frontier gauge, matrix-free local solves, and no CBE. Both conserve electron
number and spin projection; neither enforces total spin. Each method at a
given cap/order/seed receives the same HF-biased random, charge-complete MPS.
Ties are initialized by exact broadcasting, so they do not change the initial
wavefunction. The separately labelled `warm-nn` control first runs DMRG,
preserves its adapted bond charges, and then embeds the resulting MPS into
NN LETTA. Its DMRG setup cost is recorded separately. Initial tensor hashes and energies are saved. Across orderings,
the random perturbations are regenerated; these are not strictly identical
physical initial states.

The HF bias multiplies each random local tensor by 0.1 and sets its RHF
occupation-path entry to 1 before normalization. It avoids the excited
triplet found in a random-start LiF pilot, but does not guarantee the singlet
root at restricted bond dimensions. Every result includes total spin squared,
CASCI overlap, sector leakage, energy error, and the residual norm
`||H psi - E psi||`. These diagnostics use an independent PySCF CI action.
An energy-stagnation flag is not a ground-state convergence certificate.

The main CAS(6,6) runs use up to 30 directional passes with energy tolerance 1e-10,
local eigensolver tolerance 1e-10, and LETTA metric threshold 1e-12. The
untied one-site control uses 20 passes. The completed long-link pilot uses
one pass for each topology; exploratory longer attempts are recorded separately.
The completed water NN run uses five passes. DMRG's history also includes a final
recentring operation, labelled separately; its callback energy can be a
pre-truncation local energy. Tables use the independently recomputed energy
of the returned state.

Parameter counts are symmetry-allowed stored coefficients before quotienting
out gauge freedom. DMRG dynamically allocates bond sectors; the present LETTA
start retains all reachable sectors with fixed multiplicities. Equal bond
caps therefore do not imply equal parameter counts, equal representational
capacity, or equally effective optimization. Dense storage counts are saved
separately.

Timings are local single runs with BLAS/OpenMP limited to one thread, without
optional compiled DMRG acceleration. Several jobs ran concurrently on separate
processes, so these are indicative wall times rather than controlled speed
ratios. Solver time excludes molecular setup, MPO construction and independent
diagnostics; DMRG symmetry-conversion time is recorded separately.

## Reproduce

Dependencies include NumPy, SciPy, opt_einsum, PySCF, and the dependencies of
pyqed's MPS/AutoMPO modules (including tensorly, networkx, gbasis and sympy).
The session used the existing Python 3.12 environment with pure-Python
packages copied from another existing environment into
`/private/tmp/letta-qchem-test-deps`; no existing environment was modified.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.
python -m pyqed._letta_one_site_opt.benchmarks.qchem_active_space \
  --case lif_eq --bond-dims 16 32 --max-sweeps 30 \
  --output /private/tmp/lif_eq.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_active_space \
  --case water --bond-dims 25 --methods nn --max-sweeps 5 \
  --output /private/tmp/water.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_active_space \
  --case n2_stretched --order correlation --bond-dims 16 32 --max-sweeps 30 \
  --output /private/tmp/n2_ordered.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_active_space \
  --case n2_stretched --bond-dims 16 --methods nn direct carried --max-sweeps 1 \
  --output /private/tmp/n2_links.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_active_space \
  --case lif_stretched --bond-dims 32 --methods warm-nn --max-sweeps 30 \
  --output /private/tmp/lif_stretched_warm.json
python -m pytest -q tests/test_letta_qchem_active_space.py tests/test_letta_qchem.py \
  tests/test_letta_frontier_gauge.py tests/test_letta_general_ties.py tests/test_letta_symmetry.py
```

Available cases also include `lif_stretched` and `n2_eq`. Use `--seed 947` for
the second-start control, `--bond-dims 64 --methods dmrg` for the CAS(6,6)
reference checks, and `--initialization random` for the LiF initialization
diagnostic. Correlation ordering uses FCI mutual information as an oracle;
it is not a scalable orbital-selection algorithm. The default ordering is
RHF orbital-energy index order, not natural orbitals.

## Validation

The inclusion follow-up passes all 17 focused chemistry tests. The new
regression test demonstrates the old conditional-rank restriction, verifies
exact embedding with adapted DMRG charges, and checks nonincreasing LETTA
energy from that state.

61 focused tests passed, including reordered CAS(6,6) complex-vector actions
against PySCF without constructing a dense Hamiltonian, frozen-core handling,
exact initial-state tie embedding, HF occupation labels after reordering, and
singlet ground-state recovery with pyqed DMRG. The symbolic MPO also agrees
with the previous SVD backend on H4. Existing gauge, symmetry, and general-tie
tests remain passing.

For water, CASCI's energy tolerance produced an eigenvector residual of about
2.55e-7 Eh. MPO validation therefore compares the two independently computed
Hamiltonian actions directly, rather than demanding a smaller residual from
the reference eigenvector. The reference energy is stable to the displayed
precision.

A tighter water CASCI calculation (energy tolerance 1e-14, Krylov space 50)
changed the reference energy by only -1.42e-14 Eh and reduced its residual to
7.88e-8 Eh.

## Interpretation

The original cold-start runs expose a sector-allocation and optimization
limitation. Their higher LETTA energies do not compare variational expressiveness.
The exact-embedding follow-up above supersedes that interpretation. At cap 32,
equilibrium LiF has a 0.0934 mEh LETTA error versus 0.00196 mEh for DMRG;
stretched LiF gives 0.0942 versus 0.0190 mEh. LETTA uses 1,732 coefficients,
while these DMRG states use 494. Equilibrium N2 reaches numerical CASCI with
DMRG cap 32 and 480 coefficients, while NN LETTA at cap 64 needs 3,508
coefficients to reach the same reference. These are conclusions about the
current solver and sector allocation, not an expressiveness bound on LETTA.

The NN and carried layouts satisfy the shared-frontier condition. Their
observed local metrics are identity or supported identity. Good conditioning
has therefore been verified, but it does not remove one-site optimization
basins or inadequate fixed sector multiplicities. The current CBE solver
explicitly rejects symmetry sectors, so it cannot simply be switched on for
these charge-conserving chemistry states.

Stretched N2 is the more discriminating case. In canonical index order, NN
LETTA cap 32 ends at 18.665 mEh error with seed 731 and 0.346 mEh with seed
947. The former has S² approximately 2 and the latter is close to a singlet.
This is substantial initialization/root sensitivity. Even a small residual
can identify the wrong state: random-start LiF DMRG cap 64 reaches a triplet
275.793 mEh above the singlet CASCI reference. That run is retained as a
separate diagnostic, not used as a ground-state baseline.

For stretched N2, the strongest three MI pairs are `(1,4)`, `(2,3)`, and
`(0,5)` in zero-based active-orbital indices. Their MI values are approximately
1.0411, 1.0411, and 0.9734 nats. The oracle order `(1,4,2,3,0,5)` puts all
three pairs next to one another and reduces the MI-weighted squared-distance
objective from 39.4944 to 12.7553. At cap 16 and seed 731, DMRG error improves
from 85.315 to 0.253 mEh and NN LETTA from 55.592 to 7.588 mEh. At cap 32,
ordered LETTA gives 1.330 mEh, but the second natural-order seed does better.
Ordering helps this tested start; it does not guarantee the best optimized
state. FCI supplies the MI oracle, and the random perturbations are not
identical physical wavefunctions across orderings.

A DMRG warm start provides a separate control for the initial sector choice.
For equilibrium N2 at cap 16, NN ties improve the DMRG energy error from
0.2826 to 0.1869 mEh, with 1,036 allowed coefficients after embedding the
adapted DMRG charges. The tie optimization takes 12.45 seconds after about
3.00 seconds of DMRG. This is a real variational improvement at fixed cap,
but DMRG at cap 32 is both more accurate and smaller in coefficient count.

The next implementation priorities are adaptive charge-sector multiplicities
or symmetry-aware enrichment, reliable singlet-root targeting, and profiling
the spatial-dimension-four contraction path. Order strongly correlated pairs
adjacently before adding long ties. Any long tie must be judged against its
larger frontier and coefficient budget, not just its nominal virtual cap.
The tested distant pairs are distant in chain index; canonical orbitals are
delocalized, so this is not evidence for a spatially distant chemical pair.

## Longer-run inspection

The original water NN cap-25 run was interrupted after roughly 15 minutes
to inspect progress. Buffered output revealed 15 completed passes and a
last logged error of about 17.85 mEh. No final state from that interrupted
process was retained, so this is not included as a validated solver result.
The separate five-pass run saves a returned state and independent diagnostics.
The interruption and logged energy history are preserved in
`letta_active_space_results/interrupted_attempts.json`. Use `python -u` for
unbuffered sweep logging when redirecting benchmark output.

The longer direct/carried N2 pilots were also interrupted to replace them
with a matched one-pass topology test. The direct pilot logged ten passes
(about 9.69 mEh error at the last logged pass) in roughly 20 minutes; the
carried pilot logged one pass in roughly 11 minutes before interruption
during the next pass. Neither retained a validated final state. Their
progress is archived alongside the water interruption, not treated as
converged or independently validated results.

## Controlled long-tie pilot

The matched one-pass pilot uses stretched N2, seed 731, cap 16, and the same
HF-biased MPS for all three layouts. The added pair is `(1,4)`, selected by
FCI MI. NN has frontier width one. Both long-tie layouts have width two,
raising explicit Hamiltonian bra/ket frontier configurations from 16 to 256.
The direct tie violates the shared-frontier condition; carrying label 4
through the intervening tensors restores it and enlarges the ansatz.

This pilot measures early optimization progress, not converged accuracy.
In particular, the NN state after one pass is triplet-like (S² about 2.054),
whereas the direct-tie state has S² about 0.152. The energy changes therefore
also reflect different spin/root trajectories. They cannot be attributed
solely to additional correlation capacity in a fixed singlet manifold.

| Layout | Error (mEh) | Coefficients | Seconds | S² | Metric kinds |
|---|---:|---:|---:|---:|---|
| nn | 99.6924 | 820 | 4.42 | 2.0542 | identity, supported_identity |
| direct | 56.7012 | 1252 | 104.58 | 0.1524 | identity, general, supported_identity |
| carried | 25.7638 | 2020 | 554.01 | 0.0434 | identity, supported_identity |

Carrying restores supported-identity metrics and lowers the one-pass energy
further, but it changes the ansatz and costs about nine minutes for this single
pass. This confirms a useful long-link direction while exposing a substantial
contraction bottleneck. It does not demonstrate a converged advantage over DMRG.

The matched initial energies and tensor fingerprints are identical across
these three runs. Direct and carried energies agree with independent PySCF
CI actions and both preserve the particle sector. Their substantial residuals
show that neither one-pass state is converged.

## Measured results

The original 39 solver runs are saved in [the JSON results](letta_active_space_results).
The separate inclusion audit adds four DMRG and four LETTA runs.
Errors below are in millihartree (mEh); residuals are in Hartree. Values near
zero denote agreement within numerical tolerance, not a larger-space result.

| Case/order | Seed | Method | Cap | Error (mEh) | Residual (Eh) | S² | Coefficients | Solver seconds |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| lif_eq / natural | 731 | DMRG-2site | 16 | 0.0394743 | 0.00857 | -2.22e-16 | 238 | 0.60 |
| lif_eq / natural | 731 | LETTA-nn | 16 | 0.304357 | 0.0189 | 0.0002637 | 820 | 68.29 |
| lif_eq / natural | 731 | DMRG-2site | 32 | 0.00195842 | 0.00231 | 0 | 494 | 0.51 |
| lif_eq / natural | 731 | LETTA-nn | 32 | 0.0933952 | 0.00966 | 0.0002757 | 1732 | 107.27 |
| lif_eq / natural | 731 | DMRG-2site | 64 | 1.42109e-11 | 1.95e-09 | -2.22e-15 | 880 | 0.79 |
| lif_eq / natural | 731 | DMRG-2site (random start) | 64 | 275.793 | 7.68e-10 | 2 | 880 | 1.71 |
| lif_stretched / natural | 731 | DMRG-2site | 16 | 0.274713 | 0.0179 | -8.882e-16 | 214 | 3.89 |
| lif_stretched / natural | 731 | LETTA-nn | 16 | 0.305359 | 0.0163 | 0.0001351 | 820 | 77.76 |
| lif_stretched / natural | 731 | DMRG-2site | 32 | 0.0189806 | 0.00687 | 1.831e-06 | 494 | 6.05 |
| lif_stretched / natural | 731 | LETTA-nn | 32 | 0.0941805 | 0.00819 | 0.0003198 | 1732 | 156.55 |
| lif_stretched / natural | 731 | DMRG-2site | 64 | 1.42109e-11 | 6.06e-10 | 8.882e-16 | 880 | 0.76 |
| n2_eq / natural | 731 | DMRG-2site | 16 | 0.282602 | 0.0274 | 3.341e-05 | 262 | 3.04 |
| n2_eq / natural | 731 | LETTA-nn | 16 | 2.61365 | 0.0603 | 0.0003241 | 820 | 20.62 |
| n2_eq / natural | 731 | DMRG-2site | 32 | 0 | 8.94e-10 | 1.332e-15 | 480 | 0.50 |
| n2_eq / natural | 731 | LETTA-nn | 32 | 0.46081 | 0.0249 | 0.0008106 | 1732 | 33.03 |
| n2_eq / natural | 731 | DMRG-2site | 64 | 0 | 9.1e-10 | 1.11e-15 | 880 | 0.77 |
| n2_eq / natural | 731 | LETTA-nn | 64 | 0 | 5.52e-14 | 3.553e-15 | 3508 | 70.31 |
| n2_eq / natural | 947 | DMRG-2site | 32 | 0 | 1.18e-09 | -2.665e-15 | 480 | 0.64 |
| n2_eq / natural | 947 | LETTA-nn | 32 | 0.46081 | 0.0249 | 0.0008106 | 1732 | 36.58 |
| n2_eq / natural | 731 | LETTA-nn_from_dmrg | 16 | 0.186872 | 0.0218 | 8.008e-05 | 1036 | 12.45 |
| n2_stretched / natural | 731 | DMRG-2site | 16 | 85.3148 | 0.28 | 2 | 296 | 10.33 |
| n2_stretched / natural | 731 | LETTA-nn | 16 | 55.592 | 0.119 | 2.005 | 820 | 96.28 |
| n2_stretched / natural | 731 | DMRG-2site | 32 | 2.60222e-05 | 0.000101 | 3.865e-06 | 480 | 1.46 |
| n2_stretched / natural | 731 | LETTA-nn | 32 | 18.6649 | 0.0441 | 2 | 1732 | 114.31 |
| n2_stretched / natural | 731 | LETTA-carried | 16 | 25.7638 | 0.102 | 0.04337 | 2020 | 554.01 |
| n2_stretched / natural | 731 | LETTA-direct | 16 | 56.7012 | 0.152 | 0.1524 | 1252 | 104.58 |
| n2_stretched / natural | 731 | DMRG-2site | 64 | 0 | 5.71e-10 | 6.661e-16 | 880 | 1.97 |
| n2_stretched / natural | 731 | LETTA-mps | 16 | 109.468 | 0.177 | 2.008 | 208 | 27.41 |
| n2_stretched / natural | 731 | LETTA-nn | 16 | 99.6924 | 0.195 | 2.054 | 820 | 4.42 |
| n2_stretched / correlation | 731 | DMRG-2site | 16 | 0.253332 | 0.0113 | -4.441e-16 | 320 | 9.24 |
| n2_stretched / correlation | 731 | LETTA-nn | 16 | 7.58846 | 0.0416 | 0.7636 | 820 | 73.63 |
| n2_stretched / correlation | 731 | DMRG-2site | 32 | 0 | 4.67e-10 | 0 | 508 | 1.81 |
| n2_stretched / correlation | 731 | LETTA-nn | 32 | 1.32987 | 0.0255 | 1.134e-09 | 1732 | 221.06 |
| n2_stretched / natural | 947 | DMRG-2site | 32 | 0 | 3.81e-09 | -2.22e-16 | 480 | 2.07 |
| n2_stretched / natural | 947 | LETTA-nn | 32 | 0.345702 | 0.0107 | 0.006817 | 1732 | 75.79 |
| water / natural | 731 | DMRG-2site | 25 | 0.874148 | 0.0595 | 4.355e-06 | 916 | 22.13 |
| water / natural | 731 | DMRG-2site | 64 | 0.0537043 | 0.0163 | 2.764e-06 | 3204 | 44.27 |
| water / natural | 731 | DMRG-2site | 128 | 0.00118503 | 0.0027 | 1.776e-15 | 6186 | 88.55 |
| water / natural | 731 | LETTA-nn | 25 | 18.8338 | 0.209 | 0.005412 | 2220 | 312.80 |

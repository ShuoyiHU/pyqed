# LETTA ground-state quantum chemistry: first controlled tests

Larger LiF/N2 CAS(6,6) and water CAS(8,8) experiments using pyqed two-site DMRG
are in [the active-space report](letta_active_space.md).

## Material Passport

- Mode: implemented experiment and reproducibility validation.
- Date: 2026-10-02 (Asia/Shanghai).
- Scope: H2 and H4, full STO-3G active spaces; real spatial orbitals.
- Source method: [Hu and Gu, arXiv:2609.30101v1](https://arxiv.org/html/2609.30101v1), especially Eqs. (2), (4), (12), and Supplement S3A.
- Local context consulted: the project chats “LETTA Gauge” and “LETTA-CBE implementation (2)”.
- Evidence: executable benchmark, independent determinant/FCI tests, and JSON records in [letta_results](letta_results).
- Status: measured small-system results, not evidence of a scalable chemical advantage.

## What is implemented

The implementation lives alongside the existing one-site solver:

- `pyqed/_letta_one_site_opt/qchem.py`: real electronic integrals, a compressed operator-product MPO, fixed electron sectors, and a shared MPS initial state.
- `pyqed/_letta_one_site_opt/orbital_ordering.py`: fermionic permutations, orbital mutual information, ordering, budgeted tie selection, and graph diagnostics.
- `pyqed/_letta_one_site_opt/benchmarks/qchem_ground_state.py`: H2/H4 comparisons and local metric audits.
- `tests/test_letta_qchem.py`: independent fermionic, sector, gauge, and FCI checks.

The spatial basis is `|0>, |alpha>, |beta>, |alpha beta>`. Integrals are in an orthonormal basis, in Hartree, with chemists' ERIs:

$$
H=E_{\rm core}+\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\tfrac12\sum_{pqrs\sigma\tau}(pq|rs)
 a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
$$

Nuclear repulsion is included in the reported total energies. No particle-number penalty is used. The original tensor-network optimizer and gauge routines remain unchanged. Neither dense state vectors nor determinant-space Hamiltonians enter the solver; they are used only by small-system diagnostics.

The MPO builder combines Jordan–Wigner operator strings and compresses their sum through right and left SVD sweeps. It removes numerical null directions using machine epsilon times matrix size times the largest singular value. It never builds a full electronic Hamiltonian during construction. Integral cutoff is zero in the saved experiments; the MPO agrees with independent FCI actions to numerical precision. The product-sum representation and its memory guard are a starting point for small active spaces, not a replacement for a scalable complementary-operator chemistry MPO.

## Charge sectors and initial states

We conserve `(Nalpha, Nbeta)` using the existing Abelian machinery. Each physical charge is counted at its owner tensor, once. This fixes particle number and spin projection, not total spin.

`max_bond_dim` is a cap on the total virtual dimension. Every reachable charge sector is included before adding multiplicities. Too-small caps raise an error instead of silently selecting a restricted subset of determinants. For half-filled H4, the middle bond needs at least nine charge labels; a cap of 9 yields bond dimensions `(4,9,4)`, and a cap of 16 yields `(4,16,4)`.

Each method at a given orbital ordering starts from the same random, charge-conserving MPS, broadcast over any added physical axes. Seeds and initial energies are recorded. Different orderings use independently generated initial states with the same seed; they are not claimed to start from the same physical state. A common seed therefore does not eliminate optimization-basin effects across orderings.

This allocation deliberately retains all MPS-reachable charge sectors. It is not an optimized charge allocation for LETTA: conditioning the virtual charges on tied labels could recover more of the low-D flexibility and is a separate research question.

## Gauge and long-range ties

For an NN-tied chain, the shared-frontier condition holds at every cut. Our tests measure identity on the **supported** local metric; rank-deficient coordinates do not become full identity by an invertible gauge. This agrees with the rank qualification in Supplement S3A.

The direct long-range example uses chain order `(0,2,3,1)` for two H2 fragments. The strongly correlated original orbitals `(0,1)` occupy chain positions `(0,3)`. Its dependencies are

```text
NN:       (0,1),   (1,2),   (2,3), (3)
Direct:   (0,1,3), (1,2),   (2,3), (3)
Carried:  (0,1,3), (1,2,3), (2,3), (3)
```

The direct tie leaves a crossing label unavailable on both adjacent tensors at two cuts. Carrying label 3 through tensor 1 restores the structural condition. **Carrying changes the ansatz and parameter count; it is not a gauge transformation of the direct-tie ansatz.** It also leaves the enlarged frontier width in place.

In the seed-731 dimer experiment:

| Layout | Allowed coefficients | Maximum frontier width | Largest supported metric condition number |
|---|---:|---:|---:|
| MPS, cap 9 | 40 | 0 | 1 |
| NN LETTA, cap 9 | 148 | 1 | 1 |
| NN plus direct `(0,3)` tie | 196 | 2 | about 2.08 million |
| Carried `(0,3)` tie | 388 | 2 | 1 |

These coefficients are symmetry-allowed tensor entries before removing gauge redundancy. Stored dense entries are also recorded separately. Condition numbers are measured after canonicalizing each center and discarding numerical metric-null eigenvalues at the stated audit tolerance; they depend on the state and seed.

For spatial dimension 4, a frontier of width one carries 4 norm configurations and 16 Hamiltonian bra/ket configurations. Width two raises these to 16 and 256, before virtual and MPO dimensions. The carried tie improves conditioning but does not remove this cost. Timings in the JSON are single local runs, not controlled performance benchmarks.

## Numerical findings

Lowdin orbitals, STO-3G, seed 731:

| System | FCI total energy (Eh) | MPS cap-9 error (mEh) | NN LETTA cap-9 error |
|---|---:|---:|---:|
| H2, 0.74 Angstrom | -1.137283834489 | numerical zero | numerical zero |
| H2, 2.0 Angstrom | -0.948641112176 | numerical zero | numerical zero |
| Linear H4, spacing 1.6 Angstrom | -1.967560309920 | 2.932375 | numerical zero |
| Two H2 fragments, grouped orbital order | -1.966945141325 | 0.000123 | numerical zero |
| Same fragments, order `(0,2,3,1)` | -1.966945141325 | 45.641377 | numerical zero |

Each fragment has bond length 1.6 Angstrom; their separation along x is 8 Angstrom. “Far apart” in the tie experiment means **chain positions**: it is the within-fragment correlated pair that was separated by the ordering. The weak correlation between fragments is not presented as a strongly correlated distant chemical pair.

For canonical H4 orbitals, the cap-9 MPS error changes from 91.768 mEh in
natural orbital-index order to 14.699 mEh in MI order `(1,2,0,3)`.
NN LETTA reaches FCI in both. This is an additional small-system ordering
control, not a guarantee that FCI-MI ordering will transfer to larger problems.
The separated-dimer NN/direct/carried results were repeated with seed 947;
all three again reached numerical FCI energies. Their residuals are recorded
separately, so close energies are not conflated with identical wavefunctions.

The H4 NN example has enough capacity to reach FCI already, so extra ties cannot demonstrate further accuracy gains here. Moreover, the cap-16 MPS also reaches FCI with **80 allowed coefficients**, compared with **148** for NN LETTA at cap 9. Thus these examples validate the implementation and expose ordering/gauge effects, but do **not** demonstrate a parameter advantage over MPS. Near-zero energy errors alone do not establish exact eigenvectors: the records also include residual norms, energy variances, FCI overlaps, and sector leakage.

An energy-stagnation flag is explicitly called `sweep_stagnation_converged`; it is not used as a certificate of ground-state accuracy.

## How to choose ordering and links next

1. Choose an orbital basis and estimate fermionic orbital mutual information from a modest pilot calculation. Basis localization and ordering are separate choices; Lowdin and canonical orbitals are both supported here.
2. Minimize the MI-weighted squared chain separation. The helper enumerates permutations for up to eight orbitals and uses a Fiedler ordering beyond that. Neither this objective nor the Fiedler heuristic guarantees the best LETTA energy.
3. Start with NN ties in that order. Inspect metric ranks, supported conditioning, energy residuals, and sector coverage.
4. Add the strongest remaining non-NN correlations subject to frontier-width and tensor-degree budgets. Evaluate the gauge condition from the actual dependencies. Compare direct and carried ties only while reporting their differing parameter counts and contraction costs.
5. Use a larger active space where NN LETTA is not already exact to test whether a long tie earns its cost. Compare both equal virtual dimensions and equal parameter/memory budgets, across multiple seeds and with a converged MPS reference.

The saved MI ordering uses FCI as an **oracle pilot** for these tiny examples. That isolates the ordering question; it is not a scalable prescription or a prediction from integrals alone. The MI helper accepts an arbitrary small pilot state and handles fermionic swap signs before tracing nonadjacent orbitals. Its convention is `I_ij = S_i + S_j - S_ij`, with natural logarithms and no factor of one half.

## Running

Use Python 3.12+ with NumPy, SciPy, opt_einsum, PySCF, and pytest for tests. From the new checkout:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.
python -m pyqed._letta_one_site_opt.benchmarks.qchem_ground_state \
  --cases h2_equilibrium h2_stretched h4_chain \
  --output /private/tmp/letta_nn.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_ground_state \
  --cases h4_dimers --orders natural scrambled correlation \
  --methods mps nn direct carried --bond-dim 9 \
  --output /private/tmp/letta_links.json
python -m pyqed._letta_one_site_opt.benchmarks.qchem_ground_state \
  --cases h4_chain h4_dimers --orders natural scrambled \
  --methods mps --bond-dim 16 --output /private/tmp/letta_mps_control.json
python -m pytest -q tests/test_letta_qchem.py tests/test_letta_frontier_gauge.py \
  tests/test_letta_general_ties.py tests/test_letta_symmetry.py
```

The session used `/Users/shuoyihu/miniforge3/bin/python` (Python 3.12). Existing pure-Python dependencies were copied to `/private/tmp/letta-qchem-test-deps`, so this machine's equivalent `PYTHONPATH` during validation was `.:/private/tmp/letta-qchem-test-deps`. No existing environment was modified. Dependency versions are recorded in each result.

For custom integrals:

```python
from pyqed._letta_one_site_opt import letta_dmrg, LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, initial_state, embed_ties
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods

problem = ElectronicProblem(h1, eri, nelec=(nalpha, nbeta), ecore=ecore)
problem = problem.reordered(order)  # reorder integrals and retain original labels
mps = initial_state(problem, max_bond_dim=16, seed=731)
state = embed_ties(mps, tie_neighborhoods(problem.norb, nearest=True))
result = letta_dmrg(problem.mpo(), state=state,
                   options=LETTADMROptions(gauge_mode='frontier', max_sweeps=30))
print(result.energy)
```

Use active-space effective one-body integrals and include frozen-core energy in `ecore` when supplying a frozen-core problem. Complex orbitals and spin-dependent integrals are intentionally rejected by this first adapter.

## Preservation of the original work

- Source checkout: `/Users/shuoyihu/Documents/GitHub/pyqed`, branch `bg`, starting commit `26addb2`.
- New checkout: `/Users/shuoyihu/Documents/ChatGPT/LETTA/pyqed-letta-qchem`, branch `codex/letta-qchem`.
- Inherited snapshot commit: `e0dff43`, containing all 112 copied changed/untracked files. A concurrent source edit to the cluster compression launcher was subsequently copied and preserved separately as `33535d8`; its hash is recorded in `concurrent_update.json`.
- Backup: `/Users/shuoyihu/Documents/ChatGPT/LETTA/qchem_preservation_20261002/manifest.json`, with SHA256 hashes, original staged/unstaged patches, stash object IDs, and ten stash archives (plus stashed untracked files where present).
- Stashes remain available through the shared repository. None was popped or dropped. The old `bg` CBE stash overlaps later implementations; it is preserved, not reapplied over newer code. Stashes from unrelated branches are also archived rather than merged into this implementation.
- The original checkout and index were not switched or rewritten. No branch was pushed.

## Validation completed

56 focused tests passed in 5.70 seconds. All 29 saved experiment records were
also checked for energy consistency, variational lower bounds within numerical
tolerance, fixed particle sectors, independent FCI actions, monotone sweeps,
and identity on the supported NN/carried metrics. These are numerical and
implementation checks; they do not establish large-system scaling or a
parameter advantage.

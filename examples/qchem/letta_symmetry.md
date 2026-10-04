# Symmetry in electronic LETTA: issue, alternatives, and implemented foundation

This document records the foundation at commit `2f92835`. The subsequent
[native SU(2) implementation record](letta_native_su2.md) supersedes its
component-contraction limitations and explains the reduced operator compiler,
native environments, and performance validation.

This work lives on `codex/letta-qchem`. It supports spin-independent, real,
orthonormal spatial-orbital Hamiltonians. Spin-orbit Hamiltonians require a
different symmetry choice. The SU(2) difficulty concerns the extra tied
physical labels, not ordinary spin-adapted MPS/DMRG.

## Why magnetic-label ties need additional constraints

The ordinary singlet is

$$
|0,0\rangle=(|\uparrow\downarrow\rangle-|\downarrow\uparrow\rangle)/\sqrt2.
$$

A tied tensor can independently change its two configuration amplitudes:

$$
a|\uparrow\downarrow\rangle-b|\downarrow\uparrow\rangle
=\frac{a+b}{\sqrt2}|0,0\rangle+\frac{a-b}{\sqrt2}|1,0\rangle.
$$

Every such state has the same electron number and $S_z=0$, but only $a=b$
gives a singlet. More generally, spin rotations mix magnetic components.
The basis-copy map $C|m\rangle=|m,m\rangle$ obeys

$$ C U(g)\ne [U(g)\otimes U(g)]C. $$

For a rotation taking $|\uparrow\rangle$ to
$(|\uparrow\rangle+|\downarrow\rangle)/\sqrt2$, the left expression gives
only $|\uparrow\uparrow\rangle$ and $|\downarrow\downarrow\rangle$,
whereas the right also gives the two opposite-spin configurations. This is
a tensor covariance issue; the copy node is not a physical cloning operation.

An ordinary SU(2) MPS uses Clebsch--Gordan coefficients to enforce relations
between magnetic components. Unrestricted dependence on a neighbor's magnetic
label can violate those relations. Spin-pure states certainly exist in the
original LETTA manifold. What is missing is an efficient local parametrization
of its entire spin-pure subset at fixed bond dimensions.

Abelian transformations are diagonal in occupation/spin-projection labels.
Owned-index charge conservation therefore suffices:

$$q_R=q_L+q(s_i).$$

Dependencies condition amplitudes but must not be counted as additional
particles. This rule applies equally to $N$ alone and $(N,2S_z)$.

## Alternatives and limits

| Construction | What it can do | Limitation |
|---|---|---|
| $N,S_z$ blocks with full physical-label ties | Retain the original magnetic-label LETTA flexibility | Does not fix $S^2$ |
| Reduced SU(2), invariant multiplet-label ties | Exact total spin; occupation-dependent ties; ordinary spin-adapted MPS inclusion | Pure spin-half sites have one invariant label, so these ties become trivial there |
| Global spin constraints on original LETTA tensors | In principle preserve original ansatz and target spin | Constraints are coupled and nonlinear across tensors; a local solver/gauge construction remains to be derived |
| Spin-coupled links/intertwiners | Add rotation-invariant spin-correlation channels | Changes network structure/cost; equivalence to original fixed-$D$ LETTA is unproved |
| Total-spin projection of a general state | Produce the desired spin sector, if the projected state is nonzero | Projected representation can require larger bonds and harder contractions |
| Penalty in $S^2$ | Bias optimization toward a desired spin; a positive $S^2$ penalty favors singlets | A finite penalty alone is not an exact spin-purity guarantee; targeting higher spin needs a suitable nonnegative penalty |

For a singlet in $M=0$, the exact constraint can be written
$S_+|\Psi\rangle=0$. It becomes a linear constraint in one core with other
cores fixed, but the feasible spaces change between updates. It is not yet a
replacement for the current unconstrained local generalized eigensolve.

A useful spin-link example, restricted to two singly occupied orbitals, is

$$L_{ij}=c_0P^{(0)}_{ij}+c_1P^{(1)}_{ij},\qquad
P^{(0)}_{ij}=\tfrac14-\mathbf S_i\cdot\mathbf S_j,\quad
P^{(1)}_{ij}=\tfrac34+\mathbf S_i\cdot\mathbf S_j.$$

This distinguishes pair singlets and triplets while commuting with global
rotations. Replacing a classical tie by this kind of operator network is a
proposed extension. It requires explicit spin channels, fusion trees, and
recoupling. The original shared-label gauge theorem cannot simply be assumed
for those nontrivial representation legs.

## Exact invariant-label construction

One spatial orbital decomposes as

$$\mathcal H_i=(0,0)\oplus(1,\tfrac12)\oplus(2,0),\qquad
\mathcal V_b=\bigoplus_{n,S}\mathbb C^{r_{b,nS}}\otimes V_S.$$

The three tied labels are empty, single, and double. The reduced core is

$$
A^{\eta_i m_i;\boldsymbol\eta_{P_i}}_{q_La_Lm_L,q_Ra_Rm_R}
=\delta_{n_R,n_L+n_i}
B^{\eta_i;\boldsymbol\eta_{P_i}}_{q_La_L,q_Ra_R}
C^{S_Rm_R}_{S_Lm_L,s_im_i}.
$$

Optimize $B$, retaining only allowed fusions. The right boundary is the target
$(N,S)$, and the left boundary is vacuum. Ties act on invariant multiplet
labels, so they cannot spoil this covariance. Fixed single occupancy leaves
only one conditioning label and therefore no additional tie flexibility.

For shared scalar frontiers $\xi$, the invariant Gram matrix obeys

$$G(\xi)=\bigoplus_q G_q(\xi)\otimes I_{2S_q+1}.$$

The new gauge whitens $G_q$ on its supported multiplicity space and absorbs
the inverse into the adjacent core. It retains an invertible unit transform
on numerically null directions. It does not truncate a physical state.
Every magnetic component receives the same transform. Right environments sum
the entire target multiplet, avoiding a fixed-$M$ non-invariant metric.

Packed reduced coordinates still carry Clebsch--Gordan norm weights. Both
one- and two-site solvers retain $H_{\rm eff}b=E N_{\rm eff}b$; whitening a
virtual environment does not justify replacing this reduced metric by $I$.
The public state normalization is one normalized target-$M$ component.

## Code reused and added

- `pyqed/mps/autompo`: graph compression of ordinary operator strings. Already
  used by `ElectronicProblem.mpo(backend='symbolic')`; it does not itself
  supply SU(2) fusion labels.
- `pyqed/mps/nonabelian/builder.py`: reduced AutoMPO, irreducible operators,
  and rank-coupled channels.
- `pyqed/mps/nonabelian/models.py`: `add_spatial_one_body_terms` and
  `SpatialSpinFreeERIBuilder`, including fermionic parity and repeated-site
  products. The chemistry adapter follows the existing DMRG builder's
  separate one-/two-body assembly and `sum_mpo_chains` combination.
- `pyqed/mps/nonabelian`: sector/fusion conventions, MPS tensors, decomposition
  machinery, and reference paths for future fully reduced environment work.

The new `ElectronicProblem.su2_mpo()` contracts the exact **local magnetic
component** operator representation against reduced LETTA states through
frontier environments. `ReducedMPOHamiltonian.factors is None` makes this
choice explicit. It neither builds a full determinant-space Hamiltonian nor
claims a verified native multiplicity-only chemistry contraction. Reusing
native DMRG reduced environments directly requires convention-by-convention
validation and an adjoint map for LETTA's repeated dependencies.

Added functionality:

1. Explicit `n`, `n_sz`, and `su2` chemistry symmetry selection, with target
   spin/electron validation. Existing `nalpha_nbeta` initialization remains.
2. Configurable invariant-label dependencies and exact reduced-MPS import,
   preserving sector multiplicities, scale, and every magnetic component.
   Metadata-less DMRG output needs explicit
   `physical_representation='fully_reduced_su2'`; expanded magnetic-site
   tensors are not silently converted.
3. Conditional SU(2) frontier gauges for shared-frontier graphs, integrated
   with one- and two-site sweeps. Direct distant links without shared
   frontiers require `gauge_mode='scalar'`; carried scalar links are eligible.
4. Optional `LETTATwoSiteOptions(reduced_sector_growth=True)` opens missing
   pair fusion sectors and multiplicity directions. New left columns are
   zero and new right rows are seeded, preserving the incumbent wavefunction.
   Refine both incumbent and projected-SVD starts in the physical norm;
   select after refinement and reject an energy-increasing result. Rejection
   restores the original allocation. The accepted split retains whole
   multiplets under the total multiplet cap.
5. Correct component normalization for non-singlets, full physical-axis
   layouts even for vacuum/polarized states, and more stable metric-nullspace
   rejection in the matrix-free eigensolver.

Growth is opt-in and uses temporary per-sector capacities that may exceed the
final cap. Rejection preserves an incumbent even if it already exceeds a newly
requested cap; inspect actual dimensions. One-site sweeps alone retain fixed
sector allocations. Charge-only LETTA storage remains masked dense storage;
fully block-sparse Abelian contractions and Abelian adaptive growth are not
part of this change.

## Usage and validation

```python
symmetry = problem.symmetry('su2', two_s=0)
state = ReducedLatticeLETTA.random((1, problem.norb), symmetry=symmetry, seed=7)
hamiltonian = problem.su2_mpo()
result = letta_two_site_dmrg(
    hamiltonian, state=state, bond_dim=16,
    options=LETTATwoSiteOptions(
        split_method='conditional-svd', reduced_sector_growth=True,
        gauge_mode='frontier', max_sweeps=8))
```

Here `bond_dim` counts complete multiplets:

$$D_{\rm multiplet}=\sum_q r_q,\qquad
D_{\rm magnetic}=\sum_q(2S_q+1)r_q.$$

Report `state.bond_dimensions` and `state.magnetic_bond_dimensions` together.
Comparisons require the same target spin, sector allocations at embedding,
and actual costs; equal printed caps are insufficient.

Run `examples/qchem/letta_symmetry.py --output /private/tmp/symmetry.json`
with `PYTHONPATH=.` and single-threaded BLAS. This is a small Hubbard-chain
consistency demonstration, not a molecular performance benchmark. It uses
PySCF only for independent full-CI references and reconstructs small vectors
only for post-run verification. The archived output is
`letta_symmetry_results.json`.

Regression tests cover independent determinant matrix elements through four
orbitals, number/spin commutators, singlet/doublet/triplet residuals, all-M
embedding, complex conditional gauges, missing-sector discovery, rejected
growth rollback, and production sweeps with full-state reconstruction disabled.
The [native follow-up](letta_native_su2.md) implements reduced contractions and
records molecular validation, matched-threshold timing, adaptive MPS controls,
failed trials and fixes, and the final 132-test regression. The spin-channel
tie extension remains separate work.

Verification on 2026-10-02: **93 tests passed** across
`test_letta_qchem{,_active_space,_symmetry}.py`,
`test_letta_reduced_{state,symmetry,frontier,mpo,one_site,two_site,gauge}.py`.
The existing NumPy deprecation warnings remain. In the archived four-orbital
Hubbard example, eight retained multiplets (14 magnetic states at the center)
give an untied energy of -1.9527643400754622 and an NN-tied energy of
-1.9527751135647136, against FCI -1.9531453086845492. The embedding distance
is zero at reported precision; the spin residual is $3.06\times10^{-16}$.

## References

- [LETTA: tied-index ansatz and conditional gauges](https://arxiv.org/html/2609.30101v1).
- [Singh and Vidal: invariant tensors and SU(2) structure](https://arxiv.org/abs/1208.3919).
- [Sharma and Chan: spin-adapted quantum chemistry DMRG](https://arxiv.org/abs/1408.5039).

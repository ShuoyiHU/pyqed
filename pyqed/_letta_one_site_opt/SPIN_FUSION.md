# SU(2) spin-fusion conditioning through pair blocking

`fuse_su2_pairs` adds a **rung-blocked** spin-sensitive LETTA option. It does
not change the default orbital ansatz. It does not copy magnetic components,
project a many-body wavefunction to a spin sector, or introduce arbitrary
nonlocal/overlapping fusion ties.

## Construction and fidelity

The standard SU(2) coupling and structural/degeneracy decomposition follows
S. Singh and G. Vidal, *Tensor network states and algorithms in the presence
of a global SU(2) symmetry*, Phys. Rev. B **86**, 195114 (2012),
<https://doi.org/10.1103/PhysRevB.86.195114>.
Using the resulting pair labels as LETTA dependencies is our adaptation,
not a reproduction of an algorithm proposed in that paper.

For adjacent physical sites, the local unitary is

$$U_{(a,m_a;b,m_b),(q,c,M)}=C^{JM}_{j_a m_a,j_b m_b}.$$

The channel c records the input irreps and multiplicity copies. No pair
states are discarded. Two Hubbard orbitals give 16 magnetic states in 10
multiplets: (N,2S,multiplicity) = (0,0,1), (1,1,2), (2,0,3), (2,2,1),
(3,1,2), (4,0,1). In particular, two singly occupied orbitals have distinct
singlet and triplet labels. Neighboring tensors may condition on them.

The original left-associated spin tree is changed using the overlap of
Clebsch–Gordan trees. For fixed total magnetic component, the overlap is
independent of that component; the implementation averages over all of them.
Reduced tensor products are multiplied by these recoupling coefficients.
The operation retains the original inter-pair bond sectors exactly.

Only the **local** Hamiltonian pair core is transformed as U† W_pair U.
The existing reduced-MPO compiler and reduced environments perform subsequent
optimization. No global 4^n matrix, determinant projection, or magnetic-state
reconstruction is used in this path. Small dense reconstructions in tests
are independent validation helpers only.

## Usage

Start from an untied ReducedLatticeLETTA MPS and ReducedMPOHamiltonian in
rung-major order (0,1), (2,3), ...:

```python
from pyqed._letta_one_site_opt import fuse_su2_pairs, letta_dmrg, LETTADMROptions

rungs = state.nsites // 2
neighbors = tuple((i, i + 1) if i + 1 < rungs else (i,) for i in range(rungs))
spin_state, spin_ham = fuse_su2_pairs(state, hamiltonian, neighborhoods=neighbors)
run = letta_dmrg(spin_ham, state=spin_state, options=LETTADMROptions(
    max_sweeps=100, tolerance=2e-8, eigensolver_tolerance=1e-10,
    matrix_free=True, gauge_mode='frontier'))
```

Omit `neighborhoods` for an untied pair-blocked MPS control. Both conversions
represent exactly the same starting physical state. Do not compare only
against orbital MPS: removing the intra-rung cut restriction already changes
the variational family. D now bounds **inter-rung** multiplets. There are more
local parameters with ties; this is not an equal-parameter comparison.

The driver's per-site energy tolerance is per *blocked* site. Multiply the
requested per-orbital tolerance by two. Sweep counts are over blocked sites.

## Restrictions and checks

- Even chains of at least four sites, identical local physical bases, open
  boundary, adjacent non-overlapping pairs; arbitrary reordering is not done.
- Untied input only. Orbital LETTA ties cannot silently be dropped.
- Exact total spin and particle number are preserved. S_z labels remain
  structural, not independent variational conditioning coordinates.
- No claim of global optimization convergence or improved cost is made.
- The one-site path is exercised by the Hubbard benchmark. Compatibility of
  every CBE selector with this enlarged local basis is not yet established.
- Tests verify complex and mixed-dtype state equivalence, Hamiltonian/energy
  equivalence, local reduced execution without global state reconstruction,
  and nonredundant spin-only expressivity: for eight spins, ties increase a
  central Schmidt rank from four to eight at fixed two backbone multiplets,
  while total S+ still annihilates the state.

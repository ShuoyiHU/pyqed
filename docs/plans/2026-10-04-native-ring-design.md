# Native closed-ring symmetry contractions: design and verified foundation

Native cyclic norm/Hamiltonian contractions, local actions and an explicit target closure are now implemented and reference-tested. **Full closed-ring LETTA optimization is not yet implemented.** The goal still requires QC/condensed models, U(1)/SU(2), arbitrary physical ties, all three update methods, all four compressors, and a clear public API. A periodic Hamiltonian or a last–first physical tie on an open virtual chain is not a closed virtual ring.

## First gate: cyclic reduced norm

An open-chain invariant norm environment carries only the scalar channel selected by its unit boundary. A closed virtual ring needs all allowed total-spin channels of the bra/ket virtual transfer, including nonzero channels. Tracing the existing scalar open environment would omit them.

For virtual ket and bra irreps, form a structural coupled basis of ket times dual bra:

$$
U^{J M}_{m_b m_k}=(-1)^{j_b-m_b}
 C^{J M}_{j_k m_k,j_b,-m_b}.
$$

For the local double-layer spin transfer, compute its reduced scalar block in each J channel by projecting with these **fixed structural** CG arrays and averaging over M. Variational arrays remain reduced multiplicity blocks; do not expand a variational magnetic MPS or global determinant vector. Contraction of the reduced bra/ket multiplicities then constructs a transfer matrix T_i^(J). The cyclic norm is

$$
\langle\Psi|\Psi\rangle=\sum_J(2J+1)
\operatorname{Tr}\left[T_0^{(J)}\cdots T_{L-1}^{(J)}\right].
$$

This formula is to be validated before use, including dual phases, orientation, multiplicity layout, charge compatibility and channel dimensions. Start with pure-spin singlet rings and independently expand small reference tensors only in tests. Include cases where omitting J>0 produces an incorrect norm. Test arbitrary complex virtual gauges, nonuniform bond allocations and zero states.

## Hamiltonian channels

For a scalar operator MPO, each cut has operator, ket and dual-bra virtual representations. A candidate basis couples ket and operator to an intermediate irrep, then couples that with dual bra to a total transfer J. The intermediate fusion channel is part of the multiplicity label; retaining J alone is insufficient. Local recoupling coefficients can be evaluated/cached using small structural CG contractions, just as the existing native open transfer does. Product-group charge labels must all be retained.

The operator MPO can retain open unit endpoints while the variational bra/ket bonds close. Validate cyclic H against independent local-spin and fermionic references before implementing local energy solves. Do not replace the missing channels with a projected determinant Hamiltonian.

## Targets and actual ring topology

A trace of symmetry-invariant tensors with unshifted physical charges is a scalar of zero net charge. Nonzero particle number and non-singlet spin require an explicit construction; the inherited periodic benchmark's unit-filling mask is not general support.

Investigate a covariant target closure carrying the dual target representation, with a separate structural closure vertex/target leg. It must preserve an actual virtual loop and expose the intended physical target component. Specify which closure multiplicities are variational and how wrap updates cross that vertex. Do not silently turn the ring into an open chain or constrain odd-electron systems to singlets. Background charge shifts may simplify singlet/Abelian cases but must not be substituted for general target closure.

Arbitrary physical dependencies can be routed by frontier memory while the base virtual bond still closes; this is distinct from making that base bond one dimensional. Verify the local embedding and wrap pair map independently. Target closure and local physical bases may require heterogeneous structural metadata; do not force a mismatched target leg through a uniform physical basis.

## Update and compression gates

1. Native cyclic H/N and local actions must agree with small independent references, including wrap bonds.
2. Keep the complete nonidentity metric. Marginal gauges are preconditioners, not proofs of N=I.
3. Reuse ordinary one-site, residual-based CBE and two-site protocols with transactional recovery and strict same-start baseline acceptance.
4. Cyclic pair metrics generally do not factor into independent left/right boundary Grams. The current OBC ReducedPairMetricRoot is not valid for that case. Build a correct cyclic metric adapter; dense local-parameter matrices are permissible for initial accuracy references, but global determinant-space solver projections are not.
5. Validate all four compressors on the same physical metric, plus exact zero-padding and charge/multiplet preservation at the wrap bond.
6. Test both target singlets and non-singlets, charge/spin sectors, arbitrary ties, QC and condensed models before claiming the PBC matrix complete.

Numerical gauge failure in a sweep is a further recovery gate: the low-level gauge now restores its tensors and raises. Full sweep-level fallback/continuation must explicitly handle that exception rather than assume atomic restoration alone completes recovery.


## Verified foundation, 2026-10-05

`reduced_ring_contraction.py` implements all transfer-spin channels for norms and Hamiltonians, including the ket/operator intermediate fusion multiplicity. `local_action` contracts the complement of one core and returns the full nonidentity H/N action in reduced source coordinates. Complex cross overlaps, unequal bra/ket multiplicities, cyclic rotations, invertible virtual gauges and local Hermitian/positive-semidefinite metrics agree with independent small magnetic reference states. Magnetic variational expansion is disabled during tested production contractions. This is exact cyclic contraction, without the optional transfer-product SVD approximation.

`reduced_ring_target.py` supplies a covariant closure core with axes `(last bond, dual target, first bond)`. Its dual-target physical leg has one copy of the requested representation; neither adjacent virtual bond is restricted to dimension one. The operator is extended with an identity on this auxiliary leg. It is not an extra electron/orbital. Site-specific MPO operator bases avoid assuming that this target representation has the same dimension as a physical site.

Signed generic `Sector` labels are necessary for the negative dual U(1) charge. The older `SpinChargeSector` remains a nonnegative particle-count type. `signed_sector` and `signed_physical_basis` preserve charges, irreps, multiplicities and basis order while changing the label representation. All additive charge factors are dualized; XOR point-group labels and SU(2) irreps are self-dual.

For physical target spin S, the invariant scalar state can be written

$$
|\chi\rangle=\frac{1}{\sqrt{2S+1}}\sum_{M=-S}^{S}(-1)^{S-M}
|\psi_M\rangle\otimes|\overline{Q},S,-M\rangle.
$$

Thus an unrescaled auxiliary slice has norm squared `norm_squared()/(2S+1)`, returned by `component_norm_squared()`. Multiplying that slice by its CG phase and sqrt(2S+1) gives the physical target component. A scalar H has the same Rayleigh quotient in every component and in the full invariant contraction. These conventions must remain explicit when adding state normalization and optimization; the open-chain boundary normalization rule must not be copied blindly.

Independent checks cover N=1/2/3 electronic targets, singlet/doublet/triplet spin, a nontrivial spin-half anchor circulating around the virtual ring, U(1) Bose-Hubbard, and independent Nalpha/Nbeta charges. Physical charge, Sz and S² are checked on every target component. Molecular expectations and local H/N at both a physical site and the target closure agree with independently assembled determinant operators. Dense states/operators occur only in tests.

### Independent builder error uncovered by these checks

Periodic Heisenberg had different family labels for the ordinary and wrap bonds. AutoMPO shared their prefix state while retaining distinct additive opening transitions, duplicating the opening operator. Prefix sharing now includes family identity. A fully reduced exchange subtraction had compensated this same duplication; it is removed, with the existing reduced-vs-independent reference tests retained and extended to a combined exchange operator. Dense and reduced shared-prefix tests cover distinct/identical/missing family labels and leading identity sites.

### Still required

- Conditional ring state storage and arbitrary-tie embeddings, including the physical last–first tie, without replacing the virtual loop by unit boundaries.
- One-site sweep integration, explicit closure update policy and symmetry-preserving conditioning with full N retained.
- Pair response maps, cyclic physical metric roots and all four compressors; the OBC product of boundary Grams is not valid here.
- Actual one-site residual CBE and two-site update protocols on ring edges, including edges incident to the covariant closure, with clearly documented semantics for last–first physical pair updates through that closure.
- Transactional recovery at sweep-level gauge failures, convergence diagnostics, public API and full combination tests.
- Long-chain numerical scaling of cyclic transfer products. Current checks establish small-system accuracy, not overflow-safe long-chain execution.

The overall goal remains incomplete. This design/checkpoint note is not the final full implementation note requested by the user.

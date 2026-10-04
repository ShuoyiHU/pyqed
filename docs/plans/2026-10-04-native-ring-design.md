# Native closed-ring symmetry contractions: proposed next implementation

This is a derivation and implementation plan, **not implemented support**. The goal still requires QC/condensed models, U(1)/SU(2), arbitrary physical ties, all three update methods, all four compressors, and a clear public API. A periodic Hamiltonian or a last–first physical tie on an open virtual chain is not a closed virtual ring.

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

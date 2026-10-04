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


## One-site ring integration, 2026-10-05

`ReducedRingLETTA` now stores conditional physical cores with L+1 virtual allocations and a separate covariant target core. It reuses the exact `ReducedFrontier` sparse copy map without imposing open unit virtual endpoints. A seeded random constructor keeps all fusion-compatible sectors for a user-selected anchor irrep/copy count, and import from a verified reduced target ring broadcasts exact ties without changing the state. Signed target labels are preserved.

`ring_local_problem` composes that sparse map with native cyclic H/N and their source adjoints. `letta_dmrg` explicitly dispatches this state type to a ring one-site sweep. All physical cores and the target closure are optimized. The shared generalized eigensolver supports dense local matrices and matrix-free actions. No production magnetic wavefunction or determinant-space frame is constructed.

Every ring edge, including both closure edges, supports invertible sector-multiplicity conditioning or scalar balancing. These are unconditional legal gauges for arbitrary ties; the full local metric remains explicit. Transactional local and gauge failure paths restore the relevant incumbent and do not report convergence. Fresh local residuals are audited at energy plateaus. Numerical scaling of much longer cyclic products and broader target/allocation tests remain open gates.

Next implementation priority: native cyclic pair-response coordinates and a valid metric root for all four compressors, followed by true expanded-one-site CBE and two-site integration. The covariant closure is a variational core, so update policies for its two adjacent graph edges and a direct last–first physical pair through the closure must be kept explicit. Open-chain sweep-level gauge recovery is also still pending.


## Cyclic pair metric and compression foundation, 2026-10-05

`CyclicPairProblem` stores untruncated coefficients with the fusion-path key

$$
(q_L,p_1,q_M,p_2,q_R)
$$

and multiplicity shape `(d_L, multiplicity(p1), multiplicity(p2), d_R)`. The structural spin tensor is the product of the two sequential Clebsch–Gordan tensors, summed over the intermediate magnetic component. The coefficient layout includes every allowed intermediate irrep, including sectors missing from the incumbent middle bond. It does not expand variational magnetic coefficients.

The norm contracts these two-site structural tensors with the coupled bra/ket bases at the OUTER cuts. The Hamiltonian additionally keeps the MPO's intermediate spin and the ket/operator fusion multiplicity at both outer cuts. The complement is the cyclic product of every remaining transfer. Its channel intersection excludes the removed internal cut: limiting transfer J by the incumbent middle bond would incorrectly remove directions that an expanded pair is meant to discover. A regression starts with only a scalar internal-spin bond, retains nontrivial outer transfer channels, and recovers the exact Hubbard dimer pair energy.

The resulting pair overlap is the full correlated cyclic Gram N. `CyclicPairMetricRoot` materializes this LOCAL reduced coefficient-space matrix and equilibrates its diagonal scales before deciding support. If D denotes those coordinate scales and the supported correlation factorization is C=V Λ V†, the maps are

$$
S=\sqrt{\Lambda}\,V^\dagger D,\qquad
W=D^{-1}V\Lambda^{-1/2}.
$$

On retained support, S†S=N and SW=I. The generalized inverse action is WW†, satisfying N WW† N=N on that support; no Euclidean Moore–Penrose claim is made after equilibration. The inverse supports metric residual operations needed by CBE. The construction has quadratic LOCAL matrix storage and cubic dense factorization cost; a conservative workspace estimate is checked before matrix construction and again before factorization. It is an accuracy backend, not a completed large-ring scalability solution.

`compress_ring_pair` composes exact physical-tie copy maps with the bilinear merge, provides analytic source adjoints, and supplies admissible `(middle sector, shared invariant labels)` gauge blocks. It uses the same `fit_metric_factors` implementation as open reduced compression: ALS, variable projection, joint LS and Grassmann Newton therefore have identical iteration-budget, stopping and best-iterate semantics. A retained rank counts complete middle-sector multiplets. The supplied target is fitted in N, not in unweighted coefficient norm or a fictitious pair of independent boundary Grams.

Independent tests cover all graph edges on a two-physical-site-plus-target ring, both U(1) components, singlet/doublet targets, missing internal spin sectors, exact source adjoints, arbitrary bidirectional ties, complex legal gauges, diagonal scale ranges of sixteen orders of magnitude, explicit ALS/LSMR budgets and all four solvers. A separate four-physical-site test verifies nonzero nonlinear chart dimension, genuine rank reduction, a full-rank nonseparable cyclic Gram and the fitted error in an independent physical frame. Global frames occur only in reference tests.

This foundation does NOT yet change the public ring CBE/two-site availability. Next work is allocation growth/shrinkage, multistart initialization, supplied-target compression plus A/B energy alternation and strict same-start ordinary one-site comparison, then actual two-site and residual-CBE sweeps. The pair graph includes last-physical/closure and closure/first-physical edges; a direct last–first physical update through the closure is a distinct operation and must not be silently conflated with these edges. Numerical/resource failures must restore the incumbent and use the ordinary local fallback. The complete feature matrix, long-chain scaling, API and final implementation note remain required.


## Ring allocation and two-site integration, 2026-10-05

Whole-sector allocation changes now include both closure-adjacent bonds. Growth preserves the original amplitudes by zero left columns and complementary seeded right rows; it opens locally reachable sectors with capacities limited by source factor dimensions. Installing compressed factors constructs and validates a new state before exposure, including the closure's virtual allocation metadata. Independent amplitude and pair-map checks cover every edge and missing internal spin sectors.

`fit_ring_pair_target` accepts an externally supplied pair vector without a pair eigensolve, compares refined factor starts in the complete cyclic metric, installs complete retained multiplets, and optionally alternates full-metric local energy solves on the two graph vertices. `optimize_ring_pair` compares this candidate against an ordinary one-site update from the same incumbent. Exceptions including resource limits recover through that ordinary baseline; a baseline that exceeds a requested smaller cap is not treated as feasible compression.

The public two-site dispatcher now accepts `ReducedRingLETTA` and sweeps all L+1 graph edges. Per-update gauge failure restores the post-update state and prevents convergence reporting. Small energy changes require successful compression diagnostics, feasible bond caps and freshly rebuilt local residuals. The SVD discarded statistic is explicitly a coefficient-space initialization diagnostic, not a cyclic physical norm bound.

Ring residual-CBE, long-chain cyclic scaling, the broader feature-matrix verification, OBC sweep-level gauge recovery, unified user entry points and the final full implementation note remain incomplete. No performance phase has begun.


## Ring CBE implementation, 2026-10-05

The residual projection and greedy whole-sector allocation loop is shared with OBC CBE through `select_metric_cbe`. Adapters supply exact pair actions, the topology-specific metric root, source-factor forward/adjoint maps, and the common compressor. The existing OBC selector passed all 18 tests immediately after extraction.

`select_ring_cbe` supplies `CyclicPairMetricRoot` and analytic conditional source adjoints. It forms W†(Hx-E Nx), projects out S[M(da,b)+M(a,db)] restricted to incumbent multiplets using implicit LSMR, and fits the supported missing residual with whole-sector source factors. Tests compare this projection with an independent physical frame, including a nonzero residual in a nonseparable cyclic metric. The root materializes a local Gram and diagnostics disclose that fact.

`expand_ring_cbe` appends the selected complementary factor and a zero active partner, in both sweep directions and through the covariant closure. The resulting state is coefficient-exact before the active one-site solve. `ring_cbe_site` first computes an ordinary baseline, then selects and optimizes on copies, supplies the expanded one-site result to the existing cyclic fitter, and accepts only strict improvement over the baseline under the retained cap. Selection, expanded-solve and trimming errors recover to that baseline; a failed baseline leaves the incumbent unchanged. No pair eigensolver is called in this protocol, including optional A/B energy refinement.

Public `letta_dmrg` ring dispatch now permits exact CBE. Its fixed cap is the maximum initial ring multiplet allocation. Every physical core and the target closure is visited in both directions. Shrewd/preselection modes, nonzero baseline allowance and nonmetric trimming are explicitly rejected. Incomplete factor fits and numerical recovery prevent convergence reporting. The complete support matrix, large cyclic scaling, API consolidation, OBC sweep-gauge recovery and final implementation note remain outstanding.

Fast validation: 19 initial ring CBE tests passed, including a missing-sector Hubbard case where ordinary one-site stays at E=0 and CBE reaches exact E=-2, all four compressors, independent residual projection, exact expansion, U(1) Bose-Hubbard and recovery. The combined run now passes 116 tests with one molecular backward-tie test deselected because that exact test is already running separately. The original separate run subsequently completed: 44 passed in 1181.73 s, including the complex three-orbital molecular backward-tie case. No restart or weaker replacement was made. Commands and outputs are in 2026-10-05-letta-ring-cbe-tests.txt; test counts overlap.

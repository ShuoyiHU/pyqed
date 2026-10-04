# LETTA symmetry implementation: derivations and code conventions

This companion to [the implementation guide](letta_symmetry.md) describes the
native implementation on `letta_oct_4_sym`. It is a code-grounded mathematical
note, not a claim that the outstanding model/topology validation has finished.
The [completion matrix](plans/2026-10-04-letta-symmetry.md) remains authoritative
for validation status. Equations below first describe exact arithmetic; numerical
support cutoffs and recovery are stated separately rather than hidden in equalities.

## 1. What is a tensor coordinate?

A physical site belongs to exactly one core. Let its owned physical label be
$s_i$, and let $\mathcal D_i$ be the other physical labels on which that core
may depend. An ordinary LETTA amplitude on an open virtual chain is

$$
\psi(s_0,\ldots,s_{L-1})=
\sum_{a_1,\ldots,a_{L-1}}
\prod_{i=0}^{L-1} A_i(a_i,s_i,s_{\mathcal D_i},a_{i+1}),
\qquad a_0=a_L=1.
$$

A tied occurrence of $s_j$ reads the same configuration variable. It is not a
second physical particle or a second copy of the site's Hilbert space.
Therefore Abelian conservation at core $i$ is

$$
q_{i+1}=q_i+q(s_i),
$$

with no additional charge contribution from $s_{\mathcal D_i}$. The equation
holds componentwise for products of U(1), such as $(N_\alpha,N_\beta)$.

For SU(2), a virtual space is a direct sum of multiplicity spaces and irreps:

$$
\mathcal V_i=\bigoplus_q \mathbb C^{r_{i,q}}\otimes V_{S_q},
\qquad q=(N_q,S_q),\qquad d_q=2S_q+1.
$$

Here $a=1,\ldots,r_{i,q}$ identifies a copy of an irrep. The magnetic number
$m=-S_q,\ldots,S_q$ is a different index. Independent variational data are
reduced blocks $B$, related to magnetic tensors by

$$
A^{(q_l,a,m_l),(q_p,p,m_p),(q_r,b,m_r)}_i(\xi)
=B^{q_lq_pq_r}_{i;a p b}(\xi)
\langle S_lm_l,S_pm_p\mid S_rm_r\rangle.
$$

The block exists only if charges add and $S_r$ occurs in $S_l\otimes S_p$.
The symbol $p$ denotes a physical multiplicity copy; $\xi$ is the tuple of
invariant conditioning labels from tied sites. Neither is a magnetic number.
A bond cap counts $\sum_q r_{i,q}$ multiplets. Its magnetic dimension is
$\sum_q r_{i,q}d_q$. These are not interchangeable definitions of $D$.

For a spatial orbital the physical irreps are $(0,0)$, $(1,1/2)$ and $(2,0)$,
labelled empty, single and double. Both spin components belong to the single
multiplet. The implemented ties copy these invariant labels. A spin-half site
with one irrep and one copy has only one possible tie label. Consequently such
ties do not add spin dependence to a pure spin-half MPS. Spin-carrying links
would define a different ansatz and require additional intertwiners.

`AbelianReducedMap` uses the same storage with trivial spin irreps. Repeated
physical charges become multiplicity copies; the original physical ordering
is retained through an explicit permutation. This conversion does not impose
SU(2) spin symmetry on a U(1) problem.

Source: `reduced_symmetry.py`, `reduced_state.py`, `abelian_backend.py` under
`pyqed/_letta_one_site_opt/`.

## 2. Arbitrary ties and the sparse frontier map

Sequential contraction must remember a physical label from its first to its
last occurrence. `ReducedFrontier` appends those remembered assignments to
virtual multiplicity coordinates. Remembered labels carry no additional charge
or magnetic index: the owned physical leg already carries that representation.

For a fixed layout let $b_i$ be the packed source core and $P_i$ its sparse
embedding into a frontier MPS core. Then

$$
\widetilde b_i=P_i b_i.
$$

One source entry may appear in more than one frontier entry. Thus the forward
operation copies entries, whereas its adjoint sums them. In particular,
$P_i^\dagger P_i$ need not be identity. If the frontier environment gives local
operators $\widetilde H_i$ and $\widetilde N_i$, the source-coordinate operators
are obtained by substituting $\widetilde b_i=P_i b_i$ into the quadratic forms:

$$
b_i^\dagger H_i b_i
=b_i^\dagger P_i^\dagger\widetilde H_iP_i b_i,
\qquad
b_i^\dagger N_i b_i
=b_i^\dagger P_i^\dagger\widetilde N_iP_i b_i.
$$

Hence $H_i=P_i^\dagger\widetilde H_iP_i$ and
$N_i=P_i^\dagger\widetilde N_iP_i$. The same adjoints must appear in factor
least squares and CBE tangent projections. Omitting the sum over copied
coordinates changes both the metric and the gradient.

Forward, backward, crossing and last-first physical ties all use this map.
The number of remembered assignments can grow exponentially with frontier
width. Exact arbitrary tying is therefore a representation capability, not
a promise of inexpensive contraction for every dependency graph.

Source: `FrontierSiteEmbedding.apply/adjoint` and `ReducedFrontier` in
`reduced_frontier.py`.

## 3. Why SU(2) norm contractions have dimension factors

An invariant open-chain norm environment in sector $q$ has magnetic structure
$G(q)\otimes I_{d_q}$. The reduced Gram $G$ acts only on multiplicity indices.
For a left-coupled CG tensor, orthogonality gives

$$
\sum_{m_l,m_p}
\overline{C^{S_rm_r}_{S_lm_l,S_pm_p}}
C^{S_rm'_r}_{S_lm_l,S_pm_p}=\delta_{m_rm'_r}.
$$

Therefore a left environment advances without an extra dimension weight:

$$
G^{L,\mathrm{new}}_{rs}(q_r)=
\sum_{q_l,q_p,p,a,l}
\overline{B^{q_lq_pq_r}_{a p r}}
G^L_{al}(q_l)B^{q_lq_pq_r}_{l p s}.
$$

For a right contraction, instead sum over the physical and right magnetic
indices. By invariance the answer is proportional to identity on $V_{S_l}$.
Its trace sums the squared CG coefficients to $d_{q_r}$, so the proportionality
constant is $d_{q_r}/d_{q_l}$. Consequently

$$
G^{R,\mathrm{new}}_{al}(q_l)=
\sum_{q_p,q_r,p,r,s}\frac{d_{q_r}}{d_{q_l}}
\overline{B^{q_lq_pq_r}_{a p r}}
G^R_{rs}(q_r)B^{q_lq_pq_r}_{l p s}.
$$

A local quadratic form closes all three magnetic legs and sums the squared CG
coefficients to $d_{q_r}$. Its Euclidean-adjoint metric action is therefore

$$
(N_i B)^{q_lq_pq_r}_{a p b}
=d_{q_r}\sum_{l,r}G^L_{al}(q_l)G^R_{br}(q_r)
B^{q_lq_pq_r}_{l p r}.
$$

The right-boundary recursion and the local action are different operations;
using the recursion's dimension ratio for the local action would be wrong.
The invariant open-chain contraction sums over the entire target multiplet.
The public open-chain norm divides that sum by $d_{\mathrm{target}}$ to report
one physical component's norm. Hamiltonian/norm ratios use matching conventions.

For an open-chain merged pair, the analogous action carries the outer-right
factor $d_{q_r}$ and the two outer boundary Grams. Factor their supported parts
as $L^\dagger L=G^L$ and $R^\dagger R=G^R$. For each pair block $X$, the map

$$
S(X)=\sqrt{d_{q_r}}\,LXR^T
$$

(with the two physical indices left untouched) satisfies
$\|S(X)\|_F^2=\langle X,NX\rangle$. Implementation log scales multiply this
map by the square root of the accumulated boundary scale. Conditional source
factors still enter through their sparse embeddings; this boundary factorization
does not turn their general compression problem into an unconstrained SVD.

Source: `advance_norm_left`, `advance_norm_right`, `ReducedNormChain` in
`reduced_norm.py`; `ReducedPairMetricRoot` in
`pyqed/_letta_two_site_opt/reduced_compression.py`.

## 4. Hamiltonian reduction and its distinction from a many-body projection

The molecular adapter first assembles a component MPO from local one-/two-body
operators. A local operator between physical spin irreps is expanded in
irreducible tensor operators

$$
T^{k\mu}_m[m_o,m_i]
=(-1)^{S_i-m_i}
\langle S_om_o,S_i,-m_i\mid km\rangle.
$$

The label $\mu$ distinguishes physical input/output irreps and multiplicity
copies. Its charge is $N_o-N_i$, which can be negative. This is why operator
sectors must support signed additive charges.

`SpinTensorMPO.compile` changes each local operator basis, performs local QR
conditioning and sector-resolved factorizations, and requires a scalar final
operator boundary. It checks reconstruction rather than silently discarding
material charge- or spin-breaking terms. The compilation tolerance is explicit;
its local reconstruction diagnostic is not a rigorous global error bound.
Independent small-system operator tests supply the end-to-end check.

The open Hamiltonian environment has structural basis

$$
\mathcal B_{m_wm_bm_k}
=\langle S_km_k,J_wm_w\mid S_bm_b\rangle,
\qquad \|\mathcal B\|_F^2=d_{q_b}.
$$

Contracting this basis with the bra CG tensor, ket CG tensor, operator CG
tensor and local tensor-operator basis gives the cached `spin_transfer`
coefficient. Advancing a boundary divides by the outgoing structural basis's
squared norm. A local quadratic-form action uses the unnormalized coefficient.
Only these small, state-independent structural sums contain magnetic indices;
variational tensors and environments remain multiplicity arrays.

This is not multiplication by a $4^L$ determinant-space projector. Such global
matrices and reconstructed state vectors are used only by explicit independent
reference tests. The production native paths use local MPO data and reduced
virtual environments.

Source: `qchem.py`, `reduced_mpo_compile.py`, `reduced_environment.py`.

## 5. Local optimization and metric support

Freeze every core except the one being optimized. Let $F_i$ denote the linear
map from that core's source coordinates to the complete physical state. This
is a mathematical definition, not a request to construct its dense matrix.
Then

$$
|\psi(b_i)\rangle=F_i b_i,
\quad N_i=F_i^\dagger F_i,
\quad H_i=F_i^\dagger\widehat H F_i,
\quad E(b_i)=\frac{b_i^\dagger H_i b_i}{b_i^\dagger N_i b_i}.
$$

Stationarity of this quotient at nonzero norm gives
$H_i b_i=E N_i b_i$. The code solves for the lowest supported generalized
root, starting from the incumbent where applicable. Gauge-redundant null
coordinates must be excluded or controlled; their Euclidean magnitude says
nothing about physical state amplitude.

The equilibrated metric routine first restricts to positive diagonal entries.
On that index set define

$$
D_{aa}=\sqrt{N_{aa}},\qquad C=D^{-1}ND^{-1},\qquad
C=U\Lambda U^\dagger.
$$

Apply the relative support threshold to eigenvalues of $C$, not raw diagonal
scales of $N$. With retained $U_r,\Lambda_r$, define

$$
S=\Lambda_r^{1/2}U_r^\dagger D,
\qquad W=D^{-1}U_r\Lambda_r^{-1/2}.
$$

Direct multiplication gives $SW=I$. If only exact null directions are removed,
$S^\dagger S=N$ and $W^\dagger NW=I$. With a finite support threshold these
identities refer to the supported metric $N_{\mathrm{supp}}=S^\dagger S$;
they are not an assertion that a discarded small positive eigenvalue is zero.
The reduced eigenproblem is $W^\dagger H_iW y=E y$, with $b_i=Wy$.
The supported inverse $WW^\dagger$ need not be the Euclidean Moore–Penrose
inverse of the original unequilibrated Gram.

The dense local eigensolver uses this equilibrated factorization. The
matrix-free solver obtains only the diagonal using native norm actions and
runs Davidson in $u=Db$ coordinates with operators

$$
H'=D^{-1}H_iD^{-1},\qquad N'=D^{-1}N_iD^{-1}.
$$

It never materializes the complete Gram for this equilibration. On positive
diagonal coordinates, $N'$ has unit diagonal, so its trace supplies a PSD
spectral upper bound for numerical dependence tests. Native open-chain
conditional support projectors are constructed from the normalized boundary
Grams and act on these equilibrated coordinates. A raw-coordinate projector
cannot be reused unchanged after this transformation. Generic correlated
layouts retain the iterative metric-orthogonality and dependence checks.


A small eigenvector residual alone does not identify the lowest eigenvalue.
For example, with $N'=I$, an initial coordinate eigenvector of a diagonal
$H'$ can have zero residual even when a different coordinate has lower energy.
The Krylov sequence from that initial vector stays in its invariant eigenspace.
Testing only a prefix of coordinate vectors does not resolve this problem.

The matrix-free solver therefore starts with the incumbent and two reproducible
complex random probes, projected and orthonormalized in $N'$. If their columns
form $V$, its Ritz problem is

$$
V^\dagger N' V=I,\qquad
(V^\dagger H'V)c_j=\theta_j c_j,\qquad
u_j=Vc_j,\qquad r_j=H'u_j-\theta_j N'u_j.
$$

The lowest Ritz vector can still be the unchanged excited incumbent while the
initial probe Rayleigh quotients lie above it. Consequently, the iteration
also examines residuals of the next two lowest Ritz vectors. The first
unconverged independent residual extends $V$. Coordinate seeds remain a
fallback for insufficient exploration; a restart keeps up to four low Ritz
vectors. Within a numerically degenerate lowest Ritz space, projection of the
incumbent selects a continuous representative whenever possible.

These probes change only the search space, never the physical Hamiltonian or
state being represented. In exact arithmetic the incumbent's inclusion makes
the lowest Ritz value no higher than its starting Rayleigh quotient. Random
probes avoid the deterministic coordinate-prefix blind spot; they are not a
proof that a finite capped iterative solve always finds the global lowest
local eigenvalue. Residual checks, independent physical-energy acceptance and
explicit iteration caps are still necessary. Regression tests compare against
independent diagonalizations with a stationary excited start, a disconnected
complex subspace, redundant/null coordinates and large coordinate scaling.
Computing the diagonal currently requires one norm action per source coordinate;
this accuracy correction is not claimed to improve runtime.

Fresh local residuals and independently contracted physical energies remain
necessary: a supported solve or a small sweep-energy change alone does not
certify convergence of the original variational problem.

## 6. Compression is a bilinear metric fit

Let $x$ be the target pair coefficient vector, and $M(a,b)$ the exact bilinear
merge of two legal source cores. The fitted state has a specified retained
middle-sector allocation. Its squared physical error is

$$
\mathcal L(a,b)
=\|F_{\mathrm{pair}}[M(a,b)-x]\|^2
=[M(a,b)-x]^\dagger N[M(a,b)-x].
$$

With a supported square root $S$, optimization uses the corresponding weighted
residual $r=S[M(a,b)-x]$. The nonlinear objective is $\tfrac12\|r\|^2$;
reported physical losses use the convention without the one-half. The source
merge and physical loss are retained for acceptance/diagnostics rather than
being replaced by an unweighted parameter norm.

A general $S$ mixes both pair sides. Although the weighted residual has a
Euclidean norm, $SM(a,b)$ is not generally an arbitrary rank-$D$ matrix in
those coordinates. This is the obstruction to solving the correlated-metric
problem with one ordinary SVD of a whitened target.

**ALS.** Fix $b$. The map $a\mapsto SM(a,b)$ is linear, so solve its weighted
least-squares problem by LSMR. Then fix the new $a$ and solve for $b$ the same
way. One ALS round comprises these two solves. The operator adjoint includes
both $S^\dagger$ and the source-embedding adjoints. Keep a factor proposal only
if the actual physical loss does not increase. Inner iteration caps, outer
round caps and factor-gradient stationarity are reported separately.

**Joint least squares.** Pack active real and imaginary parts of both factors
into real coordinates $u$. For a variation $(\delta a,\delta b)$,

$$
\delta M=M(\delta a,b)+M(a,\delta b),
\qquad \delta r=S\delta M.
$$

These expressions give the exact real Jacobian supplied to the trust-region
least-squares solver. Both factors are optimized together. Within each legal
sector/shared-label block, $a\mapsto aG$, $b\mapsto G^{-1}b$ leaves the merge
unchanged. The code balances factor products using small QR/SVD operations,
but joint coordinates still contain gauge redundancy.

## 7. Variable projection, with both factors explicitly identified

In one admissible gauge block, write the left factor as a matrix $X$ with
$r$ columns, and the right factor as $T$. The factors called "first" and
"second" here are exactly $X$ and $T$, respectively. For full-column-rank
$X_0$, choose orthonormal columns $Q_0$ spanning its column space, and an
orthonormal complement $Q_\perp$.

For a nearby $X$, completeness of these two bases gives

$$
X=Q_0(Q_0^\dagger X)+Q_\perp(Q_\perp^\dagger X).
$$

Set $C=Q_0^\dagger X$. In the chart where $C$ is invertible, define
$Z=(Q_\perp^\dagger X)C^{-1}$. Substitution gives, without omitting either term,

$$
X=Q_0C+Q_\perp ZC=(Q_0+Q_\perp Z)C,
\qquad XT=(Q_0+Q_\perp Z)(CT).
$$

Absorb $C$ into the right factor. Thus only the column subspace needs nonlinear
coordinates $Z$; the right factor contains the remaining linear freedom.
The implementation makes one such chart per allowed middle-sector/shared-label
block, never mixing inequivalent symmetry sectors. It may swap factor roles
to use the smaller chart. Wide/redundant factors are represented with zero
padding beyond the maximal effective column rank. If an initial factor is rank
deficient, QR completes its columns to the chosen chart dimension. That
completion is not unique and does not imply smoothness through a rank change.

Collect the chart entries into a real vector $z$, splitting real/imaginary
parts for complex data. The left factor is affine:

$$
a(z)=a_0+\sum_j z_j a_j.
$$

Let $e_\ell$ be the real-coordinate basis for the right factor. Realify the
weighted residual by stacking its real and imaginary parts. Define

$$
y=\operatorname{realify}(Sx),\qquad
K(z)_{:\ell}=\operatorname{realify}(S M(a(z),e_\ell)).
$$

For fixed $z$, the second factor's real coordinate vector $t$ solves

$$
\min_t\tfrac12\|K(z)t-y\|^2,
\qquad t_*(z)=K(z)^+y.
$$

This inner solve is a dense rank-revealing SVD, not an LSMR iteration. Substituting
its solution eliminates $t$ from the outer problem:

$$
f(z)=\tfrac12\|K(z)K(z)^+y-y\|^2.
$$

For the retained SVD $K=U_r\Sigma_rV_r^T$, the pseudoinverse is
$K^+=V_r\Sigma_r^{-1}U_r^T$. Therefore

$$
KK^+=U_r\Sigma_rV_r^TV_r\Sigma_r^{-1}U_r^T
=U_rU_r^T.
$$

The singular values cancel in this projector; they remain in the fitted
coefficients $t_*$ and in derivatives through $K^+$.

For completeness, let $r=Kt_*-y$, $P=KK^+$ and $K_j=\partial K/\partial z_j$.
The normal equation is $K^Tr=0$. On a constant-rank smooth region, differentiating
the fitted residual gives

$$
\frac{\partial r}{\partial z_j}
=(I-P)K_jt_*-(K^+)^T K_j^T r.
$$

The second term accounts for movement of the fitted subspace when the residual
is nonzero. Dropping it is not the implemented exact residual Jacobian.
`variable-projection` passes this Jacobian and the reduced residual to the
trust-region least-squares solver.

## 8. Grassmann-chart Newton and its Hessian

`grassmann-newton` uses the same balanced admissible factors, affine subspace
chart, and eliminated right factor. It changes the outer optimizer to a
trust-region Newton method on the reduced scalar objective.

Because $K(z)$ is affine, $K_{ij}=0$. Define columns $A_j=K_jt_*$ and a matrix
$B$ whose row $j$ is

$$
B_{j:}=A_j^T K+r^T K_j.
$$

Before eliminating $t$, differentiating
$F(z,t)=\tfrac12\|K(z)t-y\|^2$ at its fitted right factor gives

$$
F_{zz}=A^TA,\qquad F_{zt}=B,\qquad F_{tt}=K^TK.
$$

Differentiating $F_t(z,t_*(z))=0$ gives
$F_{tt}\,\partial t_*/\partial z=-F_{tz}$ on supported coordinates.
Substitution into the derivative of $f_z=F_z$ yields the Schur complement

$$
\nabla f=A^Tr,\qquad
\nabla^2 f=A^TA-B(K^TK)^+B^T.
$$

The code evaluates the second term as
$(BV_r\Sigma_r^{-1})(BV_r\Sigma_r^{-1})^T$ and symmetrizes the result.
This is not merely the Gauss–Newton matrix of the reduced residual.
The smooth interpretation requires a stable supported rank; ranks encountered
by the inner SVD are recorded. Crossing a rank threshold invalidates treating
that transition as a single smooth Hessian model. A fixed local chart also
is not a guarantee of a global optimum over all subspaces.

All nonlinear methods preserve the best finite physical-loss iterate they have
observed. Workspace estimates and some numerical failures trigger a reported
ALS fallback. For a cyclic metric, failure to allocate/factor the required
full local Gram instead triggers the caller's ordinary-step recovery. Neither
case silently substitutes a separable norm for a correlated one.

Source for sections 6–8: `pyqed/_letta_compression.py`,
`_ProjectedProblem`, `fit_metric_factors`, and the open/ring compression adapters.

## 9. Why a true virtual ring needs a target closure

With the unshifted owned-site charges used here, a trace of invariant physical
cores without an external representation leg has total charge zero and scalar spin. It cannot directly represent a charged
or non-singlet molecular target. The implementation introduces a covariant
closure core with one dual-target leg and two ordinary virtual bonds. Charges
on that auxiliary leg are the negatives of the desired total charges; its
SU(2) spin is the same because SU(2) irreps are self-dual.

Coupling the physical target multiplet to its dual produces an invariant state

$$
|\Xi\rangle=\frac1{\sqrt{2S+1}}
\sum_{M=-S}^S(-1)^{S-M}
|\psi_{SM}\rangle\otimes|S,-M\rangle_{\mathrm{aux}}.
$$

The physical states in an irreducible multiplet have equal norm. Orthogonality
of the auxiliary basis therefore gives

$$
\langle\Xi|\Xi\rangle
=\frac1{2S+1}\sum_M\langle\psi_{SM}|\psi_{SM}\rangle
=\langle\psi_{SM}|\psi_{SM}\rangle.
$$

A raw auxiliary slice has this squared norm divided by $2S+1$. Multiplication
by $\sqrt{2S+1}$ and the CG phase recovers a normalized physical-component
convention. A scalar Hamiltonian acts as $\widehat H\otimes I_{\mathrm{aux}}$;
its normalized expectation is the same in every target component.

The closure coefficients are variational. Both closure-adjacent virtual bonds
may be nonunit spaces. There are $L$ physical cores plus one closure core,
with graph edges $(0,1),\ldots,(L-1,L),(L,0)$. This graph is distinct from the
Hamiltonian interaction graph and from the physical tie graph. Two-site updates
visit these graph edges, including both closure edges; they do not omit the
closure and pretend that the last and first physical cores form a two-core pair.

Source: `ReducedRingTarget`, `ReducedRingLETTA`, and `CyclicReducedMPO`.

## 10. Cyclic reduced transfer contractions

For an open norm chain the external scalar boundary selects invariant boundary
channels. A ring instead traces a complete transfer product, so all allowed
channels of ket times dual bra must be retained. A structural coupled basis is

$$
U^{JM}_{m_bm_k}
=(-1)^{S_b-m_b}\langle S_km_k,S_b,-m_b\mid JM\rangle.
$$

Transform each local norm transfer into this basis. SU(2) invariance makes it
identity in the magnetic number $M$ within a total-$J$ channel, while a reduced
matrix $T_i^{(J)}$ acts on sector-pair and multiplicity labels. The trace is
therefore

$$
\langle\Xi|\Xi\rangle
=\sum_J(2J+1)\operatorname{Tr}
[T_0^{(J)}T_1^{(J)}\cdots T_L^{(J)}].
$$

The factor $2J+1$ counts magnetic components; keeping only $J=0$ is generally
incorrect. Each reduced local transfer coefficient is obtained by contracting
the left/right structural bases with the two state CG tensors and dividing by
$2J+1$. Closing the channel trace restores that dimension factor once.

For a Hamiltonian transfer, first couple ket spin with operator spin to an
intermediate spin, then couple with dual bra to total $J$. Keep every allowed
intermediate channel as well as every $J$. The cached `_operator_spin_transfer`
contains the corresponding structural contraction. Local H/N actions are
obtained by removing one or two transfers, contracting the cyclic complement,
and applying the open transfer legs to the trial reduced coefficients.

This environment is generally correlated across the pair's two outer legs.
`CyclicPairMetricRoot` consequently factors the full local reduced pair Gram.
It does not reuse the open-chain product of left and right boundary Grams.
The factorization is local in coefficient space, but can still be costly.
Scaled transfer products retain complete ranks/channels and separate exact
powers of two to prevent avoidable intermediate overflow or underflow.

Source: `reduced_ring_contraction.py`, `reduced_ring_pair.py`.

## 11. Relating CBE and two-site updates to these equations

Two-site optimization first solves the supported pair generalized eigenproblem
for a target $x_*$. It then fits $x_*$ with legal retained factors using the
physical metric above and optionally alternates one-site energy solves on
those factors. A smaller fitting error is not itself an energy acceptance
criterion. The resulting normalized state must pass a fresh physical energy
check and, when feasible at the requested cap, beat or match the independently
computed ordinary one-site baseline.

CBE instead starts from that ordinary baseline, forms its pair residual
$r_H=Hx-E Nx$, and converts it to supported orthonormal coordinates as
$g=W^\dagger r_H$. Variations already available within the incumbent allocation
have weighted tangent map

$$
J(\delta a,\delta b)=S[M(\delta a,b)+M(a,\delta b)].
$$

LSMR fits $Jz$ to $g$. Subtracting the fit gives $g_\perp=g-Jz_*$, and
$Wg_\perp$ is the coefficient target for new directions. Greedy physical-metric
fits choose complete middle-sector multiplets. One factor receives the chosen
new rows/columns and its partner receives zeros, so the wavefunction is unchanged
before the expanded one-site solve. Only that one-site energy problem is solved;
CBE never calls the pair energy eigensolver. Its supplied optimized pair vector
is compressed back to the cap and energy-refined. Acceptance requires strictly
lower physical energy than the same-start ordinary baseline.

The selector name `exact` identifies the native pair actions and retained
metric, not a proof that its greedy nonlinear allocation is globally best.
Likewise, one-site stationarity does not certify a global variational minimum.
Increasing a cap enlarges an available ansatz, but independent local optimization
runs are not guaranteed to find nested or globally optimal solutions.

Numerical or allocation failure discards temporary candidates. The ordinary
baseline is retained when available; otherwise the untouched incumbent remains.
The next step attempts the requested method again. Recovery/status fields and
fresh local residual audits prevent an unchanged rejected state from masquerading
as converged. Recovering a Python exception does not guarantee survival of an
OS memory kill or an inability to allocate even the restored incumbent's work.

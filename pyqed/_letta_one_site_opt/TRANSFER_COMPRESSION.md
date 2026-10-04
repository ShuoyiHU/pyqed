# Experimental segment transfer compression

This adapts the transfer-product SVD idea of Pippan, White and Evertz,
[Phys. Rev. B 81, 081103(R) (2010)](https://arxiv.org/abs/0801.1947),
to LETTA's frontier contractions. It is available on norm and Hamiltonian
environment caches, including the pair caches used by CBE and two-site
optimization. **It is an explicit experimental API, not an enabled sweep
backend or a replacement for metric-aware factor compression.**

## What is compressed

For cuts `a < b`, let a vector contain every index of the frontier at `a`.
Applying the local contractions at sites `a, ..., b-1` is a linear map:

$$
\ell_b = M_{b:a}\ell_a,
\qquad M_{b:a}=T_{b-1}\cdots T_a.
$$

Every physical index shared by several LETTA tensors stays on the frontier
until its last occurrence. A local transfer can therefore be rectangular;
it is not generally a plain MPS transfer of dimension bond-dimension squared.
The implementation retains these copy constraints, including backward and
nonlocal physical ties. Right-environment propagation uses the transpose:

$$
r_a=M_{b:a}^{\mathsf T}r_b.
$$

The matrix-free SVD also needs the *adjoint*, which differs for complex states:

$$
M_{b:a}^{\dagger}z
=\overline{M_{b:a}^{\mathsf T}\overline z}.
$$

`SegmentTransfer` freezes the segment's tensors and compiles/caches contraction
paths through the existing contraction machinery. It never constructs the
full segment matrix. Source tensors may subsequently change without corrupting
the snapshot; rebuild the snapshot to represent those changes.

## Randomized row-space SVD

Let the map have shape `m x n`, requested retained rank `p`, and sketch width
`k = min(p + oversampling, m, n)`. Draw a Gaussian matrix of shape `m x k`:

$$
\Omega\in\mathbb C^{m\times k},\qquad
Y=M^\dagger\Omega,\qquad Q=\operatorname{orth}(Y).
$$

Optional power iterations repeat the two orthogonalizations:

$$
P=\operatorname{orth}(MQ),\qquad
Q\leftarrow\operatorname{orth}(M^\dagger P).
$$

Then form and decompose a skinny matrix:

$$
B=MQ=U\Sigma W^\dagger.
$$

Retaining the first `p` singular directions gives

$$
\widehat M=U_p\Sigma_pW_p^\dagger Q^\dagger
=U_p\Sigma_p V_p^\dagger,
\qquad V_p=QW_p.
$$

Application now consists of two skinny matrix products. This is an
approximate SVD: its error is not necessarily the optimal rank-`p` error.
The bounded dense benchmark computes that optimal error independently:

$$
\min_{\operatorname{rank}(X)\le p}
\frac{\|M-X\|_F}{\|M\|_F}
=\sqrt{\frac{\sum_{j>p}\sigma_j^2}{\sum_j\sigma_j^2}}.
$$

## Checks and limitations

Independent random probes estimate relative forward and adjoint action errors.
For example, with new probe vectors independent of the construction sketch:

$$
\epsilon_{\mathrm{forward}}
=\sqrt{\frac{\sum_j\|(M-\widehat M)g_j\|^2}
                   {\sum_j\|Mg_j\|^2}}.
$$

Both estimates must meet `tolerance`, and the factors must use less storage
than a dense segment map. Otherwise `apply_left` and `apply_right` fall back
to the exact snapshot. The raw candidate remains available for diagnostics.

These tests **do not certify a worst-case error, positive norm metric, or
energy accuracy**. Transfer indices and local-metric indices have different
groupings. A truncated transfer SVD need not preserve the positive
semidefinite local metric after contraction/reshaping. Also, small absolute
metric errors can dominate weakly supported local directions. Simply taking
this SVD does not whiten the physical compression metric or remove the
nonconvexity of optimizing LETTA factors.

Singular spectra depend on the tensor gauge. The benchmark reports the
representation produced by the current solvers; it does not establish
gauge-independent lower bounds for every possible implementation.

The dense-map storage check does not imply a saving over our existing exact
frontier propagation, which already avoids a dense segment map. The snapshot
is retained for fallback. The workspace limit estimates sketch/SVD allocation;
it excludes snapshot tensors, contraction intermediates and library workspace.
Hamiltonian maps currently use generic dense local MPO factors, not the sparse
MPO-channel optimization of the standard sweep cache.

Building a compressed segment requires many exact applications. Savings need
repeated use while all segment tensors remain fixed. Our existing open-boundary
sweeps already reuse completed boundary environments; compressing each segment
anew on every update is unlikely to repay its construction cost.

## Explicit use

```python
from pyqed._letta_one_site_opt import IdentityEnvironmentCache

cache = IdentityEnvironmentCache(state)
left = cache.build_left_environments()
transfer = cache.segment_transfer(3, 6)  # sites 3, 4, 5; frozen state
compressed = transfer.compress(rank=16, tolerance=1e-8, seed=0)
print(compressed.diagnostics)
boundary_at_6 = compressed.apply_left(left[3])
```

The same method is on `LETTAEnvironmentCache`, `IdentityPairEnvironmentCache`
and `LETTAPairEnvironmentCache`. An approximate source boundary-MPS cache is
rejected to avoid silently combining two independent approximations.

## Reproduce the pilot

From the repository root:

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/letta-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv-1/bin/python -m pyqed._letta_one_site_opt.benchmarks.transfer_compression \
  --model bose_hubbard --shape 3 2 --bond-dim 3 --sweeps 4 \
  --methods one-site cbe two-site \
  --output /private/tmp/letta-transfer/bose32.json
```

This first runs exact-environment optimization, then audits compression of
frozen initial and optimized states. The reported energy shift changes **only
the norm denominator at a fixed state**, keeping the exact Hamiltonian
expectation numerator. It is an error diagnostic, not an energy from a new
optimization algorithm. Pair-metric checks report Hermiticity, eigenvalues,
and distance from the exact local metric. Timing includes SVD construction
after snapshot creation, and separately measures repeated map applications.

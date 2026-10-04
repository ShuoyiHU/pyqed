# Periodic Hubbard one-site comparison

The implementation is in `periodic.py`; the benchmark entry point is
`benchmarks/periodic_hubbard.py`. It supports a true trace over closed virtual
bonds for both states, not an open MPS used with a periodic Hamiltonian.

$$
\psi_{\mathrm{MPS}}(s_0,\ldots,s_{L-1})
=\operatorname{Tr}[A_0^{s_0}A_1^{s_1}\cdots A_{L-1}^{s_{L-1}}],
$$

$$
\psi_{\mathrm{LETTA}}(s_0,\ldots,s_{L-1})
=\operatorname{Tr}[A_0^{s_0,s_1}A_1^{s_1,s_2}\cdots A_{L-1}^{s_{L-1},s_0}].
$$

The four physical states are empty, up, down, and double occupation. The model is

$$
H=-\sum_{i=0}^{L-1}\sum_{\sigma=\uparrow,\downarrow}
(c_{i\sigma}^{\dagger}c_{(i+1)\bmod L,\sigma}+\mathrm{h.c.})
+4\sum_i n_{i\uparrow}n_{i\downarrow}.
$$

There is no chemical-potential energy shift. The last-first hopping includes the
full Jordan-Wigner string in the chosen site ordering.

## Exact half filling and bond sectors

Every nonzero tensor element obeys

$$
q_a+n(s_i)-q_b=1.
$$

Summing around the closed virtual loop cancels the virtual charges, enforcing
exactly L electrons. Total spin projection is not fixed. The same charge
allocation is used for both methods:

| D | Virtual charges |
|---|---|
| 2 | 0, 1 |
| 3 | 0, 1, −1 |
| 4 | 0, 1, −1, 0 |
| 6 | 0, 1, −1, 0, 1, −1 |

These are specific U(1) sector allocations, not an optimization over every
possible assignment of D charge channels. LETTA has additional physical
arguments and consequently more free parameters at the same D. This is not a
comparison at equal parameter count or equal runtime.

## One-site optimization and gauges

Exact cyclic double-layer transfer products construct the active metric and
Hamiltonian. The same supported-metric generalized eigenproblem is solved in
both methods:

$$
H_i a_i=E N_i a_i.
$$

The supported metric basis uses the existing diagonally equilibrated Hermitian
metric solver. The relative metric cutoff is 1e-11. Neither method assumes
that the ring environment metric is the identity. No physical-basis Hamiltonian
projection is used by the optimizer.

Before a local solve, charge-preserving square roots of the left and right
metric marginals give invertible gauge transformations, cancelled by inverse
transformations on neighboring tensors. LETTA allows these gauges to depend
on the physical value shared across the bond. The marginal gauge floor is
1e-6; it regulates a coordinate transformation and does not truncate the state.

Metric gauges alone can cause large factor norms. After each local update,
invertible diagonal bond scaling controls this problem. For virtual channel b,
let l_b be the squared norm of the corresponding column of the left factor and
r_b the squared norm of the corresponding row of the right factor. The scaling
minimizes

$$
f_b(g)=l_b g^2+r_b/g^2,\qquad g_b=(r_b/l_b)^{1/4}.
$$

The left column is multiplied by g_b and the right row divided by g_b.
Scales are bounded between 1e-3 and 1e3 per application; zero channels are left
unchanged. Every transformation is invertible, diagonal in virtual charge,
and exactly cancelled on the neighbor. LETTA can apply it independently for
each shared physical value. The full correlated metric is still solved.

No SVD truncation or rank-deficient pair refactorization is used: such a
refactorization could preserve the wavefunction while shrinking a neighboring
one-site search space. Tests check both wavefunction and one-site-space
preservation. There is no CBE, bond growth, or two-site variational optimization
during a run.

Each candidate is checked against the contracted whole-ring energy. Numerical
energy increases above 1e-10 are rejected. Gauge transformations are likewise
checked and rolled back if they change the energy beyond that tolerance.
Transfers are independently rebuilt at the end of every sweep. One sweep is
one directional pass through all sites, with the direction alternating.

## Run protocol

For each L=5,10 and D=2,3,4,6, a seeded random half-filled MPS initializes both
methods. LETTA is initialized with no dependence on its extra physical argument,
so their starting physical wavefunctions are identical. Each run is capped at
500 sweeps. Early stopping requires energy-density changes below 1e-11 for
three consecutive sweeps without a rejected variational update. This criterion
establishes a plateau, not a global variational-minimum certificate.

A second paired D6 initialization uses the D4 MPS result, padded to D6 with
small charge-allowed noise. This happens once before optimization, not through
adaptive expansion during sweeps. Both D6 methods again receive exactly the
same starting wavefunction. This check probes local-minimum sensitivity rather
than concealing it by choosing different starts for the two methods.

```bash
bash pyqed/_letta_one_site_opt/benchmarks/run_periodic_hubbard.sh /private/tmp/periodic_hubbard_fresh
```

The benchmark writes per-sweep JSON, final tensors, source hashes, and PNG/PDF
plots. The logarithmic plot uses the positive values of E_k−E_final_round;
zero and negative roundoff differences have no logarithm and are omitted.
That quantity measures progress relative to the last recorded sweep, not error
relative to the exact ground-state energy.

Independent validation constructs all C(2L,L) half-filled amplitudes in batches
and applies Hubbard hopping directly with occupation-bit fermionic signs.
This reference calculation is separate from optimization. The ground-state
reference uses sparse exact diagonalization in minimal |Sz|, which contains a
member of every SU(2) spin multiplet at fixed electron count.

## Bose–Hubbard comparison

The same state representation, transfer contractions, one-site solver and gauges
also support bosons by setting `particle_numbers=(0,1,2)` (or a larger cutoff).
The benchmark uses t=1, U=4, N=L, chemical potential zero, and occupation cutoff
n_max=2, with Hamiltonian

$$
H=-\sum_{i=0}^{L-1}(b_i^\dagger b_{(i+1)\bmod L}+\mathrm{h.c.})
+\frac{4}{2}\sum_i n_i(n_i-1).
$$

The unit-density charge mask, charge allocations, matched starts, D values,
500-sweep cap, stopping tolerances and paired D6 warm-start checks are identical
to the fermionic protocol. For both bosonic methods, the invertible marginal
gauge uses a floor of 1e-3 (selectable with `--gauge-floor`), rather than the
fermionic benchmark's 1e-6. The original smaller floor led to ill-conditioned
factors and a failed independent energy check in the 5-site D4 LETTA pilot.
The metric support cutoff remains 1e-11. Gauge regularization does not discard
state components: the neighboring inverse is applied exactly.
Bosonic hopping has no Jordan–Wigner sign.
The finite local occupation cutoff defines the tested model; these runs do not
establish convergence as that cutoff increases.

```bash
bash pyqed/_letta_one_site_opt/benchmarks/run_periodic_bose_hubbard.sh /private/tmp/periodic_bose_hubbard_fresh
```

`periodic_bose_validation.py` independently enumerates configurations with N=L
and constructs hopping amplitudes from the occupation-number square roots.
It supplies sparse exact ground-state references and checks every final state
energy and variance against this independent Hamiltonian. This occupation basis
is used only for validation, never inside the active optimizer.

## Directional normalization gauge (paper comparison)

`PeriodicOneSiteOptions(gauge_method='paper')`, or the benchmark flag
`--gauge-method paper`, selects the directional normalization used in the
periodic one-site framework of Verstraete, Porras and Cirac, PRL 93, 227205
(2004), https://doi.org/10.1103/PhysRevLett.93.227205.
In this implementation's matrix orientation, forward sweeps impose
sum_s A[s]^dagger A[s] = I and absorb the compensating factor into the next
tensor. Backward sweeps impose sum_s A[s] A[s]^dagger = I and absorb the
factor into the previous tensor. The full correlated overlap environment is
recomputed and retained in every local generalized eigenproblem.

Charge-block SVDs construct square, invertible gauges. Singular values below
1e-12 times the factor norm receive a finite scale rather than an inverse of a
near-zero value. Thus rank-deficient blocks normalize only on their supported
subspace; no virtual channel or neighboring one-site direction is deleted.
Counts of these regularized values and rejected gauges are recorded per sweep.
For LETTA the operation is performed conditionally on the physical index shared
with the receiving neighbor; this is an adaptation, not a claim that the paper
itself optimizes LETTA.

This option replaces both the pre-update marginal gauge and the post-update
diagonal balancing. It does not add expansion, compression, or a two-site solve.
The default remains `marginal` so comparisons are explicit.

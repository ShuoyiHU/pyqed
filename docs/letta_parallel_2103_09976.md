# Zhai–Chan parallel DMRG methods applied to LETTA

Paper: H. Zhai and G. K.-L. Chan, *Low communication high performance ab initio
density matrix renormalization group algorithms*, arXiv:2103.09976v3 (2021).
Reviewed all theory sections II A–G, results III A–D, and appendices A–B.

## Point-by-point assessment

| Paper point | LETTA assessment and implementation status |
| --- | --- |
| II A: distribute normal/complementary operators, retain partial R operators | The key observation is to communicate the local action vector rather than large renormalized operators. This requires distributed ownership of environments; current LETTA has no MPI sweep owner. **Not implemented as distributed parallelism.** |
| II B, Eqs. 1–6: symmetry-preserving sub-Hamiltonian partition | Applicable in principle: H_eff x = sum_p H_eff,p x, with the same metric and trial vector on every rank. Integral permutations must share an owner; ecore must be counted once. Independently minimizing each part and summing states is incorrect. A real implementation must shard construction, storage, environment updates, local diagonals and residual checks, and all-reduce every action. Simply duplicating the existing MPO on every rank would not deliver the paper's memory benefit. **Not implemented.** |
| II B advantages (i)–(vi): communication, simplicity, MPO independence, index symmetry, balance, sparse scaling | Existing streamed compiler avoids expanded all-core storage, but is not a partitioned NC/CN builder and still has dense coefficient arrays. These advantages must be demonstrated separately on cluster-scale chemistry; no distributed speedup is claimed. |
| II C, Eqs. 7–10: multi-center site sweeps | LETTA has nonlocal ties and a nontrivial local metric. The MPS connection-matrix formula is not a proof that independently updated tied regions can be merged consistently. Would need a tie-aware boundary representation, refreshed environments, singular-support handling and comparison of converged energies/iterations. **Not implemented; sweeps remain sequential.** |
| II C: dynamic connection sites | Balance by measured work rather than equal site counts is relevant. Dynamic *site* boundaries require the multi-center algorithm above. This patch balances independent operator/output tasks instead; it is not dynamic-site DMRG. |
| II D: shared-memory operator sums | **Implemented:** a two-phase compiled action schedule produces shared first GEMMs, then consumes them in independently owned output tiles. No full output vector per worker and no vector reduction are needed. |
| II E: collapse operator and symmetry-sector tasks | **Implemented:** prepared batches combine routes across operators/sectors; longest-estimated-work-first placement balances batches. Each output row tile has one owner, preserving its contribution order. No nested worker pools. |
| II F: threaded dense BLAS | Already available through the linked BLAS. Keep BLAS threads at one when using outer workers, as the paper recommends for its measured systems. No claim that the same optimum holds for every block size or machine. |
| II G: stack storage and delayed contraction | Existing packed boundaries and lazy environments remain. **Added:** reusable per-caller product arenas and per-worker workspaces; compiled plans are reused over Davidson actions. Products are refreshed for every trial vector, never reused as valid numerical data across vectors. |
| II G: identity / single-entry matrices | **Added:** exact scaled-identity and single-entry first-product kernels; no threshold or approximate sparsification. Detection avoids allocating a dense identity matrix. Existing structured boundary optimizations remain. |
| II G(i): symbolic simplification and unused operators | Existing AutoMPO prefix sharing, exact fixed-channel combination, operator-path pruning and lazy environment preparation cover parts of this idea. This patch preserves them; it does not implement a new symbolic NC/CN engine. |
| II G(ii): delayed GEMM list and reusable block skeleton | **Implemented:** route, tile, special-kernel and product-offset arrays are prepared once per operator/worker count. Python dispatch is per batch, not per route. |
| II G(iii): raw I/O and lossy floating compression | Existing optional product scratch uses raw memory maps. The new compiled path handles RAM-only arenas; disk-backed cases retain the Python reference path. **No lossy compression added:** the paper's 1e-14 per-number tolerance is an additional approximation, not authorized by a request to preserve accuracy. |
| III A / appendix B: mixed NC/CN, index symmetries | Existing general builder/compiler is not this specialized representation. A mixed builder needs independent Hamiltonian-action validation, including its midpoint transform; not replaced by this patch. |
| III B–D: multi-level scaling and benchmark protocol | This patch reports matched actions and complete tiny chemistry stages. It does not extrapolate the paper's 448–2800-core results to LETTA or claim multi-node scaling. |
| Results: perturbative noise / minimum state per quantum number | These change state-space exploration and truncation. No changes made to LETTA's sector allocation, CBE policy, solver iterations or tolerances. |

## Mathematical work and memory

Each nonzero reduced route evaluates `Y += c L X R^T`, with L:(a,l), X:(l,r),
R:(b,r). These are internal multiplicity dimensions, not the user's D_mul.
Tied frontier spaces may be larger. The chosen order needs
`min(a*l*r + a*r*b, l*r*b + a*l*b)` multiply-accumulates (MACs), excluding
coefficient scaling and output accumulation. Real FLOPs are approximately twice
MACs; complex arithmetic needs a different FLOP multiplier.

For a shared first product, the production cost is paid once for the current
vector. Every distinct second multiplication remains necessary. Shared output
row tiles never overlap, so the original route accumulation order is retained
per element. Uncached first products are recomputed, and their outputs are not
tiled in ways that would duplicate that work. Exact identity/single-entry
products need only scaling/copying/zero-filling rather than general GEMMs.

The previous shared-product path already eliminated repeated first GEMMs but
executed each second contraction from Python. The new path removes that
interpreter dispatch and reuses buffers; it does not claim the entire naive
arithmetic saving as a new improvement over the previous implementation.

In the synthetic n=96, four-sector replay: 1536 routes; naive work 2,717,908,992
MACs. Prepared work: 28,311,552 first-product MACs plus 1,358,954,496 second-phase
MACs. The four second-stage workers each receive 339,738,624 MACs. The real
product arena is 2,359,296 bytes and workspace is 221,184 bytes per worker,
excluding boundaries, packed schedules, input/output and Python process storage.
Complex arrays double these byte counts. These are allocation sizes, not RSS.

## Matched measurements

macOS local Python 3.13; NumPy/SciPy BLAS threads limited to one; fixed random
seed, same matrices/vector; seven warm timed actions (compilation excluded).
Times below are median milliseconds per action:

| Implementation | 1 worker | 2 workers | 4 workers |
| --- | ---: | ---: | ---: |
| Existing compiled sector path | 13.02 | 11.29 | 8.53 |
| Previous Python shared-product path | 28.20 | 16.44 | 26.87 |
| New compiled shared-product path | 12.38 | 11.08 | 7.89 |

At four workers this is 3.41x faster than Python shared products, but only 1.08x
faster than the existing compiled sector path. Compared with the existing
single-worker compiled path it is 1.65x faster. The new compiled action was
bitwise identical to the existing compiled action in this replay; validation
elsewhere uses numerical tolerances to allow BLAS-dependent rounding.

Three-orbital chemistry, same seeded initial state, D_mul=4, two workers,
two directional sweeps in each optimization stage, median of three replays:

| Stage | Existing sector | Python shared | New compiled shared |
| --- | ---: | ---: | ---: |
| Two-site MPS | 135.11 ms | 147.59 ms | 139.08 ms |
| One-site MPS | 39.24 ms | 43.28 ms | 40.02 ms |
| One-site LETTA | 43.48 ms | 47.62 ms | 44.43 ms |

All three final energies were -0.05907746417949217 Ha. This tiny full workflow
shows that the shared path is not universally faster than the ordinary sector
path. Therefore global shared-product selection remains opt-in. These are not
FeCO5 or Cr2 production timings, and no whole-job speedup is promised.
The full raw report is `docs/benchmarks/letta_parallel_2103_09976.json`.

## Use and validation

The compiled backend is now the default **within the opt-in RAM-only shared
product path**, used by both one-site and two-site LETTA actions. Existing jobs
with PYQED_SHARED_PRODUCTS unset retain their existing dispatch.

```bash
export PYQED_SHARED_PRODUCTS=1
export PYQED_SHARED_PRODUCT_BACKEND=compiled
export PYQED_PRODUCT_CACHE_MB=64
export PYQED_PRODUCT_DISK_MB=0
export PYQED_SECTOR_THREADS=4  # request at least four CPUs for this process
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Use `PYQED_SHARED_PRODUCT_BACKEND=python` for the previous reference or
`PYQED_SHARED_PRODUCTS=0` for the ordinary compiled path. Existing RAM/disk
budgets are preserved. Disk-backed shared products currently use the Python
path rather than silently dropping or exceeding their disk budget.

Tests cover real/complex operators and vectors, changing vectors and dtypes,
zero-size sectors, zero budgets, disk fallback, repeated arena reuse, concurrent
callers, row tiling, special kernels, and complete small MPS/LETTA energy
histories. No production determinant-space projection was added.

```bash
PYTHONPATH=. python -m pyqed._letta_one_site_opt.benchmarks.paper_parallel \
  --whole-sweeps --output /tmp/letta-paper-parallel.json
```

For production follow-up, capture representative FeCO5/Cr2 local operators
and time preparation, solve actions, updates and whole sweeps separately on
allocated cluster CPUs before changing cluster defaults. MPI sub-Hamiltonians
and multi-center tied sweeps remain substantial unimplemented portions of the
paper, not hidden behind these shared-memory switches.

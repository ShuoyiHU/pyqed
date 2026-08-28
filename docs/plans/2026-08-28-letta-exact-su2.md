# Exact Reduced SU(2) LETTA Implementation Plan

> **For Codex:** Follow TDD for every task: add the focused test, observe the
> intended failure, implement the smallest correct change, and rerun both the
> focused and neighboring regression tests.

**Goal:** Add user-defined U(1) and exact reduced SU(2) sectors to the one-site
and two-site LETTA optimizers, reuse the existing Wigner-Eckart/non-Abelian DMRG
engine, and provide reproducible energy/resource benchmarks.

**Representation:** An SU(2)-adapted LETTA tensor stores only reduced
multiplicity coefficients.  Its owned physical leg fuses the incoming virtual
irrep with a user-declared local irrep into the outgoing virtual irrep.  A
positive-neighbor conditioning leg may contain only symmetry-scalar labels
(irrep/multiplicity labels, never magnetic quantum numbers).  This restriction
is essential: the ordinary LETTA copy tensor for repeated magnetic indices is
not an SU(2) intertwiner.  For a pure spin-1/2 site there is one reduced local
label, so the conditional dimensions are one and the exact SU(2) ansatz becomes
a reduced MPS backbone.  Multi-irrep sites (for example empty/single/double
spatial orbitals) retain nontrivial scalar LETTA conditioning.

**Execution model:** Convert the reduced conditional LETTA factor graph to an
exact reduced MPS with frontier-memory multiplicities.  The memory labels are
SU(2) scalars and therefore enlarge multiplicity spaces without changing irrep
fusion.  Local effective problems are projected through the sparse linear
embedding from LETTA parameters to the frontier MPS site (or pair) parameters.
This preserves the LETTA parameter tying while reusing the tested reduced MPO,
environment, recoupling, Davidson, and irrep-aware SVD code.

---

## Task 1: Establish and repair the reduced-engine baseline

**Files:**

- Modify: `pyqed/mps/symmetry.py`
- Modify only as required: `pyqed/mps/nonabelian/contraction.py`
- Modify only as required: `pyqed/mps/nonabelian/solver.py`
- Test: `tests/test_nonabelian_tensor.py`

1. Record the current focused baseline and separate unrelated legacy Abelian
   complementary-operator failures from the non-Abelian reduced path.
2. Add/retain focused tests for `Sector.is_abelian`, single-axis contracted
   channel metadata, and reduced-Krylov reporting.
3. Implement the missing sector property and correct only the reduced-engine
   metadata/bookkeeping defects needed by LETTA.
4. Run the non-Abelian tensor/state/operator/model tests and record any
   remaining pre-existing failures explicitly.

## Task 2: Add a user-defined reduced symmetry descriptor

**Files:**

- Create: `pyqed/_letta_one_site_opt/reduced_symmetry.py`
- Modify: `pyqed/_letta_one_site_opt/__init__.py`
- Test: `tests/test_letta_reduced_symmetry.py`

1. Test a generic local decomposition `H = direct_sum(V_j tensor C^m)` with
   user labels, optional U(1) charges, irrep labels, and multiplicities.
2. Test convenient constructors for a spin-1/2 site and the existing fully
   reduced spatial-orbital site.
3. Test Clebsch-Gordan reachability and target-sector validation.
4. Test that magnetic-component labels are rejected as LETTA conditioning
   labels with a clear explanation.
5. Implement immutable `ReducedPhysicalBasis`, `ReducedSymmetry`, and bond
   sector/multiplicity allocation helpers using `SU2Irrep`,
   `SpinChargeSector`, and existing fusion functions.

## Task 3: Implement exact reduced LETTA storage and dense reconstruction

**Files:**

- Create: `pyqed/_letta_one_site_opt/reduced_state.py`
- Modify: `pyqed/_letta_one_site_opt/__init__.py`
- Test: `tests/test_letta_reduced_state.py`

1. Test block shapes and selection rules for one-dimensional and rectangular
   lattices.
2. Test that forbidden fusion blocks cannot be inserted.
3. Test exact expansion of a two-spin singlet against the Clebsch-Gordan
   reference vector and verify `S^2 = 0` to numerical precision.
4. Test U(1)-only reduced storage against the existing Abelian LETTA state.
5. Implement `ReducedLatticeLETTA`, block copy/normalization, parameter counts,
   symmetry violation diagnostics, reduced path amplitudes, and small-system
   dense reconstruction for verification only.

## Task 4: Build the exact frontier-memory MPS embedding

**Files:**

- Create: `pyqed/_letta_one_site_opt/reduced_frontier.py`
- Test: `tests/test_letta_reduced_frontier.py`

1. Test frontier sets for 1D, 2D, 3D, and custom site orderings.
2. Test sparse one-site embedding maps and their adjoints.
3. Test that the embedded reduced MPS reconstructs the same dense state as the
   reduced LETTA object on small lattices.
4. Test that memory multiplies virtual multiplicities but never creates or
   changes SU(2) irreps.
5. Implement frontier enumeration, deterministic consistency constraints,
   block-sparse embedding, and projected parameter pack/unpack helpers.

## Task 5: Add exact reduced one-site optimization

**Files:**

- Create: `pyqed/_letta_one_site_opt/reduced_solver.py`
- Modify: `pyqed/_letta_one_site_opt/solver.py`
- Test: `tests/test_letta_reduced_one_site.py`

1. Test projected local norm and Hamiltonian matrices against dense local-frame
   references on two and four sites.
2. Test that every one-site update remains in the requested total-spin sector.
3. Test energy monotonicity for a short SU(2)-invariant chain.
4. Implement reduced one-site environment construction, projected Davidson
   matvec/diagonal, block gauge movement, sweep history, and dispatch from the
   public LETTA optimizer.

## Task 6: Add exact reduced two-site optimization and truncation

**Files:**

- Create: `pyqed/_letta_two_site_opt/reduced_pair.py`
- Modify: `pyqed/_letta_two_site_opt/solver.py`
- Modify: `pyqed/_letta_two_site_opt/truncation.py`
- Test: `tests/test_letta_reduced_two_site.py`

1. Test pair embedding and adjoint projection against dense references.
2. Test blockwise SVD by intermediate irrep, including whole-multiplet keeping
   and discarded-weight accounting with `(2j+1)` weights.
3. Test exact merge/split round trips without truncation.
4. Test that two-site sweeps retain the target irrep and lower the energy.
5. Implement projected pair optimization and irrep-aware conditional split,
   then dispatch from the public two-site optimizer.

## Task 7: Add reduced Heisenberg operators and comparison benchmarks

**Files:**

- Create: `pyqed/_letta_two_site_opt/benchmarks/heisenberg_symmetry.py`
- Create: `tests/test_letta_su2_benchmark.py`
- Modify: one-site and two-site READMEs

1. Implement a fully reduced spin-1/2 vector operator and scalar Heisenberg
   MPO using the existing `ReducedTensorOperator`/rank-coupled AutoMPO path.
2. Compare dense/no-symmetry, U(1) fixed-`Sz`, and SU(2) singlet runs from
   equivalent initial states and convergence tolerances.
3. Report energy, error against exact diagonalization, wall time, tensor
   storage, stored parameters, largest local solve dimension, cubic local-work
   proxy, symmetry violation, and whole-multiplet bond dimensions.
4. Assert energy agreement and exact symmetry diagnostics in the smoke test;
   keep timing assertions ratio-based and limited to a size where reduced
   arithmetic is expected to dominate Python overhead.

## Task 8: Final verification and handoff

1. Run all new reduced-LETTA tests, existing LETTA symmetry tests, and the
   relevant non-Abelian engine suite with deterministic BLAS thread counts.
2. Run the benchmark at a smoke-test size and at a larger performance size.
3. Inspect the branch diff for accidental changes and document known
   limitations, especially the scalar-conditioning rule.
4. Provide exact commands and machine-readable JSON/CSV output so the user can
   reproduce energy and efficiency comparisons.

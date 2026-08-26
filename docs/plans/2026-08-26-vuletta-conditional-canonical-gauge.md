# Conditional-Canonical VULETTA Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Make the nearest-neighbor uniform LETTA solver use an explicit physical-state-conditioned mixed-canonical gauge and horizontal tangent coordinates by default, then benchmark observables and runtime against VUMPS across bond dimensions.

**Architecture:** Represent the two equivalent copy-tensor embeddings of a nearest-neighbor LETTA and canonicalize their conditioned sectors into `TL`, `TR`, and per-state center matrices `C`. Parameterize every active variation in the orthogonal complement of the left-canonical conditioned blocks and whiten it with the supported right environment, so scale, phase, and virtual-gauge directions never enter the active coordinates. Retain the existing dense tangent-Gram solver as a numerical oracle and legacy update path.

**Tech Stack:** Python, NumPy, SciPy, pytest, `time.perf_counter`, existing `_vuletta` thermodynamic contractions and `_vumps` reference solver.

---

### Task 1: Conditional-canonical state and exact NN gauge

**Files:**
- Modify: `pyqed/_vuletta/state.py`
- Modify: `pyqed/_vuletta/__init__.py`
- Test: `tests/test_vuletta.py`

**Step 1: Write the failing public-API tests**

Add tests that request the wished-for API:

```python
from pyqed._vuletta import ConditionalCanonicalLETTA, conditional_canonicalize


@pytest.mark.parametrize("real", [True, False])
def test_conditional_canonicalize_preserves_state_and_enforces_sector_isometries(real):
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=41, real=real)
    canonical = conditional_canonicalize(state)

    assert isinstance(canonical, ConditionalCanonicalLETTA)
    assert canonical.left_isometry_error() < 1.0e-10
    assert canonical.right_isometry_error() < 1.0e-10
    assert canonical.center_error() < 1.0e-10
    for configuration in ((0, 0, 1), (0, 1, 1, 0)):
        np.testing.assert_allclose(
            canonical.state.periodic_amplitude(configuration),
            state.periodic_amplitude(configuration)
            / canonical.amplitude_scale(len(configuration)),
            atol=1.0e-10,
        )
```

Also add focused tests for:

- shapes `TL.shape == TC.shape == TR.shape == (D,d,d,D)` and `C.shape == (d,D,D)`;
- the exact relations `TC[p,q] = TL[p,q] @ C[q] = C[p] @ TR[p,q]`;
- invariance of thermodynamic energy and one-/two-site observables;
- finite/full-rank validation and clear rank-deficiency errors;
- the shifted structured-MPS embedding having the same periodic amplitudes.

**Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/test_vuletta.py -k conditional_canonical
```

Expected: collection/import failure because the new class and function do not exist.

**Step 3: Implement the two exact structured embeddings**

Add `UniformLETTA.shifted_structured_mps_tensor()` with

$$
\widetilde B^s_{(\alpha,p),(\beta,q)}
=\delta_{p,s}T^{p,q}_{\alpha\beta}.
$$

Keep `structured_mps_tensor()` as the right-copy embedding

$$
B^s_{(\alpha,p),(\beta,q)}
=\delta_{q,s}T^{p,q}_{\alpha\beta}.
$$

**Step 4: Implement supported positive square roots**

Add private Hermitian helpers that:

- symmetrize each fixed-point block;
- reject eigenvalues below a negative numerical tolerance;
- return square root, inverse square root, and retained rank;
- require full conditioned rank in `conditional_canonicalize` by default;
- expose `allow_rank_deficient=True` only for supported-subspace diagnostics, not active optimization.

**Step 5: Implement `ConditionalCanonicalLETTA`**

Store:

```python
@dataclass(frozen=True)
class ConditionalCanonicalLETTA:
    TL: np.ndarray
    C: np.ndarray
    TR: np.ndarray
    TC: np.ndarray
    transfer_eigenvalue: float
    conditioned_ranks: tuple[int, ...]
```

Provide `state`, `left_isometry_error`, `right_isometry_error`, `center_error`, and `validate` methods. `state` returns the normalized thermodynamic state represented by `TL`.

**Step 6: Implement uniform conditional canonicalization**

For normalized transfer tensor $T_0=T/\sqrt\lambda$:

1. Obtain block-diagonal left fixed points $l_q$ from the right-copy embedding.
2. Obtain block-diagonal right fixed points $r_p$ from the shifted embedding.
3. Form

$$
T_L^{p,q}=l_p^{1/2}T_0^{p,q}l_q^{-1/2},
$$

$$
T_R^{p,q}=r_p^{-1/2}T_0^{p,q}r_q^{1/2},
$$

$$
C_p=l_p^{1/2}r_p^{1/2},
\qquad
T_C^{p,q}=T_L^{p,q}C_q.
$$

Normalize all `C_p` and `TC` by one common center norm without changing `TL` or `TR`.

**Step 7: Verify GREEN and refactor**

Run the focused tests, then all `tests/test_vuletta.py`. Keep the tensor-index loops explicit where they encode copy-tensor structure.

**Step 8: Commit**

```bash
git add pyqed/_vuletta/state.py pyqed/_vuletta/__init__.py tests/test_vuletta.py
git commit -m "feat: add conditional-canonical LETTA gauge"
```

---

### Task 2: Horizontal, whitened tangent coordinates

**Files:**
- Modify: `pyqed/_vuletta/gradients.py`
- Modify: `pyqed/_vuletta/__init__.py`
- Test: `tests/test_vuletta.py`

**Step 1: Write failing tangent-coordinate tests**

Specify a `ConditionalTangentData` API containing the canonical state, conditioned complements, right metric factors, reduced gradient, tensor direction, residual, and reduced dimension.

Tests must verify:

```python
for q in range(d):
    horizontal = sum(
        canonical.TL[:, p, q, :].conj().T @ direction[:, p, q, :]
        for p in range(d)
    )
    np.testing.assert_allclose(horizontal, 0.0, atol=1.0e-10)
```

and:

- reduced complex dimension is $d(d-1)D^2$;
- real models retain real directions;
- `real(vdot(gradient, direction)) == -residual**2` for the descent direction;
- the reduced residual and physical tangent step agree with the dense Gram oracle at small `d=2`, `D=1,2` to tolerance;
- gauge-transformed inputs yield the same energy slope and residual.

**Step 2: Run tests and verify RED**

Expected: import or missing-symbol failure for `conditional_tangent_direction`.

**Step 3: Construct conditioned orthogonal complements**

For each $q$, stack

$$
M_q[(p,\alpha),\beta]=T_L^{p,q}_{\alpha\beta}.
$$

Use a complete QR/SVD to form an orthonormal complement $N_q$ to the $D$ columns of $M_q$.

**Step 4: Whiten the surviving right metric**

Compute the diagonal conditioned blocks $\rho_q$ of the right transfer fixed point of `TL`. Parameterize

$$
X_q=N_q Z_q\rho_q^{-1/2}.
$$

Pull the analytic tensor gradient back to $Z_q$:

$$
g_{Z_q}=N_q^\dagger g_q\rho_q^{-1/2}.
$$

Set $\delta Z_q=-g_{Z_q}$ and reconstruct the tensor descent direction. Use only supported eigenvectors when rank-deficiency is explicitly allowed.

**Step 5: Verify GREEN and refactor**

Run every new test and the existing analytic-gradient, gauge-nullity, and dense-metric tests.

**Step 6: Commit**

```bash
git add pyqed/_vuletta/gradients.py pyqed/_vuletta/__init__.py tests/test_vuletta.py
git commit -m "feat: add horizontal conditional LETTA tangent"
```

---

### Task 3: Make conditional-canonical optimization the default

**Files:**
- Modify: `pyqed/_vuletta/solver.py`
- Modify: `pyqed/_vuletta/README.md`
- Test: `tests/test_vuletta.py`
- Test: `tests/test_vuletta_example.py`

**Step 1: Write failing solver tests**

Require:

- `VULETTAOptions().update_method == "conditional_canonical"`;
- the result exposes `canonical_state` and `canonical_residual_norm`;
- accepted iterates and final state satisfy conditional canonical invariants;
- convergence uses the reduced horizontal residual;
- the legacy dense `natural_gradient` and `lbfgs` paths remain callable as reference paths;
- the existing TFIM $D=1,2,3$ energy and observable checks remain valid.

**Step 2: Run tests and verify RED**

Expected: default method and result fields do not yet exist.

**Step 3: Implement the conditional-canonical iteration**

At initialization and after every accepted step:

1. call `conditional_canonicalize`;
2. evaluate the exact analytic energy gradient at `canonical.state`;
3. construct the horizontal whitened descent;
4. cap the reduced-coordinate step norm;
5. use Armijo backtracking with the exact thermodynamic objective;
6. recanonicalize the accepted tensor;
7. converge only when the reduced residual and canonical residual are both below tolerance.

Do not call `tangent_gram_matrix` in this active path. Keep the dense path under `update_method="natural_gradient"` as an oracle.

**Step 4: Update diagnostics and documentation**

Record reduced tangent dimension, canonical residual, step size, and energy change per iteration. Document the conditional isometries, center relation, horizontal condition, rank assumption, and distinction from the ordinary MPS center-eigenproblem.

**Step 5: Verify GREEN and refactor**

Run solver tests, example tests, and a monkeypatch test that makes `tangent_gram_matrix` raise if called from the conditional path.

**Step 6: Commit**

```bash
git add pyqed/_vuletta/solver.py pyqed/_vuletta/README.md tests/test_vuletta.py tests/test_vuletta_example.py
git commit -m "feat: use conditional gauge in VULETTA solver"
```

---

### Task 4: Cross-model observables and runtime benchmark

**Files:**
- Create: `pyqed/_vuletta/benchmarks/__init__.py`
- Create: `pyqed/_vuletta/benchmarks/one_dimensional_models.py`
- Create: `tests/test_vuletta_benchmark.py`
- Modify: `pyqed/_vuletta/README.md`

**Step 1: Write failing benchmark-schema tests**

Define immutable rows with fields:

```python
model, method, bond_dim, transfer_bond_dim, tensor_entries,
energy, reference_energy, energy_error, observables,
converged, iterations, residual, runtime_seconds
```

Tests must use tiny dimensions/iteration limits and verify deterministic row ordering, finite observables, nonnegative runtimes, correct tensor-entry counts, and both LETTA and VUMPS methods. Never assert one runtime is faster.

**Step 2: Run tests and verify RED**

Expected: benchmark module does not exist.

**Step 3: Implement model definitions**

Provide:

- TFIM $H=-ZZ-gX$ at $J=1$, $g=1.5$, with analytic energy, $\langle X\rangle$, and $\langle ZZ\rangle$;
- antiferromagnetic Heisenberg $H=\mathbf S_i\cdot\mathbf S_{i+1}$, with exact energy $1/4-\ln2$, $\langle S_z\rangle=0$, and SU(2) reference $\langle S_zS_z\rangle=(1/4-\ln2)/3$.

**Step 4: Implement deterministic benchmark execution**

Use `perf_counter`, fixed seeds, one warm-up excluded from timing when repeats exceed one, median runtime over measured repeats, LETTA continuation in increasing `D`, and independent VUMPS solves. Default dimensions:

```python
letta_bond_dimensions=(1, 2, 3)
mps_bond_dimensions=(1, 2, 3, 4, 6)
```

The report must mark matched transfer dimensions explicitly (`chi == 2*D_letta`).

**Step 5: Add CLI and Markdown/CSV output**

Support an executable module that prints compact observable/error/runtime tables and optionally writes CSV. Do not write output during import or unit tests.

**Step 6: Verify GREEN and refactor**

Run `tests/test_vuletta_benchmark.py`, then the VULETTA/VUMPS example suites.

**Step 7: Commit**

```bash
git add pyqed/_vuletta/benchmarks tests/test_vuletta_benchmark.py pyqed/_vuletta/README.md
git commit -m "bench: compare canonical VULETTA and VUMPS"
```

---

### Task 5: Full benchmark run and final verification

**Files:**
- Create: `pyqed/_vuletta/benchmarks/results/one_dimensional_models.csv`
- Create: `pyqed/_vuletta/benchmarks/results/one_dimensional_models.md`
- Modify: `pyqed/_vuletta/README.md`

**Step 1: Run the requested benchmark**

Use one BLAS/OpenMP thread and run TFIM plus Heisenberg for LETTA `D=1,2,3` and VUMPS `D=1,2,3,4,6`, with fixed seeds and enough iterations for the reported convergence target.

**Step 2: Inspect convergence before accepting timings**

If a run does not converge, report it as nonconverged rather than hiding it or comparing its timing as a converged solve. Increase iteration limits only through explicit benchmark parameters recorded in the output.

**Step 3: Write reproducible result artifacts**

Save exact command/configuration, Python/NumPy/SciPy versions, model parameters, observables, errors, residuals, iterations, and runtime. Make clear that wall times are machine-specific.

**Step 4: Run final verification**

Run:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_vuletta.py \
  tests/test_vuletta_example.py \
  tests/test_vuletta_benchmark.py \
  tests/test_vumps.py \
  tests/test_vumps_example.py
```

Also run the benchmark CLI from a clean process and inspect both output artifacts.

**Step 5: Commit**

```bash
git add pyqed/_vuletta/benchmarks/results pyqed/_vuletta/README.md
git commit -m "docs: record 1D VULETTA benchmark"
```

---

### Task 6: Integrate without disturbing the user's checkout

**Files:**
- Review all feature-branch changes

**Step 1: Verify the original checkout still contains only its pre-existing unrelated changes**

Run `git status --short` in `/Users/shuoyihu/Documents/GitHub/pyqed`.

**Step 2: Follow the Finishing a Development Branch skill**

Present or perform the safe fast-forward integration appropriate to the user's explicit “deploy” request. Do not overwrite `_letta_one_site_opt` or other unrelated working-tree changes.

**Step 3: Re-run focused verification from the integrated checkout**

Use the same Python 3.12 pytest runner and thread limits as the feature worktree.

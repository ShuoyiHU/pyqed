import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyqed._letta_one_site_opt.cbe as cbe_module


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    LatticeLETTA,
    letta_dmrg,
)
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo,
)
from pyqed._letta_one_site_opt.cbe import (
    _directional_one_site_metric_trim,
    _directional_one_site_trim,
    _streamed_shrewd_final_tensor,
    _streamed_shrewd_preselection_tensor,
    _streamed_shrewd_weighted_preselection_tensor,
    cbe_cached_mpo_sweep,
    embed_cbe_pair,
    exact_missing_pair_direction,
    matrix_free_missing_pair_direction,
    metric_trim_pair,
    metric_orthogonal_complement,
    select_cbe_directions,
    select_shrewd_cbe_directions,
    streamed_shrewd_cbe_selection,
)
from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_one_site_opt.operators import LatticeMPO
from pyqed._letta_two_site_opt import (
    IdentityPairEnvironmentCache,
    LETTAPairEnvironmentCache,
    LETTAPairLayout,
    LETTATwoSiteOptions,
    letta_two_site_dmrg,
)
from pyqed._letta_one_site_opt.benchmarks.cbe_convergence import (
    SOLVERS,
    run_comparison,
)
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
    make_shared_initial_state,
)


def _pair_problem(
    seed=401,
    *,
    shape=(2, 2),
    bond_dim=2,
    left_site=0,
    real=True,
):
    state = LatticeLETTA.random(
        shape,
        physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        real=real,
    )
    mpo = transverse_field_ising_mpo(shape, coupling=0.8, field=1.3)
    layout = LETTAPairLayout.from_state(state, left_site)
    hamiltonian_cache = LETTAPairEnvironmentCache(state, mpo)
    metric_cache = IdentityPairEnvironmentCache(state)
    hamiltonian_left = hamiltonian_cache.build_left_environments()
    hamiltonian_right = hamiltonian_cache.build_right_environments()
    metric_left = metric_cache.build_left_environments()
    metric_right = metric_cache.build_right_environments()
    pair_metric = metric_cache.effective_pair_metric(
        metric_left[left_site], metric_right[left_site + 2], layout
    )

    def pair_action(vector):
        return hamiltonian_cache.effective_pair_action(
            hamiltonian_left[left_site],
            hamiltonian_right[left_site + 2],
            layout,
            vector,
        )

    return state, layout, pair_action, pair_metric


def _streamed_pair_problem(seed=501, *, bond_dim=2, real=True):
    state = LatticeLETTA.random(
        (2, 3),
        physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        real=real,
    )
    mpo = transverse_field_ising_mpo(
        (2, 3), coupling=0.8, field=1.3
    )
    layout = LETTAPairLayout.from_state(state, 1)
    cache = LETTAPairEnvironmentCache(state, mpo)
    left = cache.build_left_environments()[layout.left_site]
    right = cache.build_right_environments()[layout.left_site + 2]
    return state, layout, cache, left, right


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_streamed_shrewd_selection_uses_one_site_parent_spaces(direction):
    state, layout, cache, left, right = _streamed_pair_problem()

    selection = streamed_shrewd_cbe_selection(
        cache,
        left,
        right,
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        expansion_dimension=1,
        preselection_dimension=5,
        direction=direction,
        tolerance=1.0e-12,
    )

    assert selection.selector == "shrewd"
    assert selection.preselection_dimension <= 5
    assert selection.left_direction.shape == (
        state.tensors[layout.left_site].shape[:-1] + (1,)
    )
    assert selection.right_direction.shape == (
        (1,) + state.tensors[layout.left_site + 1].shape[1:]
    )
    assert selection.missing_norm > 0.0
    assert 0.0 <= selection.captured_weight <= 1.0
    assert selection.pair_action_count == 0
    assert selection.pair_metric_count == 0
    assert selection.merged_pair_count == 0
    assert selection.preselection_output_size <= max(
        state.tensors[layout.left_site].size
        * cache.mpo.factors[layout.left_site].shape[1],
        state.tensors[layout.left_site + 1].size
        * cache.mpo.factors[layout.left_site + 1].shape[0],
    )


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_streamed_shrewd_selection_returns_zero_for_zero_complement(direction):
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=503
    )
    width = 3
    factors = [
        np.zeros(
            (
                1 if site == 0 else width,
                1 if site == state.nsites - 1 else width,
                2,
                2,
            )
        )
        for site in range(state.nsites)
    ]
    cache = LETTAPairEnvironmentCache(
        state,
        LatticeMPO(factors, lattice_shape=state.lattice_shape),
    )
    layout = LETTAPairLayout.from_state(state, 1)
    left = cache.build_left_environments()[layout.left_site]
    right = cache.build_right_environments()[layout.left_site + 2]

    selection = streamed_shrewd_cbe_selection(
        cache,
        left,
        right,
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        expansion_dimension=1,
        preselection_dimension=width,
        direction=direction,
    )

    assert selection.sector_ranks == (0,)
    assert selection.missing_norm == 0.0
    assert selection.preselection_dimension == 0
    assert not np.any(selection.left_direction)
    assert not np.any(selection.right_direction)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_sparse_streamed_selector_contractions_equal_dense(direction):
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=2, seed=504
    )
    mpo = transverse_field_ising_mpo(
        (2, 3), coupling=0.8, field=1.3
    )
    layout = LETTAPairLayout.from_state(state, 1)
    sparse_cache = LETTAPairEnvironmentCache(
        state, mpo, use_sparse_mpo=True
    )
    dense_cache = LETTAPairEnvironmentCache(
        state, mpo, use_sparse_mpo=False
    )
    sparse_left = sparse_cache.build_left_environments()[layout.left_site]
    sparse_right = sparse_cache.build_right_environments()[
        layout.left_site + 2
    ]
    dense_left = dense_cache.build_left_environments()[layout.left_site]
    dense_right = dense_cache.build_right_environments()[
        layout.left_site + 2
    ]
    left_tensor = state.tensors[layout.left_site]
    right_tensor = state.tensors[layout.left_site + 1]

    sparse_preselection = _streamed_shrewd_preselection_tensor(
        sparse_cache,
        sparse_left,
        sparse_right,
        layout,
        left_tensor,
        right_tensor,
        direction,
    )
    dense_preselection = _streamed_shrewd_preselection_tensor(
        dense_cache,
        dense_left,
        dense_right,
        layout,
        left_tensor,
        right_tensor,
        direction,
    )
    np.testing.assert_allclose(
        sparse_preselection, dense_preselection, atol=1.0e-11
    )

    rng = np.random.default_rng(505)
    width = 2
    middle_width = mpo.factors[layout.left_site].shape[1]
    if direction == "rl":
        weighted_half = rng.normal(
            size=(left_tensor.shape[-1], middle_width, width)
        )
    else:
        weighted_half = rng.normal(
            size=(width, right_tensor.shape[0], middle_width)
        )
    sparse_weighted = _streamed_shrewd_weighted_preselection_tensor(
        sparse_cache,
        sparse_left,
        sparse_right,
        layout,
        left_tensor,
        right_tensor,
        weighted_half,
        direction,
    )
    dense_weighted = _streamed_shrewd_weighted_preselection_tensor(
        dense_cache,
        dense_left,
        dense_right,
        layout,
        left_tensor,
        right_tensor,
        weighted_half,
        direction,
    )
    np.testing.assert_allclose(sparse_weighted, dense_weighted, atol=1.0e-11)

    if direction == "rl":
        preselected = rng.normal(size=left_tensor.shape[:-1] + (width,))
    else:
        preselected = rng.normal(size=(width,) + right_tensor.shape[1:])
    sparse_final = _streamed_shrewd_final_tensor(
        sparse_cache,
        sparse_left,
        sparse_right,
        layout,
        left_tensor,
        right_tensor,
        preselected,
        direction,
    )
    dense_final = _streamed_shrewd_final_tensor(
        dense_cache,
        dense_left,
        dense_right,
        layout,
        left_tensor,
        right_tensor,
        preselected,
        direction,
    )
    np.testing.assert_allclose(sparse_final, dense_final, atol=1.0e-11)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_one_site_metric_trim_improves_over_euclidean_trim(direction):
    rng = np.random.default_rng(506)
    parent_dimension = 7
    expanded_dimension = 4
    retained_dimension = 2
    if direction == "lr":
        left_tensor = rng.normal(
            size=(parent_dimension, expanded_dimension)
        )
        right_tensor = np.eye(expanded_dimension)
        target = left_tensor
    else:
        left_tensor = np.eye(expanded_dimension)
        right_tensor = rng.normal(
            size=(expanded_dimension, parent_dimension)
        )
        target = right_tensor
    factor = rng.normal(size=(target.size, target.size))
    dense_metric = factor.T @ factor + 0.1 * np.eye(target.size)
    metric = BlockDiagonalMetric(
        target.size,
        (dense_metric,),
        (np.arange(target.size),),
    )

    euclidean = _directional_one_site_trim(
        left_tensor,
        right_tensor,
        bond_dimension=retained_dimension,
        direction=direction,
    )
    refined = _directional_one_site_metric_trim(
        left_tensor,
        right_tensor,
        metric,
        bond_dimension=retained_dimension,
        direction=direction,
        tolerance=1.0e-11,
        max_iterations=4,
        metric_tolerance=1.0e-12,
    )

    euclidean_approximation = euclidean.left_tensor @ euclidean.right_tensor
    euclidean_difference = (target - euclidean_approximation).reshape(-1)
    euclidean_loss = float(
        np.real(
            np.vdot(
                euclidean_difference,
                dense_metric @ euclidean_difference,
            )
        )
    )
    assert refined.loss <= euclidean_loss + 1.0e-9
    assert refined.iterations > 0


class _PairHamiltonianPathForbidden(LETTAPairEnvironmentCache):
    def effective_pair_action(self, left, right, layout, vector):
        raise AssertionError("strict shrewd CBE called the pair action")


class _PairMetricPathForbidden(IdentityPairEnvironmentCache):
    def effective_pair_metric(self, left, right, layout):
        raise AssertionError("strict shrewd CBE built the pair metric")


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_active_shrewd_sweep_never_enters_pair_space(direction, monkeypatch):
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=1, seed=502
    )
    mpo = transverse_field_ising_mpo(
        (2, 3), coupling=0.8, field=1.3
    )
    hamiltonian_cache = _PairHamiltonianPathForbidden(state, mpo)
    metric_cache = _PairMetricPathForbidden(state)
    hamiltonian_environments = (
        hamiltonian_cache.build_left_environments()
        if direction == "rl"
        else hamiltonian_cache.build_right_environments()
    )
    metric_environments = (
        metric_cache.build_left_environments()
        if direction == "rl"
        else metric_cache.build_right_environments()
    )

    def forbidden_merge(self, left_tensor, right_tensor):
        raise AssertionError("strict shrewd CBE formed a merged pair tensor")

    monkeypatch.setattr(LETTAPairLayout, "merge", forbidden_merge)
    updates, energy, _hamiltonian, _metric = cbe_cached_mpo_sweep(
        state,
        hamiltonian_cache,
        metric_cache,
        hamiltonian_environments,
        metric_environments,
        direction,
        LETTADMROptions(
            max_sweeps=1,
            matrix_free=True,
            cbe_enabled=True,
            cbe_selector="shrewd",
            cbe_expansion_dimension=1,
            cbe_preselection_dimension=2,
        ),
    )

    assert np.isfinite(energy)
    strict_updates = [
        update for update in updates if update.cbe_expansion_dimension > 0
    ]
    assert len(strict_updates) == state.nsites - 1
    assert all(update.cbe_selector_pair_action_count == 0 for update in strict_updates)
    assert all(update.cbe_selector_pair_metric_count == 0 for update in strict_updates)
    assert all(update.cbe_selector_merged_pair_count == 0 for update in strict_updates)
    assert all(
        update.cbe_materialized_pair_tensor is False
        for update in strict_updates
    )
    assert all(update.cbe_preselection_output_size > 0 for update in strict_updates)
    assert all(update.cbe_final_output_size >= 0 for update in strict_updates)
    assert all(
        update.cbe_trim_method == "one-site-metric-als"
        for update in strict_updates
    )
    assert all(
        update.cbe_materialized_pair_metric is False
        for update in strict_updates
    )
    assert all(
        update.cbe_materialized_tangent_jacobian is False
        for update in strict_updates
    )


def test_strict_streamed_residual_vanishes_on_represented_eigenstate():
    state = LatticeLETTA.random(
        (1, 4), physical_dim=2, bond_dim=4, seed=518
    )
    mpo = transverse_field_ising_mpo(
        (1, 4), coupling=0.8, field=1.3
    )
    exact = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=4,
        options=LETTATwoSiteOptions(
            max_sweeps=8,
            tolerance=1.0e-12,
            matrix_free=True,
            split_method="metric-als",
            one_site_polish_sweeps=0,
        ),
    )

    result = letta_dmrg(
        mpo,
        state=exact.state,
        options=LETTADMROptions(
            max_sweeps=1,
            tolerance=1.0e-12,
            matrix_free=True,
            cbe_enabled=True,
            cbe_selector="shrewd",
            cbe_expansion_dimension=1,
        ),
    )

    strict_updates = [
        update
        for sweep in result.history
        for update in sweep.updates
        if update.cbe_expansion_dimension > 0
    ]
    assert strict_updates
    assert max(update.cbe_missing_norm for update in strict_updates) < 1.0e-8
    assert all(update.cbe_selector_pair_action_count == 0 for update in strict_updates)
    assert all(update.cbe_selector_pair_metric_count == 0 for update in strict_updates)
    assert all(update.cbe_selector_merged_pair_count == 0 for update in strict_updates)


def test_cbe_options_are_disabled_by_default():
    options = LETTADMROptions()

    assert options.cbe_enabled is False
    assert options.cbe_expansion_dimension == 1
    assert options.cbe_selection_tolerance > 0.0
    assert options.cbe_refinement_max_iterations > 0
    assert options.cbe_selector == "exact"
    assert options.cbe_preselection_dimension is None
    assert options.cbe_projection_tolerance > 0.0
    assert options.cbe_projection_max_iterations > 0
    assert options.cbe_baseline_guard_fraction == pytest.approx(0.2)


@pytest.mark.parametrize("selector", ["exact", "shrewd"])
def test_trimmed_cbe_update_respects_ordinary_one_site_gain_guard(
    selector,
):
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=517
    )
    result = letta_dmrg(
        transverse_field_ising_mpo((2, 2), coupling=0.8, field=1.3),
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            tolerance=1.0e-12,
            cbe_enabled=True,
            cbe_selector=selector,
            cbe_expansion_dimension=1,
        ),
    )
    attempted = [
        update
        for sweep in result.history
        for update in sweep.updates
        if update.cbe_expanded_energy is not None
    ]
    assert attempted
    for update in attempted:
        assert update.cbe_baseline_energy is not None
        assert update.cbe_baseline_allowance is not None
        assert update.energy <= (
            update.cbe_baseline_energy
            + update.cbe_baseline_allowance
            + 1.0e-9
        )
        assert update.cbe_baseline_selected is update.cbe_fallback


def test_metric_orthogonal_complement_uses_supported_nonidentity_metric():
    metric = np.diag([4.0, 1.0, 0.0])
    jacobian = np.asarray([[1.0], [1.0], [0.0]])
    residual = np.asarray([1.0, 0.0, 8.0])

    result = metric_orthogonal_complement(
        residual,
        jacobian,
        metric,
        tolerance=1.0e-12,
    )

    np.testing.assert_allclose(
        jacobian.conj().T @ metric @ result.vector,
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.vector[2], 0.0, atol=1.0e-12)
    assert result.metric_rank == 2
    assert result.tangent_rank == 1


def test_exact_missing_pair_direction_is_metric_orthogonal_to_tangent():
    state, layout, pair_action, pair_metric = _pair_problem()

    result = exact_missing_pair_direction(
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        pair_action,
        pair_metric,
        metric_tolerance=1.0e-11,
    )

    assert result.vector.shape == (int(np.prod(layout.merged_shape)),)
    assert result.metric_rank > 0
    assert 0 <= result.tangent_rank <= result.metric_rank
    assert result.missing_norm >= 0.0
    assert np.isfinite(result.missing_norm)
    assert result.tangent_overlap_norm <= 2.0e-9


class _NoDenseBlockMetric:
    def __init__(self, metric):
        self.blocks = metric.blocks
        self.indices = metric.indices
        self.size = metric.size
        self.shape = metric.shape
        self.dtype = metric.dtype

    def __matmul__(self, value):
        value = np.asarray(value)
        result = np.zeros_like(value, dtype=np.result_type(self.dtype, value))
        for block, indices in zip(self.blocks, self.indices):
            result[indices] = block @ value[indices]
        return result

    def to_dense(self):
        raise AssertionError("the shrewd selector must not densify the pair metric")


def _metric_norm(vector, metric):
    return float(np.sqrt(max(0.0, np.real(np.vdot(vector, metric @ vector)))))


def test_restricted_residual_projection_matches_dense_metric_oracle():
    assert hasattr(cbe_module, "_metric_projected_restricted_residual")
    rng = np.random.default_rng(519)
    dimension = 9
    tangent_indices = np.array([0, 1, 3, 6])
    factor = rng.normal(size=(dimension, dimension))
    dense_metric = factor.T @ factor + 0.5 * np.eye(dimension)
    metric = BlockDiagonalMetric(
        dimension,
        (dense_metric,),
        (np.arange(dimension),),
    )
    covector = rng.normal(size=dimension)

    projected, norm = cbe_module._metric_projected_restricted_residual(
        covector,
        metric,
        tangent_indices,
        metric_tolerance=1.0e-12,
    )

    raised = np.linalg.solve(dense_metric, covector)
    tangent = np.eye(dimension)[:, tangent_indices]
    coefficients = np.linalg.solve(
        tangent.T @ dense_metric @ tangent,
        tangent.T @ dense_metric @ raised,
    )
    expected = raised - tangent @ coefficients
    np.testing.assert_allclose(projected, expected, atol=1.0e-11)
    np.testing.assert_allclose(
        tangent.T @ dense_metric @ projected,
        0.0,
        atol=1.0e-11,
    )
    assert norm == pytest.approx(_metric_norm(expected, metric))


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_streamed_metric_projected_final_matches_dense_pair_oracle(direction):
    assert hasattr(cbe_module, "_metric_projected_streamed_final")
    state, layout, cache, hamiltonian_left, hamiltonian_right = (
        _streamed_pair_problem(seed=520, bond_dim=2)
    )
    metric_cache = IdentityPairEnvironmentCache(state)
    metric_left = metric_cache.build_left_environments()[layout.left_site]
    metric_right = metric_cache.build_right_environments()[layout.left_site + 2]
    left_tensor = state.tensors[layout.left_site]
    right_tensor = state.tensors[layout.left_site + 1]
    rng = np.random.default_rng(521)
    width = 2
    if direction == "rl":
        preselected = rng.normal(size=left_tensor.shape[:-1] + (width,))
    else:
        preselected = rng.normal(size=(width,) + right_tensor.shape[1:])
    energy = state.expectation(cache.mpo)

    final_matrix, missing_norm, restricted_size = (
        cbe_module._metric_projected_streamed_final(
            cache,
            hamiltonian_left,
            hamiltonian_right,
            metric_cache,
            metric_left,
            metric_right,
            layout,
            left_tensor,
            right_tensor,
            preselected,
            direction,
            energy=energy,
            metric_tolerance=1.0e-12,
        )
    )

    pair_metric = metric_cache.effective_pair_metric(
        metric_left, metric_right, layout
    ).to_dense()
    theta = layout.merge(left_tensor, right_tensor).reshape(-1)
    residual = cache.effective_pair_action(
        hamiltonian_left, hamiltonian_right, layout, theta
    ) - energy * (pair_metric @ theta)
    if direction == "rl":
        expanded_left = np.concatenate([left_tensor, preselected], axis=-1)
        variable_shape = (expanded_left.shape[-1],) + right_tensor.shape[1:]

        def merge_variable(vector):
            return layout.merge(expanded_left, vector.reshape(variable_shape))

        tangent_mask = np.zeros(variable_shape, dtype=bool)
        tangent_mask[: left_tensor.shape[-1]] = True
    else:
        expanded_right = np.concatenate([right_tensor, preselected], axis=0)
        variable_shape = left_tensor.shape[:-1] + (expanded_right.shape[0],)

        def merge_variable(vector):
            return layout.merge(vector.reshape(variable_shape), expanded_right)

        tangent_mask = np.zeros(variable_shape, dtype=bool)
        tangent_mask[..., : right_tensor.shape[0]] = True
    basis = np.eye(int(np.prod(variable_shape)))
    jacobian = np.column_stack(
        [merge_variable(column).reshape(-1) for column in basis]
    )
    restricted_metric = jacobian.conj().T @ pair_metric @ jacobian
    restricted_covector = jacobian.conj().T @ residual
    raised = np.linalg.pinv(restricted_metric, rcond=1.0e-12) @ restricted_covector
    tangent_indices = np.flatnonzero(tangent_mask.reshape(-1))
    tangent = np.eye(raised.size)[:, tangent_indices]
    tangent_metric = tangent.conj().T @ restricted_metric @ tangent
    coefficients = np.linalg.pinv(tangent_metric, rcond=1.0e-12) @ (
        tangent.conj().T @ restricted_metric @ raised
    )
    expected = raised - tangent @ coefficients
    if direction == "rl":
        expected_matrix = expected.reshape(variable_shape)[
            left_tensor.shape[-1] :
        ].reshape(width, -1)
    else:
        expected_matrix = expected.reshape(variable_shape)[
            ..., right_tensor.shape[0] :
        ].reshape(-1, width)

    assert restricted_size == int(np.prod(variable_shape))
    np.testing.assert_allclose(final_matrix, expected_matrix, atol=2.0e-9)
    np.testing.assert_allclose(
        missing_norm,
        np.sqrt(max(0.0, np.real(np.vdot(expected, restricted_metric @ expected)))),
        atol=2.0e-9,
    )


def _dense_matrix_to_exact_mpo(matrix, nsites, physical_dim, lattice_shape):
    tensor = np.asarray(matrix).reshape((physical_dim,) * (2 * nsites))
    interleaved = tuple(
        axis for site in range(nsites) for axis in (site, nsites + site)
    )
    remainder = tensor.transpose(interleaved).reshape(
        (physical_dim**2,) * nsites
    )
    factors = []
    left_rank = 1
    for _site in range(nsites - 1):
        matrix = remainder.reshape(left_rank * physical_dim**2, -1)
        vectors, values, adjoint = np.linalg.svd(matrix, full_matrices=False)
        cutoff = np.finfo(float).eps * max(matrix.shape) * values[0]
        rank = int(np.count_nonzero(values > cutoff))
        core = vectors[:, :rank].reshape(
            left_rank, physical_dim, physical_dim, rank
        )
        factors.append(core.transpose(0, 3, 1, 2))
        remainder = values[:rank, None] * adjoint[:rank]
        left_rank = rank
    core = remainder.reshape(left_rank, physical_dim, physical_dim, 1)
    factors.append(core.transpose(0, 3, 1, 2))
    return LatticeMPO(factors, lattice_shape=lattice_shape)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_metric_projected_strict_selection_is_mpo_representation_invariant(
    direction,
):
    state = LatticeLETTA.random(
        (1, 4), physical_dim=4, bond_dim=2, seed=522
    )
    direct_sum_mpo = build_model(
        "fermi_hubbard", "1d", 4, t=1.0, U=4.0, mu=2.0
    ).mpo
    compact_mpo = _dense_matrix_to_exact_mpo(
        direct_sum_mpo.to_dense(max_sites=state.nsites),
        state.nsites,
        state.physical_dim,
        state.lattice_shape,
    )
    np.testing.assert_allclose(
        direct_sum_mpo.to_dense(max_sites=state.nsites),
        compact_mpo.to_dense(max_sites=state.nsites),
        atol=1.0e-10,
    )
    layout = LETTAPairLayout.from_state(state, 1)
    metric_cache = IdentityPairEnvironmentCache(state)
    metric_left = metric_cache.build_left_environments()[layout.left_site]
    metric_right = metric_cache.build_right_environments()[layout.left_site + 2]
    energy = state.expectation(compact_mpo)

    def select(mpo):
        cache = LETTAPairEnvironmentCache(state, mpo)
        left = cache.build_left_environments()[layout.left_site]
        right = cache.build_right_environments()[layout.left_site + 2]
        return streamed_shrewd_cbe_selection(
            cache,
            left,
            right,
            layout,
            state.tensors[layout.left_site],
            state.tensors[layout.left_site + 1],
            expansion_dimension=1,
            preselection_dimension=3,
            direction=direction,
            tolerance=1.0e-11,
            metric_cache=metric_cache,
            metric_left=metric_left,
            metric_right=metric_right,
            energy=energy,
            metric_tolerance=1.0e-12,
        )

    original = select(compact_mpo)
    transformed = select(direct_sum_mpo)
    if direction == "rl":
        original_factor = original.left_direction.reshape(-1, 1)
        transformed_factor = transformed.left_direction.reshape(-1, 1)
    else:
        original_factor = original.right_direction.reshape(1, -1).T
        transformed_factor = transformed.right_direction.reshape(1, -1).T
    original_factor /= np.linalg.norm(original_factor)
    transformed_factor /= np.linalg.norm(transformed_factor)

    np.testing.assert_allclose(
        original_factor @ original_factor.conj().T,
        transformed_factor @ transformed_factor.conj().T,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        original.missing_norm, transformed.missing_norm, rtol=2.0e-8, atol=2.0e-10
    )


def test_default_metric_projected_strict_cbe_converges_fermi_reference():
    model = build_model(
        "fermi_hubbard", "1d", 4, t=1.0, U=4.0, mu=2.0
    )
    initial = make_shared_initial_state(model, bond_dim=4, seed=731).letta
    exact_energy = float(
        np.linalg.eigvalsh(model.mpo.to_dense(max_sites=model.nsites))[0]
    )

    result = letta_dmrg(
        model.mpo,
        state=initial,
        options=LETTADMROptions(
            max_sweeps=12,
            tolerance=1.0e-9,
            matrix_free=True,
            use_sparse_mpo=True,
            cbe_enabled=True,
            cbe_selector="shrewd",
            cbe_expansion_dimension=1,
        ),
    )

    assert result.converged
    assert result.energy - exact_energy < 1.0e-8
    attempted = [
        update
        for sweep in result.history
        for update in sweep.updates
        if update.cbe_preselection_dimension is not None
    ]
    assert attempted
    assert max(update.cbe_preselection_dimension for update in attempted) == 6


@pytest.mark.parametrize("real", [True, False])
def test_matrix_free_missing_direction_matches_exact_oracle_without_materializing(
    real,
):
    state, layout, pair_action, pair_metric = _pair_problem(
        seed=409,
        shape=(2, 3),
        bond_dim=1,
        left_site=1,
        real=real,
    )
    exact = exact_missing_pair_direction(
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        pair_action,
        pair_metric,
        metric_tolerance=1.0e-11,
    )
    action_calls = 0

    def counted_action(vector):
        nonlocal action_calls
        action_calls += 1
        return pair_action(vector)

    guarded_metric = _NoDenseBlockMetric(pair_metric)
    shrewd = matrix_free_missing_pair_direction(
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        counted_action,
        guarded_metric,
        metric_tolerance=1.0e-11,
        projection_tolerance=1.0e-11,
        projection_max_iterations=200,
    )

    error = _metric_norm(shrewd.vector - exact.vector, guarded_metric)
    scale = max(exact.missing_norm, 1.0e-14)
    assert action_calls == 1
    assert shrewd.selector == "shrewd"
    assert shrewd.pair_action_count == 1
    assert shrewd.materialized_pair_metric is False
    assert shrewd.materialized_tangent_jacobian is False
    assert shrewd.projection_converged
    assert shrewd.projection_iterations > 0
    np.testing.assert_allclose(shrewd.energy, exact.energy, atol=1.0e-11)
    assert error / scale <= 2.0e-7
    assert shrewd.tangent_overlap_norm <= 2.0e-7


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_shrewd_two_stage_selection_tracks_exact_selector(direction):
    state, layout, pair_action, pair_metric = _pair_problem(
        seed=410,
        shape=(2, 3),
        bond_dim=1,
        left_site=1,
    )
    exact_missing = exact_missing_pair_direction(
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        pair_action,
        pair_metric,
    )
    shrewd_missing = matrix_free_missing_pair_direction(
        layout,
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
        pair_action,
        pair_metric,
        projection_tolerance=1.0e-10,
        projection_max_iterations=100,
    )
    exact_selection = select_cbe_directions(
        exact_missing,
        layout,
        pair_metric,
        expansion_dimension=1,
        direction=direction,
    )
    shrewd_selection = select_shrewd_cbe_directions(
        shrewd_missing,
        layout,
        pair_metric,
        expansion_dimension=1,
        preselection_dimension=2,
        direction=direction,
    )

    assert shrewd_selection.selector == "shrewd"
    assert shrewd_selection.preselection_dimension >= 1
    assert np.isfinite(shrewd_selection.loss)
    assert 0.0 <= shrewd_selection.captured_weight <= 1.0
    assert shrewd_selection.captured_weight >= (
        exact_selection.captured_weight - 5.0e-4
    )


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_cbe_embedding_preserves_pair_before_site_optimization(direction):
    state, layout, _pair_action, _pair_metric = _pair_problem(seed=402)
    left = state.tensors[0]
    right = state.tensors[1]
    rng = np.random.default_rng(403)
    expansion_dimension = 1
    left_direction = rng.normal(
        size=left.shape[:-1] + (expansion_dimension,)
    )
    right_direction = rng.normal(
        size=(expansion_dimension,) + right.shape[1:]
    )
    original = layout.merge(left, right)

    expanded_left, expanded_right = embed_cbe_pair(
        left,
        right,
        left_direction,
        right_direction,
        direction=direction,
    )

    assert expanded_left.shape[-1] == left.shape[-1] + expansion_dimension
    assert expanded_right.shape[0] == right.shape[0] + expansion_dimension
    np.testing.assert_allclose(
        layout.merge(expanded_left, expanded_right),
        original,
        atol=1.0e-13,
    )


def test_selection_and_metric_trim_are_finite_and_normalized():
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=1, seed=405
    )
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    layout = LETTAPairLayout.from_state(state, 1)
    hamiltonian_cache = LETTAPairEnvironmentCache(state, mpo)
    metric_cache = IdentityPairEnvironmentCache(state)
    hamiltonian_left = hamiltonian_cache.build_left_environments()
    hamiltonian_right = hamiltonian_cache.build_right_environments()
    metric_left = metric_cache.build_left_environments()
    metric_right = metric_cache.build_right_environments()
    pair_metric = metric_cache.effective_pair_metric(
        metric_left[1], metric_right[3], layout
    )

    def pair_action(vector):
        return hamiltonian_cache.effective_pair_action(
            hamiltonian_left[1],
            hamiltonian_right[3],
            layout,
            vector,
        )

    missing = exact_missing_pair_direction(
        layout,
        state.tensors[1],
        state.tensors[2],
        pair_action,
        pair_metric,
    )
    selection = select_cbe_directions(
        missing,
        layout,
        pair_metric,
        expansion_dimension=1,
        direction="lr",
    )
    expanded = layout.merge(
        *embed_cbe_pair(
            state.tensors[1],
            state.tensors[2],
            selection.left_direction,
            selection.right_direction,
            direction="lr",
        )
    )
    trim = metric_trim_pair(
        expanded,
        layout,
        pair_metric,
        bond_dimension=1,
        direction="lr",
    )
    trimmed = layout.merge(trim.left_tensor, trim.right_tensor).reshape(-1)

    assert missing.missing_norm > 1.0e-10
    assert 0.0 <= selection.captured_weight <= 1.0
    assert np.isfinite(selection.loss)
    assert np.isfinite(trim.loss)
    np.testing.assert_allclose(
        np.real(np.vdot(trimmed, pair_metric @ trimmed)),
        1.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("selector", ["exact", "shrewd"])
def test_cbe_one_sweep_keeps_bonds_fixed_and_populates_diagnostics(
    direction, selector
):
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=1, seed=404
    )
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    initial_energy = state.expectation(mpo)
    initial_bonds = state.bond_dimensions

    result = letta_dmrg(
        mpo,
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            alternate=False,
            start_direction=direction,
            matrix_free=True,
            cbe_enabled=True,
            cbe_selector=selector,
            cbe_expansion_dimension=1,
        ),
    )

    assert result.state.bond_dimensions == initial_bonds
    assert result.energy <= initial_energy + 1.0e-9
    np.testing.assert_allclose(
        result.energy,
        result.state.expectation(mpo),
        atol=1.0e-10,
    )
    cbe_updates = [
        update
        for update in result.history[0].updates
        if update.cbe_expansion_dimension > 0
    ]
    assert len(cbe_updates) == state.nsites - 1
    assert all(update.cbe_pair_dimension > 0 for update in cbe_updates)
    assert all(update.cbe_missing_norm >= 0.0 for update in cbe_updates)
    assert all(np.isfinite(update.cbe_trim_loss) for update in cbe_updates)
    assert all(update.cbe_selector == selector for update in cbe_updates)
    expected_pair_actions = 1 if selector == "exact" else 0
    assert all(
        update.cbe_selector_pair_action_count == expected_pair_actions
        for update in cbe_updates
    )
    if selector == "shrewd":
        assert all(
            update.cbe_preselection_dimension >= 1
            for update in cbe_updates
            if not update.cbe_fallback
        )
        assert all(
            update.cbe_materialized_pair_metric is False
            for update in cbe_updates
        )
        assert all(
            update.cbe_materialized_tangent_jacobian is False
            for update in cbe_updates
        )
    else:
        assert all(
            update.cbe_materialized_pair_metric is True
            for update in cbe_updates
        )
        assert all(
            update.cbe_materialized_tangent_jacobian is True
            for update in cbe_updates
        )


def test_cbe_option_validation_and_scope_errors_are_explicit():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=406
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.8, field=1.3)

    with pytest.raises(ValueError, match="cbe_expansion_dimension"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                cbe_expansion_dimension=0,
            ),
        )
    with pytest.raises(ValueError, match="cbe_selector"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                cbe_selector="random",
            ),
        )
    with pytest.raises(ValueError, match="cbe_preselection_dimension"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                cbe_preselection_dimension=0,
            ),
        )
    with pytest.raises(ValueError, match="cbe_projection_tolerance"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                cbe_projection_tolerance=0.0,
            ),
        )
    with pytest.raises(ValueError, match="cbe_projection_max_iterations"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                cbe_projection_max_iterations=0,
            ),
        )
    for invalid_fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="cbe_baseline_guard_fraction"):
            letta_dmrg(
                mpo,
                state=state,
                options=LETTADMROptions(
                    max_sweeps=1,
                    cbe_enabled=True,
                    cbe_baseline_guard_fraction=invalid_fraction,
                ),
            )
    with pytest.raises(ValueError, match="site-granularity"):
        letta_dmrg(
            mpo,
            state=state,
            options=LETTADMROptions(
                max_sweeps=1,
                cbe_enabled=True,
                environment_granularity="column",
            ),
        )
    with pytest.raises(ValueError, match="MPO Hamiltonian"):
        letta_dmrg(
            mpo.to_dense(),
            state=state,
            options=LETTADMROptions(max_sweeps=1, cbe_enabled=True),
        )


def test_explicitly_disabled_cbe_matches_legacy_one_site_path():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=407
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.8, field=1.3)
    common = dict(
        max_sweeps=2,
        tolerance=1.0e-14,
        matrix_free=True,
        use_sparse_mpo=True,
    )

    legacy = letta_dmrg(
        mpo,
        state=state,
        options=LETTADMROptions(**common),
    )
    disabled = letta_dmrg(
        mpo,
        state=state,
        options=LETTADMROptions(**common, cbe_enabled=False),
    )

    np.testing.assert_allclose(disabled.energy, legacy.energy, atol=1.0e-13)
    for actual, expected in zip(disabled.state.tensors, legacy.state.tensors):
        np.testing.assert_allclose(actual, expected, atol=1.0e-13)


def test_cbe_benchmark_compares_exact_and_shrewd_from_one_initial_state():
    report = run_comparison(
        shape=(2, 2),
        bond_dim=1,
        expansion_dimension=1,
        max_sweeps=1,
        seed=408,
        exact_max_sites=4,
    )

    assert tuple(record["solver"] for record in report["records"]) == SOLVERS
    assert report["initial_state_fingerprint"]
    assert all(
        record["initial_state_fingerprint"]
        == report["initial_state_fingerprint"]
        for record in report["records"]
    )
    assert all(np.isfinite(record["energy"]) for record in report["records"])
    exact_cbe = next(
        record
        for record in report["records"]
        if record["solver"] == "letta_cbe_exact"
    )
    shrewd_cbe = next(
        record
        for record in report["records"]
        if record["solver"] == "letta_cbe_shrewd"
    )
    assert exact_cbe["cbe_updates"] == 3
    assert shrewd_cbe["cbe_updates"] == 3
    assert exact_cbe["selector"] == "exact"
    assert shrewd_cbe["selector"] == "shrewd"
    assert exact_cbe["mean_selector_pair_actions"] == 1.0
    assert shrewd_cbe["mean_selector_pair_actions"] == 0.0
    assert shrewd_cbe["mean_selector_pair_metrics"] == 0.0
    assert shrewd_cbe["mean_selector_merged_pairs"] == 0.0
    assert shrewd_cbe["materialized_pair_tensor"] is False
    assert exact_cbe["materialized_pair_metric"] is True
    assert exact_cbe["materialized_tangent_jacobian"] is True
    assert shrewd_cbe["materialized_pair_metric"] is False
    assert shrewd_cbe["materialized_tangent_jacobian"] is False
    assert shrewd_cbe["mean_projection_iterations"] >= 0.0
    assert shrewd_cbe["mean_missing_norm"] >= 0.0
    assert 0.0 <= shrewd_cbe["mean_captured_weight"] <= 1.0
    assert exact_cbe["cbe_baseline_selected"] >= 0
    assert shrewd_cbe["cbe_baseline_selected"] >= 0
    assert exact_cbe["mean_cbe_baseline_allowance"] >= 0.0
    assert shrewd_cbe["mean_cbe_baseline_allowance"] >= 0.0


def test_cbe_comparison_script_is_click_runnable(tmp_path):
    script = (
        _REPOSITORY_ROOT
        / "pyqed"
        / "_letta_one_site_opt"
        / "benchmarks"
        / "cbe_convergence.py"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "NUMBA_CACHE_DIR": str(tmp_path / "numba"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--shape",
            "2x2",
            "--bond-dim",
            "1",
            "--expansion-dimension",
            "1",
            "--max-sweeps",
            "1",
            "--seed",
            "408",
            "--exact-max-sites",
            "4",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "solver" in completed.stdout
    assert "one_site" in completed.stdout
    assert "letta_cbe_exact" in completed.stdout
    assert "letta_cbe_shrewd" in completed.stdout
    assert "two_site" in completed.stdout
    assert "baseline" in completed.stdout


def test_weighted_shrewd_cbe_reaches_two_site_accuracy_on_reference_case():
    report = run_comparison(
        shape=(2, 3),
        bond_dim=2,
        expansion_dimension=1,
        max_sweeps=4,
        seed=732,
    )
    records = {record["solver"]: record for record in report["records"]}
    shrewd_error = abs(records["letta_cbe_shrewd"]["energy_error"])
    one_site_error = abs(records["one_site"]["energy_error"])
    two_site_error = abs(records["two_site"]["energy_error"])

    assert shrewd_error < one_site_error - 1.0e-3
    assert shrewd_error <= two_site_error + 1.0e-8


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [str(Path(__file__).resolve()), "-q", "-p", "no:cacheprovider"]
        )
    )

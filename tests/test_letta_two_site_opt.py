import numpy as np

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo,
)
from pyqed._letta_two_site_opt import (
    IdentityPairEnvironmentCache,
    LETTAPairLayout,
    LETTAPairEnvironmentCache,
    LETTATwoSiteOptions,
    conditional_svd_split,
    energy_refine_split,
    letta_two_site_dmrg,
    metric_als_refine,
)
from pyqed._letta_two_site_opt._letta_for_2d import (
    transverse_field_ising_mpo as two_site_2d_ising_mpo,
)
from pyqed._letta_two_site_opt._letta_for_3d import (
    snake_letta_state as two_site_snake_letta_state,
    transverse_field_ising_mpo as two_site_3d_ising_mpo,
)


def _dense_pair_frame(state, layout):
    frame = np.zeros(
        (state.hilbert_dim, int(np.prod(layout.merged_shape))),
        dtype=np.result_type(*state.tensors),
    )
    dimensions = (state.physical_dim,) * state.nsites
    for row, configuration in enumerate(np.ndindex(*dimensions)):
        left = state._left_partial(layout.left_site, configuration)
        right = state._right_partial(layout.left_site + 1, configuration)
        physical = tuple(
            configuration[site] for site in layout.merged_physical_sites
        )
        for left_index in range(layout.merged_shape[0]):
            for right_index in range(layout.merged_shape[-1]):
                local = (left_index,) + physical + (right_index,)
                column = np.ravel_multi_index(local, layout.merged_shape)
                frame[row, column] = left[left_index] * right[right_index]
    return frame


def test_two_site_defaults_use_environment_weighted_truncation():
    assert LETTATwoSiteOptions().split_method == "metric-als"
    assert LETTATwoSiteOptions().energy_refinement_max_iterations == 8


def test_pair_layout_separates_shared_and_exclusive_physical_sites():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=101
    )

    layout = LETTAPairLayout.from_state(state, 0)

    assert layout.sites == (0, 1)
    assert layout.left_neighborhood == (0, 1, 2)
    assert layout.right_neighborhood == (1, 3)
    assert layout.left_only == (0, 2)
    assert layout.shared == (1,)
    assert layout.right_only == (3,)
    assert layout.merged_physical_sites == (0, 2, 1, 3)
    assert layout.merged_shape == (1, 2, 2, 2, 2, 2)


def test_pair_merge_identifies_the_shared_physical_axis():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=102
    )
    layout = LETTAPairLayout.from_state(state, 0)

    merged = layout.merge(state.tensors[0], state.tensors[1])

    for configuration in np.ndindex(*(2,) * state.nsites):
        merged_physical = tuple(
            configuration[site] for site in layout.merged_physical_sites
        )
        actual = merged[(0,) + merged_physical + (slice(None),)]
        left_physical = tuple(
            configuration[site] for site in layout.left_neighborhood
        )
        right_physical = tuple(
            configuration[site] for site in layout.right_neighborhood
        )
        left = state.tensors[0][(0,) + left_physical + (slice(None),)]
        right = state.tensors[1][
            (slice(None),) + right_physical + (slice(None),)
        ]
        expected = left @ right
        np.testing.assert_allclose(actual, expected, atol=1.0e-13)


def test_conditional_svd_reconstructs_each_shared_sector_independently():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=103
    )
    layout = LETTAPairLayout.from_state(state, 0)
    rng = np.random.default_rng(104)
    merged = rng.normal(size=layout.merged_shape)

    split = conditional_svd_split(
        merged,
        layout,
        max_bond_dim=4,
        direction="lr",
    )
    reconstructed = layout.merge(split.left_tensor, split.right_tensor)

    assert split.left_tensor.shape == (1, 2, 2, 2, 4)
    assert split.right_tensor.shape == (4, 2, 2, 2)
    assert split.sector_ranks == (4, 4)
    assert split.discarded_weight < 1.0e-28
    np.testing.assert_allclose(reconstructed, merged, atol=1.0e-12)


def test_pair_merge_maps_have_complex_adjoint_actions():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=121, real=False
    )
    layout = LETTAPairLayout.from_state(state, 0)
    rng = np.random.default_rng(122)
    gradient = rng.normal(size=layout.merged_shape)
    gradient = gradient + 1j * rng.normal(size=layout.merged_shape)
    left = state.tensors[0]
    right = state.tensors[1]
    merged = layout.merge(left, right)

    left_inner = np.vdot(merged, gradient)
    left_adjoint = np.vdot(
        left, layout.left_adjoint(gradient, right)
    )
    right_adjoint = np.vdot(
        right, layout.right_adjoint(left, gradient)
    )

    np.testing.assert_allclose(left_adjoint, left_inner, atol=1.0e-12)
    np.testing.assert_allclose(right_adjoint, left_inner, atol=1.0e-12)


def test_pair_layout_supports_adjacent_tensors_without_shared_physical_sites():
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=2, seed=105
    )

    layout = LETTAPairLayout.from_state(state, 2)

    assert layout.left_neighborhood == (2, 5)
    assert layout.right_neighborhood == (3, 4)
    assert layout.shared == ()
    assert layout.left_only == (2, 5)
    assert layout.right_only == (3, 4)

    merged = np.random.default_rng(130).normal(size=layout.merged_shape)
    split = conditional_svd_split(
        merged,
        layout,
        max_bond_dim=8,
        direction="rl",
    )
    np.testing.assert_allclose(
        layout.merge(split.left_tensor, split.right_tensor),
        merged,
        atol=1.0e-12,
    )


def test_pair_overlap_metric_matches_dense_pair_frame():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=106
    )
    layout = LETTAPairLayout.from_state(state, 0)
    frame = _dense_pair_frame(state, layout)
    cache = IdentityPairEnvironmentCache(state)
    right = cache.build_right_environments()

    metric = cache.effective_pair_metric(
        cache.scalar_boundary(), right[2], layout
    )

    dense_metric = np.einsum(
        "ki,kj->ij", frame.conj(), frame, optimize=True
    )
    np.testing.assert_allclose(metric.to_dense(), dense_metric, atol=1.0e-11)


def test_pair_hamiltonian_action_matches_dense_pair_frame():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=107
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.8, field=1.3)
    layout = LETTAPairLayout.from_state(state, 0)
    frame = _dense_pair_frame(state, layout)
    effective = np.einsum(
        "ai,ab,bj->ij",
        frame.conj(),
        mpo.to_dense(),
        frame,
        optimize=True,
    )
    cache = LETTAPairEnvironmentCache(state, mpo)
    right = cache.build_right_environments()
    rng = np.random.default_rng(108)
    vector = rng.normal(size=int(np.prod(layout.merged_shape)))

    actual = cache.effective_pair_action(
        cache.scalar_boundary(), right[2], layout, vector
    )

    expected = np.einsum("ij,j->i", effective, vector, optimize=True)
    np.testing.assert_allclose(actual, expected, atol=1.0e-11)


def test_pair_hamiltonian_action_batches_multiple_vectors():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=136
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.8, field=1.3)
    layout = LETTAPairLayout.from_state(state, 0)
    cache = LETTAPairEnvironmentCache(state, mpo)
    right = cache.build_right_environments()
    vectors = np.random.default_rng(137).normal(
        size=(int(np.prod(layout.merged_shape)), 3)
    )

    batched = cache.effective_pair_action(
        cache.scalar_boundary(), right[2], layout, vectors
    )
    separate = np.column_stack(
        [
            cache.effective_pair_action(
                cache.scalar_boundary(),
                right[2],
                layout,
                vectors[:, column],
            )
            for column in range(vectors.shape[1])
        ]
    )

    np.testing.assert_allclose(batched, separate, atol=1.0e-11)


def test_sparse_pair_action_uses_mpo_transitions_instead_of_dense_factors():
    state = LatticeLETTA.random(
        (2, 3), physical_dim=2, bond_dim=2, seed=127
    )
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    dense = LETTAPairEnvironmentCache(state, mpo, use_sparse_mpo=False)
    sparse = LETTAPairEnvironmentCache(state, mpo, use_sparse_mpo=True)
    dense_right = dense.build_right_environments()
    sparse_right = sparse.build_right_environments()
    dense_left = dense.build_left_environments()
    sparse_left = sparse.build_left_environments()
    rng = np.random.default_rng(128)
    cases = []
    for site in range(state.nsites - 1):
        layout = LETTAPairLayout.from_state(state, site)
        vector = rng.normal(size=int(np.prod(layout.merged_shape)))
        expected = dense.effective_pair_action(
            dense_left[site], dense_right[site + 2], layout, vector
        )
        cases.append((site, layout, vector, expected))
    mpo.factors = tuple(np.full_like(factor, np.nan) for factor in mpo.factors)

    for site, layout, vector, expected in cases:
        actual = sparse.effective_pair_action(
            sparse_left[site], sparse_right[site + 2], layout, vector
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-11)


def test_metric_als_does_not_increase_wavefunction_truncation_loss():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=123
    )
    layout = LETTAPairLayout.from_state(state, 0)
    cache = IdentityPairEnvironmentCache(state)
    right = cache.build_right_environments()
    metric = cache.effective_pair_metric(
        cache.scalar_boundary(), right[2], layout
    )
    rng = np.random.default_rng(124)
    target = rng.normal(size=layout.merged_shape)
    initial = conditional_svd_split(
        target,
        layout,
        max_bond_dim=1,
        direction="lr",
    )
    initial_difference = target.reshape(-1) - layout.merge(
        initial.left_tensor, initial.right_tensor
    ).reshape(-1)
    initial_loss = float(
        np.real(np.vdot(initial_difference, metric @ initial_difference))
    )

    refined = metric_als_refine(
        target,
        layout,
        initial,
        metric,
        tolerance=1.0e-12,
        max_iterations=6,
        metric_tolerance=1.0e-12,
    )

    assert refined.loss <= initial_loss + 1.0e-12
    assert refined.iterations >= 1


def test_fixed_rank_energy_refinement_lowers_split_rayleigh_energy():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=133
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    layout = LETTAPairLayout.from_state(state, 0)
    hamiltonian_cache = LETTAPairEnvironmentCache(state, mpo)
    metric_cache = IdentityPairEnvironmentCache(state)
    hamiltonian_right = hamiltonian_cache.build_right_environments()
    metric_right = metric_cache.build_right_environments()
    metric = metric_cache.effective_pair_metric(
        metric_cache.scalar_boundary(), metric_right[2], layout
    )

    def action(vector):
        return hamiltonian_cache.effective_pair_action(
            hamiltonian_cache.scalar_boundary(),
            hamiltonian_right[2],
            layout,
            vector,
        )

    target = np.random.default_rng(134).normal(size=layout.merged_shape)
    initial = conditional_svd_split(
        target,
        layout,
        max_bond_dim=1,
        direction="lr",
    )
    initial_vector = layout.merge(
        initial.left_tensor, initial.right_tensor
    ).reshape(-1)
    initial_energy = float(
        np.real(
            np.vdot(initial_vector, action(initial_vector))
            / np.vdot(initial_vector, metric @ initial_vector)
        )
    )

    refined = energy_refine_split(
        layout,
        initial,
        action,
        metric,
        max_iterations=2,
        tolerance=1.0e-10,
        metric_tolerance=1.0e-12,
        energy_increase_tolerance=1.0e-10,
        max_factor_norm_growth=100.0,
    )
    refined_vector = layout.merge(
        refined.left_tensor, refined.right_tensor
    ).reshape(-1)
    refined_energy = float(
        np.real(
            np.vdot(refined_vector, action(refined_vector))
            / np.vdot(refined_vector, metric @ refined_vector)
        )
    )

    np.testing.assert_allclose(refined.initial_energy, initial_energy, atol=1.0e-10)
    np.testing.assert_allclose(refined.energy, refined_energy, atol=1.0e-10)
    assert refined.energy <= refined.initial_energy + 1.0e-10
    assert refined.iterations >= 1
    assert refined.accepted_substeps >= 1
    assert np.isfinite(refined.max_factor_norm)


def test_two_site_solver_supports_energy_refined_split_updates():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=135
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    initial_energy = state.expectation(mpo)

    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            split_method="energy-refined",
            energy_refinement_max_iterations=2,
        ),
    )

    assert result.energy <= initial_energy + 1.0e-10
    np.testing.assert_allclose(
        result.energy, result.state.expectation(mpo), atol=1.0e-10
    )
    for update in result.history[0].updates:
        assert update.energy_refinement_iterations >= 1
        assert update.energy_refinement_energy <= (
            update.energy_refinement_initial_energy + 1.0e-10
        )
        if update.accepted:
            assert update.energy <= update.old_energy + 1.0e-10


def test_two_site_sweep_is_energy_nonincreasing_at_fixed_bond_dimension():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=109
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    initial_energy = state.expectation(mpo)

    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=2,
            tolerance=1.0e-12,
            split_method="conditional-svd",
        ),
    )

    assert result.energy <= initial_energy + 1.0e-10
    np.testing.assert_allclose(
        result.energy, result.state.expectation(mpo), atol=1.0e-10
    )
    assert result.state.bond_dimensions == (2, 2, 2)
    assert len(result.history) >= 1
    assert all(
        later.energy <= earlier.energy + 1.0e-10
        for earlier, later in zip(result.history, result.history[1:])
    )
    for sweep in result.history:
        for update in sweep.updates:
            if update.accepted:
                assert update.energy <= update.old_energy + 1.0e-10


def test_matrix_free_two_site_solver_does_not_form_pair_hamiltonian(monkeypatch):
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=110
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)

    def fail(*_args, **_kwargs):
        raise AssertionError("a dense pair Hamiltonian was formed")

    monkeypatch.setattr(LETTAPairEnvironmentCache, "effective_pair_matrix", fail)
    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            matrix_free=True,
            split_method="conditional-svd",
        ),
    )

    assert np.isfinite(result.energy)


def test_matrix_free_and_dense_pair_solvers_agree_on_a_small_lattice():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=131
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    common = dict(
        max_sweeps=1,
        split_method="conditional-svd",
    )
    dense = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(**common, matrix_free=False),
    )
    matrix_free = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(**common, matrix_free=True),
    )

    np.testing.assert_allclose(matrix_free.energy, dense.energy, atol=1.0e-10)


def test_reverse_sweep_preserves_requested_bond_shapes_near_boundary():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=4, seed=132
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)

    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=4,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            start_direction="rl",
            split_method="conditional-svd",
        ),
    )

    assert result.state.bond_dimensions == (4, 4, 4)
    np.testing.assert_allclose(
        result.energy, result.state.expectation(mpo), atol=1.0e-10
    )


def test_two_site_solver_uses_metric_als_split_refinement():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=125
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    initial_energy = state.expectation(mpo)

    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            split_method="metric-als",
            truncation_max_iterations=3,
        ),
    )

    assert result.energy <= initial_energy + 1.0e-10
    np.testing.assert_allclose(
        result.energy, result.state.expectation(mpo), atol=1.0e-10
    )
    assert all(
        update.truncation_iterations >= 1
        for update in result.history[0].updates
    )


def test_optional_one_site_polish_preserves_or_lowers_two_site_energy():
    state = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=1, seed=126
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    common = dict(
        max_sweeps=1,
        split_method="metric-als",
        truncation_max_iterations=2,
    )
    unpolished = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(**common),
    )

    polished = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(
            **common,
            one_site_polish_sweeps=1,
        ),
    )

    assert polished.energy <= unpolished.energy + 1.0e-10
    assert polished.two_site_energy == polished.history[-1].energy
    assert polished.polish_sweeps == 1


def test_case_packages_reuse_existing_2d_and_3d_model_builders():
    assert two_site_2d_ising_mpo((2, 2)).nsites == 4
    assert two_site_3d_ising_mpo((2, 2, 2)).nsites == 8


def test_two_site_solver_lowers_2x2x2_snake_letta_energy():
    state = two_site_snake_letta_state(
        (2, 2, 2), bond_dim=1, seed=129
    )
    mpo = two_site_3d_ising_mpo(
        (2, 2, 2), coupling=1.0, field=1.5
    )
    initial_energy = state.expectation(mpo)

    result = letta_two_site_dmrg(
        mpo,
        state=state,
        bond_dim=1,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            split_method="conditional-svd",
            dense_solver_threshold=128,
        ),
    )

    assert result.energy <= initial_energy + 1.0e-10
    np.testing.assert_allclose(
        result.energy, result.state.expectation(mpo), atol=1.0e-10
    )
    assert all(
        update.shared_physical_sites
        for update in result.history[0].updates
    )

import numpy as np

from pyqed._letta_one_site_opt import (
    BoundaryMPS,
    IdentityEnvironmentCache,
    LETTAEnvironmentCache,
    LETTADMROptions,
    LatticeLETTA,
    exact_ground_state,
    identity_mpo,
    letta_dmrg,
    network_expectation,
    network_operator_matrix,
    network_overlap,
    automatic_bond_schedule,
)
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_hamiltonian,
    transverse_field_ising_mpo,
)
from pyqed._letta_one_site_opt._letta_for_2d.examples.ising_comparison import compare_ising
from pyqed._letta_one_site_opt.solver import (
    _build_environment_checkpoints,
    _lowest_generalized_eigenpair,
    _shift_virtual_gauge,
)
from pyqed._letta_one_site_opt.contractions import (
    BlockDiagonalMetric,
    _compiled_contraction,
    _contract_operands,
)


def test_generalized_eigensolver_is_invariant_to_effective_matrix_scale():
    scale = 1.0e-20
    hamiltonian = scale * np.diag([1.0, 2.0])
    metric = scale * np.eye(2)

    energy, vector, rank, residual = _lowest_generalized_eigenpair(
        hamiltonian,
        metric,
        metric_tolerance=1.0e-12,
    )

    np.testing.assert_allclose(energy, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.vdot(vector, metric @ vector), 1.0)
    assert rank == 2
    assert residual < 1.0e-20


def test_equivalent_relabelled_contractions_reuse_compiled_expression():
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    _compiled_contraction.cache_clear()

    first = _contract_operands([left, right], [(10, 11), (11, 12)], (10, 12))
    second = _contract_operands([left, right], [(3, 8), (8, 5)], (3, 5))

    np.testing.assert_allclose(first, left @ right)
    np.testing.assert_allclose(second, left @ right)
    assert _compiled_contraction.cache_info().hits == 1


def test_compact_ising_mpo_matches_sparse_hamiltonian():
    for lattice_shape in ((2, 3), (2, 2, 2)):
        sparse_hamiltonian = transverse_field_ising_hamiltonian(
            lattice_shape,
            coupling=0.7,
            field=1.2,
        )
        mpo = transverse_field_ising_mpo(
            lattice_shape,
            coupling=0.7,
            field=1.2,
        )

        np.testing.assert_allclose(
            mpo.to_dense(),
            sparse_hamiltonian.toarray(),
            atol=1.0e-14,
        )
        assert max(mpo.bond_dimensions) <= int(np.prod(lattice_shape[1:])) + 2


def test_direct_network_contractions_match_dense_reference():
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=8)
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.9, field=1.1)
    dense_hamiltonian = mpo.to_dense()

    np.testing.assert_allclose(network_overlap(state), state.norm())
    np.testing.assert_allclose(
        network_expectation(state, mpo),
        state.expectation(dense_hamiltonian),
    )
    for site in range(state.nsites):
        frame = state.local_frame(site)
        direct_metric = network_operator_matrix(
            state,
            identity_mpo(state.nsites, state.physical_dim),
            site,
        )
        direct_hamiltonian = network_operator_matrix(state, mpo, site)
        np.testing.assert_allclose(
            direct_metric,
            frame.conj().T @ frame,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            direct_hamiltonian,
            frame.conj().T @ dense_hamiltonian @ frame,
            atol=1.0e-11,
        )


def test_cached_frontier_environments_match_full_network_contractions():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=11)
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    cache = LETTAEnvironmentCache(state, mpo)
    right = cache.build_right_environments()
    left = cache.scalar_boundary()

    for site in range(state.nsites):
        cached = cache.effective_matrix(left, right[site + 1], site)
        direct = network_operator_matrix(state, mpo, site)
        np.testing.assert_allclose(cached, direct, atol=1.0e-11)
        left = cache.extend_left(left, site)


def test_sparse_mpo_channel_contractions_match_dense_channel_backend():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=15)
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    sparse_cache = LETTAEnvironmentCache(state, mpo, use_sparse_mpo=True)
    dense_cache = LETTAEnvironmentCache(state, mpo, use_sparse_mpo=False)
    sparse_right = sparse_cache.build_right_environments()
    dense_right = dense_cache.build_right_environments()
    sparse_left = sparse_cache.scalar_boundary()
    dense_left = dense_cache.scalar_boundary()

    assert sum(len(transitions) for transitions in mpo.transitions) < sum(
        factor.shape[0] * factor.shape[1] for factor in mpo.factors
    )
    for site in range(state.nsites):
        np.testing.assert_allclose(
            sparse_cache.effective_matrix(
                sparse_left, sparse_right[site + 1], site
            ),
            dense_cache.effective_matrix(dense_left, dense_right[site + 1], site),
            atol=1.0e-11,
        )
        sparse_left = sparse_cache.extend_left(sparse_left, site)
        dense_left = dense_cache.extend_left(dense_left, site)
        np.testing.assert_allclose(sparse_left, dense_left, atol=1.0e-11)


def test_identity_cache_fuses_bra_and_ket_frontier_labels():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=12)
    identity_cache = IdentityEnvironmentCache(state)
    generic_cache = LETTAEnvironmentCache(
        state,
        identity_mpo(state.nsites, state.physical_dim),
    )

    assert max(map(len, identity_cache.frontiers)) < max(
        map(len, generic_cache.frontiers)
    )
    right = identity_cache.build_right_environments()
    left = identity_cache.scalar_boundary()
    for site in range(state.nsites):
        frame = state.local_frame(site)
        np.testing.assert_allclose(
            identity_cache.effective_matrix(left, right[site + 1], site),
            frame.conj().T @ frame,
            atol=1.0e-11,
        )
        left = identity_cache.extend_left(left, site)


def test_block_diagonal_metric_matches_dense_overlap_matrix():
    state = LatticeLETTA.random(
        (2, 2, 2),
        physical_dim=2,
        bond_dim=2,
        seed=21,
    )
    cache = IdentityEnvironmentCache(state)
    right = cache.build_right_environments()
    left = cache.scalar_boundary()

    for site in range(state.nsites):
        block_metric = cache.effective_metric(left, right[site + 1], site)
        dense_metric = cache.effective_matrix(left, right[site + 1], site)

        assert isinstance(block_metric, BlockDiagonalMetric)
        np.testing.assert_allclose(
            block_metric.to_dense(),
            dense_metric,
            atol=1.0e-11,
        )
        left = cache.extend_left(left, site)


def test_mpo_solver_reuses_initial_energy_from_cached_environments(monkeypatch):
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=22)
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)

    def fail(*_args, **_kwargs):
        raise AssertionError("a redundant full expectation was evaluated")

    monkeypatch.setattr(LatticeLETTA, "expectation", fail)
    result = letta_dmrg(
        mpo,
        state=state,
        options=LETTADMROptions(max_sweeps=1),
    )

    assert np.isfinite(result.energy)


def test_sparse_extension_batches_repeated_local_operators(monkeypatch):
    state = LatticeLETTA.random(
        (2, 2, 2),
        physical_dim=2,
        bond_dim=1,
        seed=23,
    )
    mpo = transverse_field_ising_mpo((2, 2, 2), coupling=1.0, field=1.5)
    cache = LETTAEnvironmentCache(state, mpo, use_sparse_mpo=True)
    calls = 0
    original = _contract_operands

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pyqed._letta_one_site_opt.contractions._contract_operands",
        count,
    )
    cache.extend_left(cache.scalar_boundary(), 0)

    assert calls < len(mpo.transitions[0])


def test_streaming_boundary_update_matches_dense_without_reconstruction(
    monkeypatch,
):
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=24)
    mpo = transverse_field_ising_mpo((2, 3), coupling=0.8, field=1.3)
    dense_cache = LETTAEnvironmentCache(state, mpo, use_sparse_mpo=False)
    dense = dense_cache.scalar_boundary()
    dense = dense_cache.extend_left(dense, 0)
    dense = dense_cache.extend_left(dense, 1)
    dense_right = dense_cache.scalar_boundary()
    dense_right = dense_cache.extend_right(dense_right, state.nsites - 1)
    dense_right = dense_cache.extend_right(dense_right, state.nsites - 2)

    cache = LETTAEnvironmentCache(
        state,
        mpo,
        boundary_bond_dim=1024,
        boundary_cutoff=0.0,
    )
    boundary = cache.extend_left(cache.scalar_boundary(), 0)
    boundary_right = cache.extend_right(
        cache.scalar_boundary(),
        state.nsites - 1,
    )
    original_to_dense = BoundaryMPS.to_dense

    def fail(*_args, **_kwargs):
        raise AssertionError("the existing boundary was reconstructed densely")

    monkeypatch.setattr(BoundaryMPS, "to_dense", fail)
    boundary = cache.extend_left(boundary, 1)
    boundary_right = cache.extend_right(
        boundary_right,
        state.nsites - 2,
    )

    assert isinstance(boundary, BoundaryMPS)
    np.testing.assert_allclose(
        original_to_dense(boundary),
        dense,
        atol=1.0e-10,
    )
    assert isinstance(boundary_right, BoundaryMPS)
    np.testing.assert_allclose(
        original_to_dense(boundary_right),
        dense_right,
        atol=1.0e-10,
    )


def test_normalization_distributes_scale_across_letta_tensors():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=4, seed=10)
    tensor_norms = np.asarray(
        [np.linalg.norm(tensor) for tensor in state.tensors]
    )

    np.testing.assert_allclose(state.norm(), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        tensor_norms,
        np.full(state.nsites, tensor_norms[0]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_zero_noise_bond_expansion_preserves_every_amplitude():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=20)
    reference = state.state_vector()

    expanded = state.expand_bond_dimension(5, noise=0.0)

    assert expanded is not state
    assert expanded.bond_dimensions == (5,) * (state.nsites - 1)
    np.testing.assert_allclose(expanded.state_vector(), reference, atol=1.0e-13)


def test_automatic_bond_schedule_assigns_most_sweeps_to_target_dimension():
    dimensions, sweeps = automatic_bond_schedule(8, 20)

    assert dimensions == (2, 4, 8)
    assert sweeps == (2, 4, 14)
    assert sum(sweeps) == 20


def test_bond_continuation_improves_4x4_energy_at_fixed_sweep_budget():
    mpo = transverse_field_ising_mpo((4, 4), coupling=1.0, field=1.5)
    common = dict(
        max_sweeps=10,
        tolerance=1.0e-30,
        environment_granularity="column",
    )
    direct = letta_dmrg(
        mpo,
        lattice_shape=(4, 4),
        bond_dim=4,
        seed=4,
        options=LETTADMROptions(**common),
    )
    continued = letta_dmrg(
        mpo,
        lattice_shape=(4, 4),
        bond_dim=4,
        seed=4,
        options=LETTADMROptions(
            **common,
            bond_dimension_schedule=(2, 4),
            bond_schedule_sweeps=(2, 8),
            bond_expansion_noise=1.0e-3,
        ),
    )

    assert continued.sweeps == direct.sweeps == 10
    assert {sweep.bond_dimension for sweep in continued.history} == {2, 4}
    assert continued.energy < direct.energy - 1.0e-4
    for sweep in continued.history:
        np.testing.assert_allclose(
            sweep.energy_density_change,
            sweep.energy_change / 16,
        )


def test_mpo_sweep_does_not_build_dense_state_or_local_frame(monkeypatch):
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=9)
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)

    def fail(*_args, **_kwargs):
        raise AssertionError("dense Hilbert-space path was used")

    monkeypatch.setattr(LatticeLETTA, "state_vector", fail)
    monkeypatch.setattr(LatticeLETTA, "local_frame", fail)
    monkeypatch.setattr(
        "pyqed._letta_one_site_opt.solver.network_operator_matrix",
        fail,
    )
    result = letta_dmrg(
        mpo,
        state=state,
        options=LETTADMROptions(max_sweeps=1),
    )

    assert np.isfinite(result.energy)
    np.testing.assert_allclose(result.state.norm(), 1.0, atol=1.0e-10)


def test_alternating_sweeps_reuse_newly_built_opposite_environments(monkeypatch):
    counts = {"hamiltonian": 0, "metric": 0}
    original_h = LETTAEnvironmentCache.build_right_environments
    original_n = IdentityEnvironmentCache.build_right_environments

    def count_h(self):
        counts["hamiltonian"] += 1
        return original_h(self)

    def count_n(self):
        counts["metric"] += 1
        return original_n(self)

    monkeypatch.setattr(LETTAEnvironmentCache, "build_right_environments", count_h)
    monkeypatch.setattr(IdentityEnvironmentCache, "build_right_environments", count_n)
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    letta_dmrg(
        mpo,
        lattice_shape=(2, 2),
        bond_dim=2,
        seed=3,
        options=LETTADMROptions(max_sweeps=3, tolerance=1.0e-30),
    )

    assert counts == {"hamiltonian": 1, "metric": 1}


def test_column_checkpoints_match_site_environment_sweep():
    initial = LatticeLETTA.random(
        (3, 2), physical_dim=2, bond_dim=2, seed=16
    )
    mpo = transverse_field_ising_mpo((3, 2), coupling=1.0, field=1.5)
    cache = LETTAEnvironmentCache(initial, mpo)
    checkpoints = _build_environment_checkpoints(cache, "rl", block_size=2)

    assert set(checkpoints) == {0, 2, 4, 6}
    site_result = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(
            max_sweeps=2,
            tolerance=1.0e-30,
            environment_granularity="site",
        ),
    )
    column_result = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(
            max_sweeps=2,
            tolerance=1.0e-30,
            environment_granularity="column",
        ),
    )

    np.testing.assert_allclose(
        column_result.energy, site_result.energy, atol=1.0e-10
    )
    np.testing.assert_allclose(
        column_result.state.state_vector(),
        site_result.state.state_vector(),
        atol=1.0e-9,
    )


def test_matrix_free_local_solver_matches_dense_local_solver(monkeypatch):
    initial = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=17
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    dense_result = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(max_sweeps=1, matrix_free=False),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("dense Hamiltonian effective matrix was formed")

    monkeypatch.setattr(LETTAEnvironmentCache, "effective_matrix", fail)
    matrix_free_result = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(max_sweeps=1, matrix_free=True),
    )

    np.testing.assert_allclose(
        matrix_free_result.energy, dense_result.energy, atol=1.0e-9
    )


def test_boundary_mps_roundtrip_and_truncation_diagnostics():
    tensor = np.random.default_rng(18).normal(size=(2, 3, 2, 4))
    exact = BoundaryMPS.from_dense(
        tensor, labels=(0, 1, 2, 3), max_bond_dim=64, cutoff=0.0
    )
    compressed = BoundaryMPS.from_dense(
        tensor, labels=(0, 1, 2, 3), max_bond_dim=2, cutoff=0.0
    )

    np.testing.assert_allclose(exact.to_dense(), tensor, atol=1.0e-12)
    assert exact.discarded_weight < 1.0e-28
    assert compressed.discarded_weight > 0.0
    assert compressed.nbytes < tensor.nbytes


def test_compressed_boundary_solver_matches_exact_when_rank_is_sufficient():
    initial = LatticeLETTA.random(
        (2, 2), physical_dim=2, bond_dim=2, seed=19
    )
    mpo = transverse_field_ising_mpo((2, 2), coupling=1.0, field=1.5)
    exact = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(max_sweeps=1, boundary_bond_dim=None),
    )
    compressed = letta_dmrg(
        mpo,
        state=initial,
        options=LETTADMROptions(
            max_sweeps=1,
            boundary_bond_dim=256,
            boundary_cutoff=0.0,
        ),
    )

    np.testing.assert_allclose(compressed.energy, exact.energy, atol=1.0e-10)
    assert compressed.max_boundary_discarded_weight < 1.0e-20


def test_truncated_boundary_solver_rejects_unresolved_metric_directions():
    shape = (3, 3)
    mpo = transverse_field_ising_mpo(shape, coupling=1.0, field=1.5)
    result = letta_dmrg(
        mpo,
        lattice_shape=shape,
        bond_dim=4,
        seed=4,
        options=LETTADMROptions(
            max_sweeps=1,
            matrix_free=True,
            environment_granularity="column",
            boundary_bond_dim=4,
            boundary_cutoff=1.0e-12,
        ),
    )
    spectral_lower_bound = -(12.0 + 1.5 * 9.0)

    assert np.isfinite(result.energy)
    assert result.energy >= spectral_lower_bound - 1.0e-8


def test_qr_virtual_gauge_preserves_letta_state():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=3, seed=14)
    reference = state.state_vector()

    _shift_virtual_gauge(state, 1, "lr", "qr")
    np.testing.assert_allclose(state.state_vector(), reference, atol=1.0e-12)
    _shift_virtual_gauge(state, 4, "rl", "qr")
    np.testing.assert_allclose(state.state_vector(), reference, atol=1.0e-12)


def test_4x4_mpo_sweeps_remain_monotonic_with_ill_conditioned_metric():
    mpo = transverse_field_ising_mpo((4, 4), coupling=1.0, field=1.5)
    result = letta_dmrg(
        mpo,
        lattice_shape=(4, 4),
        physical_dim=2,
        bond_dim=4,
        seed=4,
        options=LETTADMROptions(max_sweeps=3, tolerance=1.0e-10),
    )

    energies = [sweep.energy for sweep in result.history]
    assert all(
        later <= earlier + 1.0e-9
        for earlier, later in zip(energies, energies[1:])
    )
    assert energies[-1] < -30.47


def test_square_lattice_neighborhood_matches_column_major_scratch_convention():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=1)

    assert state.coordinates == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    )
    assert state.site_neighborhood(0) == (0, 1, 3)
    assert state.site_neighborhood(1) == (1, 2, 4)
    assert state.site_neighborhood(2) == (2, 5)
    assert state.site_neighborhood(5) == (5,)
    assert state.tensors[0].shape == (1, 2, 2, 2, 2)
    assert state.tensors[-1].shape == (2, 2, 1)


def test_amplitude_is_product_of_tensors_with_shared_lattice_spins():
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=2)
    config = (0, 1, 1, 0)

    expected = state.tensors[0][0, config[0], config[1], config[2], :]
    expected = expected @ state.tensors[1][
        :, config[1], config[3], :
    ]
    expected = expected @ state.tensors[2][
        :, config[2], config[3], :
    ]
    expected = expected @ state.tensors[3][:, config[3], 0]

    np.testing.assert_allclose(state.amplitude(config), expected)


def test_column_blocking_is_exact_enlarged_physical_space_identity():
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=3)

    for config in np.ndindex(*(2,) * state.nsites):
        np.testing.assert_allclose(
            state.blocked_column_amplitude(config),
            state.amplitude(config),
            atol=1.0e-12,
        )


def test_local_frame_reconstructs_state_when_applied_to_active_tensor():
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=4)

    for site in range(state.nsites):
        frame = state.local_frame(site)
        reconstructed = frame @ state.tensors[site].reshape(-1)
        np.testing.assert_allclose(reconstructed, state.state_vector(), atol=1.0e-12)


def test_three_dimensional_positive_neighbor_ties_are_supported():
    state = LatticeLETTA.random((2, 2, 2), physical_dim=2, bond_dim=2, seed=5)

    assert state.site_neighborhood(0) == (0, 1, 2, 4)
    assert state.tensors[0].shape == (1, 2, 2, 2, 2, 2)
    assert state.state_vector().shape == (2**8,)
    np.testing.assert_allclose(np.linalg.norm(state.state_vector()), 1.0)


def test_dmrg_like_sweep_lowers_energy_for_three_dimensional_lattice():
    hamiltonian = transverse_field_ising_mpo(
        (2, 2, 2),
        coupling=1.0,
        field=1.5,
    )
    initial = LatticeLETTA.random(
        (2, 2, 2),
        physical_dim=2,
        bond_dim=2,
        seed=7,
    )
    initial_energy = initial.expectation(hamiltonian)

    result = letta_dmrg(
        hamiltonian,
        state=initial,
        options=LETTADMROptions(max_sweeps=2, tolerance=1.0e-10),
    )

    assert result.energy <= initial_energy + 1.0e-10
    assert all(
        later.energy <= earlier.energy + 1.0e-9
        for earlier, later in zip(result.history, result.history[1:])
    )


def test_lattice_tfim_hamiltonian_and_observables_match_dense_reference():
    hamiltonian = transverse_field_ising_hamiltonian(
        (2, 2),
        coupling=1.0,
        field=1.3,
    )
    exact_energy, exact_state = exact_ground_state(hamiltonian)

    dense = hamiltonian.toarray()
    values, vectors = np.linalg.eigh(dense)
    np.testing.assert_allclose(exact_energy, values[0], atol=1.0e-12)
    np.testing.assert_allclose(
        abs(np.vdot(exact_state, vectors[:, 0])),
        1.0,
        atol=1.0e-10,
    )


def test_dmrg_like_sweeps_lower_energy_and_approach_exact_2x2_ising_result():
    hamiltonian = transverse_field_ising_hamiltonian(
        (2, 2),
        coupling=1.0,
        field=1.5,
    )
    exact_energy, _exact_state = exact_ground_state(hamiltonian)
    initial = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=4, seed=6)
    initial_energy = initial.expectation(hamiltonian)

    result = letta_dmrg(
        hamiltonian,
        state=initial,
        options=LETTADMROptions(max_sweeps=8, tolerance=1.0e-10),
    )

    assert result.energy <= initial_energy + 1.0e-10
    assert all(
        later.energy <= earlier.energy + 1.0e-9
        for earlier, later in zip(result.history, result.history[1:])
    )
    assert result.energy - exact_energy < 1.0e-7


def test_2x3_ising_example_uses_fixed_eight_dimensional_local_physical_block():
    comparison = compare_ising(lattice_shape=(2, 3), bond_dim=4, max_sweeps=4)

    assert comparison.converged
    assert comparison.exact_runtime_seconds >= 0.0
    assert comparison.letta_runtime_seconds >= 0.0
    assert comparison.max_local_physical_dimension == 8
    assert abs(comparison.letta_energy - comparison.exact_energy) < 1.0e-10
    assert abs(comparison.letta_x - comparison.exact_x) < 1.0e-9
    assert abs(comparison.letta_zz - comparison.exact_zz) < 1.0e-9


def test_ising_comparison_can_skip_exact_diagonalization(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("exact Hamiltonian was constructed")

    monkeypatch.setattr(
        "pyqed._letta_one_site_opt._letta_for_2d.examples.ising_comparison."
        "transverse_field_ising_hamiltonian",
        fail,
    )
    comparison = compare_ising(
        lattice_shape=(2, 2),
        bond_dim=2,
        max_sweeps=1,
        run_exact=False,
    )

    assert comparison.exact_energy is None
    assert comparison.exact_runtime_seconds is None
    assert comparison.exact_x is None
    assert comparison.exact_zz is None


def test_ising_comparison_skips_exact_above_site_limit(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("exact Hamiltonian was constructed")

    monkeypatch.setattr(
        "pyqed._letta_one_site_opt._letta_for_2d.examples.ising_comparison."
        "transverse_field_ising_hamiltonian",
        fail,
    )
    comparison = compare_ising(
        lattice_shape=(2, 3),
        bond_dim=2,
        max_sweeps=1,
        max_exact_sites=4,
    )

    assert comparison.exact_energy is None
    assert comparison.exact_runtime_seconds is None

import numpy as np

from pyqed._letta_one_site_opt import (
    ReducedFrontier,
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedSymmetry,
    reduced_mps_state_vector,
)


def _spatial_state(shape=(2, 2), seed=9):
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(
        basis,
        target_charge=int(np.prod(shape)),
        target_two_j=0,
    )
    return ReducedLatticeLETTA.random(
        shape,
        symmetry=symmetry,
        multiplets_per_sector=1,
        seed=seed,
    )


def test_frontier_tracks_each_repeated_physical_variable_until_last_use():
    state = _spatial_state()
    frontier = ReducedFrontier.from_state(state)

    assert frontier.cuts == ((1, 2), (2, 3), (3,))
    assert frontier.left_variables(0) == ()
    assert frontier.right_variables(0) == (1, 2)
    assert frontier.left_variables(2) == (2, 3)
    assert frontier.right_variables(3) == ()


def test_frontier_embedding_reconstructs_identical_dense_state():
    state = _spatial_state(seed=13)
    frontier = ReducedFrontier.from_state(state)
    mps = frontier.to_mps(state)

    embedded = reduced_mps_state_vector(
        mps,
        state.physical_basis,
        target_sector=state.symmetry.sector,
        target_two_m=0,
    )
    np.testing.assert_allclose(embedded, state.state_vector(), atol=1.0e-12)


def test_frontier_memory_only_multiplies_multiplicity_not_irrep_content():
    state = _spatial_state()
    frontier = ReducedFrontier.from_state(state)
    mps = frontier.to_mps(state)

    for site, tensor in enumerate(mps):
        assert set(tensor.qns[0]) == set(state.left_virtual_sectors(site))
        assert set(tensor.qns[2]) == set(state.right_virtual_sectors(site))
        left_memory = state.physical_dim ** len(frontier.left_variables(site))
        right_memory = state.physical_dim ** len(frontier.right_variables(site))
        assert len(tensor.qns[0]) == len(state.left_virtual_sectors(site)) * left_memory
        assert len(tensor.qns[2]) == len(state.right_virtual_sectors(site)) * right_memory


def test_site_embedding_and_adjoint_obey_inner_product_identity():
    state = _spatial_state(seed=17)
    embedding = ReducedFrontier.from_state(state).site_embedding(state, 1)
    rng = np.random.default_rng(21)
    source = rng.normal(size=embedding.source_size)
    target = rng.normal(size=embedding.target_size)

    lhs = np.vdot(embedding.apply(source), target)
    rhs = np.vdot(source, embedding.adjoint(target))
    np.testing.assert_allclose(lhs, rhs, atol=1.0e-13)


def test_embedding_vector_matches_expanded_site_tensor_blocks():
    state = _spatial_state(seed=19)
    frontier = ReducedFrontier.from_state(state)
    embedding = frontier.site_embedding(state, 0)
    source = embedding.pack_source(state.tensors[0])
    target = embedding.apply(source)
    expanded = frontier.to_mps(state)[0]

    np.testing.assert_allclose(target, embedding.pack_target(expanded.data))
    rebuilt = embedding.unpack_target(target)
    for key in expanded.data:
        np.testing.assert_allclose(rebuilt[key], expanded.data[key])


def test_custom_coordinate_order_uses_general_factor_graph_frontiers():
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(
        basis,
        target_charge=4,
        target_two_j=0,
    )
    coordinates = ((1, 1), (0, 0), (1, 0), (0, 1))
    state = ReducedLatticeLETTA.random(
        (2, 2),
        symmetry=symmetry,
        coordinates=coordinates,
        multiplets_per_sector=1,
        seed=23,
    )
    frontier = ReducedFrontier.from_state(state)

    # Site 0's physical variable is used again by factors at sites 2 and 3;
    # the general frontier must carry it even though those are backward
    # geometric dependencies in the custom ordering.
    assert 0 in frontier.cuts[0]
    assert 0 in frontier.cuts[1]
    assert 0 in frontier.cuts[2]
    embedded = reduced_mps_state_vector(
        frontier.to_mps(state),
        basis,
        target_sector=symmetry.sector,
        target_two_m=0,
    )
    np.testing.assert_allclose(embedded, state.state_vector(), atol=1.0e-12)

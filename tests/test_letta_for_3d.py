import numpy as np

from pyqed._letta_one_site_opt._letta_for_3d import (
    MPSDMRGOptions,
    SnakeMPS,
    letta_ground_state,
    mps_dmrg,
    nearest_neighbor_bonds,
    ordered_coordinates,
    snake_coordinates,
    snake_letta_state,
    transverse_field_ising_mpo,
    transverse_field_ising_sparse,
)
from pyqed._letta_one_site_opt import LETTADMROptions, exact_ground_state


def _manhattan_distance(left, right):
    return sum(abs(a - b) for a, b in zip(left, right))


def test_2x2x2_snake_is_a_nearest_neighbor_hamiltonian_path():
    coordinates = snake_coordinates((2, 2, 2))

    assert coordinates == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (1, 0, 0),
    )
    assert all(
        _manhattan_distance(left, right) == 1
        for left, right in zip(coordinates, coordinates[1:])
    )


def test_compact_and_layer_snake_orderings_reduce_3x3x3_mpo_bandwidth():
    shape = (3, 3, 3)
    compact = ordered_coordinates(shape, ordering="compact")
    layer_snake = ordered_coordinates(shape, ordering="layer-snake")
    continuous = ordered_coordinates(shape, ordering="continuous-snake")

    assert compact == tuple(np.ndindex(*shape))
    assert layer_snake[:9] == continuous[:9]
    assert tuple(coordinate[1:] for coordinate in layer_snake[9:18]) == tuple(
        coordinate[1:] for coordinate in layer_snake[:9]
    )
    assert max(
        transverse_field_ising_mpo(
            shape,
            ordering="compact",
        ).bond_dimensions
    ) == 11
    assert max(
        transverse_field_ising_mpo(
            shape,
            ordering="continuous-snake",
        ).bond_dimensions
    ) == 19


def test_3x3x3_snake_contains_every_cubic_bond_once():
    coordinates = snake_coordinates((3, 3, 3))
    bonds = nearest_neighbor_bonds((3, 3, 3))

    assert len(coordinates) == 27
    assert len(set(coordinates)) == 27
    assert len(bonds) == 54
    assert len(set(bonds)) == 54
    assert all(left < right for left, right in bonds)
    assert all(
        _manhattan_distance(coordinates[left], coordinates[right]) == 1
        for left, right in bonds
    )


def test_snake_letta_has_current_z_y_x_physical_leg_order():
    state = snake_letta_state((2, 2, 2), bond_dim=2, seed=3)

    assert state.coordinates == snake_coordinates((2, 2, 2))
    assert state.site_neighborhood(0) == (0, 1, 3, 7)
    assert state.tensors[0].shape == (1, 2, 2, 2, 2, 2)
    assert max(
        int(np.prod(tensor.shape[1:-1])) for tensor in state.tensors
    ) == 16


def test_snake_tfim_mpo_matches_sparse_reference_on_2x2x2():
    mpo = transverse_field_ising_mpo((2, 2, 2), coupling=0.7, field=1.2)
    sparse = transverse_field_ising_sparse(
        (2, 2, 2),
        coupling=0.7,
        field=1.2,
    )

    np.testing.assert_allclose(mpo.to_dense(), sparse.toarray(), atol=1.0e-13)


def test_snake_mps_contractions_match_dense_state_vector():
    state = SnakeMPS.random(8, physical_dim=2, bond_dim=4, seed=5)
    mpo = transverse_field_ising_mpo((2, 2, 2), coupling=0.8, field=1.1)
    vector = state.state_vector()

    expected = np.vdot(vector, mpo.to_dense() @ vector) / np.vdot(vector, vector)

    np.testing.assert_allclose(state.norm(), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(state.expectation(mpo), expected, atol=1.0e-11)


def test_two_site_mps_dmrg_approaches_exact_2x2x2_ground_energy():
    mpo = transverse_field_ising_mpo((2, 2, 2), coupling=1.0, field=1.5)
    sparse = transverse_field_ising_sparse(
        (2, 2, 2),
        coupling=1.0,
        field=1.5,
    )
    exact_energy, _ = exact_ground_state(sparse)
    result = mps_dmrg(
        mpo,
        bond_dim=16,
        seed=6,
        options=MPSDMRGOptions(max_sweeps=5, tolerance=1.0e-10),
    )

    assert result.energy - exact_energy < 1.0e-8
    assert result.history[-1].energy_density_change < 1.0e-8


def test_snake_letta_sweep_lowers_2x2x2_energy():
    mpo = transverse_field_ising_mpo((2, 2, 2), coupling=1.0, field=1.5)
    initial = snake_letta_state((2, 2, 2), bond_dim=2, seed=7)
    initial_energy = initial.expectation(mpo)

    result = letta_ground_state(
        mpo,
        lattice_shape=(2, 2, 2),
        bond_dim=2,
        seed=7,
        state=initial,
        options=LETTADMROptions(max_sweeps=2, tolerance=1.0e-12),
    )

    assert result.energy < initial_energy
    assert result.state.coordinates == snake_coordinates((2, 2, 2))


def test_3x3x3_model_builds_without_dense_hilbert_space_objects():
    shape = (3, 3, 3)
    state = snake_letta_state(shape, bond_dim=2, seed=8)
    mpo = transverse_field_ising_mpo(shape, coupling=1.0, field=1.5)

    assert state.nsites == mpo.nsites == 27
    assert len(nearest_neighbor_bonds(shape)) == 54
    assert max(mpo.bond_dimensions) == 19
    assert max(
        int(np.prod(tensor.shape[1:-1])) for tensor in state.tensors
    ) == 16

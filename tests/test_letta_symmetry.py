import numpy as np

from pyqed._letta_one_site_opt import (
    AbelianSymmetry,
    LETTADMROptions,
    LatticeLETTA,
    letta_dmrg,
)
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_hamiltonian,
    transverse_field_ising_mpo,
)
from pyqed._letta_one_site_opt._letta_for_3d import (
    snake_coordinates,
    transverse_field_ising_mpo as transverse_field_ising_mpo_3d,
    transverse_field_ising_sparse,
)
from pyqed._letta_two_site_opt import (
    LETTAPairLayout,
    LETTATwoSiteOptions,
    conditional_svd_split,
    letta_two_site_dmrg,
)


def _z2_even():
    return AbelianSymmetry(
        physical_charges=(0, 1),
        sector=0,
        moduli=2,
        name="ising-z2",
    )


def _assert_only_target_sector(state):
    vector = state.state_vector()
    for flat, configuration in enumerate(
        np.ndindex(*(state.physical_dim,) * state.nsites)
    ):
        charge = state.symmetry.configuration_charge(configuration)
        if charge != state.symmetry.sector:
            np.testing.assert_allclose(vector[flat], 0.0, atol=1.0e-13)


def test_user_defined_abelian_symmetry_masks_random_copy_and_expansion():
    symmetry = AbelianSymmetry(
        physical_charges=((0, 0), (1, 1), (0, 2)),
        sector=(0, 0),
        moduli=(2, 3),
        name="custom-z2-x-z3",
    )
    state = LatticeLETTA.random(
        (1, 3),
        physical_dim=3,
        bond_dim=6,
        seed=701,
        symmetry=symmetry,
    )

    assert state.symmetry is symmetry
    assert state.symmetry_violation() == 0.0
    assert state.parameter_count < state.dense_parameter_count
    _assert_only_target_sector(state)

    copied = state.copy()
    expanded = state.expand_bond_dimension(8, noise=1.0e-3, seed=702)
    assert copied.symmetry == symmetry
    assert copied.bond_charges == state.bond_charges
    assert expanded.symmetry == symmetry
    assert expanded.bond_dimensions == (8, 8)
    assert expanded.symmetry_violation() == 0.0
    _assert_only_target_sector(expanded)


def test_ising_x_basis_exposes_z2_and_preserves_dense_spectrum_in_2d_and_3d():
    symmetry = _z2_even()
    mpo_z = transverse_field_ising_mpo(
        (2, 2), coupling=0.7, field=1.2, basis="z"
    )
    mpo_x = transverse_field_ising_mpo(
        (2, 2), coupling=0.7, field=1.2, basis="x"
    )
    sparse_z = transverse_field_ising_hamiltonian(
        (2, 2), coupling=0.7, field=1.2, basis="z"
    )
    sparse_x = transverse_field_ising_hamiltonian(
        (2, 2), coupling=0.7, field=1.2, basis="x"
    )

    np.testing.assert_allclose(
        np.linalg.eigvalsh(mpo_z.to_dense()),
        np.linalg.eigvalsh(mpo_x.to_dense()),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(mpo_z.to_dense(), sparse_z.toarray(), atol=1.0e-12)
    np.testing.assert_allclose(mpo_x.to_dense(), sparse_x.toarray(), atol=1.0e-12)
    assert symmetry.commutation_error(mpo_x.to_dense(), nsites=4) < 1.0e-12

    mpo_3d = transverse_field_ising_mpo_3d(
        (1, 2, 2), coupling=0.7, field=1.2, basis="x"
    )
    sparse_3d = transverse_field_ising_sparse(
        (1, 2, 2), coupling=0.7, field=1.2, basis="x"
    )
    np.testing.assert_allclose(mpo_3d.to_dense(), sparse_3d.toarray(), atol=1.0e-12)
    assert symmetry.commutation_error(mpo_3d.to_dense(), nsites=4) < 1.0e-12


def test_one_site_z2_matches_dense_run_and_reduces_every_local_problem():
    symmetry = _z2_even()
    mpo = transverse_field_ising_mpo(
        (1, 3), coupling=0.7, field=1.2, basis="x"
    )
    symmetric_initial = LatticeLETTA.random(
        (1, 3), bond_dim=2, seed=703, symmetry=symmetry
    )
    dense_initial = symmetric_initial.without_symmetry()
    options = LETTADMROptions(
        max_sweeps=6,
        tolerance=1.0e-11,
        matrix_free=False,
        gauge_mode="qr",
    )

    symmetric = letta_dmrg(mpo, state=symmetric_initial, options=options)
    dense = letta_dmrg(mpo, state=dense_initial, options=options)

    np.testing.assert_allclose(symmetric.energy, dense.energy, atol=1.0e-9)
    np.testing.assert_allclose(
        symmetric.energy, symmetric.state.expectation(mpo), atol=1.0e-11
    )
    assert symmetric.state.symmetry_violation() == 0.0
    _assert_only_target_sector(symmetric.state)
    for sweep in symmetric.history:
        for update in sweep.updates:
            assert update.local_dimension < update.full_local_dimension
    assert sum(
        update.local_dimension for sweep in symmetric.history for update in sweep.updates
    ) < sum(
        update.local_dimension for sweep in dense.history for update in sweep.updates
    )


def test_both_solvers_accept_symmetry_directly_without_manual_state_setup():
    symmetry = _z2_even()
    mpo = transverse_field_ising_mpo(
        (1, 2), coupling=0.7, field=1.2, basis="x"
    )

    one_site = letta_dmrg(
        mpo,
        lattice_shape=(1, 2),
        bond_dim=2,
        seed=707,
        symmetry=symmetry,
        options=LETTADMROptions(max_sweeps=1, matrix_free=True),
    )
    two_site = letta_two_site_dmrg(
        mpo,
        lattice_shape=(1, 2),
        bond_dim=2,
        seed=707,
        symmetry=symmetry,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            split_method="energy-refined",
            energy_refinement_max_iterations=2,
        ),
    )

    assert one_site.state.symmetry == symmetry
    assert two_site.state.symmetry == symmetry
    assert one_site.state.symmetry_violation() == 0.0
    assert two_site.state.symmetry_violation() == 0.0

    coordinates = snake_coordinates((1, 1, 2))
    mpo_3d = transverse_field_ising_mpo_3d(
        (1, 1, 2), coupling=0.7, field=1.2, basis="x"
    )
    ordered = letta_dmrg(
        mpo_3d,
        lattice_shape=(1, 1, 2),
        coordinates=coordinates,
        bond_dim=2,
        symmetry=symmetry,
        options=LETTADMROptions(max_sweeps=1, matrix_free=True),
    )
    ordered_two = letta_two_site_dmrg(
        mpo_3d,
        lattice_shape=(1, 1, 2),
        coordinates=coordinates,
        bond_dim=2,
        symmetry=symmetry,
        options=LETTATwoSiteOptions(max_sweeps=1),
    )
    assert ordered.state.coordinates == coordinates
    assert ordered_two.state.coordinates == coordinates


def test_symmetry_aware_conditional_split_preserves_both_site_masks():
    state = LatticeLETTA.random(
        (2, 2), bond_dim=4, seed=704, symmetry=_z2_even()
    )
    layout = LETTAPairLayout.from_state(state, 0)
    rng = np.random.default_rng(705)
    merged = rng.normal(size=layout.merged_shape)
    merged[~layout.symmetry_mask()] = 0.0

    split = conditional_svd_split(
        merged,
        layout,
        max_bond_dim=4,
        direction="lr",
    )

    assert np.linalg.norm(split.left_tensor[~state.symmetry_mask(0)]) == 0.0
    assert np.linalg.norm(split.right_tensor[~state.symmetry_mask(1)]) == 0.0
    reconstructed = layout.merge(split.left_tensor, split.right_tensor)
    assert np.linalg.norm(reconstructed[~layout.symmetry_mask()]) == 0.0


def test_symmetry_split_supports_3d_snake_pair_with_shared_left_owned_site():
    state = LatticeLETTA.random(
        (2, 2, 2),
        bond_dim=4,
        seed=708,
        coordinates=snake_coordinates((2, 2, 2)),
        symmetry=_z2_even(),
    )
    layout = LETTAPairLayout.from_state(state, 2)
    assert layout.left_site in layout.shared
    merged = np.random.default_rng(709).normal(size=layout.merged_shape)
    merged[~layout.symmetry_mask()] = 0.0

    split = conditional_svd_split(
        merged, layout, max_bond_dim=4, direction="rl"
    )

    assert np.linalg.norm(split.left_tensor[~layout.factor_mask("left")]) == 0.0
    assert np.linalg.norm(split.right_tensor[~layout.factor_mask("right")]) == 0.0


def test_two_site_z2_matches_dense_run_and_reduces_every_pair_problem():
    symmetry = _z2_even()
    mpo = transverse_field_ising_mpo(
        (1, 3), coupling=0.7, field=1.2, basis="x"
    )
    symmetric_initial = LatticeLETTA.random(
        (1, 3), bond_dim=2, seed=706, symmetry=symmetry
    )
    dense_initial = symmetric_initial.without_symmetry()
    options = LETTATwoSiteOptions(
        max_sweeps=5,
        tolerance=1.0e-11,
        split_method="metric-als",
        matrix_free=True,
        gauge_mode="qr",
    )

    symmetric = letta_two_site_dmrg(
        mpo, state=symmetric_initial, bond_dim=2, options=options
    )
    dense = letta_two_site_dmrg(
        mpo, state=dense_initial, bond_dim=2, options=options
    )

    np.testing.assert_allclose(symmetric.energy, dense.energy, atol=1.0e-8)
    np.testing.assert_allclose(
        symmetric.energy, symmetric.state.expectation(mpo), atol=1.0e-10
    )
    assert symmetric.state.symmetry_violation() == 0.0
    _assert_only_target_sector(symmetric.state)
    for sweep in symmetric.history:
        for update in sweep.updates:
            assert update.local_dimension < update.full_local_dimension

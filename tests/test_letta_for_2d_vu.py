import numpy as np

from pyqed._letta_for_2d_vu import (
    UniformCylinderLETTA,
    VULETTAOptions,
    cylinder_energy_density,
    horizontal_zz_expectation,
    random_uniform_cylinder_letta,
    tfim_cylinder_hamiltonian,
    transverse_magnetization,
    transverse_zz_expectation,
    vuletta_cylinder,
)
from pyqed._vuletta import energy_density


def _tfim_chain_bond(coupling, field):
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return (
        -coupling * np.kron(z, z)
        - 0.5 * field * np.kron(x, identity)
        - 0.5 * field * np.kron(identity, x)
    )


def test_width_one_cylinder_hamiltonian_is_the_tfim_chain_bond():
    model = tfim_cylinder_hamiltonian(
        1,
        coupling=0.7,
        field=1.2,
    )

    np.testing.assert_allclose(
        model.local_density.reshape(4, 4),
        _tfim_chain_bond(coupling=0.7, field=1.2),
        atol=1.0e-13,
    )
    assert model.column_dim == 2
    assert model.transverse_bond_count == 0


def test_cylinder_state_uses_column_configurations_as_tied_indices():
    state = random_uniform_cylinder_letta(
        width=2,
        local_physical_dim=2,
        bond_dim=3,
        seed=2,
        real=True,
    )

    assert state.tensor.shape == (3, 4, 4, 3)
    assert state.column_dim == 4
    assert state.effective_bond_dim == 12
    np.testing.assert_allclose(
        state.periodic_amplitude(((0, 0), (0, 1), (1, 1))),
        state.uniform_state.periodic_amplitude((0, 1, 3)),
    )


def test_cylinder_energy_matches_underlying_uniform_letta_contraction():
    state = random_uniform_cylinder_letta(
        width=2,
        local_physical_dim=2,
        bond_dim=1,
        seed=3,
        real=True,
        transverse_boundary="open",
    )
    model = tfim_cylinder_hamiltonian(
        2,
        coupling=0.8,
        field=1.1,
        transverse_boundary="open",
    )

    np.testing.assert_allclose(
        cylinder_energy_density(state, model),
        energy_density(state.uniform_state, model.local_density) / 2.0,
        atol=1.0e-12,
    )


def test_uniform_plus_state_has_expected_cylinder_observables():
    width = 3
    column_dim = 2**width
    state = UniformCylinderLETTA(
        width=width,
        local_physical_dim=2,
        tensor=np.ones((1, column_dim, column_dim, 1)),
        transverse_boundary="periodic",
    )
    model = tfim_cylinder_hamiltonian(
        width,
        coupling=1.0,
        field=1.5,
        transverse_boundary="periodic",
    )

    np.testing.assert_allclose(transverse_magnetization(state), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        horizontal_zz_expectation(state),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        transverse_zz_expectation(state),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cylinder_energy_density(state, model),
        -1.5,
        atol=1.0e-12,
    )


def test_width_two_vuletta_update_lowers_the_energy():
    model = tfim_cylinder_hamiltonian(
        2,
        coupling=1.0,
        field=1.5,
        transverse_boundary="open",
    )
    initial = random_uniform_cylinder_letta(
        width=2,
        local_physical_dim=2,
        bond_dim=1,
        seed=5,
        real=True,
        transverse_boundary="open",
    )
    initial_energy = cylinder_energy_density(initial, model)
    result = vuletta_cylinder(
        model,
        initial=initial,
        options=VULETTAOptions(
            max_iterations=15,
            update_method="lbfgs",
            gradient_method="analytic",
            stationarity_tolerance=1.0e-5,
        ),
    )

    assert result.energy_density < initial_energy - 1.0e-3
    assert result.state.width == 2
    assert result.state.tensor.shape == initial.tensor.shape
    assert np.isfinite(result.residual_norm)

import numpy as np
import pytest

from pyqed._vumps import canonicalize, one_site_expectation
from pyqed._vumps.examples.tfim_comparison import (
    exact_tfim_energy_density,
    exact_tfim_transverse_magnetization,
    exact_tfim_zz_correlation,
    finite_tfim_ground_state,
    tfim_bond_hamiltonian,
    vumps_tfim_ground_state,
)


def test_one_site_expectation_for_x_polarized_product_state():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    plus_x = np.array([1.0, 1.0]) / np.sqrt(2.0)
    state = canonicalize(plus_x.reshape(1, 2, 1))

    np.testing.assert_allclose(one_site_expectation(state, x), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(one_site_expectation(state, z), 0.0, atol=1.0e-12)


def test_one_site_expectation_preserves_complex_bra_ket_order():
    raising = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    plus_y = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    state = canonicalize(plus_y.reshape(1, 2, 1))

    np.testing.assert_allclose(
        one_site_expectation(state, raising),
        0.5j,
        atol=1.0e-12,
    )


def test_exact_tfim_critical_point_reference_values():
    np.testing.assert_allclose(
        exact_tfim_energy_density(coupling=1.0, field=1.0),
        -4.0 / np.pi,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        exact_tfim_transverse_magnetization(coupling=1.0, field=1.0),
        2.0 / np.pi,
        atol=1.0e-11,
    )


def test_exact_tfim_zz_reference_is_stable_for_weak_coupling():
    np.testing.assert_allclose(
        exact_tfim_zz_correlation(coupling=1.0e-10, field=1.0),
        5.0e-11,
        rtol=1.0e-5,
        atol=1.0e-15,
    )


def test_tfim_bond_hamiltonian_splits_onsite_field_equally():
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])

    expected = -np.kron(z, z) - 0.75 * (
        np.kron(x, identity) + np.kron(identity, x)
    )

    np.testing.assert_allclose(
        tfim_bond_hamiltonian(coupling=1.0, field=1.5),
        expected,
        atol=1.0e-12,
    )


def test_finite_tfim_ground_state_matches_direct_two_site_diagonalization():
    coupling = 0.8
    field = 1.2
    result = finite_tfim_ground_state(
        num_sites=2,
        coupling=coupling,
        field=field,
    )

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    identity = np.eye(2)
    # A two-site periodic ring contains the same nearest-neighbor edge twice.
    dense = (
        -2.0 * coupling * np.kron(z, z)
        - field * (np.kron(x, identity) + np.kron(identity, x))
    )
    expected_energy = np.linalg.eigvalsh(dense)[0] / 2.0

    np.testing.assert_allclose(result.energy_density, expected_energy, atol=1.0e-11)


def test_tfim_example_rejects_nonintegral_discrete_dimensions():
    with pytest.raises(ValueError, match="num_sites"):
        finite_tfim_ground_state(num_sites=2.5)
    with pytest.raises(ValueError, match="bond_dim"):
        vumps_tfim_ground_state(bond_dim=2.5)


def test_vumps_tfim_example_observables_approach_exact_references():
    result = vumps_tfim_ground_state(
        bond_dim=2,
        coupling=1.0,
        field=1.5,
        seed=3,
        tolerance=1.0e-8,
        max_iterations=40,
    )
    exact_energy = exact_tfim_energy_density(coupling=1.0, field=1.5)
    exact_magnetization = exact_tfim_transverse_magnetization(
        coupling=1.0,
        field=1.5,
    )

    assert result.converged
    assert abs(result.energy_density - exact_energy) < 2.0e-4
    assert abs(result.transverse_magnetization - exact_magnetization) < 8.0e-4

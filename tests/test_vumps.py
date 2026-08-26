import numpy as np
import pytest

from pyqed._vumps import (
    CanonicalMPS,
    VUMPSOptions,
    apply_left_transfer,
    apply_right_transfer,
    build_effective_hamiltonians,
    canonicalize,
    nearest_neighbor_energy,
    vumps,
)
from pyqed._vumps.operators import _left_source, _right_source
from pyqed.mps import UniformMPS


def _spin_half_heisenberg_bond():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def test_canonicalize_uses_left_physical_right_tensor_order():
    rng = np.random.default_rng(8)
    tensor = rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3))

    state = canonicalize(tensor)

    assert state.AL.shape == (3, 2, 3)
    assert state.AR.shape == (3, 2, 3)
    assert state.C.shape == (3, 3)
    assert state.AC.shape == (3, 2, 3)
    assert state.left_isometry_error() < 1.0e-10
    assert state.right_isometry_error() < 1.0e-10
    assert state.center_error() < 1.0e-10
    np.testing.assert_allclose(np.linalg.norm(state.C), 1.0, atol=1.0e-12)


def test_nearest_neighbor_energy_matches_existing_uniform_mps_convention():
    rng = np.random.default_rng(12)
    tensor = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)

    state = canonicalize(tensor)
    expected = UniformMPS(state.AL.transpose(1, 0, 2)).energy_density(h)

    np.testing.assert_allclose(nearest_neighbor_energy(state, h), expected, atol=1.0e-10)


def test_effective_hamiltonian_actions_are_hermitian():
    rng = np.random.default_rng(19)
    tensor = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    state = canonicalize(tensor)

    effective = build_effective_hamiltonians(state, h)
    x_ac = rng.normal(size=state.AC.shape) + 1j * rng.normal(size=state.AC.shape)
    y_ac = rng.normal(size=state.AC.shape) + 1j * rng.normal(size=state.AC.shape)
    x_c = rng.normal(size=state.C.shape) + 1j * rng.normal(size=state.C.shape)
    y_c = rng.normal(size=state.C.shape) + 1j * rng.normal(size=state.C.shape)

    np.testing.assert_allclose(
        np.vdot(x_ac, effective.apply_center(y_ac)),
        np.vdot(effective.apply_center(x_ac), y_ac),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.vdot(x_c, effective.apply_bond(y_c)),
        np.vdot(effective.apply_bond(x_c), y_c),
        atol=1.0e-10,
    )
    assert abs(np.trace(state.rho_left @ effective.HL)) < 1.0e-10
    assert abs(np.trace(state.rho_right @ effective.HR)) < 1.0e-10


def test_hamiltonian_environments_satisfy_defining_linear_equations():
    rng = np.random.default_rng(23)
    tensor = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    state = canonicalize(tensor)
    h = h.reshape(2, 2, 2, 2)

    effective = build_effective_hamiltonians(state, h)
    identity = np.eye(state.bond_dim)
    left_source = _left_source(state, h)
    right_source = _right_source(state, h)
    left_centered = left_source - identity * np.trace(state.rho_left @ left_source)
    right_centered = (
        right_source - identity * np.trace(state.rho_right @ right_source)
    )

    np.testing.assert_allclose(
        effective.HL - apply_left_transfer(state.AL, effective.HL),
        left_centered,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        effective.HR - apply_right_transfer(state.AR, effective.HR),
        right_centered,
        atol=1.0e-10,
    )


def test_effective_hamiltonians_are_covariant_under_virtual_gauge_change():
    rng = np.random.default_rng(29)
    tensor = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    state = canonicalize(tensor)
    unitary, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    )

    def transform_matrix(matrix):
        return unitary.conj().T @ matrix @ unitary

    def transform_tensor(site_tensor):
        return np.stack(
            [
                transform_matrix(site_tensor[:, physical, :])
                for physical in range(site_tensor.shape[1])
            ],
            axis=1,
        )

    transformed = CanonicalMPS(
        AL=transform_tensor(state.AL),
        C=transform_matrix(state.C),
        AR=transform_tensor(state.AR),
    )
    effective = build_effective_hamiltonians(state, h)
    transformed_effective = build_effective_hamiltonians(transformed, h)
    center_probe = rng.normal(size=state.AC.shape) + 1j * rng.normal(
        size=state.AC.shape
    )
    bond_probe = rng.normal(size=state.C.shape) + 1j * rng.normal(
        size=state.C.shape
    )

    np.testing.assert_allclose(
        transformed_effective.HL,
        transform_matrix(effective.HL),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        transformed_effective.HR,
        transform_matrix(effective.HR),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        transformed_effective.apply_center(transform_tensor(center_probe)),
        transform_tensor(effective.apply_center(center_probe)),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        transformed_effective.apply_bond(transform_matrix(bond_probe)),
        transform_matrix(effective.apply_bond(bond_probe)),
        atol=1.0e-10,
    )


def test_bond_effective_hamiltonian_contains_crossing_bond_term():
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h = np.kron(z, z)
    state = canonicalize(np.array([[[1.0], [0.0]]]))

    effective = build_effective_hamiltonians(state, h)

    np.testing.assert_allclose(effective.apply_bond(state.C), state.C, atol=1.0e-12)


def test_vumps_solves_ferromagnetic_heisenberg_product_ground_state():
    h = -_spin_half_heisenberg_bond()

    result = vumps(
        h,
        bond_dim=1,
        seed=7,
        options=VUMPSOptions(max_iterations=30, tolerance=1.0e-11),
    )

    assert result.converged
    np.testing.assert_allclose(result.energy, -0.25, atol=1.0e-11)
    assert result.residual_norm < 1.0e-10
    assert result.state.left_isometry_error() < 1.0e-12
    assert result.state.center_error() < 1.0e-12


def test_vumps_infers_bond_dimension_from_existing_uniform_mps():
    h = -_spin_half_heisenberg_bond()
    initial = UniformMPS(np.array([[[1.0]], [[0.0]]]))

    result = vumps(
        h,
        initial=initial,
        options=VUMPSOptions(max_iterations=10, tolerance=1.0e-11),
    )

    assert result.converged
    assert result.state.bond_dim == 1
    np.testing.assert_allclose(result.energy, -0.25, atol=1.0e-12)


def test_vumps_rejects_nonhermitian_hamiltonian():
    h = np.zeros((4, 4))
    h[0, 1] = 1.0

    with pytest.raises(ValueError, match="Hermitian"):
        vumps(h, bond_dim=1)


def test_canonical_mps_rejects_noncanonical_tensors():
    tensor = np.array([[[2.0], [0.0]]])

    with pytest.raises(ValueError, match="left-canonical"):
        CanonicalMPS(AL=tensor, C=np.ones((1, 1)), AR=tensor)


def test_vumps_recanonicalizes_provisional_canonical_initial_state():
    rng = np.random.default_rng(31)
    exact = canonicalize(
        rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    )
    physical_rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    unrelated_right = np.einsum(
        "st,atb->asb",
        physical_rotation,
        exact.AR,
        optimize=True,
    )
    provisional = CanonicalMPS(
        AL=exact.AL,
        C=exact.C,
        AR=unrelated_right,
        center_tensor=exact.AC,
    )
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h = -np.kron(z, z) - 0.75 * (
        np.kron(x, identity) + np.kron(identity, x)
    )
    options = VUMPSOptions(max_iterations=1, tolerance=1.0e-14)

    from_provisional = vumps(h, initial=provisional, options=options)
    from_left_tensor = vumps(h, initial=exact.AL, options=options)

    assert provisional.center_error() > 0.1
    np.testing.assert_allclose(
        from_provisional.energy,
        from_left_tensor.energy,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        from_provisional.residual_norm,
        from_left_tensor.residual_norm,
        atol=1.0e-12,
    )


def test_bond_one_vumps_iterates_to_self_consistent_product_state():
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h = -np.kron(z, z) - 0.75 * (
        np.kron(x, identity) + np.kron(identity, x)
    )

    result = vumps(
        h,
        bond_dim=1,
        seed=4,
        options=VUMPSOptions(max_iterations=100, tolerance=1.0e-10),
    )

    assert result.converged
    assert result.iterations > 1
    np.testing.assert_allclose(result.energy, -1.5625, atol=1.0e-10)


def test_nonconverged_result_energy_matches_exported_uniform_mps():
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h = -np.kron(z, z) - 0.75 * (
        np.kron(x, identity) + np.kron(identity, x)
    )

    result = vumps(
        h,
        bond_dim=2,
        seed=3,
        options=VUMPSOptions(max_iterations=1, tolerance=1.0e-14),
    )

    assert not result.converged
    np.testing.assert_allclose(
        result.energy,
        result.to_uniform_mps().energy_density(h),
        atol=1.0e-10,
    )


def test_vumps_solves_translation_invariant_ising_ground_state():
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    field = 1.5
    h = -np.kron(z, z) - 0.5 * field * (
        np.kron(x, identity) + np.kron(identity, x)
    )

    result = vumps(
        h,
        bond_dim=2,
        seed=3,
        options=VUMPSOptions(max_iterations=40, tolerance=1.0e-8),
    )

    assert result.converged
    np.testing.assert_allclose(result.energy, -1.67173662, atol=2.0e-7)
    assert result.residual_norm < 1.0e-8
    assert result.state.center_error() < 1.0e-8


def test_vumps_matrix_free_eigensolver_and_environment_paths():
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h = -np.kron(z, z) - 0.75 * (
        np.kron(x, identity) + np.kron(identity, x)
    )

    result = vumps(
        h,
        bond_dim=2,
        seed=3,
        options=VUMPSOptions(
            max_iterations=40,
            tolerance=1.0e-7,
            dense_eigensolver_threshold=0,
            dense_environment_threshold=0,
        ),
    )

    assert result.converged
    np.testing.assert_allclose(result.energy, -1.67173662, atol=3.0e-7)

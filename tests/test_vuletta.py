import numpy as np
import pytest

from pyqed._vuletta import (
    ConditionalCanonicalLETTA,
    ConditionalTangentData,
    UniformLETTA,
    VULETTAOptions,
    conditional_canonicalize,
    conditional_tangent_direction,
    energy_density,
    energy_gradient,
    expand_uniform_letta,
    natural_gradient,
    one_site_expectation,
    random_uniform_letta,
    tangent_gram_matrix,
    transfer_data,
    two_site_expectation,
    vuletta,
)
from pyqed.mps import UniformMPS


def _tfim_bond(coupling=1.0, field=1.5):
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return -coupling * np.kron(z, z) - 0.5 * field * (
        np.kron(x, identity) + np.kron(identity, x)
    )


def test_structured_mps_amplitude_equals_periodic_letta_amplitude():
    rng = np.random.default_rng(4)
    tensor = rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(
        size=(2, 2, 2, 2)
    )
    state = UniformLETTA(tensor)
    configuration = (0, 1, 1, 0, 1)

    direct = state.periodic_amplitude(configuration)
    structured = state.structured_mps_tensor()
    product = np.eye(state.effective_bond_dim, dtype=complex)
    for physical in configuration:
        product = product @ structured[:, physical, :]

    np.testing.assert_allclose(np.trace(product), direct, atol=1.0e-12)


def test_shifted_structured_mps_amplitude_equals_periodic_letta_amplitude():
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=40)
    configuration = (0, 1, 1, 0, 1)

    product = np.eye(state.effective_bond_dim, dtype=complex)
    shifted = state.shifted_structured_mps_tensor()
    for physical in configuration:
        product = product @ shifted[:, physical, :]

    np.testing.assert_allclose(
        np.trace(product),
        state.periodic_amplitude(configuration),
        atol=1.0e-12,
    )


@pytest.mark.parametrize("real", [True, False])
def test_conditional_canonicalize_preserves_state_and_sector_isometries(real):
    state = random_uniform_letta(
        physical_dim=2,
        bond_dim=2,
        seed=41,
        real=real,
    )

    canonical = conditional_canonicalize(state)

    assert isinstance(canonical, ConditionalCanonicalLETTA)
    assert canonical.TL.shape == state.tensor.shape
    assert canonical.TR.shape == state.tensor.shape
    assert canonical.TC.shape == state.tensor.shape
    assert canonical.C.shape == (
        state.physical_dim,
        state.bond_dim,
        state.bond_dim,
    )
    assert canonical.left_isometry_error() < 1.0e-10
    assert canonical.right_isometry_error() < 1.0e-10
    assert canonical.center_error() < 1.0e-10
    canonical.validate()

    hamiltonian = _tfim_bond()
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(
        energy_density(canonical.state, hamiltonian),
        energy_density(state, hamiltonian),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        one_site_expectation(canonical.state, x),
        one_site_expectation(state, x),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        two_site_expectation(canonical.state, np.kron(z, z)),
        two_site_expectation(state, np.kron(z, z)),
        atol=1.0e-10,
    )

    for configuration in ((0, 0, 1), (0, 1, 1, 0)):
        np.testing.assert_allclose(
            canonical.state.periodic_amplitude(configuration),
            state.periodic_amplitude(configuration)
            * canonical.amplitude_scale(len(configuration)),
            atol=1.0e-10,
        )


def test_conditional_center_relations_hold_blockwise():
    canonical = conditional_canonicalize(
        random_uniform_letta(physical_dim=3, bond_dim=2, seed=42)
    )

    for previous in range(canonical.physical_dim):
        for current in range(canonical.physical_dim):
            left_center = (
                canonical.TL[:, previous, current, :] @ canonical.C[current]
            )
            right_center = (
                canonical.C[previous] @ canonical.TR[:, previous, current, :]
            )
            np.testing.assert_allclose(
                canonical.TC[:, previous, current, :],
                left_center,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(left_center, right_center, atol=1.0e-10)


@pytest.mark.parametrize("real", [True, False])
def test_conditional_tangent_direction_is_horizontal_and_descending(real):
    canonical = conditional_canonicalize(
        random_uniform_letta(
            physical_dim=2,
            bond_dim=2,
            seed=43,
            real=real,
        )
    )
    gradient = energy_gradient(canonical.state, _tfim_bond())

    tangent = conditional_tangent_direction(canonical, gradient, real=real)

    assert isinstance(tangent, ConditionalTangentData)
    expected_dimension = 8 if real else 16
    assert tangent.reduced_dimension == expected_dimension
    if real:
        assert not np.iscomplexobj(tangent.direction)
    for current in range(canonical.physical_dim):
        horizontal = np.zeros(
            (canonical.bond_dim, canonical.bond_dim),
            dtype=np.result_type(canonical.TL.dtype, tangent.direction.dtype),
        )
        for previous in range(canonical.physical_dim):
            horizontal += (
                canonical.TL[:, previous, current, :].conj().T
                @ tangent.direction[:, previous, current, :]
            )
        np.testing.assert_allclose(horizontal, 0.0, atol=1.0e-10)
    slope = float(np.real(np.vdot(gradient, tangent.direction)))
    np.testing.assert_allclose(
        slope,
        -(tangent.residual_norm**2),
        rtol=1.0e-9,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("real", [True, False])
def test_conditional_tangent_residual_matches_dense_gram_oracle(real):
    canonical = conditional_canonicalize(
        random_uniform_letta(
            physical_dim=2,
            bond_dim=2,
            seed=44,
            real=real,
        )
    )
    gradient = energy_gradient(canonical.state, _tfim_bond())
    tangent = conditional_tangent_direction(canonical, gradient, real=real)
    packed_gradient = (
        np.real(gradient).reshape(-1)
        if real
        else np.concatenate(
            [np.real(gradient).reshape(-1), np.imag(gradient).reshape(-1)]
        )
    )
    metric = tangent_gram_matrix(canonical.state, real=real)
    _dense_direction, dense_residual, dense_rank = natural_gradient(
        packed_gradient,
        metric,
    )

    assert dense_rank == tangent.reduced_dimension
    np.testing.assert_allclose(
        tangent.residual_norm,
        dense_residual,
        rtol=2.0e-7,
        atol=2.0e-9,
    )


def test_conditional_tangent_residual_is_gauge_invariant():
    rng = np.random.default_rng(45)
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=45)
    gauges = []
    for _physical in range(state.physical_dim):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        gauges.append(matrix + 3.0 * np.eye(2))
    transformed = state.gauge_transform(np.asarray(gauges))

    residuals = []
    slopes = []
    for candidate in (state, transformed):
        canonical = conditional_canonicalize(candidate)
        gradient = energy_gradient(canonical.state, _tfim_bond())
        tangent = conditional_tangent_direction(canonical, gradient)
        residuals.append(tangent.residual_norm)
        slopes.append(float(np.real(np.vdot(gradient, tangent.direction))))

    np.testing.assert_allclose(residuals[0], residuals[1], rtol=2.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=2.0e-8, atol=1.0e-10)


def test_uniform_letta_is_invariant_under_physical_dependent_virtual_gauge():
    rng = np.random.default_rng(7)
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=7)
    gauges = []
    for _physical in range(2):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        matrix += 2.0 * np.eye(2)
        gauges.append(matrix)
    transformed = state.gauge_transform(np.asarray(gauges))
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)

    np.testing.assert_allclose(
        transformed.periodic_amplitude((0, 1, 1, 0)),
        state.periodic_amplitude((0, 1, 1, 0)),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        energy_density(transformed, h),
        energy_density(state, h),
        atol=1.0e-10,
    )


def test_expand_uniform_letta_activates_a_larger_real_virtual_space():
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=8, real=True)

    expanded = expand_uniform_letta(
        state,
        3,
        seed=9,
        relative_noise=3.0e-2,
    )

    assert expanded.tensor.shape == (3, 2, 2, 3)
    assert not np.iscomplexobj(expanded.tensor)
    np.testing.assert_allclose(np.linalg.norm(expanded.tensor), 1.0)
    assert np.linalg.norm(expanded.tensor[2, :, :, :]) > 0.0
    assert np.linalg.norm(expanded.tensor[:, :, :, 2]) > 0.0


def test_expand_uniform_letta_rejects_nonincreasing_bond_dimension():
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=8)

    with pytest.raises(ValueError, match="larger"):
        expand_uniform_letta(state, 2)


def test_uniform_letta_product_state_observables():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    tensor = np.full((1, 2, 2, 1), 1.0 / np.sqrt(2.0))
    state = UniformLETTA(tensor)

    np.testing.assert_allclose(one_site_expectation(state, x), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(one_site_expectation(state, z), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        two_site_expectation(state, np.kron(z, z)),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        energy_density(state, _tfim_bond()),
        -1.5,
        atol=1.0e-12,
    )


def test_transfer_fixed_points_satisfy_normalized_equations():
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=9)
    data = transfer_data(state)
    tensor = data.structured_tensor
    left_action = np.zeros_like(data.left_fixed_point)
    right_action = np.zeros_like(data.right_fixed_point)
    for physical in range(state.physical_dim):
        site = tensor[:, physical, :]
        left_action += site.conj().T @ data.left_fixed_point @ site
        right_action += site @ data.right_fixed_point @ site.conj().T

    np.testing.assert_allclose(left_action, data.left_fixed_point, atol=1.0e-10)
    np.testing.assert_allclose(right_action, data.right_fixed_point, atol=1.0e-10)
    np.testing.assert_allclose(
        np.trace(data.left_fixed_point @ data.right_fixed_point),
        1.0,
        atol=1.0e-12,
    )


def test_noninjective_uniform_letta_is_rejected_explicitly():
    tensor = np.zeros((1, 2, 2, 1))
    tensor[0, 0, 0, 0] = 1.0
    tensor[0, 1, 1, 0] = 1.0

    with pytest.raises(ValueError, match="noninjective"):
        transfer_data(UniformLETTA(tensor))


def test_transfer_data_rejects_invalid_injectivity_tolerance():
    state = random_uniform_letta(physical_dim=2, bond_dim=1, seed=1)

    with pytest.raises(ValueError, match="injectivity_tolerance"):
        transfer_data(state, injectivity_tolerance=-1.0)


def test_real_analytic_gradient_matches_central_differences():
    rng = np.random.default_rng(21)
    state = random_uniform_letta(
        physical_dim=2,
        bond_dim=2,
        seed=21,
        real=True,
    )
    hamiltonian = rng.normal(size=(4, 4))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    analytic = energy_gradient(state, hamiltonian)
    step = 2.0e-6

    for flat_index in (0, 3, 7, 12):
        direction = np.zeros(state.tensor.size)
        direction[flat_index] = 1.0
        direction = direction.reshape(state.tensor.shape)
        forward = energy_density(
            UniformLETTA(state.tensor + step * direction),
            hamiltonian,
        )
        backward = energy_density(
            UniformLETTA(state.tensor - step * direction),
            hamiltonian,
        )
        finite_difference = (forward - backward) / (2.0 * step)
        np.testing.assert_allclose(
            analytic.reshape(-1)[flat_index],
            finite_difference,
            rtol=2.0e-5,
            atol=2.0e-7,
        )


def test_complex_analytic_gradient_matches_real_and_imaginary_directions():
    rng = np.random.default_rng(22)
    state = random_uniform_letta(physical_dim=2, bond_dim=1, seed=22)
    hamiltonian = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    analytic = energy_gradient(state, hamiltonian)
    direction = np.zeros(state.tensor.shape, dtype=complex)
    direction.reshape(-1)[2] = 1.0
    step = 2.0e-6

    real_fd = (
        energy_density(UniformLETTA(state.tensor + step * direction), hamiltonian)
        - energy_density(UniformLETTA(state.tensor - step * direction), hamiltonian)
    ) / (2.0 * step)
    imag_fd = (
        energy_density(
            UniformLETTA(state.tensor + 1j * step * direction),
            hamiltonian,
        )
        - energy_density(
            UniformLETTA(state.tensor - 1j * step * direction),
            hamiltonian,
        )
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic.reshape(-1)[2].real, real_fd, atol=2.0e-7)
    np.testing.assert_allclose(analytic.reshape(-1)[2].imag, imag_fd, atol=2.0e-7)


def test_analytic_gradient_annihilates_infinitesimal_letta_gauge_direction():
    rng = np.random.default_rng(23)
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=23)
    hamiltonian = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    generators = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    direction = np.empty_like(state.tensor)
    for previous in range(state.physical_dim):
        for current in range(state.physical_dim):
            site = state.tensor[:, previous, current, :]
            direction[:, previous, current, :] = (
                -generators[previous] @ site
                + site @ generators[current]
            )

    gradient = energy_gradient(state, hamiltonian)
    directional_derivative = np.real(np.vdot(gradient, direction))

    np.testing.assert_allclose(directional_derivative, 0.0, atol=2.0e-9)


@pytest.mark.parametrize("real, expected_rank", [(True, 8), (False, 16)])
def test_tangent_gram_matrix_removes_scale_phase_and_gauge(real, expected_rank):
    state = random_uniform_letta(
        physical_dim=2,
        bond_dim=2,
        seed=24,
        real=real,
    )

    metric = tangent_gram_matrix(state, real=real)
    eigenvalues = np.linalg.eigvalsh(metric)
    cutoff = 1.0e-10 * eigenvalues[-1]

    np.testing.assert_allclose(metric, metric.T, atol=1.0e-12)
    assert eigenvalues[0] > -1.0e-11
    assert np.count_nonzero(eigenvalues > cutoff) == expected_rank


def test_letta_thermodynamic_contraction_matches_structured_uniform_mps():
    rng = np.random.default_rng(11)
    state = random_uniform_letta(physical_dim=2, bond_dim=2, seed=11)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    structured_mps = UniformMPS(
        state.structured_mps_tensor().transpose(1, 0, 2)
    )

    np.testing.assert_allclose(
        energy_density(state, h),
        structured_mps.energy_density(h),
        atol=1.0e-10,
    )


def test_vuletta_rejects_fractional_bond_dimension():
    with pytest.raises(ValueError, match="bond_dim"):
        vuletta(_tfim_bond(), bond_dim=1.5)


def test_vuletta_rejects_invalid_function_evaluation_limit():
    with pytest.raises(ValueError, match="max_function_evaluations"):
        vuletta(
            _tfim_bond(),
            bond_dim=1,
            options=VULETTAOptions(max_function_evaluations=0),
        )


def test_vuletta_rejects_initial_bond_dimension_mismatch():
    initial = random_uniform_letta(physical_dim=2, bond_dim=1, seed=2, real=True)

    with pytest.raises(ValueError, match="bond dimension"):
        vuletta(_tfim_bond(), bond_dim=2, initial=initial)


def test_vuletta_finds_nontrivial_pair_tied_ising_state():
    result = vuletta(
        _tfim_bond(),
        bond_dim=1,
        seed=3,
        options=VULETTAOptions(
            max_iterations=100,
            tolerance=1.0e-8,
            finite_difference_scheme="3-point",
        ),
    )

    assert result.converged
    np.testing.assert_allclose(result.energy, -5.0 / 3.0, atol=1.0e-9)
    assert result.residual_norm < 1.0e-6
    np.testing.assert_allclose(
        result.gradient_norm,
        result.parameter_norm * result.coordinate_gradient_norm,
        atol=1.0e-14,
    )
    assert result.energy < -1.56
    assert result.gradient_method == "analytic"
    assert result.update_method == "natural_gradient"
    assert result.metric_rank == 2


def test_vuletta_does_not_claim_convergence_above_stationarity_tolerance():
    result = vuletta(
        _tfim_bond(),
        bond_dim=1,
        seed=3,
        options=VULETTAOptions(
            max_iterations=100,
            tolerance=1.0e-8,
            stationarity_tolerance=1.0e-20,
        ),
    )

    assert not result.converged
    assert "exceeds stationarity_tolerance" in result.message


def test_bond_two_vuletta_approaches_exact_infinite_ising_energy():
    exact_energy = -1.6719262215361948
    result = vuletta(
        _tfim_bond(),
        bond_dim=2,
        seed=3,
        options=VULETTAOptions(
            max_iterations=120,
            tolerance=1.0e-8,
            finite_difference_scheme="3-point",
        ),
    )

    assert result.converged
    assert abs(result.energy - exact_energy) < 1.0e-5
    np.testing.assert_allclose(
        result.energy,
        energy_density(result.state, _tfim_bond()),
        atol=1.0e-12,
    )

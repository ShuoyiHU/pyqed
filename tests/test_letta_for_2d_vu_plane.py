import numpy as np
import opt_einsum as oe
import pytest
import warnings

from scipy.optimize import OptimizeResult, OptimizeWarning

from pyqed._letta_for_2d_vu import (
    PlaneEnvironmentOptions,
    UniformPlaneLETTA,
    VULETTA2DOptions,
    contract_plane_window,
    double_layer_cell,
    plane_energy_density,
    plane_observables,
    random_uniform_plane_letta,
    tfim_square_lattice,
    vuletta_plane,
)


def test_plane_state_has_four_virtual_and_three_tied_physical_legs():
    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=3,
        seed=4,
        real=True,
    )

    assert state.tensor.shape == (3, 3, 3, 3, 2, 2, 2)
    assert state.bond_dim == 3
    assert state.local_physical_dim == 2
    assert state.parameter_count == 3**4 * 2**3


def test_plane_periodic_amplitude_uses_right_and_down_physical_ties():
    tensor = np.arange(1.0, 9.0).reshape(1, 1, 1, 1, 2, 2, 2)
    state = UniformPlaneLETTA(tensor)
    configuration = np.array([[0, 1], [1, 0]])

    expected = 1.0
    for row in range(2):
        for column in range(2):
            expected *= tensor[
                0,
                0,
                0,
                0,
                configuration[row, column],
                configuration[row, (column + 1) % 2],
                configuration[(row + 1) % 2, column],
            ]

    np.testing.assert_allclose(
        state.periodic_amplitude(configuration),
        expected,
    )


def test_double_layer_cell_has_finite_directional_bond_dimension():
    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=2,
        seed=7,
        real=True,
    )

    cell = double_layer_cell(state, np.eye(2))

    assert cell.shape == (16, 16, 16, 16)
    assert np.all(np.isfinite(cell))


def test_double_layer_cells_match_direct_periodic_letta_expectations():
    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=1,
        seed=9,
        real=True,
    )
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    identity_cell = double_layer_cell(state, identity)
    x_cell = double_layer_cell(state, x)
    size = 3
    center = (size // 2, size // 2)

    def contract_cells(replacements):
        horizontal = np.arange(size * size).reshape(size, size)
        vertical = horizontal + size * size
        operands = []
        for row in range(size):
            for column in range(size):
                cell = replacements.get((row, column), identity_cell)
                labels = (
                    int(horizontal[row, (column - 1) % size]),
                    int(horizontal[row, column]),
                    int(vertical[(row - 1) % size, column]),
                    int(vertical[row, column]),
                )
                operands.extend((cell, labels))
        return oe.contract(*operands, optimize="auto")

    norm = 0.0
    x_numerator = 0.0
    for flat in range(2 ** (size * size)):
        ket = np.array(
            np.unravel_index(flat, (2,) * (size * size))
        ).reshape(size, size)
        ket_amplitude = state.periodic_amplitude(ket)
        norm += abs(ket_amplitude) ** 2
        bra = ket.copy()
        bra[center] = 1 - bra[center]
        x_numerator += (
            np.conj(state.periodic_amplitude(bra)) * ket_amplitude
        )

    network_norm = contract_cells({})
    network_x = contract_cells({center: x_cell})
    np.testing.assert_allclose(network_x / network_norm, x_numerator / norm)


def test_boundary_mps_window_matches_exact_open_contraction_without_truncation():
    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=1,
        seed=11,
        real=True,
    )
    cell = double_layer_cell(state, np.eye(2))
    size = 3
    horizontal = np.arange(size * (size - 1)).reshape(size, size - 1)
    vertical = (
        np.arange((size - 1) * size).reshape(size - 1, size)
        + horizontal.size
    )
    next_label = horizontal.size + vertical.size
    boundary = np.eye(2).reshape(-1)
    boundary = boundary / np.linalg.norm(boundary)
    operands = []
    for row in range(size):
        for column in range(size):
            labels = []
            for internal in (
                None if column == 0 else horizontal[row, column - 1],
                None if column == size - 1 else horizontal[row, column],
                None if row == 0 else vertical[row - 1, column],
                None if row == size - 1 else vertical[row, column],
            ):
                if internal is None:
                    label = next_label
                    next_label += 1
                    operands.extend((boundary, (label,)))
                else:
                    label = int(internal)
                labels.append(label)
            operands.extend((cell, tuple(labels)))
    exact = oe.contract(*operands, optimize="auto")

    approximate = contract_plane_window(
        state,
        cell,
        size,
        boundary_bond_dim=128,
        cutoff=0.0,
    )
    reconstructed = (
        approximate.phase * np.exp(approximate.log_magnitude)
    )

    np.testing.assert_allclose(reconstructed, exact, rtol=1.0e-11, atol=1.0e-13)


def test_uniform_plus_state_has_exact_infinite_plane_observables():
    state = UniformPlaneLETTA(np.ones((1, 1, 1, 1, 2, 2, 2)))
    model = tfim_square_lattice(coupling=1.0, field=1.5)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3, 5),
        boundary_bond_dim=16,
        convergence_tolerance=1.0e-10,
    )

    observables = plane_observables(state, environment=environment)

    np.testing.assert_allclose(
        observables.transverse_magnetization,
        1.0,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(observables.horizontal_zz, 0.0, atol=1.0e-11)
    np.testing.assert_allclose(observables.vertical_zz, 0.0, atol=1.0e-11)
    np.testing.assert_allclose(
        plane_energy_density(state, model, environment=environment),
        -1.5,
        atol=1.0e-10,
    )
    assert observables.window_converged
    assert observables.boundary_converged
    assert observables.converged
    assert observables.window_change <= environment.convergence_tolerance
    assert observables.boundary_change <= environment.convergence_tolerance


def test_plane_environment_requires_window_and_boundary_convergence():
    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=1,
        seed=1,
        real=True,
    )
    environment = PlaneEnvironmentOptions(
        window_sizes=(3, 5),
        boundary_bond_dim=16,
        boundary_bond_dims=(8, 16),
        convergence_tolerance=1.0,
        boundary_convergence_tolerance=1.0e-14,
    )

    observables = plane_observables(state, environment=environment)

    assert observables.window_converged
    assert not observables.boundary_converged
    assert not observables.converged
    assert observables.boundary_change > (
        environment.boundary_convergence_tolerance
    )


def test_uniform_z_product_state_counts_both_plane_bond_directions():
    tensor = np.zeros((1, 1, 1, 1, 2, 2, 2))
    tensor[0, 0, 0, 0, 0, 0, 0] = 1.0
    state = UniformPlaneLETTA(tensor)
    model = tfim_square_lattice(coupling=0.7, field=1.2)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3,),
        boundary_bond_dim=8,
    )

    observables = plane_observables(state, environment=environment)

    np.testing.assert_allclose(observables.transverse_magnetization, 0.0)
    np.testing.assert_allclose(observables.horizontal_zz, 1.0)
    np.testing.assert_allclose(observables.vertical_zz, 1.0)
    np.testing.assert_allclose(
        plane_energy_density(state, model, environment=environment),
        -1.4,
    )


def test_plane_vuletta_lowers_a_fixed_environment_energy():
    model = tfim_square_lattice(coupling=1.0, field=1.5)
    initial = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=1,
        seed=12,
        real=True,
    )
    environment = PlaneEnvironmentOptions(
        window_sizes=(3,),
        boundary_bond_dim=12,
    )
    initial_energy = plane_energy_density(
        initial,
        model,
        environment=environment,
    )

    result = vuletta_plane(
        model,
        initial=initial,
        environment=environment,
        options=VULETTA2DOptions(
            max_iterations=5,
            function_tolerance=1.0e-12,
            finite_difference_step=2.0e-5,
            gradient_method="finite_difference",
        ),
    )

    assert result.energy_density < initial_energy - 1.0e-4
    assert result.state.tensor.shape == initial.tensor.shape
    assert np.isfinite(result.gradient_norm)


def test_autodiff_plane_gradient_matches_directional_finite_difference():
    from pyqed._letta_for_2d_vu.plane_autodiff import (
        make_plane_energy_value_and_gradient,
    )

    state = random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=2,
        seed=4,
        real=True,
    )
    model = tfim_square_lattice(coupling=1.0, field=0.2)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3,),
        boundary_bond_dim=8,
    )
    evaluate = make_plane_energy_value_and_gradient(
        state.tensor.shape,
        model,
        environment,
    )
    parameters = state.tensor.reshape(-1)

    energy, gradient = evaluate(parameters)

    np.testing.assert_allclose(
        energy,
        plane_energy_density(state, model, environment=environment),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert np.all(np.isfinite(gradient))
    rng = np.random.default_rng(17)
    direction = rng.normal(size=parameters.shape)
    direction /= np.linalg.norm(direction)
    step = 1.0e-5

    def displaced_energy(distance):
        tensor = (parameters + distance * direction).reshape(
            state.tensor.shape
        )
        return plane_energy_density(
            UniformPlaneLETTA(tensor),
            model,
            environment=environment,
        )

    directional_difference = (
        displaced_energy(step) - displaced_energy(-step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.dot(gradient, direction),
        directional_difference,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_plane_solver_autodiff_avoids_parameterwise_evaluations():
    model = tfim_square_lattice(coupling=1.0, field=1.5)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3,),
        boundary_bond_dim=8,
    )

    result = vuletta_plane(
        model,
        bond_dim=1,
        seed=12,
        real=True,
        environment=environment,
        options=VULETTA2DOptions(
            max_iterations=1,
            gradient_method="autodiff",
        ),
    )

    assert result.function_evaluations < 17
    assert np.isfinite(result.gradient_norm)


def test_plane_solver_separates_environment_and_optimizer_convergence():
    result = vuletta_plane(
        tfim_square_lattice(coupling=0.0, field=1.0),
        bond_dim=1,
        seed=3,
        real=True,
        environment=PlaneEnvironmentOptions(
            window_sizes=(3,),
            boundary_bond_dim=8,
        ),
        options=VULETTA2DOptions(
            max_iterations=5,
            function_tolerance=1.0e-12,
            gradient_tolerance=1.0e-5,
        ),
    )

    assert result.optimizer_converged
    assert not result.environment_converged
    assert not result.converged
    assert "environment" in result.message.lower()


def test_plane_solver_autodiff_callback_reuses_compiled_energy(monkeypatch):
    from pyqed._letta_for_2d_vu import plane_solver

    calls = 0
    original = plane_solver.plane_energy_density

    def counted_energy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        plane_solver,
        "plane_energy_density",
        counted_energy,
    )
    result = vuletta_plane(
        tfim_square_lattice(coupling=1.0, field=1.5),
        bond_dim=1,
        seed=12,
        real=True,
        environment=PlaneEnvironmentOptions(
            window_sizes=(3,),
            boundary_bond_dim=8,
        ),
        options=VULETTA2DOptions(
            max_iterations=1,
            gradient_method="autodiff",
        ),
    )

    assert result.iterations == 1
    assert calls == 0


def test_high_field_solver_cannot_exploit_truncated_environment():
    model = tfim_square_lattice(coupling=1.0, field=10.0)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3, 5),
        boundary_bond_dim=24,
    )

    result = vuletta_plane(
        model,
        bond_dim=1,
        seed=4,
        real=True,
        environment=environment,
        options=VULETTA2DOptions(
            max_iterations=30,
            function_tolerance=1.0e-12,
        ),
    )

    observables = result.observables
    assert -12.0 <= result.energy_density <= 12.0
    assert abs(observables.transverse_magnetization) <= 1.0
    assert abs(observables.horizontal_zz) <= 1.0
    assert abs(observables.vertical_zz) <= 1.0
    np.testing.assert_allclose(result.energy_density, -10.05, atol=2.0e-2)
    np.testing.assert_allclose(
        observables.transverse_magnetization,
        0.995,
        atol=1.0e-2,
    )
    np.testing.assert_allclose(observables.horizontal_zz, 0.05, atol=2.0e-2)
    np.testing.assert_allclose(observables.vertical_zz, 0.05, atol=2.0e-2)


def test_unphysical_truncated_environment_is_rejected():
    tensor = np.array(
        [
            0.01643372,
            0.92154710,
            -0.07436242,
            0.25662853,
            -0.00248098,
            0.24593500,
            -0.07875364,
            -0.11135789,
        ]
    ).reshape(1, 1, 1, 1, 2, 2, 2)
    state = UniformPlaneLETTA(tensor)
    environment = PlaneEnvironmentOptions(
        window_sizes=(5,),
        boundary_bond_dim=24,
    )

    with pytest.raises(ValueError, match="unphysical"):
        plane_observables(state, environment=environment)


def test_plane_solver_uses_only_supported_scipy_options():
    model = tfim_square_lattice(coupling=0.0, field=1.0)
    environment = PlaneEnvironmentOptions(
        window_sizes=(3,),
        boundary_bond_dim=8,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vuletta_plane(
            model,
            bond_dim=1,
            seed=3,
            real=True,
            environment=environment,
            options=VULETTA2DOptions(max_iterations=1),
        )

    assert not any(
        issubclass(item.category, OptimizeWarning) for item in caught
    )


def test_plane_solver_restarts_nonstationary_termination_with_remaining_budget(
    monkeypatch,
):
    from pyqed._letta_for_2d_vu import plane_solver

    calls = []

    def fake_minimize(
        objective,
        parameters,
        *,
        method,
        jac,
        callback,
        options,
    ):
        calls.append(dict(options))
        call_index = len(calls)
        return OptimizeResult(
            x=np.asarray(parameters).copy(),
            fun=-1.0,
            jac=(
                np.zeros_like(parameters)
                if call_index == 3
                else np.full_like(parameters, 1.0e-2)
            ),
            success=True,
            status=0,
            message=(
                "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
                if call_index == 1
                else "ABNORMAL: "
                if call_index == 2
                else "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
            ),
            nit=2 if call_index == 1 else 0 if call_index == 2 else 1,
            nfev=3 if call_index == 1 else 2,
        )

    monkeypatch.setattr(plane_solver, "minimize", fake_minimize)
    result = vuletta_plane(
        tfim_square_lattice(coupling=0.0, field=1.0),
        bond_dim=1,
        seed=3,
        real=True,
        environment=PlaneEnvironmentOptions(
            window_sizes=(3,),
            boundary_bond_dim=8,
        ),
        options=VULETTA2DOptions(
            max_iterations=5,
            max_function_evaluations=10,
            function_tolerance=1.0e-12,
            gradient_tolerance=1.0e-5,
        ),
    )

    assert len(calls) == 3
    assert calls[0]["maxiter"] == 5
    assert calls[1]["maxiter"] == 3
    assert calls[2]["maxiter"] == 3
    assert calls[0]["maxfun"] == 10
    assert calls[1]["maxfun"] == 7
    assert calls[2]["maxfun"] == 5
    assert calls[0]["maxls"] == 20
    assert calls[1]["maxls"] == 40
    assert calls[2]["maxls"] == 60
    assert all(call["ftol"] == 1.0e-12 for call in calls)
    assert all(call["gtol"] == 1.0e-5 for call in calls)
    assert result.iterations == 3
    assert result.function_evaluations == 7
    assert result.optimizer_converged
    assert "restarted 2 times" in result.message

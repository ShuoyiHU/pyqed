"""Direct variational optimizer for a uniform LETTA."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.optimize import minimize

from .gradients import (
    conditional_tangent_direction,
    energy_and_gradient,
    natural_gradient,
    tangent_gram_matrix,
)
from .operators import _as_two_site_operator, energy_density
from .state import (
    ConditionalCanonicalLETTA,
    UniformLETTA,
    conditional_canonicalize,
    random_uniform_letta,
)


@dataclass(frozen=True)
class VULETTAOptions:
    max_iterations: int = 200
    max_function_evaluations: int | None = None
    tolerance: float = 1.0e-8
    energy_tolerance: float = 1.0e-13
    update_method: str = "conditional_canonical"
    gradient_method: str = "analytic"
    finite_difference_scheme: str = "3-point"
    max_line_search_steps: int = 30
    max_parameter_step: float = 2.0
    metric_rcond: float = 1.0e-10
    canonical_rcond: float = 1.0e-12
    armijo_coefficient: float = 1.0e-4
    stationarity_tolerance: float = 1.0e-6
    verbosity: int = 0


@dataclass(frozen=True)
class VULETTAIteration:
    iteration: int
    energy: float
    energy_change: float | None
    residual_norm: float | None = None
    canonical_residual_norm: float | None = None
    step_size: float | None = None


@dataclass(frozen=True)
class VULETTAResult:
    state: UniformLETTA
    energy: float
    converged: bool
    iterations: int
    function_evaluations: int
    gradient_norm: float
    coordinate_gradient_norm: float
    parameter_norm: float
    residual_norm: float
    metric_rank: int | None
    canonical_state: ConditionalCanonicalLETTA | None
    canonical_residual_norm: float | None
    reduced_dimension: int | None
    update_method: str
    gradient_method: str
    history: tuple[VULETTAIteration, ...]
    message: str


def _pack_tensor(tensor, real):
    tensor = np.asarray(tensor)
    if real:
        if np.max(np.abs(np.imag(tensor))) > 1.0e-12:
            raise ValueError("a complex initial tensor cannot be used with real=True.")
        return np.real(tensor).reshape(-1)
    return np.concatenate([np.real(tensor).reshape(-1), np.imag(tensor).reshape(-1)])


def _unpack_tensor(parameters, shape, real):
    parameters = np.asarray(parameters)
    size = int(np.prod(shape))
    if real:
        tensor = parameters.reshape(shape)
    else:
        tensor = (
            parameters[:size] + 1j * parameters[size:]
        ).reshape(shape)
    norm = np.linalg.norm(tensor)
    if norm <= np.finfo(float).tiny:
        raise ValueError("the variational LETTA tensor became numerically zero.")
    return tensor / norm


def _pack_gradient(gradient, real):
    gradient = np.asarray(gradient)
    if real:
        return np.real(gradient).reshape(-1)
    return np.concatenate(
        [np.real(gradient).reshape(-1), np.imag(gradient).reshape(-1)]
    )


def _initial_state(initial, physical_dim, bond_dim, seed, real):
    if initial is None:
        bond_dim = 2 if bond_dim is None else bond_dim
        return random_uniform_letta(
            physical_dim,
            bond_dim,
            seed=seed,
            real=real,
        )
    state = initial if isinstance(initial, UniformLETTA) else UniformLETTA(initial)
    if state.physical_dim != physical_dim:
        raise ValueError("the initial LETTA has an incompatible physical dimension.")
    if bond_dim is not None and state.bond_dim != bond_dim:
        raise ValueError("the initial LETTA has an incompatible bond dimension.")
    return state.normalized_parameters()


def vuletta(
    hamiltonian,
    *,
    physical_dim=None,
    bond_dim=None,
    initial=None,
    seed=None,
    real=None,
    options=None,
):
    """Minimize an infinite-chain energy directly over a uniform LETTA tensor."""

    options = VULETTAOptions() if options is None else options
    if not isinstance(options, VULETTAOptions):
        raise TypeError("options must be a VULETTAOptions instance.")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if options.max_function_evaluations is not None:
        try:
            max_function_evaluations = index(options.max_function_evaluations)
        except TypeError as error:
            raise ValueError(
                "max_function_evaluations must be an integer."
            ) from error
        if max_function_evaluations <= 0:
            raise ValueError("max_function_evaluations must be positive.")
    if (
        options.tolerance <= 0.0
        or options.energy_tolerance <= 0.0
        or options.stationarity_tolerance <= 0.0
        or options.max_parameter_step <= 0.0
        or options.metric_rcond <= 0.0
        or options.canonical_rcond <= 0.0
        or not 0.0 < options.armijo_coefficient < 1.0
    ):
        raise ValueError("solver tolerances must be positive.")
    if options.finite_difference_scheme not in {"2-point", "3-point"}:
        raise ValueError("finite_difference_scheme must be '2-point' or '3-point'.")
    if options.gradient_method not in {"analytic", "finite_difference"}:
        raise ValueError(
            "gradient_method must be 'analytic' or 'finite_difference'."
        )
    if options.update_method not in {
        "conditional_canonical",
        "natural_gradient",
        "lbfgs",
    }:
        raise ValueError(
            "update_method must be 'conditional_canonical', "
            "'natural_gradient', or 'lbfgs'."
        )
    if (
        options.update_method in {"conditional_canonical", "natural_gradient"}
        and options.gradient_method != "analytic"
    ):
        raise ValueError(
            "conditional and natural-gradient updates require the analytic gradient."
        )

    array = np.asarray(hamiltonian)
    if array.ndim == 2:
        inferred = int(round(np.sqrt(array.shape[0])))
    elif array.ndim == 4:
        inferred = int(array.shape[0])
    else:
        raise ValueError("the Hamiltonian must be rank two or rank four.")
    if physical_dim is not None and int(physical_dim) != inferred:
        raise ValueError("physical_dim is inconsistent with the Hamiltonian.")
    physical_dim = inferred
    h = _as_two_site_operator(array, physical_dim)
    matrix = h.reshape(physical_dim**2, physical_dim**2)
    if not np.allclose(matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("the Hamiltonian must be Hermitian.")

    if bond_dim is not None:
        try:
            bond_dim = index(bond_dim)
        except TypeError as error:
            raise ValueError("bond_dim must be an integer.") from error
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
    if real is None:
        real = not np.iscomplexobj(matrix)
    real = bool(real)
    state = _initial_state(initial, physical_dim, bond_dim, seed, real)
    shape = state.tensor.shape
    parameters = _pack_tensor(state.tensor, real)
    if options.max_function_evaluations is None:
        if options.update_method in {"conditional_canonical", "natural_gradient"}:
            max_function_evaluations = (
                1
                + options.max_iterations
                * (1 + options.max_line_search_steps)
            )
        elif options.gradient_method == "analytic":
            evaluations_per_line_search = 1
        else:
            stencil_size = (
                2 if options.finite_difference_scheme == "3-point" else 1
            )
            evaluations_per_line_search = 1 + stencil_size * parameters.size
        if options.update_method not in {"conditional_canonical", "natural_gradient"}:
            max_function_evaluations = (
                1
                + options.max_iterations
                * options.max_line_search_steps
                * evaluations_per_line_search
            )
    history = []
    previous_energy = None

    def objective(values):
        tensor = _unpack_tensor(values, shape, real)
        return energy_density(UniformLETTA(tensor), h)

    def objective_with_gradient(values):
        parameter_norm = np.linalg.norm(values)
        tensor = _unpack_tensor(values, shape, real)
        energy, tensor_gradient = energy_and_gradient(UniformLETTA(tensor), h)
        packed_tensor = _pack_tensor(tensor, real)
        packed_gradient = _pack_gradient(tensor_gradient, real)
        radial_component = np.dot(packed_tensor, packed_gradient)
        coordinate_gradient = (
            packed_gradient - radial_component * packed_tensor
        ) / parameter_norm
        return energy, coordinate_gradient

    def callback(values):
        nonlocal previous_energy
        energy = objective(values)
        change = None if previous_energy is None else abs(energy - previous_energy)
        record = VULETTAIteration(
            iteration=len(history) + 1,
            energy=energy,
            energy_change=change,
        )
        history.append(record)
        previous_energy = energy
        if options.verbosity:
            change_text = "-" if change is None else f"{change:.3e}"
            print(
                f"VULETTA {record.iteration:4d}  energy={energy: .14f}  "
                f"dE={change_text}"
            )

    if options.update_method == "conditional_canonical":
        function_evaluations = 0
        success = False
        message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        iterations = 0
        canonical_state = conditional_canonicalize(
            state,
            rcond=options.canonical_rcond,
        )
        residual_norm = np.inf
        canonical_residual_norm = max(
            canonical_state.left_isometry_error(),
            canonical_state.right_isometry_error(),
            canonical_state.center_error(),
        )
        reduced_dimension = None

        for iteration in range(1, options.max_iterations + 1):
            current_state = canonical_state.state
            energy, tensor_gradient = energy_and_gradient(current_state, h)
            function_evaluations += 1
            tangent = conditional_tangent_direction(
                canonical_state,
                tensor_gradient,
                real=real,
                rcond=options.canonical_rcond,
            )
            residual_norm = tangent.residual_norm
            reduced_dimension = tangent.reduced_dimension
            canonical_residual_norm = max(
                canonical_state.left_isometry_error(),
                canonical_state.right_isometry_error(),
                canonical_state.center_error(),
            )
            iterations = iteration - 1
            if (
                residual_norm <= options.stationarity_tolerance
                and canonical_residual_norm <= options.stationarity_tolerance
            ):
                success = True
                message = (
                    "CONVERGENCE: CONDITIONAL TANGENT AND CANONICAL "
                    "RESIDUALS <= STATIONARITY TOLERANCE"
                )
                break

            descent = np.asarray(tangent.direction).copy()
            coordinate_step_norm = residual_norm
            if coordinate_step_norm > options.max_parameter_step:
                scale = options.max_parameter_step / coordinate_step_norm
                descent *= scale
            slope = float(np.real(np.vdot(tensor_gradient, descent)))
            if not np.isfinite(slope) or slope >= 0.0:
                message = "STOP: CONDITIONAL TANGENT DIRECTION IS NOT DESCENDING"
                break

            step_size = 1.0
            accepted = False
            trial_energy = energy
            trial_tensor = current_state.tensor
            for _line_search in range(options.max_line_search_steps):
                if function_evaluations >= max_function_evaluations:
                    message = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
                    break
                trial_tensor = current_state.tensor + step_size * descent
                try:
                    trial_energy = energy_density(UniformLETTA(trial_tensor), h)
                except ValueError:
                    trial_energy = np.inf
                function_evaluations += 1
                if trial_energy <= (
                    energy + options.armijo_coefficient * step_size * slope
                ):
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                if function_evaluations < max_function_evaluations:
                    message = "STOP: ARMIJO LINE SEARCH FAILED"
                break

            canonical_state = conditional_canonicalize(
                UniformLETTA(trial_tensor),
                rcond=options.canonical_rcond,
            )
            change = abs(trial_energy - energy)
            canonical_residual_norm = max(
                canonical_state.left_isometry_error(),
                canonical_state.right_isometry_error(),
                canonical_state.center_error(),
            )
            history.append(
                VULETTAIteration(
                    iteration=iteration,
                    energy=trial_energy,
                    energy_change=change,
                    residual_norm=residual_norm,
                    canonical_residual_norm=canonical_residual_norm,
                    step_size=step_size,
                )
            )
            iterations = iteration
            if options.verbosity:
                print(
                    f"VULETTA {iteration:4d}  energy={trial_energy: .14f}  "
                    f"dE={change:.3e}  residual={residual_norm:.3e}  "
                    f"canonical={canonical_residual_norm:.3e}  "
                    f"step={step_size:.3e}"
                )

        final_state = canonical_state.state
        final_energy, final_tensor_gradient = energy_and_gradient(final_state, h)
        function_evaluations += 1
        final_tangent = conditional_tangent_direction(
            canonical_state,
            final_tensor_gradient,
            real=real,
            rcond=options.canonical_rcond,
        )
        residual_norm = final_tangent.residual_norm
        reduced_dimension = final_tangent.reduced_dimension
        canonical_residual_norm = max(
            canonical_state.left_isometry_error(),
            canonical_state.right_isometry_error(),
            canonical_state.center_error(),
        )
        packed_tensor = _pack_tensor(final_state.tensor, real)
        packed_gradient = _pack_gradient(final_tensor_gradient, real)
        parameter_norm = float(np.linalg.norm(packed_tensor))
        radial_coefficient = np.dot(packed_tensor, packed_gradient) / (
            parameter_norm * parameter_norm
        )
        coordinate_gradient = packed_gradient - radial_coefficient * packed_tensor
        coordinate_gradient_norm = float(np.linalg.norm(coordinate_gradient))
        gradient_norm = parameter_norm * coordinate_gradient_norm
        converged = bool(
            (success or residual_norm <= options.stationarity_tolerance)
            and residual_norm <= options.stationarity_tolerance
            and canonical_residual_norm <= options.stationarity_tolerance
        )
        if converged and not success:
            message = (
                "CONVERGENCE: CONDITIONAL TANGENT AND CANONICAL "
                "RESIDUALS <= STATIONARITY TOLERANCE"
            )
        elif not converged:
            message += (
                f"; conditional tangent residual {residual_norm:.3e}, "
                f"canonical residual {canonical_residual_norm:.3e}, "
                "exceeds stationarity_tolerance "
                f"{options.stationarity_tolerance:.3e}"
            )
        return VULETTAResult(
            state=final_state,
            energy=final_energy,
            converged=converged,
            iterations=iterations,
            function_evaluations=function_evaluations,
            gradient_norm=gradient_norm,
            coordinate_gradient_norm=coordinate_gradient_norm,
            parameter_norm=parameter_norm,
            residual_norm=residual_norm,
            metric_rank=reduced_dimension,
            canonical_state=canonical_state,
            canonical_residual_norm=canonical_residual_norm,
            reduced_dimension=reduced_dimension,
            update_method=options.update_method,
            gradient_method=options.gradient_method,
            history=tuple(history),
            message=message,
        )

    if options.update_method == "natural_gradient":
        function_evaluations = 0
        success = False
        message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        metric_rank = None
        residual_norm = np.inf
        iterations = 0

        for iteration in range(1, options.max_iterations + 1):
            tensor = _unpack_tensor(parameters, shape, real)
            current_state = UniformLETTA(tensor)
            energy, tensor_gradient = energy_and_gradient(current_state, h)
            function_evaluations += 1
            packed_tensor = _pack_tensor(tensor, real)
            packed_gradient = _pack_gradient(tensor_gradient, real)
            radial_component = np.dot(packed_tensor, packed_gradient)
            coordinate_gradient = packed_gradient - radial_component * packed_tensor
            metric = tangent_gram_matrix(current_state, real=real)
            natural_direction, residual_norm, metric_rank = natural_gradient(
                coordinate_gradient,
                metric,
                rcond=options.metric_rcond,
            )
            coordinate_gradient_norm = float(np.linalg.norm(coordinate_gradient))
            iterations = iteration - 1
            if residual_norm <= options.stationarity_tolerance:
                success = True
                message = (
                    "CONVERGENCE: TANGENT-METRIC RESIDUAL <= "
                    "STATIONARITY TOLERANCE"
                )
                break

            descent = -natural_direction
            descent_norm = np.linalg.norm(descent)
            if descent_norm > options.max_parameter_step:
                descent *= options.max_parameter_step / descent_norm
            slope = float(np.dot(coordinate_gradient, descent))
            if not np.isfinite(slope) or slope >= 0.0:
                message = "STOP: NATURAL-GRADIENT DIRECTION IS NOT DESCENDING"
                break

            step_size = 1.0
            accepted = False
            trial_energy = energy
            for _line_search in range(options.max_line_search_steps):
                if function_evaluations >= max_function_evaluations:
                    message = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
                    break
                trial_parameters = parameters + step_size * descent
                trial_energy = objective(trial_parameters)
                function_evaluations += 1
                if trial_energy <= (
                    energy + options.armijo_coefficient * step_size * slope
                ):
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                if function_evaluations < max_function_evaluations:
                    message = "STOP: ARMIJO LINE SEARCH FAILED"
                break

            parameters = _pack_tensor(
                _unpack_tensor(trial_parameters, shape, real),
                real,
            )
            change = abs(trial_energy - energy)
            history.append(
                VULETTAIteration(
                    iteration=iteration,
                    energy=trial_energy,
                    energy_change=change,
                    residual_norm=residual_norm,
                    step_size=step_size,
                )
            )
            iterations = iteration
            if options.verbosity:
                print(
                    f"VULETTA {iteration:4d}  energy={trial_energy: .14f}  "
                    f"dE={change:.3e}  residual={residual_norm:.3e}  "
                    f"step={step_size:.3e}"
                )

        final_tensor = _unpack_tensor(parameters, shape, real)
        final_state = UniformLETTA(final_tensor)
        final_energy, final_tensor_gradient = energy_and_gradient(final_state, h)
        function_evaluations += 1
        final_packed_tensor = _pack_tensor(final_tensor, real)
        final_packed_gradient = _pack_gradient(final_tensor_gradient, real)
        final_coordinate_gradient = final_packed_gradient - np.dot(
            final_packed_tensor,
            final_packed_gradient,
        ) * final_packed_tensor
        final_metric = tangent_gram_matrix(final_state, real=real)
        _final_direction, residual_norm, metric_rank = natural_gradient(
            final_coordinate_gradient,
            final_metric,
            rcond=options.metric_rcond,
        )
        parameter_norm = float(np.linalg.norm(parameters))
        coordinate_gradient_norm = float(
            np.linalg.norm(final_coordinate_gradient)
        )
        gradient_norm = parameter_norm * coordinate_gradient_norm
        converged = bool(
            (success or residual_norm <= options.stationarity_tolerance)
            and residual_norm <= options.stationarity_tolerance
        )
        if converged and not success:
            message = (
                "CONVERGENCE: TANGENT-METRIC RESIDUAL <= "
                "STATIONARITY TOLERANCE"
            )
        elif not converged and residual_norm > options.stationarity_tolerance:
            message += (
                f"; tangent-metric residual {residual_norm:.3e} exceeds "
                f"stationarity_tolerance {options.stationarity_tolerance:.3e}"
            )
        return VULETTAResult(
            state=final_state,
            energy=final_energy,
            converged=converged,
            iterations=iterations,
            function_evaluations=function_evaluations,
            gradient_norm=gradient_norm,
            coordinate_gradient_norm=coordinate_gradient_norm,
            parameter_norm=parameter_norm,
            residual_norm=residual_norm,
            metric_rank=metric_rank,
            canonical_state=None,
            canonical_residual_norm=None,
            reduced_dimension=None,
            update_method=options.update_method,
            gradient_method=options.gradient_method,
            history=tuple(history),
            message=message,
        )

    optimizer_objective = (
        objective_with_gradient
        if options.gradient_method == "analytic"
        else objective
    )
    optimizer_jacobian = (
        True
        if options.gradient_method == "analytic"
        else options.finite_difference_scheme
    )
    optimized = minimize(
        optimizer_objective,
        parameters,
        method="L-BFGS-B",
        jac=optimizer_jacobian,
        callback=callback,
        options={
            "maxiter": int(options.max_iterations),
            "maxfun": int(max_function_evaluations),
            "gtol": float(options.tolerance),
            "ftol": float(options.energy_tolerance),
            "maxls": int(options.max_line_search_steps),
        },
    )
    final_tensor = _unpack_tensor(optimized.x, shape, real)
    final_state = UniformLETTA(final_tensor)
    final_energy = energy_density(final_state, h)
    parameter_norm = float(np.linalg.norm(np.asarray(optimized.x)))
    coordinate_gradient_norm = float(np.linalg.norm(np.asarray(optimized.jac)))
    gradient_norm = parameter_norm * coordinate_gradient_norm
    converged = bool(
        optimized.success
        and gradient_norm <= options.stationarity_tolerance
    )
    message = str(optimized.message)
    if optimized.success and not converged:
        message += (
            f"; projected gradient {gradient_norm:.3e} exceeds "
            f"stationarity_tolerance {options.stationarity_tolerance:.3e}"
        )
    return VULETTAResult(
        state=final_state,
        energy=final_energy,
        converged=converged,
        iterations=int(optimized.nit),
        function_evaluations=int(optimized.nfev),
        gradient_norm=gradient_norm,
        coordinate_gradient_norm=coordinate_gradient_norm,
        parameter_norm=parameter_norm,
        residual_norm=gradient_norm,
        metric_rank=None,
        canonical_state=None,
        canonical_residual_norm=None,
        reduced_dimension=None,
        update_method=options.update_method,
        gradient_method=options.gradient_method,
        history=tuple(history),
        message=message,
    )

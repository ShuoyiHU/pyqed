"""Direct variational optimization of a uniform LETTA on the 2D plane."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.optimize import minimize

from .plane_environment import PlaneEnvironmentOptions
from .plane_operators import (
    PlaneObservables,
    PlaneTFIM,
    UnreliablePlaneEnvironmentError,
    plane_energy_density,
    plane_observables,
)
from .plane_state import (
    UniformPlaneLETTA,
    random_uniform_plane_letta,
)


@dataclass(frozen=True)
class VULETTA2DOptions:
    max_iterations: int = 50
    max_function_evaluations: int | None = None
    function_tolerance: float = 1.0e-12
    gradient_tolerance: float = 1.0e-5
    finite_difference_step: float = 1.0e-5
    max_line_search_steps: int = 20
    max_restarts: int = 5
    gradient_method: str = "autodiff"
    verbosity: int = 0


@dataclass(frozen=True)
class VULETTA2DIteration:
    iteration: int
    energy_density: float
    energy_change: float | None


@dataclass(frozen=True)
class VULETTA2DResult:
    state: UniformPlaneLETTA
    energy_density: float
    observables: PlaneObservables
    converged: bool
    optimizer_converged: bool
    environment_converged: bool
    iterations: int
    function_evaluations: int
    gradient_norm: float
    history: tuple[VULETTA2DIteration, ...]
    message: str


def _validated_options(options):
    if options is None:
        options = VULETTA2DOptions()
    if not isinstance(options, VULETTA2DOptions):
        raise TypeError("options must be a VULETTA2DOptions instance.")
    try:
        max_iterations = index(options.max_iterations)
    except TypeError as error:
        raise ValueError("max_iterations must be an integer.") from error
    max_evaluations = options.max_function_evaluations
    if max_evaluations is not None:
        try:
            max_evaluations = index(max_evaluations)
        except TypeError as error:
            raise ValueError(
                "max_function_evaluations must be an integer."
            ) from error
    try:
        max_line_search_steps = index(options.max_line_search_steps)
    except TypeError as error:
        raise ValueError(
            "max_line_search_steps must be an integer."
        ) from error
    try:
        max_restarts = index(options.max_restarts)
    except TypeError as error:
        raise ValueError("max_restarts must be an integer.") from error
    if max_iterations <= 0 or (
        max_evaluations is not None and max_evaluations <= 0
    ) or max_line_search_steps <= 0 or max_restarts < 0:
        raise ValueError("iteration and evaluation limits must be positive.")
    function_tolerance = float(options.function_tolerance)
    gradient_tolerance = float(options.gradient_tolerance)
    step = float(options.finite_difference_step)
    gradient_method = str(options.gradient_method)
    if (
        function_tolerance <= 0.0
        or gradient_tolerance <= 0.0
        or step <= 0.0
        or not np.isfinite(function_tolerance)
        or not np.isfinite(gradient_tolerance)
        or not np.isfinite(step)
    ):
        raise ValueError("solver tolerances must be finite and positive.")
    if gradient_method not in {"autodiff", "finite_difference"}:
        raise ValueError(
            "gradient_method must be 'autodiff' or 'finite_difference'."
        )
    return type(options)(
        max_iterations=max_iterations,
        max_function_evaluations=max_evaluations,
        function_tolerance=function_tolerance,
        gradient_tolerance=gradient_tolerance,
        finite_difference_step=step,
        max_line_search_steps=max_line_search_steps,
        max_restarts=max_restarts,
        gradient_method=gradient_method,
        verbosity=index(options.verbosity),
    )


def _pack_tensor(tensor, real):
    if real:
        return np.real(tensor).reshape(-1)
    return np.concatenate(
        (np.real(tensor).reshape(-1), np.imag(tensor).reshape(-1))
    )


def _unpack_tensor(parameters, shape, real):
    size = int(np.prod(shape))
    if real:
        tensor = np.asarray(parameters).reshape(shape)
    else:
        tensor = (
            np.asarray(parameters[:size])
            + 1j * np.asarray(parameters[size:])
        ).reshape(shape)
    norm = np.linalg.norm(tensor)
    if norm <= np.finfo(float).tiny:
        raise ValueError("the variational plane LETTA became numerically zero.")
    return tensor / norm


def _initial_plane_state(model, bond_dim, seed, real):
    if bond_dim == 1 and abs(model.field) >= 2.0 * abs(model.coupling):
        tensor = np.ones((1, 1, 1, 1, 2, 2, 2))
        if model.field < 0.0:
            tensor[..., 1, :, :] *= -1.0
        return UniformPlaneLETTA(tensor).normalized_parameters()
    return random_uniform_plane_letta(
        local_physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        real=real,
    )


def vuletta_plane(
    model,
    *,
    bond_dim=None,
    initial=None,
    seed=None,
    real=None,
    environment=None,
    options=None,
):
    """Optimize a one-site uniform LETTA directly for the infinite plane."""

    if not isinstance(model, PlaneTFIM):
        raise TypeError("model must be a PlaneTFIM.")
    options = _validated_options(options)
    if environment is None:
        environment = PlaneEnvironmentOptions()
    if not isinstance(environment, PlaneEnvironmentOptions):
        raise TypeError("environment must be a PlaneEnvironmentOptions instance.")
    environment = environment.validated()
    optimization_environment = PlaneEnvironmentOptions(
        window_sizes=(environment.window_sizes[-1],),
        boundary_bond_dim=environment.boundary_bond_dim,
        boundary_bond_dims=(environment.boundary_bond_dim,),
        cutoff=environment.cutoff,
        convergence_tolerance=environment.convergence_tolerance,
        boundary_convergence_tolerance=(
            environment.boundary_convergence_tolerance
        ),
    )

    if bond_dim is not None:
        try:
            bond_dim = index(bond_dim)
        except TypeError as error:
            raise ValueError("bond_dim must be an integer.") from error
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
    if initial is None:
        bond_dim = 1 if bond_dim is None else bond_dim
        if real is None:
            real = True
        state = _initial_plane_state(
            model,
            bond_dim,
            seed,
            bool(real),
        )
    else:
        if not isinstance(initial, UniformPlaneLETTA):
            initial = UniformPlaneLETTA(initial)
        if initial.local_physical_dim != 2:
            raise ValueError(
                "the square-lattice TFIM requires physical dimension two."
            )
        if bond_dim is not None and bond_dim != initial.bond_dim:
            raise ValueError("initial state and requested bond_dim do not agree.")
        state = initial.normalized_parameters()
        if real is None:
            real = not np.iscomplexobj(state.tensor)
    real = bool(real)
    if real and np.max(np.abs(np.imag(state.tensor))) > 1.0e-12:
        raise ValueError("a complex initial tensor cannot be used with real=True.")

    shape = state.tensor.shape
    parameters = _pack_tensor(state.tensor, real)
    history = []
    previous_energy = None
    invalid_objective = 2.0 * abs(model.coupling) + abs(model.field) + 1.0

    def objective(values):
        tensor = _unpack_tensor(values, shape, real)
        trial = UniformPlaneLETTA(tensor)
        try:
            return plane_energy_density(
                trial,
                model,
                environment=optimization_environment,
            )
        except UnreliablePlaneEnvironmentError:
            return invalid_objective

    cached_values = None
    cached_energy = None
    if options.gradient_method == "autodiff":
        if not real:
            raise ValueError(
                "automatic differentiation currently requires real=True."
            )
        from .plane_autodiff import (
            make_plane_energy_value_and_gradient,
        )

        autodiff_objective = make_plane_energy_value_and_gradient(
            shape,
            model,
            optimization_environment,
        )

        def objective_with_gradient(values):
            nonlocal cached_values, cached_energy
            try:
                value, gradient = autodiff_objective(values)
            except UnreliablePlaneEnvironmentError:
                value = invalid_objective
                gradient = np.zeros_like(values)
            cached_values = np.asarray(values).copy()
            cached_energy = float(value)
            return value, gradient
    else:
        objective_with_gradient = objective

    def callback(values):
        nonlocal previous_energy
        if (
            options.gradient_method == "autodiff"
            and cached_values is not None
            and np.array_equal(values, cached_values)
        ):
            energy = cached_energy
        else:
            energy = objective(values)
        change = (
            None
            if previous_energy is None
            else abs(energy - previous_energy)
        )
        history.append(
            VULETTA2DIteration(
                iteration=len(history) + 1,
                energy_density=energy,
                energy_change=change,
            )
        )
        previous_energy = energy
        if options.verbosity:
            change_text = "-" if change is None else f"{change:.3e}"
            print(
                f"VULETTA-2D {len(history):4d}  "
                f"energy={energy: .14f}  dE={change_text}"
            )

    total_iterations = 0
    total_function_evaluations = 0
    restart_count = 0
    current_parameters = parameters
    jacobian = (
        True if options.gradient_method == "autodiff" else "3-point"
    )
    while True:
        remaining_iterations = options.max_iterations - total_iterations
        minimize_options = {
            "maxiter": remaining_iterations,
            "ftol": options.function_tolerance,
            "gtol": options.gradient_tolerance,
            "finite_diff_rel_step": options.finite_difference_step,
            "maxls": options.max_line_search_steps * (restart_count + 1),
        }
        if options.max_function_evaluations is not None:
            remaining_evaluations = (
                options.max_function_evaluations
                - total_function_evaluations
            )
            minimize_options["maxfun"] = remaining_evaluations
        result = minimize(
            objective_with_gradient,
            current_parameters,
            method="L-BFGS-B",
            jac=jacobian,
            callback=callback,
            options=minimize_options,
        )
        run_iterations = int(result.nit)
        total_iterations += run_iterations
        total_function_evaluations += int(result.nfev)
        gradient_norm = float(np.linalg.norm(np.asarray(result.jac)))
        iteration_budget_exhausted = (
            total_iterations >= options.max_iterations
        )
        evaluation_budget_exhausted = (
            options.max_function_evaluations is not None
            and total_function_evaluations
            >= options.max_function_evaluations
        )
        stationary = gradient_norm <= options.gradient_tolerance
        if (
            stationary
            or iteration_budget_exhausted
            or evaluation_budget_exhausted
            or restart_count >= options.max_restarts
        ):
            break
        current_parameters = np.asarray(result.x).copy()
        restart_count += 1
    final_state = UniformPlaneLETTA(
        _unpack_tensor(result.x, shape, real)
    )
    observables = plane_observables(final_state, environment=environment)
    energy = float(
        -model.coupling
        * (observables.horizontal_zz + observables.vertical_zz)
        - model.field * observables.transverse_magnetization
    )
    gradient = np.asarray(result.jac)
    gradient_norm = float(np.linalg.norm(gradient))
    optimizer_converged = bool(
        result.success and gradient_norm <= options.gradient_tolerance
    )
    environment_converged = bool(observables.converged)
    converged = optimizer_converged and environment_converged
    messages = [str(result.message)]
    if restart_count:
        suffix = "" if restart_count == 1 else "s"
        messages.append(
            f"optimizer restarted {restart_count} time{suffix} after "
            "nonstationary termination"
        )
    if not optimizer_converged:
        messages.append(
            "optimizer stationarity failed: "
            f"|grad|={gradient_norm:.3e} > "
            f"{options.gradient_tolerance:.3e}"
        )
    if not environment_converged:
        messages.append(
            "environment convergence failed: "
            f"window change={observables.window_change:.3e}, "
            f"boundary change={observables.boundary_change:.3e}"
        )
    return VULETTA2DResult(
        state=final_state,
        energy_density=energy,
        observables=observables,
        converged=converged,
        optimizer_converged=optimizer_converged,
        environment_converged=environment_converged,
        iterations=total_iterations,
        function_evaluations=total_function_evaluations,
        gradient_norm=gradient_norm,
        history=tuple(history),
        message="; ".join(messages),
    )

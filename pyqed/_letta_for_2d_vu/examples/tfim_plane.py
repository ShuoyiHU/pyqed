"""Optimize a genuine two-dimensional uniform LETTA for the square TFIM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

from pyqed._letta_for_2d_vu import (
    PlaneEnvironmentOptions,
    VULETTA2DOptions,
    expand_uniform_plane_letta,
    tfim_square_lattice,
    vuletta_plane,
)


@dataclass(frozen=True)
class PlaneRun:
    method: str
    gradient_method: str
    bond_dimension: int
    energy_density: float
    transverse_magnetization: float
    horizontal_zz: float
    vertical_zz: float
    environment_window: int
    environment_boundary_bond: int
    environment_window_change: float
    environment_boundary_change: float
    environment_converged: bool
    solver_converged: bool
    overall_converged: bool
    iterations: int
    gradient_norm: float
    runtime_seconds: float


def run_tfim_plane(
    *,
    coupling=1.0,
    field=1.5,
    bond_dimensions=(1,),
    window_sizes=(3, 5, 7),
    boundary_bond_dim=24,
    boundary_cutoff=1.0e-10,
    environment_tolerance=1.0e-6,
    max_iterations=50,
    function_tolerance=1.0e-12,
    gradient_tolerance=1.0e-5,
    gradient_method="autodiff",
    seed=4,
):
    """Run bond-dimension continuation for the infinite square lattice."""

    bond_dimensions = tuple(int(value) for value in bond_dimensions)
    if not bond_dimensions or any(value <= 0 for value in bond_dimensions):
        raise ValueError("bond_dimensions must contain positive integers.")
    if any(
        right <= left
        for left, right in zip(bond_dimensions, bond_dimensions[1:])
    ):
        raise ValueError("bond_dimensions must be strictly increasing.")
    model = tfim_square_lattice(coupling=coupling, field=field)
    environment = PlaneEnvironmentOptions(
        window_sizes=tuple(window_sizes),
        boundary_bond_dim=boundary_bond_dim,
        cutoff=boundary_cutoff,
        convergence_tolerance=environment_tolerance,
    )
    solver_options = VULETTA2DOptions(
        max_iterations=max_iterations,
        function_tolerance=function_tolerance,
        gradient_tolerance=gradient_tolerance,
        gradient_method=gradient_method,
    )

    rows = []
    previous = None
    for stage, bond_dim in enumerate(bond_dimensions):
        initial = None
        if previous is not None:
            initial = expand_uniform_plane_letta(
                previous,
                bond_dim,
                seed=seed + stage,
            )
        start = perf_counter()
        result = vuletta_plane(
            model,
            bond_dim=bond_dim,
            initial=initial,
            seed=seed,
            real=True,
            environment=environment,
            options=solver_options,
        )
        runtime = perf_counter() - start
        previous = result.state
        observables = result.observables
        rows.append(
            PlaneRun(
                method="VULETTA-2D",
                gradient_method=gradient_method,
                bond_dimension=bond_dim,
                energy_density=result.energy_density,
                transverse_magnetization=observables.transverse_magnetization,
                horizontal_zz=observables.horizontal_zz,
                vertical_zz=observables.vertical_zz,
                environment_window=observables.window_size,
                environment_boundary_bond=(
                    observables.maximum_boundary_bond_dimension
                ),
                environment_window_change=observables.window_change,
                environment_boundary_change=observables.boundary_change,
                environment_converged=result.environment_converged,
                solver_converged=result.optimizer_converged,
                overall_converged=result.converged,
                iterations=result.iterations,
                gradient_norm=result.gradient_norm,
                runtime_seconds=runtime,
            )
        )
    return tuple(rows)


def _parse_positive_ints(text):
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive integers."
        )
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=3.04438)
    parser.add_argument(
        "--bonds",
        type=_parse_positive_ints,
        default=(1,),
        help="comma-separated LETTA virtual bond dimensions",
    )
    parser.add_argument(
        "--windows",
        type=_parse_positive_ints,
        default=(3, 5, 7, 9),
        help="comma-separated odd plane-window sizes",
    )
    parser.add_argument("--environment-bond", type=int, default=64)
    parser.add_argument("--environment-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--environment-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument(
        "--function-tolerance",
        type=float,
        default=1.0e-12,
        help="relative L-BFGS energy-change tolerance",
    )
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--gradient-method",
        choices=("autodiff", "finite_difference"),
        default="autodiff",
        help="energy-gradient backend (default: autodiff)",
    )
    parser.add_argument("--seed", type=int, default=4)
    arguments = parser.parse_args(argv)

    rows = run_tfim_plane(
        coupling=arguments.coupling,
        field=arguments.field,
        bond_dimensions=arguments.bonds,
        window_sizes=arguments.windows,
        boundary_bond_dim=arguments.environment_bond,
        boundary_cutoff=arguments.environment_cutoff,
        environment_tolerance=arguments.environment_tolerance,
        max_iterations=arguments.max_iterations,
        function_tolerance=arguments.function_tolerance,
        gradient_tolerance=arguments.gradient_tolerance,
        gradient_method=arguments.gradient_method,
        seed=arguments.seed,
    )
    print("Infinite square-lattice transverse-field Ising model")
    print(
        "method       gradient          D       energy/site       <X>          "
        "<ZZ-x>       <ZZ-y>       window  chi  dL          dchi        env-conv  "
        "opt-conv  all-conv  iterations  |grad|"
    )
    for row in rows:
        print(
            f"{row.method:12s} {row.gradient_method:16s} "
            f"{row.bond_dimension:3d}  "
            f"{row.energy_density: .10f}  "
            f"{row.transverse_magnetization: .10f}  "
            f"{row.horizontal_zz: .10f}  "
            f"{row.vertical_zz: .10f}  "
            f"{row.environment_window:6d}  "
            f"{row.environment_boundary_bond:3d}  "
            f"{row.environment_window_change:.3e}  "
            f"{row.environment_boundary_change:.3e}  "
            f"{str(row.environment_converged):>8s}  "
            f"{str(row.solver_converged):>8s}  "
            f"{str(row.overall_converged):>8s}  "
            f"{row.iterations:10d}  {row.gradient_norm:.3e}"
        )


if __name__ == "__main__":
    main()

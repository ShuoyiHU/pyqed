"""Compare one-site and two two-site LETTA methods on Ising lattices.

Run the default 2D and 3D cases with::

    python -m pyqed._letta_two_site_opt.benchmarks.ising_convergence

The benchmark deliberately keeps exact diagonalization, state construction,
and optional contraction warmup outside the measured solver interval.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy

from ..._letta_one_site_opt import (
    LETTADMROptions,
    LatticeLETTA,
    exact_ground_state,
    letta_dmrg,
)
from ..._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_hamiltonian as transverse_field_ising_hamiltonian_2d,
)
from ..._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo as transverse_field_ising_mpo_2d,
)
from ..._letta_one_site_opt._letta_for_3d import (
    snake_letta_state,
    transverse_field_ising_mpo as transverse_field_ising_mpo_3d,
    transverse_field_ising_sparse as transverse_field_ising_hamiltonian_3d,
)
from ..solver import LETTATwoSiteOptions, letta_two_site_dmrg


SOLVERS = (
    "one_site",
    "two_site_metric_als",
    "two_site_energy_refined",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One open-boundary Ising lattice included in a benchmark run."""

    dimension: str
    shape: tuple[int, ...]
    ordering: str = "continuous-snake"

    def __post_init__(self):
        dimension = self.dimension.lower()
        shape = tuple(int(length) for length in self.shape)
        if dimension not in {"2d", "3d"}:
            raise ValueError("dimension must be '2d' or '3d'.")
        expected_rank = 2 if dimension == "2d" else 3
        if len(shape) != expected_rank:
            raise ValueError(
                f"a {dimension} benchmark shape must have {expected_rank} axes."
            )
        if any(length <= 0 for length in shape):
            raise ValueError("all benchmark shape lengths must be positive.")
        if int(np.prod(shape)) < 2:
            raise ValueError("two-site benchmarking requires at least two sites.")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "shape", shape)

    @property
    def label(self):
        return f"{self.dimension}-{'x'.join(str(length) for length in self.shape)}"

    @property
    def nsites(self):
        return int(np.prod(self.shape))


def _build_problem(case, bond_dim, seed, coupling, field):
    if case.dimension == "2d":
        hamiltonian = transverse_field_ising_mpo_2d(
            case.shape,
            coupling=coupling,
            field=field,
        )
        state = LatticeLETTA.random(
            case.shape,
            physical_dim=2,
            bond_dim=bond_dim,
            seed=seed,
        )
    else:
        hamiltonian = transverse_field_ising_mpo_3d(
            case.shape,
            coupling=coupling,
            field=field,
            ordering=case.ordering,
        )
        state = snake_letta_state(
            case.shape,
            physical_dim=2,
            bond_dim=bond_dim,
            seed=seed,
            ordering=case.ordering,
        )
    return hamiltonian, state


def _exact_energy(case, coupling, field, exact_max_sites):
    if exact_max_sites == 0 or case.nsites > exact_max_sites:
        return None, None
    started = time.perf_counter()
    if case.dimension == "2d":
        hamiltonian = transverse_field_ising_hamiltonian_2d(
            case.shape,
            coupling=coupling,
            field=field,
        )
    else:
        hamiltonian = transverse_field_ising_hamiltonian_3d(
            case.shape,
            coupling=coupling,
            field=field,
            max_sites=exact_max_sites,
            ordering=case.ordering,
        )
    energy, _state = exact_ground_state(hamiltonian)
    elapsed = time.perf_counter() - started
    return float(energy), float(elapsed)


def _solver_options(
    solver,
    *,
    max_sweeps,
    tolerance,
    split_method,
    truncation_max_iterations,
    energy_refinement_max_iterations,
    energy_refinement_tolerance,
    energy_refinement_max_factor_norm_growth,
):
    if solver == "one_site":
        return LETTADMROptions(
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            matrix_free=True,
            use_sparse_mpo=True,
        )
    selected_split = (
        "energy-refined"
        if solver == "two_site_energy_refined"
        else split_method
    )
    return LETTATwoSiteOptions(
        max_sweeps=max_sweeps,
        tolerance=tolerance,
        matrix_free=True,
        use_sparse_mpo=True,
        split_method=selected_split,
        truncation_max_iterations=truncation_max_iterations,
        energy_refinement_max_iterations=energy_refinement_max_iterations,
        energy_refinement_tolerance=energy_refinement_tolerance,
        energy_refinement_max_factor_norm_growth=(
            energy_refinement_max_factor_norm_growth
        ),
        one_site_polish_sweeps=0,
    )


def _solve(
    solver,
    hamiltonian,
    initial_state,
    *,
    bond_dim,
    max_sweeps,
    tolerance,
    split_method,
    truncation_max_iterations,
    energy_refinement_max_iterations,
    energy_refinement_tolerance,
    energy_refinement_max_factor_norm_growth,
):
    options = _solver_options(
        solver,
        max_sweeps=max_sweeps,
        tolerance=tolerance,
        split_method=split_method,
        truncation_max_iterations=truncation_max_iterations,
        energy_refinement_max_iterations=energy_refinement_max_iterations,
        energy_refinement_tolerance=energy_refinement_tolerance,
        energy_refinement_max_factor_norm_growth=(
            energy_refinement_max_factor_norm_growth
        ),
    )
    if solver == "one_site":
        return letta_dmrg(
            hamiltonian,
            state=initial_state,
            bond_dim=bond_dim,
            options=options,
        )
    return letta_two_site_dmrg(
        hamiltonian,
        state=initial_state,
        bond_dim=bond_dim,
        options=options,
    )


def _warm_solvers(
    hamiltonian,
    initial_state,
    *,
    bond_dim,
    tolerance,
    split_method,
    truncation_max_iterations,
    energy_refinement_max_iterations,
    energy_refinement_tolerance,
    energy_refinement_max_factor_norm_growth,
):
    for solver in SOLVERS:
        _solve(
            solver,
            hamiltonian,
            initial_state,
            bond_dim=bond_dim,
            max_sweeps=1,
            tolerance=tolerance,
            split_method=split_method,
            truncation_max_iterations=truncation_max_iterations,
            energy_refinement_max_iterations=(
                energy_refinement_max_iterations
            ),
            energy_refinement_tolerance=energy_refinement_tolerance,
            energy_refinement_max_factor_norm_growth=(
                energy_refinement_max_factor_norm_growth
            ),
        )


def _run_record(result, solver, repeat, order_position, elapsed, exact_energy):
    local_updates = sum(len(sweep.updates) for sweep in result.history)
    accepted_updates = sum(
        int(update.accepted)
        for sweep in result.history
        for update in sweep.updates
    )
    energy_error = (
        None
        if exact_energy is None
        else abs(float(result.energy) - float(exact_energy))
    )
    return {
        "solver": solver,
        "repeat": int(repeat),
        "order_position": int(order_position),
        "elapsed_seconds": float(elapsed),
        "converged": bool(result.converged),
        "sweeps": int(result.sweeps),
        "sweeps_to_convergence": (
            int(result.sweeps) if result.converged else None
        ),
        "local_updates": int(local_updates),
        "accepted_updates": int(accepted_updates),
        "final_energy": float(result.energy),
        "absolute_error_to_exact": energy_error,
        "final_energy_density_change": float(
            result.history[-1].energy_density_change
        ),
        "sweep_energies": [float(sweep.energy) for sweep in result.history],
        "sweep_energy_density_changes": [
            float(sweep.energy_density_change) for sweep in result.history
        ],
        "message": result.message,
    }


def _median(records, key):
    return float(statistics.median(record[key] for record in records))


def _summarize(records):
    elapsed = [record["elapsed_seconds"] for record in records]
    errors = [
        record["absolute_error_to_exact"]
        for record in records
        if record["absolute_error_to_exact"] is not None
    ]
    energies = [record["final_energy"] for record in records]
    convergence_sweeps = [
        record["sweeps_to_convergence"]
        for record in records
        if record["sweeps_to_convergence"] is not None
    ]
    return {
        "run_count": len(records),
        "converged_runs": sum(int(record["converged"]) for record in records),
        "converged_all": all(record["converged"] for record in records),
        "sweeps_median": _median(records, "sweeps"),
        "sweeps_min": min(record["sweeps"] for record in records),
        "sweeps_max": max(record["sweeps"] for record in records),
        "sweeps_to_convergence_median": (
            None
            if not convergence_sweeps
            else float(statistics.median(convergence_sweeps))
        ),
        "local_updates_median": _median(records, "local_updates"),
        "accepted_updates_median": _median(records, "accepted_updates"),
        "elapsed_seconds_median": float(statistics.median(elapsed)),
        "elapsed_seconds_mean": float(statistics.mean(elapsed)),
        "elapsed_seconds_min": float(min(elapsed)),
        "elapsed_seconds_max": float(max(elapsed)),
        "elapsed_seconds_std": float(statistics.pstdev(elapsed)),
        "final_energy_median": float(statistics.median(energies)),
        "final_energy_spread": float(max(energies) - min(energies)),
        "absolute_error_to_exact_median": (
            None if not errors else float(statistics.median(errors))
        ),
        "final_energy_density_change_median": _median(
            records, "final_energy_density_change"
        ),
    }


def _benchmark_case(
    case,
    *,
    bond_dim,
    max_sweeps,
    tolerance,
    repeats,
    seed,
    coupling,
    field,
    exact_max_sites,
    warmup,
    split_method,
    truncation_max_iterations,
    energy_refinement_max_iterations,
    energy_refinement_tolerance,
    energy_refinement_max_factor_norm_growth,
):
    hamiltonian, initial_state = _build_problem(
        case,
        bond_dim,
        seed,
        coupling,
        field,
    )
    initial_energy = float(initial_state.expectation(hamiltonian))
    exact_energy, exact_elapsed = _exact_energy(
        case,
        coupling,
        field,
        exact_max_sites,
    )
    if warmup:
        _warm_solvers(
            hamiltonian,
            initial_state,
            bond_dim=bond_dim,
            tolerance=tolerance,
            split_method=split_method,
            truncation_max_iterations=truncation_max_iterations,
            energy_refinement_max_iterations=(
                energy_refinement_max_iterations
            ),
            energy_refinement_tolerance=energy_refinement_tolerance,
            energy_refinement_max_factor_norm_growth=(
                energy_refinement_max_factor_norm_growth
            ),
        )

    runs = []
    for repeat_index in range(repeats):
        offset = repeat_index % len(SOLVERS)
        order = SOLVERS[offset:] + SOLVERS[:offset]
        for order_position, solver in enumerate(order, start=1):
            started = time.perf_counter()
            result = _solve(
                solver,
                hamiltonian,
                initial_state,
                bond_dim=bond_dim,
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                split_method=split_method,
                truncation_max_iterations=truncation_max_iterations,
                energy_refinement_max_iterations=(
                    energy_refinement_max_iterations
                ),
                energy_refinement_tolerance=energy_refinement_tolerance,
                energy_refinement_max_factor_norm_growth=(
                    energy_refinement_max_factor_norm_growth
                ),
            )
            elapsed = time.perf_counter() - started
            runs.append(
                _run_record(
                    result,
                    solver,
                    repeat_index + 1,
                    order_position,
                    elapsed,
                    exact_energy,
                )
            )

    summaries = {
        solver: _summarize(
            [record for record in runs if record["solver"] == solver]
        )
        for solver in SOLVERS
    }
    return {
        "label": case.label,
        "dimension": case.dimension,
        "shape": list(case.shape),
        "ordering": case.ordering if case.dimension == "3d" else "compact",
        "nsites": case.nsites,
        "initial_state_id": "shared-random-state",
        "initial_energy": initial_energy,
        "exact_energy": exact_energy,
        "exact_elapsed_seconds": exact_elapsed,
        "runs": runs,
        "summaries": summaries,
    }


def _environment_record():
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": thread_variables,
    }


def run_ising_benchmarks(
    *,
    cases: Sequence[BenchmarkCase] = (
        BenchmarkCase("2d", (2, 3)),
        BenchmarkCase("3d", (2, 2, 2)),
    ),
    bond_dim=2,
    max_sweeps=8,
    tolerance=1.0e-8,
    repeats=1,
    seed=11,
    coupling=1.0,
    field=1.0,
    exact_max_sites=12,
    warmup=True,
    split_method="metric-als",
    truncation_max_iterations=8,
    energy_refinement_max_iterations=8,
    energy_refinement_tolerance=1.0e-10,
    energy_refinement_max_factor_norm_growth=100.0,
):
    """Run a fair three-method comparison and return JSON-ready data.

    Each solver receives a copy of the same random state. Timed execution order
    rotates across repeats, and exact-reference work is not timed as part of a
    solver.
    """

    cases = tuple(cases)
    if not cases or not all(isinstance(case, BenchmarkCase) for case in cases):
        raise ValueError("cases must be a nonempty sequence of BenchmarkCase values.")
    bond_dim = int(bond_dim)
    max_sweeps = int(max_sweeps)
    repeats = int(repeats)
    exact_max_sites = int(exact_max_sites)
    truncation_max_iterations = int(truncation_max_iterations)
    energy_refinement_max_iterations = int(
        energy_refinement_max_iterations
    )
    if bond_dim <= 0 or max_sweeps <= 0 or repeats <= 0:
        raise ValueError("bond_dim, max_sweeps, and repeats must be positive.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if exact_max_sites < 0:
        raise ValueError("exact_max_sites must be nonnegative.")
    if split_method not in {"metric-als", "conditional-svd"}:
        raise ValueError(
            "split_method must be 'metric-als' or 'conditional-svd'."
        )
    if truncation_max_iterations <= 0:
        raise ValueError("truncation_max_iterations must be positive.")
    if energy_refinement_max_iterations <= 0:
        raise ValueError("energy_refinement_max_iterations must be positive.")
    if energy_refinement_tolerance <= 0.0:
        raise ValueError("energy_refinement_tolerance must be positive.")
    if energy_refinement_max_factor_norm_growth < 1.0:
        raise ValueError(
            "energy_refinement_max_factor_norm_growth must be at least one."
        )

    configuration = {
        "bond_dim": bond_dim,
        "max_sweeps": max_sweeps,
        "tolerance": float(tolerance),
        "repeats": repeats,
        "seed": int(seed),
        "coupling": float(coupling),
        "field": float(field),
        "exact_max_sites": exact_max_sites,
        "warmup": bool(warmup),
        "two_site_split_method": split_method,
        "truncation_max_iterations": truncation_max_iterations,
        "energy_refinement_max_iterations": energy_refinement_max_iterations,
        "energy_refinement_tolerance": float(energy_refinement_tolerance),
        "energy_refinement_max_factor_norm_growth": float(
            energy_refinement_max_factor_norm_growth
        ),
        "one_site_polish_sweeps": 0,
    }
    case_results = [
        _benchmark_case(
            case,
            bond_dim=bond_dim,
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            repeats=repeats,
            seed=seed,
            coupling=coupling,
            field=field,
            exact_max_sites=exact_max_sites,
            warmup=warmup,
            split_method=split_method,
            truncation_max_iterations=truncation_max_iterations,
            energy_refinement_max_iterations=(
                energy_refinement_max_iterations
            ),
            energy_refinement_tolerance=energy_refinement_tolerance,
            energy_refinement_max_factor_norm_growth=(
                energy_refinement_max_factor_norm_growth
            ),
        )
        for case in cases
    ]
    return {
        "schema_version": 2,
        "configuration": configuration,
        "environment": _environment_record(),
        "cases": case_results,
    }


def _table(rows):
    widths = [
        max(len(str(row[column])) for row in rows)
        for column in range(len(rows[0]))
    ]
    return "\n".join(
        "  ".join(
            str(value).ljust(widths[column])
            for column, value in enumerate(row)
        ).rstrip()
        for row in rows
    )


def format_benchmark_report(report):
    """Format the summary portion of a benchmark report as plain text."""

    rows = [
        (
            "case",
            "solver",
            "conv",
            "sweeps_run",
            "sweeps_conv",
            "updates",
            "accepted",
            "median_s",
            "energy",
            "abs_exact_err",
        )
    ]
    for case in report["cases"]:
        for solver in SOLVERS:
            summary = case["summaries"][solver]
            error = summary["absolute_error_to_exact_median"]
            convergence_sweeps = summary["sweeps_to_convergence_median"]
            rows.append(
                (
                    case["label"],
                    solver,
                    f"{summary['converged_runs']}/{summary['run_count']}",
                    f"{summary['sweeps_median']:.1f}",
                    (
                        "-"
                        if convergence_sweeps is None
                        else f"{convergence_sweeps:.1f}"
                    ),
                    f"{summary['local_updates_median']:.1f}",
                    f"{summary['accepted_updates_median']:.1f}",
                    f"{summary['elapsed_seconds_median']:.6f}",
                    f"{summary['final_energy_median']:.12f}",
                    "-" if error is None else f"{error:.3e}",
                )
            )
    configuration = report["configuration"]
    lines = [
        "LETTA Ising convergence benchmark",
        (
            "settings: "
            f"D={configuration['bond_dim']}, "
            f"max_sweeps={configuration['max_sweeps']}, "
            f"tolerance={configuration['tolerance']:.3e}, "
            f"repeats={configuration['repeats']}, "
            f"old_two_site_split={configuration['two_site_split_method']}, "
            "new_two_site_split=energy-refined"
        ),
        (
            "timing: solver only; identical initial state per pair; "
            "execution order rotates across repeats"
        ),
        "reported sweeps, updates, accepted updates, time, and energy are medians",
        "",
        _table(rows),
    ]
    return "\n".join(lines)


def _parse_shape(value, rank, option_name):
    normalized = value.lower().replace(",", "x")
    try:
        shape = tuple(int(piece) for piece in normalized.split("x"))
    except ValueError as error:
        raise ValueError(f"{option_name} must contain integer lengths.") from error
    if len(shape) != rank or any(length <= 0 for length in shape):
        raise ValueError(
            f"{option_name} must contain {rank} positive lengths separated by x."
        )
    return shape


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-site, metric-ALS two-site, and energy-refined "
            "two-site LETTA sweeps on open-boundary 2D and 3D "
            "transverse-field Ising models."
        )
    )
    parser.add_argument("--dimension", choices=("all", "2d", "3d"), default="all")
    parser.add_argument("--shape-2d", default="2x3")
    parser.add_argument("--shape-3d", default="2x2x2")
    parser.add_argument("--ordering", default="continuous-snake")
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument(
        "--exact-max-sites",
        type=int,
        default=12,
        help="largest lattice for exact comparison; use 0 to disable",
    )
    parser.add_argument(
        "--two-site-split",
        choices=("metric-als", "conditional-svd"),
        default="metric-als",
    )
    parser.add_argument("--truncation-max-iterations", type=int, default=8)
    parser.add_argument("--energy-refinement-max-iterations", type=int, default=8)
    parser.add_argument("--energy-refinement-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--energy-refinement-max-factor-norm-growth",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="include first-use contraction planning overhead in timings",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for raw runs, sweep histories, and summaries",
    )
    return parser


def main(argv=None):
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        shape_2d = _parse_shape(arguments.shape_2d, 2, "--shape-2d")
        shape_3d = _parse_shape(arguments.shape_3d, 3, "--shape-3d")
        cases = []
        if arguments.dimension in {"all", "2d"}:
            cases.append(BenchmarkCase("2d", shape_2d))
        if arguments.dimension in {"all", "3d"}:
            cases.append(BenchmarkCase("3d", shape_3d, arguments.ordering))
        report = run_ising_benchmarks(
            cases=cases,
            bond_dim=arguments.bond_dim,
            max_sweeps=arguments.max_sweeps,
            tolerance=arguments.tolerance,
            repeats=arguments.repeats,
            seed=arguments.seed,
            coupling=arguments.coupling,
            field=arguments.field,
            exact_max_sites=arguments.exact_max_sites,
            warmup=not arguments.no_warmup,
            split_method=arguments.two_site_split,
            truncation_max_iterations=arguments.truncation_max_iterations,
            energy_refinement_max_iterations=(
                arguments.energy_refinement_max_iterations
            ),
            energy_refinement_tolerance=arguments.energy_refinement_tolerance,
            energy_refinement_max_factor_norm_growth=(
                arguments.energy_refinement_max_factor_norm_growth
            ),
        )
    except ValueError as error:
        parser.error(str(error))

    print(format_benchmark_report(report))
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(f"\nraw JSON written to {arguments.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

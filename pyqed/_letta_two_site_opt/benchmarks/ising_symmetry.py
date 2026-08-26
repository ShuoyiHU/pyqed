"""Reproducible dense-versus-Z2 LETTA optimization benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from ..._letta_one_site_opt import (
    AbelianSymmetry,
    LETTADMROptions,
    LatticeLETTA,
    letta_dmrg,
)
from ..._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo as transverse_field_ising_mpo_2d,
)
from ..._letta_one_site_opt._letta_for_3d import (
    snake_coordinates,
    transverse_field_ising_mpo as transverse_field_ising_mpo_3d,
)
from ..solver import LETTATwoSiteOptions, letta_two_site_dmrg


@dataclass(frozen=True)
class SymmetryBenchmarkCase:
    dimension: str
    shape: tuple[int, ...]

    def __post_init__(self):
        dimension = str(self.dimension).lower()
        shape = tuple(int(value) for value in self.shape)
        expected_rank = {"2d": 2, "3d": 3}.get(dimension)
        if expected_rank is None:
            raise ValueError("dimension must be '2d' or '3d'.")
        if len(shape) != expected_rank or any(value <= 0 for value in shape):
            raise ValueError("benchmark shape has the wrong rank or extent.")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "shape", shape)

    @property
    def nsites(self):
        return int(np.prod(self.shape))


def _z2_even():
    return AbelianSymmetry(
        physical_charges=(0, 1),
        sector=0,
        moduli=2,
        name="ising-z2",
    )


def _problem(case, bond_dim, seed, coupling, field):
    symmetry = _z2_even()
    if case.dimension == "2d":
        coordinates = None
        hamiltonian = transverse_field_ising_mpo_2d(
            case.shape, coupling=coupling, field=field, basis="x"
        )
    else:
        coordinates = snake_coordinates(case.shape)
        hamiltonian = transverse_field_ising_mpo_3d(
            case.shape, coupling=coupling, field=field, basis="x"
        )
    symmetric = LatticeLETTA.random(
        case.shape,
        physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        coordinates=coordinates,
        symmetry=symmetry,
    )
    return hamiltonian, symmetric, symmetric.without_symmetry(), symmetry


def _solve(solver, hamiltonian, state, bond_dim, max_sweeps, tolerance):
    if solver == "one_site":
        return letta_dmrg(
            hamiltonian,
            state=state,
            options=LETTADMROptions(
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                matrix_free=True,
                gauge_mode="qr",
            ),
        )
    if solver == "two_site":
        return letta_two_site_dmrg(
            hamiltonian,
            state=state,
            bond_dim=bond_dim,
            options=LETTATwoSiteOptions(
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                matrix_free=True,
                split_method="metric-als",
                gauge_mode="qr",
            ),
        )
    raise ValueError("unknown solver.")


def _record(result, elapsed, repeat):
    updates = [
        update for sweep in result.history for update in sweep.updates
    ]
    dimensions = [int(update.local_dimension) for update in updates]
    full_dimensions = [
        int(
            update.full_local_dimension
            if update.full_local_dimension is not None
            else update.local_dimension
        )
        for update in updates
    ]
    max_dimension = max(dimensions, default=0)
    dtype_bytes = int(np.dtype(np.result_type(*result.state.tensors)).itemsize)
    return {
        "repeat": int(repeat),
        "energy": float(result.energy),
        "converged": bool(result.converged),
        "sweeps": int(result.sweeps),
        "local_updates": len(updates),
        "elapsed_seconds": float(elapsed),
        "parameter_count": int(result.state.parameter_count),
        "dense_parameter_count": int(result.state.dense_parameter_count),
        "tensor_storage_bytes": int(
            sum(tensor.nbytes for tensor in result.state.tensors)
        ),
        "max_local_dimension": max_dimension,
        "max_full_local_dimension": max(full_dimensions, default=0),
        "local_dimension_sum": int(sum(dimensions)),
        "dense_matrix_bytes": int(2 * max_dimension**2 * dtype_bytes),
        "cubic_work_proxy": int(sum(dimension**3 for dimension in dimensions)),
        "symmetry_violation": float(result.state.symmetry_violation()),
    }


def _summary(records):
    return {
        "energy_median": float(median(record["energy"] for record in records)),
        "energy_min": float(min(record["energy"] for record in records)),
        "converged_fraction": float(
            sum(record["converged"] for record in records) / len(records)
        ),
        "sweeps_median": float(median(record["sweeps"] for record in records)),
        "elapsed_seconds_median": float(
            median(record["elapsed_seconds"] for record in records)
        ),
        "parameter_count": int(records[0]["parameter_count"]),
        "dense_parameter_count": int(records[0]["dense_parameter_count"]),
        "tensor_storage_bytes": int(records[0]["tensor_storage_bytes"]),
        "max_local_dimension": int(
            max(record["max_local_dimension"] for record in records)
        ),
        "max_full_local_dimension": int(
            max(record["max_full_local_dimension"] for record in records)
        ),
        "local_dimension_sum_median": float(
            median(record["local_dimension_sum"] for record in records)
        ),
        "dense_matrix_bytes": int(
            max(record["dense_matrix_bytes"] for record in records)
        ),
        "cubic_work_proxy_median": float(
            median(record["cubic_work_proxy"] for record in records)
        ),
        "symmetry_violation_max": float(
            max(record["symmetry_violation"] for record in records)
        ),
        "runs": records,
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _comparison(dense_records, symmetry_records):
    dense = _summary(dense_records)
    symmetry = _summary(symmetry_records)
    return {
        "without_symmetry": dense,
        "with_symmetry": symmetry,
        "energy_difference": float(
            abs(symmetry["energy_median"] - dense["energy_median"])
        ),
        "ratios": {
            "elapsed_seconds": _ratio(
                symmetry["elapsed_seconds_median"],
                dense["elapsed_seconds_median"],
            ),
            "parameter_count": _ratio(
                symmetry["parameter_count"], dense["parameter_count"]
            ),
            "max_local_dimension": _ratio(
                symmetry["max_local_dimension"], dense["max_local_dimension"]
            ),
            "dense_matrix_bytes": _ratio(
                symmetry["dense_matrix_bytes"], dense["dense_matrix_bytes"]
            ),
            "cubic_work_proxy": _ratio(
                symmetry["cubic_work_proxy_median"],
                dense["cubic_work_proxy_median"],
            ),
        },
    }


def run_symmetry_benchmarks(
    *,
    cases=(
        SymmetryBenchmarkCase("2d", (2, 3)),
        SymmetryBenchmarkCase("3d", (2, 2, 2)),
    ),
    bond_dim=4,
    max_sweeps=8,
    tolerance=1.0e-9,
    repeats=3,
    seed=7,
    coupling=1.0,
    field=1.5,
):
    """Compare identical initial states with and without exact Z2 sectors."""

    cases = tuple(cases)
    bond_dim = int(bond_dim)
    max_sweeps = int(max_sweeps)
    repeats = int(repeats)
    if bond_dim <= 0 or max_sweeps <= 0 or repeats <= 0:
        raise ValueError("bond_dim, max_sweeps, and repeats must be positive.")
    case_reports = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, SymmetryBenchmarkCase):
            case = SymmetryBenchmarkCase(*case)
        hamiltonian, symmetric, dense, symmetry = _problem(
            case,
            bond_dim,
            int(seed) + case_index,
            float(coupling),
            float(field),
        )
        solver_reports = {}
        for solver_index, solver in enumerate(("one_site", "two_site")):
            records = {"without_symmetry": [], "with_symmetry": []}
            for repeat in range(repeats):
                labels = ["without_symmetry", "with_symmetry"]
                if (repeat + solver_index) % 2:
                    labels.reverse()
                for label in labels:
                    initial = dense if label == "without_symmetry" else symmetric
                    start = perf_counter()
                    result = _solve(
                        solver,
                        hamiltonian,
                        initial,
                        bond_dim,
                        max_sweeps,
                        float(tolerance),
                    )
                    elapsed = perf_counter() - start
                    records[label].append(_record(result, elapsed, repeat))
            solver_reports[solver] = _comparison(
                records["without_symmetry"], records["with_symmetry"]
            )
        exact_energy = None
        if case.nsites <= 12:
            exact_energy = float(np.linalg.eigvalsh(hamiltonian.to_dense())[0])
        case_reports.append(
            {
                "dimension": case.dimension,
                "shape": list(case.shape),
                "nsites": case.nsites,
                "bond_dim": bond_dim,
                "exact_energy": exact_energy,
                "symmetry": {
                    "name": symmetry.name,
                    "physical_charges": list(symmetry.physical_charges),
                    "sector": symmetry.sector,
                    "moduli": symmetry.moduli,
                },
                "solvers": solver_reports,
            }
        )
    return {
        "schema_version": 1,
        "settings": {
            "bond_dim": bond_dim,
            "max_sweeps": max_sweeps,
            "tolerance": float(tolerance),
            "repeats": repeats,
            "seed": int(seed),
            "coupling": float(coupling),
            "field": float(field),
            "one_site_matrix_free": True,
            "two_site_matrix_free": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "cases": case_reports,
    }


def format_report(report):
    lines = [
        "case solver variant energy time_s parameters max_dim dimension_ratio"
    ]
    for case in report["cases"]:
        label = f"{case['dimension']}:{'x'.join(map(str, case['shape']))}"
        for solver, comparison in case["solvers"].items():
            dense_dimension = comparison["without_symmetry"]["max_local_dimension"]
            for variant in ("without_symmetry", "with_symmetry"):
                row = comparison[variant]
                ratio = _ratio(row["max_local_dimension"], dense_dimension)
                lines.append(
                    f"{label} {solver} {variant} "
                    f"{row['energy_median']:.12g} "
                    f"{row['elapsed_seconds_median']:.6g} "
                    f"{row['parameter_count']} {row['max_local_dimension']} "
                    f"{ratio:.6g}"
                )
    return "\n".join(lines)


def _parse_shape(value, rank, option):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must contain integers.") from error
    if len(shape) != rank or any(length <= 0 for length in shape):
        raise argparse.ArgumentTypeError(f"{option} must have {rank} positive extents.")
    return shape


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=("2d", "3d", "both"), default="both")
    parser.add_argument("--shape-2d", default="2x3")
    parser.add_argument("--shape-3d", default="2x2x2")
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.5)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    cases = []
    if arguments.dimension in {"2d", "both"}:
        cases.append(
            SymmetryBenchmarkCase(
                "2d", _parse_shape(arguments.shape_2d, 2, "--shape-2d")
            )
        )
    if arguments.dimension in {"3d", "both"}:
        cases.append(
            SymmetryBenchmarkCase(
                "3d", _parse_shape(arguments.shape_3d, 3, "--shape-3d")
            )
        )
    report = run_symmetry_benchmarks(
        cases=cases,
        bond_dim=arguments.bond_dim,
        max_sweeps=arguments.max_sweeps,
        tolerance=arguments.tolerance,
        repeats=arguments.repeats,
        seed=arguments.seed,
        coupling=arguments.coupling,
        field=arguments.field,
    )
    print(format_report(report))
    if arguments.json_output is not None:
        arguments.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {arguments.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

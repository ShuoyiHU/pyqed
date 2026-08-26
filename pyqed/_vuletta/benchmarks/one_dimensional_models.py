"""Benchmark conditional-canonical VULETTA and VUMPS on standard 1D models.

Run from the repository root with

```
PYTHONPATH=. python -m pyqed._vuletta.benchmarks.one_dimensional_models
```
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from operator import index
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from pyqed._vuletta.operators import (
    one_site_expectation as letta_one_site_expectation,
    two_site_expectation as letta_two_site_expectation,
)
from pyqed._vuletta.solver import VULETTAOptions, vuletta
from pyqed._vuletta.state import expand_uniform_letta
from pyqed._vumps import (
    VUMPSOptions,
    nearest_neighbor_energy as mps_two_site_expectation,
    one_site_expectation as mps_one_site_expectation,
    vumps,
)
from pyqed._vumps.examples.tfim_comparison import (
    exact_tfim_energy_density,
    exact_tfim_transverse_magnetization,
    exact_tfim_zz_correlation,
    tfim_bond_hamiltonian,
)


@dataclass(frozen=True)
class ObservableDefinition:
    name: str
    operator: np.ndarray
    sites: int
    reference: float


@dataclass(frozen=True)
class OneDimensionalModel:
    name: str
    hamiltonian: np.ndarray
    reference_energy: float
    observables: tuple[ObservableDefinition, ...]

    @property
    def reference_observables(self):
        return tuple((observable.name, observable.reference) for observable in self.observables)


@dataclass(frozen=True)
class ObservableBenchmark:
    name: str
    value: float
    reference: float
    error: float


@dataclass(frozen=True)
class BenchmarkRow:
    model: str
    method: str
    bond_dim: int
    transfer_bond_dim: int
    tensor_entries: int
    tangent_dimension: int | None
    energy: float
    reference_energy: float
    energy_error: float
    observables: tuple[ObservableBenchmark, ...]
    converged: bool
    iterations: int
    residual: float
    runtime_seconds: float
    repeats: int
    message: str


def common_one_dimensional_models():
    """Return deterministic TFIM and antiferromagnetic Heisenberg models."""

    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    tfim = OneDimensionalModel(
        name="tfim",
        hamiltonian=tfim_bond_hamiltonian(coupling=1.0, field=1.5),
        reference_energy=exact_tfim_energy_density(coupling=1.0, field=1.5),
        observables=(
            ObservableDefinition(
                name="<X>",
                operator=x,
                sites=1,
                reference=exact_tfim_transverse_magnetization(
                    coupling=1.0,
                    field=1.5,
                ),
            ),
            ObservableDefinition(
                name="<Z Z>",
                operator=np.kron(z, z),
                sites=2,
                reference=exact_tfim_zz_correlation(coupling=1.0, field=1.5),
            ),
        ),
    )

    sx = 0.5 * x
    sy = 0.5 * y
    sz = 0.5 * z
    heisenberg_energy = 0.25 - np.log(2.0)
    heisenberg = OneDimensionalModel(
        name="heisenberg",
        hamiltonian=(
            np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
        ),
        reference_energy=heisenberg_energy,
        observables=(
            ObservableDefinition(
                name="<Sz>",
                operator=sz,
                sites=1,
                reference=0.0,
            ),
            ObservableDefinition(
                name="<Sz Sz>",
                operator=np.kron(sz, sz),
                sites=2,
                reference=heisenberg_energy / 3.0,
            ),
        ),
    )
    return {"tfim": tfim, "heisenberg": heisenberg}


def _positive_integer(value, name):
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _timed_solve(factory, repeats):
    repeats = _positive_integer(repeats, "repeats")
    if repeats > 1:
        factory()
    results = []
    timings = []
    for _repeat in range(repeats):
        start = perf_counter()
        results.append(factory())
        timings.append(perf_counter() - start)
    return results[0], float(median(timings))


def _observable_rows(model, state, method):
    values = []
    for observable in model.observables:
        if method == "VULETTA":
            evaluator = (
                letta_one_site_expectation
                if observable.sites == 1
                else letta_two_site_expectation
            )
        else:
            evaluator = (
                mps_one_site_expectation
                if observable.sites == 1
                else mps_two_site_expectation
            )
        value = float(np.real(evaluator(state, observable.operator)))
        values.append(
            ObservableBenchmark(
                name=observable.name,
                value=value,
                reference=float(observable.reference),
                error=abs(value - float(observable.reference)),
            )
        )
    return tuple(values)


def run_one_dimensional_benchmark(
    *,
    model_names=("tfim", "heisenberg"),
    letta_bond_dimensions=(1, 2, 3),
    mps_bond_dimensions=(1, 2, 3, 4, 6),
    seed=3,
    tolerance=1.0e-7,
    max_iterations=300,
    repeats=1,
    growth_noise=3.0e-2,
):
    """Run deterministic observable and wall-time comparisons."""

    models = common_one_dimensional_models()
    unknown = tuple(name for name in model_names if name not in models)
    if unknown:
        raise ValueError(f"unknown model names: {unknown}.")
    letta_dimensions = tuple(
        _positive_integer(value, "LETTA bond dimension")
        for value in letta_bond_dimensions
    )
    mps_dimensions = tuple(
        _positive_integer(value, "MPS bond dimension")
        for value in mps_bond_dimensions
    )
    if tuple(sorted(set(letta_dimensions))) != letta_dimensions:
        raise ValueError("LETTA bond dimensions must be unique and increasing.")
    if tuple(sorted(set(mps_dimensions))) != mps_dimensions:
        raise ValueError("MPS bond dimensions must be unique and increasing.")
    repeats = _positive_integer(repeats, "repeats")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    growth_noise = float(growth_noise)
    if not np.isfinite(growth_noise) or growth_noise <= 0.0:
        raise ValueError("growth_noise must be finite and positive.")

    rows = []
    for model_name in model_names:
        model = models[model_name]
        previous_letta_result = None
        for bond_dim in letta_dimensions:
            initial = None
            if previous_letta_result is not None:
                initial = expand_uniform_letta(
                    previous_letta_result.state,
                    bond_dim,
                    seed=seed + 1009 * bond_dim,
                    relative_noise=growth_noise,
                )

            def solve_letta(initial=initial, bond_dim=bond_dim):
                return vuletta(
                    model.hamiltonian,
                    bond_dim=bond_dim,
                    initial=initial,
                    seed=seed,
                    real=True,
                    options=VULETTAOptions(
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                        stationarity_tolerance=tolerance,
                    ),
                )

            result, runtime = _timed_solve(solve_letta, repeats)
            previous_letta_result = result
            rows.append(
                BenchmarkRow(
                    model=model.name,
                    method="VULETTA",
                    bond_dim=bond_dim,
                    transfer_bond_dim=2 * bond_dim,
                    tensor_entries=4 * bond_dim * bond_dim,
                    tangent_dimension=result.reduced_dimension,
                    energy=float(result.energy),
                    reference_energy=float(model.reference_energy),
                    energy_error=abs(float(result.energy) - model.reference_energy),
                    observables=_observable_rows(model, result.state, "VULETTA"),
                    converged=bool(result.converged),
                    iterations=int(result.iterations),
                    residual=float(result.residual_norm),
                    runtime_seconds=runtime,
                    repeats=repeats,
                    message=str(result.message),
                )
            )

        for bond_dim in mps_dimensions:

            def solve_mps(bond_dim=bond_dim):
                return vumps(
                    model.hamiltonian,
                    bond_dim=bond_dim,
                    seed=seed,
                    real=True,
                    options=VUMPSOptions(
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    ),
                )

            result, runtime = _timed_solve(solve_mps, repeats)
            rows.append(
                BenchmarkRow(
                    model=model.name,
                    method="VUMPS",
                    bond_dim=bond_dim,
                    transfer_bond_dim=bond_dim,
                    tensor_entries=2 * bond_dim * bond_dim,
                    tangent_dimension=None,
                    energy=float(result.energy),
                    reference_energy=float(model.reference_energy),
                    energy_error=abs(float(result.energy) - model.reference_energy),
                    observables=_observable_rows(model, result.state, "VUMPS"),
                    converged=bool(result.converged),
                    iterations=int(result.iterations),
                    residual=float(result.residual_norm),
                    runtime_seconds=runtime,
                    repeats=repeats,
                    message=str(result.message),
                )
            )
    return tuple(rows)


def format_benchmark_markdown(rows):
    """Return a compact Markdown table for benchmark rows."""

    lines = [
        "| model | method | D | transfer | entries | tangent dim | converged | "
        "iterations | energy | abs(dE) | observables | residual | runtime_s |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        observables = "; ".join(
            f"{item.name}={item.value:.10g} (err {item.error:.2e})"
            for item in row.observables
        )
        lines.append(
            f"| {row.model} | {row.method} | {row.bond_dim} | "
            f"{row.transfer_bond_dim} | {row.tensor_entries} | "
            f"{row.tangent_dimension if row.tangent_dimension is not None else '-'} | "
            f"{row.converged} | {row.iterations} | {row.energy:.12g} | "
            f"{row.energy_error:.3e} | {observables} | "
            f"{row.residual:.3e} | {row.runtime_seconds:.6f} |"
        )
    return "\n".join(lines) + "\n"


def write_benchmark_csv(rows, path):
    """Write benchmark rows to CSV, encoding observables as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "method",
        "bond_dim",
        "transfer_bond_dim",
        "tensor_entries",
        "tangent_dimension",
        "energy",
        "reference_energy",
        "energy_error",
        "converged",
        "iterations",
        "residual",
        "runtime_seconds",
        "repeats",
        "observables",
        "message",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {field: getattr(row, field) for field in fields if field != "observables"}
            record["observables"] = json.dumps(
                [
                    {
                        "name": item.name,
                        "value": item.value,
                        "reference": item.reference,
                        "error": item.error,
                    }
                    for item in row.observables
                ],
                sort_keys=True,
            )
            writer.writerow(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=("tfim", "heisenberg"))
    parser.add_argument("--letta-bond-dimensions", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--mps-bond-dimensions", type=int, nargs="+", default=(1, 2, 3, 4, 6))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args(argv)
    rows = run_one_dimensional_benchmark(
        model_names=tuple(arguments.models),
        letta_bond_dimensions=tuple(arguments.letta_bond_dimensions),
        mps_bond_dimensions=tuple(arguments.mps_bond_dimensions),
        seed=arguments.seed,
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
        repeats=arguments.repeats,
    )
    markdown = format_benchmark_markdown(rows)
    print(markdown, end="")
    if arguments.csv is not None:
        write_benchmark_csv(rows, arguments.csv)
    if arguments.markdown is not None:
        arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
        arguments.markdown.write_text(markdown)


if __name__ == "__main__":
    main()

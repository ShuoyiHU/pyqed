"""Compare uniform LETTA, uniform MPS, and exact TFIM observables.

Run from the repository root with

```
PYTHONPATH=. python -m pyqed._vuletta.examples.tfim_comparison
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

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

from pyqed._vuletta.operators import one_site_expectation, two_site_expectation
from pyqed._vuletta.solver import VULETTAOptions, vuletta
from pyqed._vuletta.state import expand_uniform_letta


@dataclass(frozen=True)
class ComparisonRow:
    method: str
    tensor_entry_count: int | None
    transfer_bond_dim: int | None
    energy_density: float
    transverse_magnetization: float
    zz_correlation: float
    converged: bool | None = None
    iterations: int | None = None
    function_evaluations: int | None = None
    residual: float | None = None
    message: str | None = None


def compare_tfim_methods(
    *,
    coupling=1.0,
    field=1.5,
    letta_bond_dimensions=(1, 2),
    mps_bond_dimensions=(2, 4),
    seed=3,
    tolerance=1.0e-8,
    max_iterations=150,
    bond_dimension_continuation=True,
    growth_noise=3.0e-2,
):
    """Return analytical, VULETTA, and VUMPS comparison rows."""

    growth_noise = float(growth_noise)
    if not np.isfinite(growth_noise) or growth_noise <= 0.0:
        raise ValueError("growth_noise must be finite and positive.")

    hamiltonian = tfim_bond_hamiltonian(coupling=coupling, field=field)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    zz = np.kron(z, z)
    physical_dim = 2
    rows = [
        ComparisonRow(
            method="exact, infinite",
            tensor_entry_count=None,
            transfer_bond_dim=None,
            energy_density=exact_tfim_energy_density(
                coupling=coupling,
                field=field,
            ),
            transverse_magnetization=exact_tfim_transverse_magnetization(
                coupling=coupling,
                field=field,
            ),
            zz_correlation=exact_tfim_zz_correlation(
                coupling=coupling,
                field=field,
            ),
        )
    ]

    previous_letta_result = None
    for bond_dim in letta_bond_dimensions:
        initial = None
        if (
            bond_dimension_continuation
            and previous_letta_result is not None
            and bond_dim > previous_letta_result.state.bond_dim
        ):
            initial = expand_uniform_letta(
                previous_letta_result.state,
                bond_dim,
                seed=seed + 1009 * bond_dim,
                relative_noise=growth_noise,
            )
        result = vuletta(
            hamiltonian,
            bond_dim=bond_dim,
            seed=seed,
            initial=initial,
            options=VULETTAOptions(
                max_iterations=max_iterations,
                tolerance=tolerance,
                energy_tolerance=1.0e-13,
            ),
        )
        rows.append(
            ComparisonRow(
                method=f"VULETTA, D={bond_dim}",
                tensor_entry_count=physical_dim**2 * bond_dim**2,
                transfer_bond_dim=physical_dim * bond_dim,
                energy_density=result.energy,
                transverse_magnetization=float(
                    one_site_expectation(result.state, x)
                ),
                zz_correlation=float(two_site_expectation(result.state, zz)),
                converged=result.converged,
                iterations=result.iterations,
                function_evaluations=result.function_evaluations,
                residual=result.residual_norm,
                message=result.message,
            )
        )
        previous_letta_result = result

    for bond_dim in mps_bond_dimensions:
        result = vumps(
            hamiltonian,
            bond_dim=bond_dim,
            seed=seed,
            options=VUMPSOptions(
                max_iterations=max_iterations,
                tolerance=tolerance,
            ),
        )
        rows.append(
            ComparisonRow(
                method=f"VUMPS, chi={bond_dim}",
                tensor_entry_count=physical_dim * bond_dim**2,
                transfer_bond_dim=bond_dim,
                energy_density=result.energy,
                transverse_magnetization=float(
                    mps_one_site_expectation(result.state, x)
                ),
                zz_correlation=float(
                    mps_two_site_expectation(result.state, zz)
                ),
                converged=result.converged,
                iterations=result.iterations,
                function_evaluations=None,
                residual=result.residual_norm,
                message=None,
            )
        )
    return tuple(rows)


def _format_optional_integer(value):
    return "-" if value is None else str(value)


def print_comparison(rows, *, coupling, field):
    reference = rows[0]
    print("Transverse-field Ising chain")
    print(f"H = -{coupling:g} sum(Z_n Z_(n+1)) - {field:g} sum(X_n)")
    print()
    print(
        f"{'method':<19} {'entries':>7} {'transfer':>8} "
        f"{'energy/site':>15} {'|dE|':>10} {'<X>':>13} {'|dX|':>10} "
        f"{'<ZZ>':>13} {'|dZZ|':>10}"
    )
    print("-" * 121)
    for row in rows:
        print(
            f"{row.method:<19} "
            f"{_format_optional_integer(row.tensor_entry_count):>7} "
            f"{_format_optional_integer(row.transfer_bond_dim):>8} "
            f"{row.energy_density:15.10f} "
            f"{abs(row.energy_density-reference.energy_density):10.3e} "
            f"{row.transverse_magnetization:13.10f} "
            f"{abs(row.transverse_magnetization-reference.transverse_magnetization):10.3e} "
            f"{row.zz_correlation:13.10f} "
            f"{abs(row.zz_correlation-reference.zz_correlation):10.3e}"
        )
    print()
    for row in rows:
        if row.converged is not None:
            print(
                f"{row.method}: converged={row.converged}, "
                f"iterations={row.iterations}, "
                f"function_evaluations="
                f"{_format_optional_integer(row.function_evaluations)}, "
                f"residual={row.residual:.3e}"
            )
            if row.message is not None:
                print(f"  termination: {row.message}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.5)
    parser.add_argument(
        "--letta-bond-dimensions",
        type=int,
        nargs="*",
        default=(1, 2, 3, 4),
    )
    parser.add_argument(
        "--mps-bond-dimensions",
        type=int,
        nargs="*",
        default=(2, 3, 4, 5, 6),
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--independent-letta-initializations",
        action="store_true",
        help="start each LETTA bond dimension from an independent random tensor",
    )
    parser.add_argument("--growth-noise", type=float, default=3.0e-2)
    arguments = parser.parse_args(argv)
    rows = compare_tfim_methods(
        coupling=arguments.coupling,
        field=arguments.field,
        letta_bond_dimensions=arguments.letta_bond_dimensions,
        mps_bond_dimensions=arguments.mps_bond_dimensions,
        seed=arguments.seed,
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
        bond_dimension_continuation=(
            not arguments.independent_letta_initializations
        ),
        growth_noise=arguments.growth_noise,
    )
    print_comparison(
        rows,
        coupling=arguments.coupling,
        field=arguments.field,
    )


if __name__ == "__main__":
    main()

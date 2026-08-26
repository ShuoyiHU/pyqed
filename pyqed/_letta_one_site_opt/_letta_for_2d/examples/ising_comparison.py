"""Compare site-factorized lattice LETTA with exact 2D Ising results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pyqed._letta_one_site_opt._letta_for_2d import (
    LETTADMROptions,
    automatic_bond_schedule,
    exact_ground_state,
    letta_dmrg,
    nearest_neighbor_bonds,
    network_expectation,
    state_vector_one_site_expectation,
    state_vector_two_site_expectation,
    transverse_field_ising_hamiltonian,
    transverse_field_ising_mpo,
)


@dataclass(frozen=True)
class IsingComparison:
    lattice_shape: tuple[int, int]
    contraction_shape: tuple[int, int]
    bond_dim: int
    exact_energy: float | None
    letta_energy: float
    exact_runtime_seconds: float | None
    letta_runtime_seconds: float
    exact_x: float | None
    letta_x: float
    exact_zz: float | None
    letta_zz: float
    converged: bool
    sweeps: int
    max_local_physical_dimension: int
    max_boundary_discarded_weight: float
    bond_schedule: tuple[tuple[int, int], ...]
    final_energy_density_change: float


def compare_ising(
    *,
    lattice_shape=(2, 3),
    coupling=1.0,
    field=1.5,
    bond_dim=4,
    seed=4,
    max_sweeps=8,
    tolerance=1.0e-8,
    run_exact=True,
    max_exact_sites=16,
    auto_orient=True,
    environment_granularity="column",
    sparse_mpo=True,
    matrix_free=False,
    gauge_mode="qr",
    boundary_bond_dim=None,
    boundary_cutoff=1.0e-12,
    use_bond_schedule=True,
    bond_expansion_noise=1.0e-3,
):
    lattice_shape = tuple(int(length) for length in lattice_shape)
    contraction_shape = lattice_shape
    if auto_orient and lattice_shape[1] > lattice_shape[0]:
        contraction_shape = lattice_shape[::-1]
    hamiltonian_mpo = transverse_field_ising_mpo(
        contraction_shape,
        coupling=coupling,
        field=field,
    )
    exact_energy = None
    exact_state = None
    exact_runtime_seconds = None
    exact_x = None
    exact_zz = None
    exact_enabled = run_exact and int(np.prod(lattice_shape)) <= max_exact_sites
    if exact_enabled:
        exact_start = perf_counter()
        hamiltonian = transverse_field_ising_hamiltonian(
            contraction_shape,
            coupling=coupling,
            field=field,
        )
        exact_energy, exact_state = exact_ground_state(hamiltonian)
        exact_runtime_seconds = perf_counter() - exact_start
    if use_bond_schedule and bond_dim > 2:
        schedule_dimensions, schedule_sweeps = automatic_bond_schedule(
            bond_dim,
            max_sweeps,
        )
    else:
        schedule_dimensions = (int(bond_dim),)
        schedule_sweeps = (int(max_sweeps),)
    staged = len(schedule_dimensions) > 1
    letta_start = perf_counter()
    result = letta_dmrg(
        hamiltonian_mpo,
        lattice_shape=contraction_shape,
        physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        options=LETTADMROptions(
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            environment_granularity=environment_granularity,
            use_sparse_mpo=sparse_mpo,
            matrix_free=matrix_free,
            gauge_mode=gauge_mode,
            boundary_bond_dim=boundary_bond_dim,
            boundary_cutoff=boundary_cutoff,
            bond_dimension_schedule=(schedule_dimensions if staged else None),
            bond_schedule_sweeps=(schedule_sweeps if staged else None),
            bond_expansion_noise=bond_expansion_noise,
        ),
    )
    letta_runtime_seconds = perf_counter() - letta_start
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    bonds = nearest_neighbor_bonds(contraction_shape)
    if exact_enabled:
        exact_x = np.mean(
            [
                state_vector_one_site_expectation(exact_state, x, site)
                for site in range(result.state.nsites)
            ]
        )
    x_sum_mpo = transverse_field_ising_mpo(
        contraction_shape,
        coupling=0.0,
        field=1.0,
    )
    letta_x = -network_expectation(result.state, x_sum_mpo) / result.state.nsites
    if exact_enabled:
        exact_zz = np.mean(
            [
                state_vector_two_site_expectation(
                    exact_state, z, left, z, right
                )
                for left, right in bonds
            ]
        )
    zz_sum_mpo = transverse_field_ising_mpo(
        contraction_shape,
        coupling=1.0,
        field=0.0,
    )
    letta_zz = -network_expectation(result.state, zz_sum_mpo) / len(bonds)
    max_local_physical_dimension = max(
        int(np.prod(tensor.shape[1:-1])) for tensor in result.state.tensors
    )
    return IsingComparison(
        lattice_shape=lattice_shape,
        contraction_shape=contraction_shape,
        bond_dim=int(bond_dim),
        exact_energy=exact_energy,
        letta_energy=result.energy,
        exact_runtime_seconds=exact_runtime_seconds,
        letta_runtime_seconds=letta_runtime_seconds,
        exact_x=None if exact_x is None else float(exact_x),
        letta_x=float(letta_x),
        exact_zz=None if exact_zz is None else float(exact_zz),
        letta_zz=float(letta_zz),
        converged=result.converged,
        sweeps=result.sweeps,
        max_local_physical_dimension=max_local_physical_dimension,
        max_boundary_discarded_weight=(
            result.max_boundary_discarded_weight
        ),
        bond_schedule=tuple(zip(schedule_dimensions, schedule_sweeps)),
        final_energy_density_change=result.history[-1].energy_density_change,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=(3,3))
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.5)
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--max-exact-sites", type=int, default=16)
    parser.add_argument("--no-auto-orient", action="store_true")
    parser.add_argument(
        "--environment-granularity",
        choices=("site", "column"),
        default="column",
    )
    parser.add_argument("--dense-mpo-channels", action="store_true")
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument(
        "--gauge-mode", choices=("qr", "scalar", "none"), default="qr"
    )
    parser.add_argument("--boundary-bond-dim", type=int)
    parser.add_argument("--boundary-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--bond-expansion-noise", type=float, default=1.0e-3)
    parser.add_argument(
        "--direct-initialization",
        action="store_true",
        help="disable lower-bond continuation",
    )
    arguments = parser.parse_args(argv)
    comparison = compare_ising(
        lattice_shape=tuple(arguments.shape),
        coupling=arguments.coupling,
        field=arguments.field,
        bond_dim=arguments.bond_dim,
        seed=arguments.seed,
        max_sweeps=arguments.max_sweeps,
        tolerance=arguments.tolerance,
        run_exact=not arguments.skip_exact,
        max_exact_sites=arguments.max_exact_sites,
        auto_orient=not arguments.no_auto_orient,
        environment_granularity=arguments.environment_granularity,
        sparse_mpo=not arguments.dense_mpo_channels,
        matrix_free=arguments.matrix_free,
        gauge_mode=arguments.gauge_mode,
        boundary_bond_dim=arguments.boundary_bond_dim,
        boundary_cutoff=arguments.boundary_cutoff,
        use_bond_schedule=not arguments.direct_initialization,
        bond_expansion_noise=arguments.bond_expansion_noise,
    )
    print(f"Open {comparison.lattice_shape[0]}x{comparison.lattice_shape[1]} TFIM")
    if comparison.contraction_shape != comparison.lattice_shape:
        print(f"contraction orientation: {comparison.contraction_shape}")
    print(f"LETTA bond dimension: {comparison.bond_dim}")
    if len(comparison.bond_schedule) > 1:
        schedule = " -> ".join(
            f"D={dimension} ({sweeps} sweeps)"
            for dimension, sweeps in comparison.bond_schedule
        )
        print(f"bond continuation: {schedule}")
    print(
        "largest local physical block: "
        f"{comparison.max_local_physical_dimension}"
    )
    if comparison.exact_energy is None:
        print(f"energy: exact=skipped, LETTA={comparison.letta_energy:.12f}")
        print(
            "runtime: exact=skipped, "
            f"LETTA={comparison.letta_runtime_seconds:.6f} s"
        )
        print(f"average X: exact=skipped, LETTA={comparison.letta_x:.12f}")
        print(
            "average nearest-neighbor ZZ: exact=skipped, "
            f"LETTA={comparison.letta_zz:.12f}"
        )
    else:
        print(
            f"energy: exact={comparison.exact_energy:.12f}, "
            f"LETTA={comparison.letta_energy:.12f}, "
            f"error={comparison.letta_energy-comparison.exact_energy:.3e}"
        )
        print(
            f"runtime: exact={comparison.exact_runtime_seconds:.6f} s, "
            f"LETTA={comparison.letta_runtime_seconds:.6f} s"
        )
        print(
            f"average X: exact={comparison.exact_x:.12f}, "
            f"LETTA={comparison.letta_x:.12f}"
        )
        print(
            f"average nearest-neighbor ZZ: exact={comparison.exact_zz:.12f}, "
            f"LETTA={comparison.letta_zz:.12f}"
        )
    print(
        f"converged={comparison.converged}, sweeps={comparison.sweeps}, "
        "final dE/site="
        f"{comparison.final_energy_density_change:.3e}"
    )
    if arguments.boundary_bond_dim is not None:
        print(
            "maximum boundary discarded weight: "
            f"{comparison.max_boundary_discarded_weight:.3e}"
        )


if __name__ == "__main__":
    main()

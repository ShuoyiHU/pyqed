"""Compare LETTA and MPS on an open snake-ordered 3D TFIM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    exact_ground_state,
    network_expectation,
)
from pyqed._letta_one_site_opt._letta_for_3d import (
    MPSDMRGOptions,
    letta_ground_state,
    mps_dmrg,
    nearest_neighbor_bonds,
    ordered_coordinates,
    transverse_field_ising_mpo,
    transverse_field_ising_sparse,
)


@dataclass(frozen=True)
class Ising3DComparison:
    lattice_shape: tuple[int, int, int]
    ordering: str
    environment_granularity: str
    nsites: int
    nbonds: int
    mpo_bond_dimension: int
    letta_energy: float | None
    mps_energy: float | None
    exact_energy: float | None
    letta_runtime_seconds: float | None
    mps_runtime_seconds: float | None
    exact_runtime_seconds: float | None
    letta_converged: bool | None
    mps_converged: bool | None
    letta_x: float | None
    mps_x: float | None
    letta_zz: float | None
    mps_zz: float | None


def compare_3d_ising(
    *,
    lattice_shape=(3, 3, 3),
    coupling=1.0,
    field=1.5,
    letta_bond_dim=1,
    mps_bond_dim=8,
    letta_sweeps=5,
    mps_sweeps=2,
    tolerance=1.0e-8,
    seed=4,
    run_letta=True,
    run_mps=True,
    run_exact=False,
    observables=False,
    ordering="compact",
    environment_granularity="site",
    matrix_free=False,
    boundary_bond_dim=None,
    boundary_cutoff=1.0e-12,
):
    shape = tuple(int(length) for length in lattice_shape)
    coordinates = ordered_coordinates(shape, ordering=ordering)
    bonds = nearest_neighbor_bonds(shape, ordering=ordering)
    hamiltonian = transverse_field_ising_mpo(
        shape,
        coupling=coupling,
        field=field,
        ordering=ordering,
    )

    letta_result = None
    letta_runtime = None
    if run_letta:
        start = perf_counter()
        letta_result = letta_ground_state(
            hamiltonian,
            lattice_shape=shape,
            bond_dim=letta_bond_dim,
            seed=seed,
            options=LETTADMROptions(
                max_sweeps=letta_sweeps,
                tolerance=tolerance,
                environment_granularity=environment_granularity,
                use_sparse_mpo=True,
                matrix_free=matrix_free,
                boundary_bond_dim=boundary_bond_dim,
                boundary_cutoff=boundary_cutoff,
            ),
            use_bond_schedule=letta_bond_dim > 2,
            ordering=ordering,
        )
        letta_runtime = perf_counter() - start

    mps_result = None
    mps_runtime = None
    if run_mps:
        start = perf_counter()
        mps_result = mps_dmrg(
            hamiltonian,
            bond_dim=mps_bond_dim,
            seed=seed,
            options=MPSDMRGOptions(
                max_sweeps=mps_sweeps,
                tolerance=tolerance,
            ),
        )
        mps_runtime = perf_counter() - start

    exact_energy = None
    exact_runtime = None
    if run_exact:
        start = perf_counter()
        sparse_hamiltonian = transverse_field_ising_sparse(
            shape,
            coupling=coupling,
            field=field,
            ordering=ordering,
        )
        exact_energy, _ = exact_ground_state(sparse_hamiltonian)
        exact_runtime = perf_counter() - start

    letta_x = mps_x = letta_zz = mps_zz = None
    if observables:
        x_sum = transverse_field_ising_mpo(
            shape, coupling=0.0, field=1.0, ordering=ordering
        )
        zz_sum = transverse_field_ising_mpo(
            shape, coupling=1.0, field=0.0, ordering=ordering
        )
        if letta_result is not None:
            letta_x = -network_expectation(letta_result.state, x_sum) / len(
                coordinates
            )
            letta_zz = -network_expectation(letta_result.state, zz_sum) / len(
                bonds
            )
        if mps_result is not None:
            mps_x = -mps_result.state.expectation(x_sum) / len(coordinates)
            mps_zz = -mps_result.state.expectation(zz_sum) / len(bonds)

    return Ising3DComparison(
        lattice_shape=shape,
        ordering=ordering,
        environment_granularity=environment_granularity,
        nsites=len(coordinates),
        nbonds=len(bonds),
        mpo_bond_dimension=max(hamiltonian.bond_dimensions, default=1),
        letta_energy=None if letta_result is None else letta_result.energy,
        mps_energy=None if mps_result is None else mps_result.energy,
        exact_energy=exact_energy,
        letta_runtime_seconds=letta_runtime,
        mps_runtime_seconds=mps_runtime,
        exact_runtime_seconds=exact_runtime,
        letta_converged=(
            None if letta_result is None else letta_result.converged
        ),
        mps_converged=None if mps_result is None else mps_result.converged,
        letta_x=None if letta_x is None else float(np.real(letta_x)),
        mps_x=None if mps_x is None else float(np.real(mps_x)),
        letta_zz=None if letta_zz is None else float(np.real(letta_zz)),
        mps_zz=None if mps_zz is None else float(np.real(mps_zz)),
    )


def _format_optional(value, format_spec=".12f"):
    return "skipped" if value is None else format(value, format_spec)


def _format_runtime(value):
    return "skipped" if value is None else f"{value:.6f} s"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.5)
    parser.add_argument("--letta-bond-dim", type=int, default=1)
    parser.add_argument("--mps-bond-dim", type=int, default=8)
    parser.add_argument("--letta-sweeps", type=int, default=100)
    parser.add_argument("--mps-sweeps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--skip-letta", action="store_true")
    parser.add_argument("--skip-mps", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--observables", action="store_true")
    parser.add_argument(
        "--ordering",
        choices=("compact", "layer-snake", "continuous-snake"),
        default="compact",
    )
    parser.add_argument(
        "--environment-granularity",
        choices=("site", "column"),
        default="site",
    )
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument("--boundary-bond-dim", type=int)
    parser.add_argument("--boundary-cutoff", type=float, default=1.0e-12)
    arguments = parser.parse_args(argv)

    result = compare_3d_ising(
        lattice_shape=tuple(arguments.shape),
        coupling=arguments.coupling,
        field=arguments.field,
        letta_bond_dim=arguments.letta_bond_dim,
        mps_bond_dim=arguments.mps_bond_dim,
        letta_sweeps=arguments.letta_sweeps,
        mps_sweeps=arguments.mps_sweeps,
        tolerance=arguments.tolerance,
        seed=arguments.seed,
        run_letta=not arguments.skip_letta,
        run_mps=not arguments.skip_mps,
        run_exact=arguments.exact,
        observables=arguments.observables,
        ordering=arguments.ordering,
        environment_granularity=arguments.environment_granularity,
        matrix_free=arguments.matrix_free,
        boundary_bond_dim=arguments.boundary_bond_dim,
        boundary_cutoff=arguments.boundary_cutoff,
    )
    shape = "x".join(map(str, result.lattice_shape))
    print(f"Open {shape} TFIM in {result.ordering} order")
    print(
        f"sites={result.nsites}, bonds={result.nbonds}, "
        f"MPO bond dimension={result.mpo_bond_dimension}, "
        f"environment granularity={result.environment_granularity}"
    )
    print(
        "energy: "
        f"LETTA={_format_optional(result.letta_energy)}, "
        f"MPS={_format_optional(result.mps_energy)}, "
        f"exact={_format_optional(result.exact_energy)}"
    )
    print(
        "runtime: "
        f"LETTA={_format_runtime(result.letta_runtime_seconds)}, "
        f"MPS={_format_runtime(result.mps_runtime_seconds)}, "
        f"exact={_format_runtime(result.exact_runtime_seconds)}"
    )
    print(
        f"converged: LETTA={result.letta_converged}, "
        f"MPS={result.mps_converged}"
    )
    if arguments.observables:
        print(
            "average X: "
            f"LETTA={_format_optional(result.letta_x)}, "
            f"MPS={_format_optional(result.mps_x)}"
        )
        print(
            "average nearest-neighbor ZZ: "
            f"LETTA={_format_optional(result.letta_zz)}, "
            f"MPS={_format_optional(result.mps_zz)}"
        )


if __name__ == "__main__":
    main()

"""Reproducible none/U(1)/SU(2) LETTA benchmarks for Heisenberg chains."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from ..._letta_one_site_opt import (
    AbelianSymmetry,
    LETTADMROptions,
    LatticeLETTA,
    LatticeMPO,
    ReducedFrontier,
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedSymmetry,
    letta_dmrg,
    su2_heisenberg_mpo,
)
from ..._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site
from ..solver import LETTATwoSiteOptions, letta_two_site_dmrg


def dense_heisenberg_mpo(nsites, *, coupling=1.0):
    """Return the open spin-1/2 Heisenberg chain as a five-channel MPO."""

    nsites = int(nsites)
    if nsites < 2:
        raise ValueError("nsites must be at least two")
    coupling = np.asarray(coupling)
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.diag([1.0, -1.0]).astype(complex)
    identity = np.eye(2, dtype=complex)
    operators = (coupling * sx, coupling * sy, coupling * sz)

    first = np.zeros((1, 5, 2, 2), dtype=complex)
    first[0, 0] = identity
    for channel, operator in enumerate(operators, start=1):
        first[0, channel] = operator
    factors = [first]
    for _site in range(1, nsites - 1):
        core = np.zeros((5, 5, 2, 2), dtype=complex)
        core[0, 0] = identity
        for channel, operator in enumerate(operators, start=1):
            core[0, channel] = operator
        core[1, 4] = sx
        core[2, 4] = sy
        core[3, 4] = sz
        core[4, 4] = identity
        factors.append(core)
    last = np.zeros((5, 1, 2, 2), dtype=complex)
    last[1, 0] = sx
    last[2, 0] = sy
    last[3, 0] = sz
    last[4, 0] = identity
    factors.append(last)
    return LatticeMPO(factors, lattice_shape=(1, nsites))


def _u1_symmetry():
    return AbelianSymmetry(
        physical_charges=(1, -1),
        sector=0,
        moduli=None,
        name="U(1) total 2Sz=0",
    )


def _su2_problem(nsites, multiplets_per_sector, seed, coupling):
    basis = ReducedPhysicalBasis.spin_half()
    symmetry = ReducedSymmetry.su2(basis, target_two_j=0)
    state = ReducedLatticeLETTA.random(
        (1, nsites),
        symmetry=symmetry,
        multiplets_per_sector=multiplets_per_sector,
        seed=seed,
    )
    hamiltonian = su2_heisenberg_mpo(
        nsites, physical_basis=basis, coupling=coupling
    )
    frontier_sites = ReducedFrontier.from_state(state).to_mps(state)
    expanded_bonds = tuple(
        expand_reduced_mps_site(site).shape[2] for site in frontier_sites[:-1]
    )
    return state, hamiltonian, expanded_bonds


def _solve(label, solver, hamiltonian, state, bond_dim, max_sweeps, tolerance):
    if solver == "one-site":
        result = letta_dmrg(
            hamiltonian,
            state=state,
            options=LETTADMROptions(
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                metric_tolerance=1.0e-12,
                gauge_mode="scalar",
                matrix_free=True,
            ),
        )
    elif solver == "two-site":
        result = letta_two_site_dmrg(
            hamiltonian,
            state=state,
            bond_dim=bond_dim,
            options=LETTATwoSiteOptions(
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                metric_tolerance=1.0e-12,
                split_method="conditional-svd",
                gauge_mode="scalar",
                matrix_free=True,
            ),
        )
    else:
        raise ValueError("solver must be 'one-site' or 'two-site'")
    return result


def _state_resources(state):
    if isinstance(state, ReducedLatticeLETTA):
        storage = sum(
            np.asarray(block).nbytes
            for tensor in state.tensors
            for block in tensor.values()
        )
        frontier_sites = ReducedFrontier.from_state(state).to_mps(state)
        expanded_entries = sum(
            expand_reduced_mps_site(site).size for site in frontier_sites
        )
        return {
            "parameter_count": int(state.parameter_count),
            "allocated_tensor_entries": int(state.parameter_count),
            "tensor_storage_bytes": int(storage),
            "magnetic_expanded_mps_entries": int(expanded_entries),
        }
    return {
        "parameter_count": int(state.parameter_count),
        "allocated_tensor_entries": int(state.dense_parameter_count),
        "tensor_storage_bytes": int(sum(tensor.nbytes for tensor in state.tensors)),
        "magnetic_expanded_mps_entries": None,
    }


def _record(result, elapsed, peak_memory_bytes, repeat, exact_energy):
    updates = [update for sweep in result.history for update in sweep.updates]
    dimensions = [int(update.local_dimension) for update in updates]
    full_dimensions = [
        int(
            update.full_local_dimension
            if update.full_local_dimension is not None
            else update.local_dimension
        )
        for update in updates
    ]
    resources = _state_resources(result.state)
    return {
        "repeat": int(repeat),
        "energy": float(result.energy),
        "energy_error": (
            None if exact_energy is None else float(abs(result.energy - exact_energy))
        ),
        "converged": bool(result.converged),
        "sweeps": int(result.sweeps),
        "local_updates": len(updates),
        "elapsed_seconds": float(elapsed),
        "peak_traced_memory_bytes": int(peak_memory_bytes),
        "max_local_dimension": max(dimensions, default=0),
        "max_full_local_dimension": max(full_dimensions, default=0),
        "local_dimension_sum": int(sum(dimensions)),
        "cubic_local_work_proxy": int(sum(value**3 for value in dimensions)),
        "symmetry_violation": float(result.state.symmetry_violation()),
        **resources,
    }


def _summary(records):
    first = records[0]
    return {
        "energy_median": float(median(item["energy"] for item in records)),
        "energy_error_max": (
            None
            if first["energy_error"] is None
            else float(max(item["energy_error"] for item in records))
        ),
        "elapsed_seconds_median": float(
            median(item["elapsed_seconds"] for item in records)
        ),
        "converged_fraction": float(
            sum(item["converged"] for item in records) / len(records)
        ),
        "sweeps_median": float(median(item["sweeps"] for item in records)),
        "parameter_count": first["parameter_count"],
        "allocated_tensor_entries": first["allocated_tensor_entries"],
        "tensor_storage_bytes": first["tensor_storage_bytes"],
        "peak_traced_memory_bytes_median": float(
            median(item["peak_traced_memory_bytes"] for item in records)
        ),
        "magnetic_expanded_mps_entries": first["magnetic_expanded_mps_entries"],
        "max_local_dimension": max(item["max_local_dimension"] for item in records),
        "max_full_local_dimension": max(
            item["max_full_local_dimension"] for item in records
        ),
        "local_dimension_sum_median": float(
            median(item["local_dimension_sum"] for item in records)
        ),
        "cubic_local_work_proxy_median": float(
            median(item["cubic_local_work_proxy"] for item in records)
        ),
        "symmetry_violation_max": float(
            max(item["symmetry_violation"] for item in records)
        ),
        "runs": records,
    }


def _ratio(numerator, denominator):
    return None if not denominator else float(numerator / denominator)


def run_heisenberg_symmetry_benchmark(
    *,
    nsites=6,
    solvers=("one-site", "two-site"),
    bond_dim=None,
    multiplets_per_sector=3,
    max_sweeps=8,
    tolerance=1.0e-9,
    repeats=3,
    seed=7,
    coupling=1.0,
    exact_max_sites=10,
    energy_match_tolerance=None,
):
    """Compare no symmetry, U(1), and exact reduced SU(2)."""

    nsites = int(nsites)
    repeats = int(repeats)
    if nsites < 2 or nsites % 2:
        raise ValueError("benchmark requires an even nsites >= 2 singlet chain")
    if repeats <= 0 or max_sweeps <= 0 or multiplets_per_sector <= 0:
        raise ValueError("repeats, max_sweeps, and multiplets_per_sector must be positive")
    if energy_match_tolerance is None:
        energy_match_tolerance = max(1.0e-8, 100.0 * float(tolerance))
    energy_match_tolerance = float(energy_match_tolerance)
    if energy_match_tolerance <= 0.0:
        raise ValueError("energy_match_tolerance must be positive")
    solvers = tuple(solvers)
    su2_initial, su2_hamiltonian, expanded_bonds = _su2_problem(
        nsites, multiplets_per_sector, seed, coupling
    )
    if bond_dim is None:
        bond_dim = max(expanded_bonds, default=1)
    bond_dim = int(bond_dim)
    dense_hamiltonian = dense_heisenberg_mpo(nsites, coupling=coupling)
    u1_initial = LatticeLETTA.random(
        (1, nsites),
        physical_dim=2,
        bond_dim=bond_dim,
        seed=seed,
        real=False,
        symmetry=_u1_symmetry(),
    )
    dense_initial = u1_initial.without_symmetry()
    exact_energy = None
    if nsites <= int(exact_max_sites):
        exact_energy = float(np.linalg.eigvalsh(dense_hamiltonian.to_dense())[0])

    report = {
        "metadata": {
            "nsites": nsites,
            "bond_dim": bond_dim,
            "multiplets_per_sector": int(multiplets_per_sector),
            "su2_expanded_bond_dimensions": expanded_bonds,
            "max_sweeps": int(max_sweeps),
            "tolerance": float(tolerance),
            "repeats": repeats,
            "seed": int(seed),
            "coupling": float(coupling),
            "exact_energy": exact_energy,
            "energy_match_tolerance": energy_match_tolerance,
            "gauge_mode": "scalar",
            "matrix_free": True,
            "memory_measurement": (
                "Python tracemalloc peak from a separate untimed replay"
            ),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "solvers": {},
    }
    for solver_index, solver in enumerate(solvers):
        records = {"none": [], "u1": [], "su2": []}
        inputs = {
            "none": (dense_hamiltonian, dense_initial),
            "u1": (dense_hamiltonian, u1_initial),
            "su2": (su2_hamiltonian, su2_initial),
        }
        for repeat in range(repeats):
            labels = list(inputs)
            shift = (repeat + solver_index) % len(labels)
            labels = labels[shift:] + labels[:shift]
            for label in labels:
                hamiltonian, initial = inputs[label]
                gc.collect()
                start = perf_counter()
                result = _solve(
                    label,
                    solver,
                    hamiltonian,
                    initial.copy(),
                    bond_dim,
                    max_sweeps,
                    tolerance,
                )
                elapsed = perf_counter() - start
                gc.collect()
                tracemalloc.start()
                try:
                    _solve(
                        label,
                        solver,
                        hamiltonian,
                        initial.copy(),
                        bond_dim,
                        max_sweeps,
                        tolerance,
                    )
                    _current_memory, peak_memory = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                records[label].append(
                    _record(result, elapsed, peak_memory, repeat, exact_energy)
                )
        summaries = {label: _summary(items) for label, items in records.items()}
        su2 = summaries["su2"]
        median_energies = [
            summaries[label]["energy_median"] for label in ("none", "u1", "su2")
        ]
        energy_spread = float(max(median_energies) - min(median_energies))
        exact_error = (
            None
            if exact_energy is None
            else float(max(abs(value - exact_energy) for value in median_energies))
        )
        summaries["agreement"] = {
            "energy_spread": energy_spread,
            "max_exact_energy_error": exact_error,
            "tolerance": energy_match_tolerance,
            "efficiency_comparison_valid": bool(
                energy_spread <= energy_match_tolerance
                and (exact_error is None or exact_error <= energy_match_tolerance)
            ),
        }
        summaries["comparisons"] = {
            label: {
                "energy_difference_from_su2": float(
                    abs(summary["energy_median"] - su2["energy_median"])
                ),
                "su2_over_baseline_elapsed": _ratio(
                    su2["elapsed_seconds_median"],
                    summary["elapsed_seconds_median"],
                ),
                "su2_over_baseline_parameters": _ratio(
                    su2["parameter_count"], summary["parameter_count"]
                ),
                "su2_over_baseline_peak_memory": _ratio(
                    su2["peak_traced_memory_bytes_median"],
                    summary["peak_traced_memory_bytes_median"],
                ),
                "su2_over_baseline_local_dimension": _ratio(
                    su2["max_local_dimension"], summary["max_local_dimension"]
                ),
                "su2_over_baseline_cubic_work": _ratio(
                    su2["cubic_local_work_proxy_median"],
                    summary["cubic_local_work_proxy_median"],
                ),
            }
            for label, summary in summaries.items()
            if label in {"none", "u1"}
        }
        report["solvers"][solver] = summaries
    return report


def _print_report(report):
    metadata = report["metadata"]
    print(
        f"Heisenberg N={metadata['nsites']} bond_dim={metadata['bond_dim']} "
        f"SU2 multiplets/sector={metadata['multiplets_per_sector']}"
    )
    for solver, data in report["solvers"].items():
        print(f"\n{solver}")
        for label in ("none", "u1", "su2"):
            item = data[label]
            print(
                f"  {label:4s} E={item['energy_median']:.14f} "
                f"time={item['elapsed_seconds_median']:.4f}s "
                f"params={item['parameter_count']} "
                f"peak={item['peak_traced_memory_bytes_median'] / 1024**2:.2f}MiB "
                f"max-local={item['max_local_dimension']}"
            )
        agreement = data["agreement"]
        if not agreement["efficiency_comparison_valid"]:
            print(
                "  WARNING: energies do not match within "
                f"{agreement['tolerance']:.3e}; increase bond/multiplet capacity "
                "before interpreting efficiency ratios"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsites", type=int, default=6)
    parser.add_argument(
        "--solver", choices=("one-site", "two-site", "both"), default="both"
    )
    parser.add_argument("--bond-dim", type=int, default=None)
    parser.add_argument("--multiplets-per-sector", type=int, default=3)
    parser.add_argument("--max-sweeps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--exact-max-sites", type=int, default=10)
    parser.add_argument("--energy-match-tolerance", type=float, default=None)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    solvers = (
        ("one-site", "two-site")
        if args.solver == "both"
        else (args.solver,)
    )
    report = run_heisenberg_symmetry_benchmark(
        nsites=args.nsites,
        solvers=solvers,
        bond_dim=args.bond_dim,
        multiplets_per_sector=args.multiplets_per_sector,
        max_sweeps=args.max_sweeps,
        tolerance=args.tolerance,
        repeats=args.repeats,
        seed=args.seed,
        coupling=args.coupling,
        exact_max_sites=args.exact_max_sites,
        energy_match_tolerance=args.energy_match_tolerance,
    )
    _print_report(report)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json_output}")
    return report


if __name__ == "__main__":
    main()


__all__ = ["dense_heisenberg_mpo", "run_heisenberg_symmetry_benchmark"]

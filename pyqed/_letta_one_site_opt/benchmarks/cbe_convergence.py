"""Compare one-site, exact/shrewd LETTA-CBE, and two-site LETTA.

Example:

    python pyqed/_letta_one_site_opt/benchmarks/cbe_convergence.py

or:

    python -m pyqed._letta_one_site_opt.benchmarks.cbe_convergence
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pyqed._letta_two_site_opt import (
    LETTATwoSiteOptions,
    letta_two_site_dmrg,
)
from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    LatticeLETTA,
    exact_ground_state,
    letta_dmrg,
)
from pyqed._letta_one_site_opt._letta_for_2d import (
    transverse_field_ising_mpo,
)


SOLVERS = (
    "one_site",
    "letta_cbe_exact",
    "letta_cbe_shrewd",
    "two_site",
)


def _quiet_numerical_call(function, *args, **kwargs):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return function(*args, **kwargs)


def _state_fingerprint(state):
    digest = hashlib.sha256()
    for tensor in state.tensors:
        contiguous = np.ascontiguousarray(tensor)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else 0.0


def _one_site_record(result, solver, elapsed, fingerprint, exact_energy):
    updates = [
        update
        for sweep in result.history
        for update in sweep.updates
    ]
    cbe_updates = [
        update for update in updates if update.cbe_expansion_dimension > 0
    ]
    expanded_updates = [
        update
        for update in cbe_updates
        if update.cbe_expanded_energy is not None
    ]
    accepted_cbe = [
        update for update in expanded_updates if not update.cbe_fallback
    ]
    fallback_cbe = [
        update for update in expanded_updates if update.cbe_fallback
    ]
    selectors = {
        update.cbe_selector
        for update in cbe_updates
        if update.cbe_selector is not None
    }
    selector = next(iter(selectors)) if len(selectors) == 1 else None
    metric_materializations = {
        update.cbe_materialized_pair_metric for update in cbe_updates
    }
    tangent_materializations = {
        update.cbe_materialized_tangent_jacobian for update in cbe_updates
    }
    pair_tensor_materializations = {
        update.cbe_materialized_pair_tensor for update in cbe_updates
    }
    trim_methods = {
        update.cbe_trim_method
        for update in cbe_updates
        if update.cbe_trim_method is not None
    }
    record = {
        "solver": solver,
        "initial_state_fingerprint": fingerprint,
        "energy": float(result.energy),
        "energy_error": (
            None
            if exact_energy is None
            else float(result.energy - exact_energy)
        ),
        "elapsed_seconds": float(elapsed),
        "sweeps": int(result.sweeps),
        "updates": len(updates),
        "accepted_updates": sum(int(update.accepted) for update in updates),
        "cbe_updates": len(cbe_updates),
        "cbe_accepted": sum(
            int(not update.cbe_fallback) for update in cbe_updates
        ),
        "cbe_fallbacks": sum(
            int(update.cbe_fallback) for update in cbe_updates
        ),
        "cbe_baseline_selected": sum(
            int(update.cbe_baseline_selected) for update in expanded_updates
        ),
        "selector": selector,
        "mean_preselection_dimension": _mean(
            update.cbe_preselection_dimension for update in cbe_updates
        ),
        "mean_projection_iterations": _mean(
            update.cbe_projection_iterations for update in cbe_updates
        ),
        "mean_selector_pair_actions": _mean(
            update.cbe_selector_pair_action_count for update in cbe_updates
        ),
        "mean_selector_pair_metrics": _mean(
            update.cbe_selector_pair_metric_count for update in cbe_updates
        ),
        "mean_selector_merged_pairs": _mean(
            update.cbe_selector_merged_pair_count for update in cbe_updates
        ),
        "mean_preselection_output_size": _mean(
            update.cbe_preselection_output_size for update in cbe_updates
        ),
        "mean_final_output_size": _mean(
            update.cbe_final_output_size for update in cbe_updates
        ),
        "materialized_pair_tensor": (
            next(iter(pair_tensor_materializations))
            if len(pair_tensor_materializations) == 1
            else None
        ),
        "materialized_pair_metric": (
            next(iter(metric_materializations))
            if len(metric_materializations) == 1
            else None
        ),
        "materialized_tangent_jacobian": (
            next(iter(tangent_materializations))
            if len(tangent_materializations) == 1
            else None
        ),
        "trim_method": (
            next(iter(trim_methods)) if len(trim_methods) == 1 else None
        ),
        "mean_missing_norm": _mean(
            update.cbe_missing_norm for update in cbe_updates
        ),
        "mean_captured_weight": _mean(
            update.cbe_captured_weight for update in cbe_updates
        ),
        "mean_selection_loss": _mean(
            update.cbe_selection_loss for update in cbe_updates
        ),
        "mean_trim_loss": _mean(
            update.cbe_trim_loss for update in cbe_updates
        ),
        "mean_expanded_energy_gain": _mean(
            update.cbe_old_energy - update.cbe_expanded_energy
            for update in expanded_updates
        ),
        "mean_trimmed_energy_gain": _mean(
            update.cbe_old_energy - update.cbe_trimmed_energy
            for update in expanded_updates
        ),
        "mean_accepted_trim_loss": _mean(
            update.cbe_trim_loss for update in accepted_cbe
        ),
        "mean_fallback_trim_loss": _mean(
            update.cbe_trim_loss for update in fallback_cbe
        ),
        "mean_cbe_vs_baseline_energy": _mean(
            update.cbe_trimmed_energy - update.cbe_baseline_energy
            for update in expanded_updates
            if update.cbe_baseline_energy is not None
        ),
        "mean_cbe_baseline_allowance": _mean(
            update.cbe_baseline_allowance for update in expanded_updates
        ),
        "sweep_energies": [float(sweep.energy) for sweep in result.history],
    }
    return record


def _two_site_record(result, elapsed, fingerprint, exact_energy):
    updates = [
        update
        for sweep in result.history
        for update in sweep.updates
    ]
    return {
        "solver": "two_site",
        "initial_state_fingerprint": fingerprint,
        "energy": float(result.energy),
        "energy_error": (
            None
            if exact_energy is None
            else float(result.energy - exact_energy)
        ),
        "elapsed_seconds": float(elapsed),
        "sweeps": int(result.sweeps),
        "updates": len(updates),
        "accepted_updates": sum(int(update.accepted) for update in updates),
        "cbe_updates": 0,
        "cbe_accepted": 0,
        "cbe_fallbacks": 0,
        "cbe_baseline_selected": 0,
        "selector": None,
        "mean_preselection_dimension": 0.0,
        "mean_projection_iterations": 0.0,
        "mean_selector_pair_actions": 0.0,
        "mean_selector_pair_metrics": 0.0,
        "mean_selector_merged_pairs": 0.0,
        "mean_preselection_output_size": 0.0,
        "mean_final_output_size": 0.0,
        "materialized_pair_tensor": None,
        "materialized_pair_metric": None,
        "materialized_tangent_jacobian": None,
        "trim_method": None,
        "mean_missing_norm": 0.0,
        "mean_captured_weight": 0.0,
        "mean_selection_loss": 0.0,
        "mean_trim_loss": _mean(
            update.metric_truncation_loss for update in updates
        ),
        "mean_expanded_energy_gain": 0.0,
        "mean_trimmed_energy_gain": 0.0,
        "mean_accepted_trim_loss": 0.0,
        "mean_fallback_trim_loss": 0.0,
        "mean_cbe_vs_baseline_energy": 0.0,
        "mean_cbe_baseline_allowance": 0.0,
        "sweep_energies": [float(sweep.energy) for sweep in result.history],
    }


def run_comparison(
    *,
    shape=(2, 3),
    bond_dim=1,
    expansion_dimension=1,
    max_sweeps=4,
    seed=731,
    coupling=1.0,
    field=1.0,
    tolerance=1.0e-12,
    exact_max_sites=10,
):
    """Run all methods from the same deterministic initial LETTA state."""

    shape = tuple(int(length) for length in shape)
    bond_dim = int(bond_dim)
    expansion_dimension = int(expansion_dimension)
    max_sweeps = int(max_sweeps)
    hamiltonian = transverse_field_ising_mpo(
        shape, coupling=float(coupling), field=float(field)
    )
    initial_state = LatticeLETTA.random(
        shape,
        physical_dim=2,
        bond_dim=bond_dim,
        seed=int(seed),
    )
    fingerprint = _state_fingerprint(initial_state)
    initial_energy = float(initial_state.expectation(hamiltonian))
    nsites = int(np.prod(shape))
    exact_energy = None
    if exact_max_sites and nsites <= int(exact_max_sites):
        exact_energy, _vector = _quiet_numerical_call(
            exact_ground_state, hamiltonian.to_dense()
        )

    records = []
    for solver in SOLVERS:
        started = time.perf_counter()
        if solver == "two_site":
            result = _quiet_numerical_call(
                letta_two_site_dmrg,
                hamiltonian,
                state=initial_state,
                bond_dim=bond_dim,
                options=LETTATwoSiteOptions(
                    max_sweeps=max_sweeps,
                    tolerance=tolerance,
                    matrix_free=True,
                    use_sparse_mpo=True,
                    split_method="metric-als",
                    one_site_polish_sweeps=0,
                ),
            )
            elapsed = time.perf_counter() - started
            record = _two_site_record(
                result, elapsed, fingerprint, exact_energy
            )
        else:
            result = _quiet_numerical_call(
                letta_dmrg,
                hamiltonian,
                state=initial_state,
                options=LETTADMROptions(
                    max_sweeps=max_sweeps,
                    tolerance=tolerance,
                    matrix_free=True,
                    use_sparse_mpo=True,
                    cbe_enabled=solver.startswith("letta_cbe_"),
                    cbe_selector=(
                        "shrewd"
                        if solver == "letta_cbe_shrewd"
                        else "exact"
                    ),
                    cbe_expansion_dimension=expansion_dimension,
                ),
            )
            elapsed = time.perf_counter() - started
            record = _one_site_record(
                result, solver, elapsed, fingerprint, exact_energy
            )
        if _state_fingerprint(initial_state) != fingerprint:
            raise RuntimeError("a benchmark solver mutated the shared initial state.")
        records.append(record)

    return {
        "shape": list(shape),
        "bond_dim": bond_dim,
        "expansion_dimension": expansion_dimension,
        "max_sweeps": max_sweeps,
        "seed": int(seed),
        "coupling": float(coupling),
        "field": float(field),
        "initial_energy": initial_energy,
        "exact_energy": exact_energy,
        "initial_state_fingerprint": fingerprint,
        "records": records,
        "cost_note": (
            "The exact selector materializes the pair metric and tangent "
            "Jacobian. The strict shrewd selector streams weighted half "
            "contractions through sparse MPO transitions, then raises and "
            "tangent-projects (H-E*N)psi in a restricted expanded one-site "
            "metric. It solves only that one-site problem and trims in the "
            "one-site LETTA metric. It invokes no pair action, pair metric, "
            "or merged-pair tensor. See the cbe_scaling benchmark for "
            "opt_einsum path evidence for the Hamiltonian contractions."
        ),
    }


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 2x3") from error
    if len(shape) != 2 or any(length <= 0 for length in shape):
        raise argparse.ArgumentTypeError("shape must contain two positive lengths")
    return shape


def _print_table(report):
    print(
        "solver             energy             error          seconds  "
        "cbe-ok  fallback  baseline  missing-norm  captured  dE-expand   dE-trim"
    )
    for record in report["records"]:
        error = record["energy_error"]
        error_text = "n/a" if error is None else f"{error:.3e}"
        print(
            f"{record['solver']:<18} "
            f"{record['energy']: .12f}  "
            f"{error_text:>11}  "
            f"{record['elapsed_seconds']:7.3f}  "
            f"{record['cbe_accepted']:6d}  "
            f"{record['cbe_fallbacks']:8d}  "
            f"{record['cbe_baseline_selected']:8d}  "
            f"{record['mean_missing_norm']:12.3e}  "
            f"{record['mean_captured_weight']:8.3f}  "
            f"{record['mean_expanded_energy_gain']:9.3e}  "
            f"{record['mean_trimmed_energy_gain']:9.3e}"
        )
    print(report["cost_note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, default=(3, 3))
    parser.add_argument("--bond-dim", type=int, default=5)
    parser.add_argument("--expansion-dimension", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--exact-max-sites", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_comparison(
        shape=arguments.shape,
        bond_dim=arguments.bond_dim,
        expansion_dimension=arguments.expansion_dimension,
        max_sweeps=arguments.max_sweeps,
        seed=arguments.seed,
        coupling=arguments.coupling,
        field=arguments.field,
        tolerance=arguments.tolerance,
        exact_max_sites=arguments.exact_max_sites,
    )
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(report)
    return report


if __name__ == "__main__":
    main()

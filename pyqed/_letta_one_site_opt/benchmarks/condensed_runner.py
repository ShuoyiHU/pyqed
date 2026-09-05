"""Shared, fair five-solver runner for condensed-model benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

from pyqed.mps.dmrg import DMRG
from pyqed.mps.mps import MPS

from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg

from ..operators import exact_ground_state
from ..solver import LETTADMROptions, letta_dmrg
from ..state import LatticeLETTA
from .condensed_models import CondensedModel, build_model


SOLVERS = (
    "letta_one_site",
    "letta_cbe_exact",
    "letta_cbe_strict",
    "letta_two_site",
    "mps_two_site",
)


@dataclass(frozen=True)
class SharedInitialState:
    """One physical MPS represented simultaneously as MPS and LETTA."""

    mps: MPS
    letta: LatticeLETTA
    fingerprint: str
    energy: float


def _hash_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def mps_state_vector(factors):
    """Contract a small open-boundary MPS to a dense vector for tests."""

    factors = tuple(np.asarray(factor) for factor in factors)
    if not factors:
        raise ValueError("an MPS needs at least one factor.")
    if factors[0].shape[0] != 1 or factors[-1].shape[-1] != 1:
        raise ValueError("the MPS must have open boundaries.")
    tensor = factors[0][0]
    for site, factor in enumerate(factors[1:], start=1):
        if tensor.shape[-1] != factor.shape[0]:
            raise ValueError(f"MPS bond mismatch before site {site}.")
        tensor = np.tensordot(tensor, factor, axes=([-1], [0]))
    return np.squeeze(tensor, axis=-1).reshape(-1)


def mps_to_letta(factors, lattice_shape):
    """Embed an MPS exactly by making LETTA neighbor legs state-independent."""

    factors = tuple(np.asarray(factor) for factor in factors)
    physical_dim = int(factors[0].shape[1])
    template = LatticeLETTA.random(
        lattice_shape,
        physical_dim=physical_dim,
        bond_dim=1,
        seed=0,
    )
    if len(factors) != template.nsites:
        raise ValueError("MPS length does not match the LETTA lattice.")
    tensors = []
    for site, (factor, neighborhood) in enumerate(
        zip(factors, (template.site_neighborhood(i) for i in range(template.nsites)))
    ):
        if factor.ndim != 3 or factor.shape[1] != physical_dim:
            raise ValueError(f"MPS factor {site} has incompatible dimensions.")
        extra_legs = len(neighborhood) - 1
        reshaped = factor.reshape(
            (factor.shape[0], physical_dim)
            + (1,) * extra_legs
            + (factor.shape[2],)
        )
        target_shape = (
            (factor.shape[0], physical_dim)
            + (physical_dim,) * extra_legs
            + (factor.shape[2],)
        )
        tensors.append(np.broadcast_to(reshaped, target_shape).copy())
    return LatticeLETTA(lattice_shape, physical_dim, tensors)


def make_shared_initial_state(model: CondensedModel, *, bond_dim=4, seed=731):
    """Generate one normalized random MPS and its exact LETTA embedding."""

    if not isinstance(model, CondensedModel):
        raise TypeError("model must be a CondensedModel.")
    bond_dim = int(bond_dim)
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive.")
    rng = np.random.default_rng(int(seed))
    factors = []
    for site in range(model.nsites):
        left = 1 if site == 0 else bond_dim
        right = 1 if site == model.nsites - 1 else bond_dim
        factor = rng.normal(size=(left, model.physical_dim, right))
        factor /= np.sqrt(factor.size)
        factors.append(factor)
    norm = np.linalg.norm(mps_state_vector(factors))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("random MPS initialization produced a zero state.")
    factors[0] = factors[0] / norm
    mps = MPS([factor.copy() for factor in factors], labels=["lv", "p", "rv"])
    letta = mps_to_letta(factors, model.lattice_shape)
    fingerprint = _hash_arrays(factors)
    energy = float(letta.expectation(model.mpo))
    return SharedInitialState(mps, letta, fingerprint, energy)


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else 0.0


def _error(energy, exact_energy):
    return None if exact_energy is None else float(energy - exact_energy)


def _materialized(updates, attribute):
    values = [getattr(update, attribute) for update in updates]
    known = [bool(value) for value in values if value is not None]
    return any(known) if known else None


def _letta_one_site_record(result, solver, elapsed, fingerprint, exact_energy):
    updates = [update for sweep in result.history for update in sweep.updates]
    cbe_updates = [
        update for update in updates if update.cbe_expansion_dimension > 0
    ]
    accepted_cbe = [update for update in cbe_updates if not update.cbe_fallback]
    selectors = {
        update.cbe_selector
        for update in cbe_updates
        if update.cbe_selector is not None
    }
    selector = next(iter(selectors)) if len(selectors) == 1 else None
    return {
        "solver": solver,
        "representation": "LETTA",
        "initial_state_fingerprint": fingerprint,
        "energy": float(result.energy),
        "energy_error": _error(result.energy, exact_energy),
        "elapsed_seconds": float(elapsed),
        "sweeps": int(result.sweeps),
        "converged": bool(result.converged),
        "message": str(result.message),
        "updates": len(updates),
        "accepted_updates": sum(int(update.accepted) for update in updates),
        "parameter_count": int(result.state.parameter_count),
        "cbe_updates": len(cbe_updates),
        "cbe_accepted": len(accepted_cbe),
        "cbe_fallbacks": sum(int(update.cbe_fallback) for update in cbe_updates),
        "cbe_baseline_selected": sum(
            int(update.cbe_baseline_selected) for update in cbe_updates
        ),
        "selector": selector,
        "selector_pair_actions": sum(
            int(update.cbe_selector_pair_action_count or 0)
            for update in cbe_updates
        ),
        "selector_pair_metrics": sum(
            int(update.cbe_selector_pair_metric_count or 0)
            for update in cbe_updates
        ),
        "selector_merged_pairs": sum(
            int(update.cbe_selector_merged_pair_count or 0)
            for update in cbe_updates
        ),
        "materialized_pair_tensor": _materialized(
            cbe_updates, "cbe_materialized_pair_tensor"
        ),
        "materialized_pair_metric": _materialized(
            cbe_updates, "cbe_materialized_pair_metric"
        ),
        "materialized_tangent_jacobian": _materialized(
            cbe_updates, "cbe_materialized_tangent_jacobian"
        ),
        "mean_missing_norm": _mean(
            update.cbe_missing_norm for update in cbe_updates
        ),
        "mean_captured_weight": _mean(
            update.cbe_captured_weight for update in cbe_updates
        ),
        "mean_trim_loss": _mean(update.cbe_trim_loss for update in cbe_updates),
        "mean_accepted_trim_loss": _mean(
            update.cbe_trim_loss for update in accepted_cbe
        ),
        "mean_cbe_vs_baseline_energy": _mean(
            update.cbe_trimmed_energy - update.cbe_baseline_energy
            for update in cbe_updates
            if update.cbe_trimmed_energy is not None
            and update.cbe_baseline_energy is not None
        ),
        "mean_cbe_baseline_allowance": _mean(
            update.cbe_baseline_allowance for update in cbe_updates
        ),
        "sweep_energies": [float(sweep.energy) for sweep in result.history],
    }


def _letta_two_site_record(result, elapsed, fingerprint, exact_energy):
    updates = [update for sweep in result.history for update in sweep.updates]
    return {
        "solver": "letta_two_site",
        "representation": "LETTA",
        "initial_state_fingerprint": fingerprint,
        "energy": float(result.energy),
        "energy_error": _error(result.energy, exact_energy),
        "elapsed_seconds": float(elapsed),
        "sweeps": int(result.sweeps),
        "converged": bool(result.converged),
        "message": str(result.message),
        "updates": len(updates),
        "accepted_updates": sum(int(update.accepted) for update in updates),
        "parameter_count": int(result.state.parameter_count),
        "cbe_updates": 0,
        "cbe_accepted": 0,
        "cbe_fallbacks": 0,
        "cbe_baseline_selected": 0,
        "selector": None,
        "selector_pair_actions": 0,
        "selector_pair_metrics": 0,
        "selector_merged_pairs": 0,
        "materialized_pair_tensor": None,
        "materialized_pair_metric": None,
        "materialized_tangent_jacobian": None,
        "mean_missing_norm": 0.0,
        "mean_captured_weight": 0.0,
        "mean_trim_loss": _mean(
            update.metric_truncation_loss for update in updates
        ),
        "mean_accepted_trim_loss": 0.0,
        "mean_cbe_vs_baseline_energy": 0.0,
        "mean_cbe_baseline_allowance": 0.0,
        "sweep_energies": [float(sweep.energy) for sweep in result.history],
    }


def _mps_two_site_record(dmrg, elapsed, fingerprint, exact_energy):
    energy = float(np.real(dmrg.e_tot))
    factors = dmrg.ground_state.factors
    history = list(dmrg.sweep_history)
    return {
        "solver": "mps_two_site",
        "representation": "MPS",
        "initial_state_fingerprint": fingerprint,
        "energy": energy,
        "energy_error": _error(energy, exact_energy),
        "elapsed_seconds": float(elapsed),
        "sweeps": max(1, len(history)),
        "converged": bool(dmrg.converged),
        "message": "CONVERGED" if dmrg.converged else "STOP: MAXIMUM SWEEPS REACHED",
        "updates": 0,
        "accepted_updates": 0,
        "parameter_count": int(sum(np.asarray(factor).size for factor in factors)),
        "cbe_updates": 0,
        "cbe_accepted": 0,
        "cbe_fallbacks": 0,
        "cbe_baseline_selected": 0,
        "selector": None,
        "selector_pair_actions": 0,
        "selector_pair_metrics": 0,
        "selector_merged_pairs": 0,
        "materialized_pair_tensor": None,
        "materialized_pair_metric": None,
        "materialized_tangent_jacobian": None,
        "mean_missing_norm": 0.0,
        "mean_captured_weight": 0.0,
        "mean_trim_loss": _mean(row.get("truncation") for row in history),
        "mean_accepted_trim_loss": 0.0,
        "mean_cbe_vs_baseline_energy": 0.0,
        "mean_cbe_baseline_allowance": 0.0,
        "sweep_energies": [
            float(np.real(row["energy"]))
            for row in history
            if row.get("energy") is not None
        ],
    }


def _run_one_solver(
    solver,
    model,
    initial,
    *,
    bond_dim,
    expansion_dimension,
    cbe_baseline_guard_fraction,
    max_sweeps,
    tolerance,
    exact_energy,
):
    started = time.perf_counter()
    if solver == "letta_two_site":
        result = letta_two_site_dmrg(
            model.mpo,
            state=initial.letta,
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
        return _letta_two_site_record(
            result, time.perf_counter() - started, initial.fingerprint, exact_energy
        )
    if solver == "mps_two_site":
        standard_mpo = [factor.copy() for factor in model.mpo.factors]
        dmrg = DMRG(
            standard_mpo,
            bond_dim,
            init_guess=[factor.copy() for factor in initial.mps.factors],
            nsweeps=max_sweeps,
            opt="2site",
            not_conv_err=False,
            verbose=0,
            sweep_tol=tolerance,
            davidson_tol=min(1.0e-8, tolerance),
            noise=0.0,
            local_dense_max_dim=4096,
        ).run()
        return _mps_two_site_record(
            dmrg, time.perf_counter() - started, initial.fingerprint, exact_energy
        )

    cbe = solver.startswith("letta_cbe_")
    selector = "shrewd" if solver == "letta_cbe_strict" else "exact"
    result = letta_dmrg(
        model.mpo,
        state=initial.letta,
        options=LETTADMROptions(
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            matrix_free=True,
            use_sparse_mpo=True,
            cbe_enabled=cbe,
            cbe_selector=selector,
            cbe_expansion_dimension=expansion_dimension,
            cbe_baseline_guard_fraction=cbe_baseline_guard_fraction,
        ),
    )
    return _letta_one_site_record(
        result,
        solver,
        time.perf_counter() - started,
        initial.fingerprint,
        exact_energy,
    )


def run_benchmark(
    model_name,
    *,
    dimension,
    size,
    model_parameters=None,
    bond_dim=4,
    expansion_dimension=1,
    cbe_baseline_guard_fraction=0.2,
    max_sweeps=50,
    seed=731,
    tolerance=1.0e-9,
    exact_max_dimension=4096,
    solvers=SOLVERS,
    raise_on_failure=False,
):
    """Run selected solvers from one physical state and return JSON-safe data."""

    bond_dim = int(bond_dim)
    expansion_dimension = int(expansion_dimension)
    cbe_baseline_guard_fraction = float(cbe_baseline_guard_fraction)
    max_sweeps = int(max_sweeps)
    tolerance = float(tolerance)
    exact_max_dimension = int(exact_max_dimension)
    if bond_dim <= 0 or expansion_dimension <= 0 or max_sweeps <= 0:
        raise ValueError("bond_dim, expansion_dimension, and max_sweeps must be positive.")
    if tolerance <= 0.0 or exact_max_dimension < 0:
        raise ValueError("tolerance must be positive and exact_max_dimension nonnegative.")
    if not 0.0 <= cbe_baseline_guard_fraction <= 1.0:
        raise ValueError("cbe_baseline_guard_fraction must be between zero and one.")
    solvers = tuple(solvers)
    unknown = set(solvers) - set(SOLVERS)
    if unknown:
        raise ValueError(f"unknown solver(s): {sorted(unknown)}.")
    model = build_model(
        model_name, dimension, size, **dict(model_parameters or {})
    )
    initial = make_shared_initial_state(model, bond_dim=bond_dim, seed=seed)
    initial_tensor_hash = _hash_arrays(initial.letta.tensors)

    exact_energy = None
    if exact_max_dimension and model.hilbert_dim <= exact_max_dimension:
        exact_energy, _ = exact_ground_state(
            model.mpo.to_dense(max_sites=model.nsites)
        )

    records = []
    failures = {}
    for solver in solvers:
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                records.append(
                    _run_one_solver(
                        solver,
                        model,
                        initial,
                        bond_dim=bond_dim,
                        expansion_dimension=expansion_dimension,
                        cbe_baseline_guard_fraction=cbe_baseline_guard_fraction,
                        max_sweeps=max_sweeps,
                        tolerance=tolerance,
                        exact_energy=exact_energy,
                    )
                )
        except Exception as error:
            if raise_on_failure:
                raise
            failures[solver] = f"{type(error).__name__}: {error}"
        if _hash_arrays(initial.letta.tensors) != initial_tensor_hash:
            raise RuntimeError("a benchmark solver mutated the shared initial state.")

    return {
        "model": model.name,
        "dimension": model.dimension,
        "shape": list(model.lattice_shape),
        "physical_dim": model.physical_dim,
        "nsites": model.nsites,
        "hilbert_dim": model.hilbert_dim,
        "parameters": dict(model.parameters),
        "bond_dim": bond_dim,
        "expansion_dimension": expansion_dimension,
        "cbe_baseline_guard_fraction": cbe_baseline_guard_fraction,
        "max_sweeps": max_sweeps,
        "seed": int(seed),
        "tolerance": tolerance,
        "initial_energy": initial.energy,
        "exact_energy": exact_energy,
        "initial_state_fingerprint": initial.fingerprint,
        "records": records,
        "solver_failures": failures,
        "cost_note": (
            "Exact CBE is a pair-metric oracle. Strict CBE streams sparse-MPO "
            "half contractions, then raises and tangent-projects (H-E*N)psi "
            "in a restricted expanded one-site metric. It solves only that "
            "one-site problem and trims in the one-site LETTA metric; its "
            "pair-action, pair-metric, and merged-pair counters must stay zero."
        ),
    }


def format_table(report):
    """Format the stable human-readable comparison used by every entry point."""

    lines = [
        "solver                 energy             error      seconds  swp  conv  cbe-ok  fallback   missing    captured       trim"
    ]
    for record in report["records"]:
        error = record["energy_error"]
        error_text = "n/a" if error is None else f"{error:.3e}"
        lines.append(
            f"{record['solver']:<22} {record['energy']: .12f}  "
            f"{error_text:>11}  {record['elapsed_seconds']:8.3f}  "
            f"{record['sweeps']:3d}  {str(record['converged']):>5}  "
            f"{record['cbe_accepted']:6d}  {record['cbe_fallbacks']:8d}  "
            f"{record['mean_missing_norm']:8.2e}  "
            f"{record['mean_captured_weight']:10.3f}  "
            f"{record['mean_trim_loss']:9.2e}"
        )
    for solver, error in report["solver_failures"].items():
        lines.append(f"{solver:<22} FAILED: {error}")
    lines.append(report["cost_note"])
    return "\n".join(lines)


__all__ = [
    "SOLVERS",
    "SharedInitialState",
    "format_table",
    "make_shared_initial_state",
    "mps_state_vector",
    "mps_to_letta",
    "run_benchmark",
]

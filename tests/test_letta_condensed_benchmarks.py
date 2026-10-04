"""Contracts for the shared five-solver condensed-model benchmark runner."""

from __future__ import annotations

import json

import numpy as np

from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
    SOLVERS,
    format_table,
    make_shared_initial_state,
    mps_state_vector,
    run_benchmark,
)


def test_mps_embedding_is_the_same_normalized_physical_state():
    model = build_model("ising", "2d", (2, 2), J=0.8, h=1.1)
    initial = make_shared_initial_state(model, bond_dim=2, seed=19)
    mps_vector = mps_state_vector(initial.mps.factors)
    letta_vector = initial.letta.state_vector()
    assert np.isclose(np.vdot(mps_vector, mps_vector), 1.0)
    assert np.allclose(mps_vector, letta_vector, atol=1.0e-12)
    dense = model.mpo.to_dense(max_sites=model.nsites)
    expected = float(np.real(np.vdot(mps_vector, dense @ mps_vector)))
    assert np.isclose(initial.energy, expected, atol=1.0e-12)
    assert np.isclose(initial.letta.expectation(model.mpo), expected, atol=1.0e-12)


def test_per_sweep_cbe_counts_exclude_fallbacks_and_unexpanded_updates():
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import _letta_one_site_record
    from pyqed._letta_one_site_opt.solver import LETTADMRGResult, LETTASiteUpdate, LETTASweep
    from pyqed._letta_one_site_opt.state import LatticeLETTA

    def update(expansion, fallback=False):
        return LETTASiteUpdate(
            site=0, local_energy=-1., energy=-1., metric_rank=1,
            local_dimension=1, residual_norm=0., accepted=True,
            cbe_expansion_dimension=expansion, cbe_fallback=fallback,
        )

    history = tuple(
        LETTASweep(index, direction, -1., 0., 0., 1, updates)
        for index, (direction, updates) in enumerate([
            ("LR", (update(0), update(1, True))),
            ("RL", (update(1), update(2), update(1, True))),
            ("LR", (update(0), update(1))),
        ], start=1)
    )
    result = LETTADMRGResult(
        LatticeLETTA.random((1, 2), physical_dim=2, bond_dim=1, seed=0),
        -1., False, 3, history, "test",
    )
    record = _letta_one_site_record(result, "letta_cbe_strict", 0., "test", None)
    assert record["sweep_cbe_accepted"] == [0, 2, 1]
    assert record["cbe_accepted"] == 3
    json.dumps(record)


def test_tiny_run_returns_all_five_solver_records_and_strict_cost_contract():
    report = run_benchmark(
        "ising",
        dimension="1d",
        size=2,
        model_parameters={"J": 0.7, "h": 1.2},
        bond_dim=1,
        expansion_dimension=1,
        max_sweeps=1,
        seed=5,
        tolerance=1.0e-10,
        exact_max_dimension=64,
    )
    assert tuple(record["solver"] for record in report["records"]) == SOLVERS
    assert report["exact_energy"] is not None
    assert report["initial_state_fingerprint"]
    assert report["hilbert_dim"] == 4
    assert report["cbe_baseline_guard_fraction"] == 0.2
    for record in report["records"]:
        assert np.isfinite(record["energy"])
        assert np.isfinite(record["energy_error"])
        assert record["elapsed_seconds"] >= 0.0
        assert record["sweeps"] >= 1
        assert isinstance(record["converged"], bool)
        assert record["initial_state_fingerprint"] == report["initial_state_fingerprint"]
        assert record["parameter_count"] > 0
        assert record["sweep_definition"]
        assert len(record["sweep_cbe_accepted"]) == len(record["sweep_energies"])
        assert sum(record["sweep_cbe_accepted"]) == record["cbe_accepted"]
        if record["representation"] == "LETTA":
            assert record["final_energy_density_change"] is not None
            assert record["local_hamiltonian_applications"] >= 0
            times = record["sweep_elapsed_seconds"]
            assert len(times) == record["sweeps"]
            assert times == sorted(times) and min(times) > 0.
            assert times[-1] <= record["elapsed_seconds"]

    strict = next(
        record for record in report["records"] if record["solver"] == "letta_cbe_strict"
    )
    assert strict["selector"] == "shrewd"
    assert strict["selector_pair_actions"] == 0
    assert strict["selector_pair_metrics"] == 0
    assert strict["selector_merged_pairs"] == 0
    assert strict["materialized_pair_tensor"] is False
    assert strict["materialized_pair_metric"] is False
    assert strict["materialized_tangent_jacobian"] is False
    assert strict["cbe_selection_seconds"]["routing"] >= 0.
    assert strict["cbe_selection_seconds"]["tangent_projection"] >= 0.
    assert strict["cbe_max_connector_dimension"] > 0
    assert np.isfinite(strict["cbe_max_tangent_relative_residual"])
    assert isinstance(strict["cbe_metric_kinds"], dict)
    assert isinstance(strict["cbe_trim_metric_kinds"], dict)

    assert set(report["solver_failures"]) == set()
    json.dumps(report)
    table = format_table(report)
    assert "dE/site" in table
    assert "swp" in table
    assert "local-Hmv" in table
    for solver in SOLVERS:
        assert solver in table


def test_exact_reference_can_be_disabled_by_hilbert_dimension():
    report = run_benchmark(
        "fermi_hubbard",
        dimension="1d",
        size=2,
        bond_dim=1,
        expansion_dimension=1,
        max_sweeps=1,
        exact_max_dimension=1,
        solvers=("letta_one_site",),
    )
    assert report["exact_energy"] is None
    assert report["records"][0]["energy_error"] is None


def test_cbe_regression_on_two_dimensional_bose_hubbard():
    report = run_benchmark(
        "bose_hubbard",
        dimension="2d",
        size=(2, 2),
        bond_dim=2,
        expansion_dimension=1,
        max_sweeps=2,
        seed=737,
        exact_max_dimension=4096,
        solvers=(
            "letta_one_site",
            "letta_cbe_exact",
            "letta_cbe_strict",
        ),
        raise_on_failure=True,
    )
    energies = {record["solver"]: record["energy"] for record in report["records"]}
    errors = {
        record["solver"]: abs(record["energy_error"])
        for record in report["records"]
    }
    # Before the baseline-candidate guard, exact CBE ended 0.279 above the
    # ordinary one-site result on this deterministic case.  Different accepted
    # CBE steps can still create a different sweep trajectory, so final global
    # dominance is neither expected nor asserted.
    assert energies["letta_cbe_exact"] <= energies["letta_one_site"] + 0.02
    assert errors["letta_cbe_exact"] <= errors["letta_cbe_strict"]

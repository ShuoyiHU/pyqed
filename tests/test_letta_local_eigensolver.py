"""Numerical contracts for the warm-started LETTA local eigensolver."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence

from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_one_site_opt.solver import (
    LETTADMROptions,
    _lowest_matrix_free_eigenpair,
    _optimize_site_matrix_free,
)


@pytest.mark.parametrize("method", ["one_site", "two_site"])
def test_real_initial_vector_does_not_discard_complex_hamiltonian(method):
    from pyqed._letta_two_site_opt.solver import LETTATwoSiteOptions, _lowest_pair_vector
    rng = np.random.default_rng(129)
    size = 24
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    matrix = (matrix + matrix.conj().T) / 2
    initial = rng.normal(size=size)
    metric = BlockDiagonalMetric(size, [np.eye(size)], [np.arange(size)])
    expected = np.linalg.eigvalsh(matrix)[0]
    if method == "one_site":
        energy, vector, _, _ = _lowest_matrix_free_eigenpair(
            lambda x: matrix @ x, metric, 1.e-12, initial_vector=initial)
    else:
        energy, vector, _, _ = _lowest_pair_vector(
            lambda x: matrix @ x, metric, initial,
            LETTATwoSiteOptions(dense_solver_threshold=2))
    assert energy == pytest.approx(expected, abs=1.e-8)
    assert np.linalg.norm(matrix @ vector - energy * vector) < 1.e-7


def test_local_update_reuses_nearly_converged_tensor_and_counts_actions():
    dimension = 128
    diagonal = np.linspace(-2.0, 3.0, dimension)
    metric = BlockDiagonalMetric(dimension, [np.eye(dimension)], [np.arange(dimension)])
    initial = np.zeros(dimension)
    initial[0] = 1.0
    state = SimpleNamespace(tensors=[initial.copy()])
    calls = 0

    def action(vector):
        nonlocal calls
        calls += 1
        return diagonal * vector

    update = _optimize_site_matrix_free(state, 0, action, metric, LETTADMROptions())
    assert update.energy == pytest.approx(-2.0, abs=1.0e-10)
    assert calls < 60  # The old cold-start path needs 82 applications.
    assert update.hamiltonian_applications == calls


@pytest.mark.parametrize("complex_problem", [False, True])
def test_generalized_warm_start_matches_supported_dense_reference(complex_problem):
    rng = np.random.default_rng(110)
    size = 24
    matrix = rng.normal(size=(size, size))
    if complex_problem:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
    matrix = (matrix + matrix.conj().T) / 2
    weights = np.linspace(0.4, 2.0, size)
    weights[-2:] = 0.0
    metric = BlockDiagonalMetric(size, [np.diag(weights)], [np.arange(size)])
    basis, rank = metric.whitening_basis(1.0e-12)
    reduced = basis.conj().T @ matrix @ basis
    values, vectors = np.linalg.eigh(reduced)
    initial = basis @ vectors[:, 0] + 0.01 * rng.normal(size=size)
    energy, vector, actual_rank, _ = _lowest_matrix_free_eigenpair(
        lambda x: matrix @ x, metric, 1.0e-12,
        initial_vector=initial, tolerance=1.0e-10, max_iterations=300,
    )
    assert actual_rank == rank
    assert energy == pytest.approx(values[0], abs=1.0e-8)
    assert np.vdot(vector, metric @ vector) == pytest.approx(1.0, abs=1.0e-10)
    assert np.linalg.norm(basis.conj().T @ (matrix @ vector - energy * (metric @ vector))) < 1.0e-7


def test_excited_eigenvector_initial_guess_does_not_cause_false_convergence():
    size = 64
    diagonal = np.linspace(-2.0, 3.0, size)
    initial = np.zeros(size)
    initial[-1] = 1.0
    energy, _, _, _ = _lowest_matrix_free_eigenpair(
        lambda x: diagonal * x, np.eye(size), 1.0e-12,
        initial_vector=initial, tolerance=1.0e-10, max_iterations=300,
    )
    assert energy == pytest.approx(-2.0, abs=1.0e-8)


def test_iteration_limit_is_honored_and_does_not_mutate_the_state():
    size = 256
    diagonal = np.linspace(-2.0, 3.0, size)
    initial = np.ones(size) / np.sqrt(size)
    state = SimpleNamespace(tensors=[initial.copy()])
    with pytest.raises(ArpackNoConvergence):
        _optimize_site_matrix_free(
            state, 0, lambda x: diagonal * x, np.eye(size),
            LETTADMROptions(eigensolver_max_iterations=1, eigensolver_tolerance=1.0e-14),
        )
    np.testing.assert_array_equal(state.tensors[0], initial)


def test_requested_tolerance_changes_work_and_controls_supported_residual():
    size = 160
    diagonal = np.linspace(-2.0, 3.0, size)
    initial = np.ones(size) / np.sqrt(size)
    counts = []
    for tolerance in (1.0e-3, 1.0e-11):
        calls = 0

        def action(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector

        energy, vector, _, _ = _lowest_matrix_free_eigenpair(
            action, np.eye(size), 1.0e-12, initial_vector=initial,
            tolerance=tolerance, max_iterations=300,
        )
        assert np.linalg.norm(diagonal * vector - energy * vector) < tolerance * 3
        counts.append(calls)
    assert counts[0] < counts[1]


@pytest.mark.parametrize("selector", ["exact", "shrewd"])
def test_cbe_reports_both_candidate_and_baseline_work(selector, monkeypatch):
    import pyqed._letta_one_site_opt.solver as solver
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model

    model = build_model("ising", "2d", (2, 3))
    initial = solver.LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=2, seed=731)
    original = solver._optimize_site_matrix_free
    calls = []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result.hamiltonian_applications)
        return result

    monkeypatch.setattr(solver, "_optimize_site_matrix_free", recorded)
    result = solver.letta_dmrg(
        model.mpo, state=initial,
        options=LETTADMROptions(max_sweeps=1, cbe_enabled=True, cbe_selector=selector),
    )
    updates = result.history[0].updates
    coupled = sum(update.cbe_coupled_hamiltonian_applications for update in updates)
    assert sum(update.hamiltonian_applications for update in updates) == sum(calls) + coupled
    cbe_updates = [update for update in updates if update.cbe_expansion_dimension]
    assert any(update.cbe_expanded_energy is not None for update in cbe_updates)
    for update in cbe_updates:
        assert update.cbe_timings["selection"] >= 0.0
        assert update.cbe_timings["baseline"] >= 0.0
        if update.cbe_expanded_energy is not None:
            assert update.cbe_timings["expanded_solve"] >= 0.0
            assert update.cbe_timings["trim"] >= 0.0


def test_four_method_convergence_entry_point_reports_sweeps_and_energy_change(capsys):
    from pyqed._letta_one_site_opt.benchmarks.cbe_convergence import run_comparison, _print_table

    report = run_comparison(shape=(2, 2), bond_dim=1, max_sweeps=1)
    for record in report["records"]:
        assert record["sweeps"] == 1
        assert isinstance(record["converged"], bool)
        assert np.isfinite(record["final_energy_density_change"])
        assert len(record["sweep_elapsed_seconds"]) == record["sweeps"]
        assert 0 < record["sweep_elapsed_seconds"][-1] <= record["elapsed_seconds"]
    _print_table(report)
    table = capsys.readouterr().out
    assert "swp" in table and "dE/site" in table

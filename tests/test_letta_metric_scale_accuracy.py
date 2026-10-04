"""Coordinate rescaling must not remove physical variational directions."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric, DiagonalMetric
from pyqed._letta_one_site_opt.solver import (
    _lowest_generalized_eigenpair, _lowest_matrix_free_eigenpair,
)
from pyqed._letta_two_site_opt.solver import LETTATwoSiteOptions, _lowest_pair_vector


@pytest.mark.parametrize("solver", ["dense", "iterative", "pair"])
@pytest.mark.parametrize("representation", ["array", "block", "diagonal"])
@pytest.mark.parametrize("correlated", [False, True])
def test_local_minimum_is_invariant_under_coordinate_scaling(
        solver, representation, correlated):
    if correlated and representation == "diagonal":
        pytest.skip("a diagonal representation cannot encode correlations")
    physical_h = np.array([[1., .2, .1], [.2, -2., .3], [.1, .3, .5]])
    frame = np.eye(3)
    if correlated:
        frame[0, 1], frame[1, 2] = .4, -.3
    # All three physical directions remain independent, despite 40 decades
    # between their Gram weights. This is only a change of coordinates.
    frame = frame * np.array([1e10, 1., 1e-10])
    norm = frame.T @ frame
    hamiltonian = frame.T @ physical_h @ frame
    if representation == "block":
        metric = BlockDiagonalMetric(3, [norm], [np.arange(3)])
    elif representation == "diagonal":
        metric = DiagonalMetric(np.diag(norm))
    else:
        metric = norm
    initial = np.linalg.solve(frame, np.ones(3) / np.sqrt(3))
    if solver == "dense":
        energy, vector, rank, _ = _lowest_generalized_eigenpair(hamiltonian, metric, 1e-10)
    elif solver == "iterative":
        energy, vector, rank, _ = _lowest_matrix_free_eigenpair(
            lambda x: hamiltonian @ x, metric, 1e-10, initial_vector=initial)
    else:
        if representation == "array":
            metric = BlockDiagonalMetric(3, [norm], [np.arange(3)])
        energy, vector, rank, _ = _lowest_pair_vector(
            lambda x: hamiltonian @ x, metric, initial, LETTATwoSiteOptions())
    expected = np.linalg.eigvalsh(physical_h)[0]
    assert rank == 3
    assert energy == pytest.approx(expected, abs=2e-12)
    physical = frame @ vector
    assert np.linalg.norm(physical_h @ physical - expected * physical) < 2e-11
    assert np.vdot(physical, physical) == pytest.approx(1., abs=2e-12)


def test_certified_numerical_null_support_survives_restriction_and_pair_whitening():
    from pyqed._letta_two_site_opt.solver import _BlockMetricWhitening
    metric = DiagonalMetric([1., 1e-32, 2.], support=[True, False, True])
    restricted = metric.restrict(np.array([0, 1]))
    retained, _ = restricted.coordinate_whitening(1e-10)
    np.testing.assert_array_equal(retained, [0])
    whitening = _BlockMetricWhitening(restricted, 1e-10)
    assert whitening.rank == 1
    np.testing.assert_array_equal(whitening.to_full(np.ones(1)), [1., 0.])


def test_compression_metric_counts_small_weight_large_coefficient_directions():
    from pyqed._letta_two_site_opt.truncation import _MetricSquareRoot
    weights = np.array([1e20, 1., 1e-20])
    metric = BlockDiagonalMetric(3, [np.diag(weights)], [np.arange(3)])
    error = 1 / np.sqrt(weights)
    root = _MetricSquareRoot(metric, 1e-10)
    weighted = root.apply(error)
    assert np.vdot(weighted, weighted) == pytest.approx(3., abs=1e-12)

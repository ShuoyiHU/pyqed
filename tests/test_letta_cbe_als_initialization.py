import numpy as np
import pytest
from pyqed._letta_one_site_opt.cbe import _metric_low_rank_factorization
from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric


def test_seeded_als_reduces_loss_without_mutating_seed():
    target = np.diag([3., 1.]).astype(complex)
    metric = BlockDiagonalMetric(4, [np.eye(4)], [np.arange(4)])
    left = np.array([[1.], [.2j]])
    right = np.array([[2., .1j]])
    before = left.copy(), right.copy()
    initial_loss = np.linalg.norm(target - left @ right)**2
    a, b, loss, iterations = _metric_low_rank_factorization(
        target, metric, 1, tolerance=1.e-12, max_iterations=40,
        metric_tolerance=1.e-12, initial_factors=(left, right))
    assert loss < initial_loss
    assert loss == pytest.approx(1., abs=1.e-10)
    np.testing.assert_allclose(np.linalg.norm(target-a@b)**2, loss)
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    assert iterations > 0

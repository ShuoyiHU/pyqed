"""Regression tests for the dense MPS local Hamiltonian wrapper."""

import numpy as np
from scipy.sparse.linalg import eigsh

from pyqed.mps.mps import HamiltonianMultiply


def test_dense_local_hamiltonian_is_a_fully_initialized_linear_operator():
    left = np.ones((1, 1, 1))
    right = np.ones((1, 1, 1))
    local = np.diag([3.0, -2.0, 0.5, 1.0]).reshape(1, 1, 4, 4)
    operator = HamiltonianMultiply(left, local, right)
    values = eigsh(operator, k=1, which="SA", return_eigenvectors=False)
    assert np.allclose(values, [-2.0])

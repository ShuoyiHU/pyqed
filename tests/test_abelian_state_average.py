import numpy as np

from pyqed.mps.abelian_direct import (
    abelian_state_averaged_two_site_svd_from_permuted_data,
)
from pyqed.mps.mps import _state_average_energy_converged


def test_state_averaged_svd_accepts_numpy_weights():
    data_list = [
        {(0, 0, 0, 0): np.ones((1, 1, 1, 1))},
        {(0, 0, 0, 0): 2.0 * np.ones((1, 1, 1, 1))},
        {(0, 0, 0, 0): 3.0 * np.ones((1, 1, 1, 1))},
    ]

    result = abelian_state_averaged_two_site_svd_from_permuted_data(
        data_list,
        np.full(3, 1.0 / 3.0),
        "right",
        m_max=1,
    )

    assert result.kept_states == 1


def test_state_average_convergence_compares_matching_sweep_directions():
    previous = {}
    tol = 1.0e-8

    assert not _state_average_energy_converged(
        previous, "lr", -3.3620002545320755, tol
    )
    assert not _state_average_energy_converged(
        previous, "rl", -3.3620000590965800, tol
    )
    assert _state_average_energy_converged(
        previous, "lr", -3.3620002545643530, tol
    )

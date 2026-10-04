"""A gauge change must preserve small physical rows beside large null rows."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_one_site_opt.solver import _shift_virtual_gauge


def crossing_null_rows(direction, complex_values):
    rng = np.random.default_rng(12)
    tensors = [rng.normal(size=shape) for shape in
               ((1, 2, 2, 2), (2, 2, 2), (2, 2, 1))]
    if complex_values:
        tensors = [a + 1j * rng.normal(size=a.shape) for a in tensors]
    # The last tensor kills physical value 1 on the crossing dependency.
    # Large entries in the first tensor on that value are physically null.
    tensors[0][:, :, 1, :] *= 1e20
    tensors[-1][:, 1, :] = 0
    neighborhoods = ((0, 2), (1,), (2,))
    if direction == 'rl':
        tensors = [a.transpose((-1,) + tuple(range(1, a.ndim-1)) + (0,))
                   for a in reversed(tensors)]
        neighborhoods = ((0,), (1,), (2, 0))
    state = LatticeLETTA((1, 3), 2, tensors, neighborhoods=neighborhoods)
    # Retain the gauge under test instead of the constructor's global scaling.
    state.tensors = [a.copy() for a in tensors]
    state.tensors[0] /= np.linalg.norm(state.state_vector())
    return state


@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('mode', ['qr', 'frontier'])
def test_qr_gauge_preserves_physical_rows_with_twenty_decades_of_scale(
        direction, complex_values, mode):
    state = crossing_null_rows(direction, complex_values)
    before = state.state_vector()
    shapes = [a.shape for a in state.tensors]
    site = 0 if direction == 'lr' else 2
    _shift_virtual_gauge(state, site, direction, mode)
    np.testing.assert_allclose(state.state_vector(), before, atol=2e-13, rtol=2e-13)
    assert [a.shape for a in state.tensors] == shapes
    assert all(np.all(np.isfinite(a)) for a in state.tensors)

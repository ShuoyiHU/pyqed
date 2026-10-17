import numpy as np
import pytest
from pyqed._letta_one_site_opt.frontier_contraction import ScatterBlock, contract_block


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('right_order', ['C', 'F'])
def test_compact_right_gauge_omits_empty_rows_exactly(complex_values, right_order, monkeypatch):
    rng = np.random.default_rng(851)
    shape = (384, 2, 300)
    rows = np.arange(0, shape[0]*shape[1], 3)
    indices = (rows[:, None]*shape[-1]+np.array([1, 72, 299])).ravel()
    values = rng.normal(size=len(indices))
    right = rng.normal(size=(23, shape[-1]))
    if complex_values:
        values = values+1j*rng.normal(size=len(values))
        right = right+1j*rng.normal(size=right.shape)
    right = np.array(right, order=right_order)
    block = ScatterBlock(shape, indices, values)
    expected = np.einsum('kr,lpr->lpk', right, np.asarray(block))
    def forbidden(*args, **kwargs):
        raise AssertionError('materialized the entire compact gauge block')
    monkeypatch.setattr(ScatterBlock, '__array__', forbidden)
    np.testing.assert_array_equal(contract_block(block, right=right), expected)


def test_nonfinite_right_gauge_keeps_dense_semantics():
    shape = (384, 2, 300)
    block = ScatterBlock(shape, np.array([0]), np.array([1.]))
    right = np.ones((2, 300)); right[0, 1] = np.inf
    with np.errstate(invalid='ignore'):
        expected = np.einsum('kr,lpr->lpk', right, np.asarray(block))
        actual = contract_block(block, right=right)
    np.testing.assert_array_equal(actual, expected)

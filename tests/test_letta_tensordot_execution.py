"""Shape caching must preserve NumPy's arithmetic and array layout."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import numpy_contractions as execution


CASES = [
    ((2, 3), (3, 4), ((1,), (0,))),
    ((2, 3, 4), (4, 3, 5), ((2, 1), (0, 1))),
    ((2, 3, 4), (4, 3, 5), ((1, 2), (1, 0))),
    ((3,), (3,), ((0,), (0,))),
    ((), (2, 3), ((), ())),
    ((2, 0), (0, 3), ((1,), (0,))),
    ((0, 2), (2, 3), ((1,), (0,))),
    ((1, 2, 3), (3, 2, 1), ((0, 2), (2, 0))),
    ((2,), (3,), ((), ())),
    ((2, 3), (3, 4), ((-1,), (0,))),
]


@pytest.mark.parametrize("shape_a,shape_b,axes", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["c", "fortran", "reversed"])
def test_cached_tensordot_matches_numpy_bits_and_strides(shape_a, shape_b, axes, dtype, layout):
    rng = np.random.default_rng(1732)
    operands = []
    for shape in (shape_a, shape_b):
        value = np.asarray(rng.normal(size=shape))
        if np.issubdtype(dtype, np.complexfloating):
            value = value + 1j * rng.normal(size=shape)
        value = value.astype(dtype)
        if layout == "fortran" and value.ndim:
            value = np.asfortranarray(value)
        elif layout == "reversed" and value.ndim:
            value = value[tuple(slice(None, None, -1) for _ in shape)]
        operands.append(value)
    expected = np.tensordot(*operands, axes=axes)
    actual = execution._cached_tensordot(*operands, axes=axes)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert actual.strides == expected.strides
    # Reuse the cached shapes with different operand dtypes and values.
    operands[0] = operands[0].astype(np.complex128) * (0.5 + 0.7j)
    expected = np.tensordot(*operands, axes=axes)
    actual = execution._cached_tensordot(*operands, axes=axes)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert actual.strides == expected.strides


@pytest.mark.parametrize("axes", [((0,), (0,)), ((1, 1), (0, 0)), ((2,), (0,))])
def test_invalid_axes_keep_numpy_error(axes):
    a, b = np.ones((2, 3)), np.ones((3, 4))
    with pytest.raises((ValueError, IndexError)) as reference:
        np.tensordot(a, b, axes=axes)
    with pytest.raises(type(reference.value)):
        execution._cached_tensordot(a, b, axes=axes)

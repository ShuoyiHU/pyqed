"""Exact values and layouts for planned pair contractions."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_two_site_opt import LETTAPairLayout


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("storage", ["C", "F", "reversed"])
@pytest.mark.parametrize("bond", [1, 2, 4])
def test_pair_operations_preserve_numpy_values_and_strides(dtype, storage, bond):
    ties = ((0, 3, 1), (1, 0, 3), (2, 0), (3, 1, 2))
    state = LatticeLETTA.random((2, 2), bond_dim=bond, neighborhoods=ties,
                                real=not np.issubdtype(dtype, np.complexfloating), seed=19373)
    rng = np.random.default_rng(8091)

    def stored(array):
        array = np.array(array, dtype=dtype, order="F" if storage == "F" else "C")
        return array[:, ::-1, ...] if storage == "reversed" else array

    for site in range(3):
        layout = LETTAPairLayout.from_state(state, site)
        left, right = stored(state.tensors[site]), stored(state.tensors[site + 1])
        gradient = rng.normal(size=layout.merged_shape)
        if np.issubdtype(dtype, np.complexfloating):
            gradient = gradient + 1j * rng.normal(size=gradient.shape)
        gradient = stored(gradient)
        before = [x.copy() for x in (left, right, gradient)]
        equations = layout._contraction_equations
        results = [
            (layout.merge(left, right), equations[0], left, right),
            (layout.left_adjoint(gradient, right), equations[1], gradient, right.conj()),
            (layout.right_adjoint(left, gradient), equations[2], left.conj(), gradient),
        ]
        for actual, equation, first, second in results:
            expected = np.einsum(equation, first, second, optimize=True)
            np.testing.assert_array_equal(actual, expected, strict=True)
            assert actual.strides == expected.strides
        for actual, expected in zip((left, right, gradient), before):
            np.testing.assert_array_equal(actual, expected)
        # Cache keys must follow the current internal bond, not the original layout.
        left, right = left[..., :1], right[:1]
        actual = layout.merge(left, right)
        expected = np.einsum(equations[0], left, right, optimize=True)
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert actual.strides == expected.strides

"""Cached equations preserve integer-label NumPy contractions exactly."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_two_site_opt import LETTAPairLayout


@pytest.mark.parametrize("site", [0, 1, 2])
@pytest.mark.parametrize("real", [True, False])
@pytest.mark.parametrize("bond", [1, 3])
def test_pair_equations_match_integer_labels(site, real, bond):
    # Multiple shared indices occur in different orders on the two factors.
    ties = ((0, 3, 1), (1, 0, 3), (2, 0), (3, 1, 2))
    state = LatticeLETTA.random((2, 2), bond_dim=bond, neighborhoods=ties, real=real, seed=8901)
    layout = LETTAPairLayout.from_state(state, site)
    left = state.tensors[site][:, ::-1, ...]
    right = state.tensors[site + 1][:, ::-1, ...]
    rng = np.random.default_rng(993)
    gradient = rng.normal(size=layout.merged_shape)
    if not real:
        gradient = gradient + 1j * rng.normal(size=gradient.shape)
    gradient = gradient[:, ::-1, ...]
    al, bl, ml = layout._contraction_labels()
    cases = [
        (layout.merge(left, right), (left, al, right, bl, ml)),
        (layout.left_adjoint(gradient, right), (gradient, ml, right.conj(), bl, al)),
        (layout.right_adjoint(left, gradient), (left.conj(), al, gradient, ml, bl)),
    ]
    for actual, operands in cases:
        expected = np.einsum(*operands, optimize=True)
        np.testing.assert_array_equal(actual, expected, strict=True)
        assert actual.strides == expected.strides

    # The same layout also accepts a different internal rank after splitting;
    # cached equations must not bake in the original bond dimension.
    left = left[..., :1]
    right = right[:1]
    np.testing.assert_array_equal(
        layout.merge(left, right), np.einsum(left, al, right, bl, ml, optimize=True),
        strict=True,
    )

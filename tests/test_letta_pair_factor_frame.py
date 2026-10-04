"""Batched frames must preserve the original individual basis-vector merges."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_two_site_opt import LETTAPairLayout
from pyqed._letta_two_site_opt.energy_refinement import _factor_frame


@pytest.mark.parametrize("shape,site", [((4, 1), 1), ((2, 3), 0), ((2, 3), 2), ((2, 2, 2), 2)])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("restricted", [False, True])
def test_batched_factor_frame_matches_individual_merges(shape, site, complex_values,
                                                       side, restricted):
    state = LatticeLETTA.random(shape, bond_dim=2, seed=897)
    layout = LETTAPairLayout.from_state(state, site)
    rng = np.random.default_rng(691)
    tensors = []
    for tensor_shape in (layout.left_shape, layout.right_shape):
        tensor = rng.normal(size=tensor_shape)
        if complex_values:
            tensor = tensor + 1j * rng.normal(size=tensor_shape)
        # Exercise non-contiguous physical-axis slices.
        tensors.append(tensor[:, ::-1, ...])
    left, right = tensors
    variable = left if side == "left" else right
    selected = np.array([variable.size - 1, 0, variable.size // 2]) if restricted else None
    frame, indices = _factor_frame(layout, left, right, side, selected)
    expected = []
    for index in indices:
        basis = np.zeros_like(variable)
        basis.flat[index] = 1
        merged = layout.merge(basis, right) if side == "left" else layout.merge(left, basis)
        expected.append(merged.reshape(-1))
    np.testing.assert_array_equal(frame, np.column_stack(expected), strict=True)
    assert frame.flags.c_contiguous

    coefficients = rng.normal(size=len(indices))
    full = np.zeros(variable.size, dtype=variable.dtype)
    full[indices] = coefficients
    merged = (layout.merge(full.reshape(variable.shape), right) if side == "left"
              else layout.merge(left, full.reshape(variable.shape)))
    np.testing.assert_allclose(frame @ coefficients, merged.reshape(-1), atol=2e-14, rtol=2e-14)

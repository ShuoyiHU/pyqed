import importlib.util
from pathlib import Path

import numpy as np


SCAN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "_local_tests"
    / "_yujuan_tddmrg"
    / "_July21th_H4_sadmrg_pec"
    / "_h4_sadmrg_scan.py"
)


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("h4_sadmrg_scan", SCAN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_15_point_scan_grid_matches_reference_indices_24_through_38():
    scan = _load_scan_module()
    scan_grid = scan.interior_dyadic_grid((-0.5, 0.5), 15)
    reference_grid = scan.interior_dyadic_grid((-2.0, 2.0), 63)

    indices = scan.match_shared_grid_indices(scan_grid, reference_grid)

    np.testing.assert_allclose(scan_grid, np.arange(-0.4375, 0.5, 0.0625))
    np.testing.assert_array_equal(indices, np.arange(24, 39))


def test_reference_subset_keeps_first_three_states_and_both_scan_axes():
    scan = _load_scan_module()
    reference = np.arange(4 * 63 * 63).reshape(4, 63, 63)
    indices = np.arange(24, 39)

    subset = scan.extract_reference_subset(reference, indices, indices, nstates=3)

    assert subset.shape == (3, 15, 15)
    np.testing.assert_array_equal(subset, reference[:3, indices][:, :, indices])


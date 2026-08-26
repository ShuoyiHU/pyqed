import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "_local_tests"
    / "_yujuan_tddmrg"
    / "_July21th_H4_sadmrg_pec"
    / "plot_full_gto_fci_pes.py"
)


def _load_plotter():
    spec = importlib.util.spec_from_file_location("plot_full_gto_fci_pes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_grid_is_the_63_interior_points_of_minus2_to_2():
    plotter = _load_plotter()

    grid = plotter.reference_grid()

    assert grid.shape == (63,)
    np.testing.assert_allclose(grid[[0, 31, -1]], [-1.9375, 0.0, 1.9375])
    np.testing.assert_allclose(np.diff(grid), 0.0625)


def test_load_reference_requires_four_63_by_63_surfaces(tmp_path):
    plotter = _load_plotter()
    good = np.arange(4 * 63 * 63, dtype=float).reshape(4, 63, 63)
    good_path = tmp_path / "good.npy"
    np.save(good_path, good)

    grid, loaded = plotter.load_reference(good_path)

    np.testing.assert_array_equal(loaded, good)
    np.testing.assert_allclose(grid, plotter.reference_grid())

    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, good[:, :-1, :])
    with pytest.raises(ValueError, match=r"\(4, 63, 63\)"):
        plotter.load_reference(bad_path)


def test_central_cuts_preserve_q1_q2_axis_meaning():
    plotter = _load_plotter()
    values = np.empty((4, 63, 63), dtype=float)
    for root in range(4):
        for iq1 in range(63):
            for iq2 in range(63):
                values[root, iq1, iq2] = 10000 * root + 100 * iq1 + iq2

    cuts = plotter.central_cuts(values)

    # q2=0 fixes the last array axis and varies iq1.
    np.testing.assert_array_equal(cuts["q2_zero"], values[:, :, 31])
    # q1=0 fixes the first surface axis and varies iq2.
    np.testing.assert_array_equal(cuts["q1_zero"], values[:, 31, :])


def test_summary_reports_physical_coordinates_without_transposing_axes():
    plotter = _load_plotter()
    grid = plotter.reference_grid()
    values = np.ones((4, 63, 63), dtype=float)
    values[0, 4, 52] = -3.0

    summary = plotter.reference_summary(grid, values)

    root0 = summary["roots"][0]
    assert root0["minimum_indices_iq1_iq2"] == [4, 52]
    np.testing.assert_allclose(
        root0["minimum_coordinates_q1_q2"], [grid[4], grid[52]]
    )
    assert root0["energy_at_q1_0_q2_0_hartree"] == 1.0

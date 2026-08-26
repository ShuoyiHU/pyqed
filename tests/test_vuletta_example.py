import numpy as np

from pyqed._vuletta.examples.tfim_comparison import compare_tfim_methods


def test_vuletta_tfim_example_matches_analytical_reference():
    rows = compare_tfim_methods(
        letta_bond_dimensions=(1,),
        mps_bond_dimensions=(),
        seed=3,
        tolerance=1.0e-8,
        max_iterations=100,
    )
    exact, letta = rows

    np.testing.assert_allclose(exact.energy_density, -1.6719262215361948)
    np.testing.assert_allclose(letta.energy_density, -5.0 / 3.0, atol=1.0e-9)
    np.testing.assert_allclose(
        letta.transverse_magnetization,
        8.0 / 9.0,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(letta.zz_correlation, 1.0 / 3.0, atol=2.0e-7)
    assert letta.converged
    assert letta.tensor_entry_count == 4
    assert letta.transfer_bond_dim == 2


def test_bond_dimension_continuation_avoids_seed_three_d3_basin():
    rows = compare_tfim_methods(
        letta_bond_dimensions=(1, 2, 3),
        mps_bond_dimensions=(),
        seed=3,
        tolerance=1.0e-8,
        max_iterations=200,
    )
    d3 = rows[-1]

    assert d3.converged
    assert abs(d3.energy_density + 1.6719262215361948) < 1.0e-7

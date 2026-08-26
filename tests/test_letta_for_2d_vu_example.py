import numpy as np

from pyqed._letta_for_2d_vu.examples.tfim_plane import run_tfim_plane


def test_plane_example_runs_without_a_vumps_reference():
    rows = run_tfim_plane(
        bond_dimensions=(1,),
        window_sizes=(3,),
        boundary_bond_dim=12,
        max_iterations=2,
        function_tolerance=1.0e-12,
        gradient_method="autodiff",
        seed=4,
    )

    assert len(rows) == 1
    assert rows[0].method == "VULETTA-2D"
    assert rows[0].gradient_method == "autodiff"
    assert rows[0].bond_dimension == 1
    assert np.isfinite(rows[0].energy_density)
    assert np.isfinite(rows[0].transverse_magnetization)
    assert np.isfinite(rows[0].horizontal_zz)
    assert np.isfinite(rows[0].vertical_zz)


def test_plane_example_reports_environment_and_optimizer_convergence_separately():
    row = run_tfim_plane(
        coupling=0.0,
        field=1.0,
        bond_dimensions=(1,),
        window_sizes=(3,),
        boundary_bond_dim=8,
        max_iterations=5,
        function_tolerance=1.0e-12,
        gradient_tolerance=1.0e-5,
        gradient_method="autodiff",
        seed=3,
    )[0]

    assert row.solver_converged
    assert not row.environment_converged
    assert not row.overall_converged
    assert np.isinf(row.environment_window_change)
    assert row.environment_boundary_change <= 1.0e-6

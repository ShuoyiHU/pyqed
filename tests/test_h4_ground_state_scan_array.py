import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "_local_tests"
    / "_yujuan_tddmrg"
    / "_July22nd_H4_ground_state_dmrg"
)
if not RUN_DIR.exists():
    RUN_DIR = Path(__file__).resolve().parents[1]
WORKER = RUN_DIR / "h4_gdvr_ground_state_array.py"
PLOTTER = RUN_DIR / "plot_h4_ground_state_scan.py"
SUBMIT = RUN_DIR / "submit_h4_ground_state_arrays.sh"
PLOT_SH = RUN_DIR / "plot_h4_ground_state_scan.sh"


def _load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_maps_all_225_interior_geometries():
    worker = _load(WORKER, "h4_gdvr_ground_state_array")

    points = [worker.task_point(index) for index in range(225)]

    assert points[0][:4] == (0, 0, -0.4375, -0.4375)
    assert points[112][2:4] == (0.0, 0.0)
    assert points[-1][:4] == (14, 14, 0.4375, 0.4375)
    assert len({(point[2], point[3]) for point in points}) == 225


def test_worker_configuration_is_single_state_and_grid_specific():
    worker = _load(WORKER, "h4_gdvr_ground_state_array_config")
    args = SimpleNamespace(
        lz=15.0,
        nz=63,
        transverse_orbitals=1,
        bond_dim=20,
        max_sweeps=20,
        spin_penalty=0.2,
        spin_tol=1.0e-6,
        sweep_tol=1.0e-8,
        davidson_tol=1.0e-9,
        davidson_max_iter=100,
    )
    iq1, iq2, q1, q2, coordinates = worker.task_point(112)

    config = worker.task_config(args, 112, iq1, iq2, q1, q2, coordinates)

    assert config["Lz"] == 15.0
    assert config["Nz"] == 63
    assert config["nstates"] == 1
    assert config["state_average"] is False
    assert config["bond_dimension"] == 20


def _write_result(root, nz, task_index, *, converged=True):
    iq1, iq2 = divmod(task_index, 15)
    grid = np.linspace(-0.5, 0.5, 16, endpoint=False)[1:]
    task_dir = root / f"Nz{nz}" / f"task_{task_index:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_index": task_index,
        "iq1": iq1,
        "iq2": iq2,
        "q1": float(grid[iq1]),
        "q2": float(grid[iq2]),
        "Lz": 15.0,
        "Nz": nz,
        "physical_energy_hartree": -2.0 + 1.0e-3 * task_index + 1.0e-5 * nz,
        "objective_energy_hartree": -1.99 + 1.0e-3 * task_index,
        "s2": 1.0e-8 * (task_index + 1),
        "dmrg_converged": converged,
    }
    (task_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_plot_collector_builds_separate_15_by_15_surfaces(tmp_path):
    plotter = _load(PLOTTER, "plot_h4_ground_state_scan")
    for nz in (31, 63):
        for task_index in range(225):
            _write_result(tmp_path, nz, task_index)

    scan31 = plotter.collect_scan(tmp_path, 31)
    scan63 = plotter.collect_scan(tmp_path, 63)

    assert scan31["energy"].shape == (15, 15)
    assert scan63["energy"].shape == (15, 15)
    assert np.isfinite(scan31["energy"]).all()
    assert np.isfinite(scan63["energy"]).all()
    assert scan31["energy"][7, 7] != scan63["energy"][7, 7]
    np.testing.assert_allclose(scan31["q1"], scan63["q1"])
    np.testing.assert_allclose(scan31["q2"], scan63["q2"])


def test_plot_collector_rejects_missing_or_unconverged_points(tmp_path):
    plotter = _load(PLOTTER, "plot_h4_ground_state_scan_strict")
    for task_index in range(224):
        _write_result(tmp_path, 31, task_index)

    try:
        plotter.collect_scan(tmp_path, 31)
    except RuntimeError as exc:
        assert "225 converged results" in str(exc)
    else:
        raise AssertionError("an incomplete scan was accepted")

    _write_result(tmp_path, 31, 224, converged=False)
    try:
        plotter.collect_scan(tmp_path, 31)
    except RuntimeError as exc:
        assert "225 converged results" in str(exc)
    else:
        raise AssertionError("an unconverged scan was accepted")


def test_plot_collector_preview_accepts_latest_unconverged_results(tmp_path):
    plotter = _load(PLOTTER, "plot_h4_ground_state_scan_preview")
    for task_index in range(225):
        _write_result(tmp_path, 31, task_index, converged=task_index < 45)

    scan = plotter.collect_scan(tmp_path, 31, allow_unconverged=True)

    assert scan["energy"].shape == (15, 15)
    assert np.isfinite(scan["energy"]).all()
    assert int(np.count_nonzero(scan["converged"])) == 45


def test_reference_loader_selects_ground_state_center_15_by_15(tmp_path):
    plotter = _load(PLOTTER, "plot_h4_ground_state_reference")
    source = np.arange(4 * 63 * 63, dtype=float).reshape(4, 63, 63)
    path = tmp_path / "reference.npy"
    np.save(path, source)

    reference = plotter.load_fci_reference(path, expected_grid=np.linspace(-0.5, 0.5, 16, endpoint=False)[1:])

    assert reference["source_shape"] == (4, 63, 63)
    assert reference["root"] == 0
    np.testing.assert_array_equal(reference["energy"], source[0, 24:39, 24:39])
    np.testing.assert_allclose(reference["q1"], np.linspace(-0.5, 0.5, 16, endpoint=False)[1:])
    np.testing.assert_allclose(reference["q2"], reference["q1"])


def test_reference_comparison_reports_absolute_and_shape_errors():
    plotter = _load(PLOTTER, "plot_h4_ground_state_reference_metrics")
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    calculated = reference + np.array([[0.1, -0.2], [0.3, -0.4]])

    metrics = plotter.comparison_metrics(calculated, reference)

    np.testing.assert_allclose(metrics["mean_signed_error_millihartree"], -50.0)
    np.testing.assert_allclose(metrics["mean_absolute_error_millihartree"], 250.0)
    np.testing.assert_allclose(metrics["root_mean_square_error_millihartree"], np.sqrt(0.075) * 1000.0)
    np.testing.assert_allclose(metrics["max_absolute_error_millihartree"], 400.0)
    assert metrics["relative_surface_rmse_ev"] > 0.0


def test_submission_scripts_launch_two_arrays_then_dependent_plot():
    submit_text = SUBMIT.read_text(encoding="utf-8")
    plot_text = PLOT_SH.read_text(encoding="utf-8")

    assert "GRID_NZ=31" in submit_text
    assert "GRID_NZ=63" in submit_text
    assert submit_text.count("--array=0-224") == 2
    assert "--array=0-224%" not in submit_text
    assert "--time=" not in submit_text
    assert "--dependency=afterok:${JOB31}:${JOB63}" in submit_text
    assert "plot_h4_ground_state_scan.sh" in submit_text
    assert "plot_h4_ground_state_scan.py" in plot_text
    assert '"${1:-}" = "preview"' in plot_text
    assert "--allow-unconverged" in plot_text
    assert "--reference" in plot_text

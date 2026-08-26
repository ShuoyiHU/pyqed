import json

import numpy as np


def test_tiny_symmetry_benchmark_reports_accuracy_and_resource_ratios():
    from pyqed._letta_two_site_opt.benchmarks.ising_symmetry import (
        SymmetryBenchmarkCase,
        run_symmetry_benchmarks,
    )

    report = run_symmetry_benchmarks(
        cases=(
            SymmetryBenchmarkCase("2d", (1, 3)),
            SymmetryBenchmarkCase("3d", (1, 1, 3)),
        ),
        bond_dim=2,
        max_sweeps=3,
        repeats=1,
        seed=17,
    )

    assert report["schema_version"] == 1
    assert report["settings"]["one_site_matrix_free"] is True
    assert len(report["cases"]) == 2
    json.dumps(report)
    for case in report["cases"]:
        assert case["symmetry"]["name"] == "ising-z2"
        for solver in ("one_site", "two_site"):
            comparison = case["solvers"][solver]
            dense = comparison["without_symmetry"]
            symmetric = comparison["with_symmetry"]
            np.testing.assert_allclose(
                dense["energy_median"], symmetric["energy_median"], atol=1.0e-8
            )
            assert symmetric["symmetry_violation_max"] < 1.0e-13
            assert symmetric["parameter_count"] < dense["parameter_count"]
            assert (
                symmetric["max_local_dimension"]
                < dense["max_local_dimension"]
            )
            assert comparison["ratios"]["parameter_count"] < 1.0
            assert comparison["ratios"]["max_local_dimension"] < 1.0
            assert comparison["ratios"]["dense_matrix_bytes"] < 1.0
            assert comparison["energy_difference"] < 1.0e-8


def test_symmetry_benchmark_cli_writes_json(tmp_path, capsys):
    from pyqed._letta_two_site_opt.benchmarks.ising_symmetry import main

    output = tmp_path / "symmetry.json"
    status = main(
        [
            "--dimension",
            "2d",
            "--shape-2d",
            "1x3",
            "--bond-dim",
            "2",
            "--max-sweeps",
            "2",
            "--repeats",
            "1",
            "--json-output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text())
    assert payload["cases"][0]["shape"] == [1, 3]
    rendered = capsys.readouterr().out
    assert "with_symmetry" in rendered
    assert "dimension_ratio" in rendered
    assert str(output) in rendered

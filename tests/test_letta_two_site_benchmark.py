import json

import numpy as np


def test_tiny_2d_and_3d_benchmark_records_fairness_metrics():
    from pyqed._letta_two_site_opt.benchmarks.ising_convergence import (
        BenchmarkCase,
        run_ising_benchmarks,
    )

    report = run_ising_benchmarks(
        cases=(
            BenchmarkCase("2d", (1, 2)),
            BenchmarkCase("3d", (1, 1, 2)),
        ),
        bond_dim=1,
        max_sweeps=1,
        tolerance=1.0e-8,
        repeats=2,
        seed=7,
        exact_max_sites=2,
        warmup=False,
    )

    assert report["schema_version"] == 2
    assert len(report["cases"]) == 2
    json.dumps(report)

    for case in report["cases"]:
        assert case["initial_state_id"] == "shared-random-state"
        assert case["exact_energy"] is not None
        assert case["exact_elapsed_seconds"] >= 0.0
        assert set(case["summaries"]) == {
            "one_site",
            "two_site_metric_als",
            "two_site_energy_refined",
        }
        assert [run["solver"] for run in case["runs"]] == [
            "one_site",
            "two_site_metric_als",
            "two_site_energy_refined",
            "two_site_metric_als",
            "two_site_energy_refined",
            "one_site",
        ]

        one_site = case["summaries"]["one_site"]
        old_two_site = case["summaries"]["two_site_metric_als"]
        energy_refined = case["summaries"]["two_site_energy_refined"]
        assert one_site["sweeps_median"] == 1.0
        assert old_two_site["sweeps_median"] == 1.0
        assert energy_refined["sweeps_median"] == 1.0
        assert one_site["local_updates_median"] == 2.0
        assert old_two_site["local_updates_median"] == 1.0
        assert energy_refined["local_updates_median"] == 1.0
        assert one_site["elapsed_seconds_median"] >= 0.0
        assert old_two_site["elapsed_seconds_median"] >= 0.0
        assert energy_refined["elapsed_seconds_median"] >= 0.0
        assert np.isfinite(one_site["final_energy_median"])
        assert np.isfinite(old_two_site["final_energy_median"])
        assert np.isfinite(energy_refined["final_energy_median"])
        assert one_site["absolute_error_to_exact_median"] >= 0.0
        assert old_two_site["absolute_error_to_exact_median"] >= 0.0
        assert energy_refined["absolute_error_to_exact_median"] >= 0.0


def test_benchmark_cli_writes_machine_readable_results(tmp_path, capsys):
    from pyqed._letta_two_site_opt.benchmarks.ising_convergence import main

    output_path = tmp_path / "benchmark.json"
    status = main(
        [
            "--dimension",
            "2d",
            "--shape-2d",
            "1x2",
            "--bond-dim",
            "1",
            "--max-sweeps",
            "1",
            "--repeats",
            "1",
            "--exact-max-sites",
            "2",
            "--no-warmup",
            "--json-output",
            str(output_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_path.read_text())
    assert payload["cases"][0]["dimension"] == "2d"
    assert payload["cases"][0]["shape"] == [1, 2]
    output = capsys.readouterr().out
    assert "one_site" in output
    assert "two_site_metric_als" in output
    assert "two_site_energy_refined" in output
    assert "sweeps_conv" in output
    assert str(output_path) in output

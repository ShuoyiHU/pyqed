import csv

import numpy as np

from pyqed._vuletta.benchmarks.one_dimensional_models import (
    BenchmarkRow,
    common_one_dimensional_models,
    format_benchmark_markdown,
    run_one_dimensional_benchmark,
    write_benchmark_csv,
)


def test_common_model_references_include_tfim_and_heisenberg_observables():
    models = common_one_dimensional_models()

    assert tuple(models) == ("tfim", "heisenberg")
    np.testing.assert_allclose(
        models["heisenberg"].reference_energy,
        0.25 - np.log(2.0),
    )
    heisenberg_references = dict(models["heisenberg"].reference_observables)
    np.testing.assert_allclose(
        heisenberg_references["<Sz Sz>"],
        (0.25 - np.log(2.0)) / 3.0,
    )
    tfim_references = dict(models["tfim"].reference_observables)
    assert set(tfim_references) == {"<X>", "<Z Z>"}


def test_small_benchmark_returns_letta_and_vumps_rows_with_timings():
    rows = run_one_dimensional_benchmark(
        model_names=("tfim",),
        letta_bond_dimensions=(1,),
        mps_bond_dimensions=(1,),
        seed=3,
        tolerance=1.0e-5,
        max_iterations=3,
        repeats=1,
    )

    assert len(rows) == 2
    assert all(isinstance(row, BenchmarkRow) for row in rows)
    assert [(row.method, row.bond_dim) for row in rows] == [
        ("VULETTA", 1),
        ("VUMPS", 1),
    ]
    letta, mps = rows
    assert letta.transfer_bond_dim == 2
    assert letta.tensor_entries == 4
    assert letta.tangent_dimension == 2
    assert mps.transfer_bond_dim == 1
    assert mps.tensor_entries == 2
    assert mps.tangent_dimension is None
    for row in rows:
        assert np.isfinite(row.energy)
        assert np.isfinite(row.energy_error)
        assert row.runtime_seconds >= 0.0
        assert row.iterations >= 0
        assert np.isfinite(row.residual)
        assert row.observables
        assert all(np.isfinite(observable.value) for observable in row.observables)


def test_benchmark_markdown_and_csv_include_observables_and_runtime(tmp_path):
    rows = run_one_dimensional_benchmark(
        model_names=("tfim",),
        letta_bond_dimensions=(1,),
        mps_bond_dimensions=(),
        seed=3,
        tolerance=1.0e-5,
        max_iterations=2,
        repeats=1,
    )

    markdown = format_benchmark_markdown(rows)
    assert "VULETTA" in markdown
    assert "runtime_s" in markdown
    assert "<X>" in markdown

    path = tmp_path / "benchmark.csv"
    write_benchmark_csv(rows, path)
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert records[0]["method"] == "VULETTA"
    assert records[0]["tangent_dimension"] == "2"
    assert float(records[0]["runtime_seconds"]) >= 0.0

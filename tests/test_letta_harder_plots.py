"""Convergence plots preserve the actual per-pass energies and case metadata."""
import json

import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks import cbe_harder_cases as harder
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark


@pytest.fixture
def small_case():
    case = run_benchmark("bose_hubbard", dimension="1d", size=3, bond_dim=2,
                         max_sweeps=3, solvers=harder.SOLVERS[:4], raise_on_failure=True)
    case["case_name"] = "bose_hubbard_3_99"
    return case


def test_plot_contains_signed_differences_to_final_two_site_energy(small_case):
    assert hasattr(harder, "build_convergence_figure")
    fig = harder.build_convergence_figure(small_case)
    ax = fig.axes[0]
    methods = ("letta_one_site", "letta_two_site", "letta_cbe_strict")
    rows = {row["solver"]: row for row in small_case["records"]}
    lines = [line for line in ax.lines if not line.get_label().startswith("_")]
    assert len(lines) == 3
    reference = rows["letta_two_site"]["energy"]
    for line, method in zip(lines, methods):
        energies = rows[method]["sweep_energies"]
        np.testing.assert_array_equal(line.get_xdata(), np.arange(1, len(energies) + 1))
        np.testing.assert_array_equal(line.get_ydata(), np.asarray(energies) - reference)
    heading = "\n".join(text.get_text() for text in fig.texts)
    assert "Bose" in heading and "H=" in heading
    assert "D=2" in heading and "L=3" in heading and "99" not in heading
    assert "directional pass" in ax.get_xlabel()
    assert ax.get_yscale() == "symlog"
    assert "two-site" in ax.get_ylabel()


def test_save_separate_figures_and_source_data(small_case, tmp_path):
    assert hasattr(harder, "save_convergence_figure")
    first = harder.save_convergence_figure(small_case, tmp_path)
    second_case = dict(small_case, case_name="another_case", bond_dim=4)
    second = harder.save_convergence_figure(second_case, tmp_path)
    assert set(first) == {"png", "pdf", "json"}
    assert set(first.values()).isdisjoint(second.values())
    from pathlib import Path
    for path in (*first.values(), *second.values()):
        assert Path(path).is_file() and Path(path).stat().st_size > 100
    saved = json.loads(Path(first["json"]).read_text())
    assert saved["records"] == json.loads(json.dumps(small_case["records"]))
    assert saved["bond_dim"] == 2 and saved["nsites"] == 3


def test_click_run_plot_defaults_and_partial_case_export(tmp_path, monkeypatch):
    args = harder._parser().parse_args([])
    assert hasattr(args, "figure_dir")
    assert args.figure_dir.is_absolute()
    assert args.no_plots is False
    monkeypatch.setattr(harder, "HARD_CASES", {
        "small": ("ising", "1d", 3, 2), "invalid": ("bad_model", "1d", 3, 2),
    })
    report = harder.run_harder_suite(cases=("small", "invalid"), max_sweeps=2,
        figure_dir=tmp_path / "figs", output=tmp_path / "data" / "results.json")
    assert "invalid" in report["failures"]
    assert not report["plot_failures"]
    assert report["cases"][0]["figures"]["png"].endswith(".png")
    saved = json.loads((tmp_path / "data" / "results.json").read_text())
    assert saved["cases"][0]["figures"] == report["cases"][0]["figures"]


def test_missing_solver_is_not_fabricated(small_case):
    assert hasattr(harder, "build_convergence_figure")
    small_case["records"] = small_case["records"][:1]
    fig = harder.build_convergence_figure(small_case)
    assert len(fig.axes[0].lines) == 0
    assert any("not available" in text.get_text() for text in fig.texts)


def test_signed_scale_keeps_negative_positive_and_zero_without_abs(small_case):
    # Deterministic plotting input, not a physical benchmark result.
    small_case["records"] = [
        {"solver": "letta_one_site", "energy": -2.1, "sweep_energies": [-1., -2., -2.1]},
        {"solver": "letta_two_site", "energy": -2., "sweep_energies": [-1.5, -2.]},
        {"solver": "letta_cbe_strict", "energy": -2.01, "sweep_energies": [-1.9, -2.01]},
    ]
    fig = harder.build_convergence_figure(small_case)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1., 0., -.1])
    assert ax.get_yscale() == "symlog"
    fig.canvas.draw()
    assert ax.get_ylim()[0] < 0 < ax.get_ylim()[1]


@pytest.mark.parametrize("counts", [[0, 2, 1], [0, 0, 0], []])
def test_only_cbe_ok_passes_have_filled_magenta_markers(counts):
    from matplotlib.colors import to_rgba

    case = dict(model="ising", dimension="1d", nsites=3, bond_dim=2,
                parameters={"J": 1., "h": 1.}, records=[
        dict(solver="letta_two_site", energy=-2., sweep_energies=[-1., -2.],
             sweep_cbe_accepted=[1, 1]),
        dict(solver="letta_cbe_strict", energy=-2.1,
             sweep_energies=[-1.9, -2., -2.1], sweep_cbe_accepted=counts),
    ])
    fig = harder.build_convergence_figure(case)
    ax = fig.axes[0]
    if any(counts):
        assert len(ax.collections) == 1
        highlight = ax.collections[0]
        np.testing.assert_allclose(highlight.get_offsets(), [[2, 0.], [3, -.1]])
        np.testing.assert_allclose(highlight.get_facecolors(), [to_rgba("magenta")])
        np.testing.assert_allclose(highlight.get_edgecolors(), [to_rgba("magenta")])
        assert "CBE-ok" in highlight.get_label()
    else:
        assert not ax.collections
    assert ax.get_yscale() == "symlog"
    fig.canvas.draw()


@pytest.mark.parametrize("model", list(harder.HAMILTONIAN_LABELS))
def test_model_headings_render_inside_figure(model, tmp_path):
    case = run_benchmark(model, dimension="1d", size=3, bond_dim=1,
        max_sweeps=1, exact_max_dimension=0, solvers=("letta_one_site",),
        raise_on_failure=True)
    case["case_name"] = model
    fig = harder.build_convergence_figure(case)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width
        assert box.y0 >= 0 and box.y1 <= fig.bbox.height
    assert harder.save_convergence_figure(case, tmp_path)["pdf"].endswith(".pdf")


def test_plot_failure_does_not_lose_solver_results(tmp_path, monkeypatch):
    monkeypatch.setattr(harder, "HARD_CASES", {"small": ("ising", "1d", 3, 1)})
    not_a_directory = tmp_path / "file"
    not_a_directory.touch()
    report = harder.run_harder_suite(cases=("small",), max_sweeps=1,
        solvers=("letta_one_site",), figure_dir=not_a_directory)
    assert not report["failures"]
    assert "small" in report["plot_failures"]
    assert len(report["cases"][0]["records"]) == 1

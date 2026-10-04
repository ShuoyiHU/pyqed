import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def small_harder_cases(monkeypatch):
    from pyqed._letta_one_site_opt.benchmarks import cbe_harder_cases

    # Change only input configuration; run the real benchmark and solvers.
    monkeypatch.setattr(cbe_harder_cases, "HARD_CASES", {
        "ising_2x3": ("ising", "2d", (2, 3), 1),
        "ising_4_D4": ("ising", "1d", 4, 4),
        "ising_4_D6": ("ising", "1d", 4, 6),
        "ising_4_D8": ("ising", "1d", 4, 8),
    })
    return cbe_harder_cases


def test_harder_suite_defaults_to_three_methods_without_exact_cbe(small_harder_cases):
    from pyqed._letta_one_site_opt.benchmarks.cbe_harder_cases import _parser, run_harder_suite

    args = _parser().parse_args([])
    assert args.bond_dim is None and args.max_sweeps == 50
    assert args.include_cbe_exact is False
    report = run_harder_suite(cases=("ising_2x3",), bond_dim=1, max_sweeps=1)
    assert not report["failures"]
    case = report["cases"][0]
    from pyqed._letta_one_site_opt.benchmarks.run_condensed_suite import format_suite_table
    assert "ising_2x3" in format_suite_table(report)
    assert {row["solver"] for row in case["records"]} == {
        "letta_one_site", "letta_two_site", "letta_cbe_strict"}
    assert not case["solver_failures"]
    assert len(set(row["initial_state_fingerprint"] for row in case["records"])) == 1
    for row in case["records"]:
        assert row["sweeps"] == 1
        assert row["energy"] is not None


@pytest.mark.parametrize("override", [None, 2])
def test_fourth_case_value_sets_D_without_changing_seed(small_harder_cases, override):
    cases = ("ising_4_D4", "ising_4_D6", "ising_4_D8")
    options = {} if override is None else {"bond_dim": override}
    report = small_harder_cases.run_harder_suite(
        cases=cases, max_sweeps=1, seed=123,
        solvers=("letta_one_site",), **options,
    )
    assert not report["failures"]
    assert all(not case["solver_failures"] for case in report["cases"])
    assert [case["bond_dim"] for case in report["cases"]] == (
        [4, 6, 8] if override is None else [override] * 3
    )
    assert [case["seed"] for case in report["cases"]] == [123] * 3
    subset = small_harder_cases.run_harder_suite(
        cases=(cases[1],), max_sweeps=1, seed=123,
        solvers=("letta_one_site",), **options,
    )
    assert subset["cases"][0]["initial_state_fingerprint"] == report["cases"][1]["initial_state_fingerprint"]


def test_main_uses_case_D_unless_explicitly_overridden(small_harder_cases, capsys, tmp_path):
    report = small_harder_cases.main([
        "--cases", "ising_4_D6", "--max-sweeps", "1",
        "--solvers", "letta_one_site", "--json", "--figure-dir", str(tmp_path),
    ])
    saved = json.loads(capsys.readouterr().out)
    assert report["cases"][0]["bond_dim"] == saved["cases"][0]["bond_dim"] == 6
    assert saved["cases"][0]["seed"] == 731


def test_global_trim_ablation_is_exposed_and_preserves_initial_state():
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark

    args = dict(dimension="2d", size=(2, 3), bond_dim=2, max_sweeps=1,
                solvers=("letta_cbe_strict",), raise_on_failure=True)
    old = run_benchmark("ising", **args, cbe_conditional_trim=False)
    new = run_benchmark("ising", **args, cbe_conditional_trim=True)
    assert old["initial_state_fingerprint"] == new["initial_state_fingerprint"]
    assert old["cbe_conditional_trim"] is False
    assert new["cbe_conditional_trim"] is True
    assert old["records"][0]["energy"] != new["records"][0]["energy"]


def test_harder_suite_click_runs_and_saves_partial_results(tmp_path):
    script = Path(__file__).resolve().parents[1] / "pyqed/_letta_one_site_opt/benchmarks/cbe_harder_cases.py"
    output = tmp_path / "results.json"
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    # Load the real entry point with a tiny case independent of the user's
    # editable case selection. No benchmark or solver is replaced.
    bootstrap = (
        "import runpy, sys; scope = runpy.run_path(sys.argv[1]); "
        "scope['HARD_CASES']['smoke_test'] = ('ising', '2d', (2, 3), 6); "
        "scope['main'](sys.argv[2:])"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(script), "--cases", "smoke_test", "--bond-dim", "1",
         "--max-sweeps", "1", "--exact-max-dimension", "0",
         "--global-trim", "--output", str(output), "--figure-dir", str(tmp_path / "figs")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "FAILED" not in result.stdout
    assert "swp" in result.stdout and "dE/site" in result.stdout
    report = json.loads(output.read_text())
    assert len(report["cases"][0]["records"]) == 3
    assert report["cases"][0]["cbe_conditional_trim"] is False
    assert report["cases"][0]["bond_dim"] == 1
    assert report["cases"][0]["seed"] == 731
    assert Path(report["cases"][0]["figures"]["png"]).is_file()


def test_harder_suite_rejects_unknown_cases():
    from pyqed._letta_one_site_opt.benchmarks.cbe_harder_cases import run_harder_suite
    with pytest.raises(ValueError, match="unknown"):
        run_harder_suite(cases=("typo",))


def test_hubbard_holstein_case_runs_with_four_methods(small_harder_cases):
    from pyqed._letta_one_site_opt.benchmarks.cbe_harder_cases import HARD_CASES, _parser, run_harder_suite

    args = _parser().parse_args([])
    name = "hubbard_holstein_3_4"
    # Case selection is user-editable; enable this input only within the test.
    HARD_CASES[name] = ("hubbard_holstein", "1d", 3, 4)
    report = run_harder_suite(cases=(name,), bond_dim=1, max_sweeps=1,
                              exact_max_dimension=0, include_cbe_exact=True)
    assert not report["failures"]
    case = report["cases"][0]
    assert case["case_name"] == name and case["physical_dim"] == 8
    assert not case["solver_failures"] and len(case["records"]) == 4
    assert all(row["sweeps"] == 1 for row in case["records"])


def test_exact_cbe_can_be_enabled_in_main(small_harder_cases, capsys):
    report = small_harder_cases.main([
        "--cases", "ising_4_D4", "--bond-dim", "1", "--max-sweeps", "1",
        "--include-cbe-exact", "--no-plots", "--json",
    ])
    assert not report["failures"]
    methods = [row["solver"] for row in report["cases"][0]["records"]]
    assert len(methods) == 4 and methods.count("letta_cbe_exact") == 1
    assert json.loads(capsys.readouterr().out)["solvers"] == methods

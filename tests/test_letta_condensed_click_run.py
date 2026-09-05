"""Direct-execution contracts for all condensed-model benchmark scripts."""

from __future__ import annotations

from inspect import signature
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "pyqed" / "_letta_one_site_opt" / "benchmarks"
SCRIPTS = (
    "ising_1d.py",
    "heisenberg_1d.py",
    "bose_hubbard_1d.py",
    "fermi_hubbard_1d.py",
    "ising_2d.py",
    "heisenberg_2d.py",
    "bose_hubbard_2d.py",
    "fermi_hubbard_2d.py",
)
SOLVERS = (
    "letta_one_site",
    "letta_cbe_exact",
    "letta_cbe_strict",
    "letta_two_site",
    "mps_two_site",
)


def test_all_condensed_jobs_default_to_D4_and_fifty_sweeps():
    from pyqed._letta_one_site_opt.benchmarks.condensed_cli import _parser
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
        make_shared_initial_state,
        run_benchmark,
    )
    from pyqed._letta_one_site_opt.benchmarks.run_condensed_suite import (
        run_suite,
    )

    for script_name in SCRIPTS:
        dimension = "1d" if "_1d" in script_name else "2d"
        model_name = script_name.removesuffix(f"_{dimension}.py")
        arguments = _parser(model_name, dimension).parse_args([])
        assert arguments.bond_dim == 4
        assert arguments.max_sweeps == 50

    assert (
        signature(make_shared_initial_state).parameters["bond_dim"].default
        == 4
    )
    assert signature(run_benchmark).parameters["bond_dim"].default == 4
    assert signature(run_benchmark).parameters["max_sweeps"].default == 50
    assert signature(run_suite).parameters["bond_dim"].default == 4
    assert signature(run_suite).parameters["max_sweeps"].default == 50


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_each_model_script_click_runs_from_an_unrelated_directory(
    script_name, tmp_path
):
    size = ["--length", "2"] if "_1d" in script_name else [
        "--rows",
        "2",
        "--columns",
        "2",
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "NUMBA_CACHE_DIR": "/private/tmp/numba-cache",
            "MPLCONFIGDIR": "/private/tmp/mplconfig",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARKS / script_name),
            *size,
            "--bond-dim",
            "1",
            "--expansion-dimension",
            "1",
            "--max-sweeps",
            "1",
            "--exact-max-dimension",
            "1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    for solver in SOLVERS:
        assert solver in completed.stdout


def test_suite_registry_contains_all_eight_cases():
    from pyqed._letta_one_site_opt.benchmarks.run_condensed_suite import (
        SUITE_CASES,
        run_suite,
    )

    assert len(SUITE_CASES) == 8
    report = run_suite(
        length=2,
        shape=(2, 2),
        bond_dim=1,
        expansion_dimension=1,
        max_sweeps=1,
        exact_max_dimension=1,
        solvers=("letta_one_site",),
    )
    assert len(report["cases"]) == 8
    assert report["failures"] == {}

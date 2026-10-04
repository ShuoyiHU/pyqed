"""Independent dense Hamiltonians and runnable CBE contracts for NN chains."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.condensed_models import MODEL_CASES, build_model
from pyqed._letta_one_site_opt.benchmarks.nn_chain_models import CHAIN_MODEL_DEFAULTS


def embed(operator, site, length):
    result = np.ones((1, 1))
    for index in range(length):
        result = np.kron(result, operator if index == site else np.eye(len(operator)))
    return result


def fermion(site, length):
    result = np.ones((1, 1))
    for index in range(length):
        operator = (np.diag([1., -1.]) if index < site else
                    np.array([[0., 1.], [0., 0.]]) if index == site else np.eye(2))
        result = np.kron(result, operator)
    return result


@pytest.mark.parametrize("name,parameters", [
    ("ssh", dict(t1=0.3, t2=1.2, mu=-0.2)),
    ("rice_mele", dict(t1=0.3, t2=1.2, mu=-0.2, stagger=0.7)),
    ("spinless_tv", dict(t=0.8, V=1.7, mu=-0.2)),
    ("kitaev", dict(t=0.8, pairing=-0.6, mu=0.7)),
])
def test_fermion_mpos_match_full_jordan_wigner_algebra(name, parameters):
    length = 4
    model = build_model(name, "1d", length, **parameters)
    c = [fermion(i, length) for i in range(length)]
    n = [operator.T @ operator for operator in c]
    q = [number - 0.5 * np.eye(2 ** length) for number in n]
    expected = np.zeros((2 ** length, 2 ** length))
    for i in range(length - 1):
        hopping = parameters["t1" if i % 2 == 0 else "t2"] if name in {
            "ssh", "rice_mele"} else parameters["t"]
        hop = c[i].T @ c[i + 1]
        expected -= hopping * (hop + hop.T)
        if name == "spinless_tv":
            expected += parameters["V"] * q[i] @ q[i + 1]
        if name == "kitaev":
            pair = c[i] @ c[i + 1]
            expected += parameters["pairing"] * (pair + pair.T)
    expected -= parameters["mu"] * sum(q if name in {"spinless_tv", "kitaev"} else n)
    if name == "rice_mele":
        expected += parameters["stagger"] * sum((-1) ** i * n[i] for i in range(length))
    np.testing.assert_allclose(model.mpo.to_dense(max_sites=length), expected, atol=1e-13)


@pytest.mark.parametrize("name,parameters", [
    ("xy", dict(J=0.7, gamma=-0.4, h=0.3)),
    ("xyz", dict(Jx=0.7, Jy=-0.4, Jz=1.3, h=0.3)),
    ("dimerized_heisenberg", dict(J=0.7, dimer=-0.4, delta=1.3, h=0.3)),
    ("spin1_heisenberg", dict(J=0.7, delta=1.3, single_ion=-0.4, h=0.3)),
    ("blume_capel", dict(J=0.7, single_ion=-0.4, h=0.3)),
])
def test_spin_mpos_match_cartesian_spin_operators(name, parameters):
    length = 4
    model = build_model(name, "1d", length, **parameters)
    if model.physical_dim == 2:
        sx = np.array([[0., 1.], [1., 0.]]) / 2
        sy = np.array([[0., -1j], [1j, 0.]]) / 2
        sz = np.diag([0.5, -0.5])
    else:
        sx = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]) / np.sqrt(2)
        sy = np.array([[0., -1j, 0.], [1j, 0., -1j], [0., 1j, 0.]]) / np.sqrt(2)
        sz = np.diag([1., 0., -1.])
    x, y, z = [[embed(op, site, length) for site in range(length)] for op in (sx, sy, sz)]
    expected = np.zeros((model.hilbert_dim, model.hilbert_dim), dtype=complex)
    p = parameters
    for i in range(length - 1):
        if name == "blume_capel":
            expected -= p["J"] * z[i] @ z[i + 1]
            continue
        if name == "xy":
            couplings = (p["J"] * (1 + p["gamma"]), p["J"] * (1 - p["gamma"]), 0.)
        elif name == "xyz":
            couplings = (p["Jx"], p["Jy"], p["Jz"])
        else:
            j = p["J"] * (1 + (-1) ** i * p.get("dimer", 0.))
            couplings = (j, j, j * p["delta"])
        for coupling, operators in zip(couplings, (x, y, z)):
            expected += coupling * operators[i] @ operators[i + 1]
    expected -= p["h"] * sum(x if name == "blume_capel" else z)
    expected += p.get("single_ion", 0.) * sum(operator @ operator for operator in z)
    np.testing.assert_allclose(model.mpo.to_dense(max_sites=length), expected, atol=1e-13)


@pytest.mark.parametrize("name", CHAIN_MODEL_DEFAULTS)
def test_chain_geometry_validation_and_three_solver_smoke(name):
    from pyqed._letta_one_site_opt.benchmarks.cbe_harder_cases import DEFAULT_SOLVERS
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark

    model = build_model(name, "1d", 3)
    assert (name, "1d") in MODEL_CASES and (name, "2d") not in MODEL_CASES
    assert model.lattice_shape == (1, 3) and model.bonds == ((0, 1), (1, 2))
    for term in model.terms:
        sites = sorted(term.operators)
        assert len(sites) <= 2
        assert len(sites) == 1 or sites[1] == sites[0] + 1
    dense = model.mpo.to_dense(max_sites=3)
    np.testing.assert_allclose(dense, dense.T.conj(), atol=1e-13)
    with pytest.raises(ValueError, match="only in 1d"):
        build_model(name, "2d", (2, 2))
    with pytest.raises(ValueError, match="unknown model parameter"):
        build_model(name, "1d", 3, typo=1.)
    parameter = next(iter(CHAIN_MODEL_DEFAULTS[name]))
    with pytest.raises(ValueError, match="finite"):
        build_model(name, "1d", 3, **{parameter: np.nan})
    report = run_benchmark(name, dimension="1d", size=3, bond_dim=1,
                           max_sweeps=2, solvers=DEFAULT_SOLVERS, raise_on_failure=True)
    assert not report["solver_failures"]
    assert len(report["records"]) == 3
    assert len({row["initial_state_fingerprint"] for row in report["records"]}) == 1
    for row in report["records"]:
        assert np.isfinite(row["energy"])
        assert row["energy"] >= report["exact_energy"] - 1e-8
        assert sum(row["sweep_cbe_accepted"]) == row["cbe_accepted"]
        if row["solver"] == "letta_cbe_strict":
            assert row["selector_pair_actions"] == 0
            assert row["selector_pair_metrics"] == 0
            assert row["selector_merged_pairs"] == 0
    json.dumps(report)


@pytest.mark.parametrize("name", CHAIN_MODEL_DEFAULTS)
def test_model_cli_exposes_all_parameters(name):
    from pyqed._letta_one_site_opt.benchmarks.condensed_cli import _parser, _model_parameters

    parser = _parser(name, "1d")
    values = {key: value + 0.125 for key, value in CHAIN_MODEL_DEFAULTS[name].items()}
    argv = [item for key, value in values.items() for item in ("--" + key.replace("_", "-"), str(value))]
    assert _model_parameters(name, parser.parse_args(argv)) == values


def test_chain_presets_reach_runner_and_figures(tmp_path, monkeypatch, capsys):
    from pyqed._letta_one_site_opt.benchmarks import cbe_harder_cases as harder
    from pyqed._letta_one_site_opt.benchmarks.nn_chain_cases import NN_CHAIN_CASES, NN_CHAIN_PARAMETERS

    assert len(NN_CHAIN_CASES) == 29
    for name, (model, dimension, length, bond_dim) in NN_CHAIN_CASES.items():
        assert harder.HARD_CASES[name] == (model, dimension, length, bond_dim)
        assert dimension == "1d"
        built = build_model(model, dimension, 3, **NN_CHAIN_PARAMETERS[name])
        assert built.name == model
    harder.main(["--list-cases"])
    assert "ssh_weak_end_8_4" in capsys.readouterr().out
    # Keep preset parameters and the real solvers; reduce only the run sizes.
    monkeypatch.setattr(harder, "HARD_CASES", {
        name: (model, dimension, 3, 1)
        for name, (model, dimension, length, bond_dim) in NN_CHAIN_CASES.items()
    })
    report = harder.main(["--cases", "ssh_strong_end_8_4,xxz_easy_axis_8_4",
                          "--max-sweeps", "1", "--figure-dir", str(tmp_path), "--json"])
    capsys.readouterr()
    assert not report["failures"] and not report["plot_failures"]
    assert report["cases"][0]["parameters"]["t1"] == 1.4
    assert report["cases"][1]["parameters"]["delta"] == 2.
    for case in report["cases"]:
        assert len(case["records"]) == 3 and not case["solver_failures"]
        saved = json.loads(Path(case["figures"]["json"]).read_text())
        assert saved["parameters"] == case["parameters"]
        assert saved["plot"]["scale"] == "symlog"
    parser = harder._parser()
    assert parser.parse_args(["--nn-chains"]).nn_chains
    with pytest.raises(SystemExit):
        parser.parse_args(["--nn-chains", "--cases", "ssh_strong_end_8_4"])
    all_cases = harder.main(["--nn-chains", "--max-sweeps", "1",
                             "--solvers", "letta_one_site", "--no-plots", "--json"])
    capsys.readouterr()
    assert not all_cases["failures"]
    assert all_cases["requested_cases"] == list(NN_CHAIN_CASES)
    assert len(all_cases["cases"]) == len(NN_CHAIN_CASES)
    assert all(not case["solver_failures"] for case in all_cases["cases"])


def test_generic_chain_entry_point_runs_from_other_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "pyqed/_letta_one_site_opt/benchmarks/nn_chain_1d.py"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([
        sys.executable, str(script), "--model", "ssh", "--length", "3",
        "--t1", "0.4", "--t2", "1.6", "--bond-dim", "1", "--max-sweeps", "1", "--json",
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["parameters"] == {"t1": 0.4, "t2": 1.6, "mu": 0.}
    assert not report["solver_failures"]
    assert {row["solver"] for row in report["records"]} == {
        "letta_one_site", "letta_two_site", "letta_cbe_strict"}

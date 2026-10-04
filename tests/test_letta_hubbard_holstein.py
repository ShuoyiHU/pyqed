"""Independent occupation-basis checks for the Hubbard-Holstein benchmark."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model


def occupation_hamiltonian(length, cutoff, *, t, U, mu, omega, g):
    # Electron code: 0, up, down, up+down; phonon index varies fastest.
    d = 4 * (cutoff + 1)
    states = list(np.ndindex(*((d,) * length)))
    lookup = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for column, state in enumerate(states):
        electrons = [q // (cutoff + 1) for q in state]
        phonons = [q % (cutoff + 1) for q in state]
        bits = [int(bool(e & (1 << spin))) for e in electrons for spin in (0, 1)]
        for site, (electron, phonon) in enumerate(zip(electrons, phonons)):
            n = electron.bit_count()
            matrix[column, column] += U * (electron == 3) - mu * n + omega * phonon
            for step in (-1, 1):
                new_phonon = phonon + step
                if 0 <= new_phonon <= cutoff:
                    target = list(state)
                    target[site] += step
                    matrix[lookup[tuple(target)], column] += g * (n - 1) * np.sqrt(max(phonon, new_phonon))
        for site in range(length - 1):
            for spin in (0, 1):
                for source, destination in ((site, site + 1), (site + 1, site)):
                    a, b = 2 * source + spin, 2 * destination + spin
                    if not bits[a] or bits[b]:
                        continue
                    moved = bits.copy()
                    sign = (-1) ** sum(moved[:a])
                    moved[a] = 0
                    sign *= (-1) ** sum(moved[:b])
                    moved[b] = 1
                    target = tuple((moved[2*k] + 2*moved[2*k+1]) * (cutoff + 1) + phonons[k]
                                   for k in range(length))
                    matrix[lookup[target], column] -= t * sign
    return matrix


@pytest.mark.parametrize("length,cutoff", [(2, 1), (2, 2), (3, 1)])
def test_mpo_matches_independent_occupation_matrix(length, cutoff):
    parameters = dict(t=.7, U=1.3, mu=.2, omega=.6, g=.37)
    model = build_model("hubbard_holstein", "1d", length, max_phonons=cutoff, **parameters)
    dense = model.mpo.to_dense(max_sites=length)
    expected = occupation_hamiltonian(length, cutoff, **parameters)
    assert model.physical_dim == 4 * (cutoff + 1)
    np.testing.assert_allclose(dense, expected, atol=1e-12)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)
    number = np.array([sum((q // (cutoff + 1)).bit_count() for q in state)
                       for state in np.ndindex(*((model.physical_dim,) * length))])
    np.testing.assert_allclose((number[:, None] - number[None, :]) * dense, 0., atol=1e-12)


def test_zero_phonon_cutoff_reduces_to_hubbard():
    hh = build_model("hubbard_holstein", "1d", 3, max_phonons=0, t=.7, U=1.2, mu=.3, g=2.)
    hubbard = build_model("fermi_hubbard", "1d", 3, t=.7, U=1.2, mu=.3)
    assert hh.physical_dim == 4
    np.testing.assert_allclose(hh.mpo.to_dense(max_sites=3), hubbard.mpo.to_dense(max_sites=3), atol=1e-12)


def test_paper_couplings_and_cli_cutoff():
    from pyqed._letta_one_site_opt.benchmarks.condensed_cli import _parser, _model_parameters
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import MODEL_CASES

    model = build_model("hubbard_holstein", "1d", 2)
    assert model.parameters == dict(t=1., U=.8, mu=0., omega=.5, g=np.sqrt(.2), max_phonons=1)
    assert ("hubbard_holstein", "1d") in MODEL_CASES
    assert ("hubbard_holstein", "2d") not in MODEL_CASES
    args = _parser("hubbard_holstein", "1d").parse_args(["--max-phonons", "3", "--omega", ".9", "--g", ".2"])
    assert args.bond_dim == 4 and args.max_sweeps == 50
    parameters = _model_parameters("hubbard_holstein", args)
    assert parameters["max_phonons"] == 3 and parameters["omega"] == .9 and parameters["g"] == .2


@pytest.mark.parametrize("parameters,message", [
    ({"max_phonons": -1}, "max_phonons"),
    ({"max_phonons": 1.5}, "max_phonons"),
    ({"omega": 0.}, "omega"),
    ({"omega": -1.}, "omega"),
    ({"g": float("nan")}, "g"),
])
def test_invalid_phonon_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        build_model("hubbard_holstein", "1d", 2, **parameters)


def test_hubbard_holstein_is_explicitly_1d_only():
    with pytest.raises(ValueError, match="1d"):
        build_model("hubbard_holstein", "2d", (2, 2))


def test_four_letta_methods_report_energy_passes_and_times():
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import SOLVERS, run_benchmark
    report = run_benchmark("hubbard_holstein", dimension="1d", size=2,
                           bond_dim=1, max_sweeps=2, exact_max_dimension=64,
                           solvers=SOLVERS[:4], raise_on_failure=True)
    assert not report["solver_failures"]
    assert report["physical_dim"] == 8 and report["exact_energy"] is not None
    assert len(report["records"]) == 4
    for row in report["records"]:
        assert row["energy"] >= report["exact_energy"] - 1e-8
        assert row["initial_state_fingerprint"] == report["initial_state_fingerprint"]
        assert len(row["sweep_elapsed_seconds"]) == row["sweeps"]
        assert row["elapsed_seconds"] > 0.

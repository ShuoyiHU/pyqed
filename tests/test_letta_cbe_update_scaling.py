"""Full-solver cost measurements must not substitute selector-only timings."""

import importlib
import json

import numpy as np


def test_complete_pass_profile_repeats_identical_initial_states():
    runner = importlib.import_module(
        "pyqed._letta_one_site_opt.benchmarks.cbe_update_scaling")
    point = runner.profile_update_point(
        lattice_shape=(1, 4), bond_dimension=1, physical_dimension=2,
        mpo_width=2, preselection_dimension=2, repeats=2, passes=1, seed=817,
    )
    assert point["scope"] == "complete_solver_including_environment_setup"
    assert set(point["records"]) == {"one_site", "strict_cbe", "two_site"}
    for method, record in point["records"].items():
        assert len(record["elapsed_seconds"]) == 2
        assert min(record["elapsed_seconds"]) > 0.
        assert record["passes"] == [1, 1]
        np.testing.assert_allclose(record["energies"], record["profiled_energy"], atol=1.e-10)
        assert record["profile"]["contractions"] > 0
        assert record["profile"]["opt_cost"] > 0
        assert record["profile"]["largest_live_tensor"] > 0
        assert record["profile"]["estimated_largest_array_bytes"] > 0
        assert record["initial_state_fingerprint"] == point["initial_state_fingerprint"]
        assert record["energies"][0] <= point["initial_energy"] + 1.e-8
    assert point["records"]["strict_cbe"]["selection_diagnostics"]
    assert "not measured peak RSS" in point["memory_note"]
    json.dumps(point)


def test_probe_mpo_is_hermitian_for_higher_local_dimension():
    runner = importlib.import_module(
        "pyqed._letta_one_site_opt.benchmarks.cbe_update_scaling")
    mpo = runner.probe_mpo((2, 2), physical_dimension=3, width=3, seed=42)
    matrix = mpo.to_dense()
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.e-13)
    assert mpo.lattice_shape == (2, 2)

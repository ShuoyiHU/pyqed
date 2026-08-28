import numpy as np

from pyqed._letta_one_site_opt import (
    ReducedPhysicalBasis,
    physical_leg_from_reduced_basis,
    su2_heisenberg_mpo,
)
from pyqed._letta_two_site_opt.benchmarks.heisenberg_symmetry import (
    dense_heisenberg_mpo,
    run_heisenberg_symmetry_benchmark,
)


def _dense_matrix_from_factors(factors):
    states = {0: np.array([[1.0]], dtype=complex)}
    for core in factors:
        dense = core.as_dense()
        updated = {}
        for left, accumulated in states.items():
            for right in range(dense.shape[1]):
                if np.any(dense[left, right]):
                    updated[right] = updated.get(right, 0.0) + np.kron(
                        accumulated, dense[left, right]
                    )
        states = updated
    return states[0]


def test_dense_and_reduced_heisenberg_builders_describe_same_operator():
    basis = ReducedPhysicalBasis.spin_half()
    canonical_leg = physical_leg_from_reduced_basis(basis, fully_reduced=False)
    reduced = su2_heisenberg_mpo(
        4, physical_basis=basis, physical_leg=canonical_leg, coupling=0.61
    )

    np.testing.assert_allclose(
        _dense_matrix_from_factors(reduced.canonical_factors),
        dense_heisenberg_mpo(4, coupling=0.61).to_dense(),
        atol=1.0e-12,
    )


def test_none_u1_su2_benchmark_matches_energy_and_reports_reduction():
    report = run_heisenberg_symmetry_benchmark(
        nsites=4,
        solvers=("one-site",),
        bond_dim=4,
        multiplets_per_sector=1,
        max_sweeps=4,
        tolerance=1.0e-10,
        repeats=1,
        seed=5,
    )
    data = report["solvers"]["one-site"]
    exact = report["metadata"]["exact_energy"]

    assert report["metadata"]["gauge_mode"] == "scalar"
    assert report["metadata"]["matrix_free"] is True
    assert data["agreement"]["efficiency_comparison_valid"] is True

    for label in ("none", "u1", "su2"):
        assert abs(data[label]["energy_median"] - exact) < 1.0e-9
        assert data[label]["symmetry_violation_max"] < 1.0e-12
        assert data[label]["peak_traced_memory_bytes_median"] > 0
    assert data["su2"]["parameter_count"] < data["u1"]["parameter_count"]
    assert data["u1"]["parameter_count"] < data["none"]["parameter_count"]
    assert data["su2"]["max_local_dimension"] < data["none"]["max_local_dimension"]

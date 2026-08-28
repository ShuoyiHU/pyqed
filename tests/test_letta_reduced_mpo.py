import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    LETTADMROptions,
    ReducedLatticeLETTA,
    ReducedPhysicalBasis,
    ReducedMPOHamiltonian,
    ReducedSymmetry,
    letta_dmrg,
    physical_leg_from_reduced_basis,
    reduced_local_problem,
    su2_heisenberg_mpo,
    su2_spin_operator,
)
from pyqed.mps.nonabelian.builder import identity_operator
from pyqed.mps.nonabelian.mpo import MPO


def _heisenberg_dense(nsites, coupling=1.0):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    identity = np.eye(2)
    hamiltonian = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites - 1):
        for operator in (sx, sy, sz):
            factors = [identity] * nsites
            factors[site] = operator
            factors[site + 1] = operator
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            hamiltonian += coupling * term
    return hamiltonian


def _dense_matrix_from_mpo_list(mpo):
    states = {0: np.array([[1.0]], dtype=complex)}
    for core in mpo:
        dense_core = core.as_dense()
        new_states = {}
        for left_index, accumulated in states.items():
            for right_index in range(dense_core.shape[1]):
                local = dense_core[left_index, right_index]
                if not np.any(local):
                    continue
                contribution = np.kron(accumulated, local)
                new_states[right_index] = (
                    new_states.get(right_index, 0.0) + contribution
                )
        states = new_states
    return states[0]


def _random_singlet(seed=9):
    basis = ReducedPhysicalBasis.spin_half()
    symmetry = ReducedSymmetry.su2(basis, target_two_j=0)
    return ReducedLatticeLETTA.random(
        (1, 4), symmetry=symmetry, multiplets_per_sector=1, seed=seed
    )


def test_spin_half_reduced_heisenberg_mpo_has_exact_dense_normalization():
    basis = ReducedPhysicalBasis.spin_half()
    reduced_leg = physical_leg_from_reduced_basis(basis)
    canonical_leg = physical_leg_from_reduced_basis(basis, fully_reduced=False)
    hamiltonian = su2_heisenberg_mpo(
        4,
        physical_basis=basis,
        physical_leg=canonical_leg,
        coupling=0.73,
    )

    assert all(
        core.phys_in_leg == reduced_leg and core.phys_out_leg == reduced_leg
        for core in hamiltonian.factors
    )
    np.testing.assert_allclose(
        _dense_matrix_from_mpo_list(hamiltonian.canonical_factors),
        _heisenberg_dense(4, coupling=0.73),
        atol=1.0e-12,
    )


def test_projected_frontier_mpo_local_problem_matches_dense_reference():
    state = _random_singlet(seed=13)
    factors = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    dense_problem = reduced_local_problem(
        state, _heisenberg_dense(state.nsites), site=1
    )
    mpo_problem = reduced_local_problem(state, factors, site=1)

    np.testing.assert_allclose(
        mpo_problem.hamiltonian, dense_problem.hamiltonian, atol=1.0e-12
    )
    np.testing.assert_allclose(
        mpo_problem.metric, dense_problem.metric, atol=1.0e-12
    )
    assert mpo_problem.frame is None


def test_frontier_mpo_path_never_materializes_full_state_vector(monkeypatch):
    state = _random_singlet(seed=17)
    factors = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the reduced-MPO path materialized a dense state vector")

    monkeypatch.setattr(ReducedLatticeLETTA, "state_vector", forbidden)
    problem = reduced_local_problem(state, factors, site=2)

    assert problem.hamiltonian.shape == problem.metric.shape
    assert problem.local_dimension > 0


def test_one_site_frontier_mpo_sweep_matches_dense_ground_energy():
    state = _random_singlet(seed=23)
    factors = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    result = letta_dmrg(
        factors,
        state=state,
        options=LETTADMROptions(
            max_sweeps=12,
            tolerance=1.0e-12,
            metric_tolerance=1.0e-13,
            gauge_mode="none",
        ),
    )

    exact = np.linalg.eigvalsh(_heisenberg_dense(state.nsites))[0]
    assert abs(result.energy - exact) < 1.0e-10
    assert result.state.symmetry_violation() == 0.0


def test_user_defined_multi_irrep_basis_uses_same_general_mpo_entry_point():
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(
        basis, target_charge=4, target_two_j=0
    )
    state = ReducedLatticeLETTA.random(
        (2, 2), symmetry=symmetry, multiplets_per_sector=1, seed=29
    )
    reduced_leg = physical_leg_from_reduced_basis(basis)
    canonical_leg = physical_leg_from_reduced_basis(
        basis, fully_reduced=False
    )
    reduced_identity = MPO.from_site_operator(identity_operator(reduced_leg))
    canonical_identity = MPO.from_site_operator(identity_operator(canonical_leg))
    hamiltonian = ReducedMPOHamiltonian(
        factors=(reduced_identity,) * state.nsites,
        canonical_factors=(canonical_identity,) * state.nsites,
        name="identity",
    )

    problem = reduced_local_problem(state, hamiltonian, site=1)

    np.testing.assert_allclose(
        problem.hamiltonian, problem.metric, atol=1.0e-12
    )
    assert problem.local_dimension < problem.full_local_dimension

    result = letta_dmrg(
        hamiltonian,
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            matrix_free=True,
            dense_solver_threshold=1,
            gauge_mode="scalar",
        ),
    )
    assert result.energy == pytest.approx(1.0, abs=1.0e-11)
    assert result.state.norm() == pytest.approx(1.0, abs=1.0e-11)


def test_raw_rank_coupled_chain_requires_exact_hamiltonian_wrapper():
    state = _random_singlet(seed=31)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )

    with pytest.raises(ValueError, match="ReducedMPOHamiltonian"):
        reduced_local_problem(state, tuple(hamiltonian.factors), site=1)


def test_reduced_environment_energy_remains_accurate_after_large_local_gauges():
    basis = ReducedPhysicalBasis.spin_half()
    state = ReducedLatticeLETTA.random(
        (1, 8),
        symmetry=ReducedSymmetry.su2(basis, target_two_j=0),
        multiplets_per_sector=3,
        seed=7,
    )
    result = letta_dmrg(
        su2_heisenberg_mpo(8, physical_basis=basis),
        state=state,
        options=LETTADMROptions(
            max_sweeps=1,
            tolerance=1.0e-14,
            metric_tolerance=1.0e-12,
            gauge_mode="none",
        ),
    )
    vector = np.asarray(result.state.state_vector(), dtype=complex)
    dense = _heisenberg_dense(8)
    reference = np.vdot(vector, dense @ vector) / np.vdot(vector, vector)

    assert result.energy == pytest.approx(reference.real, abs=1.0e-12)


def test_canonical_mpo_rejects_wrong_physical_representation():
    state = _random_singlet(seed=37)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    malformed = ReducedMPOHamiltonian(
        factors=hamiltonian.factors,
        canonical_factors=hamiltonian.factors,
        name="wrong canonical physical leg",
    )

    with pytest.raises(ValueError, match="canonical MPO physical leg"):
        reduced_local_problem(state, malformed, site=1)


def test_canonical_mpo_rejects_broken_virtual_chain():
    state = _random_singlet(seed=41)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    canonical = list(hamiltonian.canonical_factors)
    canonical[1] = canonical[1].as_dense()[1:, :, :, :]
    malformed = ReducedMPOHamiltonian(
        factors=hamiltonian.factors,
        canonical_factors=tuple(canonical),
        name="broken canonical virtual bond",
    )

    with pytest.raises(ValueError, match="virtual dimensions"):
        reduced_local_problem(state, malformed, site=1)


def test_spin_operator_requires_explicit_reduced_or_canonical_representation():
    basis = ReducedPhysicalBasis.spin_half()
    reduced_leg = physical_leg_from_reduced_basis(basis)
    canonical_leg = physical_leg_from_reduced_basis(basis, fully_reduced=False)

    reduced = su2_spin_operator(
        reduced_leg, physical_basis=basis, fully_reduced=True
    )
    canonical = su2_spin_operator(
        canonical_leg, physical_basis=basis, fully_reduced=False
    )

    assert reduced.phys_in_leg == reduced_leg
    assert canonical.phys_in_leg == canonical_leg
    with pytest.raises(ValueError, match="fully reduced"):
        su2_spin_operator(
            canonical_leg, physical_basis=basis, fully_reduced=True
        )


def test_spin_operator_rejects_ambiguous_outer_multiplicity_action():
    sector = ReducedPhysicalBasis.spin_half().sectors[0]
    basis = ReducedPhysicalBasis(
        labels=("flavor",), sectors=(sector,), multiplicities=(2,)
    )
    leg = physical_leg_from_reduced_basis(basis)

    with pytest.raises(NotImplementedError, match="outer-multiplicity"):
        su2_spin_operator(leg, physical_basis=basis, fully_reduced=True)

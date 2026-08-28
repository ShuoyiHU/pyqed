from collections import Counter
import warnings

import numpy as np
import pytest
import pyqed._letta_two_site_opt.reduced_solver as reduced_solver_module

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA,
    ReducedMPOHamiltonian,
    ReducedPhysicalBasis,
    ReducedSymmetry,
    physical_leg_from_reduced_basis,
    su2_heisenberg_mpo,
)
from pyqed._letta_two_site_opt import (
    LETTATwoSiteOptions,
    letta_two_site_dmrg,
    reduced_pair_problem,
)
from pyqed.mps.nonabelian.builder import identity_operator
from pyqed.mps.nonabelian.mpo import MPO
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep


def _heisenberg_dense(nsites):
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
            hamiltonian += term
    return hamiltonian


def _state(seed=31, multiplets_per_sector=1):
    basis = ReducedPhysicalBasis.spin_half()
    symmetry = ReducedSymmetry.su2(basis, target_two_j=0)
    return ReducedLatticeLETTA.random(
        (1, 4),
        symmetry=symmetry,
        multiplets_per_sector=multiplets_per_sector,
        seed=seed,
    )


def test_reduced_pair_rayleigh_quotient_matches_dense_state_energy():
    state = _state(seed=37)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    problem = reduced_pair_problem(state, hamiltonian, left_site=1)
    vector = problem.old_vector
    local_energy = np.vdot(vector, problem.hamiltonian @ vector) / np.vdot(
        vector, problem.metric @ vector
    )

    dense = state.state_vector()
    dense_energy = np.vdot(dense, _heisenberg_dense(state.nsites) @ dense) / np.vdot(
        dense, dense
    )
    np.testing.assert_allclose(local_energy, dense_energy, atol=1.0e-12)
    assert problem.local_dimension < 2**state.nsites


def test_reduced_two_site_sweep_does_not_materialize_full_state(monkeypatch):
    state = _state(seed=41)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("two-site reduced MPO path materialized the full state")

    monkeypatch.setattr(ReducedLatticeLETTA, "state_vector", forbidden)
    result = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=2,
            tolerance=1.0e-12,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    assert np.isfinite(result.energy)
    assert result.state.symmetry_violation() == 0.0


def test_reduced_two_site_heisenberg_energy_matches_exact_ground_state():
    state = _state(seed=43)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    result = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=8,
            tolerance=1.0e-12,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    exact = np.linalg.eigvalsh(_heisenberg_dense(state.nsites))[0]
    assert abs(result.energy - exact) < 1.0e-10
    assert result.state.symmetry_violation() == 0.0
    assert all(
        update.sector_ranks
        for sweep in result.history
        for update in sweep.updates
    )


def test_reduced_two_site_truncation_keeps_complete_multiplets():
    state = _state(seed=47, multiplets_per_sector=2)
    original_bonds = state.bond_sectors
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    result = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=max(len(bond) for bond in original_bonds),
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            tolerance=1.0e-12,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    for retained, allocated in zip(result.state.bond_sectors, original_bonds):
        retained_counts = Counter(retained)
        allocated_counts = Counter(allocated)
        assert set(retained_counts) <= set(allocated_counts)
        assert all(
            retained_counts[sector] <= allocated_counts[sector]
            for sector in retained_counts
        )
    for sweep in result.history:
        for update in sweep.updates:
            assert sum(update.sector_ranks) <= len(original_bonds[update.left_site])
            assert update.conditional_discarded_weight >= 0.0


def test_reduced_two_site_rejects_unsupported_dense_split_modes():
    state = _state(seed=53)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )

    with pytest.raises(NotImplementedError, match="conditional-svd"):
        letta_two_site_dmrg(
            hamiltonian,
            state=state,
            bond_dim=2,
            options=LETTATwoSiteOptions(
                max_sweeps=1,
                split_method="metric-als",
                gauge_mode="none",
            ),
        )


def test_multi_irrep_two_site_projection_preserves_normalized_state():
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(
        basis, target_charge=4, target_two_j=0
    )
    state = ReducedLatticeLETTA.random(
        (2, 2), symmetry=symmetry, multiplets_per_sector=1, seed=3
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = letta_two_site_dmrg(
            hamiltonian,
            state=state,
            bond_dim=max(
                sum(sector.irrep.dim for sector in bond)
                for bond in state.bond_sectors
            ),
            options=LETTATwoSiteOptions(
                max_sweeps=1,
                tolerance=1.0e-12,
                split_method="conditional-svd",
                gauge_mode="none",
            ),
        )

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert result.state.norm() == pytest.approx(1.0, abs=1.0e-11)
    assert result.energy == pytest.approx(1.0, abs=1.0e-11)
    assert all(
        np.isfinite(update.metric_truncation_loss)
        for sweep in result.history
        for update in sweep.updates
    )


def test_two_site_bond_cap_structurally_removes_discarded_multiplets():
    state = _state(seed=61, multiplets_per_sector=2)
    initial_parameters = state.parameter_count
    initial_bonds = state.bond_sectors
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )

    result = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=2,
            tolerance=1.0e-14,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    assert result.state.parameter_count < initial_parameters
    assert result.state.bond_sectors != initial_bonds
    assert all(
        len(bond) <= 2
        for bond in result.state.bond_sectors
    )
    assert all(
        sum(update.sector_ranks) <= 2
        for sweep in result.history
        for update in sweep.updates
    )


def test_global_multiplet_selection_uses_irrep_weighted_discarded_norm():
    singlet = SpinChargeSector(0, SU2Irrep(0))
    triplet = SpinChargeSector(0, SU2Irrep(2))
    decompositions = {
        singlet: {
            "singular_values": np.array([2.0]),
            "available": 1,
        },
        triplet: {
            "singular_values": np.array([1.3]),
            "available": 1,
        },
    }

    retained = reduced_solver_module._select_multiplet_ranks(
        decompositions, bond_dim=1
    )

    # 3 * 1.3**2 > 1 * 2.0**2, so the complete triplet carries more norm.
    assert retained == {singlet: 0, triplet: 1}


def test_reduced_pair_matrix_free_actions_match_dense_projection():
    state = _state(seed=79)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    dense = reduced_pair_problem(state, hamiltonian, left_site=1)
    matrix_free = reduced_pair_problem(
        state,
        hamiltonian,
        left_site=1,
        matrix_free=True,
        dense_solver_threshold=1,
    )
    rng = np.random.default_rng(83)
    vector = rng.normal(size=dense.local_dimension) + 1.0j * rng.normal(
        size=dense.local_dimension
    )

    assert matrix_free.hamiltonian is None
    assert matrix_free.metric is None
    np.testing.assert_allclose(
        matrix_free.hamiltonian_action(vector),
        dense.hamiltonian @ vector,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        matrix_free.metric_action(vector),
        dense.metric @ vector,
        atol=2.0e-12,
    )


def test_reduced_two_site_matrix_free_sweep_avoids_dense_pair_frame(monkeypatch):
    state = _state(seed=89)
    hamiltonian = su2_heisenberg_mpo(
        state.nsites, physical_basis=state.physical_basis
    )
    reference = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            matrix_free=False,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("matrix-free reduced solve built a dense pair frame")

    monkeypatch.setattr(reduced_solver_module, "_pair_source_frame", forbidden)
    result = letta_two_site_dmrg(
        hamiltonian,
        state=state,
        bond_dim=2,
        options=LETTATwoSiteOptions(
            max_sweeps=1,
            matrix_free=True,
            dense_solver_threshold=1,
            split_method="conditional-svd",
            gauge_mode="none",
        ),
    )

    assert np.isfinite(result.energy)
    assert result.state.norm() == pytest.approx(1.0, abs=1.0e-10)
    assert result.energy == pytest.approx(reference.energy, abs=2.0e-10)

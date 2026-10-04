"""Independent symmetry, chemistry-MPO, and reduced-MPS interoperability checks."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA, ReducedPhysicalBasis, ReducedSymmetry,
    LETTADMROptions, letta_dmrg, reduced_local_problem,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, initial_state
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_one_site_opt.reduced_frontier import reduced_mps_state_vector
from pyqed.mps.nonabelian.states import build_random_reduced_spatial_mps
from test_letta_qchem import integrals, determinant_hamiltonian


def dense_mpo(factors):
    acc = np.ones((1, 1, 1))
    for core in factors:
        w = core.as_dense() if hasattr(core, 'as_dense') else np.asarray(core)
        acc = np.einsum('ijb,bcst->isjtc', acc, w).reshape(
            acc.shape[0]*w.shape[2], acc.shape[1]*w.shape[3], w.shape[1])
    return acc[:, :, 0]


def total_operators(n):
    number = np.diag([0., 1., 1., 2.])
    sz = np.diag([0., .5, -.5, 0.])
    plus = np.zeros((4, 4)); plus[1, 2] = 1.
    def total(local):
        result = np.zeros((4**n, 4**n))
        for i in range(n):
            term = np.array([[1.]])
            for j in range(n):
                term = np.kron(term, local if i == j else np.eye(4))
            result += term
        return result
    N, Z, P = map(total, (number, sz, plus))
    return N, Z, Z@Z + .5*(P@P.T+P.T@P)


@pytest.mark.parametrize('mode', ['n', 'n_sz'])
def test_abelian_chemistry_modes_keep_owned_charges(mode):
    h, g = integrals(3)
    p = ElectronicProblem(h, g, (2, 1))
    mps = initial_state(p, max_bond_dim=12, symmetry=mode)
    from pyqed._letta_one_site_opt.qchem import embed_ties
    tied = embed_ties(mps, tie_neighborhoods(3, nearest=True))
    N, Z, _ = total_operators(3)
    v = tied.state_vector()
    np.testing.assert_allclose(N@v, 3*v, atol=1e-12)
    if mode == 'n_sz':
        np.testing.assert_allclose(Z@v, .5*v, atol=1e-12)
    np.testing.assert_allclose(v, mps.state_vector(), atol=1e-13)


@pytest.mark.parametrize('two_s,nelec', [(0, (1, 1)), (1, (2, 1)), (2, (2, 0))])
def test_reduced_dmrg_import_preserves_all_magnetic_components(two_s, nelec):
    p = ElectronicProblem(*integrals(3), nelec)
    symmetry = p.symmetry('su2', two_s=two_s)
    sites = build_random_reduced_spatial_mps(
        3, target_sector=symmetry.sector, bond_multiplicity=2, seed=414)
    N, Z, S2 = total_operators(3)
    for neighborhoods in [tuple((i,) for i in range(3)),
                          tie_neighborhoods(3, nearest=True),
                          tie_neighborhoods(3, [(0, 2)], nearest=True)]:
        state = ReducedLatticeLETTA.from_mps(
            sites, symmetry=symmetry, neighborhoods=neighborhoods)
        assert state.neighborhoods == neighborhoods
        for m in range(-two_s, two_s+1, 2):
            v = state.state_vector(target_two_m=m)
            ref = reduced_mps_state_vector(sites, symmetry.physical_basis,
                    target_sector=symmetry.sector, target_two_m=m)
            np.testing.assert_allclose(v, ref, atol=2e-13)
            np.testing.assert_allclose(N@v, sum(nelec)*v, atol=2e-13)
            np.testing.assert_allclose(Z@v, m/2*v, atol=2e-13)
            np.testing.assert_allclose(S2@v, two_s*(two_s+2)/4*v, atol=2e-13)
        np.testing.assert_allclose(state.copy().state_vector(), state.state_vector())
        assert all(d >= r for d, r in zip(state.magnetic_bond_dimensions,
                                          state.bond_dimensions))


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_su2_chemistry_operator_matches_independent_fermionic_action(n):
    h, g = integrals(n)
    p = ElectronicProblem(h, g, (1, 0), .31)
    H = dense_mpo(p.su2_mpo().canonical_factors)
    ref = determinant_hamiltonian(p)
    np.testing.assert_allclose(H, ref, atol=8e-12)
    N, Z, S2 = total_operators(n)
    for generator in (N, Z, S2):
        np.testing.assert_allclose(H@generator, generator@H, atol=8e-12)


@pytest.mark.parametrize('nelec,two_s', [((1, 1), 0), ((2, 0), 2), ((0, 0), 0)])
def test_molecular_singlet_triplet_and_vacuum_without_dense_projection(nelec, two_s, monkeypatch):
    p = ElectronicProblem(*integrals(2), nelec, .12)
    symmetry = p.symmetry('su2', two_s=two_s)
    state = ReducedLatticeLETTA.random((1, 2), symmetry=symmetry, seed=3)
    H = determinant_hamiltonian(p)
    N, Z, S2 = total_operators(2)
    # Independent simultaneous eigenspace of the requested N, S, highest M.
    allowed = np.flatnonzero((np.diag(N) == sum(nelec)) & (np.diag(Z) == two_s/2))
    s, U = np.linalg.eigh(S2[np.ix_(allowed, allowed)])
    frame = np.eye(16)[:, allowed] @ U[:, np.abs(s-two_s*(two_s+2)/4) < 1e-10]
    exact = np.linalg.eigvalsh(frame.T@H@frame)[0]
    mpo = p.su2_mpo()
    def forbidden(*args, **kwargs):
        raise AssertionError('full state reconstruction entered production sweep')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    result = letta_dmrg(mpo, state=state, options=LETTADMROptions(
        max_sweeps=8, tolerance=1e-11, matrix_free=True, dense_solver_threshold=1))
    assert result.energy == pytest.approx(exact, abs=2e-9)
    assert result.state.norm() == pytest.approx(1., abs=2e-10)
    assert all(u.accepted for sweep in result.history for u in sweep.updates)


def test_target_validation_and_no_silent_magnetic_import():
    p = ElectronicProblem(*integrals(3), (2, 1))
    with pytest.raises(ValueError, match='spin'):
        p.symmetry('su2', two_s=0)
    from pyqed.mps.nonabelian.states import build_random_spatial_mps
    sym = p.symmetry('su2', two_s=1)
    with pytest.raises(ValueError, match='fully reduced'):
        ReducedLatticeLETTA.from_mps(build_random_spatial_mps(3), symmetry=sym)


def test_two_site_growth_discovers_missing_charge_spin_multiplets(monkeypatch):
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
    p = ElectronicProblem(np.array([[0., -1.], [-1., 0.]]), np.zeros((2,)*4), (1, 1))
    sym = p.symmetry('su2')
    vacuum, single, double = sym.physical_basis.sectors
    state = ReducedLatticeLETTA((1, 2), sym,
        [{(vacuum, double, double): np.ones((1, 1, 3, 1))},
         {(double, vacuum, double): np.ones((1, 1, 1))}], bond_sectors=((double,),))
    original = state.state_vector()
    def forbidden(*args, **kwargs):
        raise AssertionError('dense projection during bond growth')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    result = letta_two_site_dmrg(p.su2_mpo(), state=state, bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=3, split_method='conditional-svd',
            reduced_sector_growth=True, gauge_mode='frontier', dense_solver_threshold=1))
    assert result.energy == pytest.approx(-2., abs=1e-9)
    assert set(result.state.bond_sectors[0]) == {vacuum, single, double}
    assert result.state.bond_dimensions == (3,)
    assert state.bond_sectors == ((double,),)
    monkeypatch.undo()
    np.testing.assert_array_equal(state.state_vector(), original)


def test_import_after_dmrg_decomposition_with_explicit_representation():
    from pyqed.mps.nonabelian import MPS
    p = ElectronicProblem(*integrals(4), (2, 2))
    sym = p.symmetry('su2')
    mps = MPS(build_random_reduced_spatial_mps(4, target_sector=sym.sector,
              bond_multiplicity=2, seed=714))
    mps.left_canonicalize()
    ref = reduced_mps_state_vector(mps, sym.physical_basis, target_sector=sym.sector)
    state = ReducedLatticeLETTA.from_mps(mps, symmetry=sym,
        physical_representation='fully_reduced_su2',
        neighborhoods=tie_neighborhoods(4, nearest=True))
    np.testing.assert_allclose(state.state_vector(), ref, atol=2e-12)


def test_growth_padding_preserves_incumbent_and_rejection_restores_bond(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as rs
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions
    from types import SimpleNamespace
    p = ElectronicProblem(*integrals(3), (2, 1))
    state = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), seed=9)
    original = state.state_vector()
    expanded = rs._expand_reduced_pair_space(state, 1, 5)
    np.testing.assert_allclose(expanded.state_vector(), original, atol=1e-14)
    old_bonds = state.bond_sectors
    monkeypatch.setattr(rs, '_optimize_allocated_reduced_pair',
                        lambda *a: SimpleNamespace(accepted=False))
    rs._optimize_reduced_pair(state, p.su2_mpo(), 1, 'lr', 5,
                             LETTATwoSiteOptions(reduced_sector_growth=True))
    assert state.bond_sectors == old_bonds
    np.testing.assert_array_equal(state.state_vector(), original)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_zero_hamiltonian_and_single_orbital_open_shell(n):
    p = ElectronicProblem(np.zeros((n, n)), np.zeros((n,)*4), (1, 0))
    state = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2'), seed=12)
    hamiltonian = p.su2_mpo()
    np.testing.assert_array_equal(dense_mpo(hamiltonian.canonical_factors), np.zeros((4**n,)*2))
    result = letta_dmrg(hamiltonian, state=state,
                       options=LETTADMROptions(max_sweeps=2))
    assert result.energy == pytest.approx(0., abs=1e-12)
    assert result.state.norm() == pytest.approx(1., abs=1e-12)

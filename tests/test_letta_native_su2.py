"""Native SU(2) operator compilation, recoupling, and production-path checks."""
from dataclasses import replace

import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    ReducedLatticeLETTA, ReducedPhysicalBasis, LETTADMROptions, letta_dmrg,
    reduced_local_problem,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_one_site_opt.reduced_mpo_compile import SpinTensorMPO, operator_basis
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain
from pyqed._letta_one_site_opt.reduced_contraction import (
    CanonicalEnvironmentChain, expand_reduced_mps_site, reduce_expanded_mps_site,
)
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
from test_letta_qchem import integrals, determinant_hamiltonian
from test_letta_qchem_symmetry import dense_mpo


@pytest.mark.parametrize('basis', [ReducedPhysicalBasis.spatial_orbital(), ReducedPhysicalBasis.spin_half()])
def test_local_operator_basis_is_complete_and_orthonormal(basis):
    op, channels, u = operator_basis(basis)
    assert op.dense_dim == basis.dense_dim**2
    np.testing.assert_allclose(u.T.conj()@u, np.eye(len(u)), atol=1e-14)
    assert any(c.sector[1].two_j > 0 for c in channels)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_native_compiler_preserves_independent_chemistry_operator(n):
    p = ElectronicProblem(*integrals(n), (1, 0), .37)
    h = p.su2_mpo().native_mpo(ReducedPhysicalBasis.spatial_orbital())
    np.testing.assert_allclose(dense_mpo(h.component_factors()),
                               determinant_hamiltonian(p), atol=2e-11)
    assert h.relative_reconstruction_error < 1e-12


def test_compiler_rejects_spin_and_charge_breaking_operators():
    basis = ReducedPhysicalBasis.spatial_orbital()
    for matrix in [np.diag([0., 1., -1., 0.]), np.eye(4, k=1)]:
        with pytest.raises(ValueError, match='not a charge-conserving SU'):
            SpinTensorMPO.compile((matrix[None, None],), basis)


@pytest.mark.parametrize('two_s,nelec', [(0, (1, 1)), (1, (2, 1)), (2, (2, 0))])
@pytest.mark.parametrize('ties', ['none', 'nn', 'carried'])
def test_native_random_actions_match_component_reference(two_s, nelec, ties):
    n = 3
    p = ElectronicProblem(*integrals(n), nelec, .23)
    neighborhoods = (tuple((i,) for i in range(n)) if ties == 'none' else
        tie_neighborhoods(n, [(0, n-1)] if ties == 'carried' else [],
                          nearest=True, carry=ties == 'carried'))
    s = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2', two_s=two_s),
        neighborhoods=neighborhoods, real=False, multiplets_per_sector=2, seed=474)
    h = p.su2_mpo()
    sites = ReducedFrontier.from_state(s).to_mps(s)
    compiled = h.native_mpo(s.physical_basis)
    native = ReducedEnvironmentChain.build(sites, compiled)
    # The compiler is independently checked against fermionic action above.
    # Use its compressed component view to isolate contraction conventions.
    component_factors = compiled.component_factors()
    components = CanonicalEnvironmentChain.build(sites, component_factors)
    np.testing.assert_allclose(native.expectation(), components.expectation(), atol=2e-11)
    rng = np.random.default_rng(191)
    for i, a in enumerate(sites):
        trial = a.copy()
        trial.data = {key: rng.normal(size=b.shape)+1j*rng.normal(size=b.shape)
                      for key, b in a.data.items()}
        ref = reduce_expanded_mps_site(a, components.local_action(i, expand_reduced_mps_site(trial)))
        actual = native.local_action(i, trial.data)
        for key in ref:
            np.testing.assert_allclose(actual[key], ref[key], atol=2e-10)
    oracle = replace(h, canonical_factors=component_factors, contraction_backend='components')
    for i in range(n-1):
        a = reduced_pair_problem(s, h, i, matrix_free=True, dense_solver_threshold=0)
        b = reduced_pair_problem(s, oracle, i, matrix_free=True, dense_solver_threshold=0)
        v = rng.normal(size=a.local_dimension)+1j*rng.normal(size=a.local_dimension)
        np.testing.assert_allclose(a.apply_hamiltonian(v), b.apply_hamiltonian(v), atol=2e-10)
        np.testing.assert_allclose(a.apply_metric(v), b.apply_metric(v), atol=2e-10)


@pytest.mark.parametrize('two_site', [False, True])
def test_complete_sweeps_do_not_expand_magnetic_mps_or_global_states(monkeypatch, two_site):
    import pyqed._letta_one_site_opt.reduced_contraction as component
    import pyqed._letta_one_site_opt.reduced_solver as one
    import pyqed._letta_two_site_opt.reduced_solver as two
    n = 3
    h = np.diag([-1., -.5], 1)+np.diag([-1., -.5], -1)
    g = np.zeros((n,)*4)
    for i in range(n):
        g[i, i, i, i] = 2.
    p = ElectronicProblem(h, g, (2, 1), .17)
    sym = p.symmetry('su2')
    s = ReducedLatticeLETTA.random((1, n), symmetry=sym, seed=53, multiplets_per_sector=2)
    from pyscf.fci import direct_spin1
    exact = direct_spin1.kernel(h, g, n, p.nelec, ecore=p.ecore)[0]
    def forbidden(*args, **kwargs):
        raise AssertionError('magnetic/global expansion entered a native sweep')
    monkeypatch.setattr(component, 'expand_reduced_mps_site', forbidden)
    monkeypatch.setattr(one, 'expand_reduced_mps_site', forbidden)
    monkeypatch.setattr(two, '_expand_pair_blocks', forbidden)
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    if two_site:
        result = letta_two_site_dmrg(p.su2_mpo(), state=s, bond_dim=8,
            options=LETTATwoSiteOptions(max_sweeps=4, gauge_mode='frontier', split_method='conditional-svd',
                reduced_sector_growth=True, dense_solver_threshold=1))
    else:
        result = letta_dmrg(p.su2_mpo(), state=s,
            options=LETTADMROptions(max_sweeps=5, gauge_mode='frontier', dense_solver_threshold=1))
    assert result.energy == pytest.approx(exact, abs=2e-9)
    assert result.state.norm() == pytest.approx(1., abs=1e-11)


def test_moving_boundaries_match_rebuild_after_adjacent_complex_updates():
    from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
    n = 4
    p = ElectronicProblem(*integrals(n), (2, 2))
    s = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2'), real=False, seed=241)
    sites = list(ReducedFrontier.from_state(s).to_mps(s))
    mpo = p.su2_mpo().native_mpo(s.physical_basis)
    h, norm = ReducedEnvironmentChain.build(sites, mpo), ReducedNormChain.build(sites)
    rng = np.random.default_rng(257)
    for i in (0, 1, 2, 3, 2, 1, 0):
        changed = (i, i+1) if i+1 < n else (i,)
        updates = {}
        for j in changed:
            a = sites[j].copy()
            a.data = {k: v*(.7+.2j)+.01*rng.normal(size=v.shape) for k, v in a.data.items()}
            sites[j] = updates[j] = a
        h.replace_sites(updates); norm.replace_sites(updates)
        fresh_h = ReducedEnvironmentChain.build(sites, mpo)
        fresh_n = ReducedNormChain.build(sites)
        for cache, reference in [(h, fresh_h), (norm, fresh_n)]:
            actual, expected = cache.local_action(i, sites[i].data), reference.local_action(i, sites[i].data)
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], atol=2e-12)
    np.testing.assert_allclose(h.expectation(), fresh_h.expectation(), atol=2e-12)
    np.testing.assert_allclose(norm.expectation(), fresh_n.expectation(), atol=2e-12)


def test_matrix_free_six_orbital_sweep_rejects_metric_null_directions():
    n = 6
    h = -np.eye(n, k=1)-np.eye(n, k=-1)
    g = np.zeros((n,)*4)
    for i in range(n):
        g[i, i, i, i] = 4.
    p = ElectronicProblem(h, g, (3, 3))
    s = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(n)), multiplets_per_sector=3, seed=71)
    mpo = p.su2_mpo()
    result = letta_dmrg(mpo, state=s, options=LETTADMROptions(
        max_sweeps=2, gauge_mode='frontier', matrix_free=True, dense_solver_threshold=32))
    from pyscf import fci
    exact = fci.direct_spin1.kernel(h, g, n, p.nelec, verbose=0)[0]
    assert exact-1e-9 <= result.energy < -3.05
    energies = [u.energy for sweep in result.history for u in sweep.updates]
    assert min(energies) >= exact-1e-9
    assert max(np.diff(energies)) < 1e-8
    sites = ReducedFrontier.from_state(result.state).to_mps(result.state)
    reference = CanonicalEnvironmentChain.build(sites, mpo.canonical_factors).stable_expectation()
    assert result.energy == pytest.approx(reference/result.state.norm(), abs=1e-9)

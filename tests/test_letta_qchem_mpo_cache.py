"""Exact chemistry compilation and optional persistent reuse, on tiny systems."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.qchem_mpo_cache import _cache_key
from test_letta_qchem import integrals, determinant_hamiltonian
from test_letta_qchem_symmetry import dense_mpo


def problem(n=3, offset=.37):
    h, g = integrals(n)
    return ElectronicProblem(h, g, (n//2, n//2), offset)


@pytest.mark.parametrize('n', [2, 3, 4])
def test_streamed_compiler_matches_independent_operator_without_dense_cores(n, monkeypatch):
    from pyqed.mps.nonabelian.mpo import RankCoupledMPO
    p = problem(n)
    ham = p.su2_mpo(cache_dir=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('whole expanded builder core must not be materialized')
    monkeypatch.setattr(RankCoupledMPO, 'as_dense', forbidden)
    native = ham.native_mpo(p.symmetry('su2').physical_basis)
    np.testing.assert_allclose(dense_mpo(native.component_factors()),
                               determinant_hamiltonian(p), atol=2e-11, rtol=0.)
    assert all(not getattr(w, '_block_cache', {}) for w in ham.canonical_factors)


def test_cache_hit_skips_builder_and_preserves_offset(tmp_path, monkeypatch):
    p = problem()
    cold = p.su2_mpo(cache_dir=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('cache hit rebuilt Hamiltonian')
    monkeypatch.setattr(ElectronicProblem, '_build_su2_mpo', forbidden)
    warm = p.su2_mpo(cache_dir=tmp_path)
    expected = determinant_hamiltonian(p)
    for ham in (cold, warm):
        np.testing.assert_allclose(dense_mpo(ham.canonical_factors), expected, atol=2e-12)
        basis = p.symmetry('su2').physical_basis
        assert basis in ham._native_mpo_cache
        assert all(hasattr(w, 'component_shape') for w in ham.canonical_factors)
    assert len(list(tmp_path.glob('*.pkl'))) == 1
    assert not list(tmp_path.glob('*.tmp'))


def test_key_separates_inputs_and_implementation(monkeypatch):
    p = problem(); key = _cache_key(p, 0.)
    assert _cache_key(problem(offset=.38), 0.) != key
    assert _cache_key(p, 1e-12) != key
    assert _cache_key(p.reordered([2, 1, 0]), 0.) != key
    h = p.h1.copy(); h[0, 0] += .01
    assert _cache_key(ElectronicProblem(h, p.eri, p.nelec, p.ecore), 0.) != key
    from pyqed._letta_one_site_opt import qchem_mpo_cache
    monkeypatch.setattr(qchem_mpo_cache, '_implementation_signature', lambda: 'new compiler')
    assert _cache_key(p, 0.) != key


def test_cache_default_and_explicit_disable(tmp_path, monkeypatch):
    monkeypatch.delenv('PYQED_QC_MPO_CACHE', raising=False)
    p = problem(2)
    assert not hasattr(p.su2_mpo(), '_native_mpo_cache')
    monkeypatch.setenv('PYQED_QC_MPO_CACHE', str(tmp_path))
    p.su2_mpo(cache_dir=False)
    assert not list(tmp_path.iterdir())
    p.su2_mpo()
    assert len(list(tmp_path.glob('*.pkl'))) == 1


def test_cached_and_uncached_optimized_energies_match(tmp_path, monkeypatch):
    from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
    p = problem(2)
    energies = []
    for cache in ('', str(tmp_path)):
        monkeypatch.setenv('PYQED_QC_MPO_CACHE', cache)
        job = QCLETTA(p, 4, seed=71)
        job.two_site(2).polish(2)
        job.tie([(0, 1)])
        job.optimize_letta(2)
        energies.append(job.energy)
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-11, rtol=0.)


def test_cached_cpp_dmrg_matches_uncached(tmp_path, monkeypatch):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA, ReducedFrontier
    from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG
    p = problem(2)
    state = ReducedLatticeLETTA.random((1, 2), symmetry=p.symmetry('su2'),
        neighborhoods=((0,), (1,)), multiplets_per_sector=1, seed=71, real=True)
    sites = ReducedFrontier.from_state(state).to_mps(state)
    energies = []
    for cache in ('', str(tmp_path)):
        monkeypatch.setenv('PYQED_QC_MPO_CACHE', cache)
        solve = ReducedDMRG(p, sites, max_bond=4).run(max_sweeps=2)
        energies.append([row['energy'] for row in solve.history])
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-11, rtol=0.)


def test_cache_separates_construction_options(tmp_path, monkeypatch):
    p = problem(3)
    expected = determinant_hamiltonian(p)
    keys = set()
    for storage in ('dense', 'sparse'):
        for layout in ('prefix', 'balanced'):
            monkeypatch.setenv('PYQED_QC_SYMBOLIC_STORAGE', storage)
            monkeypatch.setenv('PYQED_QC_AUTOMPO_LAYOUT', layout)
            keys.add(_cache_key(p, 0.))
            ham = p.su2_mpo(cache_dir=tmp_path)
            np.testing.assert_allclose(dense_mpo(ham.canonical_factors),
                                       expected, atol=2e-11, rtol=0.)
    assert len(keys) == 4
    assert len(list(tmp_path.glob('*.pkl'))) == 4

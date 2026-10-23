import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions
from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext
from pyqed._letta_one_site_opt.qchem import ElectronicProblem


def coordinates(nelec=(2, 1), site=1):
    h = np.array([[.1, -.7, .2], [-.7, .2, -.4], [.2, -.4, .3]])
    g = np.zeros((3,)*4)
    for i in range(3):
        g[i,i,i,i] = 4.
    electronic = ElectronicProblem(h, g, nelec)
    state = ReducedLatticeLETTA.random((1,3), symmetry=electronic.symmetry('su2'),
        multiplets_per_sector=2, real=False, seed=43,
        neighborhoods=((0,1,2), (1,2), (2,)))
    options = LETTADMROptions(matrix_free=True, gauge_mode='none', simplify_fixed_ties=True)
    problem = DependencySweepContext(state, electronic.su2_mpo()).local_problem(site, options)
    c = problem.coordinates
    c.prepare(options.metric_tolerance)
    return c


@pytest.mark.parametrize('nelec', [(1,1), (2,1), (2,0)])
@pytest.mark.parametrize('site', [0,1,2])
def test_streamed_compact_operator(nelec, site, monkeypatch):
    from pyqed._letta_one_site_opt.compact_hamiltonian import CompactHamiltonian
    monkeypatch.setenv('PYQED_QR_ABSORBED_MB', '0')
    monkeypatch.setenv('PYQED_COMPACT_H_MB', '0')
    c = coordinates(nelec, site)
    reference = np.column_stack([c.apply(v) for v in np.eye(c.rank)])
    compact = CompactHamiltonian(c, budget=16*1024**2, workspace=1024**2)
    np.testing.assert_allclose(compact.matrix, reference, atol=3e-12, rtol=3e-12)
    z = np.random.default_rng(61).normal(size=c.rank)+1j*np.random.default_rng(63).normal(size=c.rank)
    np.testing.assert_allclose(compact(z), reference@z, atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(compact.matrix), np.linalg.eigvalsh(reference), atol=3e-12)
    diagonal = CompactHamiltonian(c, budget=16*1024**2, workspace=1024**2, diagonal=True)
    for block, m in zip(diagonal.blocks, c.maps):
        np.testing.assert_allclose(block, reference[m[1],m[1]], atol=3e-12, rtol=3e-12)
    with pytest.raises(MemoryError):
        CompactHamiltonian(c, budget=0)
    assert compact.report['matrix_bytes'] == compact.matrix.nbytes
    assert compact.report['boundary_cache_bytes'] <= 1024**2
    from pyqed._letta_one_site_opt.compact_hamiltonian import AssemblyWorkLimit
    with pytest.raises(AssemblyWorkLimit) as caught:
        CompactHamiltonian(c,max_equivalent_actions=1e-30)
    assert caught.value.report['equivalent_actions']>0


def test_production_compact_and_preconditioner(monkeypatch):
    c = coordinates()
    z = np.random.default_rng(9).normal(size=c.rank)
    monkeypatch.setenv('PYQED_COMPACT_H_MB', '0')
    reference = c.apply(z)
    monkeypatch.setenv('PYQED_COMPACT_H_MB', '16')
    monkeypatch.setenv('PYQED_COMPACT_H_MAX_MAC', '0')
    np.testing.assert_allclose(c.apply(z), reference, atol=4e-12, rtol=4e-12)
    assert c.compact_report['selected']
    monkeypatch.setenv('PYQED_QR_ABSORBED_MB', '0')
    actual = c.preconditioned_residual(z, -2.)
    assert c.preconditioner_report['selected']
    assert np.all(np.isfinite(actual))

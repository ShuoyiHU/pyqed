import numpy as np
import pytest
from test_letta_compact_hamiltonian import coordinates


def test_pair_raw_environment_is_lazy_and_metric_root_reused(monkeypatch):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root
    from pyqed._letta_one_site_opt import frontier_contraction
    h=np.array([[0.,-.7,.2],[-.7,0.,-.4],[.2,-.4,0.]])
    problem=ElectronicProblem(h,np.zeros((3,)*4),(2,1))
    state=ReducedLatticeLETTA.random((1,3),symmetry=problem.symmetry('su2'),
        multiplets_per_sector=2,real=False,seed=8,
        neighborhoods=((0,1,2),(1,2),(2,)))
    original=frontier_contraction.raw_pair_environments
    calls=[]
    def measured(*a,**kw):
        calls.append(1)
        return original(*a,**kw)
    monkeypatch.setattr(frontier_contraction,'raw_pair_environments',measured)
    from pyqed._letta_two_site_opt.pair_frontier import ReducedPairFrontier
    monkeypatch.setattr(ReducedPairFrontier,'requires_compact',lambda *args: True)
    p=reduced_pair_problem(state,problem.su2_mpo(),0,matrix_free=True,dense_solver_threshold=0)
    assert not calls
    root=pair_metric_root(p,state,1e-12)
    assert pair_metric_root(p,state,1e-12) is root
    p.apply_metric(p.old_vector)
    p.apply_hamiltonian(p.old_vector)
    assert len(calls)==1
    assert pair_metric_root(p,state,1e-10) is not root


def test_compact_preconditioner_independent_of_qr_cache(monkeypatch):
    monkeypatch.setenv('PYQED_COMPACT_H_MB','0')
    monkeypatch.setenv('PYQED_QR_ABSORBED_MB','0')
    c=coordinates()
    c.preconditioned_residual(np.ones(c.rank),-1.)
    assert c.preconditioner_report['selected']
    assert c._absorbed_action is None


def test_stationarity_reuses_exteriors_but_checks_changed_state():
    from test_letta_reduced_compression import example
    from test_letta_qchem import integrals
    from pyqed._letta_one_site_opt import LETTADMROptions
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext
    from pyqed._letta_one_site_opt.reduced_updates import reduced_stationarity
    state,*_=example(((0,2),(1,0),(2,1)))
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    options=LETTADMROptions(gauge_mode='none')
    context=DependencySweepContext(state,ham,structured=False)
    for change in (False,True):
        if change:
            key=next(iter(state.tensors[1]))
            state.tensors[1][key]*=1.2+.1j
            context.synchronize([1])
        cached=reduced_stationarity(state,ham,(0,1),options,context=context)
        fresh=reduced_stationarity(state,ham,(0,1),options)
        for a,b in zip(cached,fresh):
            np.testing.assert_allclose(a['relative_residual'],b['relative_residual'],atol=2e-11,rtol=2e-11)
    with pytest.raises(ValueError,match='must own'):
        reduced_stationarity(state.copy(),ham,(0,),options,context=context)


@pytest.mark.parametrize('structured',[False,True])
def test_energy_certificate_keeps_the_preparation_cache_mode(structured):
    from test_letta_reduced_compression import example
    from test_letta_qchem import integrals
    from pyqed._letta_one_site_opt import LETTADMROptions
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    context=DependencySweepContext(state,ham,structured=structured)
    context.local_problem(1,LETTADMROptions(gauge_mode='none'))
    old=dict(context.boundary_cache.entries)
    assert old
    actual=context.stable_energy()
    assert context.boundary_cache.structured==structured
    assert any(context.boundary_cache.entries.get(k) is v for k,v in old.items())
    assert actual==pytest.approx(_energy(state,ham,stable=True),abs=2e-11)

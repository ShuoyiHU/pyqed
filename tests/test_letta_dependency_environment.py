import numpy as np
import pytest
from pyqed._letta_one_site_opt.dependency_environment import fixed_dependency_view,DependencySweepContext
from pyqed._letta_one_site_opt import LETTADMROptions
from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext,_energy


def setup():
    from test_letta_pair_coordinates import setup_pair
    return setup_pair(1.)[:2]


def test_exact_only_and_active_untouched():
    from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import integrals,initial_mps
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    h,g,edges=integrals(2,3,dict(U=4.,t=1.,t_diagonal=0.))
    p=ElectronicProblem(h,g,(3,3));sites,_=initial_mps(6,6,0,10,'multiplets',4)
    state=ReducedLatticeLETTA.from_mps(sites,symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,)+tuple(j for a,j in edges if a==i) for i in range(6)))
    view,removed=fixed_dependency_view(state,0)
    assert removed and view.neighborhoods[0]==state.neighborhoods[0]
    assert all(i!=0 for i,j in removed)
    assert state.neighborhoods[1]!=view.neighborhoods[1]
    # Even a one-ULP variation is a real dependency, not rounded away.
    key=next(iter(state.tensors[1]));a=state.tensors[1][key]
    index=(0,0,1)+(0,)*(a.ndim-3)
    a[index]=np.nextafter(a[index],np.inf)
    view,removed=fixed_dependency_view(state,0)
    assert (1,state.neighborhoods[1][1]) not in removed


@pytest.mark.parametrize('active',[0,1,2])
def test_action_and_cache_follow_updated_dependencies(active):
    state,ham=setup();opts=LETTADMROptions(gauge_mode='none',simplify_fixed_ties=True,dense_solver_threshold=1)
    context=DependencySweepContext(state,ham);rng=np.random.default_rng(45)
    for iteration in range(2):
        a=ReducedSweepContext(state,ham,prune=False).local_problem(active,opts)
        b=context.local_problem(active,opts)
        x=rng.normal(size=a.embedding.source_size)+1j*rng.normal(size=a.embedding.source_size)
        assert a.embedding.source_layout.shapes == b.embedding.source_layout.shapes
        supported=b.embedding.supported_source_indices
        if supported is None:
            supported=np.arange(b.embedding.source_size)
        for field in ['hamiltonian_action','metric_action']:
            expected=getattr(a,field)(x)
            actual=np.zeros_like(expected)
            actual[supported]=getattr(b,field)(x[supported])
            np.testing.assert_allclose(expected,actual,rtol=2e-11,atol=2e-11)
        other=(active+1)%state.nsites
        for k,v in state.tensors[other].items():state.tensors[other][k]=v*(1+.01*rng.normal(size=v.shape))
        context.synchronize([other])
    assert abs(context.stable_energy()-_energy(state,ham,stable=True))<1e-12


def test_requires_explicit_local_gauge_path():
    from pyqed._letta_one_site_opt import letta_dmrg
    state,ham=setup()
    with pytest.raises(ValueError,match='gauge_mode'):
        letta_dmrg(ham,state=state,options=LETTADMROptions(simplify_fixed_ties=True))


def test_frontier_gauge_sweep_matches_unpruned_context(monkeypatch):
    from pyqed._letta_one_site_opt import letta_dmrg
    from pyqed._letta_one_site_opt import reduced_solver as solver
    state,ham=setup()
    options=LETTADMROptions(max_sweeps=2,tolerance=1e-12,gauge_mode='frontier',
        dense_solver_threshold=1,eigensolver_tolerance=1e-10)
    compact=letta_dmrg(ham,state=state,options=options)
    class CartesianContext(ReducedSweepContext):
        def __init__(self,state,hamiltonian):
            super().__init__(state,hamiltonian,prune=False)
    monkeypatch.setattr(solver,'ReducedSweepContext',CartesianContext)
    reference=letta_dmrg(ham,state=state,options=options)
    assert len(compact.history)==len(reference.history)==2
    assert any(a.local_dimension < b.local_dimension
               for sa,sb in zip(compact.history,reference.history)
               for a,b in zip(sa.updates,sb.updates))
    assert all(u.accepted and u.local_converged and u.recovery_reason is None
               for result in (compact,reference) for sweep in result.history for u in sweep.updates)
    np.testing.assert_allclose([s.energy for s in compact.history],
                               [s.energy for s in reference.history],rtol=2e-10,atol=2e-10)
    assert compact.energy==pytest.approx(reference.energy,abs=2e-10)


def test_sweep_context_rebuilds_after_sector_layout_growth():
    from pyqed._letta_two_site_opt.reduced_solver import _expand_reduced_pair_space
    from test_letta_nonzero_spin import setup as setup_random
    problem,state=setup_random(1,True)
    ham=problem.su2_mpo()
    context=ReducedSweepContext(state,ham)
    previous=context.frontier
    grown=_expand_reduced_pair_space(state,1,2)
    assert grown.bond_sectors != state.bond_sectors
    state.tensors,state.bond_sectors=grown.tensors,grown.bond_sectors
    context.synchronize([1,2])
    assert context.frontier is not previous
    fresh=ReducedSweepContext(state,ham)
    for got,expected in zip(context.embeddings,fresh.embeddings):
        np.testing.assert_array_equal(got.source_indices,expected.source_indices)
        np.testing.assert_array_equal(got.target_indices,expected.target_indices)
    options=LETTADMROptions(gauge_mode='none',dense_solver_threshold=1)
    old=context.local_problem(0,options)
    new=fresh.local_problem(0,options)
    x=np.random.default_rng(93).normal(size=old.embedding.source_size)
    for field in ['hamiltonian_action','metric_action']:
        np.testing.assert_allclose(getattr(old,field)(x),getattr(new,field)(x),rtol=2e-11,atol=2e-11)

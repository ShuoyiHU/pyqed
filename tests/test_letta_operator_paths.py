import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_environment import ReducedTransfer
from pyqed._letta_one_site_opt.operator_paths import prune_transfers


def test_conditional_budget_returns_exact_unfiltered_input():
    from types import SimpleNamespace
    from pyqed._letta_one_site_opt.operator_paths import prune_conditioned_transfers
    sites=[];transfers=[]
    for i in range(2):
        key=(i,'physical',i+1)
        support=(((*key,0,(),(),(0,)),),{i:((),)},{i+1:((),)},False)
        sites.append(SimpleNamespace(metadata={'letta_operator_support':support}))
        transfers.append((ReducedTransfer(key,key,i,i+1,np.ones((1,1,1,1))),))
    transfers=tuple(transfers)
    actual,report=prune_conditioned_transfers(sites,transfers,{0},{2},max_edges=1)
    assert actual is transfers
    assert report['conditional_fallback']=='edge budget exceeded'
    actual,report=prune_conditioned_transfers(sites,transfers,{0},{2},max_edges=10)
    assert report['conditional_routes_removed']==0
    assert actual==transfers


def test_coarse_cache_ignores_pending_labels_but_fine_cache_does_not(monkeypatch):
    from test_letta_reduced_compression import example
    from pyqed._letta_one_site_opt import ReducedFrontier
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_environment import chain_transfers
    from test_letta_qchem import integrals
    state,*_=example()
    sites=ReducedFrontier.from_state(state).to_mps(state,compact=True)
    mpo=ElectronicProblem(*integrals(3),(2,1)).su2_mpo().native_mpo(state.physical_basis)
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES','0')
    first,key,_=chain_transfers(sites,mpo)
    sites[1].metadata['letta_frontier_support']='changed label basis'
    second,newkey,_=chain_transfers(sites,mpo)
    assert first is second and key==newkey
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES','100000')
    _,key,_=chain_transfers(sites,mpo)
    sites[1].metadata['letta_frontier_support']='another label basis'
    assert chain_transfers(sites,mpo)[1]!=key


def test_requires_both_boundaries_and_keeps_physical_transitions():
    edge=lambda l,r,a: ReducedTransfer(('bra',),('ket',),l,r,np.asarray(a)[:,:,None,None])
    source=(0,0,0);middle=(1,1,1);target=(2,2,2);dead=(3,3,3)
    ts=((edge(source,middle,[[1.,2.,0.]]),edge(dead,middle,[[3.,0.,0.]])),
        (edge(middle,target,[[4.],[0.],[5.]]),edge(middle,dead,[[0.],[6.],[0.]])))
    result,report=prune_transfers(ts,{source},{target})
    assert report['removed_channel_routes']==4
    assert len(result[0])==len(result[1])==1
    np.testing.assert_array_equal(result[0][0].kernel[:,:,0,0],[[1.,0.,0.]])
    np.testing.assert_array_equal(result[1][0].kernel[:,:,0,0],[[4.],[0.],[0.]])


@pytest.mark.parametrize('pending',[0,100000])
def test_reduced_local_action_matches_unpruned(monkeypatch,pending):
    from test_letta_compact_hamiltonian import coordinates
    rng=np.random.default_rng(39)
    monkeypatch.setenv('PYQED_OPERATOR_PATHS','0')
    old=coordinates()
    monkeypatch.setenv('PYQED_OPERATOR_PATHS','1')
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES',str(pending))
    new=coordinates()
    x=rng.normal(size=old.rank)+1j*rng.normal(size=old.rank)
    np.testing.assert_allclose(new.apply(x),old.apply(x),rtol=2e-12,atol=2e-12)


def test_conditioned_pair_keeps_free_pair_domain(monkeypatch):
    from test_letta_reduced_compression import example
    from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
    from pyqed._letta_two_site_opt.reduced_solver import _orthogonal_pair_action
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from test_letta_qchem import integrals
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES','0')
    state,pair,*_=example(((0,2),(1,0),(2,1)))
    root=ReducedPairMetricRoot(pair,state,1e-12)
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    old=_orthogonal_pair_action(pair,root,state,ham)
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES','100000')
    # Fresh problem and boundaries; an arbitrary free pair is not a product
    # of the two legal tied factors.
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    p2=reduced_pair_problem(state,ham,pair.left_site,matrix_free=True,dense_solver_threshold=0)
    r2=ReducedPairMetricRoot(p2,state,1e-12)
    new=_orthogonal_pair_action(p2,r2,state,ham)
    x=np.random.default_rng(420).normal(size=root.size)+1j*np.random.default_rng(422).normal(size=root.size)
    np.testing.assert_allclose(new(x),old(x),atol=2e-11,rtol=2e-11)

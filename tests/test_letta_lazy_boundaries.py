"""Lazy reduced boundaries preserve eager arithmetic and invalidation."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain

@pytest.mark.parametrize('nelec', [(1,1), (2,1)])
def test_lazy_boundaries_match_eager_before_and_after_updates(nelec):
    h=np.diag([.2,-.1,.3]);h[0,1]=h[1,0]=-.8;h[1,2]=h[2,1]=-.4
    g=np.zeros((3,)*4)
    for i in range(3):g[i,i,i,i]=4.
    problem=ElectronicProblem(h,g,nelec)
    state=ReducedLatticeLETTA.random((1,3),symmetry=problem.symmetry('su2'),
        neighborhoods=((0,1,2),(1,2),(2,)),multiplets_per_sector=2,real=False,seed=19)
    sites=ReducedFrontier.from_state(state).to_mps(state)
    mpo=problem.su2_mpo().native_mpo(state.physical_basis)
    eager=ReducedEnvironmentChain.build(sites,mpo)
    lazy=ReducedEnvironmentChain.build(sites,mpo,lazy=True)
    assert lazy.left_valid==0 and lazy.right_valid==3
    for site in [1,0,2]:
        expected=eager.local_action(site,sites[site].data)
        actual=lazy.local_action(site,sites[site].data)
        prepared=lazy.prepare_local_action(site)
        for blocks in (sites[site].data, {k:(1.+.3j)*a for k,a in sites[site].data.items()}):
            bound=prepared(blocks); direct=lazy.local_action(site,blocks)
            for key in direct:np.testing.assert_array_equal(bound[key],direct[key])
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    assert lazy.expectation()==eager.expectation()
    changed=sites[1].copy()
    changed.data={key:(.7+.2j)*a for key,a in changed.data.items()}
    for chain in [eager,lazy]:chain.replace_sites({1:changed})
    for site in [2,0,1]:
        blocks=changed.data if site==1 else sites[site].data
        expected=eager.local_action(site,blocks);actual=lazy.local_action(site,blocks)
        for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    assert lazy.expectation()==eager.expectation()

import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache,canonical_environment

@pytest.mark.parametrize('complex_values',[False,True])
def test_cache_matches_rebuild_after_inplace_changes(complex_values):
 state,ham,_=setup_pair(1.)
 if complex_values:
  for blocks in state.tensors:
   for k,a in blocks.items():blocks[k]=a*(1+.2j)
 mpo=ham.native_mpo(state.physical_basis);cache=CanonicalBoundaryCache()
 for step,center in enumerate(list(range(state.nsites))+list(reversed(range(state.nsites)))):
  sites=ReducedFrontier.from_state(state).to_mps(state,compact=True)
  a,l,r,h=canonical_environment(sites,center,mpo,cache=cache)
  ar,lr,rr,hr=canonical_environment(sites,center,mpo)
  for lhs,rhs in ((l,lr),(r,rr),(a[center].data,ar[center].data)):
   assert lhs.keys()==rhs.keys()
   for k in lhs:np.testing.assert_array_equal(lhs[k],rhs[k])
  x=h.local_action(center,a[center].data);y=hr.local_action(center,ar[center].data)
  for k in x:np.testing.assert_array_equal(x[k],y[k])
  # Actual in-place change, including sites that were in a cached boundary.
  key=next(iter(state.tensors[center]));state.tensors[center][key]*=1.01
 assert cache.hits>0

def test_cache_budget_and_repeat():
 state,ham,_=setup_pair(1.);sites=ReducedFrontier.from_state(state).to_mps(state,compact=True);mpo=ham.native_mpo(state.physical_basis)
 cache=CanonicalBoundaryCache();canonical_environment(sites,1,mpo,cache=cache);n=cache.advances
 canonical_environment(sites,1,mpo,cache=cache);assert cache.advances==n
 cache=CanonicalBoundaryCache(max_bytes=1);canonical_environment(sites,1,mpo,cache=cache)
 assert cache.nbytes<=1 and not cache.entries

def test_energy_reuse_is_value_based(monkeypatch):
 from pyqed._letta_one_site_opt import reduced_solver as solver
 state,ham,_=setup_pair(1.);context=solver.ReducedSweepContext(state,ham)
 original=solver._energy;calls=[]
 def counted(*args,**kwargs):
  calls.append(1);return original(*args,**kwargs)
 monkeypatch.setattr(solver,'_energy',counted)
 expected=context.stable_energy()
 assert context.stable_energy(state.copy())==expected
 assert len(calls)==1
 key=next(iter(state.tensors[1]));state.tensors[1][key]*=1.01
 assert context.stable_energy()==original(state,ham,stable=True)
 assert len(calls)==2

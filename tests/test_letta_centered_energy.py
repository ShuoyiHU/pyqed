"""Two-sided QR expectation versus independently streamed expectation."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.frontier_contraction import centered_energy,frontier_energy,CanonicalBoundaryCache

@pytest.mark.parametrize('nelec',[(1,1),(2,1),(2,0)])
@pytest.mark.parametrize('budget',[0,16*1024**2])
def test_centered_energy_matches_streamed_after_updates(nelec,budget):
 h=np.diag([.1,.2,.3]);h[0,1]=h[1,0]=-.7;h[0,2]=h[2,0]=.2
 g=np.zeros((3,)*4)
 for i in range(3):g[i,i,i,i]=4.
 p=ElectronicProblem(h,g,nelec)
 state=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=43,neighborhoods=((0,1,2),(1,2),(2,)))
 mpo=p.su2_mpo().native_mpo(state.physical_basis)
 cache=CanonicalBoundaryCache(max_bytes=budget)
 for center in [0,1,2,1,0]:
  key=next(iter(state.tensors[center]));state.tensors[center][key]*=1.001+.002j
  expected=frontier_energy(state,mpo);actual=centered_energy(state,mpo,center,cache=cache)
  np.testing.assert_allclose(actual[0]/actual[1],expected[0]/expected[1],rtol=3e-11,atol=3e-11)
  assert cache.nbytes<=budget
 # Strong gauge rescaling must not change the answer or reuse stale values.
 for blocks,scale in zip(state.tensors,[1e5,1e-5,1.]):
  for key in blocks:blocks[key]*=scale
 expected=frontier_energy(state,mpo);actual=centered_energy(state,mpo,1,cache=cache)
 np.testing.assert_allclose(actual[0]/actual[1],expected[0]/expected[1],rtol=3e-11,atol=3e-11)

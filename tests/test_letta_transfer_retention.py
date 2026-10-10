import numpy as np
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_environment import cached_transfers,compile_transfers

def test_changed_layouts_do_not_accumulate_transfer_kernels():
 h=np.eye(3);g=np.zeros((3,)*4);g[0,0,0,0]=4
 p=ElectronicProblem(h,g,(2,1))
 state=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),multiplets_per_sector=2,seed=9)
 site=ReducedFrontier.from_state(state).to_mps(state)[1]
 mpo=p.su2_mpo().native_mpo(state.physical_basis)
 keys=list(site.data);assert len(keys)>2
 for shift in range(len(keys)):
  changed=site.copy();changed.data={k:site.data[k] for k in keys[shift:]+keys[:shift]}
  actual=cached_transfers(mpo,1,changed);expected=compile_transfers(changed,mpo.sites[1],mpo.channels)
  assert len(actual)==len(expected)
  for a,b in zip(actual,expected):
   assert (a.bra_key,a.ket_key,a.left_key,a.right_key)==(b.bra_key,b.ket_key,b.left_key,b.right_key)
   np.testing.assert_array_equal(a.kernel,b.kernel)
 assert sum(k[0]==1 for k in mpo._transfer_cache)<=2

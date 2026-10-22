import numpy as np
import pytest
from pyqed._letta_one_site_opt.numpy_contractions import accumulate_einsum,cached_einsum
from pyqed._letta_one_site_opt.frontier_contraction import contract_block,contract_boundary_block,ScatterBlock

@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('equation,shapes',[
 ('ybr,xyop,aob,lpr->xal',[(2,5,6),(2,2,2,2),(3,2,5),(4,2,6)]),
 ('xal,xyop,aob,lpr->ybr',[(2,3,4),(2,2,2,2),(3,2,5),(4,2,6)]),
 ('ij,jk->ik',[(3,4),(4,5)])])
def test_accumulated_final_product(equation,shapes,complex_values):
 rng=np.random.default_rng(172)
 arrays=[rng.normal(size=s)+(1j*rng.normal(size=s) if complex_values else 0) for s in shapes]
 reference=cached_einsum(equation,*arrays)/3
 value=accumulate_einsum(equation,arrays,alpha=1/3)
 np.testing.assert_allclose(value,reference,rtol=3e-13,atol=3e-13)
 for scale in [.7,-.2]:
  value=accumulate_einsum(equation,arrays,alpha=scale,out=value)
  reference=reference+scale*cached_einsum(equation,*arrays)
  np.testing.assert_allclose(value,reference,rtol=3e-13,atol=3e-13)

@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_boundary_product_matches_contraction(direction,complex_values):
 rng=np.random.default_rng(18);a=rng.normal(size=(6,2,7))
 if complex_values:a=a+1j*rng.normal(size=a.shape)
 a[2:]=0;indices=np.flatnonzero(a);block=ScatterBlock(a.shape,indices,a.ravel()[indices])
 kwargs={'left':rng.normal(size=(3,6))} if direction=='lr' else {'right':rng.normal(size=(4,7))}
 np.testing.assert_allclose(contract_boundary_block(block,**kwargs),contract_block(block,**kwargs),atol=2e-13,rtol=2e-13)

@pytest.mark.parametrize('workers',[1,2,4])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_environment_workers_match_reference(workers,direction,monkeypatch):
 from test_letta_pair_coordinates import setup_pair
 from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
 from pyqed._letta_one_site_opt.reduced_coordinates import _canonical_sites
 from pyqed._letta_one_site_opt import reduced_environment as e
 from pyqed._letta_one_site_opt.reduced_symmetry import _sector_irrep
 state,ham,_=setup_pair(1.)
 for blocks in state.tensors:
  for k,a in blocks.items():blocks[k]=a*(1+.2j)
 sites=ReducedFrontier.from_state(state).to_mps(state)
 mpo=ham.native_mpo(state.physical_basis)
 lr=direction=='lr';indices=range(len(sites)) if lr else reversed(range(len(sites)))
 end=0 if lr else len(sites)-1
 env=e._boundary(sites[end],mpo.sites[end],0 if lr else 2)
 monkeypatch.setenv('PYQED_ENVIRONMENT_THREADS',str(workers))
 for i in indices:
  transfers=e.cached_transfers(mpo,i,sites[i]);reference={}
  for t in transfers:
   incoming=env.get(t.left_key if lr else t.right_key)
   if incoming is None:continue
   value=cached_einsum('xal,xyop,aob,lpr->ybr' if lr else 'ybr,xyop,aob,lpr->xal',incoming,t.kernel,sites[i].data[t.bra_key].conj(),sites[i].data[t.ket_key])
   value/=_sector_irrep(t.bra_key[2 if lr else 0]).dim
   key=t.right_key if lr else t.left_key;reference[key]=reference.get(key,0.)+value
  actual=(e.advance_left if lr else e.advance_right)(env,sites[i],transfers)
  assert actual.keys()==reference.keys()
  for key in actual:np.testing.assert_allclose(actual[key],reference[key],atol=1e-10,rtol=2e-12)
  env=actual

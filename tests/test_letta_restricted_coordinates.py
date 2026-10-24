"""Dropping unused scatter columns must preserve the complete physical map."""
from copy import copy
import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates,_OverlapFactor

@pytest.mark.parametrize('complex_data',[False,True])
def test_restriction_compacts_only_unused_columns(complex_data):
 rng=np.random.default_rng(42)
 def random(shape):
  a=rng.normal(size=shape)
  return a+1j*rng.normal(size=shape) if complex_data else a
 c=ReducedLocalCoordinates.__new__(ReducedLocalCoordinates);c.dimension=6
 left,right=random((12,12)),random((20,20))
 outer_l=np.linalg.qr(random((25,12)),mode='reduced')[0]
 outer_r=np.linalg.qr(random((30,20)),mode='reduced')[0]
 source=np.array([0,1,2,3,0,4,5]);lm=np.array([0,1,0,1,1,6,7]);rm=np.array([1,1,2,2,2,8,9]);pm=np.array([0,0,0,0,0,1,1])
 f=_OverlapFactor(left,right,source,lm,pm,rm,2,6,np.sqrt(3.))
 c.blocks=[(np.arange(6),f)];c.images=[('sector',np.array([0,1]),outer_l,outer_r,2)]
 reference=copy(c)
 keep=source<4
 reference.blocks=[(np.arange(4),_OverlapFactor(left,right,source[keep],lm[keep],pm[keep],rm[keep],2,4,np.sqrt(3.)))]
 c.restrict(np.arange(4))
 assert c.blocks[0][1].left.shape==(2,2)
 assert c.blocks[0][1].right.shape==(2,2)
 assert c.blocks[0][1].nphysical==1
 c.prepare(1e-12);reference.prepare(1e-12)
 assert c.rank==reference.rank
 x=random((6,));x[4:]=0
 np.testing.assert_allclose(c.metric(x),reference.metric(x),rtol=2e-13,atol=2e-11)
 a=c.orthogonal_expand(c.coordinates(x));b=reference.orthogonal_expand(reference.coordinates(x))
 np.testing.assert_allclose(a['sector'],b['sector'],rtol=2e-12,atol=2e-12)
 z=random((c.rank,));y={'sector':random(a['sector'].shape)}
 np.testing.assert_allclose(np.vdot(c.orthogonal_expand(z)['sector'],y['sector']),np.vdot(z,c.orthogonal_adjoint(y)),atol=2e-12)


@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('ties',[None,((0,2),(1,0),(2,1))])
def test_constructor_restriction_preserves_physical_map(side,ties):
 from test_letta_reduced_compression import example
 from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root
 state,problem,*_=example(ties)
 root=pair_metric_root(problem,state,1e-12)
 sites=problem.frontier.to_mps(state,compact=True)
 emb=problem.frontier.site_embedding(state,1+side)
 kw=dict(retain_outer=False,boundary_seeds=root.boundary_seeds,factor_only=True)
 reference=ReducedLocalCoordinates(sites,emb,1+side,**kw)
 active=np.arange(reference.dimension)[::2]
 actual=ReducedLocalCoordinates(sites,emb,1+side,active_indices=active,**kw)
 # Select source entries in the original full geometry, without QR compaction.
 blocks=[];images=[]
 for (indices,f),image in zip(reference.blocks,reference.images):
  take=np.flatnonzero(np.isin(indices,active))
  if not len(take):continue
  inverse=np.full(len(indices),-1,dtype=int);inverse[take]=np.arange(len(take))
  keep=inverse[f.source]>=0
  blocks.append((indices[take],_OverlapFactor(f.left,f.right,inverse[f.source[keep]],
      f.lmap[keep],f.pmap[keep],f.rmap[keep],f.nphysical,len(take),f.weight)))
  images.append(image)
 reference.blocks,reference.images=blocks,images
 reference.prepare(1e-12);actual.prepare(1e-12)
 assert actual.rank==reference.rank
 rng=np.random.default_rng(71)
 x=rng.normal(size=reference.dimension)+1j*rng.normal(size=reference.dimension)
 x[np.setdiff1d(np.arange(reference.dimension),active)]=0
 np.testing.assert_allclose(actual.metric(x),reference.metric(x),atol=2e-12)
 a=actual.orthogonal_expand(actual.coordinates(x))
 b=reference.orthogonal_expand(reference.coordinates(x))
 assert a.keys()==b.keys()
 for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12)


@pytest.mark.parametrize('complex_data',[False,True])
@pytest.mark.parametrize('zero',[False,True])
def test_sparse_left_columns_keep_original_reduction(complex_data,zero):
 from pyqed._letta_one_site_opt.frontier_contraction import ScatterBlock,contract_block
 rng=np.random.default_rng(93)
 shape=(130,3,600);width=shape[1]*shape[2]
 indices=np.unique(rng.integers(0,shape[0],300)*width+rng.choice([0,7,100,777,1000],300))
 values=rng.normal(size=len(indices));left=rng.normal(size=(70,shape[0]))
 if complex_data:
  values=values+1j*rng.normal(size=len(indices));left=left+1j*rng.normal(size=left.shape)
 if zero:values[:]=0
 sparse=ScatterBlock(shape,indices,values)
 expected=np.einsum('kl,lpr->kpr',left,np.asarray(sparse))
 np.testing.assert_array_equal(contract_block(sparse,left=left),expected)

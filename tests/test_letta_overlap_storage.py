import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_coordinates import _OverlapFactor
@pytest.mark.parametrize('complex_data',[False,True])
def test_implicit_overlap_equals_dense_with_repeated_sources(complex_data):
 rng=np.random.default_rng(47)
 def values(shape):
  a=rng.normal(size=shape)
  return a+1j*rng.normal(size=shape) if complex_data else a
 left,right=values((7,5)),values((6,4))
 source=np.array([0,0,1,1,2,2,3,3]);l=np.array([0,1,1,2,2,3,3,4]);r=np.array([0,1,1,2,2,3,3,0]);p=np.array([0,1,0,1,0,1,0,1])
 f=_OverlapFactor(left,right,source,l,p,r,2,4,3**.5)
 a=f.dense();x=values((4,));y=values((84,))
 np.testing.assert_allclose(f.apply(x),a@x,rtol=3e-14,atol=3e-14)
 np.testing.assert_allclose(f.adjoint(y),a.conj().T@y,rtol=3e-14,atol=3e-14)
 assert f.stored_elements<a.size

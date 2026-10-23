import numpy as np
import pytest
from pyqed._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
from pyqed._letta_two_site_opt.pair_metric import PairMetricMap

@pytest.mark.parametrize('sizes',[(2,3,4,5,2,3),(5,4,2,3,3,2),(0,3,2,4,1,2),(3,0,2,4,1,2),(3,4,0,2,1,2),(3,4,2,0,1,2)])
@pytest.mark.parametrize('complex_factors',[False,True])
@pytest.mark.parametrize('complex_vector',[False,True])
def test_metric_map(sizes,complex_factors,complex_vector):
 l,r,a,b,p,q=sizes;rng=np.random.default_rng(7)
 def draw(shape,complex):
  x=rng.normal(size=shape)
  return x+1j*rng.normal(size=shape) if complex else x
 src=_BlockVectorLayout({0:(l,p,q,r),1:(2,1,1,3)})
 dst=_BlockVectorLayout({0:(a,p,q,b),1:(4,1,1,2)})
 factors={0:(draw((a,l),complex_factors),draw((b,r),complex_factors),.7),1:(draw((4,2),complex_factors),draw((2,3),complex_factors),1.5)}
 v=draw((src.size,),complex_vector)
 expected=dst.pack({k:s*np.einsum('al,lpqr,br->apqb',L,src.unpack(v)[k],R,optimize=True) for k,(L,R,s) in factors.items()})
 mapper=PairMetricMap(src,dst,factors)
 np.testing.assert_allclose(mapper(v),expected,rtol=2e-13,atol=2e-13)
 v.flags.writeable=False
 np.testing.assert_allclose(mapper(v),expected,rtol=2e-13,atol=2e-13)
 adj=PairMetricMap(dst,src,{k:(L.conj().T,R.conj().T,s) for k,(L,R,s) in factors.items()})
 w=draw((dst.size,),True)
 np.testing.assert_allclose(np.vdot(w,mapper(v)),np.vdot(adj(w),v),rtol=2e-13,atol=2e-13)


def test_empty_and_invalid():
 empty=_BlockVectorLayout({})
 assert PairMetricMap(empty,empty,{})(np.array([])).size==0
 with pytest.raises(ValueError):PairMetricMap(empty,empty,{})(np.ones(1))


def test_equal_fixed_factors_are_stored_once_without_changing_routes():
 rng=np.random.default_rng(19)
 left=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3))
 right=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
 src=_BlockVectorLayout({k:(3,1,2,2) for k in range(6)})
 dst=_BlockVectorLayout({k:(5,1,2,4) for k in range(6)})
 factors={k:(left.copy(),right.copy(),1.+k/10) for k in range(6)}
 mapper=PairMetricMap(src,dst,factors)
 assert mapper.factor_bytes==left.nbytes+right.nbytes
 assert mapper.unshared_factor_bytes==6*mapper.factor_bytes
 x=rng.normal(size=src.size)+1j*rng.normal(size=src.size)
 expected=dst.pack({k:s*np.einsum('al,lpqr,br->apqb',l,src.unpack(x)[k],r,optimize=True)
                    for k,(l,r,s) in factors.items()})
 np.testing.assert_allclose(mapper(x),expected,rtol=1e-13,atol=1e-13)
 factors[0][0][0,0]+=1e-8
 changed=PairMetricMap(src,dst,factors)
 assert changed.factor_bytes==2*left.nbytes+right.nbytes


def test_cancellation_preserves_original_contraction_accuracy():
 from pathlib import Path
 # Captured D30 Hubbard ALS residual; oracle evaluated at 70 decimal digits.
 data=np.load(Path(__file__).with_name('data')/'letta_metric_cancellation.npz')
 left,right,x,scale,expected=(data[k] for k in ('left','right','vector','scale','expected'))
 src=_BlockVectorLayout({0:x.shape});dst=_BlockVectorLayout({0:expected.shape})
 actual=PairMetricMap(src,dst,{0:(left,right,float(scale))})(x.ravel()).reshape(expected.shape)
 reference=float(scale)*np.einsum('al,br,lpqr->apqb',left,right,x,optimize=True)
 # Preserve the established double-precision error envelope despite cancellation.
 assert np.linalg.norm(actual-expected)<=1.25*np.linalg.norm(reference-expected)+2e-12

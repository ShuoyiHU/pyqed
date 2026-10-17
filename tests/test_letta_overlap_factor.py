import numpy as np
import pytest
from pyqed._letta_one_site_opt.overlap_factor import overlap_factor_matrix


@pytest.mark.parametrize('complex_factors',[False,True])
@pytest.mark.parametrize('empty',[False,True])
def test_overlap_factor_routes_accumulate_and_match_adjoint(complex_factors,empty):
    from pyqed._letta_one_site_opt.reduced_coordinates import _OverlapFactor
    rng=np.random.default_rng(93)
    def draw(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_factors else x
    left,right=draw((5,3)),draw((4,2))
    count=0 if empty else 25
    source=rng.integers(0,7,count);l=rng.integers(0,3,count)
    p=rng.integers(0,2,count);r=rng.integers(0,2,count)
    factor=_OverlapFactor(left,right,source,l,p,r,2,7,np.sqrt(3))
    expected=np.zeros((5,2,4,7),dtype=np.result_type(left,right))
    for col,li,pi,ri in zip(source,l,p,r):
        expected[:,pi,:,col]+=factor.weight*left[:,li,None]*right[None,:,ri]
    actual=factor.dense()
    np.testing.assert_allclose(actual,expected.reshape(-1,7),rtol=1e-15,atol=1e-15)
    v=draw((7,));w=draw((40,))
    np.testing.assert_allclose(actual@v,factor.apply(v),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(actual.conj().T@w,factor.adjoint(w),rtol=1e-13,atol=1e-13)


def test_overlap_factor_rejects_invalid_route():
    z=np.array([0],dtype=int)
    with pytest.raises(ValueError,match='outside'):
        overlap_factor_matrix(np.eye(2),np.eye(2),z,z,np.array([2]),z,2,1,1.)

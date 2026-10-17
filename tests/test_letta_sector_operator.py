import numpy as np
import pytest
from pyqed._letta_two_site_opt.sector_operator import SectorOperator


@pytest.mark.parametrize('complex_factors',[False,True])
@pytest.mark.parametrize('complex_vector',[False,True])
@pytest.mark.parametrize('physical',[(1,1),(2,3),(3,2)])
def test_compiled_sector_action(complex_factors,complex_vector,physical):
    rng=np.random.default_rng(837)
    def draw(shape,complex):
        value=rng.normal(size=shape)
        return value+1j*rng.normal(size=shape) if complex else value
    p,q=physical
    shapes={0:(2,p,3),1:(4,q,2),2:(0,1,3)}
    blocks={key:draw(shape,complex_vector) for key,shape in shapes.items()}
    terms=[]
    for bra,ket in [(0,1),(1,0),(0,0),(1,1),(0,1),(2,0)]:
        out,src=shapes[bra],shapes[ket]
        left=draw((2,out[0],src[0]),complex_factors)
        right=draw((3,out[-1],src[-1]),complex_factors)
        kernel=draw((2,3,out[1],src[1]),complex_factors)
        kernel[0,1]=0
        terms.append((bra,ket,left,kernel,right))
    expected={key:np.zeros(shape,dtype=complex) for key,shape in shapes.items()}
    for bra,ket,l,w,r in terms:
        expected[bra]+=np.einsum('xal,xyop,ybr,lpr->aob',l,w,r,blocks[ket],optimize=True)
    operator=SectorOperator(shapes,terms,scale=.7)
    actual=operator(blocks)
    for key in shapes:np.testing.assert_allclose(actual[key],.7*expected[key],rtol=3e-13,atol=3e-13)


def test_empty_operator():
    action=SectorOperator({0:(2,1,2)},[])
    np.testing.assert_array_equal(action({0:np.ones((2,1,2))})[0],0)


def test_pair_schedule_matches_reference_and_invalidates_on_replacement():
    from test_letta_pair_coordinates import setup_pair
    from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
    from pyqed._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain
    state,ham,pair=setup_pair()
    root=ReducedPairMetricRoot(pair,state,1e-12)
    from pyqed._letta_one_site_opt.reduced_coordinates import _canonical_sites
    canonical,_,_=_canonical_sites(pair.frontier.to_mps(state),1,last_center=2)
    chain=ReducedEnvironmentChain.build(canonical,ham.native_mpo(state.physical_basis),lazy=True)
    rng=np.random.default_rng(263)
    blocks={key:rng.normal(size=shape)+1j*rng.normal(size=shape) for key,shape in root.physical_layout.shapes.items()}
    def compare():
        expected=chain._pair_action_reference(1,blocks)
        actual=chain.pair_action(1,blocks)
        for key in blocks:np.testing.assert_allclose(actual[key],expected[key],rtol=2e-12,atol=2e-12)
    compare()
    assert chain._prepared_pairs
    changed=chain.sites[0].copy()
    changed.data={key:1.1*a for key,a in changed.data.items()}
    chain.replace_sites({0:changed})
    assert not chain._prepared_pairs
    compare()

@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('dims',[(2,3,5,7),(7,5,3,2)])
def test_reused_first_products_preserve_route_order(complex_values,dims):
    rng=np.random.default_rng(838)
    l,r,a,b=dims
    def draw(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_values else x
    shapes={0:(l,3,r),1:(a,4,b)}
    left,right=draw((2,a,l)),draw((3,b,r))
    terms=[(1,0,left,draw((2,3,4,3)),right)]
    action=SectorOperator(shapes,terms)
    count=action.reuse_count
    assert count>0 and action.reuse_elements*16<=64*1024**2
    for _ in range(3):
        blocks={k:draw(v) for k,v in shapes.items()}
        action.reuse_count=count
        actual=action(blocks)
        action.reuse_count=0
        expected=action(blocks)
        for key in shapes:
            np.testing.assert_array_equal(actual[key],expected[key])

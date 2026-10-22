import numpy as np
import pytest
from types import SimpleNamespace
from scipy.linalg import block_diag
from pyqed._letta_one_site_opt.boundary_structure import support_qr, BoundaryProducts

@pytest.mark.parametrize('complex_values',[False,True])
def test_support_qr_reconstructs_rectangular_disconnected_components(complex_values):
    rng=np.random.default_rng(48)
    chunks=[rng.normal(size=s) for s in [(4,7),(8,3),(5,5)]]
    if complex_values:chunks=[a+1j*rng.normal(size=a.shape) for a in chunks]
    a=np.pad(block_diag(*chunks),((3,4),(2,3)))
    a=a[rng.permutation(len(a))][:,rng.permutation(a.shape[1])]
    q,r=support_qr(a)
    np.testing.assert_allclose(q@r,a,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(q.conj().T@q,np.eye(q.shape[1]),atol=1e-13)
    assert q.shape[1]==12
    # A tiny nonzero connection is retained rather than thresholded away.
    a[0,0]=1e-100
    q,r=support_qr(a)
    np.testing.assert_allclose(q@r,a,rtol=1e-13,atol=1e-13)

@pytest.mark.parametrize('direction',['lr','rl'])
@pytest.mark.parametrize('complex_values',[False,True])
def test_sparse_products_preserve_complex_bra_and_accumulation(direction,complex_values):
    rng=np.random.default_rng(15)
    def draw(shape):
        a=rng.normal(size=shape)
        return a+1j*rng.normal(size=shape) if complex_values else a
    a=draw((32,1,40));b=draw((36,1,44))
    a*=rng.random(a.shape)<.02;b*=rng.random(b.shape)<.02
    site=SimpleNamespace(data={'a':a,'b':b})
    kernel=draw((2,2,1,1));env=draw((2,32,36) if direction=='lr' else (2,40,44))
    equation='xal,xyop,aob,lpr->ybr' if direction=='lr' else 'ybr,xyop,aob,lpr->xal'
    expected=np.einsum(equation,env,kernel,a.conj(),b,optimize=True)
    t=SimpleNamespace(bra_key='a',ket_key='b',kernel=kernel.transpose(1,0,2,3) if direction=='lr' else kernel)
    products=BoundaryProducts(site,direction)
    result=products.accumulate(env,t,.3,None)
    assert result is not None
    result=products.accumulate(env,t,.7,result)
    np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('direction',['lr','rl'])
@pytest.mark.parametrize('workers',[1,2])
def test_structured_full_absorption_matches_dense(direction,workers,monkeypatch):
    from test_letta_pair_coordinates import setup_pair
    from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
    from pyqed._letta_one_site_opt import reduced_environment as env
    state,h,_=setup_pair(1.)
    sites=ReducedFrontier.from_state(state).to_mps(state)
    site=sites[1];transfers=env.cached_transfers(h.native_mpo(state.physical_basis),1,site)
    rng=np.random.default_rng(83)
    for key in site.data:
        a=rng.normal(size=(32,1,40))+1j*rng.normal(size=(32,1,40))
        a*=rng.random(a.shape)<.02;site.data[key]=a
    incoming={}
    for t in transfers:
        channels=t.kernel.shape[0 if direction=='lr' else 1]
        size=32 if direction=='lr' else 40
        key=t.left_key if direction=='lr' else t.right_key
        incoming[key]=rng.normal(size=(channels,size,size))+1j*rng.normal(size=(channels,size,size))
    monkeypatch.setenv('PYQED_ENVIRONMENT_THREADS',str(workers))
    a=env._advance_environment(incoming,site,transfers,direction)
    b=env._advance_environment(incoming,site,transfers,direction,structured=True)
    assert a.keys()==b.keys()
    for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)


def test_oversized_sparse_boundary_cache_is_lossless():
    from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache
    a=np.eye(64,dtype=complex);a[2,7]=1e-100j
    cache=CanonicalBoundaryCache(max_bytes=12000);cache.structured=True
    factor={'q':a};environment={'h':a[None]};signatures=(b'a',b'b')
    cache.store('rl',1,signatures,factor,environment,.2)
    assert cache.entries and cache.nbytes<cache.max_bytes
    cut,(f,e,scale)=cache.nearest('rl',1,signatures)
    np.testing.assert_array_equal(f['q'],a)
    np.testing.assert_array_equal(e['h'],a[None])
    assert cut==1 and scale==.2
    assert cache.nearest('rl',1,(b'a',b'changed')) is None

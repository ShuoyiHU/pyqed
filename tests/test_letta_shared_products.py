import numpy as np
import pytest
from pyqed._letta_two_site_opt.sector_operator import SectorOperator
from pyqed._letta_two_site_opt.shared_sector_products import SharedSectorProducts
from concurrent.futures import ThreadPoolExecutor

def test_action_cache_does_not_keep_discarded_operator_alive(monkeypatch):
    import gc,weakref
    monkeypatch.setenv('PYQED_SHARED_PRODUCTS','1')
    monkeypatch.setenv('PYQED_SECTOR_THREADS','1')
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        op=SectorOperator({0:(2,1,2)},[(0,0,np.eye(2)[None],np.ones((1,1,1,1)),np.eye(2)[None])])
        ref=weakref.ref(op)
        op.apply_packed(np.ones(4))
        del op
        assert ref() is None
    finally:
        if enabled:gc.enable()

@pytest.mark.parametrize('budget',[(0,0),(10000000,0),(0,10000000),(1024,10000000)])
@pytest.mark.parametrize('complex_values',[False,True])
def test_cache(budget,complex_values,monkeypatch):
    rng=np.random.default_rng(22)
    def rand(shape):
        x=rng.normal(size=shape)
        return x+1j*rng.normal(size=shape) if complex_values else x
    shapes={0:(5,2,7),1:(137,3,4),2:(0,1,2)}
    terms=[]
    for bra,ket in [(1,0),(0,1),(0,0),(1,1),(2,0)]:
        a,no,b=shapes[bra];l,ni,r=shapes[ket]
        terms.append((bra,ket,rand((2,a,l)),rand((2,3,no,ni)),rand((3,b,r))))
    op=SectorOperator(shapes,terms,scale=.7)
    plan=SharedSectorProducts(op,*budget)
    monkeypatch.setenv('PYQED_SHARED_PRODUCTS','0')
    with ThreadPoolExecutor(3) as pool:
        for _ in range(2):
            x=rand((op.layout.size,))
            expected=op.apply_packed(x)
            actual,report=plan.apply(x,3,pool)
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-10)

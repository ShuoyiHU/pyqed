"""Prepared shared GEMM actions: exact fast paths, tiling and solver parity."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from pyqed._letta_two_site_opt.sector_operator import SectorOperator
from pyqed._letta_two_site_opt.shared_sector_products import SharedSectorProducts


@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('kind',['identity','single'])
def test_special_first_products(side,kind,complex_values,monkeypatch):
    rng=np.random.default_rng(221)
    a,l,r,b=(5,5,3,8) if side=='left' else (9,4,3,3)
    left=rng.normal(size=(a,l));right=rng.normal(size=(b,r))
    special=np.eye(a) if side=='left' else np.eye(b)
    if kind=='single':special[:]=0;special[-1,0]=2
    special=special*(1+2j if complex_values else 2.)
    if side=='left':left=special
    else:right=special
    shapes={0:(l,2,r),1:(a,3,b)}
    op=SectorOperator(shapes,[(1,0,left[None],np.ones((1,1,3,2)),right[None])])
    plan=SharedSectorProducts(op,1024**2)
    with ThreadPoolExecutor(3) as pool:
        for complex_vector in [False,True,False]:
            x=rng.normal(size=op.layout.size)
            if complex_vector:x=x+1j*rng.normal(size=x.size)
            expected=op.apply_packed(x,shared_products=False)
            for backend in ['python','compiled']:
                monkeypatch.setenv('PYQED_SHARED_PRODUCT_BACKEND',backend)
                actual,report=plan.apply(x,3,pool)
                np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-12)
                if backend=='compiled':assert report['special_first_products']>0


def test_tiles_concurrent_vectors_and_arena_reuse(monkeypatch):
    rng=np.random.default_rng(222);n=260
    op=SectorOperator({0:(n,2,3)},[(0,0,rng.normal(size=(2,n,n)),
                      rng.normal(size=(2,3,2,2)),rng.normal(size=(3,3,3)))])
    plan=SharedSectorProducts(op,8*1024**2)
    monkeypatch.setenv('PYQED_SHARED_PRODUCT_BACKEND','compiled')
    x=rng.normal(size=op.layout.size)
    with ThreadPoolExecutor(4) as pool:
        y,report=plan.apply(x,4,pool);arena=plan._scratch.arena
        z,_=plan.apply(x*2,4,pool)
        assert plan._scratch.arena is arena
        np.testing.assert_allclose(z,2*y,rtol=2e-13,atol=2e-11)
        assert report['output_tiles']>2
        with ThreadPoolExecutor(2) as callers:
            got=list(callers.map(lambda v:plan.apply(v,4,pool)[0],[x,x*3]))
        np.testing.assert_allclose(got,[y,3*y],rtol=2e-13,atol=2e-11)
        # Returned results never alias reused scratch storage.
        np.testing.assert_array_equal(y,got[0])


def test_letta_energies_same_with_compiled_shared_actions(monkeypatch):
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
    from test_letta_qchem import integrals
    p=ElectronicProblem(*integrals(3),(1,1),.37)
    results=[]
    for backend in ['python','compiled']:
        monkeypatch.setenv('PYQED_SHARED_PRODUCTS','1')
        monkeypatch.setenv('PYQED_SHARED_PRODUCT_BACKEND',backend)
        monkeypatch.setenv('PYQED_SECTOR_THREADS','2')
        monkeypatch.setenv('PYQED_PRODUCT_CACHE_MB','8')
        job=QCLETTA(p,4,seed=71)
        job.two_site(2).polish(2);job.tie([(0,2)]);job.optimize_letta(2)
        results.append([r['total_energy'] for r in job.history])
    np.testing.assert_allclose(results[0],results[1],atol=2e-10,rtol=0.)

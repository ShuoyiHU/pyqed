import numpy as np
import pytest
from pyqed._letta_one_site_opt.streamed_overlap import streamed_svd
from pyqed._letta_one_site_opt.reduced_coordinates import _OverlapFactor


@pytest.mark.parametrize('complex_data',[False,True])
def test_streamed_svd_retains_small_directions(complex_data):
    rng=np.random.default_rng(67)
    a=rng.normal(size=(173,9))
    if complex_data:a=a+1j*rng.normal(size=a.shape)
    a[:,-1]=a[:,0]+1e-8*a[:,-1]
    a[:,3]=0
    roots,u,s,vh=streamed_svd(a.shape,lambda i,j:a[i:j],workspace_mb=.05)
    normalized=np.divide(a,roots,out=np.zeros_like(a),where=roots>0)
    expected=np.linalg.svd(normalized,compute_uv=False)
    np.testing.assert_allclose(s,expected,atol=2e-15,rtol=1e-6)
    np.testing.assert_allclose((u*s)@vh,normalized,atol=3e-15)
    np.testing.assert_allclose(u.conj().T@u,np.eye(9),atol=3e-15)


def test_overlap_rows_preserve_scatter_duplicates():
    rng=np.random.default_rng(52)
    f=_OverlapFactor(rng.normal(size=(3,4)),rng.normal(size=(5,2)),
        np.array([0,1,0,2]),np.array([0,1,3,1]),np.array([0,1,0,0]),
        np.array([0,1,1,0]),2,3,np.sqrt(2.))
    np.testing.assert_allclose(np.vstack([f.rows(0,7),f.rows(7,30)]),f.dense(),atol=1e-15)


def test_coordinate_streaming_preserves_physical_actions(monkeypatch):
    from test_letta_compact_hamiltonian import coordinates
    monkeypatch.setenv('PYQED_OVERLAP_STREAM_MB','-1')
    old=coordinates()
    monkeypatch.setenv('PYQED_OVERLAP_STREAM_MB','0')
    new=coordinates()
    assert new.rank==old.rank
    x=np.random.default_rng(123).normal(size=old.dimension)
    np.testing.assert_allclose(new.metric(x),old.metric(x),atol=1e-12)
    np.testing.assert_allclose(new.expand(new.coordinates(x)),old.expand(old.coordinates(x)),atol=1e-11)

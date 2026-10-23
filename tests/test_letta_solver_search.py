from types import SimpleNamespace
import numpy as np
import pytest
from pyqed._letta_one_site_opt import LETTADMROptions
from pyqed._letta_one_site_opt.reduced_solver import _davidson_in_metric_coordinates


def test_budgeted_restart_preserves_lowest_root(monkeypatch):
    rng=np.random.default_rng(27)
    q,_=np.linalg.qr(rng.normal(size=(60,60)))
    h=(q*np.linspace(-3.,5.,60))@q.T
    p=SimpleNamespace(local_dimension=60,apply_hamiltonian=lambda x:h@x,
        apply_metric=lambda x:x,metric_scale=1.)
    monkeypatch.setenv('PYQED_DAVIDSON_SPACE','24')
    monkeypatch.setenv('PYQED_DAVIDSON_RETAIN','8')
    options=LETTADMROptions(eigensolver_max_iterations=300,eigensolver_tolerance=1e-10)
    e,v,_=_davidson_in_metric_coordinates(p,options,q[:,-1])
    assert abs(e+3.)<1e-10
    assert np.linalg.norm(h@v-e*v)<5e-10
    assert p.search_report['max_space']==24
    assert p.search_report['retained']==8
    assert p.search_report['hamiltonian_applications']>0


def test_recycled_search_recomputes_changed_hamiltonian():
    rng=np.random.default_rng(42)
    q,_=np.linalg.qr(rng.normal(size=(30,30)))
    h=(q*np.linspace(-4.,6.,30))@q.T
    old=q[:,:3]
    h[0,0]-=2.
    p=SimpleNamespace(local_dimension=30,apply_hamiltonian=lambda x:h@x,
        apply_metric=lambda x:x,metric_scale=1.,initial_search_vectors=old.T)
    options=LETTADMROptions(eigensolver_max_iterations=300,eigensolver_tolerance=1e-10)
    e,v,_=_davidson_in_metric_coordinates(p,options,q[:,-1])
    assert abs(e-np.linalg.eigvalsh(h)[0])<1e-11
    assert np.linalg.norm(h@v-e*v)<1e-9
    assert p.search_report['recycled_vectors']==3


@pytest.mark.parametrize('retry_succeeds',[False,True])
def test_physical_stopping_still_certifies_source_action(monkeypatch,retry_succeeds):
    from test_letta_compact_hamiltonian import coordinates
    from pyqed._letta_one_site_opt import reduced_solver as solver
    monkeypatch.setenv('PYQED_PHYSICAL_RESIDUAL_SOLVE','1')
    c=coordinates()
    h=np.column_stack([c.apply(v) for v in np.eye(c.rank)])
    _,u=np.linalg.eigh((h+h.conj().T)*.5)
    bad=u[:,0]+.01*u[:,-1];bad/=np.linalg.norm(bad)
    calls=[]
    def stopped_search(local,options,initial):
        calls.append(options.eigensolver_tolerance)
        local.search_report={'hamiltonian_applications':1}
        v=u[:,0] if len(calls)==2 and retry_succeeds else bad
        return float(np.vdot(v,h@v).real),v,1
    monkeypatch.setattr(solver,'_davidson_in_metric_coordinates',stopped_search)
    p=SimpleNamespace(coordinates=c,apply_metric=c.metric)
    options=LETTADMROptions(matrix_free=True,dense_solver_threshold=0,eigensolver_tolerance=1e-10)
    solver._lowest_coordinate_root(p,options,c.expand(bad))
    assert len(calls)==2 and calls[1]<calls[0]
    assert c.search_report['retry']
    certified=c.solution_diagnostics[1]['relative_residual']<=options.eigensolver_tolerance
    assert certified==retry_succeeds

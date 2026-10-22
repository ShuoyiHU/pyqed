"""Exact solve actions, ownership, and reusable final diagnostics."""
import numpy as np
import pytest
from pyqed._letta_two_site_opt.sector_operator import SectorOperator

@pytest.mark.parametrize('workers',[1,4,16])
def test_packed_action_ownership_and_timing(monkeypatch,workers):
    monkeypatch.setenv('PYQED_SECTOR_THREADS',str(workers))
    rng=np.random.default_rng(92);shapes={0:(3,2,4)}
    l=rng.normal(size=(2,3,3));r=rng.normal(size=(2,4,4));w=rng.normal(size=(2,2,2,2))
    op=SectorOperator(shapes,[(0,0,l,w,r)],.8);records=[];op.timing_callback=records.append
    for _ in range(3):
        x=rng.normal(size=shapes[0])+1j*rng.normal(size=shapes[0])
        actual=op.apply_packed(op.layout.pack({0:x}));saved=actual.copy()
        expected=.8*np.einsum('xal,xyop,ybr,lpr->aob',l,w,r,x,optimize=True)
        np.testing.assert_allclose(op.layout.unpack(actual)[0],expected,rtol=2e-13,atol=2e-13)
        op.apply_packed(op.layout.pack({0:2*x}))
        np.testing.assert_array_equal(actual,saved)
    assert records and all(x['kernel_wall_seconds']>=0 for x in records)

def test_final_diagnostics_use_single_h_action():
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_solver import reduced_local_problem
    from pyqed._letta_one_site_opt.reduced_updates import local_residual_diagnostics
    h=np.array([[0.,-1.,0.],[-1.,0.,-1.],[0.,-1.,0.]])
    p=ElectronicProblem(h,np.zeros((3,)*4),(1,1))
    s=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=43,neighborhoods=((0,1,2),(1,2),(2,)))
    problem=reduced_local_problem(s,p.su2_mpo(),1,matrix_free=True)
    c=problem.coordinates;x=problem.embedding.pack_source(s.tensors[1]);original=c.hamiltonian_action;calls=[]
    def counted(v):
        calls.append(1);return original(v)
    c.hamiltonian_action=counted
    report=local_residual_diagnostics(problem,x)
    assert len(calls)==1
    assert report['norm_squared']==pytest.approx(c.norm(x),rel=2e-12)
    assert report['energy']==pytest.approx(c.energy(x),abs=2e-12)

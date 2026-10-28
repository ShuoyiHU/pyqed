"""Parity gauges must preserve frozen sides and their physical norm support."""
import numpy as np
from pyqed._letta_one_site_opt import AbelianSymmetry, LatticeLETTA
from pyqed._letta_one_site_opt.gauge import canonicalize_frontier
from pyqed._letta_one_site_opt._letta_for_3d import transverse_field_ising_mpo
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg

def make_mps(n=16, D=10):
    rng=np.random.default_rng(4)
    tensors=[]
    for i in range(n):
        a=rng.normal(size=(1 if i==0 else D,2,1 if i==n-1 else D))
        tensors.append(a/np.sqrt(a.size))
    state=LatticeLETTA((4,4,1),2,tensors,neighborhoods=[(i,) for i in range(n)],
        symmetry=AbelianSymmetry(physical_charges=(0,1),sector=0,moduli=2),normalize=False)
    canonicalize_frontier(state,0,tolerance=1e-12)
    state.tensors[0]/=np.max(np.abs(state.tensors[0]))
    state.tensors[0]/=np.linalg.norm(state.tensors[0])
    return state

def test_parity_gauge_does_not_require_later_symmetry_cleanup():
    s=make_mps()
    before=[a.copy() for a in s.tensors]
    s.enforce_symmetry()
    for a,b in zip(before,s.tensors):np.testing.assert_array_equal(a,b)

def test_two_site_parity_mps_keeps_null_directions_out_of_local_solve():
    s=make_mps();H=transverse_field_ising_mpo((4,4,1),coupling=.25,field=1.,ordering='compact',basis='x')
    r=letta_two_site_dmrg(H,state=s,bond_dim=10,options=LETTATwoSiteOptions(
        max_sweeps=12,metric_tolerance=1e-12,split_method='conditional-svd'))
    assert 'FAILED' not in r.message
    for sweep in r.history:
        for update in sweep.updates:
            assert -22.00000001 <= update.local_energy <= 22.00000001
    np.testing.assert_allclose(r.energy,r.state.expectation(H),atol=1e-9,rtol=0)
    assert r.state.symmetry_violation()==0

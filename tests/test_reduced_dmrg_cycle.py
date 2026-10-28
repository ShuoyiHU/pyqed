import numpy as np
import pytest
from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import initial_mps
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG, _DirectionalCycle
from test_letta_qchem import determinant_hamiltonian


@pytest.mark.parametrize('low',['lr','rl'])
def test_cycle_requires_both_stationary_directions_and_nonzero_gap(low):
    c=_DirectionalCycle(1e-6)
    other='rl' if low=='lr' else 'lr'
    assert c.observe(low,-2.) is None
    assert c.observe(other,-1.) is None
    assert c.observe(low,-2.-1e-7) is None
    assert c.observe(other,-1.-1e-7)==low
    # A direction starts moving again: this is no longer a stationary cycle.
    assert c.observe(low,-2.1) is None


def test_no_cycle_for_shared_plateau_or_single_stationary_endpoint():
    c=_DirectionalCycle(1e-6)
    for direction,e in [('lr',-2.),('rl',-2.+2e-7),('lr',-2.-1e-7),('rl',-2.+1e-7)]:
        assert c.observe(direction,e) is None
    c=_DirectionalCycle(1e-6)
    for direction,e in [('lr',-2.),('rl',-1.),('lr',-2.),('rl',-1.1)]:
        assert c.observe(direction,e) is None


def problem():
    r=np.random.default_rng(0);h=r.normal(size=(4,4));h=(h+h.T)*.2
    a=r.normal(size=(5,4,4));a=(a+a.transpose(0,2,1))*.15
    p=ElectronicProblem(h,np.einsum('apq,ars->pqrs',a,a),(2,2),.31)
    sites,_=initial_mps(4,4,0,30,'multiplets',7)
    return p,sites


def test_real_directional_cycle_restores_earlier_lower_endpoint_and_polishes():
    p,sites=problem();solver=ReducedDMRG(p,sites,max_bond=3)
    sectors={}
    def record(row):
        if row['stage']=='two_site':sectors[row['sweep']]=solver.state.bond_sectors
    solver.run(max_sweeps=20,energy_tolerance=1e-6,residual_tolerance=1e-10,
               one_site_max_sweeps=30,callback=record)
    switch=solver.cycle_handoff
    assert switch['selected_direction']=='lr'
    assert switch['selected_pass']==3  # Switch happens at the higher RL pass4.
    assert switch['endpoint_gap']>1e-6
    assert all(abs(x)<=1e-6 for x in switch['same_direction_changes'].values())
    assert solver.state.bond_sectors==sectors[3]
    polish=[r for r in solver.history if r['stage']=='one_site']
    assert polish[0]['direction']=='rl'
    assert all(r['delta_energy']<=1e-10 for r in polish)
    assert solver.energy<switch['selected_energy']-1e-3
    assert solver.success and len(polish)>=2
    assert all(abs(r['delta_energy'])<=1e-6 for r in polish[-2:])
    vector=solver.state.state_vector()
    expected=np.vdot(vector,determinant_hamiltonian(p)@vector)/np.vdot(vector,vector)
    assert solver.energy==pytest.approx(expected.real,abs=1e-10)
    assert max(len(s.qns[2]) for s in solver.sites[:-1])<=3


def test_polish_cap_is_not_convergence_and_resume_stays_one_site():
    p,sites=problem()
    first=ReducedDMRG(p,sites,max_bond=3).run(max_sweeps=20,
        energy_tolerance=1e-6,one_site_max_sweeps=1)
    assert first.cycle_handoff and not first.success
    assert first.history[-1]['stage']=='one_site'
    resumed=ReducedDMRG(p,first.sites,max_bond=3).run(start_stage='one_site',
        start_direction='lr',energy_tolerance=1e-6,one_site_max_sweeps=30)
    assert resumed.success
    assert all(r['stage']=='one_site' for r in resumed.history)
    assert resumed.energy<=first.energy+1e-10


@pytest.mark.parametrize('options',[dict(polish_on_cycle=False),dict(energy_tolerance=0.)])
def test_fixed_two_site_runs_do_not_switch(options):
    p,sites=problem()
    solver=ReducedDMRG(p,sites,max_bond=3).run(max_sweeps=6,**options)
    assert solver.cycle_handoff is None
    assert len(solver.history)==6 and all(r['stage']=='two_site' for r in solver.history)

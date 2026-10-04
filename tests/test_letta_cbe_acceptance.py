"""Acceptance compares against the ordinary candidate, including near ties."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt import LETTADMROptions,letta_dmrg
from pyqed._letta_one_site_opt.cbe import _cbe_candidate_is_preferred
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state


@pytest.mark.parametrize('candidate,expected',[
    (-2.1,True),(-2.-1e-12,True),(-2.,False),(-2.+1e-12,False),
    (-1.9,False),(-0.9,False),(np.nan,False),(np.inf,False),(-np.inf,False),
])
def test_strict_baseline_requires_a_real_gain_even_with_large_increase_tolerance(candidate,expected):
    options=LETTADMROptions(energy_increase_tolerance=1.0)
    assert _cbe_candidate_is_preferred(candidate,-1.,-2.,options)==expected


@pytest.mark.parametrize('candidate,expected',[
    (-1.1,True),(-1.-1e-12,True),(-1.,False),(-1.+1e-12,False),
])
def test_previous_energy_control_uses_pre_update_energy(candidate,expected):
    options=LETTADMROptions(cbe_baseline_guard_fraction=1.0)
    assert _cbe_candidate_is_preferred(candidate,-1.,-2.,options)==expected


def test_old_exploratory_allowance_is_available_as_an_explicit_control():
    options=LETTADMROptions(cbe_baseline_guard_fraction=0.2)
    assert _cbe_candidate_is_preferred(-1.85,-1.,-2.,options)
    assert not _cbe_candidate_is_preferred(-1.75,-1.,-2.,options)


@pytest.mark.parametrize('selector',['shrewd','exact'])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_accepted_cbe_updates_beat_the_same_state_ordinary_candidate(selector,direction):
    model=build_model('ising','2d',(2,2))
    initial=make_shared_initial_state(model,bond_dim=1,seed=732).letta
    result=letta_dmrg(model.mpo,state=initial,options=LETTADMROptions(
        cbe_enabled=True,cbe_selector=selector,start_direction=direction,
        max_sweeps=2,metric_tolerance=1e-10))
    attempted=[u for s in result.history for u in s.updates if u.cbe_baseline_energy is not None]
    assert attempted
    assert any(not u.cbe_baseline_selected for u in attempted)
    for u in attempted:
        assert u.cbe_baseline_allowance==0
        if u.cbe_baseline_selected:
            assert u.energy==u.cbe_baseline_energy
        else:
            assert u.energy < u.cbe_baseline_energy
            assert u.energy < u.cbe_old_energy
    vector=result.state.state_vector()
    physical=np.vdot(vector,model.mpo.to_dense()@vector).real/np.vdot(vector,vector).real
    np.testing.assert_allclose(result.energy,physical,atol=1e-10,rtol=0)

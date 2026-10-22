import numpy as np
import pytest
from test_letta_nonzero_spin import setup
from pyqed._letta_one_site_opt import reduced_gauge as gauge
from pyqed._letta_one_site_opt.reduced_coordinates import _canonical_sites
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier

@pytest.mark.parametrize('direction',['lr','rl'])
def test_factors_do_not_build_unused_boundary(direction,monkeypatch):
    _,state=setup(1,True)
    import pyqed._letta_one_site_opt.reduced_coordinates as coords
    def forbidden(*a,**kw):raise AssertionError('gauge must not construct both canonical boundaries')
    monkeypatch.setattr(coords,'_canonical_sites',forbidden)
    gauge.reduced_frontier_factors(state,2,direction)

@pytest.mark.parametrize('spin',[0,1,2])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_moving_qr_matches_full_rebuild_and_preserves_state(spin,direction):
    _,state=setup(spin,True)
    before=state.state_vector()
    sweep=gauge.GaugeQRBoundary(state,direction)
    cuts=range(1,state.nsites) if direction=='lr' else range(state.nsites-1,0,-1)
    for cut in cuts:
        frontier=ReducedFrontier.from_state(state)
        _,left,right=_canonical_sites(frontier.to_mps(state),cut if direction=='lr' else cut-1,
                                      transform_active=False)
        expected=left if direction=='lr' else right
        actual=sweep.at(cut)
        for q,a in expected.items():
            np.testing.assert_allclose(actual[q].conj().T@actual[q],a.conj().T@a,rtol=2e-9,atol=2e-10)
        gauge.shift_reduced_frontier_gauge(state,cut,direction,qr_boundary=sweep)
    np.testing.assert_allclose(state.state_vector(),before,rtol=2e-10,atol=2e-10)
    assert sweep.site_contractions==(state.nsites-1)

@pytest.mark.parametrize('direction',['lr','rl'])
def test_moving_qr_invalidates_in_place_changes(direction):
    _,state=setup(1,True)
    sweep=gauge.GaugeQRBoundary(state,direction);cut=2;sweep.at(cut)
    site=0 if direction=='lr' else state.nsites-1
    for a in state.tensors[site].values():a*=1.17
    actual=sweep.at(cut)
    fresh=gauge.GaugeQRBoundary(state,direction).at(cut)
    for q in fresh:np.testing.assert_array_equal(actual[q],fresh[q])

@pytest.mark.parametrize('center',[0,1,3])
def test_initial_conditioning_has_linear_site_work(center,monkeypatch):
    _,state=setup(1,True)
    before=state.state_vector();calls=[]
    advance=gauge.GaugeQRBoundary._advance
    def counted(self,site,incoming):
        calls.append(site)
        return advance(self,site,incoming)
    monkeypatch.setattr(gauge.GaugeQRBoundary,'_advance',counted)
    gauge.canonicalize_reduced_frontier(state,center)
    assert len(calls)==(state.nsites-1)
    np.testing.assert_allclose(state.state_vector(),before,rtol=2e-10,atol=2e-10)


def test_refresh_failure_restores_gauge_transaction(monkeypatch):
    _,state=setup(1,True)
    original=state.copy();cache=gauge.GaugeQRBoundary(state,'lr')
    def fail(*args):raise FloatingPointError('injected QR failure')
    monkeypatch.setattr(cache,'refresh',fail)
    with pytest.raises(FloatingPointError,match='injected QR failure'):
        gauge.shift_reduced_frontier_gauge(state,1,'lr',qr_boundary=cache)
    for got,expected in zip(state.tensors,original.tensors):
        for key in expected:np.testing.assert_array_equal(got[key],expected[key])

"""Stable energy reuse must retain the independent QR contraction exactly."""
import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt.frontier_contraction import EnergyBoundaryCache, frontier_energy

@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('budget',[0,16*1024**2])
def test_energy_prefix_reuse_matches_fresh(complex_values,budget):
    state,ham,_=setup_pair(1e5)
    if complex_values:
        for blocks in state.tensors:
            for key in blocks:blocks[key]=blocks[key]*(1+.2j)
    mpo=ham.native_mpo(state.physical_basis);cache=EnergyBoundaryCache(budget)
    for site in [None,2,3,1,0,3]:
        if site is not None:
            key=next(iter(state.tensors[site]));state.tensors[site][key]*=1.001
        expected=frontier_energy(state,mpo)
        actual=frontier_energy(state,mpo,cache=cache)
        np.testing.assert_array_equal(actual,expected)
        assert cache.nbytes<=budget
    assert (cache.hits>0)==bool(budget)
    # A new Hamiltonian object cannot reuse old Hamiltonian boundaries.
    old_hits=cache.hits
    import copy
    changed=copy.deepcopy(mpo)
    np.testing.assert_array_equal(frontier_energy(state,changed,cache=cache),frontier_energy(state,changed))
    assert cache.hits==old_hits


def test_stable_energy_uses_independent_prefixes(monkeypatch):
    from pyqed._letta_one_site_opt import reduced_solver as solver
    from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
    state,ham,_=setup_pair()
    monkeypatch.setattr(ReducedFrontier,'requires_compact',lambda self,state:True)
    cache=solver.StableEnergyCache(ham)
    assert cache(state)==solver._energy(state,ham,stable=True)
    key=next(iter(state.tensors[2]));state.tensors[2][key]*=1.002
    assert cache(state)==solver._energy(state,ham,stable=True)
    assert cache.boundaries.hits>0
    # Topology changes must invalidate even when the tensors are unchanged.
    from copy import copy
    changed=copy(state);changed.bond_sectors=list(state.bond_sectors)
    changed.bond_sectors[1]=tuple(reversed(changed.bond_sectors[1]))
    cache.boundaries.signatures(changed,ham.native_mpo(state.physical_basis))
    if changed.bond_sectors!=state.bond_sectors:assert not cache.boundaries.entries

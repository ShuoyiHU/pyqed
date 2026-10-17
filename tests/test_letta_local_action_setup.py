"""Active tensor values are unnecessary for a boundary-defined local action."""
import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates

@pytest.mark.parametrize('center',[0,1,2,3])
def test_action_only_setup_matches_full_canonical_contraction(center):
    state,ham,_=setup_pair()
    frontier=ReducedFrontier.from_state(state)
    sites=frontier.to_mps(state,compact=True)
    embedding=frontier.site_embedding(state,center)
    mpo=ham.native_mpo(state.physical_basis)
    reference=ReducedLocalCoordinates(sites,embedding,center,mpo=mpo)
    action=ReducedLocalCoordinates(sites,embedding,center,mpo=mpo,action_only=True)
    assert action.sites is None
    rng=np.random.default_rng(171)
    vector=rng.normal(size=embedding.source_size)+1j*rng.normal(size=embedding.source_size)
    np.testing.assert_array_equal(action.source_hamiltonian(vector),reference.source_hamiltonian(vector))
    np.testing.assert_array_equal(action.metric(vector),reference.metric(vector))
    assert action.energy(vector)==reference.energy(vector)

"""Staged raw-coordinate action agrees with the direct contraction reference."""
import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates

@pytest.mark.parametrize('scale',[1.,1e5])
@pytest.mark.parametrize('center',[0,1,2,3])
def test_staged_source_hamiltonian_matches_direct(scale,center):
    state,ham,_=setup_pair(scale)
    frontier=ReducedFrontier.from_state(state)
    embedding=frontier.site_embedding(state,center)
    c=ReducedLocalCoordinates(frontier.to_mps(state,compact=True),embedding,center,
                              mpo=ham.native_mpo(state.physical_basis),action_only=True)
    rng=np.random.default_rng(920)
    x=rng.normal(size=embedding.source_size)+1j*rng.normal(size=embedding.source_size)
    y=rng.normal(size=x.size)+1j*rng.normal(size=x.size)
    from pyqed._letta_one_site_opt.frontier_contraction import scatter_blocks
    raw=scatter_blocks(embedding,embedding.unpack_source(x))
    direct={k:np.einsum('al,lpr,br->apb',c.left[k[0]],np.asarray(a),c.right[k[2]])
            for k,a in raw.items()}
    transformed=c.source_blocks(x)
    for k in direct:
        np.testing.assert_allclose(transformed[k],direct[k],rtol=2e-12,
            atol=2e-12*max(1.,np.linalg.norm(direct[k])))
    acted=c.hamiltonian_action(direct)
    pulled={k:np.einsum('al,apb,br->lpr',c.left[k[0]].conj(),a,c.right[k[2]].conj())
            for k,a in acted.items()}
    expected=embedding.adjoint(embedding.pack_target(pulled))
    actual=c.source_hamiltonian(x)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12*max(1.,np.linalg.norm(expected)))
    np.testing.assert_allclose(np.vdot(y,actual),np.vdot(c.source_hamiltonian(y),x),rtol=2e-12,
                              atol=2e-12*max(1.,np.linalg.norm(expected)*np.linalg.norm(y)))

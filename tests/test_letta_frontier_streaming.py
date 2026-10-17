import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_norm import left_canonical_reduced_sites, ReducedNormChain
from pyqed._letta_one_site_opt.frontier_contraction import frontier_qr, frontier_norm, frontier_energy
from pyqed._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain

@pytest.mark.parametrize('scale',[1.,1e5])
@pytest.mark.parametrize('complex_values',[False,True])
def test_streamed_qr_norm_energy_matches_dense(scale,complex_values,monkeypatch):
    state,ham,_=setup_pair(scale)
    if complex_values:
        rng=np.random.default_rng(271)
        for tensor in state.tensors:
            for key,a in tensor.items():tensor[key]=a*(1+.1j*rng.normal(size=a.shape))
    dense=left_canonical_reduced_sites(ReducedFrontier.from_state(state).to_mps(state))
    expected_norm=ReducedNormChain.build(dense).expectation()
    mpo=ham.native_mpo(state.physical_basis)
    expected_energy=ReducedEnvironmentChain.build(dense,mpo,lazy=True).expectation()/expected_norm
    def forbidden(*args,**kwargs):raise AssertionError('dense whole-chain expansion')
    monkeypatch.setattr(ReducedFrontier,'to_mps',forbidden)
    assert frontier_norm(state)==pytest.approx(expected_norm/(state.target_two_j+1),rel=2e-10)
    numerator,denominator=frontier_energy(state,mpo)
    assert numerator/denominator==pytest.approx(expected_energy,abs=2e-10)
    streamed=list(frontier_qr(state))
    # Compare physical invariants rather than arbitrary QR phases.
    assert ReducedNormChain.build(streamed).expectation()==pytest.approx(expected_norm,rel=2e-10)


def test_constant_ties_only_removed_in_evaluation_view():
    from pyqed._letta_one_site_opt.frontier_contraction import evaluation_state
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    state,ham,_=setup_pair()
    sites=ReducedFrontier.from_state(state).to_mps(state)
    # An ordinary MPS broadcast into ties is exactly the same wavefunction.
    ordinary=ReducedLatticeLETTA.from_mps(sites,symmetry=state.symmetry,normalize=False)
    tied=ReducedLatticeLETTA.from_mps(sites,symmetry=state.symmetry,
        neighborhoods=state.neighborhoods,normalize=False)
    view=evaluation_state(tied)
    assert view.neighborhoods==tuple((i,) for i in range(tied.nsites))
    assert tied.neighborhoods==state.neighborhoods
    assert frontier_norm(tied)==pytest.approx(frontier_norm(ordinary),rel=1e-12)
    i=next(i for i,n in enumerate(tied.neighborhoods) if len(n)>1)
    key=next(iter(tied.tensors[i]));a=tied.tensors[i][key]
    a[(0,0,1)+(0,)*(a.ndim-3)]+=1e-12
    assert len(evaluation_state(tied).site_neighborhood(i))>1


@pytest.mark.parametrize('center,last',[(0,0),(1,1),(1,2),(3,3)])
def test_compact_canonical_coordinates_match_dense(center,last,monkeypatch):
    from pyqed._letta_one_site_opt.reduced_coordinates import _canonical_sites
    from pyqed._letta_one_site_opt.frontier_contraction import ScatterBlock
    state,ham,_=setup_pair()
    frontier=ReducedFrontier.from_state(state)
    dense=frontier.to_mps(state)
    compact=frontier.to_mps(state,compact=True)
    expected,left,right=_canonical_sites(dense,center,last_center=last)
    actual,sl,sr=_canonical_sites(compact,center,last_center=last)
    for q in left:np.testing.assert_allclose(sl[q].conj().T@sl[q],left[q].conj().T@left[q],rtol=2e-12,atol=2e-12)
    for q in right:np.testing.assert_allclose(sr[q].conj().T@sr[q],right[q].conj().T@right[q],rtol=2e-12,atol=2e-12)
    for sa,se in zip(actual,expected):
        for key in se.data:np.testing.assert_array_equal(sa.data[key],se.data[key])
    a=ReducedNormChain.build(actual).expectation();e=ReducedNormChain.build(expected).expectation()
    assert a==pytest.approx(e,rel=2e-12)

@pytest.mark.parametrize('complex_values',[False,True])
def test_streamed_pair_actions_match_dense(complex_values,monkeypatch):
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    state,ham,_=setup_pair()
    rng=np.random.default_rng(73)
    if complex_values:
        for tensor in state.tensors:
            for key,a in tensor.items():tensor[key]=a*(1+.1j*rng.normal(size=a.shape))
    monkeypatch.setattr(ReducedFrontier,'requires_compact',lambda *_:False)
    dense=reduced_pair_problem(state,ham,1,matrix_free=True,dense_solver_threshold=0)
    monkeypatch.setattr(ReducedFrontier,'requires_compact',lambda *_:True)
    streamed=reduced_pair_problem(state,ham,1,matrix_free=True,dense_solver_threshold=0)
    v=rng.normal(size=dense.layout.size)+1j*rng.normal(size=dense.layout.size)
    for name in ('apply_hamiltonian','apply_metric'):
        np.testing.assert_allclose(getattr(streamed,name)(v),getattr(dense,name)(v),rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(streamed.old_vector,dense.old_vector,rtol=2e-12,atol=2e-12)

"""Pair metric whitening must preserve the physical Hamiltonian and Ritz root."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import integrals, initial_mps
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt import LETTATwoSiteOptions
from pyqed._letta_two_site_opt import reduced_solver as solver
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot


def setup_pair(gauge_scale=1.):
    h,g,edges=integrals(2,2,dict(U=4.,t=1.,t_diagonal=-.5))
    problem=ElectronicProblem(h,g,(2,1))
    sites,_=initial_mps(4,3,1,3,'multiplets',4)
    neighbors=tuple((i,)+tuple(j for a,j in edges if a==i) for i in range(4))
    state=ReducedLatticeLETTA.from_mps(sites,symmetry=problem.symmetry('su2'),neighborhoods=neighbors,normalize=True)
    state=solver._expand_reduced_pair_space(state,1,3)
    scales={q:gauge_scale**((i%3)-1) for i,q in enumerate(set(state.bond_sectors[0]))}
    for key,block in state.tensors[0].items():
        state.tensors[0][key]=block*scales[key[-1]]
    for key,block in state.tensors[1].items():
        state.tensors[1][key]=block/scales[key[0]]
    ham=problem.su2_mpo()
    pair=solver.reduced_pair_problem(state,ham,1,matrix_free=True,dense_solver_threshold=0)
    return state,ham,pair


@pytest.mark.parametrize('gauge_scale',[1.,1e5])
def test_pair_orthogonal_action_matches_raw_and_dense_reference(gauge_scale):
    state,ham,pair=setup_pair(gauge_scale)
    root=ReducedPairMetricRoot(pair,state,1e-12)
    action=solver._orthogonal_pair_action(pair,root,state,ham)
    rng=np.random.default_rng(36)
    z=rng.normal(size=root.size)+1j*rng.normal(size=root.size)
    x=root.unwhiten(z)
    np.testing.assert_allclose(root.apply(x),z,atol=1e-10)
    np.testing.assert_allclose(action(z),root.unwhiten_adjoint(pair.apply_hamiltonian(x)),atol=1e-9)
    matrix=np.column_stack([action(v) for v in np.eye(root.size)])
    np.testing.assert_allclose(matrix,matrix.conj().T,atol=1e-10)
    opts=LETTATwoSiteOptions(metric_tolerance=1e-12,eigensolver_tolerance=1e-10,dense_solver_threshold=0)
    e,v,rank,res=solver._solve_reduced_pair_root(pair,state,ham,opts)
    np.testing.assert_allclose(e,np.linalg.eigvalsh(matrix)[0],atol=1e-9)
    assert rank==root.size
    assert res<=opts.eigensolver_tolerance*max(1.,abs(e))
    np.testing.assert_allclose(np.vdot(v,pair.apply_hamiltonian(v))/np.vdot(v,pair.apply_metric(v)),e,atol=1e-9)


def test_pair_root_rejects_unconverged_ritz_vector():
    state,ham,pair=setup_pair()
    opts=LETTATwoSiteOptions(metric_tolerance=1e-12,eigensolver_tolerance=1e-12,dense_solver_threshold=0,eigensolver_max_iterations=1)
    with pytest.raises(FloatingPointError,match='pair eigensolver did not converge'):
        solver._solve_reduced_pair_root(pair,state,ham,opts)


def test_coordinate_geometry_cache_does_not_cache_state_factors():
    from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates
    state,ham,pair=setup_pair()
    embedding=pair.frontier.site_embedding(state,1)
    sites=pair.frontier.to_mps(state)
    old=ReducedLocalCoordinates(sites,embedding,1)
    routes=embedding.coordinate_routes
    changed=[site.copy() for site in sites]
    changed[0].data={key:1.3*block for key,block in changed[0].data.items()}
    cached=ReducedLocalCoordinates(changed,embedding,1)
    assert embedding.coordinate_routes is routes
    embedding.__dict__.pop('coordinate_routes')
    fresh=ReducedLocalCoordinates(changed,embedding,1)
    assert embedding.coordinate_routes is not routes
    vector=np.random.default_rng(63).normal(size=embedding.source_size)
    np.testing.assert_array_equal(cached.metric(vector),fresh.metric(vector))
    np.testing.assert_allclose(cached.metric(vector),1.3**2*old.metric(vector),atol=1e-12)


def test_frontier_embedding_cache_tracks_layout_and_current_values():
    state,ham,pair=setup_pair()
    frontier=pair.frontier
    first=frontier.site_embedding(state,1)
    changed=state.copy()
    changed.tensors[1]={key:1.7*value for key,value in changed.tensors[1].items()}
    assert frontier.site_embedding(changed,1) is first
    before=frontier.to_mps(state)[1].data
    after=frontier.to_mps(changed)[1].data
    for key in before:np.testing.assert_allclose(after[key],1.7*before[key])
    # A layout change replaces the map and drops its coordinate-route cache.
    key=next(iter(changed.tensors[1]))
    changed.tensors[1][key]=np.repeat(changed.tensors[1][key],2,axis=-1)
    second=frontier.site_embedding(changed,1)
    assert second is not first
    fresh=type(frontier).from_state(changed).site_embedding(changed,1)
    np.testing.assert_array_equal(second.source_indices,fresh.source_indices)
    np.testing.assert_array_equal(second.target_indices,fresh.target_indices)
    assert len(frontier._embedding_cache)<=state.nsites


def test_frontier_cache_eviction_preserves_exact_embedding():
    state,ham,pair=setup_pair()
    frontier=pair.frontier
    first=frontier.site_embedding(state,1)
    frontier._embedding_cache_budget=0
    # A changed layout forces admission/eviction rather than a cache hit.
    changed=state.copy();key=next(iter(changed.tensors[1]))
    changed.tensors[1][key]=np.repeat(changed.tensors[1][key],2,axis=-1)
    actual=frontier.site_embedding(changed,1)
    assert not frontier._embedding_cache
    assert frontier._embedding_cache_bytes==0
    fresh=type(frontier).from_state(changed).site_embedding(changed,1)
    np.testing.assert_array_equal(actual.source_indices,fresh.source_indices)
    np.testing.assert_array_equal(actual.target_indices,fresh.target_indices)

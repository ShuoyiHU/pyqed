"""A pair update must keep its fixed exterior available for exact reuse."""
import numpy as np
import pytest
from test_letta_reduced_compression import example
from test_letta_qchem import integrals
from pyqed._letta_one_site_opt import LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import _energy
from pyqed._letta_one_site_opt.reduced_cbe import reduced_cbe_site
from pyqed._letta_two_site_opt import LETTATwoSiteOptions
from pyqed._letta_two_site_opt.reduced_solver import _optimize_reduced_pair


@pytest.mark.parametrize('side',[0,1])
def test_factor_coordinates_skip_unused_center_without_changing_maps(monkeypatch,side):
    from pyqed._letta_one_site_opt import reduced_coordinates as rc
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root, _physical_factor_map
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    problem=reduced_pair_problem(state,ham,1,matrix_free=True,dense_solver_threshold=0)
    root=pair_metric_root(problem,state,1e-12)
    sites=problem.frontier.to_mps(state,compact=True)
    embedding=problem.frontier.site_embedding(state,1+side)
    expected=rc.ReducedLocalCoordinates(sites,embedding,1+side,
        retain_outer=False,boundary_seeds=root.boundary_seeds)
    contract=rc.contract_block
    def checked(block,left=None,right=None):
        assert left is None or right is None, 'Unused active center was transformed'
        return contract(block,left,right)
    monkeypatch.setattr(rc,'contract_block',checked)
    actual=rc.ReducedLocalCoordinates(sites,embedding,1+side,
        retain_outer=False,boundary_seeds=root.boundary_seeds,factor_only=True)
    assert actual.sites[1+side] is None
    expected.prepare(1e-12);actual.prepare(1e-12)
    a,b=_physical_factor_map(actual,root,1,side),_physical_factor_map(expected,root,1,side)
    rng=np.random.default_rng(42)
    z=rng.normal(size=actual.rank)+1j*rng.normal(size=actual.rank)
    y=rng.normal(size=root.physical_layout.size)+1j*rng.normal(size=root.physical_layout.size)
    np.testing.assert_array_equal(a[0](z),b[0](z))
    np.testing.assert_array_equal(a[1](y),b[1](y))


@pytest.mark.parametrize('method', ['pair', 'cbe'])
def test_pair_transactions_preserve_fixed_exterior(method):
    state, *_ = example()
    ham = ElectronicProblem(*integrals(3), (2, 1)).su2_mpo()
    exterior = {k: v.copy() for k, v in state.tensors[0].items()}
    old = _energy(state, ham, stable=True)
    if method == 'pair':
        result = _optimize_reduced_pair(state, ham, 1, 'lr', 4,
            LETTATwoSiteOptions(gauge_mode='none', energy_refinement_max_iterations=3))
    else:
        result = reduced_cbe_site(state, ham, 1, 'lr', 4,
            LETTADMROptions(gauge_mode='none', cbe_enabled=True,
                cbe_energy_refinement_max_iterations=3))
    assert result.accepted
    assert _energy(state, ham, stable=True) == pytest.approx(result.energy, abs=2e-10)
    assert result.energy <= old + 1e-10
    assert state.norm() == pytest.approx(1., abs=2e-10)
    for key, value in exterior.items():
        np.testing.assert_array_equal(state.tensors[0][key], value)


def test_boundary_cache_uses_local_support_not_global_fingerprint():
    from pyqed._letta_one_site_opt import ReducedFrontier
    from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache, canonical_environment
    state, *_ = example()
    mpo=ElectronicProblem(*integrals(3),(2,1)).su2_mpo().native_mpo(state.physical_basis)
    sites=ReducedFrontier.from_state(state).to_mps(state,compact=True)
    cache=CanonicalBoundaryCache()
    canonical_environment(sites,1,mpo,cache=cache)
    count=cache.advances
    for site in sites:
        site.metadata['letta_frontier_support']='new global graph, unchanged local coordinates'
    canonical_environment(sites,1,mpo,cache=cache)
    assert cache.advances==count


@pytest.mark.parametrize('pending', [0,100000])
def test_cached_operator_boundaries_follow_changed_allocations(monkeypatch,pending):
    from test_letta_reduced_cbe import incomplete, controls
    from pyqed._letta_one_site_opt.reduced_cbe import select_reduced_cbe, expand_reduced_cbe
    from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES',str(pending))
    state,ham=incomplete()
    context=DependencySweepContext(state,ham,structured=False)
    options=controls(gauge_mode='none')
    context.local_problem(0,options)
    selection=select_reduced_cbe(state,ham,0,options,boundary_cache=context.boundary_cache)
    expanded=expand_reduced_cbe(state,selection,'lr')
    shared=context.for_state(expanded,center=0).local_problem(0,options)
    fresh=DependencySweepContext(expanded,ham,structured=False).local_problem(0,options)
    x=np.random.default_rng(14).normal(size=shared.local_dimension)
    np.testing.assert_allclose(shared.apply_hamiltonian(x),fresh.apply_hamiltonian(x),atol=1e-12)
    np.testing.assert_allclose(shared.apply_metric(x),fresh.apply_metric(x),atol=1e-12)


def test_boundary_signatures_track_numerical_inputs_and_filtered_kernels():
    from dataclasses import replace
    from pyqed._letta_one_site_opt import ReducedFrontier
    from pyqed._letta_one_site_opt.frontier_contraction import CanonicalBoundaryCache
    from pyqed._letta_one_site_opt.reduced_environment import chain_transfers
    state,*_=example()
    mpo=ElectronicProblem(*integrals(3),(2,1)).su2_mpo().native_mpo(state.physical_basis)
    sites=ReducedFrontier.from_state(state).to_mps(state,compact=True)
    transfers=chain_transfers(sites,mpo)[0]
    cache=CanonicalBoundaryCache()
    old=cache.signatures(sites,mpo,transfers)
    changed=list(transfers)
    changed[1]=(replace(transfers[1][0],kernel=1.1*transfers[1][0].kernel),*transfers[1][1:])
    new=cache.signatures(sites,mpo,changed)
    assert old[0]==new[0] and old[1]!=new[1] and old[2]==new[2]
    key=next(iter(sites[0].data))
    block=sites[0].data[key]
    sites[0].data[key]=np.asarray(block)*1.1
    assert cache.signatures(sites,mpo,transfers)[0]!=old[0]


def test_norm_only_pair_does_not_build_h_environment_but_preserves_actions(monkeypatch):
    from pyqed._letta_one_site_opt import frontier_contraction as fc
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem, _orthogonal_pair_action
    from pyqed._letta_two_site_opt.reduced_compression import pair_metric_root
    from pyqed._letta_two_site_opt.pair_frontier import ReducedPairFrontier
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    monkeypatch.setattr(ReducedPairFrontier,'requires_compact',lambda *a:True)
    old=reduced_pair_problem(state,ham,1,matrix_free=True,dense_solver_threshold=0)
    root=pair_metric_root(old,state,1e-12)
    calls=[]
    original=fc.canonical_environment
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(fc,'canonical_environment',counted)
    new=reduced_pair_problem(state,ham,1,norm_only=True)
    assert new.hamiltonian is None and new.metric is None
    newroot=pair_metric_root(new,state,1e-12)
    x=np.random.default_rng(32).normal(size=old.local_dimension)
    np.testing.assert_allclose(newroot.physical_apply(x),root.physical_apply(x),atol=1e-12)
    assert not calls
    z=np.random.default_rng(33).normal(size=root.size)
    np.testing.assert_allclose(_orthogonal_pair_action(new,newroot,state,ham)(z),
        _orthogonal_pair_action(old,root,state,ham)(z),atol=2e-11)
    assert len(calls)==1


def test_refinement_reuses_source_energy_and_keeps_independent_final_check(monkeypatch):
    from pyqed._letta_one_site_opt import reduced_solver as one
    from pyqed._letta_one_site_opt.reduced_updates import refine_reduced_pair_energy
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    options=LETTADMROptions(gauge_mode='none')
    calls=[]
    original=one._energy
    def counted(*a,**kw):
        calls.append(1)
        return original(*a,**kw)
    monkeypatch.setattr(one,'_energy',counted)
    expected=refine_reduced_pair_energy(state,ham,1,options,max_iterations=4,
        tolerance=1e-10,reuse_local_energy=False)
    n=len(calls);calls.clear()
    actual=refine_reduced_pair_energy(state,ham,1,options,max_iterations=4,
        tolerance=1e-10,reuse_local_energy=True)
    assert 2<=len(calls)<n
    np.testing.assert_allclose(actual.energies,expected.energies,atol=2e-11,rtol=0)
    assert actual.energy==pytest.approx(original(actual.state,ham,stable=True),abs=2e-11)
    assert actual.iterations==expected.iterations


def test_refinement_rejects_incorrect_source_energy_certificate(monkeypatch):
    from pyqed._letta_one_site_opt.reduced_solver import StableEnergyCache
    from pyqed._letta_one_site_opt.reduced_updates import refine_reduced_pair_energy
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    remember=StableEnergyCache.remember
    monkeypatch.setattr(StableEnergyCache,'remember',
        lambda self,state,energy:remember(self,state,energy-1.))
    with pytest.raises(FloatingPointError,match='Independent pair energy disagrees'):
        refine_reduced_pair_energy(state,ham,1,LETTADMROptions(gauge_mode='none'),
            max_iterations=1,tolerance=1e-10)


@pytest.mark.parametrize('pending',[0,100000])
def test_cached_boundaries_after_remote_support_changes(monkeypatch,pending):
    from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext
    monkeypatch.setenv('PYQED_OPERATOR_PENDING_EDGES',str(pending))
    state,*_=example()
    ham=ElectronicProblem(*integrals(3),(2,1)).su2_mpo()
    context=DependencySweepContext(state,ham,structured=False)
    options=LETTADMROptions(gauge_mode='none')
    context.local_problem(0,options)
    state.tensors[-1].pop(next(iter(state.tensors[-1])))
    actual=context.local_problem(0,options)
    reference=DependencySweepContext(state,ham,structured=False).local_problem(0,options)
    x=np.random.default_rng(45).normal(size=actual.local_dimension)
    np.testing.assert_allclose(actual.apply_hamiltonian(x),reference.apply_hamiltonian(x),atol=2e-11)
    np.testing.assert_allclose(actual.apply_metric(x),reference.apply_metric(x),atol=2e-11)

"""Dependency-aware reduced gauges: full shared frontiers or legal marginals."""
import numpy as np
import pytest
from dataclasses import replace
from collections import Counter

from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions, letta_dmrg, abelian_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, initial_state, embed_ties
from pyqed._letta_one_site_opt.reduced_gauge import (
    reduced_gauge_variables, reduced_frontier_grams, shift_reduced_frontier_gauge,
    canonicalize_reduced_frontier)
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
from pyqed._letta_one_site_opt.reduced_solver import ReducedSweepContext, reduced_local_problem, _energy
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from test_letta_qchem import integrals, determinant_hamiltonian

TIES = ((0,3), (1,0), (2,1,3), (3,0))


def example(two_s=0):
    p = ElectronicProblem(*integrals(4), (2,2))
    s = ReducedLatticeLETTA.random((1,4), symmetry=p.symmetry('su2',two_s=two_s),
        neighborhoods=TIES, multiplets_per_sector=2, seed=448, real=False)
    return p, s


@pytest.mark.parametrize('direction',['lr','rl'])
@pytest.mark.parametrize('two_s',[0,2])
def test_general_gauge_preserves_every_target_component_and_whitens_marginal(direction,two_s):
    p,s=example(two_s)
    original={m:s.state_vector(target_two_m=m) for m in range(-two_s,two_s+1,2)}
    bonds=s.bond_sectors
    cut=2
    variables=reduced_gauge_variables(s,cut)
    assert variables==(1,)
    frontier=ReducedFrontier.from_state(s)
    assert set(variables)<set(frontier.cuts[cut-1])
    chain=ReducedNormChain.build(frontier.to_mps(s))
    env=chain.left[cut] if direction=='lr' else chain.right[cut]
    log=chain.left_log_scales[cut] if direction=='lr' else chain.right_log_scales[cut]
    expected={}
    for memory,config in enumerate(frontier._assignments(frontier.cuts[cut-1])):
        shared=tuple(dict(zip(frontier.cuts[cut-1],config))[v] for v in variables)
        for q,r in Counter(bonds[cut-1]).items():
            part=env[q][memory*r:(memory+1)*r,memory*r:(memory+1)*r]*np.exp(log)
            expected[shared,q]=expected.get((shared,q),0.)+part
    for key,gram in reduced_frontier_grams(s,cut,direction).items():
        np.testing.assert_allclose(gram,expected[key],atol=2e-12)
    shift_reduced_frontier_gauge(s,cut,direction)
    for gram in reduced_frontier_grams(s,cut,direction).values():
        values=np.linalg.eigvalsh(gram)
        np.testing.assert_allclose(values[values>1e-8],1.,atol=3e-9)
    for m,vector in original.items():
        np.testing.assert_allclose(s.state_vector(target_two_m=m),vector,atol=3e-11)
    assert s.bond_sectors==bonds
    assert tuple(s.site_neighborhood(i) for i in range(4))==TIES
    assert s.symmetry_violation()==0.


def test_full_frontier_remains_correlated_after_marginal_gauge():
    p,s=example()
    shift_reduced_frontier_gauge(s,2,'lr')
    chain=ReducedNormChain.build(ReducedFrontier.from_state(s).to_mps(s))
    blocks=[g*np.exp(chain.left_log_scales[2]) for g in chain.left[2].values()]
    assert max(np.linalg.norm(g-np.eye(len(g))) for g in blocks)>1e-2
    # Local solver must keep this metric, rather than assuming N=I.
    problem=reduced_local_problem(s,p.su2_mpo(),2,matrix_free=False)
    assert problem.metric is not None
    assert np.linalg.norm(problem.metric-np.eye(problem.local_dimension))>1e-2


def test_strict_full_frontier_mode_rejects_before_any_mutation():
    _,s=example()
    old=[[a.copy() for a in core.values()] for core in s.tensors]
    with pytest.raises(ValueError,match='shared frontier'):
        canonicalize_reduced_frontier(s,2,strict=True)
    for before,core in zip(old,s.tensors):
        for a,b in zip(before,core.values()):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('direction',['lr','rl'])
def test_general_gauge_with_cached_environments_matches_fresh_actions(direction):
    p,s=example();h=p.su2_mpo();context=ReducedSweepContext(s,h)
    shift_reduced_frontier_gauge(s,2,direction,environment=(context.n_chain,context.frontier))
    context.synchronize([1,2])
    rng=np.random.default_rng(77)
    for site in (1,2):
        a=context.local_problem(site,LETTADMROptions(dense_solver_threshold=1))
        b=reduced_local_problem(s,h,site,matrix_free=True,dense_solver_threshold=1)
        x=rng.normal(size=a.local_dimension)+1j*rng.normal(size=a.local_dimension)
        np.testing.assert_allclose(a.apply_metric(x),b.apply_metric(x),atol=2e-10)
        np.testing.assert_allclose(a.apply_hamiltonian(x),b.apply_hamiltonian(x),atol=2e-9)


@pytest.mark.parametrize('symmetry',['su2','nalpha_nbeta'])
@pytest.mark.parametrize('method',['one-site','cbe','two-site'])
def test_arbitrary_ties_work_with_default_gauge_for_all_symmetry_updates(symmetry,method):
    p=ElectronicProblem(*integrals(3),(2,1),.17)
    ties=((0,2),(1,0),(2,1))
    if symmetry=='su2':
        s=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),neighborhoods=ties,seed=75)
        h=p.su2_mpo();before=_energy(s,h,stable=True)
    else:
        s=embed_ties(initial_state(p,max_bond_dim=8,symmetry=symmetry),ties)
        h=p.mpo();v=s.state_vector();before=np.vdot(v,determinant_hamiltonian(p)@v).real
    opts=(LETTATwoSiteOptions(max_sweeps=2,reduced_sector_growth=True,energy_refinement_max_iterations=2)
        if method=='two-site' else LETTADMROptions(max_sweeps=3,cbe_enabled=method=='cbe',cbe_energy_refinement_max_iterations=2))
    if symmetry=='su2':
        result=(letta_two_site_dmrg(h,state=s,bond_dim=8,options=opts) if method=='two-site'
                else letta_dmrg(h,state=s,options=opts))
    else:
        result=abelian_dmrg(h,state=s,options=opts,bond_dim=8 if method=='two-site' else None)
    v=result.state.state_vector()
    energy=np.vdot(v,determinant_hamiltonian(p)@v).real/np.vdot(v,v).real
    assert result.energy==pytest.approx(energy,abs=3e-9)
    assert result.energy<=before+1e-9
    assert result.state.symmetry_violation()==0.
    assert tuple(result.state.site_neighborhood(i) for i in range(3))==ties


@pytest.mark.parametrize('whole_sweep',[False,True])
def test_gauge_failure_restores_both_cores_and_previous_gauge_steps(monkeypatch,whole_sweep):
    import pyqed._letta_one_site_opt.reduced_gauge as gauge
    _,s=example()
    old=[{k:a.copy() for k,a in core.items()} for core in s.tensors]
    original=gauge._apply_conditional
    calls=0
    def fail(state,site,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==(4 if whole_sweep else 2):
            # Emulate a numerical operation that failed after a partial write.
            for a in state.tensors[site].values():a.fill(np.nan)
            raise FloatingPointError('injected gauge failure')
        return original(state,site,*args,**kwargs)
    monkeypatch.setattr(gauge,'_apply_conditional',fail)
    with pytest.raises(FloatingPointError,match='injected gauge failure'):
        if whole_sweep:canonicalize_reduced_frontier(s,2)
        else:shift_reduced_frontier_gauge(s,2,'lr')
    for before,core in zip(old,s.tensors):
        for key in before:np.testing.assert_array_equal(before[key],core[key])


@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('two_s', [0, 2])
def test_frontier_qr_factors_reproduce_complex_marginal_grams(direction, two_s):
    from pyqed._letta_one_site_opt.reduced_gauge import reduced_frontier_factors
    _, state = example(two_s)
    for cut in range(1, state.nsites):
        grams = reduced_frontier_grams(state, cut, direction)
        factors = reduced_frontier_factors(state, cut, direction)
        assert grams.keys() == factors.keys()
        for key, factor in factors.items():
            np.testing.assert_allclose(factor.conj().T@factor, grams[key], atol=2e-12, rtol=2e-12)


def test_qr_gauge_preserves_captured_ill_conditioned_ladder():
    import pickle
    from pathlib import Path
    with (Path(__file__).parent/'data/letta_overlap_2x4_d20.pkl').open('rb') as stream:
        state = pickle.load(stream)['state']
    original = state.state_vector()
    for center in (0, state.nsites-1):
        canonicalize_reduced_frontier(state, center)
        np.testing.assert_allclose(state.state_vector(), original, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_gauge_does_not_amplify_negligible_sector(direction):
    _, state = example()
    site, axis, cut = (0, 2, 1) if direction == 'lr' else (3, 0, 3)
    sector = next(iter(state.tensors[site]))[axis]
    assert len({key[axis] for key in state.tensors[site]}) > 1
    for key, block in state.tensors[site].items():
        if key[axis] == sector:
            block *= 1e-9
    original = state.state_vector()
    ranks = shift_reduced_frontier_gauge(state, cut, direction)
    selected = [rank for (_, q), rank in ranks.items() if q == sector]
    assert selected and all(rank == 0 for rank in selected)
    np.testing.assert_allclose(state.state_vector(), original, atol=3e-11, rtol=3e-11)

"""Sector allocation reuses independent metric fits without changing tolerances."""
from collections import Counter
import numpy as np
import pytest
from test_letta_reduced_cbe import controls, incomplete
from test_letta_qchem import integrals
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt import reduced_cbe as cbe
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot,compress_reduced_pair
from pyqed._letta_compression import MetricCompressionOptions

@pytest.mark.parametrize('budget',[3,9])
def test_sector_profiles_are_fitted_once_per_rank(monkeypatch,budget):
 state,ham=incomplete()
 original=cbe.compress_reduced_pair
 calls=Counter()
 def counted(target,problem,state,left,right,allocation,**kwargs):
  assert len(allocation)==1
  assert kwargs.get('root') is not None
  calls[tuple(allocation.items())]+=1
  return original(target,problem,state,left,right,allocation,**kwargs)
 monkeypatch.setattr(cbe,'compress_reduced_pair',counted)
 selection=cbe.select_reduced_cbe(state,ham,0,controls(cbe_expansion_dimension=budget))
 assert selection.multiplicities
 assert selection.factor_fit_attempts==sum(calls.values())
 assert max(calls.values())<=3
 assert selection.diagnostics[0]['allocation_method']=='cached-sector-metric-fits'
 assert selection.captured_weight>0
 assert sum(selection.multiplicities.values())<=budget
 for direction in ('lr','rl'):
  expanded=cbe.expand_reduced_cbe(state,selection,direction)
  np.testing.assert_allclose(expanded.state_vector(),state.state_vector(),atol=2e-12)

def test_reused_outer_root_matches_fresh_compression():
 p=ElectronicProblem(*integrals(3),(2,1))
 s=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),neighborhoods=((0,2),(1,0),(2,1)),real=False,seed=31)
 pair=reduced_pair_problem(s,p.su2_mpo(),1,matrix_free=True,dense_solver_threshold=0)
 root=ReducedPairMetricRoot(pair,s,1e-12)
 le,re=(pair.frontier.site_embedding(s,k) for k in (1,2))
 rng=np.random.default_rng(19)
 target=rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
 a=le.pack_source(s.tensors[1]);b=re.pack_source(s.tensors[2]);allocation=Counter(s.bond_sectors[1])
 options=MetricCompressionOptions(als_max_iterations=3)
 fresh=compress_reduced_pair(target,pair,s,a,b,allocation,options=options)
 reused=compress_reduced_pair(target,pair,s,a,b,allocation,options=options,root=root)
 assert reused.loss==pytest.approx(fresh.loss,rel=1e-12,abs=1e-12)
 np.testing.assert_allclose(reused.left,fresh.left,rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(reused.right,fresh.right,rtol=1e-12,atol=1e-12)


def test_middle_sector_restriction_preserves_physical_metric():
    p=ElectronicProblem(*integrals(3),(2,1))
    s=ReducedLatticeLETTA.random((1,3),symmetry=p.symmetry('su2'),
        neighborhoods=((0,2),(1,0),(2,1)),real=False,seed=31)
    pair=reduced_pair_problem(s,p.su2_mpo(),1,matrix_free=True,dense_solver_threshold=0)
    root=ReducedPairMetricRoot(pair,s,1e-12)
    rng=np.random.default_rng(31)
    x=rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
    blocks=pair.layout.unpack(x)
    total=0.
    for q in {k[2] for k in pair.layout.shapes}:
        local=root.restrict_middle_sector(q)
        v=local.input_layout.pack({k:blocks[k] for k in local.input_layout.shapes})
        physical=local.physical_apply(v)
        total+=np.linalg.norm(physical)**2
        full=pair.layout.pack({k:blocks[k] if k[2]==q else np.zeros_like(blocks[k]) for k in blocks})
        np.testing.assert_allclose(np.linalg.norm(physical),np.linalg.norm(root.physical_apply(full)),atol=1e-12)
        full_adjoint=pair.layout.unpack(root.physical_adjoint(root.physical_apply(full)))
        np.testing.assert_allclose(local.physical_adjoint(physical),
            local.input_layout.pack({k:full_adjoint[k] for k in local.input_layout.shapes}),atol=1e-12)
    assert total==pytest.approx(np.linalg.norm(root.physical_apply(x))**2,rel=1e-12)


def test_compression_sector_allocation_is_invariant_under_external_gauge():
    from pyqed._letta_two_site_opt.reduced_solver import _metric_pair_split, _split_reduced_pair
    p=ElectronicProblem(*integrals(4),(2,2))
    h=p.su2_mpo()
    state=ReducedLatticeLETTA.random((2,2),symmetry=p.symmetry('su2'),
        neighborhoods=((0,2),(1,3),(2,),(3,)),real=False,seed=31)
    expected=None
    raw_allocations=[]
    for exponent in (-4,0,4):
        s=state.copy()
        sectors=sorted(set(s.bond_sectors[0]))
        scales={q:10.**(exponent*(j-len(sectors)/2)) for j,q in enumerate(sectors)}
        s.tensors[0]={k:v*scales[k[2]] for k,v in s.tensors[0].items()}
        s.tensors[1]={k:v/scales[k[0]] for k,v in s.tensors[1].items()}
        np.testing.assert_allclose(s.state_vector(),state.state_vector(),atol=1e-12)
        pair=reduced_pair_problem(s,h,1,matrix_free=True,dense_solver_threshold=0)
        root=ReducedPairMetricRoot(pair,s,1e-12)
        sites=pair.frontier.to_mps(s)
        kwargs=dict(bond_dim=2,sector_capacities=Counter(s.bond_sectors[1]),direction='lr',cutoff=1e-12)
        split=_metric_pair_split(pair,root,pair.old_vector,sites,**kwargs)
        allocation=dict(split.retained_multiplicities)
        if expected is None:expected=allocation
        assert allocation==expected
        assert sum(allocation.values())==2
        raw=_split_reduced_pair(pair.layout.unpack(pair.old_vector),sites[1],sites[2],**kwargs)
        raw_allocations.append(dict(raw.retained_multiplicities))
    # The previous raw-coordinate criterion demonstrably changes the decision
    # under a gauge transformation that leaves the physical state unchanged.
    assert raw_allocations[0]!=raw_allocations[-1]


def test_sector_tangent_svd_fits_budget_that_excludes_full_matrix():
    from pyqed._letta_one_site_opt.reduced_cbe import _tangent_coordinates
    from pyqed._letta_one_site_opt.tangent_projection import project_tangent,_factor_tangent_matrix
    p=ElectronicProblem(*integrals(4),(2,2))
    s=ReducedLatticeLETTA.random((2,2),symmetry=p.symmetry('su2'),
        neighborhoods=((0,2),(1,3),(2,),(3,)),real=False,seed=31)
    pair=reduced_pair_problem(s,p.su2_mpo(),1,matrix_free=True,dense_solver_threshold=0)
    root=ReducedPairMetricRoot(pair,s,1e-12)
    embeddings=tuple(pair.frontier.site_embedding(s,k) for k in (1,2))
    maps=_tangent_coordinates(s,pair,embeddings,Counter(s.bond_sectors[1]),controls())
    rng=np.random.default_rng(17)
    gradient=rng.normal(size=root.size)+1j*rng.normal(size=root.size)
    kwargs=dict(rank_tolerance=1e-12,normal_tolerance=1e-9)
    answer,report=project_tangent(maps,root,1,gradient,workspace_mb=256.,**kwargs)
    full_bytes=6*np.dtype(complex).itemsize*root.size*sum(c.rank for c in maps)
    assert report['workspace_bytes']<full_bytes
    limit=(report['workspace_bytes']+full_bytes)/2/1024**2
    bounded,bounded_report=project_tangent(maps,root,1,gradient,workspace_mb=limit,**kwargs)
    assert bounded_report['materialized'] and bounded_report['sector_blocks']>1
    matrix=np.concatenate([_factor_tangent_matrix(c,root,1,side) for side,c in enumerate(maps)],axis=1)
    u,singular,_=np.linalg.svd(matrix,full_matrices=False)
    q=u[:,singular>kwargs['rank_tolerance']*singular.max()]
    np.testing.assert_allclose(bounded,gradient-q@(q.conj().T@gradient),atol=2e-12)
    np.testing.assert_allclose(answer,bounded,atol=2e-12)

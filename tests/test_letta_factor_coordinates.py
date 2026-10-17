import numpy as np
import pytest
from collections import Counter
from test_letta_reduced_compression import example
from pyqed._letta_two_site_opt.reduced_compression import _factor_coordinates
from pyqed._letta_compression import MetricCompressionOptions
def factor_map(*args):return _factor_coordinates(*args,MetricCompressionOptions())
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
from pyqed._letta_two_site_opt.reduced_solver import _pair_vector_from_sources,_active_source_indices

@pytest.mark.parametrize('ties',[None,((0,),(1,),(2,)),((0,2),(1,0),(2,1))])
@pytest.mark.parametrize('scale',[1.,1e6])
@pytest.mark.parametrize('multiplicity',[1,2])
@pytest.mark.parametrize('side',[0,1])
def test_factor_projection_matches_full_physical_least_squares(ties,scale,side,multiplicity):
 if multiplicity==1:
  s,p,le,re,a,b,retained=example(ties)
 else:
  from pyqed._letta_one_site_opt import ReducedLatticeLETTA
  from pyqed._letta_one_site_opt.qchem import ElectronicProblem
  from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
  from test_letta_qchem import integrals
  model=ElectronicProblem(*integrals(3),(2,1))
  s=ReducedLatticeLETTA.random((1,3),symmetry=model.symmetry('su2'),neighborhoods=ties,
      multiplets_per_sector=2,real=False,seed=87)
  p=reduced_pair_problem(s,model.su2_mpo(),1,matrix_free=True,dense_solver_threshold=0)
  le,re=(p.frontier.site_embedding(s,i) for i in (1,2))
  retained=Counter(s.bond_sectors[1])
 # An internal gauge changes conditioning but not the target state.
 for key,v in s.tensors[1].items():
  s.tensors[1][key]=v*np.geomspace(1.,scale,v.shape[-1])
 for key,v in s.tensors[2].items():
  s.tensors[2][key]=v/np.geomspace(1.,scale,v.shape[0]).reshape((-1,)+(1,)*(v.ndim-1))
 a=le.pack_source(s.tensors[1]);b=re.pack_source(s.tensors[2])
 root=ReducedPairMetricRoot(p,s,1e-12)
 c,apply,adjoint=factor_map(s,p,root,le,re,a,b,retained,side)
 rng=np.random.default_rng(102);z=rng.normal(size=c.rank)+1j*rng.normal(size=c.rank)
 raw=c.expand(z)
 merge=lambda x,y:_pair_vector_from_sources(p.layout,le,re,x,y)
 expected=root.physical_apply(merge(raw,b) if side==0 else merge(a,raw))
 np.testing.assert_allclose(apply(z),expected,atol=1e-10)
 np.testing.assert_allclose(adjoint(apply(z)),z,atol=1e-10)
 from pyqed._letta_one_site_opt.tangent_projection import _factor_tangent_matrix as factor_matrix
 matrix=factor_matrix(c,root,p.left_site,side)
 np.testing.assert_allclose(matrix@z,root.project_physical(apply(z)),atol=1e-10)
 w=rng.normal(size=root.size)+1j*rng.normal(size=root.size)
 np.testing.assert_allclose(matrix.conj().T@w,adjoint(root.embed_physical(w)),atol=1e-10)
 rhs=rng.normal(size=root.physical_layout.size)+1j*rng.normal(size=root.physical_layout.size)
 assert abs(np.vdot(apply(z),rhs)-np.vdot(z,adjoint(rhs)))<1e-10
 indices=_active_source_indices(le if side==0 else re,retained,'left' if side==0 else 'right')
 columns=[]
 for j in indices:
  v=np.zeros_like(a if side==0 else b);v[j]=1
  columns.append(root.physical_apply(merge(v,b) if side==0 else merge(a,v)))
 matrix=np.column_stack(columns);norm=np.linalg.norm(matrix,axis=0)
 inv=np.divide(1.,norm,out=np.zeros_like(norm),where=norm>0)
 balanced=matrix*inv
 old=a if side==0 else b
 correction=c.least_squares_correction(adjoint(rhs)-c.coordinates(old))
 expected_correction=np.linalg.lstsq(matrix,rhs-matrix@old[indices],rcond=1e-12)[0]
 np.testing.assert_allclose(correction[indices],expected_correction,atol=1e-8,rtol=1e-8)
 solution=np.linalg.lstsq(balanced,rhs,rcond=1e-12)[0]
 np.testing.assert_allclose(apply(adjoint(rhs)),balanced@solution,atol=1e-10)


def test_small_workspace_uses_lsmr_without_changing_objective():
 from pyqed._letta_two_site_opt.reduced_compression import compress_reduced_pair
 s,p,le,re,a,b,retained=example()
 rng=np.random.default_rng(51)
 target=p.old_vector+.03*(rng.normal(size=p.local_dimension)+1j*rng.normal(size=p.local_dimension))
 opts=MetricCompressionOptions(als_max_iterations=2,max_workspace_mb=1e-6)
 r=compress_reduced_pair(target,p,s,a,b,retained,options=opts)
 assert all(x['solver']=='lsmr' and 'workspace' in x['fallback_reason'] for x in r.diagnostics['linear_solves'])
 root=ReducedPairMetricRoot(p,s,1e-12)
 residual=root.physical_apply(_pair_vector_from_sources(p.layout,le,re,r.left,r.right)-target)
 np.testing.assert_allclose(r.loss,np.linalg.norm(residual)**2,atol=1e-12)


def test_projection_workspace_guard_precedes_matrix_allocation(monkeypatch):
 from pyqed._letta_one_site_opt import tangent_projection
 s,p,le,re,a,b,retained=example()
 root=ReducedPairMetricRoot(p,s,1e-12)
 maps=[factor_map(s,p,root,le,re,a,b,retained,side)[0] for side in (0,1)]
 def forbidden(*args):raise AssertionError('allocated tangent matrix despite zero workspace')
 monkeypatch.setattr(tangent_projection,'_factor_tangent_matrix',forbidden)
 residual,report=tangent_projection.project_tangent(maps,root,p.left_site,np.ones(root.size),
     workspace_mb=0.,rank_tolerance=1e-12,normal_tolerance=1e-10)
 assert residual is None and not report['materialized']
 assert 'workspace' in report['fallback_reason']


def test_projection_svd_failure_requests_iterative_fallback(monkeypatch):
 from pyqed._letta_one_site_opt import tangent_projection
 s,p,le,re,a,b,retained=example()
 root=ReducedPairMetricRoot(p,s,1e-12)
 maps=[factor_map(s,p,root,le,re,a,b,retained,side)[0] for side in (0,1)]
 def failed(*args,**kwargs):raise np.linalg.LinAlgError('SVD did not converge')
 monkeypatch.setattr(tangent_projection.np.linalg,'svd',failed)
 residual,report=tangent_projection.project_tangent(maps,root,p.left_site,np.ones(root.size),
     workspace_mb=256.,rank_tolerance=1e-12,normal_tolerance=1e-10)
 assert residual is None and report['materialized']
 assert 'LinAlgError' in report['fallback_reason']

def test_qr_als_preserves_a_factor_when_its_partner_is_zero():
    from pyqed._letta_two_site_opt.reduced_compression import compress_reduced_pair
    state, pair, le, re, left, right, retained = example()
    fit = compress_reduced_pair(pair.old_vector, pair, state, left, np.zeros_like(right), retained,
        options=MetricCompressionOptions(als_max_iterations=1, tolerance=1e-12))
    # With the right factor zero, every left vector is an exact linear LS
    # minimizer. Preserve the incumbent so the right half-step can recover.
    assert fit.loss < 1e-20
    assert np.linalg.norm(fit.left) > 0


@pytest.mark.parametrize('side',[0,1])
def test_reused_outer_factor_sites_match_fresh_partner_coordinates(side):
    s,p,le,re,a,b,retained=example()
    root=ReducedPairMetricRoot(p,s,1e-12)
    sites=p.frontier.to_mps(s)
    originals=[{k:v.copy() for k,v in t.data.items()} for t in sites]
    a=1.3*a;b=b/1.3
    args=(s,p,root,le,re,a,b,retained,side,MetricCompressionOptions())
    fresh,forward,_=_factor_coordinates(*args)
    cached,cached_forward,_=_factor_coordinates(*args,sites=sites)
    assert fresh.rank==cached.rank
    z=np.random.default_rng(51).normal(size=fresh.rank)
    np.testing.assert_array_equal(fresh.expand(z),cached.expand(z))
    np.testing.assert_array_equal(forward(z),cached_forward(z))
    for site,original in zip(sites,originals):
        for key in original:np.testing.assert_array_equal(site.data[key],original[key])

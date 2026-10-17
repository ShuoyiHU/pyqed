"""Independent small-system references only; no dense production solver path."""
from dataclasses import replace
import numpy as np
import pytest
from test_letta_qchem import determinant_hamiltonian
from test_letta_qchem_symmetry import total_operators
from pyqed._letta_one_site_opt import ReducedLatticeLETTA,reduced_local_problem
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import _energy
from pyqed._letta_one_site_opt.reduced_gauge import canonicalize_reduced_frontier
from pyqed._letta_two_site_opt import reduced_pair_problem


def setup(two_s,tied):
 n=4;h=np.zeros((n,n));g=np.zeros((n,)*4)
 for i in range(n):g[i,i,i,i]=4.
 for i,j,t in [(0,1,-1.),(0,2,-1.),(1,3,-1.),(2,3,-1.),(0,3,.25),(1,2,.25)]:h[i,j]=h[j,i]=t
 ne=3 if two_s==1 else 4
 p=ElectronicProblem(h,g,((ne+two_s)//2,(ne-two_s)//2))
 neighborhoods=((0,1,2,3),(1,2,3),(2,3),(3,)) if tied else tuple((i,) for i in range(n))
 s=ReducedLatticeLETTA.random((2,2),symmetry=p.symmetry('su2'),multiplets_per_sector=1,seed=59,real=False,neighborhoods=neighborhoods)
 return p,s

@pytest.mark.parametrize('two_s',[0,1,2])
@pytest.mark.parametrize('tied',[False,True])
def test_spin_components_and_gauge(two_s,tied):
 p,s=setup(two_s,tied);H=determinant_hamiltonian(p);N,Z,S2=total_operators(4);ham=p.su2_mpo();energies=[];vectors=[]
 for m in range(-two_s,two_s+1,2):
  v=s.state_vector(target_two_m=m);vectors.append(v)
  np.testing.assert_allclose(N@v,(3 if two_s==1 else 4)*v,atol=1e-11)
  np.testing.assert_allclose(Z@v,m/2*v,atol=1e-11)
  np.testing.assert_allclose(S2@v,two_s*(two_s+2)/4*v,atol=1e-11)
  np.testing.assert_allclose(np.vdot(v,v),s.norm(),atol=1e-11)
  energies.append((np.vdot(v,H@v)/np.vdot(v,v)).real)
 np.testing.assert_allclose(energies,_energy(s,ham,stable=True),atol=2e-11)
 for center in [0,3,1]:
  canonicalize_reduced_frontier(s,center)
  for m,v in zip(range(-two_s,two_s+1,2),vectors):np.testing.assert_allclose(s.state_vector(target_two_m=m),v,atol=3e-10)

@pytest.mark.parametrize('two_s',[0,1,2])
@pytest.mark.parametrize('tied',[False,True])
def test_local_and_pair_actions(two_s,tied):
 n=3;ne=3 if two_s==1 else 2
 h=np.array([[0.,-1.,.25],[-1.,0.,-1.],[.25,-1.,0.]])
 g=np.zeros((n,)*4)
 for i in range(n):g[i,i,i,i]=4.
 p=ElectronicProblem(h,g,((ne+two_s)//2,(ne-two_s)//2))
 ties=((0,1,2),(1,2),(2,)) if tied else tuple((i,) for i in range(n))
 s=ReducedLatticeLETTA.random((1,n),symmetry=p.symmetry('su2'),neighborhoods=ties,multiplets_per_sector=2,real=False,seed=59)
 ham=p.su2_mpo();components=replace(ham,contraction_backend='components')
 rng=np.random.default_rng(79)
 for builder,count in [(reduced_local_problem,3),(reduced_pair_problem,2)]:
  for site in range(count):
   a=builder(s,ham,site,matrix_free=True,dense_solver_threshold=0)
   b=builder(s,components,site,matrix_free=True,dense_solver_threshold=0)
   for _ in range(2):
    v=rng.normal(size=a.local_dimension)+1j*rng.normal(size=a.local_dimension)
    np.testing.assert_allclose(a.apply_metric(v),b.apply_metric(v),atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(a.apply_hamiltonian(v),b.apply_hamiltonian(v),atol=2e-10,rtol=2e-10)

@pytest.mark.parametrize('two_s',[0,1,2])
@pytest.mark.parametrize('method',['one','two','cbe'])
def test_optimized_state_energy_and_spin(two_s,method,record_property):
 from pyqed._letta_one_site_opt import letta_dmrg,LETTADMROptions
 from pyqed._letta_two_site_opt import letta_two_site_dmrg,LETTATwoSiteOptions
 # Three orbitals allow an exact independent reference without a large run.
 n=3;h=np.array([[0.,-1.,.25],[-1.,0.,-1.],[.25,-1.,0.]])
 g=np.zeros((n,)*4)
 for i in range(n):g[i,i,i,i]=4.
 ne=3 if two_s==1 else 2
 p=ElectronicProblem(h,g,((ne+two_s)//2,(ne-two_s)//2));ham=p.su2_mpo()
 s=ReducedLatticeLETTA.random((1,n),symmetry=p.symmetry('su2'),multiplets_per_sector=2,seed=72,neighborhoods=((0,1,2),(1,2),(2,)))
 H=determinant_hamiltonian(p);N,Z,S2=total_operators(n)
 allowed=np.flatnonzero((np.diag(N)==ne)&(np.diag(Z)==two_s/2))
 vals,vec=np.linalg.eigh(S2[np.ix_(allowed,allowed)]);frame=np.eye(4**n)[:,allowed]@vec[:,abs(vals-two_s*(two_s+2)/4)<1e-10]
 exact=np.linalg.eigvalsh(frame.T@H@frame)[0]
 common=dict(max_sweeps=4,tolerance=1e-11,eigensolver_tolerance=1e-10,eigensolver_max_iterations=300,matrix_free=True,dense_solver_threshold=1,gauge_mode='frontier')
 if method=='two':
  r=letta_two_site_dmrg(ham,state=s,bond_dim=10,options=LETTATwoSiteOptions(**common,split_method='metric-als-energy',reduced_sector_growth=True,truncation_max_iterations=32,energy_refinement_max_iterations=32))
 else:
  r=letta_dmrg(ham,state=s,options=LETTADMROptions(**common,cbe_enabled=method=='cbe',cbe_expansion_dimension=3,cbe_projection_max_iterations=1000,cbe_refinement_max_iterations=32,cbe_energy_refinement_max_iterations=32))
 for m in range(-two_s,two_s+1,2):
  v=r.state.state_vector(target_two_m=m)
  np.testing.assert_allclose(S2@v,two_s*(two_s+2)/4*v,atol=2e-10)
  np.testing.assert_allclose(np.vdot(v,H@v)/np.vdot(v,v),r.energy,atol=2e-10)
 record_property('two_s',two_s);record_property('method',method)
 record_property('energy',r.energy);record_property('exact',exact)
 record_property('error',abs(r.energy-exact))
 record_property('spin_residual',float(np.linalg.norm(S2@v-two_s*(two_s+2)/4*v)))
 assert r.energy==pytest.approx(exact,abs=2e-8)

@pytest.mark.parametrize('two_s',[0,1,2])
@pytest.mark.parametrize('site',[0,1])
def test_pair_compression_metric_and_adjoint(two_s,site):
 from collections import Counter
 from pyqed._letta_compression import MetricCompressionOptions
 from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot,compress_reduced_pair
 from pyqed._letta_two_site_opt.reduced_solver import _pair_vector_from_sources
 n=3;ne=3 if two_s==1 else 2
 p=ElectronicProblem(np.eye(n),np.zeros((n,)*4),((ne+two_s)//2,(ne-two_s)//2))
 s=ReducedLatticeLETTA.random((1,n),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=87,neighborhoods=((0,2),(1,0),(2,1)))
 pair=reduced_pair_problem(s,p.su2_mpo(),site,matrix_free=True,dense_solver_threshold=0)
 root=ReducedPairMetricRoot(pair,s,1e-12);rng=np.random.default_rng(67)
 x=rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
 np.testing.assert_allclose(root.physical_adjoint(root.physical_apply(x)),pair.apply_metric(x),atol=2e-10,rtol=2e-10)
 le,re=(pair.frontier.site_embedding(s,k) for k in (site,site+1))
 a,b=le.pack_source(s.tensors[site]),re.pack_source(s.tensors[site+1]);target=pair.old_vector+.03*x
 fit=compress_reduced_pair(target,pair,s,a,b,Counter(s.bond_sectors[site]),options=MetricCompressionOptions(als_max_iterations=32))
 error=_pair_vector_from_sources(pair.layout,le,re,fit.left,fit.right)-target
 assert fit.loss==pytest.approx(np.vdot(error,pair.apply_metric(error)).real,abs=2e-10)
 assert fit.loss<=fit.diagnostics['initial_loss']+2e-10

@pytest.mark.parametrize('two_s',[0,1,2])
@pytest.mark.parametrize('method',['two','cbe'])
def test_truncated_ladder_retains_spin_and_reported_energy(two_s,method):
 from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import initial_mps
 from pyqed._letta_one_site_opt import letta_dmrg,LETTADMROptions
 from pyqed._letta_two_site_opt import letta_two_site_dmrg,LETTATwoSiteOptions
 p,template=setup(two_s,True);ne=3 if two_s==1 else 4
 sites,_=initial_mps(4,ne,two_s,3,'multiplets',4)
 s=ReducedLatticeLETTA.from_mps(sites,symmetry=p.symmetry('su2'),neighborhoods=template.neighborhoods,normalize=True)
 ham=p.su2_mpo();before=_energy(s,ham,stable=True)
 common=dict(max_sweeps=2,tolerance=1e-11,eigensolver_tolerance=1e-10,eigensolver_max_iterations=300,matrix_free=True,dense_solver_threshold=1,gauge_mode='frontier')
 if method=='two':r=letta_two_site_dmrg(ham,state=s,bond_dim=3,options=LETTATwoSiteOptions(**common,split_method='metric-als-energy',reduced_sector_growth=True,truncation_max_iterations=32,energy_refinement_max_iterations=32))
 else:r=letta_dmrg(ham,state=s,options=LETTADMROptions(**common,cbe_enabled=True,cbe_expansion_dimension=3,cbe_projection_max_iterations=1000,cbe_refinement_max_iterations=32,cbe_energy_refinement_max_iterations=32))
 H=determinant_hamiltonian(p);N,Z,S2=total_operators(4)
 for m in range(-two_s,two_s+1,2):
  v=r.state.state_vector(target_two_m=m)
  np.testing.assert_allclose(S2@v,two_s*(two_s+2)/4*v,atol=2e-10)
  np.testing.assert_allclose(np.vdot(v,H@v)/np.vdot(v,v),r.energy,atol=2e-10)
 assert r.energy<=before+1e-9
 assert max(r.state.bond_dimensions)<=3

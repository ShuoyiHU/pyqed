import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_mps_cbe import select_mps_cbe
from pyqed._letta_one_site_opt.reduced_cbe import expand_reduced_cbe

def initial():
 p=ElectronicProblem(np.array([[0.,-1.],[-1.,0.]]),np.zeros((2,)*4),(1,1));sym=p.symmetry('su2')
 vac,single,double=sym.physical_basis.sectors
 a=np.zeros((1,1,3));a[0,0,0]=1
 b=np.zeros((3,1,1));b[0,0,0]=1
 s=ReducedLatticeLETTA((1,2),sym,[{(vac,double,double):a},{(double,vac,double):b}],bond_sectors=((double,)*3,),neighborhoods=((0,),(1,)))
 return s,p.su2_mpo()

def controls():return LETTADMROptions(max_sweeps=8,cbe_enabled=True,cbe_selector='mps-dd',cbe_expansion_dimension=1,gauge_mode='frontier',metric_tolerance=1e-12,eigensolver_tolerance=1e-10)

@pytest.mark.parametrize('direction',['lr','rl'])
def test_padding_and_dd(direction):
 s,h=initial();v=s.state_vector()
 c,selection,d=select_mps_cbe(s,h,0,direction,controls())
 assert selection is not None
 assert d['tangent_overlap']<1e-12
 assert any(q not in s.bond_sectors[0] for q in selection.multiplicities)
 e=expand_reduced_cbe(c,selection,direction)
 np.testing.assert_allclose(e.state_vector(),v,atol=1e-12)
 assert e.symmetry_violation()==0


def test_missing_sectors_recovered_without_pair_solve(monkeypatch):
 import pyqed._letta_two_site_opt.reduced_solver as pair
 s,h=initial()
 baseline=letta_dmrg(h,state=s,options=LETTADMROptions(max_sweeps=2))
 assert abs(baseline.energy)<1e-12
 def forbidden(*a,**k):raise AssertionError('forbidden two-site eigensolve or determinant expansion')
 monkeypatch.setattr(pair,'_solve_local_problem',forbidden)
 monkeypatch.setattr(ReducedLatticeLETTA,'state_vector',forbidden)
 r=letta_dmrg(h,state=s,options=controls())
 assert r.energy==pytest.approx(-2.,abs=1e-8)
 assert max(r.state.bond_dimensions)<=3
 assert any(u.cbe_expansion_dimension for row in r.history for u in row.updates)
 assert all(u.cbe_energy_refinement_iterations==0 for row in r.history for u in row.updates)

@pytest.mark.parametrize('direction',['lr','rl'])
def test_multisite_canonical_metric_and_padding(direction):
 from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import integrals,initial_mps
 from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
 from pyqed._letta_one_site_opt.reduced_symmetry import _sector_irrep
 h,g,_=integrals(2,3,dict(U=12,t=1,t_diagonal=-.25));p=ElectronicProblem(h,g,(3,2))
 sites,_=initial_mps(6,5,1,8,'multiplets',4)
 # Complex-valued test, including a nontrivial external gauge.
 for site in sites:
  for key in site.data:site.data[key]=site.data[key]*(1+.2j)
 s=ReducedLatticeLETTA.from_mps(sites,symmetry=p.symmetry('su2'),neighborhoods=tuple((i,) for i in range(6)))
 c,selection,d=select_mps_cbe(s,p.su2_mpo(),2,direction,controls())
 assert selection is not None
 problem=selection.problem
 x=np.random.default_rng(2).normal(size=problem.local_dimension)
 expected=problem.layout.pack({k:v*_sector_irrep(k[-1]).dim for k,v in problem.layout.unpack(x).items()})
 np.testing.assert_allclose(problem.apply_metric(x),expected,atol=1e-10)
 np.testing.assert_allclose(c.state_vector(),s.state_vector(),atol=1e-10)
 np.testing.assert_allclose(expand_reduced_cbe(c,selection,direction).state_vector(),s.state_vector(),atol=1e-10)
 assert d['tangent_overlap']<1e-10


def test_tied_state_rejected():
 from dataclasses import replace
 s,h=initial()
 s=ReducedLatticeLETTA(s.lattice_shape,s.symmetry,[{k:np.repeat(v[:,:,None,:],s.physical_dim,axis=2) for k,v in s.tensors[0].items()},s.tensors[1]],bond_sectors=s.bond_sectors,neighborhoods=((0,1),(1,)))
 with pytest.raises(ValueError,match='untied'):
  letta_dmrg(h,state=s,options=controls())

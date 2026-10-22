import numpy as np
import pytest
from pyqed._letta_one_site_opt.qr_absorbed_action import QRAbsorbedAction as FusedAction
from pyqed._letta_one_site_opt import ReducedLatticeLETTA,LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.dependency_environment import DependencySweepContext

@pytest.mark.parametrize('nelec',[(1,1),(2,1),(2,0)])
@pytest.mark.parametrize('site',[0,1,2])
def test_all_coordinates_and_lowest_energy(nelec,site,monkeypatch):
 monkeypatch.setenv('PYQED_SHARED_PRODUCTS','0')
 h=np.diag([.1,.2,.3]);h[0,1]=h[1,0]=-.7;h[0,2]=h[2,0]=.2
 g=np.zeros((3,)*4)
 for i in range(3):g[i,i,i,i]=4.
 problem=ElectronicProblem(h,g,nelec)
 state=ReducedLatticeLETTA.random((1,3),symmetry=problem.symmetry('su2'),multiplets_per_sector=2,real=False,seed=43,neighborhoods=((0,1,2),(1,2),(2,)))
 opts=LETTADMROptions(matrix_free=True,gauge_mode='none',simplify_fixed_ties=True)
 p=DependencySweepContext(state,problem.su2_mpo()).local_problem(site,opts)
 c=p.coordinates;c.prepare(opts.metric_tolerance);f=FusedAction(c)
 assert f.report["allocated_boundary_bytes"]==f.report["boundary_bytes"]
 assert c.rank<150
 eye=np.eye(c.rank)
 reference=np.column_stack([c.apply(x) for x in eye]);actual=np.column_stack([f(x) for x in eye])
 np.testing.assert_allclose(actual,reference,rtol=2e-12,atol=2e-12)
 np.testing.assert_allclose(actual,actual.conj().T,rtol=2e-12,atol=2e-12)
 np.testing.assert_allclose(np.linalg.eigvalsh(actual),np.linalg.eigvalsh(reference),rtol=2e-12,atol=2e-12)
 from pyqed._letta_one_site_opt.qr_absorbed_action import UnprofitableQRAbsorption
 if f.report['fused_MAC']>=f.report['original_MAC']:
  with pytest.raises(UnprofitableQRAbsorption):FusedAction(c,require_reduction=True)
 from pyqed._letta_one_site_opt.local_preconditioner import coordinate_blocks
 precondition,report=coordinate_blocks(f)
 for m,block in zip(c.maps,report['blocks']):
  np.testing.assert_allclose(block,reference[m[1],m[1]],rtol=2e-12,atol=2e-12)
 with pytest.raises(MemoryError):coordinate_blocks(f,max_elements=0)
 with pytest.raises(MemoryError):FusedAction(c,budget=-1)
 monkeypatch.setenv('PYQED_QR_ABSORBED_MB','64')
 np.testing.assert_allclose(c.apply(eye[0]),reference[:,0],rtol=2e-12,atol=2e-12)
 # The production arithmetic gate may retain the original action.
 if site==1:
  from pyqed._letta_one_site_opt.reduced_solver import _solve_local_problem
  from dataclasses import replace
  monkeypatch.setenv('PYQED_LOCAL_PRECONDITIONER','blocks')
  initial=p.embedding.pack_source(state.tensors[site])
  energy,vector,rank,residual=_solve_local_problem(p,replace(opts,dense_solver_threshold=1),initial_vector=initial)
  np.testing.assert_allclose(energy,np.linalg.eigvalsh(reference)[0],rtol=1e-9,atol=1e-9)
 import weakref
 reference_owner=weakref.ref(c)
 retained_action=f
 del c,p
 assert reference_owner() is None
 assert retained_action.rank==len(eye)

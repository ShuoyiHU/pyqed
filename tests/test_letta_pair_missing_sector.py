import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot

@pytest.mark.parametrize('side',['left','right','both'])
@pytest.mark.parametrize('nelec',[(2,2),(2,1)])
def test_pair_root_missing_boundary_sector_is_exact_null(side,nelec):
 h=np.eye(4);g=np.zeros((4,)*4)
 p=ElectronicProblem(h,g,nelec)
 s=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=29)
 if side in ('left','both'):
  q=list(s.tensors[0])[0][2]
  s.tensors[0]={k:v for k,v in s.tensors[0].items() if k[2]!=q}
 if side in ('right','both'):
  q=list(s.tensors[-1])[0][0]
  s.tensors[-1]={k:v for k,v in s.tensors[-1].items() if k[0]!=q}
 pair=reduced_pair_problem(s,p.su2_mpo(),1,matrix_free=True,dense_solver_threshold=0)
 root=ReducedPairMetricRoot(pair,s,1e-12)
 projector=pair.equilibrated_projector_factory(1e-12)
 projected=projector(pair.old_vector)
 np.testing.assert_allclose(projector(projected),projected,atol=1e-11)
 rng=np.random.default_rng(81);x=rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
 nx=pair.apply_metric(x)
 np.testing.assert_allclose(root.adjoint(root.apply(x)),nx,atol=1e-10,rtol=1e-11)
 np.testing.assert_allclose(pair.apply_metric(root.inverse_action(nx)),nx,atol=1e-10,rtol=1e-11)
 y=rng.normal(size=root.size)+1j*rng.normal(size=root.size)
 np.testing.assert_allclose(root.apply(root.unwhiten(y)),y,atol=1e-10,rtol=1e-11)
 assert any(0 in shape for shape in root.output_layout.shapes.values())
 physical=s.state_vector();expected=(s.target_two_j+1)*np.vdot(physical,physical).real
 assert np.linalg.norm(root.apply(pair.old_vector))**2==pytest.approx(expected,rel=2e-11,abs=1e-12)

@pytest.mark.parametrize('site',[1,2])
def test_local_projector_missing_sector_preserves_metric(site):
 from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
 from pyqed._letta_one_site_opt.reduced_norm import ReducedNormChain
 p=ElectronicProblem(np.eye(4),np.zeros((4,)*4),(2,2))
 s=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),multiplets_per_sector=2,real=False,seed=29,neighborhoods=((0,),(1,),(2,),(3,)))
 if site==1:
  q=list(s.tensors[0])[0][2];s.tensors[0]={k:v for k,v in s.tensors[0].items() if k[2]!=q}
 else:
  q=list(s.tensors[-1])[0][0];s.tensors[-1]={k:v for k,v in s.tensors[-1].items() if k[0]!=q}
 frontier=ReducedFrontier.from_state(s);chain=ReducedNormChain.build(frontier.to_mps(s));embedding=frontier.site_embedding(s,site)
 project=chain.local_projector(site,embedding,1e-12)
 assert project is not None
 rng=np.random.default_rng(8);x=rng.normal(size=embedding.source_size)+1j*rng.normal(size=embedding.source_size)
 def metric(v):return embedding.adjoint(embedding.pack_target(chain.local_action(site,embedding.unpack_target(embedding.apply(v)))))
 np.testing.assert_allclose(metric(project(x)),metric(x),atol=1e-10,rtol=1e-11)
 np.testing.assert_allclose(project(project(x)),project(x),atol=1e-11)

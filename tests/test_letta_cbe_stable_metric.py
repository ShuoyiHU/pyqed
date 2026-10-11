"""Physical square-root objectives must not be evaluated by squared Grams."""
import numpy as np
from scipy.sparse.linalg import aslinearoperator
from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_two_site_opt.reduced_compression import fit_metric_factors

class Root:
    def __init__(self,r):self.r=r;self.size=r.shape[0]
    def physical_apply(self,x):return self.r@x
    def apply(self,x):return self.r@x
    def adjoint(self,x):return self.r.conj().T@x

def test_als_initial_loss_resolves_a_small_physical_direction():
    # Forming R†R rounds away the 1e-16 diagonal contribution; its norm says
    # zero for this vector, although the represented physical error has norm 1.
    r=np.array([[1.,1.],[0.,1e-8]])
    root=Root(r);metric=aslinearoperator(r.T@r)
    a=np.ones(1);b=np.array([1e8,-1e8]);target=np.zeros(2)
    merge=lambda a,b:a[0]*b
    def adjoint(side,a,b,v):
        return np.array([np.vdot(b,v)]) if side==0 else a[0].conjugate()*v
    result=fit_metric_factors(target,metric,a,b,merge=merge,root=root,
        adjoint=adjoint,left_indices=np.arange(1),right_indices=np.arange(2),
        gauge_blocks=[],options=MetricCompressionOptions(solver='als',als_max_iterations=2),
        als_max_iterations=2)
    assert abs(result.diagnostics['initial_loss']-1.)<1e-12
    assert abs(result.loss-np.linalg.norm(root.apply(merge(result.left,result.right)-target))**2)<1e-12


def test_state_norm_preserves_cancelling_multiplicity_paths():
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    symmetry=ElectronicProblem(np.zeros((2,2)),np.zeros((2,)*4),(1,1)).symmetry('su2')
    vacuum,_,double=symmetry.physical_basis.sectors
    rng=np.random.default_rng(0)
    a=rng.normal(size=3)*1e8;b=rng.normal(size=3)
    b-=a*np.dot(a,b)/np.dot(a,a)
    b+=a/np.dot(a,a)
    state=ReducedLatticeLETTA((1,2),symmetry,
        [{(vacuum,double,double):a.reshape(1,1,3)},
         {(double,vacuum,double):b.reshape(3,1,1)}],
        bond_sectors=((double,double,double),),neighborhoods=((0,),(1,)),normalize=False)
    vector=state.state_vector()
    expected=float(np.vdot(vector,vector).real)
    assert abs(state.norm()-expected)<1e-6
    state.normalize()
    assert abs(state.norm()-1.)<1e-6


def test_tangent_projection_is_invariant_to_bond_gauge():
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA,LETTADMROptions
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_cbe import select_reduced_cbe
    p=ElectronicProblem(-np.eye(4,k=1)-np.eye(4,k=-1),np.zeros((4,)*4),(2,1))
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),
        multiplets_per_sector=2,seed=31,real=False)
    options=LETTADMROptions(cbe_enabled=True,cbe_projection_max_iterations=100,cbe_refinement_max_iterations=2)
    original=select_reduced_cbe(state,p.su2_mpo(),1,options)
    scaled=state.copy()
    for key,b in scaled.tensors[1].items():
        scaled.tensors[1][key]=b*np.geomspace(1e-3,1e3,b.shape[-1])
    for key,b in scaled.tensors[2].items():
        scaled.tensors[2][key]=b/np.geomspace(1e-3,1e3,b.shape[0]).reshape((-1,)+(1,)*(b.ndim-1))
    np.testing.assert_allclose(state.state_vector(),scaled.state_vector(),atol=1e-12)
    projected=select_reduced_cbe(scaled,p.su2_mpo(),1,options)
    assert abs(projected.missing_norm-original.missing_norm)<1e-9
    assert projected.missing_norm<1e-9
    assert not projected.multiplicities


def test_compression_solves_the_full_physical_objective_not_inverse_support():
    # An inverse-support cutoff may discard a small singular direction, but
    # large cancelling coefficients can still carry a finite physical residual.
    class TruncatedRoot(Root):
        def __init__(self):
            super().__init__(np.array([[1.,1.],[0.,1e-4]]))
            _,singular,vh=np.linalg.svd(self.r,full_matrices=False)
            self.cut=singular[:1,None]*vh[:1];self.size=1
        def apply(self,x):return self.cut@x
        def adjoint(self,x):return self.cut.T@x
        def physical_adjoint(self,x):return self.r.T@x
    root=TruncatedRoot();target=np.array([1e4,-1e4])
    merge=lambda a,b:a[0]*b
    def adjoint(side,a,b,v):
        return np.array([np.vdot(b,v)]) if side==0 else a[0].conjugate()*v
    for solver,dtype in (("als",float),("als",complex),("variable-projection",float),
                         ("grassmann-newton",complex),("joint-ls",complex)):
        result=fit_metric_factors(target,aslinearoperator(root.r.T@root.r),
            np.ones(1,dtype=dtype),np.array([1.,.1],dtype=dtype),merge=merge,root=root,
            adjoint=adjoint,left_indices=np.arange(1),right_indices=np.arange(2),
            gauge_blocks=[(np.array([[0]]),np.array([[0,1]]))],
            options=MetricCompressionOptions(solver=solver,tolerance=1e-12,max_iterations=100,
            max_factor_norm_growth=200.,
            als_max_iterations=4),als_max_iterations=4)
        assert result.loss<1e-12
        assert np.linalg.norm(root.physical_apply(merge(result.left,result.right)-target))<1e-6


def test_pair_hamiltonian_residual_preserves_ill_conditioned_boundary_gauges():
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_cbe import _canonical_pair_residual
    from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem
    from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
    p = ElectronicProblem(-np.eye(4, k=1)-np.eye(4, k=-1), np.zeros((4,)*4), (2, 1))
    h = p.su2_mpo()
    state = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        multiplets_per_sector=2, seed=31, real=False)

    def residual(candidate):
        pair = reduced_pair_problem(candidate, h, 1, matrix_free=True, dense_solver_threshold=0)
        root = ReducedPairMetricRoot(pair, candidate, 1e-12)
        return _canonical_pair_residual(pair, candidate, h, root), pair, root

    reference, pair, root = residual(state)
    x = pair.old_vector
    hx, nx = pair.apply_hamiltonian(x), pair.apply_metric(x)
    energy = np.vdot(x, hx).real/np.vdot(x, nx).real
    np.testing.assert_allclose(reference, root.unwhiten_adjoint(hx-energy*nx), atol=1e-11)
    np.testing.assert_allclose(root.project_physical(root.physical_apply(x)), root.apply(x), atol=1e-11)
    for scale in (1e2, 1e6):
        changed = state.copy()
        # The two exterior bond gauges cancel exactly in the represented state.
        # Contracting raw pair coefficients first nevertheless loses precision.
        for bond in (0, 2):
            transforms = {}
            for key, block in changed.tensors[bond].items():
                q, d = key[-1], block.shape[-1]
                if q not in transforms:
                    rng = np.random.default_rng(42+d)
                    u, _ = np.linalg.qr(rng.normal(size=(d, d)))
                    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
                    transforms[q] = (u*np.geomspace(1, scale, d))@v.T
                changed.tensors[bond][key] = block@transforms[q]
            for key, block in changed.tensors[bond+1].items():
                changed.tensors[bond+1][key] = np.linalg.solve(transforms[key[0]],
                    block.reshape(block.shape[0], -1)).reshape(block.shape)
        np.testing.assert_allclose(changed.state_vector(), state.state_vector(), atol=1e-9)
        actual, _, _ = residual(changed)
        np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(reference), atol=1e-8, rtol=0.)

"""Cyclic two-site solves against independent physical operators and recovery."""
from dataclasses import replace

import numpy as np
import pytest

from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_one_site_opt import ReducedRingLETTA, ReducedPhysicalBasis
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_ring_solver import ring_energy, optimize_ring_site
from pyqed._letta_one_site_opt.reduced_updates import one_site_options
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_two_site_opt.reduced_ring_solver import optimize_ring_pair, refine_ring_pair_energy
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from test_letta_ring_sweeps import direct_conditional_ring
from test_letta_qchem import determinant_hamiltonian


def dimer(*, tied=True, solver='als'):
    basis = ReducedPhysicalBasis.spatial_orbital()
    state = ReducedRingLETTA.random(2, basis, SpinChargeSector(2, SU2Irrep(0)),
        anchor_sector=SpinChargeSector(0, SU2Irrep(1)),
        neighborhoods=((0,1),(1,0)) if tied else None, seed=21)
    h1 = np.array([[0.,-1.],[-1.,0.]])
    eri = np.zeros((2,)*4)
    eri[0,0,0,0] = eri[1,1,1,1] = 4.
    model = ElectronicProblem(h1,eri,(1,1))
    options = LETTATwoSiteOptions(max_sweeps=2, reduced_sector_growth=True,
        matrix_free=False, gauge_mode='none', energy_refinement_max_iterations=12,
        compression=MetricCompressionOptions(solver=solver, als_max_iterations=60,
            lsmr_max_iterations=150, max_iterations=120, tolerance=1e-11))
    return state, model, options


@pytest.mark.parametrize('solver', ['als','variable-projection','joint-ls','grassmann-newton'])
def test_ring_pair_all_compressors_reach_exact_hubbard_dimer(solver):
    state, model, options = dimer(solver=solver)
    before = ring_energy(state, model.su2_mpo())
    update = optimize_ring_pair(state, model.su2_mpo(), 0, 'lr', 4, options)
    assert update.accepted and not update.fallback, update
    assert update.recovery_reason is None
    assert update.energy < before
    assert update.energy == pytest.approx((4-np.sqrt(32))/2, abs=2e-9)
    assert update.compression_diagnostics['used_solver'] == solver
    assert update.energy_refinement_iterations > 0
    vector = direct_conditional_ring(state)
    np.testing.assert_allclose(np.vdot(vector,vector),1.,atol=3e-11)
    actual = np.vdot(vector,determinant_hamiltonian(model)@vector).real
    assert actual == pytest.approx(update.energy,abs=3e-11)
    assert state.bond_dimensions[1] <= 4


@pytest.mark.parametrize('edge',[0,1,2])
@pytest.mark.parametrize('direction',['lr','rl'])
def test_ring_pair_failure_restores_incumbent_before_one_site_fallback(monkeypatch,edge,direction):
    import pyqed._letta_two_site_opt.reduced_ring_solver as backend
    state, model, options = dimer()
    h = model.su2_mpo()
    baseline = state.copy()
    ordinary = optimize_ring_site(baseline,h,edge if direction=='lr' else (edge+1)%3,one_site_options(options))
    def failed(candidate,*args,**kwargs):
        next(iter(candidate.site_blocks(edge).values())).flat[0] += 1000
        raise MemoryError('injected candidate resource failure')
    monkeypatch.setattr(backend,'fit_ring_pair_target',failed)
    update = optimize_ring_pair(state,h,edge,direction,4,options)
    assert update.accepted == ordinary.accepted
    assert update.baseline_selected and update.fallback
    assert 'MemoryError' in update.recovery_reason
    assert update.energy == pytest.approx(ordinary.energy,abs=2e-11)
    np.testing.assert_allclose(direct_conditional_ring(state),direct_conditional_ring(baseline),atol=2e-11)
    assert state.bond_sectors == baseline.bond_sectors


def test_public_ring_two_site_sweeps_all_edges_and_preserves_input():
    state, model, options = dimer(tied=False)
    old = direct_conditional_ring(state)
    options = replace(options,gauge_mode='frontier',max_sweeps=2)
    result = letta_two_site_dmrg(model.su2_mpo(),state=state,bond_dim=4,options=options)
    np.testing.assert_array_equal(direct_conditional_ring(state),old)
    assert result.energy == pytest.approx((4-np.sqrt(32))/2,abs=2e-9)
    previous = ring_energy(state,model.su2_mpo())
    for sweep in result.history:
        assert {(u.left_site,u.right_site) for u in sweep.updates} == {(0,1),(1,2),(2,0)}
        for u in sweep.updates:
            assert u.recovery_reason is None, u.recovery_reason
            assert u.energy <= previous+2e-10
            previous = u.energy
    assert max(result.state.bond_dimensions) <= 4


def test_ring_target_fit_does_not_invoke_pair_eigensolver(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_ring_solver as backend
    from pyqed._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem
    state,model,options = dimer()
    p=CyclicPairProblem(state,model.su2_mpo(),2)
    def forbidden(*args,**kwargs):
        raise AssertionError('supplied-target fitting invoked pair eigensolve')
    monkeypatch.setattr(backend,'_solve_local_problem',forbidden)
    candidate,fit,split,refinement,diagnostic=backend.fit_ring_pair_target(
        state,model.su2_mpo(),p,p.old_vector,4,options)
    assert fit.loss < 1e-10
    assert ring_energy(candidate,model.su2_mpo()) <= ring_energy(state,model.su2_mpo())+2e-10


def test_three_site_su2_doublet_two_site_native_without_expansion(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_contraction as component
    import pyqed._letta_two_site_opt.reduced_solver as reference
    state=ReducedRingLETTA.random(3,ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(1,SU2Irrep(1)),multiplets_per_sector=1,seed=32)
    model=ElectronicProblem(np.eye(3)-np.ones((3,3)),np.zeros((3,)*4),(1,0))
    def forbidden(*args,**kwargs):
        raise AssertionError('native ring solve expanded magnetic variational tensors')
    monkeypatch.setattr(component,'expand_reduced_mps_site',forbidden)
    monkeypatch.setattr(reference,'_expand_pair_blocks',forbidden)
    result=letta_two_site_dmrg(model.su2_mpo(),state=state,bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=2,reduced_sector_growth=True,
            matrix_free=True,dense_solver_threshold=1,energy_refinement_max_iterations=4,
            compression=MetricCompressionOptions(als_max_iterations=20,lsmr_max_iterations=100)))
    assert result.energy==pytest.approx(-2.,abs=2e-8)
    assert all(not u.recovery_reason for s in result.history for u in s.updates)
    v=direct_conditional_ring(result.state)
    dense=np.kron(determinant_hamiltonian(model),np.eye(2))
    assert np.vdot(v,dense@v).real/np.vdot(v,v).real==pytest.approx(-2.,abs=2e-8)


def test_three_site_bose_pbc_arbitrary_ties_two_site_energy():
    from pyqed.mps.symmetry import Sector
    from pyqed._letta_one_site_opt import LatticeMPO
    basis=ReducedPhysicalBasis(('0','1','2'),tuple(
        Sector(('number','su2'),(n,SU2Irrep(0))) for n in range(3)),(1,1,1))
    state=ReducedRingLETTA.random(3,basis,Sector(('number','su2'),(2,SU2Irrep(0))),
        neighborhoods=((0,2),(1,0),(2,1)),seed=72)
    a=np.diag(np.sqrt([1.,2.]),1); identity=np.eye(3); number=np.diag([0.,1.,2.])
    products=[]
    for i,j in [(0,1),(1,2),(2,0)]:
        for x,y in [(a.T,a),(a,a.T)]:
            term=[identity.copy() for _ in range(3)]
            term[i],term[j]=-x,y
            products.append(term)
    for i in range(3):
        term=[identity.copy() for _ in range(3)]
        term[i]=2*number@(number-identity)
        products.append(term)
    n=len(products)
    middle=np.zeros((n,n,3,3))
    for t,term in enumerate(products):middle[t,t]=term[1]
    mpo=LatticeMPO((np.stack([p[0] for p in products])[None],middle,
                    np.stack([p[2] for p in products])[:,None]))
    dense=sum(np.kron(np.kron(p[0],p[1]),p[2]) for p in products)
    ids=[i for i,c in enumerate(np.ndindex(3,3,3)) if sum(c)==2]
    exact=np.linalg.eigvalsh(dense[np.ix_(ids,ids)])[0]
    result=letta_two_site_dmrg(mpo,state=state,bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=2,matrix_free=False,
            energy_refinement_max_iterations=6,
            compression=MetricCompressionOptions(als_max_iterations=40,lsmr_max_iterations=120)))
    assert result.energy==pytest.approx(exact,abs=2e-8)
    assert all(not u.recovery_reason for s in result.history for u in s.updates)
    v=direct_conditional_ring(result.state)
    assert np.vdot(v,dense@v).real/np.vdot(v,v).real==pytest.approx(exact,abs=2e-8)


def test_ring_two_site_gauge_partial_failure_restores_and_continues(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_ring_solver as backend
    state,model,options=dimer(tied=False)
    original=backend.ring_gauge_shift
    visited=[]
    def fail_once(candidate,site,direction,**kwargs):
        visited.append(site)
        if len(visited)==1:
            next(iter(candidate.site_blocks(site).values())).flat[0]=np.nan
            raise FloatingPointError('injected partial gauge write')
        return original(candidate,site,direction,**kwargs)
    monkeypatch.setattr(backend,'ring_gauge_shift',fail_once)
    result=letta_two_site_dmrg(model.su2_mpo(),state=state,bond_dim=4,
        options=replace(options,gauge_mode='frontier',max_sweeps=1))
    assert len(visited)==3
    assert result.history[0].updates[0].recovery_reason.startswith('gauge:')
    assert not result.converged
    assert result.energy==pytest.approx((4-np.sqrt(32))/2,abs=2e-9)
    assert np.all(np.isfinite(direct_conditional_ring(result.state)))


def test_both_ring_candidates_failing_leave_state_and_allocation_unchanged(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_ring_solver as backend
    state,model,options=dimer()
    before=direct_conditional_ring(state)
    bonds=state.bond_sectors
    def fail(*args,**kwargs):
        raise np.linalg.LinAlgError('injected solve failure')
    monkeypatch.setattr(backend,'_solve_local_problem',fail)
    monkeypatch.setattr(backend,'optimize_ring_site',fail)
    update=optimize_ring_pair(state,model.su2_mpo(),2,'rl',4,options)
    assert not update.accepted and update.fallback
    assert 'baseline:' in update.recovery_reason
    assert state.bond_sectors==bonds
    np.testing.assert_array_equal(direct_conditional_ring(state),before)


def test_ring_two_site_optional_polishing_preserves_energy_guard():
    state,model,options=dimer(tied=False)
    result=letta_two_site_dmrg(model.su2_mpo(),state=state,bond_dim=4,
        options=replace(options,max_sweeps=1,one_site_polish_sweeps=1))
    assert result.polish_sweeps==1
    assert result.energy <= result.two_site_energy+options.energy_increase_tolerance
    assert result.energy==pytest.approx((4-np.sqrt(32))/2,abs=2e-9)
    assert result.state.norm()==pytest.approx(1.,abs=2e-11)

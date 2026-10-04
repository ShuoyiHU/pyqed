"""All four shared compressors in a correlated cyclic physical metric."""
from collections import Counter

import numpy as np
import pytest

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed._letta_one_site_opt import ReducedRingLETTA, ReducedPhysicalBasis
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem, CyclicPairMetricRoot
from pyqed._letta_two_site_opt.reduced_ring_compression import compress_ring_pair, ring_factor_adjoint, ring_factor_blocks
from pyqed._letta_two_site_opt.reduced_solver import _active_source_indices
from test_letta_qchem import integrals
from test_letta_ring_pair import reference_pair_frame


def problem_and_factors(edge=0, tied=True, copies=1):
    state = ReducedRingLETTA.random(2, ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(2, SU2Irrep(0)), multiplets_per_sector=copies,
        anchor_sector=SpinChargeSector(0, SU2Irrep(1)),
        neighborhoods=((0,1),(1,0)) if tied else None, seed=17)
    problem = CyclicPairProblem(state, ElectronicProblem(*integrals(2), (1,1)).su2_mpo(), edge)
    a = problem.left_embedding.pack_source(state.site_blocks(edge))
    b = problem.right_embedding.pack_source(state.site_blocks(problem.right_site))
    retained = dict(Counter(state.bond_sectors[(edge+1) % (state.nsites+1)]))
    return state, problem, a, b, retained


@pytest.mark.parametrize('edge', [0,1,2])
def test_ring_factor_adjoint_and_legal_gauges(edge):
    state, p, a, b, retained = problem_and_factors(edge, copies=2)
    rng = np.random.default_rng(12)
    g = rng.normal(size=p.local_dimension)+1j*rng.normal(size=p.local_dimension)
    for side, old in enumerate((a,b)):
        dx = rng.normal(size=old.size)+1j*rng.normal(size=old.size)
        image = p.merge(dx,b) if side == 0 else p.merge(a,dx)
        np.testing.assert_allclose(np.vdot(g,image), np.vdot(ring_factor_adjoint(p,side,a,b,g),dx), atol=3e-12)
    aa,bb = a.copy(),b.copy()
    for li,ri in ring_factor_blocks(p,state,retained):
        k = li.shape[1]
        gauge = 2*np.eye(k)+.1*(rng.normal(size=(k,k))+1j*rng.normal(size=(k,k)))
        aa[li] = aa[li]@gauge
        bb[ri] = np.linalg.solve(gauge,bb[ri])
    np.testing.assert_allclose(p.merge(aa,bb),p.merge(a,b),atol=3e-12)


@pytest.mark.parametrize('solver',['als','variable-projection','joint-ls','grassmann-newton'])
@pytest.mark.parametrize('edge',[0,2])
def test_all_ring_compressors_fit_the_physical_metric(solver,edge):
    state,p,a,b,retained = problem_and_factors(edge)
    rng = np.random.default_rng(81)
    target = p.merge(a+.15*(rng.normal(size=a.size)+1j*rng.normal(size=a.size)),
                     b+.15*(rng.normal(size=b.size)+1j*rng.normal(size=b.size)))
    frame = reference_pair_frame(p)
    initial = np.linalg.norm(frame@(p.merge(a,b)-target))**2
    result = compress_ring_pair(target,p,state,a,b,retained,options=MetricCompressionOptions(
        solver=solver,als_max_iterations=80,lsmr_max_iterations=120,max_iterations=150,tolerance=1e-10,max_workspace_mb=256))
    measured = np.linalg.norm(frame@(p.merge(result.left,result.right)-target))**2
    assert measured < max(1e-14, initial*1e-8)
    assert result.loss == pytest.approx(measured,abs=3e-12)
    assert result.diagnostics['requested_solver'] == solver
    assert result.diagnostics['used_solver'] == solver
    assert result.diagnostics['metric_kind'] == 'full-cyclic-reduced'
    assert result.diagnostics['metric_rank'] < result.diagnostics['pair_dimension']


def test_whole_multiplet_trimming_keeps_explicit_als_lsmr_budgets():
    state,p,a,b,retained = problem_and_factors(0,copies=2)
    retained = {q:1 for q in retained}
    li = _active_source_indices(p.left_embedding,retained,'left')
    ri = _active_source_indices(p.right_embedding,retained,'right')
    aa,bb = np.zeros_like(a),np.zeros_like(b)
    rng = np.random.default_rng(73)
    aa[li],bb[ri] = rng.normal(size=len(li)),rng.normal(size=len(ri))
    target = p.merge(aa,bb)
    result = compress_ring_pair(target,p,state,a,b,retained,
        options=MetricCompressionOptions(als_max_iterations=2,lsmr_max_iterations=1))
    assert result.iterations <= 2
    assert result.diagnostics['als_max_iterations'] == 2
    assert all(r['iterations'] <= 1 and r['max_iterations'] == 1 for r in result.diagnostics['linear_solves'])
    assert not np.any(result.left[np.setdiff1d(np.arange(a.size),li)])
    assert not np.any(result.right[np.setdiff1d(np.arange(b.size),ri)])
    assert result.loss <= result.diagnostics['initial_loss']+1e-12


def test_cyclic_metric_equilibration_retains_small_independent_coordinates():
    _,p,_,_,_ = problem_and_factors(0,tied=False)
    p.materialize(hamiltonian=False)
    rank = CyclicPairMetricRoot(p).size
    scale = np.geomspace(1e-8,1e8,p.local_dimension)
    p.metric = p.metric*scale[:,None]*scale[None,:]
    root = CyclicPairMetricRoot(p)
    assert root.size == rank
    np.testing.assert_allclose(root.coordinates.conj().T@root.coordinates,p.metric,atol=1e-12,rtol=2e-12)
    np.testing.assert_allclose(root.coordinates@root.whitening,np.eye(rank),atol=3e-12)



@pytest.mark.parametrize('solver',['als','variable-projection','joint-ls','grassmann-newton'])
def test_nonlinear_rank_reduction_with_nonseparable_ring_metric(solver):
    from pyqed.mps.symmetry import Sector
    from pyqed._letta_one_site_opt import LatticeMPO
    sector = Sector(('charge','su2'),(0,SU2Irrep(0)))
    basis = ReducedPhysicalBasis(('neutral',),(sector,),(2,))
    state = ReducedRingLETTA.random(4,basis,sector,multiplets_per_sector=2,seed=42)
    h = LatticeMPO(tuple(np.eye(2)[None,None] for _ in range(4)))
    p = CyclicPairProblem(state,h,0)
    a = p.left_embedding.pack_source(state.site_blocks(0))
    b = p.right_embedding.pack_source(state.site_blocks(1))
    retained = {sector:1}
    li = _active_source_indices(p.left_embedding,retained,'left')
    ri = _active_source_indices(p.right_embedding,retained,'right')
    reference_a,reference_b = np.zeros_like(a),np.zeros_like(b)
    rng = np.random.default_rng(63)
    reference_a[li] = a[li]+.03*(rng.normal(size=len(li))+1j*rng.normal(size=len(li)))
    reference_b[ri] = b[ri]+.03*(rng.normal(size=len(ri))+1j*rng.normal(size=len(ri)))
    target = p.merge(reference_a,reference_b)
    frame = reference_pair_frame(p)
    p.materialize(hamiltonian=False)
    reshuffled = p.metric.reshape(4,4,4,4).transpose(0,2,1,3).reshape(16,16)
    assert np.linalg.matrix_rank(reshuffled,tol=1e-10) > 1
    assert np.linalg.matrix_rank(p.metric,tol=1e-10) == p.local_dimension
    result = compress_ring_pair(target,p,state,a,b,retained,options=MetricCompressionOptions(
        solver=solver,als_max_iterations=300,lsmr_max_iterations=100,max_iterations=400,tolerance=1e-10))
    physical_error = np.linalg.norm(frame@(p.merge(result.left,result.right)-target))**2
    assert physical_error < 1e-10
    assert result.loss == pytest.approx(physical_error,abs=3e-12)
    assert result.diagnostics['used_solver'] == solver
    if solver != 'als':
        assert result.diagnostics['nonlinear_parameters'] > 0
    assert not np.any(result.left[np.setdiff1d(np.arange(a.size),li)])
    assert not np.any(result.right[np.setdiff1d(np.arange(b.size),ri)])

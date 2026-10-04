"""Native cyclic pair actions against independently expanded local pair frames."""
import numpy as np
import pytest

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed._letta_one_site_opt import ReducedRingLETTA, ReducedPhysicalBasis
from pyqed._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem, CyclicPairMetricRoot
from pyqed._letta_two_site_opt.reduced_solver import _expand_pair_blocks
from test_letta_qchem import integrals, determinant_hamiltonian
from test_letta_ring_sweeps import direct_conditional_ring


def reference_pair_vector(problem, vector):
    cores = [expand_reduced_mps_site(a) for a in problem.sites]
    i, j = problem.left_site, problem.right_site
    pair = _expand_pair_blocks(problem.layout.unpack(vector), problem.sites[i], problem.sites[j])
    n = len(cores)
    result = []
    for physical in np.ndindex(*(a.shape[1] for a in cores)):
        matrix = pair[:, physical[i], physical[j], :]
        for step in range(2, n):
            site = (i+step) % n
            matrix = matrix@cores[site][:, physical[site], :]
        result.append(np.trace(matrix))
    return np.asarray(result)


def reference_pair_frame(problem):
    return np.column_stack([reference_pair_vector(problem, x) for x in np.eye(problem.local_dimension)])


@pytest.mark.parametrize('edge', [0, 1, 2])
@pytest.mark.parametrize('charge,two_s', [(2, 0), (1, 1)])
def test_pair_actions_and_root_include_full_correlated_ring_metric(edge, charge, two_s, monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as magnetic_pair
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic_site
    state = ReducedRingLETTA.random(2, ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(charge, SU2Irrep(two_s)), anchor_sector=SpinChargeSector(0, SU2Irrep(1)),
        multiplets_per_sector=1, seed=12)
    model = ElectronicProblem(*integrals(2), ((charge+two_s)//2, (charge-two_s)//2))
    pair = CyclicPairProblem(state, model.su2_mpo(), edge)
    frame = reference_pair_frame(pair)
    dense = np.kron(determinant_hamiltonian(model), np.eye(two_s+1))
    np.testing.assert_allclose(frame@pair.old_vector, direct_conditional_ring(state), atol=3e-12)
    expected_n, expected_h = frame.conj().T@frame, frame.conj().T@dense@frame
    def forbidden(*args, **kwargs):
        raise AssertionError('native ring pair expanded variational magnetic tensors')
    monkeypatch.setattr(magnetic_pair, '_expand_pair_blocks', forbidden)
    monkeypatch.setattr(magnetic_site, 'expand_reduced_mps_site', forbidden)
    # Rebuild with production expansion disabled, not only subsequent matvecs.
    pair = CyclicPairProblem(state, model.su2_mpo(), edge)
    pair.materialize()
    np.testing.assert_allclose(pair.metric, expected_n, atol=3e-12)
    np.testing.assert_allclose(pair.hamiltonian, expected_h, atol=3e-12)
    root = CyclicPairMetricRoot(pair)
    identity = np.eye(pair.local_dimension)
    rebuilt = np.column_stack([root.adjoint(root.apply(x)) for x in identity])
    np.testing.assert_allclose(rebuilt, expected_n, atol=4e-12)
    supported = np.column_stack([root.apply(root.unwhiten(x)) for x in np.eye(root.size)])
    np.testing.assert_allclose(supported, np.eye(root.size), atol=4e-12)
    rng = np.random.default_rng(19)
    x = rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
    y = pair.metric@x
    np.testing.assert_allclose(pair.metric@root.inverse_action(y), y, atol=4e-12)
    if edge == 0 and charge == 2:
        # Distinct internal spin paths are correlated by the ring complement.
        assert np.linalg.norm(pair.metric-np.diag(np.diag(pair.metric))) > 1e-3
        assert root.size < pair.local_dimension


@pytest.mark.parametrize('edge', [0, 1, 2])
def test_pair_merging_with_bidirectional_ties_matches_direct_state(edge):
    state = ReducedRingLETTA.random(2, ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(2, SU2Irrep(0)), neighborhoods=((0, 1), (1, 0)), seed=29)
    h = ElectronicProblem(*integrals(2), (1, 1)).su2_mpo()
    pair = CyclicPairProblem(state, h, edge)
    np.testing.assert_allclose(reference_pair_vector(pair, pair.old_vector), direct_conditional_ring(state), atol=4e-12)
    frame = reference_pair_frame(pair)
    x = np.random.default_rng(71).normal(size=pair.local_dimension)
    np.testing.assert_allclose(pair.apply_metric(x), frame.conj().T@frame@x, atol=4e-12)


def test_local_pair_workspace_guard_precedes_materialization():
    state = ReducedRingLETTA.random(2, ReducedPhysicalBasis.spatial_orbital(), SpinChargeSector(2, SU2Irrep(0)), seed=1)
    pair = CyclicPairProblem(state, ElectronicProblem(*integrals(2), (1, 1)).su2_mpo(), 0)
    with pytest.raises(MemoryError, match='max_workspace_mb'):
        pair.materialize(max_workspace_mb=1e-12)
    assert pair.metric is None and pair.hamiltonian is None
    pair.materialize(hamiltonian=False)
    with pytest.raises(MemoryError, match='max_workspace_mb'):
        CyclicPairMetricRoot(pair, max_workspace_mb=1e-12)



def test_expanded_pair_is_not_restricted_to_incumbent_internal_spin_channels():
    from pyqed.mps.symmetry import Sector
    from pyqed._letta_one_site_opt import LETTADMROptions
    from pyqed._letta_one_site_opt.reduced_solver import _solve_local_problem
    state = ReducedRingLETTA.random(2,ReducedPhysicalBasis.spatial_orbital(),
        SpinChargeSector(2,SU2Irrep(0)),anchor_sector=SpinChargeSector(0,SU2Irrep(1)),seed=91)
    middle = Sector(('charge','su2'),(1,SU2Irrep(0)))
    cores = [{key:a for key,a in state.tensors[0].items() if key[2] == middle},
             {key:a for key,a in state.tensors[1].items() if key[0] == middle}]
    state = ReducedRingLETTA(state.physical_basis,state.symmetry.sector,cores,
        bond_sectors=(state.bond_sectors[0],(middle,),state.bond_sectors[2]),closure=state.closure)
    h1 = np.array([[0.,-1.],[-1.,0.]])
    eri = np.zeros((2,)*4)
    eri[0,0,0,0] = eri[1,1,1,1] = 4.
    model = ElectronicProblem(h1,eri,(1,1))
    pair = CyclicPairProblem(state,model.su2_mpo(),0)
    assert any(key[2].components[-1] == SU2Irrep(2) for key in pair.layout.keys)
    frame = reference_pair_frame(pair)
    pair.materialize()
    np.testing.assert_allclose(pair.metric,frame.conj().T@frame,atol=4e-12)
    energy,vector,_,_ = _solve_local_problem(pair,LETTADMROptions(matrix_free=False),initial_vector=pair.old_vector)
    assert energy == pytest.approx((4-np.sqrt(32))/2,abs=2e-11)
    v = frame@vector
    assert np.vdot(v,determinant_hamiltonian(model)@v).real/np.vdot(v,v).real == pytest.approx(energy,abs=2e-11)


@pytest.mark.parametrize('edge',[0,1,2])
def test_cyclic_pair_keeps_each_abelian_charge_component(edge):
    from pyqed.mps.symmetry import Sector
    labels = ('nalpha','nbeta','su2')
    occupations = [(0,0),(1,0),(0,1),(1,1)]
    sectors = tuple(Sector(labels,q+(SU2Irrep(0),)) for q in occupations)
    basis = ReducedPhysicalBasis(('empty','alpha','beta','double'),sectors,(1,)*4)
    state = ReducedRingLETTA.random(2,basis,Sector(labels,(1,1,SU2Irrep(0))),seed=15)
    model = ElectronicProblem(*integrals(2),(1,1))
    p = CyclicPairProblem(state,model.su2_mpo(),edge)
    frame = reference_pair_frame(p)
    p.materialize()
    np.testing.assert_allclose(p.metric,frame.conj().T@frame,atol=3e-12)
    np.testing.assert_allclose(p.hamiltonian,frame.conj().T@determinant_hamiltonian(model)@frame,atol=3e-12)

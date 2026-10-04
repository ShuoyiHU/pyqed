"""Independent conditional-ring amplitudes, local actions and one-site sweeps."""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
from pyqed.mps.symmetry import Sector
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed._letta_one_site_opt import ReducedPhysicalBasis, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.reduced_frontier import ReducedFrontier
from pyqed._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site
from pyqed._letta_one_site_opt.reduced_ring_target import signed_physical_basis
from pyqed._letta_one_site_opt.reduced_ring_state import ReducedRingLETTA
from pyqed._letta_one_site_opt.reduced_ring_solver import (
    ring_local_problem, ring_energy, ring_gauge_shift, optimize_ring_site, ring_dmrg)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from test_letta_ring_target import random_target_ring
from test_letta_native_ring import explicit_ring_vector
from test_letta_qchem import integrals, determinant_hamiltonian


def direct_conditional_ring(state):
    """Enumerate physical configurations without using the frontier embedding."""
    basis = state.physical_basis
    physical = tuple(q for q, r in zip(basis.sectors, basis.multiplicities) for _ in range(r))
    labels = tuple(i for i, p in enumerate(basis.reduced_states) for _ in range(p.irrep.dim))
    closure = expand_reduced_mps_site(state.closure)
    output = []
    for config in np.ndindex(*((basis.dense_dim,)*state.nsites)):
        condition = tuple(labels[p] for p in config)
        product = np.eye(closure.shape[2], dtype=complex)
        for i, p in enumerate(config):
            dependencies = tuple(condition[j] for j in state.site_neighborhood(i)[1:])
            blocks = {key: a[(slice(None), slice(None))+dependencies+(slice(None),)]
                      for key, a in state.tensors[i].items()}
            core = NonabelianTensor(data=blocks,
                qns=[state.left_virtual_sectors(i), physical, state.right_virtual_sectors(i)],
                dirs=[-1, 1, 1], metadata={'physical_basis': 'fully_reduced_su2'})
            product = product@expand_reduced_mps_site(core)[:, p, :]
        output.extend(np.trace(product@closure[:, m, :]) for m in range(closure.shape[1]))
    return np.asarray(output)


def molecular_state(*, charge=2, two_s=0, tied=True, copies=1, seed=73):
    basis = signed_physical_basis(ReducedPhysicalBasis.spatial_orbital())
    ring = random_target_ring(2, basis, SpinChargeSector(charge, SU2Irrep(two_s)), copies=copies, seed=seed)
    ties = ((0, 1), (1, 0)) if tied else ((0,), (1,))
    return ReducedRingLETTA.from_target_ring(ring, basis, neighborhoods=ties, normalize=True)


def test_arbitrary_forward_backward_wrap_ties_keep_closed_base_bond():
    basis = ReducedPhysicalBasis(('0', '1'), tuple(Sector(('number', 'su2'), (n, SU2Irrep(0))) for n in (0, 1)), (1, 1))
    target = Sector(('number', 'su2'), (1, SU2Irrep(0)))
    ring = random_target_ring(3, basis, target, copies=2)
    state = ReducedRingLETTA.from_target_ring(ring, basis, neighborhoods=((0, 2), (1, 0), (2, 1)))
    np.testing.assert_allclose(direct_conditional_ring(state), explicit_ring_vector(ring.sites), atol=2e-12)
    rng = np.random.default_rng(7)
    for core in state.tensors:
        for key, a in core.items():
            core[key] = a*(1+0.2*rng.normal(size=a.shape)+0.2j*rng.normal(size=a.shape))
    reference = direct_conditional_ring(state)
    np.testing.assert_allclose(explicit_ring_vector(state.to_target_ring().sites), reference, atol=2e-12)
    assert state.norm()**2 == pytest.approx(np.vdot(reference, reference).real, abs=2e-12)
    assert state.bond_dimensions[0] == state.bond_dimensions[-1] == 2
    state.normalize()
    assert state.norm() == pytest.approx(1., abs=2e-12)
    copied = state.copy()
    next(iter(copied.tensors[0].values())).flat[0] += 1.
    assert not np.allclose(direct_conditional_ring(copied), direct_conditional_ring(state))


@pytest.mark.parametrize('site', [0, 1, 2])
def test_tied_ring_local_actions_match_independent_full_frame(site):
    state = molecular_state(charge=1, two_s=1, copies=2)
    problem = ElectronicProblem(*integrals(2), (1, 0))
    local = ring_local_problem(state, problem.su2_mpo(), site)
    columns = []
    for x in np.eye(local.local_dimension):
        trial = state.copy()
        trial.set_site_blocks(site, local.embedding.unpack_source(x))
        columns.append(direct_conditional_ring(trial))
    frame = np.column_stack(columns)
    dense = np.kron(determinant_hamiltonian(problem), np.eye(2))
    np.testing.assert_allclose(local.metric, frame.conj().T@frame, atol=3e-12)
    np.testing.assert_allclose(local.hamiltonian, frame.conj().T@dense@frame, atol=3e-12)
    assert np.linalg.norm(local.metric-np.eye(local.local_dimension)) > 1e-3


@pytest.mark.parametrize('mode', ['frontier', 'scalar'])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_sector_gauges_across_every_ring_edge_preserve_physical_state(direction, mode):
    state = molecular_state(copies=2)
    before = direct_conditional_ring(state)
    for site in range(state.nsites+1):
        ring_gauge_shift(state, site, direction, mode=mode)
        np.testing.assert_allclose(direct_conditional_ring(state), before, atol=1e-11)


@pytest.mark.parametrize('charge,two_s', [(1, 1), (2, 0), (2, 2)])
@pytest.mark.parametrize('tied', [False, True])
def test_ring_one_site_sweeps_reach_small_hubbard_sector_ground_state(charge, two_s, tied, monkeypatch):
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic
    state = molecular_state(charge=charge, two_s=two_s, tied=tied, copies=2)
    h = np.array([[0., -1.], [-1., 0.]])
    g = np.zeros((2,)*4)
    g[0,0,0,0] = g[1,1,1,1] = 4.
    problem = ElectronicProblem(h, g, ((charge+two_s)//2, (charge-two_s)//2))
    # Triplet is Pauli-blocked; N=1 doublet and N=2 singlet have known dimer roots.
    exact = 0. if two_s == 2 else -1. if charge == 1 else (4-np.sqrt(32))/2
    def forbidden(*args, **kwargs):
        raise AssertionError('native ring sweep expanded variational magnetic tensors')
    with monkeypatch.context() as patch:
        patch.setattr(magnetic, 'expand_reduced_mps_site', forbidden)
        result = letta_dmrg(problem.su2_mpo(), state=state, options=LETTADMROptions(max_sweeps=6, matrix_free=False))
    assert result.energy == pytest.approx(exact, abs=2e-9)
    assert all(row.energy <= prior.energy+1e-10 for prior, row in zip(result.history, result.history[1:]))
    assert all(len(row.updates) == 3 and sum(u.is_target_closure for u in row.updates) == 1 for row in result.history)
    vector = direct_conditional_ring(result.state)
    dense = np.kron(determinant_hamiltonian(problem), np.eye(two_s+1))
    assert result.energy == pytest.approx(np.vdot(vector, dense@vector).real/np.vdot(vector, vector).real, abs=2e-10)
    assert result.state.norm() == pytest.approx(1., abs=2e-11)


def test_matrix_free_ring_update_matches_dense_update():
    state = molecular_state(tied=False, copies=2)
    p = ElectronicProblem(*integrals(2), (1, 1)).su2_mpo()
    dense, iterative = state.copy(), state.copy()
    full = ring_local_problem(state, p, 0)
    assert np.linalg.matrix_rank(full.metric, tol=1e-10) < full.local_dimension
    matrix_free = ring_local_problem(state, p, 0, matrix_free=True, dense_solver_threshold=0)
    assert matrix_free.metric is None and matrix_free.hamiltonian is None
    options = LETTADMROptions(matrix_free=False)
    a = optimize_ring_site(dense, p, 0, options)
    b = optimize_ring_site(iterative, p, 0, replace(options, matrix_free=True, dense_solver_threshold=0))
    assert a.accepted and b.accepted
    assert b.energy == pytest.approx(a.energy, abs=2e-10)


def test_failed_local_solve_and_gauge_restore_then_continue(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_ring_solver as solver
    state = molecular_state()
    p = ElectronicProblem(*integrals(2), (1, 1)).su2_mpo()
    before = direct_conditional_ring(state)
    solve = solver._solve_local_problem
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('injected ring solver failure')
    monkeypatch.setattr(solver, '_solve_local_problem', fail)
    update = optimize_ring_site(state, p, 0, LETTADMROptions(matrix_free=False))
    assert not update.accepted and update.recovery_reason
    np.testing.assert_array_equal(direct_conditional_ring(state), before)
    monkeypatch.setattr(solver, '_solve_local_problem', solve)
    gauge = solver.ring_gauge_shift
    calls = []
    def fail_once(state, site, direction, **kwargs):
        calls.append(site)
        if len(calls) == 1:
            next(iter(state.tensors[0].values())).flat[0] = np.nan
            raise FloatingPointError('injected partial gauge write')
        return gauge(state, site, direction, **kwargs)
    monkeypatch.setattr(solver, 'ring_gauge_shift', fail_once)
    result = ring_dmrg(p, state=state, options=LETTADMROptions(max_sweeps=2, matrix_free=False))
    assert result.sweeps == 2 and len(calls) == 6
    assert result.history[0].updates[0].recovery_reason.startswith('gauge:')
    assert np.isfinite(result.energy)
    assert result.energy <= ring_energy(state, p)+1e-10



def test_three_site_periodic_hubbard_doublet_closed_ring_reference():
    basis = signed_physical_basis(ReducedPhysicalBasis.spatial_orbital())
    target = SpinChargeSector(1, SU2Irrep(1))
    ring = random_target_ring(3, basis, target, copies=2)
    state = ReducedRingLETTA.from_target_ring(ring, basis, normalize=True)
    h1 = np.eye(3)-np.ones((3, 3))
    problem = ElectronicProblem(h1, np.zeros((3,)*4), (1, 0))
    result = letta_dmrg(problem.su2_mpo(), state=state,
                        options=LETTADMROptions(max_sweeps=5, matrix_free=False))
    assert result.energy == pytest.approx(-2., abs=2e-9)
    assert result.state.bond_dimensions[0] == 2
    vector = direct_conditional_ring(result.state)
    dense = np.kron(determinant_hamiltonian(problem), np.eye(2))
    assert np.vdot(vector, dense@vector).real/np.vdot(vector, vector).real == pytest.approx(-2., abs=2e-9)



def test_seeded_bose_ring_with_wrap_ties_and_public_mpo_dispatch():
    from pyqed._letta_one_site_opt import LatticeMPO
    basis = ReducedPhysicalBasis(('0', '1', '2'), tuple(
        Sector(('number', 'su2'), (n, SU2Irrep(0))) for n in range(3)), (1, 1, 1))
    target = Sector(('number', 'su2'), (2, SU2Irrep(0)))
    kwargs = dict(multiplets_per_sector=2, neighborhoods=((0, 1), (1, 0)), seed=91)
    state = ReducedRingLETTA.random(2, basis, target, **kwargs)
    repeated = ReducedRingLETTA.random(2, basis, target, **kwargs)
    np.testing.assert_array_equal(direct_conditional_ring(state), direct_conditional_ring(repeated))
    a = np.diag(np.sqrt([1., 2.]), 1)
    number = np.diag([0., 1., 2.])
    onsite = 1.5*number@(number-np.eye(3))
    first = np.stack([-a.T, -a, onsite, np.eye(3)])[None]
    last = np.stack([a, a.T, np.eye(3), onsite])[:, None]
    mpo = LatticeMPO((first, last))
    dense = -np.kron(a.T, a)-np.kron(a, a.T)+np.kron(onsite, np.eye(3))+np.kron(np.eye(3), onsite)
    configurations = list(np.ndindex(3, 3))
    sector = [i for i, c in enumerate(configurations) if sum(c) == 2]
    exact = np.linalg.eigvalsh(dense[np.ix_(sector, sector)])[0]
    result = letta_dmrg(mpo, state=state, options=LETTADMROptions(max_sweeps=5, matrix_free=False))
    assert result.energy == pytest.approx(exact, abs=2e-9)
    vector = direct_conditional_ring(result.state)
    assert sum(abs(vector[i])**2 for i in range(9) if i not in sector) < 1e-20
    assert result.state.neighborhoods == ((0, 1), (1, 0))
    assert result.state.bond_dimensions[0] == 2


def test_random_ring_nontrivial_spin_anchor_and_unreachable_target():
    basis = ReducedPhysicalBasis.spatial_orbital()
    state = ReducedRingLETTA.random(2, basis, SpinChargeSector(2, SU2Irrep(0)),
        anchor_sector=SpinChargeSector(0, SU2Irrep(1)), multiplets_per_sector=2, seed=41)
    from test_letta_qchem_symmetry import total_operators
    number, _, spin = total_operators(2)
    vector = direct_conditional_ring(state)
    np.testing.assert_allclose(number@vector, 2*vector, atol=3e-12)
    np.testing.assert_allclose(spin@vector, 0., atol=3e-12)
    assert state.norm() == pytest.approx(1., abs=2e-12)
    with pytest.raises(ValueError, match='unreachable'):
        ReducedRingLETTA.random(2, basis, SpinChargeSector(5, SU2Irrep(1)))


def test_ring_cbe_is_not_silently_replaced_by_one_site():
    state = molecular_state()
    with pytest.raises(NotImplementedError, match='ring CBE'):
        letta_dmrg(ElectronicProblem(*integrals(2), (1, 1)).su2_mpo(),
                   state=state, options=LETTADMROptions(cbe_enabled=True))

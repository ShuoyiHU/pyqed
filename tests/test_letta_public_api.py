"""Public model/topology/method choices checked with independent small operators."""
from dataclasses import replace
import os

import numpy as np
import pytest

from pyqed.letta import (ElectronicProblem, LETTAProblem, MetricCompressionOptions,
    OptimizationOptions, molecular, hubbard, bose_hubbard, heisenberg, random_state, solve)
from pyqed._letta_one_site_opt.reduced_ring_state import ReducedRingLETTA
from test_letta_qchem import integrals, determinant_hamiltonian
from test_letta_qchem_symmetry import dense_mpo, total_operators
from test_letta_ring_sweeps import direct_conditional_ring


def kron_operators(n, d, operators):
    result = np.ones((1, 1))
    for i in range(n):
        result = np.kron(result, operators.get(i, np.eye(d)))
    return result


def independent_model(kind, n=3, periodic=True):
    edges = [(i, i+1) for i in range(n-1)]+([(0, n-1)] if periodic and n > 2 else [])
    if kind.startswith('fermion'):
        h = -.2*np.eye(n)
        for i, j in edges:
            h[i, j] = h[j, i] = -1.
        g = np.zeros((n,)*4)
        for i in range(n):
            g[i, i, i, i] = 3.
        nelec = ((n+1)//2, n//2)
        p = ElectronicProblem(h, g, nelec)
        H = determinant_hamiltonian(p)
        N, Z, S2 = total_operators(n)
        indices = np.flatnonzero((np.diag(N) == n) & (np.diag(Z) == (n % 2)/2))
        frame = np.eye(4**n)[:, indices]
        mode = kind.split('-')[1]
        if mode == 'su2':
            values, vectors = np.linalg.eigh(frame.T@S2@frame)
            frame = frame@vectors[:, abs(values-(n % 2)*.75) < 1e-9]
        model = hubbard(n, nelec=nelec, U=3., mu=.2, periodic=periodic, symmetry=mode)
    elif kind == 'bose':
        d = 3
        a = np.diag(np.sqrt([1., 2.]), 1)
        number = np.diag([0., 1., 2.])
        onsite = 1.5*number@(number-np.eye(d))-.2*number
        H = sum(kron_operators(n, d, {i: onsite}) for i in range(n))
        for i, j in edges:
            H -= kron_operators(n, d, {i: a.T, j: a})+kron_operators(n, d, {i: a, j: a.T})
        charges = np.array([sum(x) for x in np.ndindex(*((d,)*n))])
        frame = np.eye(d**n)[:, charges == n]
        model = bose_hubbard(n, particles=n, max_occupancy=2, U=3., mu=.2, periodic=periodic)
    else:
        sx = .5*np.array([[0., 1.], [1., 0.]])
        sy = .5*np.array([[0., 1j], [-1j, 0.]])
        sz = .5*np.diag([-1., 1.])
        H = sum((kron_operators(n, 2, {i: a, j: a}) for i, j in edges for a in (sx, sy, sz)),
                np.zeros((2**n, 2**n), dtype=complex))
        total = [sum(kron_operators(n, 2, {i: a}) for i in range(n)) for a in (sx, sy, sz)]
        frame = np.eye(2**n)[:, np.abs(np.diag(total[2])-(n % 2)/2) < 1e-12]
        mode = kind.split('-')[1]
        if mode == 'su2':
            values, vectors = np.linalg.eigh(frame.T@sum(a@a for a in total)@frame)
            frame = frame@vectors[:, abs(values-(n % 2)*.75) < 1e-9]
        model = heisenberg(n, periodic=periodic, symmetry=mode)
    return model, H, np.linalg.eigvalsh(frame.conj().T@H@frame)[0]


def physical_components(state):
    if isinstance(state, ReducedRingLETTA):
        return direct_conditional_ring(state).reshape(state.physical_basis.dense_dim**state.nsites, -1)
    return state.state_vector()[:, None]


def energy(components, h):
    return np.vdot(components, h@components).real/np.vdot(components, components).real


def full_ring_stress():
    """Opt in to the original simultaneous full-sector, wrap-tied fixtures."""
    return os.environ.get('LETTA_FULL_RING_STRESS') == '1'


def options(**kwargs):
    return OptimizationOptions(max_sweeps=2, energy_refinement_rounds=2,
        compression=MetricCompressionOptions(als_max_iterations=4, lsmr_max_iterations=60), **kwargs)


@pytest.mark.parametrize('kind', ['fermion-u1', 'fermion-su2', 'bose', 'spin-u1', 'spin-su2'])
@pytest.mark.parametrize('periodic', [False, True])
def test_public_models_match_independent_hamiltonians(kind, periodic):
    model, h, _ = independent_model(kind, periodic=periodic)
    np.testing.assert_allclose(dense_mpo(model.hamiltonian.canonical_factors), h, atol=3e-12)
    native = model.hamiltonian.native_mpo(model.physical_basis)
    np.testing.assert_allclose(dense_mpo(native.component_factors()), h, atol=4e-11)


@pytest.mark.parametrize('kind', ['fermion-u1', 'fermion-su2', 'bose', 'spin-u1', 'spin-su2'])
@pytest.mark.parametrize('topology', ['open', 'ring'])
@pytest.mark.parametrize('method', ['one-site', 'cbe', 'two-site'])
def test_public_method_symmetry_topology_matrix(kind, topology, method):
    model, h, exact = independent_model(kind)
    ties = ((0, 2), (1,), (2, 0))
    # Two simultaneous wrap dependencies multiply the exact frontier size.
    # Keep a wrap tie in the routine U(1) integration test; retain the original
    # combined allocation as an explicitly requested stress configuration.
    if kind == 'fermion-u1' and topology == 'ring' and not full_ring_stress():
        ties = ((0,), (1,), (2, 0))
    state = random_state(model, topology=topology, ties=ties, seed=31)
    before = physical_components(state)
    cap = max(map(len, state.bond_sectors), default=1)
    result = solve(model, state=state, method=method, bond_dim=cap, options=options())
    vector = physical_components(result.state)
    actual = energy(vector, h)
    assert result.energy == pytest.approx(actual, abs=3e-9)
    assert exact-3e-9 <= actual <= energy(before, h)+3e-9
    assert max(map(len, result.state.bond_sectors), default=1) <= cap
    assert tuple(result.state.site_neighborhood(i) for i in range(3)) == ties
    assert 1 <= result.sweeps <= 2
    np.testing.assert_array_equal(physical_components(state), before)
    # A recovered/iteration-capped result remains inspectable, not a hidden
    # change to symmetry, topology, or Hamiltonian.
    assert len(result.history[-1].updates) == (3 if topology == 'open' else 4)-(method == 'two-site' and topology == 'open')


@pytest.mark.parametrize('topology', ['open', 'ring'])
@pytest.mark.parametrize('solver', ['als', 'variable-projection', 'joint-ls', 'grassmann-newton'])
@pytest.mark.parametrize('method', ['cbe', 'two-site'])
def test_public_molecular_compression_selections(topology, solver, method):
    p = ElectronicProblem(*integrals(2), (1, 1), .37)
    model = molecular(p)
    state = random_state(model, topology=topology, ties='nn-periodic', seed=43)
    if method == 'cbe':
        # A fully allocated dimer already reaches its optimum in one ordinary
        # update and correctly skips CBE fitting. Start with a missing charge
        # sector so this test actually exercises the requested compressor.
        if topology == 'open':
            from test_letta_reduced_cbe import incomplete
        else:
            from test_letta_ring_cbe import incomplete
        state, _ = incomplete()
    initial_energy = energy(physical_components(state), determinant_hamiltonian(p))
    controls = replace(options(), max_sweeps=1, compression=MetricCompressionOptions(
        solver=solver, als_max_iterations=7, lsmr_max_iterations=53, max_iterations=8))
    result = solve(model, state=state, method=method, options=controls)
    assert result.energy == pytest.approx(energy(physical_components(result.state), determinant_hamiltonian(p)), abs=3e-9)
    if method == 'two-site':
        reports = [u.compression_diagnostics for s in result.history for u in s.updates if u.compression_diagnostics]
    else:
        reports = [d for s in result.history for u in s.updates for d in u.cbe_compression_diagnostics]
        reports += [d for s in result.history for u in s.updates
                    for d in (u.cbe_selection_diagnostics or {}).get('factor_fits', ())]
    assert reports
    updates = [u for row in result.history for u in row.updates]
    assert not any(getattr(u, 'recovery_reason', None) or getattr(u, 'cbe_recovery_reason', None) for u in updates)
    if method == 'cbe':
        assert any(u.cbe_expansion_dimension > 0 for u in updates)
        assert result.energy < initial_energy-1e-8
    assert all(r.get('requested_solver') == solver for r in reports)
    if solver == 'als':
        linear = [entry for report in reports for entry in report.get('linear_solves', ())]
        assert linear
        assert all(entry['max_iterations'] == 53 and entry['iterations'] <= 53 for entry in linear)
        if method == 'two-site':
            assert all(u.truncation_iterations <= 7 for s in result.history for u in s.updates)


@pytest.mark.parametrize('kind', ['bose', 'spin-su2', 'fermion-u1'])
@pytest.mark.parametrize('seed,copies', [(11, 1), (19, 2)])
def test_public_initial_allocations_and_nontrivial_ring(kind, seed, copies):
    model, h, _ = independent_model(kind, n=2)
    ties = 'nn-periodic'
    if kind == 'fermion-u1' and copies == 2 and not full_ring_stress():
        # This case checks multiplicity and the nonunit closing bond. The
        # copies=1 case separately checks both physical-index dependencies.
        ties = 'none'
    state = random_state(model, topology='ring', ties=ties,
                         multiplets_per_sector=copies, seed=seed)
    assert len(state.bond_sectors[0]) == len(state.bond_sectors[-1]) == copies
    before = physical_components(state)
    result = solve(model, state=state, options=replace(options(), max_sweeps=1))
    assert result.energy <= energy(before, h)+3e-9
    assert result.energy == pytest.approx(energy(physical_components(result.state), h), abs=3e-9)
    np.testing.assert_array_equal(physical_components(state), before)


def test_hamiltonian_topology_and_wrap_tie_are_independent():
    model = hubbard(3, nelec=(2, 1), periodic=True)
    for topology in ('open', 'ring'):
        for ties, last in [('none', (2,)), ('nn', (2,)), ('nn-periodic', (2, 0))]:
            state = random_state(model, topology=topology, ties=ties, seed=4)
            assert state.site_neighborhood(2) == last
            assert isinstance(state, ReducedRingLETTA) == (topology == 'ring')


@pytest.mark.parametrize('factory', [lambda: hubbard(3, nelec=(2, 1), bonds=[(0, 1), (1, 0)]),
    lambda: heisenberg(3, bonds=[(0, 0)]), lambda: bose_hubbard(3, particles=2, max_occupancy=2, bonds=[(0, 3)]),
    lambda: hubbard(3, nelec=(2, 1), periodic=True, bonds=[]),
    lambda: heisenberg(3, two_s=0), lambda: heisenberg(3, symmetry='su2', two_sz=1),
    lambda: bose_hubbard(2, particles=5, max_occupancy=2), lambda: OptimizationOptions(max_sweeps=0)])
def test_invalid_inputs_fail_before_sweeps(factory):
    with pytest.raises(ValueError):
        factory()


def test_supplied_state_model_and_budget_mismatches_are_rejected():
    model, _, _ = independent_model('fermion-su2', n=2)
    state = random_state(model, seed=13)
    for method in ('one-site', 'cbe'):
        with pytest.raises(ValueError, match='initial multiplet'):
            solve(model, state=state, method=method, bond_dim=1)
    with pytest.raises(ValueError, match='target sector'):
        solve(hubbard(2, nelec=(1, 0)), state=state)
    with pytest.raises(ValueError, match='method'):
        solve(model, state=state, method='invalid')
    with pytest.raises(ValueError, match='topology'):
        random_state(model, topology='open', anchor_sector=model.symmetry.identity)


def test_generic_graph_edges_are_preserved_in_molecular_adapter():
    model = hubbard(4, nelec=(2, 2), bonds=[(0, 3), (0, 2), (2, 1)], U=2.)
    h = np.zeros((4, 4))
    for i, j in [(0, 3), (0, 2), (2, 1)]:
        h[i, j] = h[j, i] = -1.
    g = np.zeros((4,)*4)
    for i in range(4):
        g[i, i, i, i] = 2.
    np.testing.assert_allclose(dense_mpo(model.hamiltonian.canonical_factors),
                              determinant_hamiltonian(ElectronicProblem(h, g, (2, 2))), atol=5e-12)


@pytest.mark.parametrize('mode', ['n', 'n_sz', 'nalpha_nbeta', 'su2'])
def test_molecular_modes_preserve_core_energy_and_sector(mode):
    p = ElectronicProblem(*integrals(2), (1, 0), .73)
    model = molecular(p, symmetry=mode)
    np.testing.assert_allclose(dense_mpo(model.hamiltonian.canonical_factors), determinant_hamiltonian(p), atol=4e-12)
    state = random_state(model, seed=51)
    N, Z, S2 = total_operators(2)
    v = physical_components(state)
    np.testing.assert_allclose(N@v, v, atol=3e-13)
    if mode != 'n':
        np.testing.assert_allclose(Z@v, .5*v, atol=3e-13)
    if mode == 'su2':
        np.testing.assert_allclose(S2@v, .75*v, atol=3e-13)


def test_custom_nonhermitian_scalar_mpo_is_an_input_error():
    from pyqed.letta import ReducedPhysicalBasis, ReducedSymmetry, ReducedMPOHamiltonian
    basis = ReducedPhysicalBasis.spatial_orbital()
    symmetry = ReducedSymmetry.su2(basis, target_charge=1, target_two_j=1)
    with pytest.raises(ValueError, match='Hermitian'):
        LETTAProblem(ReducedMPOHamiltonian(None, (1j*np.eye(4)[None, None],)), symmetry)


def test_negative_spin_projection_is_supported():
    model = heisenberg(3, symmetry='u1', two_sz=-1)
    v = physical_components(random_state(model, seed=11))
    z = sum(kron_operators(3, 2, {i: np.diag([-1., 1.])}) for i in range(3))
    np.testing.assert_allclose(z@v, -v, atol=3e-13)

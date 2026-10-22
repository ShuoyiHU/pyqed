"""Exact pair support: preserve the free pair space and legal factor space."""
from collections import Counter
from itertools import product

import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    LETTADMROptions, ReducedLatticeLETTA, ReducedPhysicalBasis, ReducedSymmetry,
    su2_heisenberg_mpo,
)
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site
from pyqed._letta_one_site_opt.reduced_cbe import (
    expand_reduced_cbe, reduced_cbe_site, select_reduced_cbe,
)
from pyqed._letta_two_site_opt import reduced_solver as pair_solver
from pyqed._letta_two_site_opt.reduced_compression import reduced_factor_blocks
from test_letta_frontier_support import restricted_bond
from test_letta_qchem import determinant_hamiltonian
from test_letta_reduced_cbe import controls, incomplete


def electronic_state(nelec, ties, *, copies=1):
    h = np.array([[.1, -1., .25], [-1., -.2, -.8], [.25, -.8, .3]])
    g = np.zeros((3,) * 4)
    for i in range(3):
        g[i, i, i, i] = 4.
    electronic = ElectronicProblem(h, g, nelec)
    state = ReducedLatticeLETTA.random((1, 3), symmetry=electronic.symmetry('su2'),
        neighborhoods=ties, multiplets_per_sector=copies, seed=2117,
        real=False, normalize=False)
    return state, electronic


def pair_injection(state, old, new):
    """Independently map compact outer assignment/copy rows into Cartesian rows."""
    support = new.frontier.support(state)
    i = new.left_site
    left_copies = Counter(state.left_virtual_sectors(i))
    right_copies = Counter(state.right_virtual_sectors(i + 1))

    def rows(cut, sector, copies):
        variables = old.frontier.left_variables(cut) if cut < state.nsites else ()
        assignments = tuple(product(range(state.physical_dim), repeat=len(variables)))
        lookup = {a: k for k, a in enumerate(assignments)}
        return np.array([lookup[a] * copies + j
                         for a in support.bases[cut][sector] for j in range(copies)])

    injection = np.empty(new.local_dimension, dtype=int)
    for key, shape in new.layout.shapes.items():
        original = np.arange(*old.layout.offsets[key]).reshape(old.layout.shapes[key])
        selected = original[np.ix_(rows(i, key[0], left_copies[key[0]]),
            np.arange(shape[1]), np.arange(shape[2]),
            rows(i + 2, key[-1], right_copies[key[-1]]))]
        start, stop = new.layout.offsets[key]
        injection[start:stop] = selected.ravel()
    assert len(np.unique(injection)) == len(injection)
    return injection


def pair_dense_frame(state, problem):
    """Tiny reference only: contract each free pair basis vector in magnetic space.

    Keep all target magnetic components; native reduced norms trace this leg.
    """
    sites = problem.frontier.to_mps(state)
    dense_sites = [expand_reduced_mps_site(site) for site in sites]
    i = problem.left_site
    left = np.ones((1, 1), dtype=complex)
    for tensor in dense_sites[:i]:
        left = np.tensordot(left, tensor, axes=(-1, 0)).reshape(-1, tensor.shape[-1])
    columns = []
    for vector in np.eye(problem.local_dimension):
        tensor = pair_solver._expand_pair_blocks(problem.layout.unpack(vector),
                                                 sites[i], sites[i + 1])
        value = np.tensordot(left, tensor, axes=(-1, 0))
        value = value.reshape(-1, tensor.shape[-1])
        for tensor in dense_sites[i + 2:]:
            value = np.tensordot(value, tensor, axes=(-1, 0)).reshape(-1, tensor.shape[-1])
        columns.append(value)
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize('nelec', [(1, 1), (2, 1)])
@pytest.mark.parametrize('ties', [((0, 2), (1,), (2,)), ((0, 2), (1, 0), (2, 1))])
@pytest.mark.parametrize('left_site', [0, 1])
def test_free_pair_full_actions_match_cartesian_and_dense(nelec, ties, left_site):
    state, electronic = electronic_state(nelec, ties)
    hamiltonian = electronic.su2_mpo()
    old, new = [pair_solver.reduced_pair_problem(state, hamiltonian, left_site,
        prune=prune, matrix_free=True, dense_solver_threshold=0) for prune in (False, True)]
    assert new.frontier.prune
    assert new.local_dimension <= old.local_dimension
    injection = pair_injection(state, old, new)
    frame = pair_dense_frame(state, old)
    # Dropped free-pair directions must be physically zero, not merely absent
    # from today's factorized incumbent or tangent space.
    removed = np.setdiff1d(np.arange(old.local_dimension), injection)
    np.testing.assert_allclose(frame[:, :, removed], 0., atol=3e-13)
    compact_frame = pair_dense_frame(state, new)
    np.testing.assert_allclose(compact_frame, frame[:, :, injection], atol=3e-13)
    dense_h = determinant_hamiltonian(electronic)
    expected_n = np.einsum('amx,amy->xy', compact_frame.conj(), compact_frame)
    expected_h = np.einsum('amx,ab,bmy->xy', compact_frame.conj(), dense_h, compact_frame)
    for name, expected in [('apply_metric', expected_n), ('apply_hamiltonian', expected_h)]:
        new_matrix = np.column_stack([getattr(new, name)(v) for v in np.eye(new.local_dimension)])
        old_columns = []
        for j in injection:
            v = np.zeros(old.local_dimension)
            v[j] = 1.
            old_columns.append(getattr(old, name)(v)[injection])
        np.testing.assert_allclose(new_matrix, np.column_stack(old_columns), atol=3e-11, rtol=3e-11)
        np.testing.assert_allclose(new_matrix, expected, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(new.old_vector, old.old_vector[injection], atol=1e-13)


@pytest.mark.parametrize('left_site', [0, 1])
def test_compact_factor_selection_adjoint_and_gauge_preserve_pair(left_site):
    state, electronic = electronic_state((2, 1), ((0, 1, 2), (1, 0, 2), (2, 0, 1)), copies=2)
    problem = pair_solver.reduced_pair_problem(state, electronic.su2_mpo(), left_site,
        prune=True, matrix_free=True, dense_solver_threshold=0)
    embeddings = [problem.frontier.site_embedding(state, k) for k in (left_site, left_site + 1)]
    assert any(e.source_size < e.source_layout.size for e in embeddings)
    retained = {q: 1 for q in state.bond_sectors[left_site]}
    for side, embedding in zip(('left', 'right'), embeddings):
        raw_allowed = []
        for key, shape in embedding.source_layout.shapes.items():
            grid = np.arange(*embedding.source_layout.offsets[key]).reshape(shape)
            section = grid[..., :retained[key[2]]] if side == 'left' else grid[:retained[key[0]]]
            raw_allowed.extend(section.ravel())
        expected = np.flatnonzero(np.isin(embedding.supported_source_indices, raw_allowed))
        actual = pair_solver._active_source_indices(embedding, retained, side)
        np.testing.assert_array_equal(actual, expected)

    rng = np.random.default_rng(2119)
    a, b = [rng.normal(size=e.source_size) + 1j * rng.normal(size=e.source_size) for e in embeddings]
    le, re = embeddings
    g = rng.normal(size=problem.local_dimension) + 1j * rng.normal(size=problem.local_dimension)
    merge = lambda a, b: pair_solver._pair_vector_from_sources(problem.layout, le, re, a, b)
    left_blocks = pair_solver._expanded_source_blocks(le, a)
    right_blocks = pair_solver._expanded_source_blocks(re, b)
    np.testing.assert_allclose(np.vdot(merge(a, b), g),
        np.vdot(a, pair_solver._left_source_adjoint(problem.layout, g, right_blocks, le)), atol=3e-11)
    np.testing.assert_allclose(np.vdot(merge(a, b), g),
        np.vdot(b, pair_solver._right_source_adjoint(problem.layout, left_blocks, g, re)), atol=3e-11)
    groups = reduced_factor_blocks(state, left_site, le, re, Counter(state.bond_sectors[left_site]))
    assert groups
    gauged_a, gauged_b = a.copy(), b.copy()
    for rows, columns in groups:
        assert rows.min() >= 0 and rows.max() < le.source_size
        assert columns.min() >= 0 and columns.max() < re.source_size
        assert rows.shape[1] == columns.shape[0]
        scales = np.linspace(.8, 1.3, rows.shape[1])
        gauged_a[rows] *= scales
        gauged_b[columns] /= scales[:, None]
    np.testing.assert_allclose(merge(gauged_a, gauged_b), merge(a, b), atol=3e-11)


def test_pair_pruning_does_not_block_missing_spin_sector_growth():
    basis = ReducedPhysicalBasis.spin_half()
    state = ReducedLatticeLETTA.random((1, 4),
        symmetry=ReducedSymmetry.su2(basis, target_two_j=0), seed=2121, normalize=False)
    state = restricted_bond(state, 1, lambda q: q.irrep.two_j == 0)
    expanded = pair_solver._expand_reduced_pair_space(state, 1, 2)
    np.testing.assert_allclose(expanded.state_vector(), state.state_vector(), atol=1e-13)
    assert any(q.irrep.two_j == 2 for q in expanded.bond_sectors[1])
    hamiltonian = su2_heisenberg_mpo(4, physical_basis=basis)
    problem = pair_solver.reduced_pair_problem(expanded, hamiltonian, 1, prune=True)
    assert any(key[2].irrep.two_j == 2 for key in problem.layout.keys)
    spin_one = [index for key in problem.layout.keys if key[2].irrep.two_j == 2
                for index in range(*problem.layout.offsets[key])]
    diagonal = np.diag(problem.metric)
    assert np.any(diagonal[spin_one] > 1e-12)


def missing_charge_state(doublet):
    if not doublet:
        return incomplete()
    h = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 0.]])
    p = ElectronicProblem(h, np.zeros((3,) * 4), (2, 1))
    sym = p.symmetry('su2')
    vacuum, single, double = sym.physical_basis.sectors
    target = sym.sector
    a = np.zeros((1, 1, 3, 3, 3)); a[0, 0, 0, 1, 0] = 1.
    b = np.zeros((3, 1, 3, 1)); b[0, 0, 1, 0] = 1.
    c = np.ones((1, 1, 1))
    state = ReducedLatticeLETTA((1, 3), sym,
        [{(vacuum, double, double): a}, {(double, vacuum, double): b}, {(double, single, target): c}],
        bond_sectors=((double, double, double), (double,)),
        neighborhoods=((0, 1, 2), (1, 2), (2,)), normalize=False)
    return state, p.su2_mpo()


@pytest.mark.parametrize('doublet', [False, True])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_cbe_missing_charge_growth_matches_unpruned_and_preserves_state(monkeypatch, doublet, direction):
    state, hamiltonian = missing_charge_state(doublet)
    options = controls()
    original = pair_solver.reduced_pair_problem
    results, selections = [], []
    for prune in (False, True):
        def build(*args, **kwargs):
            kwargs['prune'] = prune
            return original(*args, **kwargs)
        monkeypatch.setattr(pair_solver, 'reduced_pair_problem', build)
        selection = select_reduced_cbe(state, hamiltonian, 0, options)
        assert selection.problem.frontier.prune == prune
        assert selection.missing_norm > .1
        assert selection.projection_converged
        assert any(q not in state.bond_sectors[0] for q in selection.multiplicities)
        expanded = expand_reduced_cbe(state, selection, direction)
        np.testing.assert_allclose(expanded.state_vector(), state.state_vector(), atol=3e-13)
        candidate = state.copy()
        site = 0 if direction == 'lr' else 1
        update = reduced_cbe_site(candidate, hamiltonian, site, direction, 3, options)
        assert update.cbe_recovery_reason is None
        assert update.energy <= update.cbe_baseline_energy + 1e-10
        results.append(update.energy)
        selections.append(selection)
    np.testing.assert_allclose(results[0], results[1], atol=3e-9)
    np.testing.assert_allclose(selections[0].missing_norm, selections[1].missing_norm, atol=3e-10)
    np.testing.assert_allclose(selections[0].captured_weight, selections[1].captured_weight, atol=3e-10)


def test_pair_relaxation_keeps_nonzero_direction_forbidden_by_global_factor_support():
    from pyqed._letta_one_site_opt import ReducedFrontier
    # Physical state |empty, double, empty, empty>. The first tensor remembers
    # site 2. Globally legal factor paths require that remembered label=empty.
    # A free merged (1,2) tensor may independently use a different remembered
    # label, and that raw pair direction still contracts to a nonzero state.
    p = ElectronicProblem(np.eye(4), np.zeros((4,) * 4), (1, 1))
    sym = p.symmetry('su2')
    vacuum, single, double = sym.physical_basis.sectors
    first = np.ones((1, 1, 3, 1))
    unit = np.ones((1, 1, 1))
    state = ReducedLatticeLETTA((1, 4), sym,
        [{(vacuum, vacuum, vacuum): first},
         {(vacuum, double, double): unit.copy()},
         {(double, vacuum, double): unit.copy()},
         {(double, vacuum, double): unit.copy()}],
        bond_sectors=((vacuum,), (double,), (double,)),
        neighborhoods=((0, 2), (1,), (2,), (3,)), normalize=False)
    ordinary = ReducedFrontier.from_state(state)
    assert ordinary.support(state).bases[1][vacuum] == ((0,),)
    old, new = [pair_solver.reduced_pair_problem(state, p.su2_mpo(), 1,
        prune=prune, matrix_free=True, dense_solver_threshold=0) for prune in (False, True)]
    assert new.frontier.support(state).bases[1][vacuum] == ((0,), (1,), (2,))
    injection = pair_injection(state, old, new)
    np.testing.assert_array_equal(injection, np.arange(3))
    # Only 3 pair columns and 256 determinant rows: this explicit adversarial
    # reference is far smaller than a generic dense four-site pair test.
    frame = pair_dense_frame(state, old)
    assert np.linalg.norm(frame[:, :, 2]) == pytest.approx(1., abs=1e-13)
    for problem in (old, new):
        metric = np.column_stack([problem.apply_metric(x) for x in np.eye(3)])
        np.testing.assert_allclose(metric, np.ones((3, 3)), atol=1e-13)
    # Legal LETTA factors cannot vary this extra copy independently.
    embedding = new.frontier.site_embedding(state, 1)
    np.testing.assert_array_equal(embedding.supported_source_indices, [0])


def test_pair_support_rebuilds_after_growth():
    from pyqed._letta_two_site_opt.pair_frontier import ReducedPairFrontier
    state, _ = incomplete()
    frontier = ReducedPairFrontier.from_state(state, 0)
    original_support = frontier.support(state)
    original = frontier.site_embedding(state, 0)
    expanded = pair_solver._expand_reduced_pair_space(state, 0, 4)
    fresh = ReducedPairFrontier.from_state(expanded, 0)
    assert frontier.support(expanded) is not original_support
    assert frontier.support(expanded).bases == fresh.support(expanded).bases
    for site in (0, 1):
        reused = frontier.site_embedding(expanded, site)
        expected = fresh.site_embedding(expanded, site)
        if site == 0:
            assert reused is not original
        for name in ('supported_source_indices', 'source_indices', 'target_indices'):
            np.testing.assert_array_equal(getattr(reused, name), getattr(expected, name))
        assert reused.source_layout.shapes == expected.source_layout.shapes
        assert reused.target_layout.shapes == expected.target_layout.shapes


@pytest.mark.parametrize('nelec', [(1, 1), (2, 1)])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_tied_pair_solve_and_compression_match_unpruned(monkeypatch, nelec, direction):
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions
    state, electronic = electronic_state(nelec, ((0,1,2),(1,0,2),(2,0,1)))
    hamiltonian = electronic.su2_mpo()
    dense_h = determinant_hamiltonian(electronic)
    opts = LETTATwoSiteOptions(gauge_mode='none', split_method='metric-als')
    original = pair_solver.reduced_pair_problem
    reports = []
    for prune in (False, True):
        def build(*args, **kwargs):
            kwargs['prune'] = prune
            return original(*args, **kwargs)
        monkeypatch.setattr(pair_solver, 'reduced_pair_problem', build)
        candidate = pair_solver._expand_reduced_pair_space(state, 0, 3)
        report = pair_solver._optimize_allocated_reduced_pair(candidate, hamiltonian, 0, direction, 3, opts)
        assert report.accepted
        assert not report.fallback
        assert report.recovery_reason is None
        vector = candidate.state_vector()
        expected = np.vdot(vector, dense_h @ vector).real / np.vdot(vector, vector).real
        np.testing.assert_allclose(report.energy, expected, atol=3e-10)
        assert candidate.symmetry_violation() == 0.
        reports.append(report)
    np.testing.assert_allclose(reports[0].local_energy, reports[1].local_energy, atol=3e-10)
    np.testing.assert_allclose(reports[0].energy, reports[1].energy, atol=3e-9)

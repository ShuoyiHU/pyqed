"""Native reduced CBE: metric residual, legal expansion and variational guard."""
from dataclasses import replace
import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_cbe import select_reduced_cbe, expand_reduced_cbe, reduced_cbe_site
from pyqed._letta_one_site_opt.reduced_solver import _energy, optimize_reduced_site
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem, _expand_reduced_pair_space
from pyqed._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot
from pyqed._letta_compression import MetricCompressionOptions
from test_letta_qchem import integrals


def incomplete():
    p = ElectronicProblem(np.array([[0., -1.], [-1., 0.]]), np.zeros((2,)*4), (1, 1))
    sym = p.symmetry('su2')
    vacuum, single, double = sym.physical_basis.sectors
    a = np.zeros((1, 1, 3, 3)); a[0, 0, 0, 0] = 1.
    b = np.zeros((3, 1, 1)); b[0, 0, 0] = 1.
    s = ReducedLatticeLETTA((1, 2), sym,
        [{(vacuum, double, double): a}, {(double, vacuum, double): b}],
        bond_sectors=((double, double, double),))
    return s, p.su2_mpo()


def controls(**kwargs):
    return LETTADMROptions(cbe_enabled=True, max_sweeps=4,
        cbe_energy_refinement_max_iterations=3, cbe_refinement_max_iterations=5,
        cbe_projection_max_iterations=100,
        compression=MetricCompressionOptions(als_max_iterations=8, lsmr_max_iterations=50),
        **kwargs)


def test_supported_inverse_satisfies_native_metric_identity():
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), seed=12, real=False)
    pair = reduced_pair_problem(s, p.su2_mpo(), 1, matrix_free=True, dense_solver_threshold=0)
    root = ReducedPairMetricRoot(pair, s, 1e-12)
    rng = np.random.default_rng(7)
    x = rng.normal(size=pair.local_dimension)+1j*rng.normal(size=pair.local_dimension)
    nx = pair.apply_metric(x)
    np.testing.assert_allclose(pair.apply_metric(root.inverse_action(nx)), nx, atol=2e-10)
    y = rng.normal(size=root.size)+1j*rng.normal(size=root.size)
    np.testing.assert_allclose(root.apply(root.unwhiten(y)), y, atol=2e-10)
    np.testing.assert_allclose(np.vdot(x, root.unwhiten(y)), np.vdot(root.unwhiten_adjoint(x), y), atol=2e-10)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_selection_opens_missing_sector_and_padding_preserves_state(direction):
    s, h = incomplete()
    selection = select_reduced_cbe(s, h, 0, controls())
    assert selection.missing_norm > .1
    assert selection.projection_converged
    assert selection.tangent_overlap < 1e-8
    assert sum(selection.multiplicities.values()) == 1
    assert any(q not in s.bond_sectors[0] for q in selection.multiplicities)
    expanded = expand_reduced_cbe(s, selection, direction)
    assert len(expanded.bond_sectors[0]) == 4
    np.testing.assert_allclose(expanded.state_vector(), s.state_vector(), atol=1e-13)
    assert expanded.symmetry_violation() == 0.


def test_cbe_uses_expanded_one_site_not_pair_diagonalization(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as pair
    import pyqed._letta_one_site_opt.reduced_contraction as magnetic
    s, h = incomplete()
    initial = _energy(s, h, stable=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('CBE used pair eigensolve or magnetic/global expansion')
    monkeypatch.setattr(pair, '_solve_local_problem', forbidden)
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    monkeypatch.setattr(magnetic, 'expand_reduced_mps_site', forbidden)
    ordinary = letta_dmrg(h, state=s, options=LETTADMROptions(max_sweeps=4))
    result = letta_dmrg(h, state=s, options=controls())
    assert ordinary.energy == pytest.approx(initial, abs=1e-12)
    assert result.energy == pytest.approx(-2., abs=1e-8)
    assert max(map(len, result.state.bond_sectors)) <= 3
    updates = [u for row in result.history for u in row.updates]
    assert any(u.cbe_expansion_dimension for u in updates)
    assert not any(u.cbe_recovery_reason for u in updates)
    assert all(u.energy <= u.cbe_baseline_energy+1e-10 for u in updates if u.cbe_baseline_energy is not None)


@pytest.mark.parametrize('solver', ['als', 'variable-projection', 'joint-ls', 'grassmann-newton'])
def test_cbe_compressors_and_same_start_one_site_baseline(solver):
    s, h = incomplete()
    options = replace(controls(), compression=MetricCompressionOptions(solver=solver,
        max_iterations=15, als_max_iterations=8, lsmr_max_iterations=50))
    baseline = s.copy()
    optimize_reduced_site(baseline, h, 0, replace(options, cbe_enabled=False))
    update = reduced_cbe_site(s, h, 0, 'lr', 3, options)
    assert update.energy <= _energy(baseline, h, stable=True)+1e-10
    assert update.cbe_expanded_energy < update.cbe_baseline_energy-1e-4
    assert update.cbe_recovery_reason is None
    assert update.cbe_compression_diagnostics[0]['requested_solver'] == solver


def test_failed_expansion_restores_and_returns_ordinary_update(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_cbe as cbe
    s, h = incomplete()
    options = controls()
    expected = s.copy()
    baseline = optimize_reduced_site(expected, h, 0, replace(options, cbe_enabled=False))
    def fail(candidate, *args, **kwargs):
        for a in candidate.tensors[0].values(): a.fill(np.nan)
        candidate.bond_sectors = ()
        raise FloatingPointError('injected CBE failure')
    monkeypatch.setattr(cbe, 'select_reduced_cbe', fail)
    update = reduced_cbe_site(s, h, 0, 'lr', 3, options)
    assert update.cbe_fallback and update.cbe_baseline_selected
    assert 'injected CBE failure' in update.cbe_recovery_reason
    assert update.energy == pytest.approx(baseline.energy, abs=1e-12)
    assert s.bond_sectors == expected.bond_sectors
    np.testing.assert_array_equal(s.state_vector(), expected.state_vector())


@pytest.mark.parametrize('ties', [((0,), (1,), (2,)), ((0, 2), (1, 0), (2, 1))])
def test_complex_molecular_cbe_matches_independent_reference_and_preserves_ties(ties):
    from test_letta_qchem import determinant_hamiltonian
    p = ElectronicProblem(*integrals(3), (2, 1), .2)
    h = p.su2_mpo()
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'),
        neighborhoods=ties, real=False, seed=32)
    before = _energy(s, h, stable=True)
    result = letta_dmrg(h, state=s, options=replace(controls(), max_sweeps=3, gauge_mode='none'))
    vector = result.state.state_vector()
    dense = determinant_hamiltonian(p)
    reference = np.vdot(vector, dense@vector).real/np.vdot(vector, vector).real
    assert result.energy == pytest.approx(reference, abs=2e-9)
    assert result.energy <= before+1e-10
    assert result.state.symmetry_violation() == 0.
    assert tuple(result.state.site_neighborhood(i) for i in range(3)) == ties
    updates = [u for row in result.history for u in row.updates]
    assert not any(u.cbe_recovery_reason for u in updates)
    assert all(u.energy <= u.cbe_baseline_energy+1e-10 for u in updates)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_complex_arbitrary_tie_selection_matches_dense_tangent_projection(direction):
    from collections import Counter
    from pyqed._letta_two_site_opt.reduced_solver import (
        _active_source_indices, _pair_vector_from_sources)
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'),
        neighborhoods=((0, 2), (1, 0), (2, 1)), real=False, seed=31)
    options = controls()
    selection = select_reduced_cbe(s, p.su2_mpo(), 1, options)
    pair, scaffold = selection.problem, selection.scaffold
    root = ReducedPairMetricRoot(pair, scaffold, options.metric_tolerance)
    le, re = (pair.frontier.site_embedding(scaffold, i) for i in (1, 2))
    a, b = le.pack_source(scaffold.tensors[1]), re.pack_source(scaffold.tensors[2])
    old = Counter(s.bond_sectors[1])
    li, ri = _active_source_indices(le, old, 'left'), _active_source_indices(re, old, 'right')
    columns = []
    for idx in li:
        da = np.zeros_like(a); da[idx] = 1.
        columns.append(root.apply(_pair_vector_from_sources(pair.layout, le, re, da, b)))
    for idx in ri:
        db = np.zeros_like(b); db[idx] = 1.
        columns.append(root.apply(_pair_vector_from_sources(pair.layout, le, re, a, db)))
    jac = np.column_stack(columns)
    x = pair.old_vector; hx, nx = pair.apply_hamiltonian(x), pair.apply_metric(x)
    e = np.vdot(x, hx).real/np.vdot(x, nx).real
    g = root.unwhiten_adjoint(hx-e*nx)
    missing = g-jac@np.linalg.lstsq(jac, g, rcond=1e-12)[0]
    assert selection.missing_norm == pytest.approx(np.linalg.norm(missing), abs=2e-9)
    assert selection.tangent_overlap < 2e-8
    expanded = expand_reduced_cbe(s, selection, direction)
    np.testing.assert_allclose(expanded.state_vector(), s.state_vector(), atol=2e-12)


def test_failed_ordinary_baseline_preserves_state_and_is_not_converged(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_solver as one
    s, h = incomplete()
    vector, bonds = s.state_vector(), s.bond_sectors
    def fail(candidate, *args, **kwargs):
        for a in candidate.tensors[0].values(): a.fill(np.nan)
        candidate.bond_sectors = ()
        raise FloatingPointError('injected ordinary failure')
    monkeypatch.setattr(one, 'optimize_reduced_site', fail)
    update = reduced_cbe_site(s, h, 0, 'lr', 3, controls())
    assert not update.accepted and not update.local_converged
    assert update.cbe_recovery_rejected
    assert 'ordinary failure' in update.cbe_recovery_reason
    assert s.bond_sectors == bonds
    np.testing.assert_array_equal(s.state_vector(), vector)


def test_four_site_tied_hubbard_cbe_reaches_independent_fci():
    from pyscf import fci
    n = 4
    h1 = -np.eye(n, k=1)-np.eye(n, k=-1)
    eri = np.zeros((n,)*4)
    for i in range(n): eri[i, i, i, i] = 4.
    p = ElectronicProblem(h1, eri, (2, 2))
    s = ReducedLatticeLETTA.random((1, n), symmetry=p.symmetry('su2'), seed=29)
    result = letta_dmrg(p.su2_mpo(), state=s, options=replace(controls(), max_sweeps=6))
    exact = fci.direct_spin1.kernel(h1, eri, n, p.nelec)[0]
    assert result.energy == pytest.approx(exact, abs=2e-9)
    assert result.state.symmetry_violation() == 0.
    assert not any(u.cbe_recovery_reason for row in result.history for u in row.updates)
    assert any(u.cbe_expansion_dimension for row in result.history for u in row.updates)


def test_four_site_heisenberg_cbe_matches_independent_spin_operator():
    from pyqed._letta_one_site_opt import ReducedPhysicalBasis, ReducedSymmetry, su2_heisenberg_mpo
    from test_letta_reduced_one_site import _heisenberg_dense
    basis = ReducedPhysicalBasis.spin_half()
    s = ReducedLatticeLETTA.random((1, 4),
        symmetry=ReducedSymmetry.su2(basis, target_two_j=0), seed=76)
    h = su2_heisenberg_mpo(4, physical_basis=basis)
    result = letta_dmrg(h, state=s, options=controls())
    assert result.energy == pytest.approx(np.linalg.eigvalsh(_heisenberg_dense(4))[0], abs=2e-10)
    assert not any(u.cbe_recovery_reason for row in result.history for u in row.updates)


def test_metric_inverse_respects_scaled_and_rank_deficient_boundary_support():
    p = ElectronicProblem(*integrals(4), (2, 2))
    s = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        neighborhoods=tuple((i,) for i in range(4)), multiplets_per_sector=2,
        real=False, seed=11)
    # Every first-cut multiplicity block has rank one but two stored columns.
    # Opposite diagonal gauges make those columns differ by 16 orders.
    for key, block in s.tensors[0].items():
        scale = np.geomspace(1e-8, 1e8, block.shape[-1])
        s.tensors[0][key] = block*scale
    for key, block in s.tensors[1].items():
        scale = np.geomspace(1e-8, 1e8, block.shape[0])
        s.tensors[1][key] = block/scale[:, None, None]
    pair = reduced_pair_problem(s, p.su2_mpo(), 1, matrix_free=True, dense_solver_threshold=0)
    root = ReducedPairMetricRoot(pair, s, 1e-12)
    assert root.size < pair.local_dimension
    rng = np.random.default_rng(55)
    y = rng.normal(size=root.size)+1j*rng.normal(size=root.size)
    np.testing.assert_allclose(root.apply(root.unwhiten(y)), y, atol=2e-10)
    nx = root.adjoint(y)
    error = pair.apply_metric(root.inverse_action(nx))-nx
    assert np.linalg.norm(error) <= 2e-10*np.linalg.norm(nx)


def test_unresolved_projection_returns_baseline_and_continues_next_step(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_cbe as cbe
    s, h = incomplete()
    original = cbe.lsmr
    def limited(*args, **kwargs):
        result = list(original(*args, **kwargs))
        result[1] = 7
        return tuple(result)
    monkeypatch.setattr(cbe, 'lsmr', limited)
    failed = reduced_cbe_site(s, h, 0, 'lr', 3, controls())
    assert failed.cbe_fallback and not failed.local_converged
    assert 'LSMR 7' in failed.cbe_recovery_reason
    monkeypatch.setattr(cbe, 'lsmr', original)
    next_step = reduced_cbe_site(s, h, 0, 'lr', 3, controls())
    assert next_step.cbe_expansion_dimension == 1
    assert next_step.energy < failed.energy-1e-4
    assert next_step.cbe_recovery_reason is None

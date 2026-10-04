"""Reduced local energy refinement, convergence and rollback regressions."""
from dataclasses import replace

import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_solver import _energy, optimize_reduced_site
from pyqed._letta_one_site_opt.reduced_updates import refine_reduced_pair_energy
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from test_letta_qchem import integrals


def case():
    p = ElectronicProblem(*integrals(3), (2, 1))
    s = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2'), seed=193)
    return s, p.su2_mpo()


def test_pair_energy_refinement_is_variational_native_and_budgeted(monkeypatch):
    s, h = case()
    initial = s.state_vector()
    before = _energy(s, h, stable=True)
    def forbidden(*a, **k):
        raise AssertionError('active refinement expanded a global state')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    result = refine_reduced_pair_energy(s, h, 1, LETTADMROptions(),
        max_iterations=3, tolerance=1e-12)
    assert result.energy < before-1e-5
    assert 1 <= result.iterations <= 3
    assert result.accepted_substeps > 0
    assert result.state.bond_sectors == s.bond_sectors
    assert result.state.symmetry_violation() == 0.
    assert all(b <= a+1e-10 for a, b in zip(result.energies, result.energies[1:]))
    assert result.energy == pytest.approx(_energy(result.state, h, stable=True), abs=1e-12)
    monkeypatch.undo()
    np.testing.assert_array_equal(s.state_vector(), initial)


@pytest.mark.parametrize('split', ['metric-als', 'metric-als-energy'])
def test_reduced_two_site_supports_metric_split_and_energy_refinement(split):
    s, h = case()
    result = letta_two_site_dmrg(h, state=s, bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=1, split_method=split,
            energy_refinement_max_iterations=2))
    assert result.energy <= _energy(s, h, stable=True)+1e-10
    if split == 'metric-als-energy':
        for u in result.history[0].updates:
            assert 0 < u.energy_refinement_iterations <= 2
            assert u.energy_refinement_energy <= u.energy_refinement_initial_energy+1e-10
            assert u.energy_refinement_diagnostics is not None


def test_failed_pair_restores_allocations_and_falls_back_to_one_site(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as pair
    s, h = case()
    options = LETTATwoSiteOptions(reduced_sector_growth=True, split_method='metric-als-energy')
    expected = s.copy()
    baseline = optimize_reduced_site(expected, h, 1, LETTADMROptions())
    old_bonds = s.bond_sectors
    def fail(candidate, *a, **k):
        for block in candidate.tensors[1].values():
            block.fill(np.nan)
        candidate.bond_sectors = ()
        raise FloatingPointError('injected nonfinite split')
    monkeypatch.setattr(pair, '_optimize_allocated_reduced_pair', fail)
    update = pair._optimize_reduced_pair(s, h, 1, 'lr', 3, options)
    assert update.fallback
    assert 'injected nonfinite split' in update.recovery_reason
    assert s.bond_sectors == old_bonds
    assert s.symmetry_violation() == 0.
    assert update.energy == pytest.approx(baseline.energy, abs=1e-11)
    np.testing.assert_allclose(s.state_vector(), expected.state_vector(), atol=1e-10)


def test_one_site_normalization_failure_is_transactional(monkeypatch):
    s, h = case()
    original = s.state_vector()
    original_bonds = s.bond_sectors
    def fail(state, **kw):
        for a in state.tensors[0].values():
            a.fill(np.nan)
        raise FloatingPointError('injected normalization failure')
    monkeypatch.setattr(ReducedLatticeLETTA, 'normalize', fail)
    with pytest.raises(FloatingPointError, match='normalization'):
        optimize_reduced_site(s, h, 1, LETTADMROptions())
    assert s.bond_sectors == original_bonds
    np.testing.assert_array_equal(s.state_vector(), original)


def test_rejected_one_site_sweep_is_not_converged(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_solver as one
    s, h = case()
    original = one.optimize_reduced_site
    def reject(state, hamiltonian, site, options, **kw):
        # Obtain valid diagnostics from a real local problem; leave the state
        # unchanged to simulate a rejected numerical candidate.
        update = original(state.copy(), hamiltonian, site, options)
        return replace(update, accepted=False, energy=_energy(state, hamiltonian, stable=True))
    monkeypatch.setattr(one, 'optimize_reduced_site', reject)
    result = letta_dmrg(h, state=s, options=LETTADMROptions(max_sweeps=2, tolerance=1.))
    assert not result.converged
    assert result.sweeps == 2


def test_unresolved_local_eigenproblem_does_not_claim_convergence(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_solver as one
    s, h = case()
    def incomplete(problem, options, *, initial_vector):
        # An iteration-limited solve can return the incumbent with no decrease.
        n = np.vdot(initial_vector, problem.apply_metric(initial_vector)).real
        v = initial_vector/np.sqrt(n)
        energy = np.vdot(v, problem.apply_hamiltonian(v)).real
        residual = np.linalg.norm(problem.apply_hamiltonian(v)-energy*problem.apply_metric(v))
        return energy, v, 1, residual
    monkeypatch.setattr(one, '_solve_local_problem', incomplete)
    result = letta_dmrg(h, state=s, options=LETTADMROptions(max_sweeps=2, tolerance=1.))
    assert not result.converged
    assert any(not u.local_converged for sweep in result.history for u in sweep.updates)
    assert result.energy == pytest.approx(_energy(s, h, stable=True), abs=1e-10)


def test_failed_fallback_also_preserves_the_incumbent(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as pair
    import pyqed._letta_one_site_opt.reduced_solver as one
    s, h = case()
    initial = s.state_vector()
    bonds = s.bond_sectors
    def fail(candidate, *args, **kwargs):
        for a in candidate.tensors[0].values():
            a.fill(np.nan)
        raise FloatingPointError('injected numerical failure')
    monkeypatch.setattr(pair, '_optimize_allocated_reduced_pair', fail)
    monkeypatch.setattr(one, 'optimize_reduced_site', fail)
    update = pair._optimize_reduced_pair(s, h, 1, 'lr', 3,
        LETTATwoSiteOptions(reduced_sector_growth=True))
    assert update.fallback and not update.accepted
    assert 'one-site fallback failed' in update.recovery_reason
    assert s.bond_sectors == bonds
    np.testing.assert_array_equal(s.state_vector(), initial)


def test_two_site_default_is_energy_refined_and_never_uses_global_expansion(monkeypatch):
    s, h = case()
    def forbidden(*args, **kwargs):
        raise AssertionError('active solver reconstructed a global wavefunction')
    monkeypatch.setattr(ReducedLatticeLETTA, 'state_vector', forbidden)
    result = letta_two_site_dmrg(h, state=s, bond_dim=3,
        options=LETTATwoSiteOptions(max_sweeps=1, energy_refinement_max_iterations=2))
    assert result.energy <= _energy(s, h, stable=True)+1e-10
    assert all(u.energy_refinement_iterations > 0 for u in result.history[0].updates)


@pytest.mark.parametrize('numerator,denominator', [(0., 0.), (1., -1.), (np.nan, 1.), (1j, 1.), (1., 1+1j)])
def test_invalid_reduced_energy_triggers_recovery_instead_of_nan(numerator, denominator):
    from pyqed._letta_one_site_opt.reduced_solver import _checked_energy
    with pytest.raises(FloatingPointError):
        _checked_energy(numerator, denominator)


def test_ordinary_one_site_gain_is_preserved_when_pair_candidate_stalls(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as pair
    from pyqed._letta_one_site_opt.reduced_updates import one_site_options
    s, h = case()
    options = LETTATwoSiteOptions(split_method='metric-als', energy_refinement_max_iterations=2)
    cap = len(s.bond_sectors[1])
    actual = pair._optimize_allocated_reduced_pair(s.copy(), h, 1, 'lr', cap, options)
    before = _energy(s, h, stable=True)
    def stationary(candidate, *args):
        return replace(actual, energy=before, accepted=True)
    monkeypatch.setattr(pair, '_optimize_allocated_reduced_pair', stationary)
    baseline = s.copy()
    ordinary = optimize_reduced_site(baseline, h, 1, one_site_options(options))
    update = pair._optimize_reduced_pair(s, h, 1, 'lr', cap, options)
    assert update.baseline_selected and not update.fallback
    assert update.energy == pytest.approx(ordinary.energy, abs=1e-12)
    assert update.energy < before-1e-5
    np.testing.assert_allclose(s.state_vector(), baseline.state_vector(), atol=1e-12)

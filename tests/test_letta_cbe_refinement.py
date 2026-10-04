"""CBE's norm trim must be followed by fixed-rank energy minimization."""
import numpy as np
from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state


def test_cbe_avoids_bose_hubbard_trim_plateau():
    model = build_model('bose_hubbard', '2d', (2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    before = initial.state_vector().copy()
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        max_sweeps=50, tolerance=1e-9, metric_tolerance=1e-10,
        cbe_enabled=True, cbe_selector='shrewd'))
    # Unrefined CBE gives -11.69634515 after 50 passes; the same ansatz
    # has a stable variational minimum near -11.697094.
    assert result.energy < -11.6970
    vector = result.state.state_vector()
    reference = np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(result.energy, reference, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(initial.state_vector(), before)
    assert result.state.bond_dimensions == initial.bond_dimensions
    assert max(np.diff([initial.expectation(model.mpo)] + [s.energy for s in result.history])) < 2e-9


import inspect
import pytest
from pyqed._letta_one_site_opt import cbe, solver


@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('selector', ['exact', 'shrewd'])
def test_refined_update_has_dense_energy_and_counts_all_solves(direction, selector, monkeypatch):
    model = build_model('ising', '2d', (2, 2))
    initial = make_shared_initial_state(model, bond_dim=1, seed=732).letta
    original = solver._update_from_cached_environments
    applications = []
    def update(*args, **kwargs):
        result = original(*args, **kwargs)
        applications.append(result.hamiltonian_applications)
        return result
    monkeypatch.setattr(solver, '_update_from_cached_environments', update)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        max_sweeps=2, start_direction=direction, cbe_enabled=True,
        cbe_selector=selector, metric_tolerance=1e-10))
    updates = [u for s in result.history for u in s.updates]
    coupled = sum(u.cbe_coupled_hamiltonian_applications for u in updates)
    assert coupled == 2 * sum(u.cbe_coupled_iterations for u in updates)
    assert sum(u.hamiltonian_applications for u in updates) == sum(applications) + coupled
    refined = [u for u in updates if u.cbe_refined_energy is not None]
    assert refined
    for u in refined:
        assert u.cbe_refined_energy <= u.cbe_trimmed_energy + 1e-10
        assert u.cbe_incumbent_refined_energy <= u.cbe_old_energy + 1e-10
        assert u.cbe_energy_refinement_start in ('trim', 'incumbent')
        assert u.cbe_energy_refinement_iterations > 0
        assert u.cbe_timings['energy_refinement'] > 0
        if not u.cbe_fallback:
            assert u.energy <= min(u.cbe_refined_energy, u.cbe_incumbent_refined_energy) + 1e-10
    v = result.state.state_vector()
    np.testing.assert_allclose(result.energy, np.vdot(v, model.mpo.to_dense() @ v) / np.vdot(v, v), atol=2e-10, rtol=0)
    assert result.state.bond_dimensions == initial.bond_dimensions


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_energy_refinement_restores_state_even_on_solve_failure(direction, monkeypatch):
    model = build_model('fermi_hubbard', '1d', 4)
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    original = cbe._strict_shrewd_cbe_bond_update
    signature = inspect.signature(original)
    checked = []
    def update(*args, **kwargs):
        context = signature.bind(*args, **kwargs).arguments
        if not checked:
            state, layout = context['state'], context['layout']
            i = layout.left_site
            before = [t.copy() for t in state.tensors]
            inputs = [context[k] for k in (
                'state', 'layout', 'hamiltonian_cache', 'metric_cache',
                'hamiltonian_left', 'hamiltonian_right', 'metric_left', 'metric_right',
                'direction', 'options')]
            refined = cbe._refine_cbe_pair_energy(*inputs, before[i], before[i+1])
            for a, b in zip(state.tensors, before):
                np.testing.assert_array_equal(a, b)
            try:
                state.tensors[i], state.tensors[i+1] = refined.left_tensor, refined.right_tensor
                v = state.state_vector()
                np.testing.assert_allclose(refined.energy, np.vdot(v, model.mpo.to_dense() @ v) / np.vdot(v, v), atol=2e-10, rtol=0)
                np.testing.assert_allclose(refined.norm, np.linalg.norm(v), atol=1e-10, rtol=0)
            finally:
                state.tensors[:] = before
            def fail(*args, **kwargs):
                raise RuntimeError('test solve failure')
            with monkeypatch.context() as m:
                m.setattr(solver, '_update_from_cached_environments', fail)
                with pytest.raises(RuntimeError, match='test solve failure'):
                    cbe._refine_cbe_pair_energy(*inputs, before[i], before[i+1])
            for a, b in zip(state.tensors, before):
                np.testing.assert_array_equal(a, b)
            checked.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(cbe, '_strict_shrewd_cbe_bond_update', update)
    letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        max_sweeps=1, start_direction=direction, cbe_enabled=True, cbe_selector='shrewd'))
    assert checked


def test_alternating_refinement_respects_its_original_conditioning_budget(monkeypatch):
    model = build_model('bose_hubbard', '1d', 6)
    initial = make_shared_initial_state(model, bond_dim=4, seed=1735).letta
    original = cbe._refine_cbe_pair_energy
    observations = []

    def checked(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs).arguments
        state, layout = bound['state'], bound['layout']
        left, right = bound['left_tensor'], bound['right_tensor']
        pair = state.tensors[layout.left_site:layout.left_site + 2]
        try:
            state.tensors[layout.left_site:layout.left_site + 2] = [left, right]
            # Physical validation only; this six-site reference is small.
            norm = np.linalg.norm(state.state_vector())
        finally:
            state.tensors[layout.left_site:layout.left_site + 2] = pair
        budget = 100 * np.sqrt(np.linalg.norm(left) * np.linalg.norm(right) / norm)
        result = original(*args, **kwargs)
        actual = np.sqrt(np.linalg.norm(result.left_tensor)
                         * np.linalg.norm(result.right_tensor) / result.norm)
        assert actual <= budget * (1 + 1e-12)
        observations.append(actual)
        return result

    monkeypatch.setattr(cbe, '_refine_cbe_pair_energy', checked)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        max_sweeps=2, cbe_enabled=True, cbe_selector='shrewd'))
    assert observations
    assert result.energy < -16.26

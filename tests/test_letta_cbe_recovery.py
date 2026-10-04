"""A failed CBE trial must roll back before the ordinary site solve."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt import cbe, solver
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state


@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('failure', ['exception', 'zero_norm', 'energy_increase', 'gauge'])
def test_failed_trial_restores_state_uses_one_site_then_resumes_cbe(monkeypatch, direction, failure):
    model = build_model('heisenberg', '2d', (3, 3))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    attempted, restored = [], []
    original_bond = cbe._cbe_bond_update
    original_update = solver._update_from_cached_environments
    original_gauge = solver._shift_gauge_and_extend_metric
    pending = {}

    def bond(state, layout, *args, **kwargs):
        attempted.append(layout.left_site)
        if len(attempted) == 1:
            pending['before'] = [a.copy() for a in state.tensors]
            pending['restore_expected'] = True
            if failure == 'exception':
                state.tensors[layout.left_site].fill(np.nan)
                raise np.linalg.LinAlgError('SVD did not converge')
            # Supply a real update record, then inject a numerical failure.
            update = original_update(state, layout.left_site if direction == 'lr'
                                     else layout.left_site + 1,
                                     *fresh_local_environments(state, model.mpo, layout, direction),
                                     args[-1])
            if failure == 'zero_norm':
                state.tensors[layout.left_site].fill(0.)
            elif failure == 'energy_increase':
                # All-up product state has strictly higher Heisenberg energy.
                for a in state.tensors:
                    a.fill(0.)
                    a.flat[0] = 1.
            return update
        return original_bond(state, layout, *args, **kwargs)

    def update(state, *args, **kwargs):
        if pending.pop('restore_expected', False):
            for actual, expected in zip(state.tensors, pending['before']):
                np.testing.assert_array_equal(actual, expected)
            restored.append(True)
        return original_update(state, *args, **kwargs)

    def gauge(state, *args, **kwargs):
        if failure == 'gauge' and pending.get('restore_expected'):
            state.tensors[0].fill(np.nan)
            raise FloatingPointError('nonfinite frontier gauge transformation')
        return original_gauge(state, *args, **kwargs)

    monkeypatch.setattr(cbe, '_cbe_bond_update', bond)
    monkeypatch.setattr(solver, '_update_from_cached_environments', update)
    monkeypatch.setattr(solver, '_shift_gauge_and_extend_metric', gauge)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        cbe_enabled=True, cbe_selector='shrewd', start_direction=direction,
        max_sweeps=1, cbe_energy_refinement_max_iterations=2))
    assert restored == [True]
    assert len(attempted) == initial.nsites - 1
    recovered = result.history[0].updates[0]
    assert recovered.cbe_fallback and recovered.cbe_recovery_reason
    assert result.energy <= initial.expectation(model.mpo) + 1e-10
    vector = result.state.state_vector()
    physical = (np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)).real
    np.testing.assert_allclose(result.energy, physical, atol=1e-9, rtol=0)


def fresh_local_environments(state, mpo, layout, direction):
    from pyqed._letta_two_site_opt import LETTAPairEnvironmentCache, IdentityPairEnvironmentCache
    h, n = LETTAPairEnvironmentCache(state, mpo), IdentityPairEnvironmentCache(state)
    site = layout.left_site if direction == 'lr' else layout.left_site + 1
    return (h, n, h.build_left_environments()[site], h.build_right_environments()[site + 1],
            n.build_left_environments()[site], n.build_right_environments()[site + 1])


def test_programming_errors_are_not_hidden(monkeypatch):
    model = build_model('heisenberg', '2d', (3, 3))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    def broken(*args, **kwargs):
        raise ValueError('incompatible tensor shape')
    monkeypatch.setattr(cbe, '_cbe_bond_update', broken)
    with pytest.raises(ValueError, match='incompatible tensor shape'):
        letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(cbe_enabled=True, max_sweeps=1))


def test_failed_one_site_retry_keeps_last_state_and_does_not_claim_convergence(monkeypatch):
    model = build_model('heisenberg', '2d', (3, 3))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    snapshots = []
    def broken(state, *args, **kwargs):
        if not snapshots:
            snapshots.extend(a.copy() for a in state.tensors)
        state.tensors[0].fill(np.nan)
        raise FloatingPointError('nonfinite local solve')
    monkeypatch.setattr(cbe, '_cbe_bond_update', broken)
    monkeypatch.setattr(solver, '_update_from_cached_environments', broken)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        cbe_enabled=True, max_sweeps=2))
    assert not result.converged and result.sweeps == 2
    assert all(u.cbe_recovery_rejected and not u.accepted
               for sweep in result.history for u in sweep.updates)
    for actual, expected in zip(result.state.tensors, snapshots):
        np.testing.assert_array_equal(actual, expected)


def test_unstable_gauge_preserves_valid_one_site_update(monkeypatch):
    model = build_model('heisenberg', '2d', (3, 3))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    def broken_gauge(state, *args, **kwargs):
        state.tensors[0].fill(np.nan)
        raise FloatingPointError('nonfinite frontier gauge transformation')
    monkeypatch.setattr(solver, '_shift_gauge_and_extend_metric', broken_gauge)
    result = letta_dmrg(model.mpo, state=initial, options=LETTADMROptions(
        cbe_enabled=True, cbe_selector='shrewd', max_sweeps=1,
        cbe_energy_refinement_max_iterations=2))
    assert result.energy < initial.expectation(model.mpo)
    assert all('skipped unstable gauge' in u.cbe_recovery_reason
               and not u.cbe_recovery_rejected for u in result.history[0].updates)

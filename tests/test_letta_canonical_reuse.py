"""Canonical norm reuse and nearest-tied sweep invariants."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    IdentityEnvironmentCache, LatticeLETTA, LETTADMROptions,
    canonicalize_frontier, letta_dmrg,
)
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo


def test_canonical_environments_survive_new_cache_and_unchanged_center(monkeypatch):
    state = LatticeLETTA.random((1, 6), bond_dim=3, seed=19, real=False)
    canonicalize_frontier(state, 2)
    first = IdentityEnvironmentCache(state)
    left = first.build_left_environments()[2]
    right = first.build_right_environments()[3]
    assert ('lr', 2) in first.canonical_reports
    assert ('rl', 3) in first.canonical_reports
    state.tensors[2] = state.tensors[2] * 1.001
    second = IdentityEnvironmentCache(state)
    assert second.build_left_environments()[2] is left
    assert second.build_right_environments()[3] is right
    # The two diagonal frontiers must suffice; no D^4 local Gram is built.
    monkeypatch.setattr(second, '_reduced_metric', lambda *args: pytest.fail('built dense norm blocks'))
    metric = second.effective_metric(left, right, 2)
    frame = state.local_frame(2)
    np.testing.assert_allclose(metric.to_dense(), frame.conj().T @ frame, atol=2e-12)


def test_in_place_changes_invalidate_only_affected_side():
    state = LatticeLETTA.random((1, 6), bond_dim=2, seed=28)
    canonicalize_frontier(state, 2)
    cache = IdentityEnvironmentCache(state)
    left, right = cache.build_left_environments()[2], cache.build_right_environments()[3]
    state.tensors[0][0, 0, 0, 0] += .2
    fresh = IdentityEnvironmentCache(state)
    assert fresh.build_left_environments()[2] is not left
    assert fresh.build_right_environments()[3] is right
    assert ('lr', 2) not in fresh.canonical_reports


@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('dimension', [2, 4, 8])
def test_nearest_tied_uses_supported_identity_at_every_update(monkeypatch, direction, dimension):
    original = IdentityEnvironmentCache.effective_metric
    observations = []
    def checked(cache, left, right, site):
        metric = original(cache, left, right, site)
        coords = metric.coordinate_whitening(1e-12)
        assert coords is not None
        np.testing.assert_allclose(coords[1], 1., atol=1e-11)
        # No dense whitening or norm blocks are stored in the compact metric.
        assert hasattr(metric, 'diagonal')
        frame = cache.state.local_frame(site)
        np.testing.assert_allclose(metric.to_dense(), frame.conj().T @ frame,
                                   rtol=2e-10, atol=2e-11)
        observations.append(site)
        return metric
    monkeypatch.setattr(IdentityEnvironmentCache, 'effective_metric', checked)
    mpo = transverse_field_ising_mpo((1, 6), field=.9)
    result = letta_dmrg(mpo, lattice_shape=(1, 6), bond_dim=dimension, seed=73,
                       options=LETTADMROptions(gauge_mode='frontier', max_sweeps=2,
                                              start_direction=direction, dense_solver_threshold=1))
    assert len(observations) == 6 * result.sweeps
    np.testing.assert_allclose(result.energy, result.state.expectation(mpo), atol=2e-10)


def test_large_bond_gauge_moves_preserve_physical_state_and_do_not_amplify_null_rows(monkeypatch):
    from pyqed._letta_one_site_opt import gauge
    original = gauge.shift_frontier_gauge
    moves = []
    def checked(state, *args, **kwargs):
        before = state.state_vector()
        result = original(state, *args, **kwargs)
        np.testing.assert_allclose(state.state_vector(), before, rtol=2e-10, atol=2e-11)
        assert max(np.linalg.norm(a) for a in state.tensors) < 1e8
        moves.append(1)
        return result
    monkeypatch.setattr(gauge, 'shift_frontier_gauge', checked)
    h = transverse_field_ising_mpo((1, 8), field=.9)
    result = letta_dmrg(h, lattice_shape=(1, 8), bond_dim=16, seed=73,
                       options=LETTADMROptions(gauge_mode='frontier', max_sweeps=2,
                                              dense_solver_threshold=1))
    assert len(moves) >= 15
    assert result.energy < -9.26415764
    assert all(u.metric_kind in {'identity', 'supported_identity'}
               for sweep in result.history for u in sweep.updates)


def test_solver_copy_preserves_certificates_and_does_not_modify_input():
    state = LatticeLETTA.random((1, 6), bond_dim=3, seed=82)
    canonicalize_frontier(state, 0)
    arrays = [a.copy() for a in state.tensors]
    initial = dict(state._canonical_norm_environments)
    result = letta_dmrg(transverse_field_ising_mpo((1, 6)), state=state,
                       options=LETTADMROptions(gauge_mode='frontier', max_sweeps=1))
    assert result.canonical_environment_reuses >= 5
    assert result.canonical_metric_hits == 6
    for a, b in zip(arrays, state.tensors):
        np.testing.assert_array_equal(a, b)
    assert all(state._canonical_norm_environments[k] is v for k, v in initial.items())


@pytest.mark.parametrize('granularity, alternate', [('site', False), ('column', True)])
def test_column_and_same_direction_sweeps_reuse_all_nearest_tied_metrics(granularity, alternate):
    result = letta_dmrg(transverse_field_ising_mpo((1, 6)), lattice_shape=(1, 6),
                       bond_dim=4, seed=93, real=False,
                       options=LETTADMROptions(gauge_mode='frontier', max_sweeps=2,
                                              environment_granularity=granularity, alternate=alternate))
    assert result.canonical_environment_reuses >= 5
    assert result.canonical_metric_hits == 6 * result.sweeps
    assert all(u.metric_kind in {'identity', 'supported_identity'}
               for sweep in result.history for u in sweep.updates)

"""Sweep gauges recover physical states and invalidate partially changed caches."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt import reduced_gauge as gauge
from pyqed._letta_one_site_opt import reduced_solver as one
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_compression import MetricCompressionOptions
from test_letta_reduced_updates import case


@pytest.mark.parametrize('initial', [False, True])
def test_partial_gauge_failure_restores_tensors_and_sector_metadata(monkeypatch, initial):
    state, h = case()
    before = state.state_vector()
    bonds = state.bond_sectors
    def fail(candidate, *args, **kwargs):
        next(iter(candidate.tensors[0].values())).fill(np.nan)
        candidate.bond_sectors = ()
        raise FloatingPointError('partial gauge write')
    monkeypatch.setattr(gauge, 'canonicalize_reduced_frontier' if initial else
                        'shift_reduced_frontier_gauge', fail)
    reason = gauge.condition_reduced_sweep(state, h, LETTADMROptions(),
        **({'center': 0} if initial else {'cut': 1}))
    assert 'partial gauge write' in reason
    assert state.bond_sectors == bonds
    np.testing.assert_array_equal(state.state_vector(), before)


def test_silent_energy_changing_gauge_is_rejected(monkeypatch):
    state, h = case()
    before = state.state_vector()
    energy = one._energy(state, h, stable=True)
    def corrupt(candidate, *args, **kwargs):
        next(iter(candidate.tensors[1].values()))[...] *= 3.
        assert abs(one._energy(candidate, h, stable=True)-energy) > 1e-5
    monkeypatch.setattr(gauge, 'shift_reduced_frontier_gauge', corrupt)
    reason = gauge.condition_reduced_sweep(state, h, LETTADMROptions(), cut=1)
    assert 'changed the physical energy' in reason
    np.testing.assert_array_equal(state.state_vector(), before)


def run(state, h, method, direction):
    common = dict(max_sweeps=1, tolerance=1e3, start_direction=direction,
        compression=MetricCompressionOptions(als_max_iterations=6, lsmr_max_iterations=60))
    if method == 'two-site':
        return letta_two_site_dmrg(h, state=state, bond_dim=3,
            options=LETTATwoSiteOptions(**common, energy_refinement_max_iterations=2))
    return letta_dmrg(h, state=state, options=LETTADMROptions(**common,
        cbe_enabled=method == 'cbe', cbe_refinement_max_iterations=3,
        cbe_energy_refinement_max_iterations=2))


@pytest.mark.parametrize('method', ['one-site', 'cbe', 'two-site'])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
@pytest.mark.parametrize('phase', ['initial', 'shift'])
def test_failed_gauge_matches_skipping_that_gauge_and_continues(monkeypatch, method, direction, phase):
    state, h = case()
    before = state.state_vector()
    initial_energy = one._energy(state, h, stable=True)
    canonicalize = gauge.canonicalize_reduced_frontier
    shift = gauge.shift_reduced_frontier_gauge
    def execute(inject):
        flags = dict(initial_done=False, skipped=False)
        def selected(candidate):
            flags['skipped'] = True
            if inject:
                next(iter(candidate.tensors[0].values())).fill(np.nan)
                candidate.bond_sectors = ()
                raise FloatingPointError('injected sweep gauge failure')
        def initial(candidate, *args, **kwargs):
            if phase == 'initial':
                return selected(candidate)
            result = canonicalize(candidate, *args, **kwargs)
            flags['initial_done'] = True
            return result
        def moving(candidate, *args, **kwargs):
            if phase == 'shift' and flags['initial_done'] and not flags['skipped']:
                return selected(candidate)
            return shift(candidate, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(gauge, 'canonicalize_reduced_frontier', initial)
            patch.setattr(gauge, 'shift_reduced_frontier_gauge', moving)
            result = run(state, h, method, direction)
        assert flags['skipped']
        return result
    reference = execute(False)
    result = execute(True)
    assert not result.converged
    updates = result.history[0].updates
    assert len(updates) == (2 if method == 'two-site' else 3)
    assert sum('injected sweep gauge failure' in (u.recovery_reason or '') for u in updates) == 1
    assert result.energy <= initial_energy+1e-9
    assert result.energy == pytest.approx(reference.energy, abs=2e-8)
    # State-level comparisons also check that a stale moving environment did
    # not steer later updates along a different numerical trajectory.
    v, w = result.state.state_vector(), reference.state.state_vector()
    overlap = abs(np.vdot(v, w))/np.linalg.norm(v)/np.linalg.norm(w)
    assert overlap == pytest.approx(1., abs=2e-8)
    assert result.state.symmetry_violation() == 0.
    assert result.energy == pytest.approx(one._energy(result.state, h, stable=True), abs=1e-10)
    np.testing.assert_array_equal(state.state_vector(), before)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_partly_invalidated_environment_is_rebuilt_after_gauge_failure(monkeypatch, direction):
    state, h = case()
    synchronize = one.ReducedSweepContext.synchronize
    failed = []
    def fail(context, indices):
        synchronize(context, indices)
        if len(indices) == 2 and not failed:
            failed.append(context)
            context.n_chain.left.clear()
            raise MemoryError('partly invalidated gauge cache')
        assert context not in failed
    monkeypatch.setattr(one.ReducedSweepContext, 'synchronize', fail)
    result = run(state, h, 'one-site', direction)
    assert len(failed) == 1
    assert not result.converged
    assert len(result.history[0].updates) == 3
    assert any('partly invalidated gauge cache' in (u.recovery_reason or '')
               for u in result.history[0].updates)
    assert result.energy == pytest.approx(one._energy(result.state, h, stable=True), abs=1e-10)
    assert result.state.symmetry_violation() == 0.

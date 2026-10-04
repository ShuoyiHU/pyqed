"""Resource failures use the same transactional guarantees as numerical errors."""
from dataclasses import replace

import numpy as np
import pytest

from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt import reduced_solver as one
from pyqed._letta_one_site_opt import reduced_ring_solver as ring
from pyqed._letta_one_site_opt.reduced_cbe import reduced_cbe_site
from pyqed._letta_two_site_opt import LETTATwoSiteOptions
from pyqed._letta_two_site_opt.reduced_solver import _optimize_reduced_pair
from test_letta_reduced_updates import case
from test_letta_reduced_cbe import incomplete, controls
from test_letta_ring_sweeps import molecular_state, direct_conditional_ring
from test_letta_qchem import integrals
from pyqed._letta_one_site_opt.qchem import ElectronicProblem


def corrupt(state):
    next(iter(state.tensors[0].values())).fill(np.nan)
    state.bond_sectors = ()
    raise MemoryError('injected allocation failure after partial write')


def test_open_local_allocation_failure_restores_before_propagating(monkeypatch):
    state, h = case()
    original, bonds = state.state_vector(), state.bond_sectors
    monkeypatch.setattr(one, '_optimize_reduced_site_impl', lambda s, *a, **k: corrupt(s))
    with pytest.raises(MemoryError, match='allocation failure'):
        one.optimize_reduced_site(state, h, 0, LETTADMROptions())
    assert state.bond_sectors == bonds
    np.testing.assert_array_equal(state.state_vector(), original)


def test_open_sweep_continues_after_failed_ordinary_update(monkeypatch):
    state, h = case()
    original = state.state_vector()
    implementation = one._optimize_reduced_site_impl
    calls = []
    def once(candidate, hamiltonian, site, options, **kwargs):
        calls.append(site)
        if len(calls) == 1:
            corrupt(candidate)
        return implementation(candidate, hamiltonian, site, options, **kwargs)
    monkeypatch.setattr(one, '_optimize_reduced_site_impl', once)
    result = letta_dmrg(h, state=state, options=LETTADMROptions(max_sweeps=1, tolerance=1e3))
    assert calls == [0, 1, 2]
    assert not result.converged
    assert not result.history[0].updates[0].accepted
    assert 'allocation failure' in result.history[0].updates[0].recovery_reason
    assert result.energy <= one._energy(state, h, stable=True)+1e-9
    assert result.state.symmetry_violation() == 0.
    np.testing.assert_array_equal(state.state_vector(), original)


@pytest.mark.parametrize('phase', ['update', 'gauge'])
def test_ring_sweep_recovers_partial_allocation_failure(monkeypatch, phase):
    state = molecular_state(copies=1)
    h = ElectronicProblem(*integrals(2), (1, 1)).su2_mpo()
    before = direct_conditional_ring(state)
    name = 'ring_local_problem' if phase == 'update' else 'ring_gauge_shift'
    implementation = getattr(ring, name)
    calls = []
    def once(candidate, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            corrupt(candidate)
        return implementation(candidate, *args, **kwargs)
    monkeypatch.setattr(ring, name, once)
    result = letta_dmrg(h, state=state, options=LETTADMROptions(max_sweeps=1, tolerance=1e3))
    assert len(calls) >= 3
    assert not result.converged
    assert 'allocation failure' in result.history[0].updates[0].recovery_reason
    assert result.energy <= ring.ring_energy(state, h)+1e-9
    np.testing.assert_array_equal(direct_conditional_ring(state), before)
    assert result.state.norm() == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize('method', ['cbe', 'two-site'])
def test_compression_allocation_failure_returns_ordinary_baseline(monkeypatch, method):
    state, h = incomplete()
    from pyqed._letta_two_site_opt import reduced_compression
    compressor = reduced_compression.compress_factors
    def fail(*args, **kwargs):
        raise MemoryError('injected compression allocation failure')
    monkeypatch.setattr(reduced_compression, 'compress_factors', fail)
    compression = MetricCompressionOptions(solver='variable-projection')
    baseline = state.copy()
    expected = one.optimize_reduced_site(baseline, h, 0, LETTADMROptions())
    if method == 'cbe':
        update = reduced_cbe_site(state, h, 0, 'lr', 3, replace(controls(), compression=compression))
        reason, fallback = update.cbe_recovery_reason, update.cbe_fallback
    else:
        update = _optimize_reduced_pair(state, h, 0, 'lr', 3, LETTATwoSiteOptions(
            compression=compression, reduced_sector_growth=True))
        reason, fallback = update.recovery_reason, update.fallback
    assert fallback and 'MemoryError' in reason
    assert update.energy == pytest.approx(expected.energy, abs=1e-12)
    assert state.bond_sectors == baseline.bond_sectors
    np.testing.assert_array_equal(state.state_vector(), baseline.state_vector())

    monkeypatch.setattr(reduced_compression, 'compress_factors', compressor)
    if method == 'cbe':
        retried = reduced_cbe_site(state, h, 0, 'lr', 3, controls())
        assert retried.cbe_expansion_dimension > 0
        assert retried.cbe_recovery_reason is None
    else:
        retried = _optimize_reduced_pair(state, h, 0, 'lr', 3,
            LETTATwoSiteOptions(reduced_sector_growth=True))
        assert retried.recovery_reason is None
    assert retried.energy < update.energy-1e-4

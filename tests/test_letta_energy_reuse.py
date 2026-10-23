"""Identical-value reuse must preserve physical acceptance and normalization."""
import numpy as np
import pytest
from test_letta_pair_coordinates import setup_pair
from pyqed._letta_one_site_opt import LETTADMROptions
from pyqed._letta_one_site_opt import reduced_solver as solver

@pytest.mark.parametrize('complex_values', [False, True])
def test_cached_local_updates_match_fresh_contractions(complex_values, monkeypatch):
    state, ham, _ = setup_pair()
    if complex_values:
        for blocks in state.tensors:
            for key, value in blocks.items():
                blocks[key] = value * (1 + .2j)
    reference, cached = state.copy(), state.copy()
    original = solver._energy
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, '_energy', counted)
    options = LETTADMROptions(gauge_mode='frontier')
    expected = solver.optimize_reduced_site(reference, ham, 1, options)
    fresh_count = len(calls)
    calls.clear()
    cache = solver.StableEnergyCache(ham)
    # A moving sweep already knows the incumbent energy. Reuse that value,
    # while the changed candidate still needs an independent contraction.
    cache(cached)
    calls.clear()
    actual = solver.optimize_reduced_site(cached, ham, 1, options, energy_cache=cache)
    assert len(calls) < fresh_count
    assert actual.energy == expected.energy
    assert actual.accepted == expected.accepted
    for a, b in zip(reference.tensors, cached.tensors):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert cache(cached.copy()) == original(cached, ham, stable=True)
    before = len(calls)
    key = next(iter(cached.tensors[1]))
    cached.tensors[1][key] *= 1.01
    assert cache(cached) == original(cached, ham, stable=True)
    assert len(calls) == before + 1

def test_cbe_update_matches_without_energy_reuse(monkeypatch):
    from test_letta_reduced_cbe import incomplete, controls
    from pyqed._letta_one_site_opt.reduced_cbe import reduced_cbe_site
    state, ham = incomplete()
    cached, fresh = state.copy(), state.copy()
    actual = reduced_cbe_site(cached, ham, 0, 'lr', 3, controls())
    monkeypatch.setattr(solver, 'StableEnergyCache',
        lambda h: lambda s, **kw: solver._energy(s, h, stable=True, **kw))
    expected = reduced_cbe_site(fresh, ham, 0, 'lr', 3, controls())
    for field in ('energy', 'accepted', 'cbe_baseline_selected', 'cbe_fallback',
                  'cbe_expansion_dimension'):
        assert getattr(actual, field) == getattr(expected, field)
    assert cached.bond_sectors == fresh.bond_sectors
    for a, b in zip(cached.tensors, fresh.tensors):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

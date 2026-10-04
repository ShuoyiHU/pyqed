"""Numerical equivalence and work-count checks for scoped contraction reuse."""
from unittest.mock import patch

import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.contractions import LETTAEnvironmentCache
from pyqed._letta_two_site_opt import (
    LETTAPairLayout, LETTAPairEnvironmentCache, LETTATwoSiteOptions, letta_two_site_dmrg,
)


@pytest.mark.parametrize("sparse", [False, True])
def test_prepared_local_action_reuses_channels_and_matches_dense(sparse, monkeypatch):
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=1660, real=False,
                                neighborhoods=((0, 3), (1, 0), (2, 1, 3), (3,)))
    model = build_model("heisenberg", dimension="2d", size=(2, 2))
    cache = LETTAEnvironmentCache(state, model.mpo, use_sparse_mpo=sparse)
    left, right = cache.build_left_environments()[1], cache.build_right_environments()[2]
    matrix = cache.effective_matrix(left, right, 1)
    action = cache.prepare_effective_action(left, right, 1)
    adjoint = cache.prepare_effective_action(left, right, 1, adjoint=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("fixed channel slices were rebuilt inside a matvec")
    with monkeypatch.context() as m:
        m.setattr(cache, "_select_channel", forbidden)
        rng = np.random.default_rng(1661)
        for _ in range(3):
            vector = rng.normal(size=matrix.shape[0]) + 1j * rng.normal(size=matrix.shape[0])
            np.testing.assert_allclose(action(vector), matrix @ vector, atol=2e-12)
            np.testing.assert_allclose(adjoint(vector), matrix.conj().T @ vector, atol=2e-12)
    # Preparing a new local solve after an in-place environment edit cannot
    # return a tensor-valued cache entry from the previous solve.
    left *= 1.7
    fresh = cache.prepare_effective_action(left, right, 1)
    np.testing.assert_allclose(fresh(vector), 1.7 * matrix @ vector, atol=3e-12)


@pytest.mark.parametrize("sparse", [False, True])
def test_prepared_pair_actions_match_fresh_actions_for_all_batch_shapes(sparse, monkeypatch):
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=1662, real=False)
    model = build_model("ising", dimension="2d", size=(2, 2))
    cache = LETTAPairEnvironmentCache(state, model.mpo, use_sparse_mpo=sparse)
    layout = LETTAPairLayout.from_state(state, 1)
    left, right = cache.build_left_environments()[1], cache.build_right_environments()[3]
    action = cache.prepare_pair_action(left, right, layout)
    rng = np.random.default_rng(1663)
    size = int(np.prod(layout.merged_shape))
    for batch in (None, 1, 3, 2, 5, 4, None):
        shape = (size,) if batch is None else (size, batch)
        vector = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        expected = cache.effective_pair_action(left, right, layout, vector)
        with monkeypatch.context() as m:
            def forbidden(*args, **kwargs):
                raise AssertionError("pair action rebuilt its channel slices")
            m.setattr(cache, "_select_channel", forbidden)
            np.testing.assert_allclose(action(vector), expected, atol=2e-11, rtol=2e-12)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_two_site_reuses_completed_environments_on_reverse_sweeps(direction):
    model = build_model("ising", dimension="2d", size=(2, 3))
    state = LatticeLETTA.random((2, 3), bond_dim=2, seed=1664)
    calls = []
    original_left = LETTAPairEnvironmentCache.build_left_environments
    original_right = LETTAPairEnvironmentCache.build_right_environments
    def left(cache):
        calls.append("left")
        return original_left(cache)
    def right(cache):
        calls.append("right")
        return original_right(cache)
    with patch.object(LETTAPairEnvironmentCache, "build_left_environments", left), \
         patch.object(LETTAPairEnvironmentCache, "build_right_environments", right):
        result = letta_two_site_dmrg(model.mpo, state=state, bond_dim=2,
            options=LETTATwoSiteOptions(max_sweeps=3, tolerance=1e-14, start_direction=direction))
    assert result.sweeps == 3
    assert calls == ["right" if direction == "lr" else "left"]
    np.testing.assert_allclose(result.energy, result.state.expectation(model.mpo), atol=2e-9, rtol=0)
    vector = result.state.state_vector()
    physical_energy = np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(result.energy, physical_energy, atol=2e-9, rtol=0)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_cbe_saved_baseline_environments_match_freshly_rebuilt_ones(direction):
    from pyqed._letta_one_site_opt import cbe
    model = build_model("ising", dimension="2d", size=(2, 3))
    state = LatticeLETTA.random((2, 3), bond_dim=2, seed=1665, real=False)
    options = LETTADMROptions(max_sweeps=2, tolerance=1e-14, start_direction=direction,
                              cbe_enabled=True, cbe_selector="shrewd")
    optimized = letta_dmrg(model.mpo, state=state, options=options)
    original = cbe._ordinary_bond_fallback
    passed = []
    def rebuild(*args, **kwargs):
        passed.append(kwargs.pop("local_environments", None))
        return original(*args, **kwargs)
    with patch.object(cbe, "_ordinary_bond_fallback", rebuild):
        fresh = letta_dmrg(model.mpo, state=state, options=options)
    assert passed and all(pair is not None for pair in passed)
    np.testing.assert_allclose([s.energy for s in optimized.history],
                               [s.energy for s in fresh.history], atol=2e-9, rtol=0)
    v, w = optimized.state.state_vector(), fresh.state.state_vector()
    assert abs(np.vdot(v, w)) ** 2 / (np.vdot(v, v).real * np.vdot(w, w).real) > 1 - 1e-9


@pytest.mark.parametrize('model,shape,seed,expected', [
    ('ising', (2, 3), 1700, (-8.403040345002827, -8.413113407000909, -8.416744300744687)),
    ('fermi_hubbard', (2, 2), 1703, (-9.840027293938487, -10.080144812862793, -10.09909876542333)),
])
def test_reuse_preserves_sensitive_2d_sweep_results(model, shape, seed, expected):
    # Saved before the reuse changes. These deliberately short trajectories
    # exposed changes from rounded baseline metrics and omitted QR preparation.
    # Keep CBE refinement disabled to retain the original reuse regression.
    # The two-site references include incumbent refinement and the overlap cutoff.
    h = build_model(model, dimension='2d', size=shape).mpo
    state = LatticeLETTA.random(shape, physical_dim=h.physical_dim, bond_dim=2, seed=seed)
    results = [
        letta_dmrg(h, state=state, options=LETTADMROptions(
            max_sweeps=3, tolerance=1e-14, cbe_enabled=cbe, cbe_selector='shrewd',
            cbe_energy_refinement_max_iterations=0))
        for cbe in (False, True)
    ]
    results.append(letta_two_site_dmrg(h, state=state, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=3, tolerance=1e-14)))
    np.testing.assert_allclose([r.energy for r in results], expected, atol=2e-8, rtol=0)
    for result in results:
        np.testing.assert_allclose(result.energy, result.state.expectation(h), atol=2e-9, rtol=0)
        vector = result.state.state_vector()
        physical_energy = np.vdot(vector, h.to_dense() @ vector) / np.vdot(vector, vector)
        np.testing.assert_allclose(result.energy, physical_energy, atol=2e-9, rtol=0)


def test_qr_pair_sweeps_contract_each_side_only_when_needed(monkeypatch):
    model = build_model('ising', dimension='2d', size=(2, 3))
    state = LatticeLETTA.random((2, 3), bond_dim=2, seed=1666)
    original = LETTAPairEnvironmentCache._sparse_extend
    transfers = []
    def extend(cache, boundary, site, direction):
        transfers.append((site, direction))
        return original(cache, boundary, site, direction)
    with monkeypatch.context() as m:
        m.setattr(LETTAPairEnvironmentCache, '_sparse_extend', extend)
        result = letta_two_site_dmrg(model.mpo, state=state, bond_dim=2,
            options=LETTATwoSiteOptions(max_sweeps=3, tolerance=1e-14, gauge_mode='qr'))
    assert result.sweeps == 3
    # One initial opposite-side build, then one outgoing contraction per site
    # per sweep. A rebuild on every reverse pass would require 36 transfers.
    assert len(transfers) == (1 + result.sweeps) * state.nsites == 24


def test_cbe_does_not_substitute_a_rounded_certificate_for_a_raw_metric():
    from pyqed._letta_one_site_opt import canonicalize_frontier
    from pyqed._letta_one_site_opt.cbe import _baseline_metric_environment
    from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache
    state = LatticeLETTA.random((1, 4), bond_dim=2, seed=1667)
    canonicalize_frontier(state, center=0)
    cache = IdentityPairEnvironmentCache(state)
    right = cache.build_right_environments()
    assert _baseline_metric_environment(cache, right[1], 1, 'rl') is None
    raw = cache.extend_right(right[2], 1)
    assert _baseline_metric_environment(cache, raw, 1, 'rl') is raw

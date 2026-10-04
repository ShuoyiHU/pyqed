"""Segment maps retain physical copy constraints and complex adjoints."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    LatticeLETTA, IdentityEnvironmentCache, LETTAEnvironmentCache,
)
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache, LETTAPairEnvironmentCache


def make_cache(kind, *, simple=False):
    ties = None if simple else ((0, 3), (1, 0), (2, 1, 3), (3,))
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=814, real=False, neighborhoods=ties)
    if kind in (IdentityEnvironmentCache, IdentityPairEnvironmentCache):
        return kind(state)
    return kind(state, build_model("ising", dimension="2d", size=(2, 2)).mpo)


@pytest.mark.parametrize("kind", [IdentityEnvironmentCache, LETTAEnvironmentCache,
                                  IdentityPairEnvironmentCache, LETTAPairEnvironmentCache])
@pytest.mark.parametrize("cuts", [(0, 4), (1, 3), (0, 2), (2, 4)])
def test_segment_matches_sequential_propagation_and_adjoint(kind, cuts):
    cache = make_cache(kind)
    start, stop = cuts
    transfer = cache.segment_transfer(start, stop)
    rng = np.random.default_rng(815)
    left = rng.normal(size=transfer.input_shape) + 1j * rng.normal(size=transfer.input_shape)
    right = rng.normal(size=transfer.output_shape) + 1j * rng.normal(size=transfer.output_shape)
    expected_left, expected_right = left, right
    for site in range(start, stop):
        expected_left = cache.extend_left(expected_left, site)
    for site in reversed(range(start, stop)):
        expected_right = cache.extend_right(expected_right, site)
    np.testing.assert_allclose(transfer.apply_left(left), expected_left, atol=2e-12)
    np.testing.assert_allclose(transfer.apply_right(right), expected_right, atol=2e-12)
    np.testing.assert_allclose(np.vdot(right.ravel(), transfer @ left.ravel()),
                               np.vdot(transfer.H @ right.ravel(), left.ravel()), atol=2e-12)
    # Full sketch rank recovers the exact map, even if storage guard rejects it.
    result = transfer.compress(min(transfer.shape), power_iterations=0)
    np.testing.assert_allclose(result.candidate @ left.ravel(), expected_left.ravel(), atol=2e-12)
    assert not result.accepted


def test_snapshot_survives_in_place_state_and_operator_edits():
    cache = make_cache(LETTAEnvironmentCache)
    transfer = cache.segment_transfer(1, 3)
    v = np.arange(transfer.shape[1], dtype=float)
    before = transfer @ v
    cache.state.tensors[1] *= 2
    cache.mpo.factors[1][...] *= 3
    np.testing.assert_array_equal(transfer @ v, before)
    np.testing.assert_allclose(cache.segment_transfer(1, 3) @ v, 12 * before, atol=1e-11)


@pytest.mark.parametrize("kind", [IdentityEnvironmentCache, LETTAEnvironmentCache])
def test_complete_segment_matches_physical_wavefunction(kind):
    cache = make_cache(kind)
    vector = cache.state.state_vector()
    expected = (np.vdot(vector, cache.mpo.to_dense() @ vector)
                if hasattr(cache, "mpo") else np.vdot(vector, vector))
    actual = cache.segment_transfer(0, cache.state.nsites).apply_left(np.asarray(1.))
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_accuracy_guard_falls_back_to_exact_and_checks_both_directions():
    transfer = make_cache(IdentityEnvironmentCache).segment_transfer(1, 3)
    result = transfer.compress(1, oversampling=0, tolerance=1e-12, seed=42)
    assert not result.accepted
    assert result.diagnostics["reason"] == "probe error exceeds tolerance"
    assert result.diagnostics["forward_probe_error"] > 1e-12
    assert result.diagnostics["adjoint_probe_error"] > 1e-12
    left, right = np.ones(transfer.input_shape), np.ones(transfer.output_shape)
    np.testing.assert_array_equal(result.apply_left(left), transfer.apply_left(left))
    np.testing.assert_array_equal(result.apply_right(right), transfer.apply_right(right))


def test_low_rank_segment_is_accepted_and_reusable():
    state = LatticeLETTA.random((3, 3), bond_dim=2, seed=816, real=False)
    for tensor in state.tensors:
        tensor[:] = (1 + 0.2j) / np.sqrt(tensor.size)
    cache = IdentityEnvironmentCache(state)
    transfer = cache.segment_transfer(1, 8)
    result = transfer.compress(1, tolerance=1e-12)
    assert result.accepted
    assert result.diagnostics["factor_to_dense_storage_ratio"] < 1
    left, right = cache.build_left_environments()[1], cache.build_right_environments()[8]
    np.testing.assert_allclose(result.apply_left(left), transfer.apply_left(left), atol=1e-15)
    np.testing.assert_allclose(result.apply_right(right), transfer.apply_right(right), atol=1e-15)


def test_validation_memory_guard_and_approximate_source_rejection():
    cache = make_cache(IdentityEnvironmentCache)
    transfer = cache.segment_transfer(1, 3)
    with pytest.raises(MemoryError):
        transfer.compress(2, max_workspace_mb=1e-8)
    for kwargs in ({"rank": 0}, {"rank": 1, "tolerance": -1},
                   {"rank": 1, "validation_vectors": 0}):
        with pytest.raises(ValueError):
            transfer.compress(**kwargs)
    for cuts in ((-1, 2), (2, 2), (1, 5)):
        with pytest.raises(ValueError):
            cache.segment_transfer(*cuts)
    cache = IdentityEnvironmentCache(cache.state, boundary_bond_dim=4)
    with pytest.raises(ValueError, match="exact"):
        cache.segment_transfer(1, 3)

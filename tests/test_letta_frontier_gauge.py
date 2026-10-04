"""Compare complete-frontier gauges with independent physical wavefunctions."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import (
    AbelianSymmetry, IdentityEnvironmentCache, LatticeLETTA, LETTADMROptions,
    canonicalize_frontier, frontier_gauge_cuts, letta_dmrg, shift_frontier_gauge,
)
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo


class CrossingExample(LatticeLETTA):
    dependencies = ((0, 2), (1,), (2, 3), (3, 4), (4,))

    def _build_neighborhood(self, coordinate):
        return self.dependencies[coordinate[1]]

    def copy(self):
        return type(self)(self.lattice_shape, self.physical_dim, self.tensors)


def crossing_example(*, real=False, right_dimension=2, seed=91):
    rng = np.random.default_rng(seed)
    dimensions = (1, 2, 2, 2, right_dimension, 1)
    tensors = []
    for i, ns in enumerate(CrossingExample.dependencies):
        shape = (dimensions[i],) + (2,) * len(ns) + (dimensions[i + 1],)
        a = rng.normal(size=shape)
        if not real:
            a = a + 1j * rng.normal(size=shape)
        tensors.append(a)
    return CrossingExample((1, 5), 2, tensors)


def gram_at(state, cut, direction):
    cache = IdentityEnvironmentCache(state)
    envs = (cache.build_left_environments() if direction == "lr"
            else cache.build_right_environments())
    physical = frontier_gauge_cuts(state)[cut - 1].physical_indices
    labels = tuple(cache.physical[p] for p in physical)
    order = [cache.frontiers[cut].index(label) for label in
             labels + (cache.bra_virtual[cut], cache.ket_virtual[cut])]
    return np.asarray(envs[cut]).transpose(order)


def test_cut_condition_depends_on_exposed_indices_not_internal_crossings():
    cuts = frontier_gauge_cuts(crossing_example())
    assert [c.admissible for c in cuts] == [False, False, True, True]
    assert cuts[2].physical_indices == (3,)
    assert cuts[2].shared_indices == (3,)


@pytest.mark.parametrize("real", [False, True])
def test_users_example_full_left_and_rank_one_right(real):
    state = crossing_example(real=real)
    original = state.state_vector()
    reports = canonicalize_frontier(state, 3)
    np.testing.assert_allclose(state.state_vector(), original, atol=3e-14)
    left = gram_at(state, 3, "lr")
    right = gram_at(state, 4, "rl")
    np.testing.assert_allclose(left, np.broadcast_to(np.eye(2), left.shape), atol=2e-13)
    np.testing.assert_allclose(right, np.broadcast_to(np.diag([1., 0.]), right.shape), atol=2e-13)
    assert reports[2].full_identity
    assert reports[3].ranks == (1, 1)
    assert not reports[3].full_identity
    frame = state.local_frame(3)
    expected = np.zeros(state.tensors[3].shape)
    expected[..., 0] = 1.
    np.testing.assert_allclose(frame.conj().T @ frame, np.diag(expected.ravel()), atol=3e-13)


def test_users_example_has_full_identity_when_right_bond_is_one():
    state = crossing_example(right_dimension=1)
    canonicalize_frontier(state, 3)
    frame = state.local_frame(3)
    np.testing.assert_allclose(frame.conj().T @ frame, np.eye(frame.shape[1]), atol=2e-13)


@pytest.mark.parametrize("direction, site", [("lr", 2), ("rl", 3)])
def test_cached_outgoing_environment_matches_fresh_contraction(direction, site):
    state = crossing_example()
    cache = IdentityEnvironmentCache(state)
    envs = (cache.build_left_environments() if direction == "lr"
            else cache.build_right_environments())
    incoming = envs[site if direction == "lr" else site + 1]
    before = state.state_vector()
    result, report = shift_frontier_gauge(state, site, direction, cache=cache, incoming=incoming)
    fresh = (cache.build_left_environments() if direction == "lr"
             else cache.build_right_environments())[report.cut.cut]
    np.testing.assert_allclose(result, fresh, atol=2e-13)
    np.testing.assert_allclose(state.state_vector(), before, atol=2e-14)


def test_zero_and_tiny_sectors_are_not_truncated_or_inverted():
    state = crossing_example()
    state.tensors[-1][:, 0, :] = 0.
    state.tensors[-1][:, 1, :] *= 1e-15
    original = state.state_vector()
    _, report = shift_frontier_gauge(state, 4, "rl")
    assert report.ranks == (0, 1)
    np.testing.assert_allclose(state.state_vector(), original, rtol=2e-13, atol=1e-30)
    assert all(np.all(np.isfinite(a)) for a in state.tensors)


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_obstructed_cut_qr_preserves_state_and_dimensions(direction):
    state = crossing_example()
    original, shapes = state.state_vector(), [a.shape for a in state.tensors]
    _, report = shift_frontier_gauge(state, 0 if direction == "lr" else 1, direction)
    assert not report.applied
    assert [a.shape for a in state.tensors] == shapes
    np.testing.assert_allclose(state.state_vector(), original, atol=2e-14)


@pytest.mark.parametrize("center", [0, 2, 4])
def test_abelian_gauge_preserves_charge_masks(center):
    symmetry = AbelianSymmetry((0, 1), sector=0, moduli=2)
    state = LatticeLETTA.random((1, 5), bond_dim=4, seed=73, real=False, symmetry=symmetry)
    original = state.state_vector()
    canonicalize_frontier(state, center)
    for site, a in enumerate(state.tensors):
        np.testing.assert_allclose(a[~state.symmetry_mask(site)], 0., atol=1e-14)
    np.testing.assert_allclose(state.state_vector(), original, atol=1e-13)


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("matrix_free", [False, True])
def test_crossing_sweeps_match_dense_physical_energy(direction, matrix_free):
    state = crossing_example()
    h = transverse_field_ising_mpo((1, 5), field=.9)
    initial_energy = state.expectation(h)
    result = letta_dmrg(h, state=state, options=LETTADMROptions(
        max_sweeps=3, start_direction=direction, gauge_mode="frontier",
        matrix_free=matrix_free, dense_solver_threshold=1,
    ))
    vector = result.state.state_vector()
    expected = np.real(np.vdot(vector, h.to_dense() @ vector) / np.vdot(vector, vector))
    np.testing.assert_allclose(result.energy, expected, atol=1e-11)
    energies = [initial_energy] + [s.energy for s in result.history]
    assert np.max(np.diff(energies)) < 1e-10
    np.testing.assert_allclose(state.norm(), 1., atol=1e-13)


@pytest.mark.parametrize("granularity, cbe, alternate", [
    ("column", False, True), ("site", True, True), ("site", False, False),
])
def test_frontier_sweep_variants(granularity, cbe, alternate):
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=78, real=False)
    h = transverse_field_ising_mpo((2, 2))
    result = letta_dmrg(h, state=state, options=LETTADMROptions(
        max_sweeps=2, gauge_mode="frontier", environment_granularity=granularity,
        cbe_enabled=cbe, cbe_selector="shrewd", alternate=alternate,
    ))
    np.testing.assert_allclose(result.energy, result.state.expectation(h), atol=2e-11)
    assert result.energy <= state.expectation(h) + 1e-11


def test_frontier_rejects_compressed_environments():
    h = transverse_field_ising_mpo((1, 3))
    with pytest.raises(ValueError, match="exact boundary"):
        letta_dmrg(h, lattice_shape=(1, 3), options=LETTADMROptions(
            gauge_mode="frontier", boundary_bond_dim=2))


@pytest.mark.parametrize("matrix_free", [False, True])
def test_supported_identity_solve_skips_metric_eigendecomposition(monkeypatch, matrix_free):
    from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
    from pyqed._letta_one_site_opt.solver import (
        _lowest_generalized_eigenpair, _lowest_matrix_free_eigenpair,
    )
    state = crossing_example()
    canonicalize_frontier(state, 3)
    cache = IdentityEnvironmentCache(state)
    metric = cache.effective_metric(cache.build_left_environments()[3],
                                    cache.build_right_environments()[4], 3)
    coordinates = metric.coordinate_whitening(1e-12)
    assert coordinates is not None and len(coordinates[0]) == 8
    frame = state.local_frame(3)
    physical_h = np.diag(np.arange(32.))
    local_h = frame.conj().T @ physical_h @ frame

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical metric must not require a spectral whitening basis")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(BlockDiagonalMetric, "whitening_basis", forbidden)
    if matrix_free:
        energy, vector, rank, residual = _lowest_matrix_free_eigenpair(
            lambda v: local_h @ v, metric, 1e-12,
            initial_vector=state.tensors[3].ravel())
    else:
        energy, vector, rank, residual = _lowest_generalized_eigenpair(local_h, metric, 1e-12)
    expected = np.linalg.eigvalsh(local_h[np.ix_(coordinates[0], coordinates[0])])[0]
    np.testing.assert_allclose(energy, expected, atol=1e-11)
    assert rank == 8 and residual < 1e-10
    np.testing.assert_allclose(np.vdot(vector, metric @ vector), 1., atol=1e-13)


def test_coordinate_shortcut_rejects_correlations_but_preserves_small_scales():
    from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
    metric = BlockDiagonalMetric(2, [np.array([[1., .1], [.1, 1.]])], [np.arange(2)])
    assert metric.coordinate_whitening(1e-12) is None
    metric = BlockDiagonalMetric(2, [np.diag([1., 2e-12])], [np.arange(2)])
    retained, scales = metric.coordinate_whitening(1e-12)
    np.testing.assert_array_equal(retained, [0, 1])
    np.testing.assert_allclose(scales, 1 / np.sqrt([1., 2e-12]))


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_multiple_shared_indices_and_zero_width_frontiers(direction):
    for dependencies in (
        ((0, 2, 3), (1, 2, 3), (2, 3), (3,), (4,)),
        ((0,), (1,), (2,), (3,), (4,)),
    ):
        class Custom(CrossingExample):
            pass
        Custom.dependencies = dependencies
        rng = np.random.default_rng(89)
        tensors = [rng.normal(size=(1 if i == 0 else 2,) + (2,) * len(ns)
                                   + (1 if i == 4 else 2,)) for i, ns in enumerate(dependencies)]
        state = Custom((1, 5), 2, tensors)
        before = state.state_vector()
        reports = canonicalize_frontier(state, 4 if direction == "lr" else 0)
        np.testing.assert_allclose(state.state_vector(), before, atol=3e-13)
        assert any(r.applied for r in reports)
        assert max(r.projector_residual for r in reports if r.applied) < 1e-10

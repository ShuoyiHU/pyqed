"""Regression for physical-index-dependent LETTA bond compression."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_one_site_opt.cbe import _directional_one_site_metric_trim
from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_two_site_opt import IdentityPairEnvironmentCache, LETTAPairLayout


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_general_singular_trim_preserves_known_rank_one_completion(direction):
    # The unobserved last entry is still a useful rank-one completion. Raising
    # N @ target with N+ sets it to zero and destroys that initial matrix rank.
    target = np.ones((3, 3))
    weights = np.ones(9)
    weights[-1] = 0.
    metric = BlockDiagonalMetric(9, [np.diag(weights)], [np.arange(9)])
    a, b = (target, np.eye(3)) if direction == "lr" else (np.eye(3), target)
    trim = _directional_one_site_metric_trim(
        a, b, metric, bond_dimension=1, direction=direction,
        tolerance=1.e-12, max_iterations=4, metric_tolerance=1.e-12)
    difference = (target - trim.left_tensor @ trim.right_tensor).ravel()
    assert np.real(np.vdot(difference, metric @ difference)) < 1.e-24
    assert trim.metric_kinds == ("general",)


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("singular", [False, True])
def test_separable_trim_attains_supported_weighted_svd_optimum(direction, singular):
    rng = np.random.default_rng(981)
    rows, columns = (7, 4) if direction == "lr" else (4, 7)
    ul, _ = np.linalg.qr(rng.normal(size=(rows, rows)) + 1j * rng.normal(size=(rows, rows)))
    ur, _ = np.linalg.qr(rng.normal(size=(columns, columns)) + 1j * rng.normal(size=(columns, columns)))
    vl, vr = np.linspace(.2, 3., rows), np.linspace(.1, 2., columns)
    if singular:
        vl[0], vr[0] = 0., 0.
    gl, gr = (ul * vl) @ ul.conj().T, (ur * vr) @ ur.conj().T
    dense_metric = np.kron(gl, gr)
    metric = BlockDiagonalMetric(rows * columns, [dense_metric], [np.arange(rows * columns)])
    target = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    a, b = (target, np.eye(columns)) if direction == "lr" else (np.eye(rows), target)
    white_target = (np.sqrt(vl)[:, None] * (ul.conj().T @ target @ ur.conj())
                    * np.sqrt(vr)[None, :])
    singular_values = np.linalg.svd(white_target, compute_uv=False)
    expected_loss = np.sum(singular_values[2:] ** 2)
    trim = _directional_one_site_metric_trim(
        a, b, metric, bond_dimension=2, direction=direction,
        tolerance=1.e-11, max_iterations=1, metric_tolerance=1.e-12)
    difference = (target - trim.left_tensor @ trim.right_tensor).ravel()
    actual_loss = np.real(np.vdot(difference, dense_metric @ difference))
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1.e-10, atol=1.e-10)
    np.testing.assert_allclose(trim.loss, actual_loss, rtol=1.e-10, atol=1.e-10)
    assert trim.iterations == 0
    assert trim.metric_kinds == ("separable",)


def test_separable_compression_removes_bose_hubbard_trim_stagnation():
    from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
    model = build_model("bose_hubbard", "1d", 6)
    initial = make_shared_initial_state(model, bond_dim=4, seed=1735).letta
    result = letta_dmrg(
        model.mpo, state=initial,
        options=LETTADMROptions(max_sweeps=200, tolerance=1e-12,
                               cbe_enabled=True, cbe_selector="shrewd"))
    assert result.converged
    np.testing.assert_allclose(result.energy, -16.269443059459, atol=2e-9, rtol=0)
    vector = result.state.state_vector()
    physical = np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(result.energy, physical, atol=2e-10, rtol=0)
    updates = [u for sweep in result.history for u in sweep.updates]
    assert any("separable" in u.cbe_trim_metric_kinds for u in updates)


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("real", [True, False])
def test_trim_preserves_conditionally_rank_one_state(direction, real):
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=3, seed=871, real=real)
    layout = LETTAPairLayout.from_state(state, 1)
    rng = np.random.default_rng(872)
    active_site = 1 if direction == "lr" else 2
    neighborhood = layout.left_neighborhood if direction == "lr" else layout.right_neighborhood
    active = state.tensors[active_site]
    for configuration in np.ndindex(*((2,) * len(layout.shared))):
        section = [slice(None)] * active.ndim
        for site, value in zip(layout.shared, configuration):
            section[1 + neighborhood.index(site)] = value
        shape = active[tuple(section)].shape
        rows = int(np.prod(shape[:-1])) if direction == "lr" else shape[0]
        columns = shape[-1] if direction == "lr" else int(np.prod(shape[1:]))
        left = rng.normal(size=(rows, 1))
        right = rng.normal(size=(1, columns))
        if not real:
            left = left + 1j * rng.normal(size=left.shape)
            right = right + 1j * rng.normal(size=right.shape)
        active[tuple(section)] = (left @ right).reshape(shape)
    target = layout.merge(state.tensors[1], state.tensors[2])
    cache = IdentityPairEnvironmentCache(state)
    metric = cache.effective_metric(
        cache.build_left_environments()[active_site],
        cache.build_right_environments()[active_site + 1], active_site,
    )
    trim = _directional_one_site_metric_trim(
        state.tensors[1], state.tensors[2], metric,
        bond_dimension=1, direction=direction, tolerance=1e-10,
        max_iterations=4, metric_tolerance=1e-12, layout=layout,
    )
    reconstructed = layout.merge(trim.left_tensor, trim.right_tensor)
    np.testing.assert_allclose(reconstructed, target, rtol=1e-8, atol=1e-8)
    assert trim.loss < 1e-15


@pytest.mark.parametrize("direction", ["lr", "rl"])
@pytest.mark.parametrize("left_site", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
def test_conditional_trim_reports_physical_loss_without_pair_operations(direction, left_site, mixed, monkeypatch):
    state = LatticeLETTA.random((2, 3), physical_dim=2, bond_dim=3, seed=875, real=False)
    layout = LETTAPairLayout.from_state(state, left_site)
    active_site = left_site if direction == "lr" else left_site + 1
    if mixed:
        state.tensors[active_site] = state.tensors[active_site].real.copy()
    cache = IdentityPairEnvironmentCache(state)
    left_env = cache.build_left_environments()
    right_env = cache.build_right_environments()
    metric = cache.effective_metric(left_env[active_site], right_env[active_site + 1], active_site)
    pair_metric = cache.effective_pair_metric(left_env[left_site], right_env[left_site + 2], layout)
    target = layout.merge(*state.tensors[left_site:left_site + 2]).reshape(-1)

    def forbidden(*args, **kwargs):
        raise AssertionError("conditional trim used a pair-space operation")

    with monkeypatch.context() as patch:
        patch.setattr(LETTAPairLayout, "merge", forbidden)
        patch.setattr(IdentityPairEnvironmentCache, "effective_pair_metric", forbidden)
        trim = _directional_one_site_metric_trim(
            *state.tensors[left_site:left_site + 2], metric,
            bond_dimension=2, direction=direction, tolerance=1e-10,
            max_iterations=4, metric_tolerance=1e-12, layout=layout,
        )
    approximation = layout.merge(trim.left_tensor, trim.right_tensor).reshape(-1)
    difference = target - approximation
    loss = np.real(np.vdot(difference, pair_metric @ difference))
    norm = np.sqrt(np.real(np.vdot(approximation, pair_metric @ approximation)))
    np.testing.assert_allclose(trim.loss, loss, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(trim.norm, norm, rtol=1e-8, atol=1e-12)

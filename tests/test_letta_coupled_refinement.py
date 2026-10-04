"""Regressions for slow directions that need simultaneous factor changes."""

import numpy as np
import pytest

from pyqed._letta_one_site_opt import AbelianSymmetry, LatticeLETTA
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import (
    IdentityPairEnvironmentCache, LETTAPairEnvironmentCache, LETTAPairLayout,
    LETTATwoSiteOptions, letta_two_site_dmrg,
)
from pyqed._letta_two_site_opt.energy_refinement import _coupled_factor_descent, _metric_descent


@pytest.mark.parametrize("condition_blocks", [False, True])
def test_coupled_direction_preserves_small_physical_coordinate_scales(condition_blocks):
    """Changing parameter units cannot remove independent physical directions."""
    scales = np.array([1e-10, 1., 1e10])
    metric = np.diag(scales**2)
    gradient = scales.copy()
    if condition_blocks:
        from pyqed._letta_one_site_opt.cbe_coupled import _conditioned_direction
        direction, residual, independent = _conditioned_direction(
            metric, gradient, 2, 1e-12)
        np.testing.assert_allclose(independent, 3., rtol=1e-12)
    else:
        direction, residual = _metric_descent(metric, gradient, 1e-12)
    np.testing.assert_allclose(scales * direction, -np.ones(3), rtol=1e-12)
    np.testing.assert_allclose(residual, 3., rtol=1e-12)


@pytest.mark.parametrize("condition_blocks", [False, True])
def test_normalization_roundoff_is_not_whitened_into_a_direction(condition_blocks):
    scales = np.array([1e-10, 1., 1e10])
    metric = np.diag(scales**2)
    gradient = scales * np.array([1e-14, 1., 1.])
    overlap = scales * np.array([1. - 1e-15, 0., 0.])
    if condition_blocks:
        from pyqed._letta_one_site_opt.cbe_coupled import _conditioned_direction
        direction, residual, _ = _conditioned_direction(
            metric, gradient, 2, 1e-12, overlap=overlap)
    else:
        direction, residual = _metric_descent(metric, gradient, 1e-12, overlap=overlap)
    np.testing.assert_allclose(scales * direction, [0., -1., -1.], atol=1e-12)
    np.testing.assert_allclose(residual, 2., atol=1e-12)


def test_coupled_refinement_closes_small_bose_gap():
    model = build_model("bose_hubbard", "2d", (2, 2))
    initial = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    before = initial.state_vector().copy()
    result = letta_two_site_dmrg(
        model.mpo, state=initial, bond_dim=2,
        options=LETTATwoSiteOptions(max_sweeps=40, tolerance=1e-12),
    )
    # Independent global and pair tangent descents reach this same D2 endpoint.
    # It is a variational reference, not the exact ground-state energy.
    reference = -11.69709376305754
    assert result.converged
    np.testing.assert_allclose(result.energy, reference, atol=5e-10, rtol=0)
    vector = result.state.state_vector()
    physical = np.vdot(vector, model.mpo.to_dense() @ vector) / np.vdot(vector, vector)
    np.testing.assert_allclose(result.energy, physical, atol=2e-10, rtol=0)
    assert np.max(np.diff([initial.expectation(model.mpo)]
                          + [s.energy for s in result.history])) < 2e-10
    assert result.state.bond_dimensions == (2, 2, 2)
    np.testing.assert_array_equal(initial.state_vector(), before)
    assert sum(u.coupled_refinement_accepted_steps for s in result.history for u in s.updates) > 0


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("restricted", [False, True])
def test_coupled_descent_preserves_physical_energy_and_symmetry(complex_values, restricted):
    symmetry = AbelianSymmetry((0, 1), sector=0, moduli=2) if restricted else None
    state = LatticeLETTA.random((2, 2), bond_dim=2, seed=713, symmetry=symmetry)
    if complex_values:
        rng = np.random.default_rng(722)
        state.tensors = [a * np.exp(1j * rng.normal(size=a.shape)) for a in state.tensors]
    mpo = transverse_field_ising_mpo((2, 2), coupling=0.7, field=1.2, basis="x")
    layout = LETTAPairLayout.from_state(state, 0)
    hc, nc = LETTAPairEnvironmentCache(state, mpo), IdentityPairEnvironmentCache(state)
    hr, nr = hc.build_right_environments(), nc.build_right_environments()
    action = hc.prepare_pair_action(hc.scalar_boundary(), hr[2], layout)
    metric = nc.effective_pair_metric(nc.scalar_boundary(), nr[2], layout)
    initial = [a.copy() for a in state.tensors]
    v = state.state_vector()
    h = mpo.to_dense()  # Validation only.
    before = float(np.vdot(v, h @ v).real / np.vdot(v, v).real)
    masks = dict(left_indices=np.flatnonzero(layout.factor_mask("left")),
                 right_indices=np.flatnonzero(layout.factor_mask("right"))) if restricted else {}
    left, right, energy, iterations, accepted = _coupled_factor_descent(
        layout, *state.tensors[:2], action, metric, max_iterations=5,
        metric_tolerance=1e-12, coupling_ratio=0, norm_limit=100, **masks,
    )
    assert 0 < accepted <= iterations <= 5
    for actual, original in zip(state.tensors, initial):
        np.testing.assert_array_equal(actual, original)
    state.tensors[:2] = [left, right]
    v = state.state_vector()
    physical = float(np.vdot(v, h @ v).real / np.vdot(v, v).real)
    assert physical < before - 1e-5
    np.testing.assert_allclose(energy, physical, rtol=0, atol=2e-12)
    if restricted:
        assert state.symmetry_violation() == 0.0
    assert state.bond_dimensions == (2, 2, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_zero_rank_tangent_stops_without_backtracking(dtype):
    state = LatticeLETTA.random((1, 2), physical_dim=1, bond_dim=1, seed=9)
    layout = LETTAPairLayout.from_state(state, 0)
    left, right = [np.ones(a.shape, dtype=dtype) for a in state.tensors]
    metric = np.eye(1, dtype=dtype)
    result = _coupled_factor_descent(
        layout, left, right, lambda v: -v, metric, max_iterations=8,
        metric_tolerance=1e-12, coupling_ratio=5, norm_limit=100,
    )
    np.testing.assert_array_equal(result[0], left)
    np.testing.assert_array_equal(result[1], right)
    assert result[2:] == (-1.0, 1, 0)


def test_coupled_steps_respect_the_supplied_factor_budget():
    state = LatticeLETTA.random((1, 2), bond_dim=1, seed=139)
    layout = LETTAPairLayout.from_state(state, 0)
    left, right = state.tensors
    metric = np.eye(int(np.prod(layout.merged_shape)))
    rng = np.random.default_rng(151)
    h = rng.normal(size=metric.shape)
    h = h + h.T
    theta = layout.merge(left, right).reshape(-1)
    before = np.vdot(theta, h @ theta).real / np.vdot(theta, theta).real
    # No normalized factorization fits this budget. The inner search must
    # reject every proposal rather than use a new budget after each step.
    result = _coupled_factor_descent(
        layout, left, right, lambda v: h @ v, metric, max_iterations=8,
        metric_tolerance=1e-12, coupling_ratio=0, norm_limit=1e-30,
    )
    assert result[3:] == (1, 0)
    np.testing.assert_allclose(result[2], before, atol=2e-14, rtol=0)


@pytest.mark.parametrize("name,value", [
    ("coupled_refinement_max_iterations", -1),
    ("coupled_refinement_max_iterations", 1.5),
    ("coupled_refinement_metric_tolerance", 0),
    ("coupled_refinement_metric_tolerance", np.nan),
    ("coupled_refinement_metric_tolerance", 1),
    ("coupled_refinement_activation_ratio", -1),
    ("coupled_refinement_activation_ratio", np.inf),
])
def test_invalid_coupled_options_are_rejected(name, value):
    model = build_model("heisenberg", "2d", (2, 2))
    with pytest.raises(ValueError, match=name):
        letta_two_site_dmrg(model.mpo, lattice_shape=(2, 2), bond_dim=2,
                           options=LETTATwoSiteOptions(**{name: value}))

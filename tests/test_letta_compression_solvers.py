"""Physical-objective and derivative tests for production compression solvers."""
import numpy as np
import pytest

from pyqed._letta_compression import (
    MetricCompressionOptions, compress_factors, factor_blocks, _balance,
    _chart, _Coordinates, _ProjectedProblem,
)
from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_two_site_opt import (
    LETTAPairLayout, LETTATwoSiteOptions, conditional_svd_split, metric_refine,
    letta_two_site_dmrg,
)

METHODS = ['variable-projection', 'joint-ls', 'grassmann-newton']


def metric(a):
    return BlockDiagonalMetric(len(a), [a], [np.arange(len(a))])


def initial(a, rank):
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    return u[:, :rank], s[:rank, None]*vh[:rank]


def run(a, m, rank, method, **kwargs):
    left, right = initial(a, rank)
    d = (left@right-a).ravel()
    loss = float(np.vdot(d, m@d).real)
    return compress_factors(a, m, left, right,
                            options=MetricCompressionOptions(solver=method, **kwargs),
                            fallback=lambda: (left, right, loss, 0))


@pytest.mark.parametrize('complex_data', [False, True])
def test_projected_jacobian_and_newton_hessian(complex_data):
    rng = np.random.default_rng(913)
    a = rng.normal(size=(4, 3))
    s = rng.normal(size=(12, 12))
    if complex_data:
        a = a+1j*rng.normal(size=a.shape)
        s = s+1j*rng.normal(size=s.shape)
    left, right = initial(a, 2)
    dtype = a.dtype
    def weighted(v):
        x = s@v.ravel()
        return np.r_[x.real, x.imag] if complex_data else x
    base, directions = _chart(left, factor_blocks(left, right))
    problem = _ProjectedProblem(base, directions, _Coordinates(right.shape, dtype),
                                lambda x, t: weighted(x@t), weighted(a),
                                lambda x, t: None, 1e-12)
    z = rng.normal(size=len(directions))*.05
    d = rng.normal(size=z.shape)
    h = 1e-6
    jac = problem.jacobian(z).copy()
    hess = problem.hessian(z).copy()
    numeric = (problem.residual(z+h*d)-problem.residual(z-h*d))/(2*h)
    np.testing.assert_allclose(jac@d, numeric, rtol=2e-6, atol=1e-7)
    numeric_hess = (problem.gradient(z+h*d)-problem.gradient(z-h*d))/(2*h)
    np.testing.assert_allclose(hess@d, numeric_hess, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('transpose', [False, True])
def test_separable_problem_matches_known_global_minimum(method, transpose):
    rng = np.random.default_rng(14)
    a = rng.normal(size=(4, 3))
    l, r = np.diag([1., 2., 3., 4.]), np.diag([2., 1., 3.])
    n = np.kron(l.T@l, r@r.T)
    expected = np.linalg.svd(l@a@r, compute_uv=False)[-1]**2
    if transpose:
        indices = np.arange(a.size).reshape(a.shape).T.ravel()
        a, n = a.T, n[np.ix_(indices, indices)]
    fit = run(a, metric(n), 2, method, max_iterations=300, tolerance=1e-12)
    assert fit.diagnostics['used_solver'] == method
    assert fit.loss == pytest.approx(expected, rel=1e-7, abs=1e-9)
    np.testing.assert_allclose(fit.left.conj().T@fit.left, fit.right@fit.right.conj().T, atol=1e-10)


@pytest.mark.parametrize('method', METHODS)
def test_complex_semidefinite_full_physical_loss_and_rank(method):
    rng = np.random.default_rng(815)
    a = rng.normal(size=(5, 3))+1j*rng.normal(size=(5, 3))
    s = rng.normal(size=(10, 15))+1j*rng.normal(size=(10, 15))
    fit = run(a, metric(s.conj().T@s), 2, method)
    assert fit.diagnostics['used_solver'] == method
    assert np.linalg.matrix_rank(fit.left@fit.right, tol=1e-9) <= 2
    assert fit.loss == pytest.approx(np.linalg.norm(s@(a-fit.left@fit.right).ravel())**2, abs=1e-9)
    assert fit.loss <= fit.diagnostics['initial_loss']+1e-10


@pytest.mark.parametrize('method', METHODS)
def test_workspace_fallback_before_dense_chart_allocation(method, monkeypatch):
    import pyqed._letta_compression as module
    def fail(*args):
        pytest.fail('allocated chart before checking workspace')
    monkeypatch.setattr(module, '_chart', fail)
    a = np.arange(12.).reshape(4, 3)
    fit = run(a, metric(np.eye(12)), 1, method, max_workspace_mb=1e-8)
    assert fit.diagnostics['used_solver'] == 'als'
    assert 'workspace' in fit.diagnostics['fallback_reason']


def test_nonfinite_trial_falls_back_without_losing_valid_fit(monkeypatch):
    import pyqed._letta_compression as module
    def fail(fun, x, **kwargs):
        fun(x)
        raise FloatingPointError('nonfinite trial')
    monkeypatch.setattr(module, 'least_squares', fail)
    a = np.arange(12.).reshape(4, 3)
    fit = run(a, metric(np.eye(12)), 1, 'variable-projection')
    assert fit.diagnostics['used_solver'] == 'als'
    assert 'nonfinite trial' in fit.diagnostics['fallback_reason']
    assert np.isfinite(fit.loss)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('symmetric', [False, True])
def test_pair_shared_indices_and_charge_masks_use_full_coupled_metric(method, symmetric):
    from pyqed._letta_one_site_opt import AbelianSymmetry
    symmetry = AbelianSymmetry(physical_charges=(0, 1), sector=0, moduli=2) if symmetric else None
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=71, symmetry=symmetry)
    layout = LETTAPairLayout.from_state(state, 0)
    rng = np.random.default_rng(91)
    a = rng.normal(size=layout.merged_shape)
    if symmetric:
        a[~layout.symmetry_mask()] = 0
    split = conditional_svd_split(a, layout, max_bond_dim=2, direction='lr')
    # This metric intentionally couples shared-physical configurations. A
    # collection of independent per-sector fits would solve the wrong problem.
    s = rng.normal(size=(a.size, a.size))
    m = metric(s.T@s+np.eye(a.size))
    fit = metric_refine(a, layout, split, m,
                        compression=MetricCompressionOptions(solver=method, max_iterations=40))
    assert fit.diagnostics['used_solver'] == method
    d = (layout.merge(fit.left_tensor, fit.right_tensor)-a).ravel()
    assert fit.loss == pytest.approx(np.vdot(d, m@d).real)
    assert fit.loss <= fit.diagnostics['initial_loss']+1e-9
    if symmetric:
        assert np.count_nonzero(fit.left_tensor[~layout.factor_mask('left')]) == 0
        assert np.count_nonzero(fit.right_tensor[~layout.factor_mask('right')]) == 0


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('algorithm', ['two-site', 'cbe-exact', 'cbe-shrewd'])
def test_solver_option_reaches_live_updates_and_energy_guard(method, algorithm):
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
    from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
    model = build_model('bose_hubbard', '2d', (2, 2))
    state = make_shared_initial_state(model, bond_dim=2, seed=731).letta
    config = MetricCompressionOptions(solver=method, max_iterations=10)
    if algorithm == 'two-site':
        result = letta_two_site_dmrg(model.mpo, state=state,
            options=LETTATwoSiteOptions(max_sweeps=2, split_method="metric-energy", compression=config))
        diagnostics = [u.compression_diagnostics for s in result.history for u in s.updates]
    else:
        result = letta_dmrg(model.mpo, state=state,
            options=LETTADMROptions(max_sweeps=2, cbe_enabled=True,
                                   cbe_selector=algorithm[4:], compression=config))
        diagnostics = [d for s in result.history for u in s.updates for d in u.cbe_compression_diagnostics]
    assert diagnostics
    assert any(d['used_solver'] == method for d in diagnostics)
    assert all(d.get('fallback_reason') is None for d in diagnostics)
    assert result.history[-1].energy <= result.history[0].energy+1e-8
    assert np.isfinite(result.energy)


@pytest.mark.parametrize('method', METHODS)
def test_zero_metric_needs_no_optimizer(method):
    a = np.arange(12.).reshape(4, 3)
    fit = run(a, metric(np.zeros((12, 12))), 1, method)
    assert fit.loss == 0
    assert fit.iterations == 0
    assert fit.diagnostics['status'] == 'zero metric'


@pytest.mark.parametrize('method', METHODS)
def test_padding_and_full_row_rank_chart(method):
    rng = np.random.default_rng(73)
    a = rng.normal(size=(2, 4))
    left = np.zeros((2, 5))
    left[:, :2] = np.eye(2)
    right = rng.normal(size=(5, 4))
    m = metric(np.eye(8))
    fit = compress_factors(a, m, left, right,
                          options=MetricCompressionOptions(solver=method),
                          fallback=lambda: pytest.fail('unexpected fallback'))
    assert fit.left.shape == (2, 5)
    assert fit.right.shape == (5, 4)
    np.testing.assert_allclose(fit.left@fit.right, a, atol=1e-7)


def test_als_budget_override_and_default_agree_with_explicit_refinement():
    from pyqed._letta_two_site_opt import metric_als_refine
    state = LatticeLETTA.random((2, 2), physical_dim=2, bond_dim=2, seed=88)
    layout = LETTAPairLayout.from_state(state, 0)
    rng = np.random.default_rng(38)
    a = rng.normal(size=layout.merged_shape)
    s = rng.normal(size=(a.size, a.size))
    m = metric(s.T@s+np.eye(a.size))
    split = conditional_svd_split(a, layout, max_bond_dim=2, direction='lr')
    expected = metric_als_refine(a, layout, split, m, max_iterations=2, lsmr_max_iterations=3)
    actual = metric_refine(a, layout, split, m,
                          compression=MetricCompressionOptions(als_max_iterations=2, lsmr_max_iterations=3))
    np.testing.assert_array_equal(actual.left_tensor, expected.left_tensor)
    np.testing.assert_array_equal(actual.right_tensor, expected.right_tensor)
    assert actual.iterations == expected.iterations


@pytest.mark.parametrize('kwargs', [dict(solver='unknown'), dict(max_iterations=0),
    dict(lsmr_max_iterations=-1), dict(tolerance=float('nan')), dict(max_workspace_mb=0)])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MetricCompressionOptions(**kwargs)

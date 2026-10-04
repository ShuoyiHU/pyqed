import numpy as np
import pytest

from pyqed._letta_one_site_opt.contractions import BlockDiagonalMetric
from pyqed._letta_one_site_opt.benchmarks.metric_compression_solvers import (
    WeightedProblem, VariableProjection, nonlinear_fit, als_fit, solve,
)
from pyqed._letta_one_site_opt.cbe import _metric_low_rank_factorization
from pyqed._letta_one_site_opt.benchmarks.grassmann_compression import (
    GrassmannChart, newton_fit,
)


def metric(matrix):
    return BlockDiagonalMetric(len(matrix), [matrix], [np.arange(len(matrix))])


@pytest.mark.parametrize("complex_data", [False, True])
def test_variable_projection_derivative_correlated_metric(complex_data):
    rng = np.random.default_rng(345)
    a, s = rng.normal(size=(4, 3)), rng.normal(size=(12, 12))
    if complex_data:
        a = a + 1j*rng.normal(size=a.shape)
        s = s + 1j*rng.normal(size=s.shape)
    p = WeightedProblem(a, metric(s.conj().T @ s + np.eye(12)), 2)
    vp = VariableProjection(p)
    x = p.pack(p.left)
    direction = rng.normal(size=x.shape)
    analytic = vp.jac(x) @ direction
    h = 1e-6
    numeric = (vp.fun(x+h*direction) - vp.fun(x-h*direction))/(2*h)
    np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-6)
    # The eliminated factor really minimizes its linear subproblem.
    vp.evaluate(x)
    jt = p.joint_jacobian(vp.left, vp.right)[:, len(x):]
    np.testing.assert_allclose(jt.T @ vp.r, 0, atol=1e-10)


@pytest.mark.parametrize("method", ["varpro_trf", "varpro_lm", "joint_trf"])
def test_matches_separable_weighted_svd_global_solution(method):
    rng = np.random.default_rng(456)
    a = rng.normal(size=(4, 3))
    l = np.diag([1., 2., 3., 4.])
    r = np.diag([4., 2., 1.])
    m = metric(np.kron(l.T @ l, r @ r.T))
    expected = np.linalg.svd(l @ a @ r, compute_uv=False)[2]**2
    fit = nonlinear_fit(a, m, 2, method=method, max_nfev=500)
    assert fit.loss == pytest.approx(expected, rel=1e-7, abs=1e-9)
    assert np.linalg.matrix_rank(fit.left @ fit.right, tol=1e-9) <= 2


@pytest.mark.parametrize("method", ["varpro_trf", "varpro_lm", "joint_trf", "grassmann_newton"])
def test_semidefinite_complex_metric_no_mutation_and_physical_loss(method):
    rng = np.random.default_rng(951)
    a = rng.normal(size=(4, 3)) + 1j*rng.normal(size=(4, 3))
    s = rng.normal(size=(8, 12)) + 1j*rng.normal(size=(8, 12))
    m = metric(s.conj().T @ s)
    before = a.copy()
    fit = (newton_fit(a, m, 2, max_nfev=100) if method == 'grassmann_newton'
           else nonlinear_fit(a, m, 2, method=method, max_nfev=100))
    expected = np.linalg.norm(s @ (a - fit.left @ fit.right).ravel())**2
    assert fit.loss == pytest.approx(expected, abs=1e-9)
    assert fit.loss <= fit.diagnostics["initial_loss"]
    np.testing.assert_array_equal(a, before)


def test_als_baseline_replays_production_and_reports_inner_stops():
    rng = np.random.default_rng(55)
    a, s = rng.normal(size=(5, 4)), rng.normal(size=(20, 20))
    m = metric(s.T@s + np.eye(20))
    reference = _metric_low_rank_factorization(
        a, m, 2, tolerance=1e-10, max_iterations=4, metric_tolerance=1e-10)
    fit = als_fit(a, m, 2)
    np.testing.assert_array_equal(fit.left, reference[0])
    np.testing.assert_array_equal(fit.right, reference[1])
    assert fit.loss == reference[2]
    assert len(fit.diagnostics["lsmr"]) == 2*fit.iterations


@pytest.mark.parametrize("complex_data", [False, True])
def test_grassmann_chart_hessian_matches_gradient_difference(complex_data):
    rng = np.random.default_rng(891)
    a, s = rng.normal(size=(4, 3)), rng.normal(size=(12, 12))
    if complex_data:
        a = a + 1j*rng.normal(size=a.shape)
        s = s + 1j*rng.normal(size=s.shape)
    p = WeightedProblem(a, metric(s.conj().T@s + np.eye(12)), 2)
    chart = GrassmannChart(p)
    z = p.pack(np.zeros(chart.shape, dtype=a.dtype))
    direction = rng.normal(size=z.size)
    h = 1e-6
    analytic = chart.hess(z) @ direction
    numeric = (chart.jac(z+h*direction) - chart.jac(z-h*direction))/(2*h)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-7)
    assert chart.jac(z) @ direction == pytest.approx(
        (chart.fun(z+h*direction)-chart.fun(z-h*direction))/(2*h), abs=2e-7)


def test_grassmann_newton_matches_known_weighted_optimum():
    rng = np.random.default_rng(456)
    a = rng.normal(size=(4, 3))
    l, r = np.diag([1., 2., 3., 4.]), np.diag([4., 2., 1.])
    m = metric(np.kron(l.T @ l, r @ r.T))
    expected = np.linalg.svd(l @ a @ r, compute_uv=False)[2]**2
    fit = newton_fit(a, m, 2)
    assert fit.loss == pytest.approx(expected, rel=1e-8)


def test_smaller_side_mapping_preserves_physical_metric_objective():
    rng = np.random.default_rng(132)
    a, s = rng.normal(size=(5, 3)), rng.normal(size=(15, 15))
    m = metric(s.T@s + np.eye(15))
    fit = solve(a, m, 2, 'varpro_trf_small')
    difference = (a-fit.left@fit.right).ravel()
    assert fit.left.shape == (5, 2)
    assert fit.right.shape == (2, 3)
    assert fit.loss == pytest.approx(difference @ m.to_dense() @ difference)


def test_dense_inner_als_solves_known_weighted_problem():
    rng = np.random.default_rng(456)
    a = rng.normal(size=(4, 3))
    l, r = np.diag([1., 2., 3., 4.]), np.diag([4., 2., 1.])
    m = metric(np.kron(l.T @ l, r @ r.T))
    expected = np.linalg.svd(l @ a @ r, compute_uv=False)[2]**2
    fit = solve(a, m, 2, 'als40_dense')
    assert fit.loss == pytest.approx(expected, rel=1e-7)


def test_balancing_preserves_newton_compressed_matrix():
    rng = np.random.default_rng(1239)
    a, s = rng.normal(size=(5, 3)), rng.normal(size=(15, 15))
    m = metric(s.T@s + np.eye(15))
    raw = solve(a, m, 2, 'grassmann_newton')
    balanced = solve(a, m, 2, 'grassmann_newton_balanced')
    np.testing.assert_allclose(balanced.left @ balanced.right, raw.left @ raw.right,
                               rtol=1e-12, atol=1e-12)
    assert balanced.loss == pytest.approx(raw.loss, abs=1e-10)

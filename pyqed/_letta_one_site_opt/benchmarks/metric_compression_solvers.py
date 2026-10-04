"""Small-problem reference solvers for correlated-metric CBE compression.

These deliberately materialize the active-site square root and Jacobians.
They are benchmark references, not production dispatch or scalable solvers.
The rank constraint stays on X @ T, never on the whitened matrix.
"""
from dataclasses import dataclass
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

from .. import cbe

_PRODUCTION_ALS = cbe._metric_low_rank_factorization


@dataclass
class Fit:
    left: np.ndarray
    right: np.ndarray
    loss: float
    iterations: int
    diagnostics: dict


class WeightedProblem:
    """Real coordinates also support complex factors and Hermitian metrics."""

    def __init__(self, target, metric, rank, metric_tolerance=1e-10):
        self.target = np.asarray(target, dtype=np.result_type(target, metric.dtype))
        self.metric = metric
        self.rank = rank
        self.rows, self.columns = self.target.shape
        self.complex = np.iscomplexobj(self.target)
        root = cbe._MetricSquareRoot(metric, metric_tolerance)
        eye = np.eye(target.size)
        self.root = np.column_stack([root.apply(v) for v in eye])
        self.b = self.pack(self.root @ self.target.ravel())
        u, s, vh = np.linalg.svd(self.target, full_matrices=False)
        self.left = u[:, :rank].copy()
        self.right = s[:rank, None] * vh[:rank]

    def pack(self, a):
        a = np.asarray(a).ravel()
        return np.concatenate((a.real, a.imag)) if self.complex else a.real.copy()

    def unpack(self, a, shape):
        n = int(np.prod(shape))
        return (a[:n] + 1j * a[n:]).reshape(shape) if self.complex else a.reshape(shape)

    def basis(self, shape):
        n = int(np.prod(shape)) * (2 if self.complex else 1)
        return [self.unpack(v, shape) for v in np.eye(n)]

    def weighted(self, a):
        return self.pack(self.root @ a.ravel())

    def loss(self, left, right):
        return cbe._one_site_factorization_loss(self.target, left, right, self.metric)

    def residual(self, left, right):
        return self.weighted(left @ right) - self.b

    def joint_jacobian(self, left, right):
        columns = [self.weighted(dx @ right) for dx in self.basis(left.shape)]
        columns += [self.weighted(left @ dt) for dt in self.basis(right.shape)]
        return np.column_stack(columns)


class VariableProjection:
    """Exact residual derivative, including the nonzero-residual term.

    At locally constant rank, dr = (I-K K+) dK t - K+^T dK^T r.
    All operations below use real coordinates, even for complex tensors.
    """

    def __init__(self, problem):
        self.p = problem
        self.xbasis = problem.basis(problem.left.shape)
        self.tbasis = problem.basis(problem.right.shape)
        self.dk = [np.column_stack([problem.weighted(dx @ dt) for dt in self.tbasis])
                   for dx in self.xbasis]
        self.last_x = None
        self.evaluations = 0

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return
        left = self.p.unpack(x, self.p.left.shape)
        k = np.column_stack([self.p.weighted(left @ dt) for dt in self.tbasis])
        u, s, vh = np.linalg.svd(k, full_matrices=False)
        keep = s > 1e-12 * s[0] if s.size and s[0] else np.zeros(s.size, dtype=bool)
        u, s, vh = u[:, keep], s[keep], vh[keep]
        pinv = (vh.T / s) @ u.T
        t = pinv @ self.p.b
        self.r = k @ t - self.p.b
        self.j = np.column_stack([
            dk @ t - u @ (u.T @ (dk @ t)) - pinv.T @ (dk.T @ self.r)
            for dk in self.dk])
        self.left = left
        self.right = self.p.unpack(t, self.p.right.shape)
        self.inner_rank = len(s)
        self.last_x = x.copy()
        self.evaluations += 1

    def fun(self, x):
        self.evaluate(x)
        return self.r

    def jac(self, x):
        self.evaluate(x)
        return self.j


def nonlinear_fit(target, metric, rank, *, method="varpro_trf", max_nfev=100,
                  tolerance=1e-11, metric_tolerance=1e-10):
    started = perf_counter()
    p = WeightedProblem(target, metric, rank, metric_tolerance)
    initial_loss = p.loss(p.left, p.right)
    if method.startswith("varpro"):
        vp = VariableProjection(p)
        result = least_squares(vp.fun, p.pack(p.left), jac=vp.jac,
                               method=method.split("_")[1], max_nfev=max_nfev,
                               ftol=tolerance, xtol=tolerance, gtol=tolerance)
        vp.evaluate(result.x)
        left, right = vp.left, vp.right
        evaluations = vp.evaluations
        inner_rank = vp.inner_rank
    elif method == "joint_trf":
        nleft = p.pack(p.left).size
        def factors(x):
            return (p.unpack(x[:nleft], p.left.shape),
                    p.unpack(x[nleft:], p.right.shape))
        result = least_squares(lambda x: p.residual(*factors(x)),
                               np.r_[p.pack(p.left), p.pack(p.right)],
                               jac=lambda x: p.joint_jacobian(*factors(x)),
                               method="trf", max_nfev=max_nfev,
                               ftol=tolerance, xtol=tolerance, gtol=tolerance)
        left, right = factors(result.x)
        evaluations, inner_rank = int(result.nfev), None
    else:
        raise ValueError(method)
    loss = p.loss(left, right)
    reverted = not np.isfinite(loss) or loss > initial_loss
    if reverted:
        left, right, loss = p.left, p.right, initial_loss
    r = p.residual(left, right)
    # Report a gauge-normalized stationarity residual as well as SciPy's raw one.
    q, transfer = np.linalg.qr(left, mode="reduced")
    gradient = p.joint_jacobian(q, transfer @ right).T @ r
    return Fit(left, right, float(loss), evaluations, dict(
        seconds=perf_counter()-started, initial_loss=initial_loss,
        supported_loss=float(r @ r), stationarity=float(np.linalg.norm(gradient)),
        status=int(result.status), message=result.message, nfev=int(result.nfev),
        njev=None if result.njev is None else int(result.njev),
        optimality=float(result.optimality), inner_rank=inner_rank, reverted=reverted))


def als_fit(target, metric, rank, *, outer=4, inner=40, tolerance=1e-10,
            metric_tolerance=1e-10, **unused):
    started = perf_counter()
    original = cbe.lsmr
    solves = []
    def lsmr(*args, **kwargs):
        if inner == 'dense':
            operator, rhs = args
            matrix = operator @ np.eye(operator.shape[1])
            solution = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
            residual = matrix @ solution-rhs
            result = (solution, 2, 1, np.linalg.norm(residual),
                      np.linalg.norm(matrix.conj().T @ residual))
        else:
            result = original(*args, **dict(kwargs, maxiter=inner))
        solves.append(dict(stop=int(result[1]), iterations=int(result[2]),
                           residual=float(result[3]), normal_residual=float(result[4])))
        return result
    with patch.object(cbe, "lsmr", lsmr):
        left, right, loss, count = _PRODUCTION_ALS(
            target, metric, rank, tolerance=tolerance, max_iterations=outer,
            metric_tolerance=metric_tolerance)
    return Fit(left, right, loss, count, dict(seconds=perf_counter()-started,
                                            lsmr=solves))


METHODS = ("baseline", "als40_40", "als4_400", "als40_400",
           "varpro_trf", "varpro_lm", "joint_trf", "grassmann_newton",
           "varpro_trf_small", "grassmann_newton_balanced",
           "varpro_trf_balanced", "varpro_lm_balanced", "varpro_trf_small_balanced")


def solve(target, metric, rank, method, **kwargs):
    if method.endswith('_balanced'):
        started = perf_counter()
        fit = solve(target, metric, rank, method.removesuffix('_balanced'), **kwargs)
        product = fit.left @ fit.right
        u, s, vh = np.linalg.svd(product, full_matrices=False)
        left = u[:, :rank] * np.sqrt(s[:rank])
        right = np.sqrt(s[:rank, None]) * vh[:rank]
        loss = cbe._one_site_factorization_loss(target, left, right, metric)
        return Fit(left, right, loss, fit.iterations,
                   dict(fit.diagnostics, seconds=perf_counter()-started,
                        balance_product_relative_error=float(np.linalg.norm(left@right-product)
                            / max(np.linalg.norm(product), 1e-300))))
    if method == 'varpro_trf_small':
        if target.shape[0] <= target.shape[1]:
            return nonlinear_fit(target, metric, rank, method='varpro_trf', **kwargs)
        permutation = np.arange(target.size).reshape(target.shape).T.ravel()
        fit = nonlinear_fit(target.T, metric.restrict(permutation), rank,
                            method='varpro_trf', **kwargs)
        return Fit(fit.right.T, fit.left.T, fit.loss, fit.iterations,
                   dict(fit.diagnostics, transposed=True))
    if method == "grassmann_newton":
        from .grassmann_compression import newton_fit
        return newton_fit(target, metric, rank, **kwargs)
    if method == "baseline":
        return als_fit(target, metric, rank, **kwargs)
    if method.startswith("als"):
        outer, inner = method[3:].split("_")
        outer, inner = int(outer), int(inner) if inner != 'dense' else 'dense'
        return als_fit(target, metric, rank, outer=outer, inner=inner, **kwargs)
    return nonlinear_fit(target, metric, rank, method=method, **kwargs)

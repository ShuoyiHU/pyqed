"""Dense variable-projection Newton reference in a Grassmann coordinate chart.

The chart X = Q0 + Qperp Z removes factor-basis redundancy. Its Hessian is
the Schur complement of the joint least-squares Hessian, including residual
curvature. This local chart excludes subspaces orthogonal to Q0; this solver
is a local reference, not a globally complete search over subspaces.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .metric_compression_solvers import WeightedProblem, Fit


class GrassmannChart:
    def __init__(self, problem):
        self.p = problem
        q, _ = np.linalg.qr(problem.left, mode='complete')
        self.q0, self.qperp = q[:, :problem.rank], q[:, problem.rank:]
        self.shape = (problem.rows-problem.rank, problem.rank)
        self.xbasis = [self.qperp @ dz for dz in problem.basis(self.shape)]
        self.tbasis = problem.basis(problem.right.shape)
        self.dk = [np.column_stack([problem.weighted(dx @ dt) for dt in self.tbasis])
                   for dx in self.xbasis]
        self.last_z = None
        self.evaluations = 0

    def evaluate(self, z):
        if self.last_z is not None and np.array_equal(z, self.last_z):
            return
        self.left = self.q0 + self.qperp @ self.p.unpack(z, self.shape)
        k = np.column_stack([self.p.weighted(self.left @ dt) for dt in self.tbasis])
        u, s, vh = np.linalg.svd(k, full_matrices=False)
        keep = s > 1e-12*s[0]
        u, s, vh = u[:, keep], s[keep], vh[keep]
        t = (vh.T/s) @ (u.T @ self.p.b)
        self.right = self.p.unpack(t, self.p.right.shape)
        residual = k @ t-self.p.b
        jx = np.column_stack([dk @ t for dk in self.dk])
        cross = jx.T @ k + np.stack([dk.T @ residual for dk in self.dk])
        reduced_cross = cross @ (vh.T/s)
        self.f = float(residual @ residual / 2)
        self.g = jx.T @ residual
        h = jx.T @ jx - reduced_cross @ reduced_cross.T
        self.h = (h+h.T)/2
        self.last_z = z.copy()
        self.evaluations += 1

    def fun(self, z):
        self.evaluate(z)
        return self.f

    def jac(self, z):
        self.evaluate(z)
        return self.g

    def hess(self, z):
        self.evaluate(z)
        return self.h


def newton_fit(target, metric, rank, *, max_nfev=100, tolerance=1e-11,
               metric_tolerance=1e-10):
    if target.shape[0] > target.shape[1]:
        permutation = np.arange(target.size).reshape(target.shape).T.ravel()
        fit = newton_fit(target.T, metric.restrict(permutation), rank,
                         max_nfev=max_nfev, tolerance=tolerance,
                         metric_tolerance=metric_tolerance)
        return Fit(fit.right.T, fit.left.T, fit.loss, fit.iterations,
                   dict(fit.diagnostics, transposed=True))
    started = perf_counter()
    p = WeightedProblem(target, metric, rank, metric_tolerance)
    chart = GrassmannChart(p)
    z0 = p.pack(np.zeros(chart.shape, dtype=p.target.dtype))
    result = minimize(chart.fun, z0, jac=chart.jac, hess=chart.hess,
                      method='trust-exact', options=dict(maxiter=max_nfev, gtol=tolerance))
    chart.evaluate(result.x)
    left, right = chart.left, chart.right
    loss = p.loss(left, right)
    initial_loss = p.loss(p.left, p.right)
    reverted = not np.isfinite(loss) or loss > initial_loss
    if reverted:
        left, right, loss = p.left, p.right, initial_loss
    return Fit(left, right, loss, chart.evaluations, dict(
        seconds=perf_counter()-started, initial_loss=initial_loss,
        status=int(result.status), success=bool(result.success), message=str(result.message),
        nfev=int(result.nfev), njev=int(result.njev),
        optimality=float(np.linalg.norm(result.jac)), reverted=reverted,
        chart_coordinate_norm=float(np.linalg.norm(result.x))))

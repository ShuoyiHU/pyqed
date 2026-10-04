"""Safeguarded physical-norm compression for bilinear LETTA factors.

The nonlinear methods use bounded dense *factor* Jacobians, never a global
wavefunction. The caller supplies a topology-specific metric square root;
ALS uses its forward/adjoint actions through matrix-free linear solves. Shared-index and charge blocks restrict the factor
charts; the objective always uses the full metric, including block couplings.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class MetricCompressionOptions:
    """Common options for two-site truncation and CBE post-expansion trimming.

    max_iterations is a residual-evaluation budget for least squares and a
    trust-region iteration budget for Newton. ALS uses the calling solver's
    existing budget unless als_max_iterations is supplied.
    """
    solver: str = "als"
    max_iterations: int = 100
    tolerance: float = 1e-10
    pinv_tolerance: float = 1e-12
    max_workspace_mb: float = 128.0
    max_factor_norm_growth: float = 100.0
    als_max_iterations: int | None = None
    lsmr_max_iterations: int | None = None

    def __post_init__(self):
        if self.solver not in {"als", "variable-projection", "joint-ls", "grassmann-newton"}:
            raise ValueError(f"Unknown compression solver: {self.solver!r}")
        for name in ("max_iterations", "als_max_iterations", "lsmr_max_iterations"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("tolerance", "pinv_tolerance", "max_workspace_mb", "max_factor_norm_growth"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.max_factor_norm_growth < 1:
            raise ValueError("max_factor_norm_growth must be at least one.")
        if self.tolerance <= np.finfo(float).eps or self.pinv_tolerance >= 1:
            raise ValueError("Use tolerance > machine epsilon and pinv_tolerance < 1.")


@dataclass(frozen=True)
class CompressionResult:
    left: np.ndarray
    right: np.ndarray
    loss: float
    iterations: int
    diagnostics: dict


def factor_blocks(left, right, layout=None, left_indices=None, right_indices=None):
    """Flat-index matrices for independent shared-physical/charge gauge blocks."""
    li = np.arange(left.size).reshape(left.shape)
    ri = np.arange(right.size).reshape(right.shape)
    lm, rm = np.ones(left.size, bool), np.ones(right.size, bool)
    if left_indices is not None:
        lm[:] = False
        lm[left_indices] = True
    if right_indices is not None:
        rm[:] = False
        rm[right_indices] = True
    shared = () if layout is None else layout.shared
    configurations = np.ndindex(*((layout.physical_dim,) * len(shared))) if shared else [()]
    blocks = []
    for configuration in configurations:
        ls, rs = [slice(None)] * left.ndim, [slice(None)] * right.ndim
        for site, value in zip(shared, configuration):
            ls[1 + layout.left_neighborhood.index(site)] = value
            rs[1 + layout.right_neighborhood.index(site)] = value
        a = li[tuple(ls)].reshape(-1, left.shape[-1])
        b = ri[tuple(rs)].reshape(right.shape[0], -1)
        groups = {}
        for bond in range(left.shape[-1]):
            key = (lm[a[:, bond]].tobytes(), rm[b[bond]].tobytes())
            groups.setdefault(key, []).append(bond)
        for bonds in groups.values():
            rows = np.flatnonzero(lm[a[:, bonds[0]]])
            cols = np.flatnonzero(rm[b[bonds[0]]])
            if rows.size and cols.size:
                blocks.append((a[np.ix_(rows, bonds)], b[np.ix_(bonds, cols)]))
    return blocks


def _balance(left, right, blocks):
    """Balance each admissible product using only a bond-sized core SVD."""
    left, right = left.copy(), right.copy()
    for li, ri in blocks:
        a, b = left.ravel()[li], right.ravel()[ri]
        ql, rl = np.linalg.qr(a, mode="reduced")
        qr, rr = np.linalg.qr(b.conj().T, mode="reduced")
        u, s, vh = np.linalg.svd(rl @ rr.conj().T, full_matrices=False)
        k = len(s)
        aa, bb = np.zeros_like(a), np.zeros_like(b)
        aa[:, :k] = (ql @ u) * np.sqrt(s)
        bb[:k] = np.sqrt(s)[:, None] * (vh @ qr.conj().T)
        left.ravel()[li], right.ravel()[ri] = aa, bb
    return left, right


class _Coordinates:
    def __init__(self, shape, dtype, indices=None):
        self.shape, self.dtype = shape, dtype
        self.indices = np.arange(np.prod(shape)) if indices is None else np.asarray(indices)
        self.complex = np.issubdtype(dtype, np.complexfloating)
        self.size = len(self.indices) * (2 if self.complex else 1)

    def pack(self, array):
        v = np.asarray(array).ravel()[self.indices]
        return np.r_[v.real, v.imag] if self.complex else v.real.copy()

    def unpack(self, x):
        a = np.zeros(self.shape, dtype=self.dtype)
        n = len(self.indices)
        a.ravel()[self.indices] = x[:n] + 1j*x[n:] if self.complex else x
        return a

    def basis(self):
        for j in range(self.size):
            v = np.zeros(self.size)
            v[j] = 1
            yield self.unpack(v)


def _chart_size(blocks, complex_data):
    return sum(max(0, a.shape[0]-a.shape[1])*min(a.shape) for a, _ in blocks) * (2 if complex_data else 1)


def _chart(left, blocks):
    base = np.zeros_like(left)
    directions = []
    for indices, _ in blocks:
        a = left.ravel()[indices]
        q, _ = np.linalg.qr(a, mode="complete")
        k = min(a.shape)
        block = np.zeros_like(a)
        block[:, :k] = q[:, :k]
        base.ravel()[indices] = block
        for imaginary in range(2 if np.iscomplexobj(left) else 1):
            for row in range(k, a.shape[0]):
                for column in range(k):
                    dx = np.zeros_like(left)
                    dx.ravel()[indices[:, column]] = q[:, row] * (1j if imaginary else 1)
                    directions.append(dx)
    return base, directions


class _ProjectedProblem:
    """Exact variable-projection derivatives in real coordinates.

    Constant-rank pseudoinverse calculus is used. Rank changes are recorded;
    they invalidate a smooth local Hessian interpretation at the transition.
    """
    def __init__(self, base, directions, right_coordinates, weighted_merge, b, observe, rcond):
        self.base, self.directions = base, directions
        self.coordinates, self.merge = right_coordinates, weighted_merge
        self.b, self.observe, self.rcond = b, observe, rcond
        self.tbasis = list(right_coordinates.basis())
        self.dk = [np.column_stack([weighted_merge(dx, dt) for dt in self.tbasis]) for dx in directions]
        self.last = None
        self.evaluations = 0
        self.ranks = set()

    def evaluate(self, z):
        if self.last is not None and np.array_equal(self.last, z):
            return
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("nonfinite chart coordinates")
        left = self.base.copy()
        for coefficient, dx in zip(z, self.directions):
            left += coefficient * dx
        k = np.column_stack([self.merge(left, dt) for dt in self.tbasis])
        if not np.all(np.isfinite(k)):
            raise FloatingPointError("nonfinite projected design matrix")
        u, s, vh = np.linalg.svd(k, full_matrices=False)
        keep = s > self.rcond*s[0] if len(s) and s[0] else np.zeros(len(s), bool)
        u, s, vh = u[:, keep], s[keep], vh[keep]
        pinv = (vh.T/s) @ u.T
        t = pinv @ self.b
        self.left, self.right = left, self.coordinates.unpack(t)
        self.r = k @ t - self.b
        self.observe(self.left, self.right)
        p = len(self.dk)
        jx = np.column_stack([dk @ t for dk in self.dk]) if p else np.empty((len(self.b), 0))
        cross_residual = np.stack([dk.T @ self.r for dk in self.dk]) if p else np.empty((0, len(t)))
        self.j = jx - u @ (u.T @ jx) - pinv.T @ cross_residual.T
        cross = jx.T @ k + cross_residual
        reduced = cross @ (vh.T/s)
        h = jx.T @ jx - reduced @ reduced.T
        self.h = (h+h.T)/2
        self.g = jx.T @ self.r
        if not all(np.all(np.isfinite(v)) for v in (self.r, self.j, self.h, self.g)):
            raise FloatingPointError("nonfinite projected derivatives")
        self.f = float(self.r @ self.r / 2)
        self.last = z.copy()
        self.evaluations += 1
        self.ranks.add(len(s))

    def residual(self, z):
        self.evaluate(z)
        return self.r

    def jacobian(self, z):
        self.evaluate(z)
        return self.j

    def fun(self, z):
        self.evaluate(z)
        return self.f

    def gradient(self, z):
        self.evaluate(z)
        return self.g

    def hessian(self, z):
        self.evaluate(z)
        return self.h


def compress_factors(target, metric, left, right, *, options, fallback,
                     metric_tolerance=1e-12, layout=None,
                     left_indices=None, right_indices=None, gauge_blocks=None, square_root=None):
    """Fit two factors; fallback returns (left, right, loss, iterations).

    Dense workspace is estimated *before* building charts or Jacobians.
    The best iterate in the original physical metric is retained, including
    the incoming factors, even if the optimizer exhausts its budget or fails.
    """
    if not isinstance(options, MetricCompressionOptions):
        raise TypeError("compression must be MetricCompressionOptions.")
    dtype = np.dtype(np.result_type(target, metric.dtype, left, right, float))
    target = np.asarray(target, dtype=dtype)
    left, right = np.array(left, dtype=dtype), np.array(right, dtype=dtype)
    merge = (lambda a, b: a @ b) if layout is None else layout.merge
    physical_merge = merge
    def loss(a, b):
        d = (physical_merge(a, b)-target).ravel()
        value = float(np.real(np.vdot(d, metric @ d)))
        return max(0., value) if np.isfinite(value) and value >= -1e-12 else np.inf
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("compression target and initial factors must be finite.")
    initial_loss = loss(left, right)
    best = [left.copy(), right.copy(), initial_loss]
    factor_scale = max(1., np.linalg.norm(left)*np.linalg.norm(right))
    target_norm = float(np.real(np.vdot(target.ravel(), metric @ target.ravel())))
    rejected_candidates = [0]
    def observe(a, b):
        if np.linalg.norm(a)*np.linalg.norm(b) > options.max_factor_norm_growth**2*factor_scale:
            rejected_candidates[0] += 1
            return
        merged = physical_merge(a, b).ravel()
        norm = float(np.real(np.vdot(merged, metric @ merged)))
        if target_norm > 0 and (not np.isfinite(norm) or norm <= np.finfo(float).tiny):
            rejected_candidates[0] += 1
            return
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            value = loss(a, b)
            if np.isfinite(value) and value <= best[2]:
                best[:] = a.copy(), b.copy(), value
    diagnostics = dict(requested_solver=options.solver, used_solver=options.solver,
                       initial_loss=initial_loss, fallback_reason=None)
    def als(reason=None):
        a, b, value, count = fallback()
        if reason is None:
            diagnostics.update(status="als", final_loss=value)
            return CompressionResult(a, b, value, count, diagnostics)
        observe(a, b)
        diagnostics.update(used_solver="als", fallback_reason=reason,
                           status="fallback" if reason else "als", final_loss=best[2])
        return CompressionResult(*best, count, diagnostics)
    if options.solver == "als":
        return als()
    if hasattr(metric, "blocks") and all(not np.any(block) for block in metric.blocks):
        diagnostics.update(status="zero metric", final_loss=0., optimizer_success=True)
        return CompressionResult(left, right, 0., 0, diagnostics)
    blocks = (factor_blocks(left, right, layout, left_indices, right_indices)
              if gauge_blocks is None else gauge_blocks)
    lc, rc = _Coordinates(left.shape, dtype, left_indices), _Coordinates(right.shape, dtype, right_indices)
    complex_data = np.issubdtype(dtype, np.complexfloating)
    reverse = [(b.T, a.T) for a, b in blocks]
    flip = options.solver != "joint-ls" and _chart_size(reverse, complex_data) < _chart_size(blocks, complex_data)
    # Swapping factor roles uses a transpose of each gauge block, not a
    # permutation/approximation of the metric or the physical merge map.
    if flip:
        left, right, lc, rc, blocks = right, left, rc, lc, reverse
        original_merge, original_observe = merge, observe
        merge = lambda a, b: original_merge(b, a)
        observe = lambda a, b: original_observe(b, a)
    n = target.size*(2 if complex_data else 1)
    p, q = _chart_size(blocks, complex_data), rc.size
    joint = lc.size+rc.size
    # Includes derivative-design tensors, factorizations, trust-region work,
    # factor bases and complete Qs. Conservative estimate, not process RSS.
    doubles = (n*joint + joint**2 if options.solver == "joint-ls" else
               n*q*(p+1) + n*p + p*p + q*q + p*q)
    doubles += (left.size+right.size)*(joint+p)* (2 if complex_data else 1)
    doubles += sum(a.shape[0]**2 for a, _ in blocks)*(2 if complex_data else 1)
    workspace = 8*8*doubles
    diagnostics.update(estimated_workspace_bytes=workspace, factor_side="right" if flip else "left",
                       nonlinear_parameters=joint if options.solver == "joint-ls" else p)
    if workspace > options.max_workspace_mb*1024**2:
        # Restore physical orientation for fallback's observe.
        if flip:
            observe = original_observe
        return als("estimated nonlinear workspace exceeds max_workspace_mb")
    from ._letta_two_site_opt.truncation import _MetricSquareRoot
    try:
        root = _MetricSquareRoot(metric, metric_tolerance) if square_root is None else square_root
        def weighted(a):
            v = root.apply(np.asarray(a).ravel())
            if not np.all(np.isfinite(v)):
                raise FloatingPointError("nonfinite weighted residual/design")
            return np.r_[v.real, v.imag] if complex_data else v.real
        b = weighted(target)
        weighted_merge = lambda a, b: weighted(merge(a, b))
        if options.solver == "joint-ls":
            left, right = _balance(left, right, blocks)
            xb, tb = list(lc.basis()), list(rc.basis())
            def factors(x):
                return lc.unpack(x[:lc.size]), rc.unpack(x[lc.size:])
            def residual(x):
                a, c = factors(x)
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("nonfinite joint coordinates")
                observe(a, c)
                return weighted_merge(a, c)-b
            def jacobian(x):
                a, c = factors(x)
                return np.column_stack([weighted_merge(dx, c) for dx in xb] +
                                       [weighted_merge(a, dt) for dt in tb])
            result = least_squares(residual, np.r_[lc.pack(left), rc.pack(right)], jac=jacobian,
                                   max_nfev=options.max_iterations, ftol=options.tolerance,
                                   xtol=options.tolerance, gtol=options.tolerance, method="trf")
            count = result.nfev
            optimality = result.optimality
        else:
            base, directions = _chart(left, blocks)
            problem = _ProjectedProblem(base, directions, rc, weighted_merge, b, observe, options.pinv_tolerance)
            z = np.zeros(len(directions))
            if not len(z):
                problem.evaluate(z)
                result = None
                optimality = 0.
            elif options.solver == "variable-projection":
                result = least_squares(problem.residual, z, jac=problem.jacobian,
                                       max_nfev=options.max_iterations, ftol=options.tolerance,
                                       xtol=options.tolerance, gtol=options.tolerance, method="trf")
                optimality = result.optimality
            else:
                result = minimize(problem.fun, z, jac=problem.gradient, hess=problem.hessian,
                                  method="trust-exact", options=dict(maxiter=options.max_iterations,
                                                                    gtol=options.tolerance))
                optimality = np.linalg.norm(result.jac, ord=np.inf)
            count = problem.evaluations
            diagnostics.update(inner_ranks=sorted(problem.ranks),
                               chart_coordinate_norm=0. if result is None else float(np.linalg.norm(result.x)))
        diagnostics.update(status="linear" if result is None else str(result.message),
                           optimizer_success=True if result is None else bool(result.success),
                           optimizer_optimality=float(optimality), evaluations=int(count))
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        if flip:
            observe = original_observe
        return als(f"{type(error).__name__}: {error}")
    # Always balance in the original factor orientation and metric. Keep the
    # unbalanced best iterate if rounding would measurably spoil compression.
    if flip:
        blocks = [(b.T, a.T) for a, b in blocks]
        observe = original_observe
    try:
        a, c = _balance(best[0], best[1], blocks)
        balanced_loss = loss(a, c)
    except np.linalg.LinAlgError:
        a, c, balanced_loss = best[0], best[1], np.inf
    slack = 100*np.finfo(float).eps*max(1., initial_loss)
    balanced_norm = float(np.real(np.vdot(physical_merge(a, c).ravel(), metric @ physical_merge(a, c).ravel())))
    balanced = (np.isfinite(balanced_loss) and balanced_loss <= best[2]+slack
                and (target_norm <= 0 or balanced_norm > np.finfo(float).tiny))
    if balanced:
        best[:] = a, c, balanced_loss
    diagnostics.update(final_loss=best[2], balanced=balanced,
                       rejected_candidates=rejected_candidates[0],
                       supported_loss=float(np.linalg.norm(weighted(physical_merge(best[0], best[1])-target))**2))
    return CompressionResult(*best, int(count), diagnostics)

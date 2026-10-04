"""Reduced-space adapters for the common physical-metric compression solvers.

The square root factors boundary multiplicity Grams, not a full pair metric.
Admissible factor gauge blocks keep charge, spin and shared scalar ties fixed.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .._letta_compression import compress_factors
from .._letta_one_site_opt.contractions import _equilibrated_metric_factors
from .._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
from .._letta_one_site_opt.reduced_norm import ReducedNormChain
from .._letta_one_site_opt.reduced_symmetry import _sector_irrep


class ReducedPairMetricRoot:
    """S with S†S=N on supported boundary multiplicity coordinates."""

    def __init__(self, problem, state, tolerance):
        sites = problem.frontier.to_mps(state)
        chain = ReducedNormChain.build(sites)
        i = problem.left_site
        scale = np.exp(.5*(chain.left_log_scales[i]+chain.right_log_scales[i+2]))
        left = {q: _equilibrated_metric_factors(g, tolerance)
                for q, g in chain.left[i].items()}
        right = {q: _equilibrated_metric_factors(g, tolerance)
                 for q, g in chain.right[i+2].items()}
        self.input_layout = problem.layout
        self.factors = {}
        self.inverse_factors = {}
        shapes = {}
        for key, shape in problem.layout.shapes.items():
            wl, l = left[key[0]]
            wr, r = right[key[-1]]
            self.inverse_factors[key] = (wl, wr, scale*np.sqrt(_sector_irrep(key[-1]).dim))
            self.factors[key] = (l, r, scale*np.sqrt(_sector_irrep(key[-1]).dim))
            shapes[key] = (l.shape[0], shape[1], shape[2], r.shape[0])
        self.output_layout = _BlockVectorLayout(shapes)
        self.size = self.output_layout.size
        if not self.size:
            raise ValueError('reduced pair metric has no supported directions')

    def apply(self, vector):
        blocks = self.input_layout.unpack(vector)
        return self.output_layout.pack({key: scale*np.einsum('al,br,lpqr->apqb',
            l, r, blocks[key], optimize=True) for key, (l, r, scale) in self.factors.items()})

    def adjoint(self, vector):
        blocks = self.output_layout.unpack(vector)
        return self.input_layout.pack({key: scale*np.einsum('al,br,apqb->lpqr',
            l.conj(), r.conj(), blocks[key], optimize=True)
            for key, (l, r, scale) in self.factors.items()})

    def unwhiten(self, vector):
        """Map supported orthonormal coordinates back to pair coefficients."""
        blocks = self.output_layout.unpack(vector)
        return self.input_layout.pack({key: np.einsum('la,rb,apqb->lpqr',
            l, r, blocks[key], optimize=True)/scale
            for key, (l, r, scale) in self.inverse_factors.items()})

    def unwhiten_adjoint(self, vector):
        blocks = self.input_layout.unpack(vector)
        return self.output_layout.pack({key: np.einsum('la,rb,lpqr->apqb',
            l.conj(), r.conj(), blocks[key], optimize=True)/scale
            for key, (l, r, scale) in self.inverse_factors.items()})

    def inverse_action(self, vector):
        """Apply a supported inverse: N inverse_action(N x) = N x.

        Equilibrating boundary coordinates before the rank decision preserves
        small but independent physical directions. This is a generalized inverse,
        not necessarily the Euclidean Moore-Penrose inverse in the raw gauge.
        """
        return self.unwhiten(self.unwhiten_adjoint(vector))


def reduced_factor_blocks(state, site, left_embedding, right_embedding, retained):
    """Packed index matrices for each (middle irrep, shared-label assignment)."""
    shared = sorted(set(state.site_neighborhood(site)) & set(state.site_neighborhood(site+1)))
    labels = {(x.sector, x.copy): k for k, x in enumerate(state.physical_basis.reduced_states)}
    groups = []
    for side, embedding in enumerate((left_embedding, right_embedding)):
        current = {}
        neighborhood = state.site_neighborhood(site+side)
        for key, shape in embedding.source_layout.shapes.items():
            middle = key[2] if side == 0 else key[0]
            rank = retained.get(middle, 0)
            if not rank:
                continue
            offset = embedding.source_layout.offsets[key][0]
            indices = offset+np.arange(np.prod(shape)).reshape(shape)
            for physical in np.ndindex(shape[1:-1]):
                values = (labels[(key[1], physical[0])],)+physical[1:]
                assignment = dict(zip(neighborhood, values))
                group = (middle, tuple(assignment[x] for x in shared))
                section = ((slice(None),)+physical+(slice(0, rank),) if side == 0
                           else (slice(0, rank),)+physical+(slice(None),))
                current.setdefault(group, []).append(indices[section])
        groups.append(current)
    # A scalar tie assignment may be forbidden by the partner's owned
    # physical irrep. Such unmatched source entries represent the zero state;
    # only paired groups carry a factor gauge.
    return [(np.concatenate(groups[0][key], axis=0), np.concatenate(groups[1][key], axis=1))
            for key in groups[0] if key in groups[1]]


class _PairMetric:
    def __init__(self, problem, dtype):
        self.problem, self.dtype = problem, dtype

    def __matmul__(self, vector):
        return self.problem.apply_metric(vector)


def compress_reduced_pair(target, problem, state, left, right, retained, *, options,
                          metric_tolerance=1e-12, als_max_iterations=100):
    """Fit reduced source factors with explicit outer and inner budgets.

    All algorithms share the same native metric and legal factor gauge blocks.
    Budget exhaustion preserves the best iterate and is reported, not called
    convergence. There is no magnetic or determinant-space reconstruction.
    """
    from .reduced_solver import (_pair_vector_from_sources, _active_source_indices,
        _expanded_source_blocks, _left_source_adjoint, _right_source_adjoint)

    i = problem.left_site
    le, re = (problem.frontier.site_embedding(state, k) for k in (i, i+1))
    li, ri = _active_source_indices(le, retained, 'left'), _active_source_indices(re, retained, 'right')
    dtype = np.result_type(target, left, right, complex)
    left, right = np.asarray(left, dtype=dtype).copy(), np.asarray(right, dtype=dtype).copy()
    left[np.setdiff1d(np.arange(left.size), li)] = 0
    right[np.setdiff1d(np.arange(right.size), ri)] = 0
    merge = lambda a, b: _pair_vector_from_sources(problem.layout, le, re, a, b)
    root = ReducedPairMetricRoot(problem, state, metric_tolerance)
    groups = reduced_factor_blocks(state, i, le, re, retained)
    metric = _PairMetric(problem, dtype)
    def adjoint(side, a, b, vector):
        if side == 0:
            return _left_source_adjoint(problem.layout, vector, _expanded_source_blocks(re, b), le)[li]
        return _right_source_adjoint(problem.layout, _expanded_source_blocks(le, a), vector, re)[ri]

    return fit_metric_factors(target, metric, left, right, merge=merge, root=root,
        adjoint=adjoint, left_indices=li, right_indices=ri, gauge_blocks=groups,
        options=options, metric_tolerance=metric_tolerance,
        als_max_iterations=als_max_iterations)


def fit_metric_factors(target, metric, left, right, *, merge, root, adjoint,
                       left_indices, right_indices, gauge_blocks, options,
                       metric_tolerance=1e-12, als_max_iterations=100):
    """Shared ALS/nonlinear fitting loop for open and cyclic reduced metrics.

    ``adjoint(side, a, b, v)`` returns only the active side coordinates.
    Topology is entirely in merge/adjoint/root; budgets and loss guards match.
    """
    li, ri, groups = left_indices, right_indices, gauge_blocks
    dtype = np.result_type(target, left, right, complex)
    rhs = root.apply(target)
    reports = []
    als_info = {}

    def loss(a, b):
        error = merge(a, b)-target
        value = float(np.vdot(error, metric@error).real)
        if not np.isfinite(value) or value < -1e-10:
            raise FloatingPointError('invalid reduced compression loss')
        return max(0., value)

    def solve(side, a, b):
        indices, old = (li, a) if side == 0 else (ri, b)
        def unpack(v):
            out = np.zeros_like(old)
            out[indices] = v
            return out
        def forward(v):
            return root.apply(merge(unpack(v), b) if side == 0 else merge(a, unpack(v)))
        operator = LinearOperator((root.size, len(indices)), matvec=forward,
            rmatvec=lambda v: adjoint(side, a, b, root.adjoint(v)), dtype=dtype)
        limit = options.lsmr_max_iterations or max(50, min(1000, 5*len(indices)))
        answer = lsmr(operator, rhs, x0=old[indices], atol=options.tolerance,
                      btol=options.tolerance, maxiter=limit)
        reports.append(dict(side='left' if side == 0 else 'right', solver='lsmr',
            stop_code=int(answer[1]), iterations=int(answer[2]), max_iterations=limit,
            residual_norm=float(answer[3]), normal_residual_norm=float(answer[4]),
            converged=answer[1] in (0, 1, 2, 4, 5)))
        if not np.all(np.isfinite(answer[0])):
            raise FloatingPointError('nonfinite reduced least-squares solution')
        return unpack(answer[0])

    def fallback():
        a, b = left.copy(), right.copy()
        value = loss(a, b)
        limit = options.als_max_iterations or als_max_iterations
        status, stationarity = 'maximum ALS rounds reached', np.inf
        for iteration in range(1, limit+1):
            old_value = value
            aa = solve(0, a, b)
            candidate = loss(aa, b)
            if candidate <= value:
                a, value = aa, candidate
            bb = solve(1, a, b)
            candidate = loss(a, bb)
            if candidate <= value:
                b, value = bb, candidate
            residual = metric@(merge(a, b)-target)
            target_metric = metric@target
            stationarity = max(np.linalg.norm(adjoint(side, a, b, residual)) /
                max(np.linalg.norm(adjoint(side, a, b, target_metric)), np.finfo(float).tiny)
                for side in (0, 1))
            if stationarity <= options.tolerance and all(r['converged'] for r in reports[-2:]):
                status = 'converged'
                break
            if old_value-value <= np.finfo(float).eps*max(1., old_value):
                status = 'stagnated before stationarity'
                break
        als_info.update(status=status, optimizer_success=status == 'converged',
                        stationarity=float(stationarity), linear_solves=reports,
                        als_max_iterations=limit)
        return a, b, value, iteration

    result = compress_factors(target, metric, left, right, options=options, fallback=fallback,
        metric_tolerance=metric_tolerance, layout=SimpleNamespace(merge=merge),
        left_indices=li, right_indices=ri, gauge_blocks=groups, square_root=root)
    if result.diagnostics['used_solver'] == 'als':
        result = replace(result, diagnostics={**result.diagnostics, **als_info})
    return result

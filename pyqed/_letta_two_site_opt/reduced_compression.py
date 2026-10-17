"""Reduced-space adapters for the common physical-metric compression solvers.

The square root factors boundary multiplicity Grams, not a full pair metric.
Admissible factor gauge blocks keep charge, spin and shared scalar ties fixed.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .._letta_compression import CompressionResult, compress_factors
from .._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
from .._letta_one_site_opt.reduced_symmetry import _sector_irrep


def _overlap_factors(factor, tolerance):
    """Equilibrate a QR factor and select inverse support without squaring it."""
    diagonal = np.linalg.norm(factor, axis=0)
    indices = np.flatnonzero(diagonal > 0.)
    scaled = factor[:,indices]/diagonal[indices]
    unitary, singular, vh = np.linalg.svd(scaled, full_matrices=False)
    cutoff = max(float(tolerance),np.finfo(float).eps*factor.shape[1])
    keep = singular > np.sqrt(cutoff)*singular.max(initial=0.)
    singular, vh = singular[keep], vh[keep]
    inverse = np.zeros((factor.shape[1],len(singular)),dtype=factor.dtype)
    root = np.zeros((len(singular),factor.shape[1]),dtype=factor.dtype)
    inverse[indices] = vh.conj().T/diagonal[indices,None]/singular
    root[:,indices] = singular[:,None]*vh*diagonal[indices]
    return inverse, root, unitary[:, keep]


class ReducedPairMetricRoot:
    """Reduced QR overlap maps and SVD inverse-support coordinates.

    Adapts QR/SVD least squares from Trefethen and Bau, Numerical Linear
    Algebra (SIAM, 1997), doi:10.1137/1.9780898719574. No determinant frame
    is formed. physical_apply retains the complete QR map for physical loss
    and norm checks; apply/unwhiten use the configured equilibrated SVD support.
    This is a numerical adaptation, not a globally optimal compression method.

    A declared sector absent from a boundary environment has exactly zero
    physical support. Keep its input coordinates but give it zero output
    rows, consistently with ReducedNormChain.pair_action. This is not an
    additional numerical truncation.
    """

    def __init__(self, problem, state, tolerance):
        from .._letta_one_site_opt.reduced_coordinates import _canonical_sites
        from .._letta_one_site_opt.frontier_contraction import state_signature
        i = problem.left_site
        cached = getattr(problem,'canonical_data',None)
        if cached is not None and cached[0] == state_signature(state):
            self.canonical_sites, full_left, full_right, self.environment = cached[1:]
        else:
            sites = problem.frontier.to_mps(state, compact=True)
            self.canonical_sites, full_left, full_right = _canonical_sites(sites, i, last_center=i+1)
        left = {q: _overlap_factors(r, tolerance) for q, r in full_left.items()}
        right = {q: _overlap_factors(r, tolerance) for q, r in full_right.items()}
        scale = 1.
        self.physical_factors = {}
        self.support_factors = {}
        self.input_layout = problem.layout
        self.factors = {}
        self.inverse_factors = {}
        shapes = {}
        for key, shape in problem.layout.shapes.items():
            if key[0] not in left:
                left[key[0]] = (np.zeros((shape[0], 0)), np.zeros((0, shape[0])), np.zeros((0, 0)))
            if key[-1] not in right:
                right[key[-1]] = (np.zeros((shape[-1], 0)), np.zeros((0, shape[-1])), np.zeros((0, 0)))
            self.physical_factors[key] = (full_left.get(key[0], np.zeros((0,shape[0]))),
                full_right.get(key[-1], np.zeros((0,shape[-1]))), np.sqrt(_sector_irrep(key[-1]).dim))
            wl, l, ul = left[key[0]]
            wr, r, ur = right[key[-1]]
            self.support_factors[key] = (ul, ur)
            self.inverse_factors[key] = (wl, wr, scale*np.sqrt(_sector_irrep(key[-1]).dim))
            self.factors[key] = (l, r, scale*np.sqrt(_sector_irrep(key[-1]).dim))
            shapes[key] = (l.shape[0], shape[1], shape[2], r.shape[0])
        self.physical_layout = _BlockVectorLayout({key:(l.shape[0],problem.layout.shapes[key][1],
            problem.layout.shapes[key][2],r.shape[0]) for key,(l,r,_) in self.physical_factors.items()})
        self.output_layout = _BlockVectorLayout(shapes)
        self.size = self.output_layout.size
        self._compiled_maps = {}
        if not self.size:
            raise ValueError('reduced pair metric has no supported directions')

    def restrict_middle_sector(self, sector):
        """Restrict an orthogonal middle-irrep block without rebuilding QR maps."""
        from copy import copy
        result = copy(self)
        for name in ('physical_factors', 'support_factors', 'factors', 'inverse_factors'):
            setattr(result,name,{k:v for k,v in getattr(self,name).items() if k[2]==sector})
        for name in ('input_layout', 'output_layout', 'physical_layout'):
            setattr(result,name,_BlockVectorLayout({k:v for k,v in getattr(self,name).shapes.items()
                                                   if k[2]==sector}))
        result.size=result.output_layout.size
        result._compiled_maps = {}
        return result

    def _contract_map(self, name, vector, source, target, factors, *, adjoint=False, inverse=False):
        from .pair_metric import PairMetricMap
        if name not in self._compiled_maps:
            prepared = {}
            for key, values in factors.items():
                left, right = values[:2]
                scale = values[2] if len(values) == 3 else 1.
                if adjoint:
                    left, right = left.conj().T, right.conj().T
                prepared[key] = (left, right, 1./scale if inverse else scale)
            self._compiled_maps[name] = PairMetricMap(source, target, prepared)
        return self._compiled_maps[name](vector)

    def project_physical(self, vector):
        """Project QR coordinates onto retained SVD support without inverse factors."""
        return self._contract_map('project', vector, self.physical_layout,
            self.output_layout, self.support_factors, adjoint=True)

    def embed_physical(self, vector):
        return self._contract_map('embed', vector, self.output_layout,
            self.physical_layout, self.support_factors)

    def physical_apply(self, vector):
        """Full QR norm map, without the inverse-support SVD cutoff."""
        return self._contract_map('physical', vector, self.input_layout,
            self.physical_layout, self.physical_factors)

    def physical_adjoint(self, vector):
        return self._contract_map('physical_adjoint', vector, self.physical_layout,
            self.input_layout, self.physical_factors, adjoint=True)

    def apply(self, vector):
        return self._contract_map('apply', vector, self.input_layout,
            self.output_layout, self.factors)

    def adjoint(self, vector):
        return self._contract_map('adjoint', vector, self.output_layout,
            self.input_layout, self.factors, adjoint=True)

    def unwhiten(self, vector):
        """Map supported orthonormal coordinates back to pair coefficients."""
        return self._contract_map('unwhiten', vector, self.output_layout,
            self.input_layout, self.inverse_factors, inverse=True)

    def unwhiten_adjoint(self, vector):
        return self._contract_map('unwhiten_adjoint', vector, self.input_layout,
            self.output_layout, self.inverse_factors, adjoint=True, inverse=True)

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
    def __init__(self, problem, dtype, *, root=None):
        self.problem, self.dtype, self.root = problem, dtype, root

    def __matmul__(self, vector):
        if self.root is not None:
            return self.root.physical_adjoint(self.root.physical_apply(vector))
        return self.problem.apply_metric(vector)


def _factor_coordinates(state, problem, root, le, re, left, right, retained, side, options, *, sites=None):
    """Isometric physical map for one factor with its partner held fixed."""
    from .._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates
    from .reduced_solver import _active_source_indices
    site = problem.left_site
    # Only the two active tensors change during ALS. Reuse the expanded outer
    # sites, while rebuilding the partner-dependent QR coordinates each time.
    current=list(problem.frontier.to_mps(state, compact=True) if sites is None else sites)
    for index,embedding,vector in ((site,le,left),(site+1,re,right)):
        from .._letta_one_site_opt.frontier_contraction import shallow_site, scatter_blocks
        current[index]=shallow_site(current[index])
        current[index].data=scatter_blocks(embedding,embedding.unpack_source(vector))
    embedding = le if side == 0 else re
    coordinates = ReducedLocalCoordinates(current, embedding, site+side, retain_outer=False)
    coordinates.restrict(_active_source_indices(embedding, retained, 'left' if side == 0 else 'right'))
    coordinates.prepare(0., singular_tolerance=options.pinv_tolerance,
                        max_workspace_mb=options.max_workspace_mb)
    forward,pullback=_physical_factor_map(coordinates,root,site,side)
    return coordinates,forward,pullback


def _physical_factor_map(coordinates,root,site,side):
    partner = coordinates.sites[site+1 if side == 0 else site].data

    def apply(vector):
        center = coordinates.orthogonal_expand(vector)
        blocks = {}
        for key, shape in root.physical_layout.shapes.items():
            left_key, right_key = key[:3], key[2:]
            active_key = left_key if side == 0 else right_key
            if active_key not in center:
                blocks[key] = np.zeros(shape, dtype=complex)
            elif side == 0:
                weight = np.sqrt(_sector_irrep(key[-1]).dim/_sector_irrep(key[2]).dim)
                blocks[key] = weight*np.einsum('lpm,mqr->lpqr', center[left_key], partner[right_key])
            else:
                blocks[key] = np.einsum('lpm,mqr->lpqr', partner[left_key], center[right_key])
        return root.physical_layout.pack(blocks)

    def adjoint(vector):
        blocks = root.physical_layout.unpack(vector)
        center = coordinates.orthogonal_expand(np.zeros(coordinates.rank))
        for key, block in blocks.items():
            left_key, right_key = key[:3], key[2:]
            active_key = left_key if side == 0 else right_key
            if active_key not in center:
                continue
            if side == 0:
                weight = np.sqrt(_sector_irrep(key[-1]).dim/_sector_irrep(key[2]).dim)
                center[left_key] += weight*np.einsum('lpqr,mqr->lpm', block, partner[right_key].conj())
            else:
                center[right_key] += np.einsum('lpm,lpqr->mqr', partner[left_key].conj(), block)
        return coordinates.orthogonal_adjoint(center)

    return apply, adjoint


def compress_reduced_pair(target, problem, state, left, right, retained, *, options,
                          metric_tolerance=1e-12, als_max_iterations=100, root=None):
    """Fit reduced source factors with explicit outer and inner budgets.

    All algorithms share the same native metric and legal factor gauge blocks.
    Budget exhaustion preserves the best iterate and is reported, not called
    convergence. If the retained sectors already match the target to the
    physical relative tolerance, the omitted-sector loss certifies the global
    optimum within this fixed allocation and iterative fitting is unnecessary.
    This certificate does not optimize the allocation or the energy. There is
    no magnetic or determinant-space reconstruction.

    Open-boundary ALS uses reduced QR/SVD factor coordinates to solve each
    linear least-squares subproblem directly, adapting Trefethen and Bau,
    Numerical Linear Algebra (SIAM, 1997), doi:10.1137/1.9780898719574.
    This is a LETTA adaptation, not a global low-rank optimization guarantee.
    The configured pseudoinverse cutoff acts on column-equilibrated singular
    values. Oversized factorizations or failed physical stationarity checks
    fall back to LSMR; both paths minimize the full physical residual.
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
    root = root if root is not None else ReducedPairMetricRoot(problem, state, metric_tolerance)
    physical_target = root.physical_layout.unpack(root.physical_apply(target))
    physical_error = root.physical_layout.unpack(root.physical_apply(merge(left, right)-target))
    target_norm = sum(np.linalg.norm(v)**2 for v in physical_target.values())
    lower_bound = sum(np.linalg.norm(v)**2 for key, v in physical_target.items()
                      if not retained.get(key[2], 0))
    retained_error = sum(np.linalg.norm(v)**2 for key, v in physical_error.items()
                         if retained.get(key[2], 0))
    if (np.isfinite(target_norm) and np.isfinite(retained_error) and
            retained_error <= options.tolerance**2*max(target_norm, np.finfo(float).tiny)):
        value = float(sum(np.linalg.norm(v)**2 for v in physical_error.values()))
        return CompressionResult(left, right, value, 0,
            dict(requested_solver=options.solver, used_solver='exact-sector-projection',
                 status='orthogonal-sector loss bound reached', initial_loss=value,
                 final_loss=value, optimizer_success=True, fallback_reason=None,
                 certified_sector_optimum=True, loss_lower_bound=float(lower_bound),
                 retained_sector_error=float(retained_error)))
    groups = reduced_factor_blocks(state, i, le, re, retained)
    metric = _PairMetric(problem, dtype, root=root)
    def adjoint(side, a, b, vector):
        if side == 0:
            return _left_source_adjoint(problem.layout, vector, _expanded_source_blocks(re, b), le)[li]
        return _right_source_adjoint(problem.layout, _expanded_source_blocks(le, a), vector, re)[ri]

    factor_sites = problem.frontier.to_mps(state, compact=True)

    def direct_solve(side,a,b):
        try:
            coordinates,forward,pullback=_factor_coordinates(state,problem,root,le,re,
                                                            a,b,retained,side,options,sites=factor_sites)
        except MemoryError as error:
            return None,dict(fallback_reason=str(error))
        rhs=root.physical_apply(target)
        old = a if side == 0 else b
        # Preserve invisible source entries so the next ALS half-step can use them.
        try:
            vector = old + coordinates.least_squares_correction(
                pullback(rhs)-coordinates.coordinates(old))
        except (MemoryError, np.linalg.LinAlgError) as error:
            return None, dict(fallback_reason=f'QR source correction failed: {type(error).__name__}')
        residual=root.physical_apply((merge(vector,b) if side==0 else merge(a,vector))-target)
        normal=float(np.linalg.norm(pullback(residual)))
        # Source reconstruction can lose a few digits even when Q is isometric.
        # Correct that physical residual before resorting to an iterative solve.
        corrections = 0
        threshold = options.tolerance*max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        for _ in range(3):
            if normal <= threshold:
                break
            try:
                trial = vector-coordinates.least_squares_correction(pullback(residual))
            except (MemoryError, np.linalg.LinAlgError):
                break
            trial_residual = root.physical_apply(
                (merge(trial,b) if side==0 else merge(a,trial))-target)
            trial_normal = float(np.linalg.norm(pullback(trial_residual)))
            if not np.isfinite(trial_normal) or trial_normal >= normal:
                break
            vector, residual, normal = trial, trial_residual, trial_normal
            corrections += 1
        converged=np.isfinite(normal) and normal<=threshold
        if not converged or not np.all(np.isfinite(vector)):
            return None,dict(fallback_reason='QR factor solution failed physical stationarity check')
        return vector,dict(side='left' if side==0 else 'right',solver='qr',iterations=1,
            residual_norm=float(np.linalg.norm(residual)),normal_residual_norm=normal,
            rank=coordinates.rank,refinement_steps=corrections,converged=True)

    return fit_metric_factors(target, metric, left, right, merge=merge, root=root,
        adjoint=adjoint, left_indices=li, right_indices=ri, gauge_blocks=groups,
        options=options, metric_tolerance=metric_tolerance,
        als_max_iterations=als_max_iterations, linear_solver=direct_solve)


def fit_metric_factors(target, metric, left, right, *, merge, root, adjoint,
                       left_indices, right_indices, gauge_blocks, options,
                       metric_tolerance=1e-12, als_max_iterations=100, linear_solver=None):
    """Shared ALS/nonlinear fitting loop for open and cyclic reduced metrics.

    ``adjoint(side, a, b, v)`` returns only the active side coordinates.
    Topology is entirely in merge/adjoint/root; budgets and loss guards match.
    ALS solves the same full physical-root objective as its loss guard; the
    inverse-support root must not silently discard residual directions.
    """
    li, ri, groups = left_indices, right_indices, gauge_blocks
    dtype = np.result_type(target, left, right, complex)
    forward_root = getattr(root, "physical_apply", root.apply)
    adjoint_root = getattr(root, "physical_adjoint", root.adjoint)
    rhs = np.asarray(forward_root(target), dtype=dtype)
    reports = []
    als_info = {}

    def loss(a, b):
        error = merge(a, b)-target
        physical = getattr(root, "physical_apply", None)
        value = (float(np.linalg.norm(physical(error))**2) if physical is not None
                 else float(np.vdot(error, metric@error).real))
        if not np.isfinite(value) or value < -1e-10:
            raise FloatingPointError('invalid reduced compression loss')
        return max(0., value)

    def solve(side, a, b):
        reason=None
        if linear_solver is not None:
            vector,report=linear_solver(side,a,b)
            if vector is not None:
                reports.append(report)
                return vector
            reason=report['fallback_reason']
        indices, old = (li, a) if side == 0 else (ri, b)
        def unpack(v):
            out = np.zeros(old.shape, dtype=dtype)
            out[indices] = v
            return out
        def forward(v):
            return forward_root(merge(unpack(v), b) if side == 0 else merge(a, unpack(v)))
        operator = LinearOperator((len(rhs), len(indices)), matvec=forward,
            rmatvec=lambda v: adjoint(side, a, b, adjoint_root(v)), dtype=dtype)
        limit = options.lsmr_max_iterations or max(50, min(1000, 5*len(indices)))
        answer = lsmr(operator, rhs, x0=np.asarray(old[indices], dtype=dtype), atol=options.tolerance,
                      btol=options.tolerance, maxiter=limit)
        reports.append(dict(side='left' if side == 0 else 'right', solver='lsmr',
            stop_code=int(answer[1]), iterations=int(answer[2]), max_iterations=limit,
            residual_norm=float(answer[3]), normal_residual_norm=float(answer[4]),
            converged=answer[1] in (0, 1, 2, 4, 5), fallback_reason=reason))
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

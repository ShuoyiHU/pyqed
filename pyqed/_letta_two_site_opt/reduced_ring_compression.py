"""All shared factor compressors in the full correlated cyclic pair metric."""
from collections import Counter
from operator import index

import numpy as np

from .reduced_compression import fit_metric_factors, _PairMetric
from .reduced_solver import _active_source_indices, _expanded_source_blocks
from .reduced_ring_pair import CyclicPairMetricRoot


def ring_factor_blocks(problem, state, retained):
    sites = (problem.left_site, problem.right_site)
    neighborhoods = [state.site_neighborhood(i) if i < state.nsites else () for i in sites]
    shared = sorted(set(neighborhoods[0]) & set(neighborhoods[1]))
    labels = {(p.sector, p.copy): i for i, p in enumerate(state.physical_basis.reduced_states)}
    groups = []
    for side, embedding in enumerate((problem.left_embedding, problem.right_embedding)):
        current = {}
        for key, shape in embedding.source_layout.shapes.items():
            middle = key[2] if side == 0 else key[0]
            rank = retained.get(middle, 0)
            if not rank:
                continue
            offset = embedding.source_layout.offsets[key][0]
            indices = offset+np.arange(np.prod(shape)).reshape(shape)
            for physical in np.ndindex(shape[1:-1]):
                values = ((labels[key[1], physical[0]],)+physical[1:]) if neighborhoods[side] else ()
                assignment = dict(zip(neighborhoods[side], values))
                group = (middle, tuple(assignment[v] for v in shared))
                section = ((slice(None),)+physical+(slice(0, rank),) if side == 0
                           else (slice(0, rank),)+physical+(slice(None),))
                current.setdefault(group, []).append(indices[section])
        groups.append(current)
    return [(np.concatenate(groups[0][q], axis=0), np.concatenate(groups[1][q], axis=1))
            for q in groups[0] if q in groups[1]]


def ring_factor_adjoint(problem, side, a, b, vector):
    """Adjoint of the bilinear source-factor map, including all tie copies."""
    le, re = problem.left_embedding, problem.right_embedding
    left, right = _expanded_source_blocks(le, a), _expanded_source_blocks(re, b)
    embedding = le if side == 0 else re
    out = {key: np.zeros(shape, dtype=np.result_type(vector, a, b, complex))
           for key, shape in embedding.target_layout.shapes.items()}
    for key, gradient in problem.layout.unpack(vector).items():
        lk, rk = key[:3], key[2:]
        if side == 0 and lk in out and rk in right:
            out[lk] += np.einsum('apqb,mqb->apm', gradient, right[rk].conj(), optimize=True)
        elif side == 1 and rk in out and lk in left:
            out[rk] += np.einsum('apm,apqb->mqb', left[lk].conj(), gradient, optimize=True)
    return embedding.adjoint(embedding.pack_target(out))


def compress_ring_pair(target, problem, state, left, right, retained, *, options,
                       metric_tolerance=1e-12, als_max_iterations=100):
    """Fit a cyclic pair target without treating the metric as separable.

    Retained counts select whole middle-sector multiplets within the supplied
    allocation. The caller owns allocation growth/shrinkage and energy guards.
    """
    available = Counter(state.bond_sectors[(problem.left_site+1) % (state.nsites+1)])
    retained = {q: index(rank) for q, rank in retained.items()}
    if any(rank < 0 or rank > available.get(q, 0) for q, rank in retained.items()):
        raise ValueError('retained ring multiplicity is outside the source allocation')
    le, re = problem.left_embedding, problem.right_embedding
    li, ri = _active_source_indices(le, retained, 'left'), _active_source_indices(re, retained, 'right')
    left, right = np.array(left, dtype=complex), np.array(right, dtype=complex)
    left[np.setdiff1d(np.arange(left.size), li)] = 0
    right[np.setdiff1d(np.arange(right.size), ri)] = 0
    root = CyclicPairMetricRoot(problem, metric_tolerance, max_workspace_mb=options.max_workspace_mb)
    def adjoint(side, a, b, vector):
        return ring_factor_adjoint(problem, side, a, b, vector)[li if side == 0 else ri]
    result = fit_metric_factors(target, _PairMetric(problem, left.dtype), left, right,
        merge=problem.merge, root=root, adjoint=adjoint, left_indices=li, right_indices=ri,
        gauge_blocks=ring_factor_blocks(problem, state, retained), options=options,
        metric_tolerance=metric_tolerance, als_max_iterations=als_max_iterations)
    from dataclasses import replace
    return replace(result, diagnostics={**result.diagnostics,
        'metric_kind': 'full-cyclic-reduced', 'metric_rank': root.size,
        'pair_dimension': problem.local_dimension})

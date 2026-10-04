"""Orbital-order and tie-graph controls, with explicit frontier costs."""
from __future__ import annotations

from itertools import permutations
from operator import index

import numpy as np

from .qchem import OCCUPATIONS, orbital_permutation
from .state import _validate_neighborhoods


def tie_neighborhoods(norb, edges=(), *, nearest=False, carry=False):
    """Forward ties in chain-position indices, optionally carried to their owner.

    ``carry=True`` includes s_j on every tensor from the first dependency on j
    through j. This restores the shared-frontier condition but enlarges the
    ansatz and its local tensors; it is not a gauge transform of direct ties.
    """
    norb = index(norb)
    if norb < 1:
        raise ValueError('norb must be positive')
    sites = [{i} for i in range(norb)]
    if nearest:
        for i in range(norb-1):
            sites[i].add(i+1)
    for edge in edges:
        try:
            i, j = sorted(index(k) for k in edge)
        except (TypeError, ValueError) as error:
            raise ValueError('edges must be pairs of integer site indices') from error
        if i < 0 or j >= norb or i == j:
            raise ValueError('tie edge must connect two distinct in-range sites')
        sites[i].add(j)
    if carry:
        for j in range(norb):
            start = min(i for i in range(norb) if j in sites[i])
            for i in range(start, j):
                sites[i].add(j)
    return tuple((i,) + tuple(sorted(s-{i})) for i, s in enumerate(sites))


def graph_diagnostics(neighborhoods, *, physical_dim=4):
    neighborhoods = _validate_neighborhoods(neighborhoods, len(neighborhoods))
    n = len(neighborhoods)
    cuts = []
    for c in range(1, n):
        left = set().union(*map(set, neighborhoods[:c]))
        right = set().union(*map(set, neighborhoods[c:]))
        frontier = left & right
        shared = set(neighborhoods[c-1]) & set(neighborhoods[c])
        cuts.append(dict(cut=c, frontier=sorted(frontier), shared=sorted(shared),
                         conditional_gauge=frontier <= shared))
    width = max((len(c['frontier']) for c in cuts), default=0)
    return dict(cuts=cuts, max_frontier_width=width,
                max_norm_frontier_configurations=physical_dim**width,
                max_hamiltonian_frontier_configurations=physical_dim**(2*width),
                max_physical_axes=max(map(len, neighborhoods)),
                all_cuts_conditional=all(c['conditional_gauge'] for c in cuts))


def _affinity(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or len(w) < 1:
        raise ValueError('affinity must be a nonempty square matrix')
    if not np.all(np.isfinite(w)) or np.any(w < -1e-12) or not np.allclose(w, w.T, atol=1e-12):
        raise ValueError('affinity must be finite, symmetric and nonnegative')
    w = np.maximum(w, 0.).copy()
    np.fill_diagonal(w, 0.)
    return w


def ordering_cost(weights, order):
    """Sum_{i<j} I_ij |position(i)-position(j)|^2 (no half-MI convention)."""
    w = _affinity(weights)
    p = orbital_permutation(order, len(w))
    positions = np.argsort(p)
    return float(np.sum(np.triu(w * (positions[:, None]-positions[None, :])**2, 1)))


def correlation_order(weights):
    """Exact minimum for <=8 orbitals; Fiedler ordering beyond that size."""
    w = _affinity(weights)
    n = len(w)
    if n <= 8:
        # Reversal has the same cost: lexicographic tie-breaking is deterministic.
        return min(permutations(range(n)), key=lambda p: (ordering_cost(w, p), p))
    _, vectors = np.linalg.eigh(np.diag(w.sum(axis=1))-w)
    p = tuple(map(int, np.argsort(vectors[:, 1], kind='stable')))
    return min(p, p[::-1])


def select_long_range_ties(weights, *, max_edges=1, max_frontier_width=2,
                           max_future_neighbors=2, nearest=True):
    """Greedily add strongest non-NN pairs under explicit contraction budgets.

    ``weights`` must already be in the chosen chain order. This is a heuristic,
    not an optimal graph search or an assertion of variational improvement.
    """
    w = _affinity(weights)
    limits = tuple(index(v) for v in (max_edges, max_frontier_width, max_future_neighbors))
    if min(limits) < 0:
        raise ValueError('tie budgets must be nonnegative')
    max_edges, max_frontier_width, max_future_neighbors = limits
    edges = []
    initial = tie_neighborhoods(len(w), nearest=nearest)
    report = graph_diagnostics(initial)
    if report['max_frontier_width'] > max_frontier_width or report['max_physical_axes']-1 > max_future_neighbors:
        raise ValueError('budgets cannot accommodate the requested NN backbone')
    candidates = sorted(((i, j) for i in range(len(w)) for j in range(i+2, len(w))
                         if w[i, j] > 0), key=lambda ij: (-w[ij], ij))
    for edge in candidates:
        if len(edges) == max_edges:
            break
        sites = tie_neighborhoods(len(w), edges+[edge], nearest=nearest)
        report = graph_diagnostics(sites)
        if (report['max_frontier_width'] <= max_frontier_width and
                report['max_physical_axes']-1 <= max_future_neighbors):
            edges.append(edge)
    return tuple(edges)


def fermionic_reorder_vector(vector, order):
    """Small-system reference: permute spatial sites with fermionic swap signs."""
    order = tuple(order)
    p = orbital_permutation(order, len(order))
    v = np.asarray(vector)
    if v.size != 4**len(p):
        raise ValueError('vector dimension does not match orbital order')
    if len(p) > 8:
        raise ValueError('dense reference limited to eight spatial orbitals')
    occupations = OCCUPATIONS.sum(axis=1)
    configs = np.indices((4,)*len(p))
    parity = np.zeros((4,)*len(p), dtype=np.int8)
    for a in range(len(p)):
        for b in range(a+1, len(p)):
            if p[a] > p[b]:
                parity ^= (occupations[configs[a]] * occupations[configs[b]] % 2).astype(np.int8)
    return (v.reshape((4,)*len(p)).transpose(p) * (1-2*parity)).ravel()


def orbital_mutual_information(vector, norb):
    """Small-system fermionic orbital MI, natural logs, I_ij=S_i+S_j-S_ij.

    Bring each subsystem first using fermionic permutations before tracing.
    An ordinary transpose alone gives incorrect nonadjacent orbital RDMs.
    The input may be an approximate pilot state; no FCI assumption is made.
    """
    norb = index(norb)
    if not 1 <= norb <= 8:
        raise ValueError('dense reference limited to one through eight orbitals')
    v = np.asarray(vector).reshape(-1)
    if v.size != 4**norb or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
        raise ValueError('invalid reference vector')
    v = v / np.linalg.norm(v)
    def entropy(selected):
        p = tuple(selected) + tuple(i for i in range(norb) if i not in selected)
        matrix = fermionic_reorder_vector(v, p).reshape(4**len(selected), -1)
        values = np.linalg.svd(matrix, compute_uv=False)**2
        values = values[values > 1e-15]
        return float(-np.sum(values*np.log(values)))
    single = np.array([entropy((i,)) for i in range(norb)])
    mi = np.zeros((norb, norb))
    for i in range(norb):
        for j in range(i+1, norb):
            mi[i, j] = mi[j, i] = max(0., single[i]+single[j]-entropy((i, j)))
    return single, mi

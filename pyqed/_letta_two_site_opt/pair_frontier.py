"""Exact support for a free adjacent-pair problem and its legal LETTA factors.

The pair eigensolver varies an arbitrary reduced pair tensor. Its outer copied
labels therefore need not obey the two active factors' equality constraints.
Prefix/suffix reachability preserves that larger intermediate space, while
complete legal paths select the factor parameters used during compression.
"""
from hashlib import sha256
from itertools import product
from types import SimpleNamespace

import numpy as np

from .._letta_one_site_opt.frontier_support import get_frontier_support
from .._letta_one_site_opt.reduced_frontier import (
    ReducedFrontier, FrontierSiteEmbedding, _BlockVectorLayout,
)
from .._letta_one_site_opt.reduced_state import ReducedLatticeLETTA


def _routes(state, site, pending, nodes, *, reverse=False):
    """One exact local copy/fusion transition, in either reachability direction."""
    left_variables, right_variables = pending[site:site + 2]
    known = right_variables if reverse else left_variables
    other = left_variables if reverse else right_variables
    new = tuple(v for v in other if v not in known and v != site)
    labels = {(p.sector, p.copy): i for i, p in enumerate(state.physical_basis.reduced_states)}
    blocks = {}
    for key, shape in sorted((key, block.shape) for key, block in state.tensors[site].items()):
        if any(n == 0 for n in shape):
            continue
        for copy in range(shape[1]):
            blocks.setdefault(key[2] if reverse else key[0], []).append(
                (key, copy, labels[key[1], copy]))
    edges, following = [], set()
    for q, assignment in sorted(nodes):
        values = dict(zip(known, assignment))
        for key, copy, own in blocks.get(q, ()):
            if site in values and values[site] != own:
                continue
            fixed = dict(values)
            fixed[site] = own
            for choice in product(range(state.physical_dim), repeat=len(new)):
                assigned = dict(fixed)
                assigned.update(zip(new, choice))
                la = tuple(assigned[v] for v in left_variables)
                ra = tuple(assigned[v] for v in right_variables)
                local = tuple(assigned[v] for v in state.site_neighborhood(site))
                edges.append((*key, copy, la, ra, local))
                following.add((key[0], la) if reverse else (key[2], ra))
    return tuple(edges), following


def _pair_support(state, cuts, legal, center):
    n = state.nsites
    pending = ((),) + tuple(cuts) + ((),)
    nodes, edges = [set() for _ in range(n + 1)], [() for _ in range(n)]
    nodes[0] = {(state.symmetry.identity, ())}
    for site in range(center):
        edges[site], nodes[site + 1] = _routes(state, site, pending, nodes[site])
    nodes[n] = {(state.symmetry.sector, ())}
    for site in range(n - 1, center + 1, -1):
        edges[site], nodes[site] = _routes(state, site, pending, nodes[site + 1], reverse=True)

    left_sectors = {q for q, _ in nodes[center]}
    right_sectors = {q for q, _ in nodes[center + 2]}
    channels = {(ql, qm, qr)
                for (ql, pl, qm), a in state.tensors[center].items()
                for (qm2, pr, qr), b in state.tensors[center + 1].items()
                if qm == qm2 and ql in left_sectors and qr in right_sectors
                and all(a.shape) and all(b.shape)}
    qleft, qmiddle, qright = ({item[k] for item in channels} for k in range(3))
    nodes[center] = {(q, a) for q, a in nodes[center] if q in qleft}
    nodes[center + 2] = {(q, a) for q, a in nodes[center + 2] if q in qright}
    for site in range(center - 1, -1, -1):
        edges[site] = tuple(e for e in edges[site] if (e[2], e[5]) in nodes[site + 1])
        nodes[site] = {(e[0], e[4]) for e in edges[site]}
    for site in range(center + 2, n):
        edges[site] = tuple(e for e in edges[site] if (e[0], e[4]) in nodes[site])
        nodes[site + 1] = {(e[2], e[5]) for e in edges[site]}

    # A live prefix, pair fusion channel and live suffix concatenate into a
    # legal path: each stored factor has complete tie-label axes. Missing an
    # entire middle sector here would therefore be a support-construction bug.
    if not qmiddle.issubset(legal.bases[center + 1]):
        raise RuntimeError('live pair sector has no complete legal factor path')
    nodes[center + 1] = {(q, a) for q in qmiddle for a in legal.bases[center + 1][q]}
    for site in (center, center + 1):
        edges[site] = tuple(e for e in legal.edges[site]
                            if (e[0], e[4]) in nodes[site] and (e[2], e[5]) in nodes[site + 1])
    bases = []
    for layer in nodes:
        basis = {}
        for q, assignment in sorted(layer):
            basis.setdefault(q, []).append(assignment)
        bases.append({q: tuple(assignments) for q, assignments in basis.items()})
    fingerprint = sha256((legal.fingerprint + ':pair:' + str(center)).encode()).hexdigest()
    return SimpleNamespace(signature=(legal.signature, center), fingerprint=fingerprint,
                           bases=tuple(bases), edges=tuple(edges))


class ReducedPairFrontier(ReducedFrontier):
    """Prune only exact null directions of the existing free-pair problem.

    Outer bases use a relaxed pair bridge. Active factor sources use complete
    legal scaffold paths, including newly admitted charge and spin sectors.
    No coefficient magnitude or numerical rank is used in either selection.
    """

    @classmethod
    def from_state(cls, state, left_site, *, prune=True):
        if not isinstance(state, ReducedLatticeLETTA):
            raise TypeError('ReducedPairFrontier requires open-boundary ReducedLatticeLETTA')
        left_site = int(left_site)
        if not 0 <= left_site < state.nsites - 1:
            raise IndexError('left_site is not an internal pair start')
        result = cls(state.neighborhoods, state.physical_dim, prune=prune)
        result.left_site = left_site
        result._legal_support = None
        return result

    def support(self, state):
        if not self.prune:
            return None
        if self.neighborhoods != state.neighborhoods:
            raise ValueError('frontier topology changed; rebuild the frontier')
        legal = get_frontier_support(state, self.cuts, previous=self._legal_support)
        if legal is not self._legal_support:
            self._support_graph = _pair_support(state, self.cuts, legal, self.left_site)
            self._legal_support = legal
            self._embedding_cache.clear()
            self._embedding_cache_bytes = 0
        return self._support_graph

    def _supported_embedding(self, state, site, support):
        if site not in (self.left_site, self.left_site + 1):
            return super()._supported_embedding(state, site, support)
        source_shapes = {key: block.shape for key, block in state.tensors[site].items()}
        source_layout = _BlockVectorLayout(source_shapes)
        left = {q: {a: i for i, a in enumerate(assignments)}
                for q, assignments in support.bases[site].items()}
        right = {q: {a: i for i, a in enumerate(assignments)}
                 for q, assignments in support.bases[site + 1].items()}
        shapes = {key: (shape[0] * len(left[key[0]]), shape[1],
                        shape[-1] * len(right[key[2]]))
                  for key, shape in source_shapes.items() if key[0] in left and key[2] in right}
        target_layout = _BlockVectorLayout(shapes)
        sources, targets = [], []
        for ql, qp, qr, physical, la, ra, labels in support.edges[site]:
            key = (ql, qp, qr)
            shape = source_shapes[key]
            a = np.arange(shape[0], dtype=np.int64)[:, None]
            b = np.arange(shape[-1], dtype=np.int64)[None, :]
            source = a * shape[1] + physical
            for dim, label in zip(shape[2:-1], labels[1:]):
                source = source * dim + label
            source = source * shape[-1] + b
            target = ((left[ql][la] * shape[0] + a) * shape[1] + physical) * shapes[key][-1]
            target = target + right[qr][ra] * shape[-1] + b
            sources.append(source.ravel() + source_layout.offsets[key][0])
            targets.append(target.ravel() + target_layout.offsets[key][0])
        source = np.concatenate(sources) if sources else np.empty(0, dtype=np.int64)
        target = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
        supported = np.unique(source)
        if len(target) != len(np.unique(target)):
            raise RuntimeError('pair paths assigned one MPS entry more than once')
        return FrontierSiteEmbedding(source_layout, target_layout,
            np.searchsorted(supported, source), target,
            self.left_variables(site), self.right_variables(site), supported)

    def dense_bytes(self, state):
        if not self.prune:
            return super().dense_bytes(state)
        support = self.support(state)
        total = 0
        for site, tensor in enumerate(state.tensors):
            keys = ({key for key in tensor
                     if key[0] in support.bases[site] and key[2] in support.bases[site + 1]}
                    if site in (self.left_site, self.left_site + 1)
                    else {e[:3] for e in support.edges[site]})
            for key in keys:
                a = tensor[key]
                total += (a.shape[0] * a.shape[1] * a.shape[-1]
                          * np.dtype(np.result_type(a, float)).itemsize
                          * len(support.bases[site][key[0]]) * len(support.bases[site + 1][key[2]]))
        return total

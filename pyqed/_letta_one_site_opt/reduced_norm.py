"""Multiplicity-only norm contractions for left-coupled SU(2) tensors.

An invariant boundary is G[q] tensor I_(2S_q+1). CG orthogonality gives unit
weight advancing left and dim(q_right)/dim(q_left) advancing right. The
Euclidean adjoint local action carries dim(q_right). These factors are needed
even for singlet targets; omitting them changes the generalized eigenproblem.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .numpy_contractions import cached_einsum
from .reduced_symmetry import _sector_irrep


def left_canonical_reduced_sites(sites):
    """Condition an auxiliary MPS by exact sector QR, without changing LETTA."""
    sites = tuple(a.copy() for a in sites)
    for i, site in enumerate(sites[:-1]):
        groups = defaultdict(list)
        for key, block in site.data.items():
            groups[key[2]].append((key, block))
        multiplets = []
        for qr, entries in groups.items():
            matrix = np.concatenate([a.reshape(-1, a.shape[2]) for _, a in entries])
            q, r = np.linalg.qr(matrix, mode='reduced')
            rank, start = q.shape[1], 0
            multiplets.extend([qr]*rank)
            for key, a in entries:
                rows = a.shape[0]*a.shape[1]
                site.data[key] = q[start:start+rows].reshape(a.shape[:2]+(rank,))
                start += rows
            for key, a in sites[i+1].data.items():
                if key[0] == qr:
                    sites[i+1].data[key] = np.tensordot(r, a, axes=(1, 0))
        site.qns[2] = multiplets
        sites[i+1].qns[0] = multiplets[:]
    return sites


def _boundary(site, axis):
    return {q: np.eye(r) for q, r in Counter(site.qns[axis]).items()}


def _normalize(blocks, log_scale):
    scale = max((float(np.max(np.abs(a), initial=0.)) for a in blocks.values()),
                default=0.)
    if scale == 0:
        return blocks, float(log_scale)
    return {q: a/scale for q, a in blocks.items()}, float(log_scale+np.log(scale))


def _support_projector(gram, tolerance, *, equilibrated=False):
    if equilibrated:
        roots = np.sqrt(np.maximum(np.diag(gram).real, 0.))
        denominator = roots[:, None]*roots[None, :]
        gram = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0.)
    values, vectors = np.linalg.eigh(.5*(gram+gram.conj().T))
    keep = values > tolerance*max(float(np.max(values, initial=0.)), np.finfo(float).tiny)
    return vectors[:, keep]@vectors[:, keep].conj().T


def advance_norm_left(environment, site):
    out = {}
    for (ql, _qp, qr), a in site.data.items():
        if ql not in environment:
            continue
        value = cached_einsum('al,apr,lps->rs', environment[ql], a.conj(), a)
        out[qr] = out.get(qr, 0.) + value
    return out


def advance_norm_right(environment, site):
    out = {}
    for (ql, _qp, qr), a in site.data.items():
        if qr not in environment:
            continue
        weight = _sector_irrep(qr).dim / _sector_irrep(ql).dim
        value = weight*cached_einsum('rs,apr,lps->al', environment[qr], a.conj(), a)
        out[ql] = out.get(ql, 0.) + value
    return out


class MovingReducedEnvironments:
    """Invalidate only boundaries crossing changed tensors; rebuild lazily."""
    left_valid = None
    right_valid = None

    def replace_sites(self, changes):
        if not changes:
            return
        sites = list(self.sites)
        for i, a in changes.items():
            sites[i] = a
        self.sites = tuple(sites)
        self.left_valid = min(len(sites) if self.left_valid is None else self.left_valid,
                              min(changes))
        self.right_valid = max(0 if self.right_valid is None else self.right_valid,
                               max(changes)+1)

    def ensure(self, left_cut, right_cut):
        left_valid = len(self.sites) if self.left_valid is None else self.left_valid
        right_valid = 0 if self.right_valid is None else self.right_valid
        for i in range(left_valid, left_cut):
            self.left[i+1], self.left_log_scales[i+1] = _normalize(
                self._advance_left(i), self.left_log_scales[i])
        for i in range(right_valid-1, right_cut-1, -1):
            self.right[i], self.right_log_scales[i] = _normalize(
                self._advance_right(i), self.right_log_scales[i+1])
        self.left_valid = max(left_valid, left_cut)
        self.right_valid = min(right_valid, right_cut)


@dataclass
class ReducedNormChain(MovingReducedEnvironments):
    """Scalar-sector environments, indexed by cuts from zero through nsites."""

    sites: tuple
    left: list
    right: list
    left_log_scales: list
    right_log_scales: list

    @classmethod
    def build(cls, sites):
        sites = tuple(sites)
        if not sites:
            raise ValueError('norm contraction requires at least one site')
        left, ll = [_boundary(sites[0], 0)], [0.]
        for site in sites:
            blocks, scale = _normalize(advance_norm_left(left[-1], site), ll[-1])
            left.append(blocks)
            ll.append(scale)
        right, rl = [_boundary(sites[-1], 2)], [0.]
        for site in reversed(sites):
            blocks, scale = _normalize(advance_norm_right(right[-1], site), rl[-1])
            right.append(blocks)
            rl.append(scale)
        return cls(sites, left, list(reversed(right)), ll, list(reversed(rl)))

    def expectation(self):
        """Norm summed over the complete target multiplet."""
        self.ensure(len(self.sites), len(self.sites))
        return sum(_sector_irrep(q).dim*np.trace(a)
                   for q, a in self.left[-1].items()) * np.exp(self.left_log_scales[-1])

    def local_action(self, site, blocks):
        """Euclidean adjoint action in packed reduced tensor coordinates."""
        self.ensure(site, site+1)
        scale = np.exp(self.left_log_scales[site]+self.right_log_scales[site+1])
        out = {}
        for key, a in blocks.items():
            ql, _qp, qr = key
            left, right = self.left[site].get(ql), self.right[site+1].get(qr)
            if left is None or right is None:
                out[key] = np.zeros_like(a)
            else:
                out[key] = scale*_sector_irrep(qr).dim*cached_einsum(
                    'al,br,lpr->apb', left, right, a)
        return out

    def pair_action(self, site, blocks):
        self.ensure(site, site+2)
        scale = np.exp(self.left_log_scales[site]+self.right_log_scales[site+2])
        out = {}
        for key, a in blocks.items():
            ql, _p1, _qm, _p2, qr = key
            left, right = self.left[site].get(ql), self.right[site+2].get(qr)
            out[key] = (np.zeros_like(a) if left is None or right is None else
                scale*_sector_irrep(qr).dim*cached_einsum(
                    'al,br,lpqr->apqb', left, right, a))
        return out

    def metric_scale(self, site, keys, *, width=1, embedding=None):
        """Spectral upper bound, used to discard numerical metric nullspace."""
        self.ensure(site, site+width)
        left = {q: np.linalg.norm(a, 2) for q, a in self.left[site].items()}
        right = {q: np.linalg.norm(a, 2) for q, a in self.right[site+width].items()}
        bound = max((_sector_irrep(key[-1]).dim*left.get(key[0], 0.)*right.get(key[-1], 0.)
                     for key in keys), default=0.)
        bound *= np.exp(self.left_log_scales[site]+self.right_log_scales[site+width])
        if embedding is not None:
            # P copies parameters into distinct target entries, so ||P||^2 is
            # the maximum number of occurrences of one source coordinate.
            bound *= np.max(np.bincount(embedding.source_indices), initial=0)
        return float(bound)

    def local_projector(self, site, embedding, tolerance, *, equilibrated=False):
        """Remove locally invisible coordinates without building a metric.

        When every frontier variable occurs on the center tensor, P assigns
        each source coordinate once. The norm is diagonal in the owned and
        tied physical labels; each remaining block is a Kronecker product of
        two conditional Grams. This applies to untied, NN, and carried ties.
        More general embeddings explicitly return None. With equilibrated=True,
        the returned orthogonal projector acts on D*x, where D contains the
        source metric diagonal square roots, rather than on raw source x.
        """
        counts = np.bincount(embedding.source_indices, minlength=embedding.source_size)
        if np.any(counts != 1):
            return None
        self.ensure(site, site+1)
        targets = np.empty(embedding.source_size, dtype=int)
        targets[embedding.source_indices] = embedding.target_indices
        plan = []
        for key in embedding.source_layout.keys:
            shape = embedding.source_layout.shapes[key]
            ql, _qp, qr = key
            source_start = embedding.source_layout.offsets[key][0]
            target_start = embedding.target_layout.offsets[key][0]
            for physical in np.ndindex(shape[1:-1]):
                flat = source_start+np.ravel_multi_index((0,)+physical+(0,), shape)
                l, _p, r = np.unravel_index(targets[flat]-target_start,
                                           embedding.target_layout.shapes[key])
                left = self.left[site][ql][l:l+shape[0], l:l+shape[0]]
                right = self.right[site+1][qr][r:r+shape[-1], r:r+shape[-1]]
                plan.append((key, physical, _support_projector(left, tolerance, equilibrated=equilibrated),
                             _support_projector(right, tolerance, equilibrated=equilibrated)))
        def project(vector):
            source = embedding.unpack_source(vector)
            out = {k: np.zeros(a.shape, dtype=np.result_type(a, complex)) for k, a in source.items()}
            for key, physical, left, right in plan:
                section = (slice(None),)+physical+(slice(None),)
                out[key][section] = left@source[key][section]@right.T
            return embedding.pack_source(out)
        return project

    def pair_projector(self, site, layout, tolerance, *, equilibrated=False):
        self.ensure(site, site+2)
        left = {q: _support_projector(a, tolerance, equilibrated=equilibrated) for q, a in self.left[site].items()}
        right = {q: _support_projector(a, tolerance, equilibrated=equilibrated) for q, a in self.right[site+2].items()}
        def project(vector):
            blocks = layout.unpack(vector)
            return layout.pack({key: cached_einsum('al,br,lpqr->apqb',
                left[key[0]], right[key[-1]], a) for key, a in blocks.items()})
        return project

    def _advance_left(self, i):
        return advance_norm_left(self.left[i], self.sites[i])

    def _advance_right(self, i):
        return advance_norm_right(self.right[i+1], self.sites[i])


__all__ = ['ReducedNormChain', 'advance_norm_left', 'advance_norm_right']

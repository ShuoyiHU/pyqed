"""Exact cyclic reduced pair responses, including covariant closure edges.

Pair coefficients retain the sequential physical fusion path. All structural
magnetic indices occur only in cached CG coefficients. Environment actions
never expand a variational magnetic tensor or a determinant-space frame.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from operator import index

import numpy as np

from pyqed.mps.su2 import SU2Irrep
from .._letta_one_site_opt.reduced_symmetry import _sector_irrep, _fuse_sectors
from .._letta_one_site_opt.reduced_frontier import _BlockVectorLayout
from .._letta_one_site_opt.reduced_mpo_compile import cg_tensor
from .._letta_one_site_opt.reduced_ring_contraction import (
    CyclicReducedNorm, CyclicReducedOperator, _dual_pair, _triple_basis, _operator_spin_tensor)
from .._letta_one_site_opt.reduced_ring_solver import _native_hamiltonian, ring_site_embedding
from .._letta_one_site_opt.contractions import _equilibrated_metric_factors
from .reduced_solver import _merge_pair_blocks


@lru_cache(maxsize=None)
def _pair_cg(left, first, middle, second, right):
    return np.einsum('apm,mqb->apqb', cg_tensor(left, first, middle),
                     cg_tensor(middle, second, right), optimize=True)


@lru_cache(maxsize=None)
def _pair_norm_coefficient(bra, ket, total):
    return float(np.einsum('alm,apqb,lpqs,bsm->',
        _dual_pair(bra[0], ket[0], total).conj(), _pair_cg(*bra).conj(),
        _pair_cg(*ket), _dual_pair(bra[-1], ket[-1], total), optimize=True)/total.dim)


@lru_cache(maxsize=None)
def _pair_operator_coefficient(bra, ket, operator, left_channel, right_channel, total):
    return float(np.einsum('xalm,aopb,lqrs,xuvy,uoq,vpr,ybsm->',
        _triple_basis(bra[0], ket[0], operator[0], left_channel, total).conj(),
        _pair_cg(*bra).conj(), _pair_cg(*ket), _pair_cg(*operator),
        _operator_spin_tensor(bra[1], ket[1], operator[1]),
        _operator_spin_tensor(bra[3], ket[3], operator[3]),
        _triple_basis(bra[-1], ket[-1], operator[-1], right_channel, total),
        optimize=True)/total.dim)


def _pair_environment(chain, site):
    return chain.environment(site, removed=2)


def _pair_layout(left, right):
    shapes = {}
    for ql, dl in Counter(left.qns[0]).items():
        for p, dp in Counter(left.qns[1]).items():
            for middle in _fuse_sectors(ql, p):
                for q, dq in Counter(right.qns[1]).items():
                    for qr, dr in Counter(right.qns[2]).items():
                        if qr in _fuse_sectors(middle, q):
                            shapes[ql, p, middle, q, qr] = (dl, dp, dq, dr)
    if not shapes:
        raise ValueError('ring pair has no symmetry-allowed coefficients')
    return _BlockVectorLayout(shapes)


@dataclass(frozen=True)
class _NormEntry:
    bra: tuple
    ket: tuple
    environment: np.ndarray
    coefficient: float


@dataclass(frozen=True)
class _OperatorEntry:
    bra: tuple
    ket: tuple
    environment: np.ndarray
    kernel: np.ndarray


class CyclicPairProblem:
    """Untruncated pair space and exact correlated cyclic H/N actions.

    The edge is (left_site, left_site+1 modulo L+1), where L is the explicit
    target closure. This includes last-physical/closure and closure/first-
    physical edges, not a direct two-physical-core update across the closure.
    """
    def __init__(self, state, hamiltonian, left_site):
        i = index(left_site)
        if not 0 <= i <= state.nsites:
            raise IndexError('ring pair edge out of range')
        self.left_site = i
        self.right_site = (i+1) % (state.nsites+1)
        ring = state.to_target_ring()
        if len(ring.sites) < 2:
            raise ValueError('a pair requires at least two graph vertices')
        self.sites = ring.sites
        self.left_embedding = ring_site_embedding(state, i)
        self.right_embedding = ring_site_embedding(state, self.right_site)
        self.layout = _pair_layout(self.sites[i], self.sites[self.right_site])
        self.old_vector = self.merge(self.left_embedding.pack_source(state.site_blocks(i)),
                                     self.right_embedding.pack_source(state.site_blocks(self.right_site)))
        self.hamiltonian = self.metric = None
        self.metric_scale = self.metric_projector_factory = None
        self._norm_entries = self._compile_norm(CyclicReducedNorm(self.sites))
        h = CyclicReducedOperator(self.sites, _native_hamiltonian(state, hamiltonian))
        self._operator_entries = self._compile_operator(h)

    @property
    def local_dimension(self):
        return self.layout.size

    @property
    def full_local_dimension(self):
        return self.layout.size

    def merge(self, left, right):
        a = self.left_embedding.unpack_target(self.left_embedding.apply(left))
        b = self.right_embedding.unpack_target(self.right_embedding.apply(right))
        blocks = {key: np.zeros(shape, dtype=np.result_type(left, right, complex))
                  for key, shape in self.layout.shapes.items()}
        blocks.update(_merge_pair_blocks(a, b))
        return self.layout.pack(blocks)

    def _compile_norm(self, chain):
        i = self.left_site
        left, right = chain.cuts[i], chain.cuts[(i+2) % len(self.sites)]
        environments = _pair_environment(chain, i)
        entries = []
        for bk in self.layout.keys:
            for kk in self.layout.keys:
                if (bk[1], bk[3]) != (kk[1], kk[3]):
                    continue
                for j, environment in environments.items():
                    lk, rk = (bk[0], kk[0], j), (bk[-1], kk[-1], j)
                    if lk not in left.blocks or rk not in right.blocks:
                        continue
                    coefficient = _pair_norm_coefficient(tuple(map(_sector_irrep, bk)),
                        tuple(map(_sector_irrep, kk)), SU2Irrep(j))
                    if coefficient == 0.:
                        continue
                    ls, lshape = left.blocks[lk]
                    rs, rshape = right.blocks[rk]
                    entries.append(_NormEntry(bk, kk, environment[rs, ls].reshape(rshape+lshape),
                                              (j+1)*coefficient))
        return tuple(entries)

    def _compile_operator(self, chain):
        i, k = self.left_site, self.right_site
        left, right = chain.cuts[i], chain.cuts[(i+2) % len(self.sites)]
        environments = _pair_environment(chain, i)
        channels = []
        for site in (i, k):
            grouped = defaultdict(list)
            for c in chain.local_channels[site]:
                grouped[c.output_sector, c.input_sector, c.sector].append(c)
            channels.append(grouped)
        operators = _merge_pair_blocks(chain.mpo.sites[i], chain.mpo.sites[k])
        entries = []
        for bk in self.layout.keys:
            for kk in self.layout.keys:
                for wk, w in operators.items():
                    first = channels[0][bk[1], kk[1], wk[1]]
                    second = channels[1][bk[3], kk[3], wk[3]]
                    if not first or not second:
                        continue
                    for lk, (ls, lshape) in left.blocks.items():
                        if lk[:3] != (wk[0], bk[0], kk[0]) or lk[-1] not in environments:
                            continue
                        for rk, (rs, rshape) in right.blocks.items():
                            if rk[:3] != (wk[-1], bk[-1], kk[-1]) or rk[-1] != lk[-1]:
                                continue
                            coefficient = _pair_operator_coefficient(tuple(map(_sector_irrep, bk)),
                                tuple(map(_sector_irrep, kk)), tuple(map(_sector_irrep, wk)),
                                SU2Irrep(lk[-2]), SU2Irrep(rk[-2]), SU2Irrep(lk[-1]))
                            if coefficient == 0.:
                                continue
                            bs, ks = self.layout.shapes[bk], self.layout.shapes[kk]
                            kernel = np.zeros((w.shape[0], w.shape[-1], bs[1], ks[1], bs[2], ks[2]), dtype=w.dtype)
                            for a in first:
                                for b in second:
                                    kernel[:, :, a.output_copy, a.input_copy, b.output_copy, b.input_copy] += coefficient*(lk[-1]+1)*w[:, a.copy, b.copy, :]
                            environment = environments[lk[-1]][rs, ls].reshape(rshape+lshape)
                            entries.append(_OperatorEntry(bk, kk, environment, kernel))
        return tuple(entries)

    def apply_metric(self, vector):
        if self.metric is not None:
            return self.metric@vector
        blocks = self.layout.unpack(vector)
        out = {key: np.zeros(shape, dtype=np.result_type(vector, complex)) for key, shape in self.layout.shapes.items()}
        for e in self._norm_entries:
            out[e.bra] += e.coefficient*np.einsum('rsal,lpqs->apqr', e.environment, blocks[e.ket], optimize=True)
        return self.layout.pack(out)

    def apply_hamiltonian(self, vector):
        if self.hamiltonian is not None:
            return self.hamiltonian@vector
        blocks = self.layout.unpack(vector)
        out = {key: np.zeros(shape, dtype=np.result_type(vector, complex)) for key, shape in self.layout.shapes.items()}
        for e in self._operator_entries:
            out[e.bra] += np.einsum('ybrxal,xyopuv,lpvr->aoub', e.environment, e.kernel, blocks[e.ket], optimize=True)
        return self.layout.pack(out)

    def materialize(self, *, hamiltonian=True, max_workspace_mb=128.):
        # Include construction/factorization copies as well as resident H/N.
        count = 12 if hamiltonian else 10
        if count*16*self.local_dimension**2 > max_workspace_mb*1024**2:
            raise MemoryError('cyclic local pair matrices exceed max_workspace_mb')
        eye = np.eye(self.local_dimension)
        for name, action in [('metric', self.apply_metric)] + ([('hamiltonian', self.apply_hamiltonian)] if hamiltonian else []):
            matrix = np.column_stack([action(v) for v in eye])
            scale = max(np.linalg.norm(matrix), np.finfo(float).tiny)
            if not np.all(np.isfinite(matrix)) or np.linalg.norm(matrix-matrix.conj().T) > 1e-9*scale:
                raise FloatingPointError('nonfinite or non-Hermitian cyclic pair operator')
            setattr(self, name, (matrix+matrix.conj().T)/2)
        return self


class CyclicPairMetricRoot:
    """Equilibrated square root of the complete cyclic local pair metric.

    This initial accuracy backend materializes a LOCAL coefficient-space Gram,
    never a global physical-state frame. It makes no separable-boundary claim.
    """
    def __init__(self, problem, tolerance=1e-12, *, max_workspace_mb=128.):
        if 10*16*problem.local_dimension**2 > max_workspace_mb*1024**2:
            raise MemoryError('cyclic metric factorization exceeds max_workspace_mb')
        if problem.metric is None:
            problem.materialize(hamiltonian=False, max_workspace_mb=max_workspace_mb)
        values = np.linalg.eigvalsh(problem.metric)
        if values[0] < -1e-10*max(abs(values[-1]), np.finfo(float).tiny):
            raise FloatingPointError('cyclic pair metric is not positive semidefinite')
        self.whitening, self.coordinates = _equilibrated_metric_factors(problem.metric, tolerance)
        self.size = self.coordinates.shape[0]
        if not self.size:
            raise FloatingPointError('cyclic pair metric has no supported directions')

    def apply(self, vector):
        return self.coordinates@vector

    def adjoint(self, vector):
        return self.coordinates.conj().T@vector

    def unwhiten(self, vector):
        return self.whitening@vector

    def unwhiten_adjoint(self, vector):
        return self.whitening.conj().T@vector

    def inverse_action(self, vector):
        return self.unwhiten(self.unwhiten_adjoint(vector))


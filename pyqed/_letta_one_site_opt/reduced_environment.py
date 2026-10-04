"""Native multiplicity-space Hamiltonian environments for SU(2) LETTA.

Magnetic sums occur only in cached, state-independent structural coefficients.
All variational tensors, boundary environments, and local actions are reduced.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values

from .reduced_mpo_compile import cg_tensor
from .reduced_norm import _normalize, MovingReducedEnvironments
from .reduced_symmetry import _sector_irrep
from .numpy_contractions import cached_einsum


@lru_cache(maxsize=None)
def spin_transfer(lb, lk, pb, pk, rb, rk, wl, wp, wr):
    """Contract the six CG vertices of one bra/operator/ket transfer.

    Environment basis B[x,a,l] = <j_k m_l, J_w m_x | j_b m_a>.
    Its squared Frobenius norm is dim(j_b). The result is the *un-normalized*
    local quadratic form; divide by that dimension only when advancing an
    environment. Keeping this distinction preserves Euclidean adjoints.
    """
    left = cg_tensor(lk, wl, lb).transpose(1, 2, 0)
    right = cg_tensor(rk, wr, rb).transpose(1, 2, 0)
    if not np.any(left) or not np.any(right):
        return 0.
    physical = np.array([[[(-1.)**((pk.two_j-mi)//2)*clebsch_gordan(
        pb, pk, wp, mo, -mi, m) for mi in ordered_two_m_values(pk)]
        for mo in ordered_two_m_values(pb)] for m in ordered_two_m_values(wp)])
    return float(np.einsum('xal,ybr,aob,lpr,xzy,zop->', left, right,
        cg_tensor(lb, pb, rb), cg_tensor(lk, pk, rk),
        cg_tensor(wl, wp, wr), physical, optimize=True))


@dataclass(frozen=True)
class ReducedTransfer:
    bra_key: tuple
    ket_key: tuple
    left_key: tuple
    right_key: tuple
    kernel: np.ndarray


def compile_transfers(site, mpo_site, channels):
    """Compile immutable spin structure once for a fixed sector allocation."""
    groups = defaultdict(list)
    for channel in channels:
        groups[channel.sector].append(channel)
    kernels = {}
    for bra_key, bra in site.data.items():
        lb, pb, rb = bra_key
        for ket_key, ket in site.data.items():
            lk, pk, rk = ket_key
            for (wl, wp, wr), w in mpo_site.data.items():
                local_channels = [c for c in groups[wp]
                                  if c.output_sector == pb and c.input_sector == pk]
                if not local_channels:
                    continue
                coefficient = spin_transfer(*map(_sector_irrep,
                    (lb, lk, pb, pk, rb, rk, wl, wp, wr)))
                if abs(coefficient) < 1e-14:
                    continue
                key = (bra_key, ket_key, (wl, lb, lk), (wr, rb, rk))
                kernel = kernels.setdefault(key, np.zeros(
                    (w.shape[0], w.shape[2], bra.shape[1], ket.shape[1]), dtype=w.dtype))
                for channel in local_channels:
                    kernel[:, :, channel.output_copy, channel.input_copy] += coefficient*w[:, channel.copy, :]
    return tuple(ReducedTransfer(*key, value) for key, value in kernels.items()
                 if np.any(value))


def cached_transfers(mpo, i, site):
    cache = getattr(mpo, '_transfer_cache', None)
    if cache is None:
        cache = {}
        object.__setattr__(mpo, '_transfer_cache', cache)
    key = (i, tuple((k, a.shape[1]) for k, a in site.data.items()))
    if key not in cache:
        cache[key] = compile_transfers(site, mpo.sites[i], mpo.channels)
    return cache[key]


def _boundary(site, mpo_site, axis):
    qns = tuple(dict.fromkeys(mpo_site.qns[axis]))
    if len(qns) != 1 or _sector_irrep(qns[0]).two_j != 0:
        raise ValueError('native scalar MPO requires spin-zero open boundaries')
    return {(qns[0], q, q): np.eye(r)[None]
            for q, r in Counter(site.qns[axis]).items()}


def advance_left(environment, site, transfers):
    result = {}
    for t in transfers:
        left = environment.get(t.left_key)
        if left is None:
            continue
        value = cached_einsum('xal,xyop,aob,lpr->ybr', left, t.kernel,
            site.data[t.bra_key].conj(), site.data[t.ket_key])
        value /= _sector_irrep(t.bra_key[2]).dim
        result[t.right_key] = result.get(t.right_key, 0.) + value
    return result


def advance_right(environment, site, transfers):
    result = {}
    for t in transfers:
        right = environment.get(t.right_key)
        if right is None:
            continue
        value = cached_einsum('ybr,xyop,aob,lpr->xal', right, t.kernel,
            site.data[t.bra_key].conj(), site.data[t.ket_key])
        value /= _sector_irrep(t.bra_key[0]).dim
        result[t.left_key] = result.get(t.left_key, 0.) + value
    return result


@dataclass
class ReducedEnvironmentChain(MovingReducedEnvironments):
    sites: tuple
    mpo: object
    transfers: tuple
    left: list
    right: list
    left_log_scales: list
    right_log_scales: list

    @classmethod
    def build(cls, sites, mpo):
        sites = tuple(sites)
        if len(sites) != len(mpo.sites):
            raise ValueError('one reduced MPO core is required per site')
        transfers = tuple(cached_transfers(mpo, i, a) for i, a in enumerate(sites))
        left, ll = [_boundary(sites[0], mpo.sites[0], 0)], [0.]
        for a, ts in zip(sites, transfers):
            e, scale = _normalize(advance_left(left[-1], a, ts), ll[-1])
            left.append(e); ll.append(scale)
        right, rl = [_boundary(sites[-1], mpo.sites[-1], 2)], [0.]
        for a, ts in reversed(tuple(zip(sites, transfers))):
            e, scale = _normalize(advance_right(right[-1], a, ts), rl[-1])
            right.append(e); rl.append(scale)
        return cls(sites, mpo, transfers, left, list(reversed(right)), ll, list(reversed(rl)))

    def expectation(self):
        self.ensure(len(self.sites), len(self.sites))
        return sum(_sector_irrep(qb).dim*np.trace(a[0])
                   for (_w, qb, qk), a in self.left[-1].items() if qb == qk
                   ) * np.exp(self.left_log_scales[-1])

    def local_action(self, site, blocks):
        self.ensure(site, site+1)
        dtype = np.result_type(*[a.dtype for a in blocks.values()],
                               *[t.kernel.dtype for t in self.transfers[site]],
                               *[a.dtype for a in self.left[site].values()],
                               *[a.dtype for a in self.right[site+1].values()])
        out = {key: np.zeros(a.shape, dtype=dtype) for key, a in blocks.items()}
        for t in self.transfers[site]:
            left, right = self.left[site].get(t.left_key), self.right[site+1].get(t.right_key)
            if left is None or right is None:
                continue
            out[t.bra_key] += cached_einsum('xal,xyop,ybr,lpr->aob', left, t.kernel,
                right, blocks[t.ket_key])
        scale = np.exp(self.left_log_scales[site]+self.right_log_scales[site+1])
        return {key: a*scale for key, a in out.items()}

    def pair_action(self, site, blocks):
        self.ensure(site, site+2)
        dtype = np.result_type(*[a.dtype for a in blocks.values()],
                               *[t.kernel.dtype for t in self.transfers[site]],
                               *[t.kernel.dtype for t in self.transfers[site+1]],
                               *[a.dtype for a in self.left[site].values()],
                               *[a.dtype for a in self.right[site+2].values()])
        out = {key: np.zeros(a.shape, dtype=dtype) for key, a in blocks.items()}
        right_by_middle = defaultdict(list)
        for right in self.transfers[site+1]:
            right_by_middle[right.left_key].append(right)
        for first in self.transfers[site]:
            left = self.left[site].get(first.left_key)
            if left is None:
                continue
            for second in right_by_middle[first.right_key]:
                right = self.right[site+2].get(second.right_key)
                if right is None:
                    continue
                bra = first.bra_key + second.bra_key[1:]
                ket = first.ket_key + second.ket_key[1:]
                if bra not in out or ket not in blocks:
                    continue
                # Projecting the intermediate CG intertwiner divides by its
                # squared norm. Its shared magnetic indices are already summed.
                out[bra] += cached_einsum('xal,xyop,yzuv,zbr,lpvr->aoub', left,
                    first.kernel, second.kernel, right, blocks[ket]
                    ) / _sector_irrep(first.bra_key[2]).dim
        scale = np.exp(self.left_log_scales[site]+self.right_log_scales[site+2])
        return {key: a*scale for key, a in out.items()}

    def replace_sites(self, changes):
        super().replace_sites(changes)
        transfers = list(self.transfers)
        for i, site in changes.items():
            transfers[i] = cached_transfers(self.mpo, i, site)
        self.transfers = tuple(transfers)

    def _advance_left(self, i):
        return advance_left(self.left[i], self.sites[i], self.transfers[i])

    def _advance_right(self, i):
        return advance_right(self.right[i+1], self.sites[i], self.transfers[i])

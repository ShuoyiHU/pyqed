"""Native cyclic contractions in reduced virtual transfer channels.

A closed ring needs every irrep in ket times dual bra, not only the scalar
boundary channel of an open chain. Only structural CG arrays contain magnetic
indices; variational arrays remain reduced multiplicity blocks throughout.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values
from .reduced_mpo_compile import cg_tensor
from .reduced_symmetry import _sector_irrep, _fuse_sectors


@lru_cache(maxsize=None)
def _dual_pair(bra, ket, total):
    """Unitary structural basis for ket tensor dual(bra), ordered bra/ket/M."""
    return np.array([[[(-1.)**((bra.two_j-mb)//2)*clebsch_gordan(
        ket, bra, total, mk, -mb, m) for m in ordered_two_m_values(total)]
        for mk in ordered_two_m_values(ket)] for mb in ordered_two_m_values(bra)])


@lru_cache(maxsize=None)
def _norm_spin_transfer(lb, lk, physical, rb, rk, total):
    left = _dual_pair(lb, lk, total)
    right = _dual_pair(rb, rk, total)
    return float(np.einsum('akz,apb,kpl,blz->', left.conj(),
        cg_tensor(lb, physical, rb).conj(), cg_tensor(lk, physical, rk), right,
        optimize=True)/total.dim)


def _validate_cycle(sites):
    sites = tuple(sites)
    if not sites:
        raise ValueError('a reduced ring requires at least one core')
    for i, site in enumerate(sites):
        if site.rank != 3:
            raise ValueError('cyclic reduced cores must have rank three')
        counts = tuple(Counter(q) for q in site.qns)
        if counts[2] != Counter(sites[(i+1) % len(sites)].qns[0]):
            raise ValueError(f'cyclic bond allocation mismatch after site {i}')
        for key, value in site.data.items():
            shape = tuple(counts[axis].get(q, 0) for axis, q in enumerate(key))
            if (key[2] not in _fuse_sectors(key[0], key[1]) or
                    value.shape != shape or not np.all(np.isfinite(value))):
                raise ValueError(f'invalid reduced ring block at site {i}')
    return sites


class _NormCut:
    def __init__(self, bra_sectors, ket_sectors):
        self.blocks, self.sizes = {}, {}
        for qb, db in sorted(Counter(bra_sectors).items()):
            for qk, dk in sorted(Counter(ket_sectors).items()):
                jb, jk = _sector_irrep(qb), _sector_irrep(qk)
                for two_j in range(abs(jb.two_j-jk.two_j), jb.two_j+jk.two_j+1, 2):
                    offset = self.sizes.get(two_j, 0)
                    self.blocks[qb, qk, two_j] = (slice(offset, offset+db*dk), (db, dk))
                    self.sizes[two_j] = offset+db*dk


@dataclass(frozen=True)
class _NormEntry:
    bra_key: tuple
    ket_key: tuple
    two_j: int
    left: slice
    right: slice
    shape: tuple
    coefficient: float


def _binary_scaled(matrix):
    """Separate an exact binary scale without overflowing a complex modulus."""
    peak = max(float(np.max(np.abs(matrix.real), initial=0.)),
               float(np.max(np.abs(matrix.imag), initial=0.)))
    if not np.isfinite(peak):
        raise FloatingPointError('nonfinite cyclic transfer product')
    if peak == 0.:
        return matrix, 0
    exponent = int(np.frexp(peak)[1])
    return _binary_rescale(matrix, -exponent), exponent


def _binary_rescale(matrix, exponent):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        if np.iscomplexobj(matrix):
            result = np.empty_like(matrix)
            result.real = np.ldexp(matrix.real, exponent)
            result.imag = np.ldexp(matrix.imag, exponent)
        else:
            result = np.ldexp(matrix, exponent)
    return result


def _transfer_product(matrices, dimension):
    """Multiply every transfer exactly up to roundoff, with no rank truncation.

    Keep an integer power of two outside each multiplication. Intermediate
    overflow/underflow cannot erase a representable final environment merely
    because neighboring tensor gauges have large cancelling scalar factors.
    """
    product = np.eye(dimension, dtype=complex)
    exponent = 0
    for matrix in matrices:
        factor, factor_exponent = _binary_scaled(matrix)
        product, product_exponent = _binary_scaled(product@factor)
        exponent += factor_exponent+product_exponent
        if not np.any(product):
            exponent = 0
    result = _binary_rescale(product, exponent)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('cyclic environment exceeds floating-point range')
    if np.any(product) and not np.any(result):
        raise FloatingPointError('cyclic environment underflows floating-point range')
    return result


class _CyclicTransferChain:
    def environment(self, site, *, removed=1):
        """Complement of consecutive cores, from their right to left cut.

        Internal cuts of the removed region cannot restrict expansion channels.
        The same scaled contraction is used by one-site and two-site actions.
        """
        n = len(self.sites)
        if not 0 <= site < n or not 1 <= removed <= n:
            raise ValueError('invalid cyclic complement')
        start = (site+removed) % n
        internal = {(site+step) % n for step in range(1, removed)}
        cuts = [cut for i, cut in enumerate(self.cuts) if i not in internal]
        channels = set.intersection(*(set(c.sizes) for c in cuts))
        return {j: _transfer_product(
                    (self.transfers[(site+step) % n][j] for step in range(removed, n)),
                    self.cuts[start].sizes[j])
                for j in sorted(channels)}

    def channel_overlaps(self):
        environment = self.environment(0)
        return {j: (j+1)*np.einsum('ab,ba->', self.transfers[0][j], env)
                for j, env in environment.items()}

    def overlap(self):
        return sum(self.channel_overlaps().values(), 0j)


class CyclicReducedNorm(_CyclicTransferChain):
    """Exact reduced ring overlap and local metric actions.

    Each J channel contributes (2J+1) times a cyclic transfer trace. The returned
    norm is that of the invariant ring tensor contraction; target closures and
    their component-normalization convention are handled by the state layer.
    Inputs are immutable during the lifetime of this contraction object.
    """

    def __init__(self, sites, *, bra=None):
        self.sites = _validate_cycle(sites)
        self.bra = self.sites if bra is None else _validate_cycle(bra)
        if len(self.bra) != len(self.sites):
            raise ValueError('bra and ket rings must have the same length')
        for b, k in zip(self.bra, self.sites):
            if Counter(b.qns[1]) != Counter(k.qns[1]):
                raise ValueError('bra and ket physical bases must match')
        self.cuts = tuple(_NormCut(b.qns[0], k.qns[0]) for b, k in zip(self.bra, self.sites))
        self.entries = tuple(self._compile(i) for i in range(len(self.sites)))
        self.transfers = tuple(self.site_transfer(i) for i in range(len(self.sites)))

    def _compile(self, i):
        left, right = self.cuts[i], self.cuts[(i+1) % len(self.sites)]
        entries = []
        for bk, b in self.bra[i].data.items():
            for kk, k in self.sites[i].data.items():
                if bk[1] != kk[1]:
                    continue
                for two_j in sorted(left.sizes.keys() & right.sizes.keys()):
                    lb, rb = (bk[0], kk[0], two_j), (bk[2], kk[2], two_j)
                    if lb not in left.blocks or rb not in right.blocks:
                        continue
                    coefficient = _norm_spin_transfer(*map(_sector_irrep,
                        (bk[0], kk[0], bk[1], bk[2], kk[2])), SU2Irrep(two_j))
                    if coefficient == 0.:
                        continue
                    entries.append(_NormEntry(bk, kk, two_j, left.blocks[lb][0],
                        right.blocks[rb][0], (b.shape[0], k.shape[0], b.shape[2], k.shape[2]), coefficient))
        return tuple(entries)

    def site_transfer(self, site, ket_blocks=None, bra_blocks=None):
        ket = self.sites[site].data if ket_blocks is None else ket_blocks
        bra = self.bra[site].data if bra_blocks is None else bra_blocks
        dtype = np.result_type(*[a.dtype for a in (*ket.values(), *bra.values())], complex)
        left, right = self.cuts[site], self.cuts[(site+1) % len(self.sites)]
        result = {j: np.zeros((n, right.sizes[j]), dtype=dtype)
                  for j, n in left.sizes.items() if j in right.sizes}
        for entry in self.entries[site]:
            block = np.einsum('apr,lps->alrs', bra[entry.bra_key].conj(), ket[entry.ket_key], optimize=True)
            result[entry.two_j][entry.left, entry.right] += entry.coefficient*block.reshape(
                entry.shape[0]*entry.shape[1], entry.shape[2]*entry.shape[3])
        return result

    def local_action(self, site, ket_blocks=None):
        """Euclidean adjoint metric action in the bra source coordinates."""
        ket = self.sites[site].data if ket_blocks is None else ket_blocks
        environment = self.environment(site)
        dtype = np.result_type(*[a.dtype for a in ket.values()], complex)
        out = {key: np.zeros_like(a, dtype=dtype) for key, a in self.bra[site].data.items()}
        for entry in self.entries[site]:
            if entry.two_j not in environment:
                continue
            db, dk, rb, rk = entry.shape
            block = environment[entry.two_j][entry.right, entry.left].reshape(rb, rk, db, dk)
            out[entry.bra_key] += (entry.two_j+1)*entry.coefficient*np.einsum(
                'rsal,lps->apr', block, ket[entry.ket_key], optimize=True)
        return out


@lru_cache(maxsize=None)
def _triple_basis(bra, ket, operator, intermediate, total):
    return np.einsum('kwc,bcz->wbkz', cg_tensor(ket, operator, intermediate),
                     _dual_pair(bra, intermediate, total), optimize=True)


@lru_cache(maxsize=None)
def _operator_spin_tensor(output, input, rank):
    return np.array([[[(-1.)**((input.two_j-mi)//2)*clebsch_gordan(
        output, input, rank, mo, -mi, m) for mi in ordered_two_m_values(input)]
        for mo in ordered_two_m_values(output)] for m in ordered_two_m_values(rank)])


@lru_cache(maxsize=None)
def _operator_spin_transfer(lb, lk, pb, pk, rb, rk, wl, wp, wr, cl, cr, total):
    left = _triple_basis(lb, lk, wl, cl, total)
    right = _triple_basis(rb, rk, wr, cr, total)
    return float(np.einsum('xalm,aob,lpr,xzy,zop,ybrm->', left.conj(),
        cg_tensor(lb, pb, rb).conj(), cg_tensor(lk, pk, rk), cg_tensor(wl, wp, wr),
        _operator_spin_tensor(pb, pk, wp), right, optimize=True)/total.dim)


class _OperatorCut:
    def __init__(self, bra_sectors, ket_sectors, operator_sectors):
        self.blocks, self.sizes = {}, {}
        for qw, dw in sorted(Counter(operator_sectors).items()):
            jw = _sector_irrep(qw)
            for qb, db in sorted(Counter(bra_sectors).items()):
                jb = _sector_irrep(qb)
                for qk, dk in sorted(Counter(ket_sectors).items()):
                    jk = _sector_irrep(qk)
                    for intermediate in range(abs(jk.two_j-jw.two_j), jk.two_j+jw.two_j+1, 2):
                        for two_j in range(abs(intermediate-jb.two_j), intermediate+jb.two_j+1, 2):
                            offset = self.sizes.get(two_j, 0)
                            key = (qw, qb, qk, intermediate, two_j)
                            self.blocks[key] = (slice(offset, offset+dw*db*dk), (dw, db, dk))
                            self.sizes[two_j] = offset+dw*db*dk


@dataclass(frozen=True)
class _OperatorEntry:
    bra_key: tuple
    ket_key: tuple
    two_j: int
    left: slice
    right: slice
    shape: tuple
    kernel: np.ndarray


class CyclicReducedOperator(_CyclicTransferChain):
    """Scalar MPO between reduced rings, including all cyclic fusion channels.

    The operator MPO has ordinary unit endpoints, while bra and ket virtual
    spaces actually close. A cut retains both total transfer spin and the
    intermediate ket/operator fusion channel. No global or local variational
    magnetic tensor is expanded by this contraction.
    """

    def __init__(self, sites, mpo, *, bra=None):
        self.sites = _validate_cycle(sites)
        self.bra = self.sites if bra is None else _validate_cycle(bra)
        self.mpo = mpo
        if len(self.bra) != len(self.sites) or len(mpo.sites) != len(self.sites):
            raise ValueError('bra, ket and MPO must have the same length')
        _validate_cycle(mpo.sites)
        if hasattr(mpo, 'physical_bases'):
            bases, self.local_channels = mpo.physical_bases, mpo.channels_by_site
        else:
            bases = (mpo.physical_basis,)*len(self.sites)
            self.local_channels = (mpo.channels,)*len(self.sites)
        if len(bases) != len(self.sites) or len(self.local_channels) != len(self.sites):
            raise ValueError('MPO local metadata must cover every ring site')
        for b, k, basis in zip(self.bra, self.sites, bases):
            expected = Counter(dict(zip(basis.sectors, basis.multiplicities)))
            if Counter(b.qns[1]) != expected or Counter(k.qns[1]) != expected:
                raise ValueError('ring physical basis does not match the native MPO')
        self.cuts = tuple(_OperatorCut(b.qns[0], k.qns[0], w.qns[0])
                          for b, k, w in zip(self.bra, self.sites, mpo.sites))
        self.entries = tuple(self._compile(i) for i in range(len(self.sites)))
        self.transfers = tuple(self.site_transfer(i) for i in range(len(self.sites)))

    def _compile(self, i):
        from collections import defaultdict
        left, right = self.cuts[i], self.cuts[(i+1) % len(self.sites)]
        channels = defaultdict(list)
        for c in self.local_channels[i]:
            channels[c.output_sector, c.input_sector, c.sector].append(c)
        kernels = {}
        for bk, b in self.bra[i].data.items():
            for kk, k in self.sites[i].data.items():
                for (wl, wp, wr), w in self.mpo.sites[i].data.items():
                    physical = channels[bk[1], kk[1], wp]
                    if not physical:
                        continue
                    for lk, (ls, lshape) in left.blocks.items():
                        if lk[:3] != (wl, bk[0], kk[0]):
                            continue
                        for rk, (rs, rshape) in right.blocks.items():
                            if rk[:3] != (wr, bk[2], kk[2]) or rk[-1] != lk[-1]:
                                continue
                            coefficient = _operator_spin_transfer(*map(_sector_irrep,
                                (bk[0], kk[0], bk[1], kk[1], bk[2], kk[2], wl, wp, wr)),
                                SU2Irrep(lk[-2]), SU2Irrep(rk[-2]), SU2Irrep(lk[-1]))
                            if coefficient == 0.:
                                continue
                            key = (bk, kk, lk, rk)
                            if key not in kernels:
                                kernels[key] = np.zeros((w.shape[0], w.shape[2], b.shape[1], k.shape[1]), dtype=w.dtype)
                            for c in physical:
                                kernels[key][:, :, c.output_copy, c.input_copy] += coefficient*w[:, c.copy, :]
        return tuple(_OperatorEntry(bk, kk, lk[-1], left.blocks[lk][0], right.blocks[rk][0],
            left.blocks[lk][1]+right.blocks[rk][1], kernel)
            for (bk, kk, lk, rk), kernel in kernels.items() if np.any(kernel))

    def site_transfer(self, site, ket_blocks=None, bra_blocks=None):
        ket = self.sites[site].data if ket_blocks is None else ket_blocks
        bra = self.bra[site].data if bra_blocks is None else bra_blocks
        dtype = np.result_type(*[a.dtype for a in (*ket.values(), *bra.values())], complex)
        left, right = self.cuts[site], self.cuts[(site+1) % len(self.sites)]
        result = {j: np.zeros((n, right.sizes[j]), dtype=dtype)
                  for j, n in left.sizes.items() if j in right.sizes}
        for entry in self.entries[site]:
            block = np.einsum('xyop,aob,lpr->xalybr', entry.kernel,
                bra[entry.bra_key].conj(), ket[entry.ket_key], optimize=True)
            result[entry.two_j][entry.left, entry.right] += block.reshape(
                np.prod(entry.shape[:3]), np.prod(entry.shape[3:]))
        return result

    def local_action(self, site, ket_blocks=None):
        ket = self.sites[site].data if ket_blocks is None else ket_blocks
        environment = self.environment(site)
        dtype = np.result_type(*[a.dtype for a in ket.values()], complex)
        out = {key: np.zeros_like(a, dtype=dtype) for key, a in self.bra[site].data.items()}
        for entry in self.entries[site]:
            if entry.two_j not in environment:
                continue
            block = environment[entry.two_j][entry.right, entry.left].reshape(entry.shape[3:]+entry.shape[:3])
            out[entry.bra_key] += (entry.two_j+1)*np.einsum('ybrxal,xyop,lpr->aob',
                block, entry.kernel, ket[entry.ket_key], optimize=True)
        return out

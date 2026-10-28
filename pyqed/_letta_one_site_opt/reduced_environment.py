"""Native multiplicity-space Hamiltonian environments for SU(2) LETTA.

Magnetic sums occur only in cached, state-independent structural coefficients.
All variational tensors, boundary environments, and local actions are reduced.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CompiledReducedTransfer(ReducedTransfer):
    """Compiled, value-independent transfer with genuinely immutable storage.

    A read-only owning ndarray can be made writable again. A bytes-backed
    array cannot; aliases therefore cannot silently change a cached kernel.
    Mutable ReducedTransfer instances remain supported and are hashed afresh.
    """
    _kernel_storage: bytes = field(init=False, repr=False, compare=False)
    _kernel_digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        import hashlib
        array = np.ascontiguousarray(self.kernel)
        storage = array.tobytes()
        kernel = np.frombuffer(storage,dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self,'kernel',kernel)
        object.__setattr__(self,'_kernel_storage',storage)
        object.__setattr__(self,'_kernel_digest',hashlib.blake2b(storage).digest())

    def __reduce__(self):
        # Re-enter the constructor on load: NumPy's default unpickling can
        # otherwise restore a writable owning array and duplicate the bytes.
        return (type(self),(self.bra_key,self.ket_key,self.left_key,self.right_key,self.kernel))

    def immutable_digest(self):
        # Shapes/dtypes are separately part of the cache's structural key.
        # A noncontiguous reinterpretation still uses ordinary value hashing.
        base = self.kernel
        while isinstance(base,np.ndarray):
            base = base.base
        if (base is self._kernel_storage and self.kernel.flags.c_contiguous
                and self.kernel.nbytes == len(self._kernel_storage)):
            return self._kernel_digest
        return None


def compile_transfers(site, mpo_site, channels):
    """Compile immutable spin structure once for a fixed sector allocation."""
    groups = defaultdict(list)
    for channel in channels:
        groups[channel.sector].append(channel)
    # Physical channels are fixed for the MPO, independent of bra/ket virtual
    # sectors. Index compatible blocks once instead of filtering them for
    # every pair of site blocks. Preserve MPO/channel accumulation order.
    compatible = defaultdict(list)
    for (wl, wp, wr), w in mpo_site.data.items():
        by_physical = defaultdict(list)
        for channel in groups[wp]:
            by_physical[channel.output_sector, channel.input_sector].append(channel)
        for physical, local_channels in by_physical.items():
            compatible[physical].append((wl, wp, wr, w, local_channels))
    kernels = {}
    for bra_key, bra in site.data.items():
        lb, pb, rb = bra_key
        for ket_key, ket in site.data.items():
            lk, pk, rk = ket_key
            for wl, wp, wr, w, local_channels in compatible.get((pb, pk), ()):
                coefficient = spin_transfer(*map(_sector_irrep,
                    (lb, lk, pb, pk, rb, rk, wl, wp, wr)))
                if abs(coefficient) < 1e-14:
                    continue
                key = (bra_key, ket_key, (wl, lb, lk), (wr, rb, rk))
                if key not in kernels:
                    kernels[key] = np.zeros(
                        (w.shape[0], w.shape[2], bra.shape[1], ket.shape[1]), dtype=w.dtype)
                kernel = kernels[key]
                for channel in local_channels:
                    kernel[:, :, channel.output_copy, channel.input_copy] += coefficient*w[:, channel.copy, :]
    return tuple(CompiledReducedTransfer(*key, value) for key, value in kernels.items()
                 if np.any(value))


def cached_transfers(mpo, i, site):
    cache = getattr(mpo, '_transfer_cache', None)
    if cache is None:
        cache = {}
        object.__setattr__(mpo, '_transfer_cache', cache)
    key = (i, tuple((k, a.shape[1]) for k, a in site.data.items()))
    if key not in cache:
        compiled = compile_transfers(site, mpo.sites[i], mpo.channels)
        # CBE changes layouts. Retain only the current and previous layout per
        # site; live environment chains own the kernels they still need.
        prior = [old for old in cache if old[0] == i]
        for old in prior[:-1]:
            del cache[old]
        cache[key] = compiled
    else:
        cache[key] = cache.pop(key)
    return cache[key]


def chain_transfers(sites,mpo):
    """Cache global structural pruning, retaining inputs to validate identity."""
    import os
    transfers=tuple(cached_transfers(mpo,i,a) for i,a in enumerate(sites))
    enabled=os.environ.get('PYQED_OPERATOR_PATHS','1')=='1'
    conditional=int(os.environ.get('PYQED_OPERATOR_PENDING_EDGES','0'))
    # The coarse graph depends only on sector/channel transitions. Including
    # pending-label fingerprints there would flush valid exterior caches when
    # an exactly constant tie is removed from a fixed tensor view.
    pending=(tuple(a.metadata.get('letta_frontier_support') for a in sites)
             if conditional>0 else None)
    key=(enabled,conditional,pending,
         tuple(id(t) for t in transfers),tuple(sites[0].qns[0]),tuple(sites[-1].qns[2]))
    cache=getattr(mpo,'_operator_paths_cache',None)
    if cache is None or cache[0]!=key:
        if enabled:
            from .operator_paths import prune_transfers
            filtered,report=prune_transfers(transfers,
                _boundary(sites[0],mpo.sites[0],0),_boundary(sites[-1],mpo.sites[-1],2))
        else:filtered,report=transfers,dict(disabled=True)
        if conditional>0:
            from .operator_paths import prune_conditioned_transfers
            filtered,extra=prune_conditioned_transfers(sites,filtered,
                _boundary(sites[0],mpo.sites[0],0),_boundary(sites[-1],mpo.sites[-1],2),max_edges=conditional)
            report={**report,**extra}
        cache=(key,transfers,filtered,report)
        object.__setattr__(mpo,'_operator_paths_cache',cache)
    return cache[2],key,cache[3]


def _boundary(site, mpo_site, axis):
    qns = tuple(dict.fromkeys(mpo_site.qns[axis]))
    if len(qns) != 1 or _sector_irrep(qns[0]).two_j != 0:
        raise ValueError('native scalar MPO requires spin-zero open boundaries')
    return {(qns[0], q, q): np.eye(r)[None]
            for q, r in Counter(site.qns[axis]).items()}


@lru_cache(maxsize=None)
def _environment_pool(workers):
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=workers,thread_name_prefix='letta-environment')


def _advance_single_physical(incoming, kernel, bra, ket, lr):
    """Absorb physical-copy-one blocks with three exact matrix products.

    SU(2) spatial-orbital reduced physical sectors each have one copy. Their
    transfer is sum_x K[y,x] B[a,b] E[x,b,c] A[c,d]. Pick whether to contract
    K before or after B/E/A by the actual multiplicity and channel dimensions.
    No dense local Hamiltonian, magnetic expansion or contraction planner.
    """
    bra, ket = np.asarray(bra)[:,0,:], np.asarray(ket)[:,0,:]
    left = bra.conj().T if lr else bra.conj()
    right = ket if lr else ket.T
    channels = kernel[:,:,0,0].T if lr else kernel[:,:,0,0]
    count_out, count_in = channels.shape
    a, b = left.shape
    c, d = right.shape
    left_first = a*b*c+a*c*d <= b*c*d+a*b*d
    per_channel = min(a*b*c+a*c*d, b*c*d+a*b*d)
    first_cost = count_out*count_in*b*c + count_out*per_channel
    last_cost = count_in*per_channel + count_out*count_in*a*d
    def absorb(value):
        return ((left@value)@right if left_first else left@(value@right))
    if first_cost <= last_cost:
        middle = (channels@incoming.reshape(count_in,b*c)).reshape(count_out,b,c)
        return absorb(middle)
    middle = absorb(incoming)
    return (channels@middle.reshape(count_in,a*d)).reshape(count_out,a,d)


def _advance_environment(environment,site,transfers,direction, *, structured=False):
    """Independent output-sector groups; preserve each group's term order."""
    import os
    from .numpy_contractions import accumulate_einsum
    from .boundary_structure import BoundaryProducts
    products=BoundaryProducts(site,direction) if structured else None
    bra_blocks=({k:(a.conj() if np.iscomplexobj(a) else a) for k,a in site.data.items()}
                if structured else None)
    lr=direction=='lr';groups={}
    for t in transfers:
        incoming=t.left_key if lr else t.right_key
        if incoming in environment:
            output=t.right_key if lr else t.left_key
            groups.setdefault(output,[]).append(t)
    def build(item):
        key,terms=item;value=None
        for t in terms:
            incoming=environment[t.left_key if lr else t.right_key]
            if products is not None:
                # Transpose MPO channel axes for a left absorption.
                transfer=t if not lr else ReducedTransfer(t.bra_key,t.ket_key,t.left_key,t.right_key,t.kernel.transpose(1,0,2,3))
                sparse=products.accumulate(incoming,transfer,1./_sector_irrep(t.bra_key[2 if lr else 0]).dim,value)
                if sparse is not None:
                    value=sparse;continue
            if t.kernel.shape[2:] == (1,1):
                contribution = _advance_single_physical(incoming, t.kernel,
                    site.data[t.bra_key], site.data[t.ket_key], lr)
                contribution /= _sector_irrep(t.bra_key[2 if lr else 0]).dim
                if value is None:
                    value = contribution
                elif np.result_type(value, contribution) == value.dtype:
                    value += contribution
                else:
                    value = value + contribution
                continue
            equation='xal,xyop,aob,lpr->ybr' if lr else 'ybr,xyop,aob,lpr->xal'
            value=accumulate_einsum(equation,(incoming,t.kernel,
                (bra_blocks[t.bra_key] if bra_blocks is not None else site.data[t.bra_key].conj()),site.data[t.ket_key]),
                alpha=1./_sector_irrep(t.bra_key[2 if lr else 0]).dim,out=value)
        return key,value
    workers=int(os.environ.get('PYQED_ENVIRONMENT_THREADS',os.environ.get('PYQED_SECTOR_THREADS','1')))
    if workers<1:raise ValueError('environment worker count must be positive')
    if workers==1 or len(groups)<2:return dict(map(build,groups.items()))
    return dict(_environment_pool(min(workers,len(groups))).map(build,groups.items()))


def advance_left(environment,site,transfers, *, structured=False):
    return _advance_environment(environment,site,transfers,'lr',structured=structured)


def advance_right(environment,site,transfers, *, structured=False):
    return _advance_environment(environment,site,transfers,'rl',structured=structured)


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
    def build(cls, sites, mpo, *, lazy=False,transfers=None):
        """Build boundaries eagerly, or defer them to exact ``ensure`` calls."""
        sites = tuple(sites)
        if len(sites) != len(mpo.sites):
            raise ValueError('one reduced MPO core is required per site')
        if transfers is None:transfers=chain_transfers(sites,mpo)[0]
        if lazy:
            n = len(sites)
            left = [_boundary(sites[0], mpo.sites[0], 0)] + [None]*n
            right = [None]*n + [_boundary(sites[-1], mpo.sites[-1], 2)]
            chain = cls(sites, mpo, transfers, left, right, [0.]*(n+1), [0.]*(n+1))
            chain.left_valid, chain.right_valid = 0, n
            return chain
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

    def prepare_local_action(self, site, *, shapes=None):
        """Bind an immutable reduced Hamiltonian schedule for compiled actions."""
        from .._letta_two_site_opt.sector_operator import prepare_sector_operator
        self.ensure(site, site+1)
        terms=[]
        for transfer in self.transfers[site]:
            left=self.left[site].get(transfer.left_key)
            right=self.right[site+1].get(transfer.right_key)
            if left is not None and right is not None:
                terms.append((transfer.bra_key,transfer.ket_key,left,transfer.kernel,right))
        if shapes is None:
            shapes={key:a.shape for key,a in self.sites[site].data.items()}
        scale=np.exp(self.left_log_scales[site]+self.right_log_scales[site+1])
        return prepare_sector_operator(shapes,terms,scale)

    def pair_action(self, site, blocks):
        """Apply a cached compiled pair schedule; rebuild after site replacement."""
        signature=(int(site),tuple((key,a.shape) for key,a in blocks.items()))
        if not hasattr(self,'_prepared_pairs'):
            self._prepared_pairs={}
        if signature not in self._prepared_pairs:
            self._prepared_pairs[signature]=self.prepare_pair_action(site,blocks)
        return self._prepared_pairs[signature](blocks)

    def prepare_pair_action(self, site, blocks):
        from .._letta_two_site_opt.sector_operator import prepare_pair_sector_operator
        self.ensure(site,site+2)
        right_by_middle=defaultdict(list)
        for transfer in self.transfers[site+1]:
            right_by_middle[transfer.left_key].append(transfer)
        terms=[]
        for first in self.transfers[site]:
            left=self.left[site].get(first.left_key)
            if left is None:
                continue
            for second in right_by_middle[first.right_key]:
                right=self.right[site+2].get(second.right_key)
                bra=first.bra_key+second.bra_key[1:]
                ket=first.ket_key+second.ket_key[1:]
                if right is None or bra not in blocks or ket not in blocks:
                    continue
                terms.append((bra,ket,left,first.kernel,second.kernel,right,
                              1./_sector_irrep(first.bra_key[2]).dim))
        scale=np.exp(self.left_log_scales[site]+self.right_log_scales[site+2])
        return prepare_pair_sector_operator({key:a.shape for key,a in blocks.items()},terms,scale)

    def _pair_action_reference(self, site, blocks):
        """Python contraction reference for validation of compiled pair actions."""
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
        self._prepared_pairs={}
        previous=chain_transfers(self.sites,self.mpo)[1]
        super().replace_sites(changes)
        self.transfers,current,_=chain_transfers(self.sites,self.mpo)
        if current!=previous:
            # A new sector can revive a formerly disconnected operator channel
            # anywhere in the chain, not only next to the changed tensor.
            self.left_valid,self.right_valid=0,len(self.sites)
            self.left[0]=_boundary(self.sites[0],self.mpo.sites[0],0)
            self.right[-1]=_boundary(self.sites[-1],self.mpo.sites[-1],2)

    def _advance_left(self, i):
        return advance_left(self.left[i], self.sites[i], self.transfers[i])

    def _advance_right(self, i):
        return advance_right(self.right[i+1], self.sites[i], self.transfers[i])

"""Exact sparse frontier contraction and streamed sector QR.

The frontier embedding is a scatter map, not a dense variational tensor.
Apply the incoming QR factor to that map before allocating a dense output.
Sector QR follows Trefethen and Bau, Numerical Linear Algebra (1997),
doi:10.1137/1.9780898719574; this is an implementation adaptation, without
rank truncation or a new numerical cutoff. Floating-point order may differ.
"""
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from .reduced_frontier import ReducedFrontier
from .reduced_symmetry import _sector_irrep


def contract_frontier_left(embedding, blocks, left):
    """Contract a reduced boundary into a scatter embedding, omitting zeros."""
    source = embedding.pack_source(blocks)
    result = {}
    for key, shape in embedding.target_layout.shapes.items():
        start, stop = embedding.target_layout.offsets[key]
        take = (embedding.target_indices >= start) & (embedding.target_indices < stop)
        indices = embedding.target_indices[take] - start
        rows, columns = np.divmod(indices, shape[1]*shape[2])
        matrix = coo_matrix((source[embedding.source_indices[take]], (rows, columns)),
                            shape=(shape[0], shape[1]*shape[2])).tocsr()
        result[key] = (matrix.T @ left[key[0]].T).T.reshape(
            left[key[0]].shape[0], shape[1], shape[2])
    return result


def evaluation_state(state):
    """Remove only exactly constant tie axes from an expectation-value view.

    The variational state and its local tangent spaces are never modified.
    No tolerance is used: even arbitrarily small nonzero tie dependence stays.
    """
    from copy import copy
    tensors, neighborhoods = [], []
    for i, tensor in enumerate(state.tensors):
        blocks = dict(tensor)
        neighborhood = list(state.site_neighborhood(i))
        for axis in range(len(neighborhood), 1, -1):
            if all(np.array_equal(a, np.broadcast_to(np.take(a, [0], axis=axis), a.shape))
                   for a in blocks.values()):
                blocks = {k: np.take(a, 0, axis=axis) for k,a in blocks.items()}
                del neighborhood[axis-1]
        tensors.append(blocks)
        neighborhoods.append(tuple(neighborhood))
    result = copy(state)
    result.tensors = tensors
    result._neighborhoods = tuple(neighborhoods)
    return result


def frontier_qr(state, *, norm_only=False, start=0, left=None, boundary_callback=None):
    """Yield left-canonical sites, or only the final core for a norm.

    At most one raw contracted site and the current QR boundary are retained.
    No dense full frontier chain is constructed or copied. QR keeps all rows
    returned by reduced QR, including numerical null directions.
    """
    frontier = ReducedFrontier.from_state(state)
    if left is None:
        left = {q: np.eye(d) for q, d in Counter(state.left_virtual_sectors(0)).items()}
    for i in range(start,state.nsites):
        embedding = frontier.site_embedding(state, i)
        raw = scatter_blocks(embedding, state.tensors[i])
        blocks = {key:(np.asarray(a) if i==0 else np.tensordot(left[key[0]],np.asarray(a),axes=(1,0)))
                  for key,a in raw.items()}
        left_qns = [q for q, factor in left.items() for _ in range(factor.shape[0])]
        if i == state.nsites-1:
            right_qns = list(state.right_virtual_sectors(i))
        else:
            groups = defaultdict(list)
            for key, block in blocks.items():
                groups[key[2]].append((key, block))
            next_left, data, right_qns = {}, {}, []
            for qr, entries in groups.items():
                matrix = np.concatenate([a.reshape(-1, a.shape[2]) for _, a in entries])
                if norm_only:
                    factor = np.linalg.qr(matrix, mode='r')
                else:
                    unitary, factor = np.linalg.qr(matrix, mode='reduced')
                    offset = 0
                    for key, a in entries:
                        rows = a.shape[0]*a.shape[1]
                        data[key] = unitary[offset:offset+rows].reshape(a.shape[:2]+(factor.shape[0],))
                        offset += rows
                next_left[qr] = factor
                right_qns.extend([qr]*factor.shape[0])
            left, blocks = next_left, data
            if norm_only:
                continue
        physical = [q for q,d in zip(state.physical_basis.sectors,
                    state.physical_basis.multiplicities) for _ in range(d)]
        yield NonabelianTensor(blocks, [left_qns, physical, right_qns], [-1,1,1],
                              metadata={'physical_basis':'fully_reduced_su2'})
        if boundary_callback is not None and i < state.nsites-1:
            boundary_callback(i+1,left)


def frontier_norm(state):
    from .reduced_norm import _normalize, advance_norm_left, _boundary
    environment, scale = None, 0.
    for site in frontier_qr(evaluation_state(state) if ReducedFrontier.from_state(state).dense_bytes(state)>64*1024**2 else state):
        if environment is None: environment = _boundary(site, 0)
        environment, scale = _normalize(advance_norm_left(environment, site), scale)
    return float(np.real(sum(_sector_irrep(q).dim*np.trace(a)
        for q,a in environment.items())*np.exp(scale))) / (state.target_two_j+1)


def frontier_energy(state, mpo, *, cache=None):
    """Stream exact canonical cores into a reduced MPO expectation."""
    from .reduced_environment import _boundary, cached_transfers, advance_left
    from .reduced_norm import _normalize, advance_norm_left, _boundary as norm_boundary
    environment, log_scale, norm, norm_scale = None, 0., None, 0.
    view=evaluation_state(state) if ReducedFrontier.from_state(state).dense_bytes(state)>64*1024**2 else state
    start,left,signature=0,None,None
    if cache is not None:
        signature=cache.signatures(view,mpo)
        found=cache.nearest(signature)
        if found is not None:
            start,(left,environment,log_scale,norm,norm_scale)=found
    def store(cut,factor):
        if cache is not None:
            cache.store(signature,cut,(factor,environment,log_scale,norm,norm_scale))
    for i, site in enumerate(frontier_qr(view,start=start,left=left,boundary_callback=store),start=start):
        if environment is None:
            environment = _boundary(site, mpo.sites[0], 0)
            norm = norm_boundary(site, 0)
        environment, log_scale = _normalize(
            advance_left(environment, site, cached_transfers(mpo, i, site)), log_scale)
        norm, norm_scale = _normalize(advance_norm_left(norm, site), norm_scale)
    numerator = sum(_sector_irrep(qb).dim*np.trace(a[0])
        for (_w,qb,qk),a in environment.items() if qb==qk)*np.exp(log_scale)
    denominator = sum(_sector_irrep(q).dim*np.trace(a) for q,a in norm.items())*np.exp(norm_scale)
    return numerator, denominator


class EnergyBoundaryCache:
    """Bounded prefixes of the independent streamed QR energy contraction.

    Reuses unchanged prefixes by values, including both H and norm boundaries.
    Uses the same sector QR (Trefethen and Bau, 1997,
    doi:10.1137/1.9780898719574), transfer order, and normalization as a fresh
    contraction. This is exact memoization, with no cutoff or local-H shortcut.
    """
    def __init__(self,max_bytes=64*1024**2):
        from collections import OrderedDict
        self.entries=OrderedDict();self.max_bytes=max_bytes;self.nbytes=0
        self.hits=0;self.mpo=None;self.structure=None

    def signatures(self,state,mpo):
        import hashlib
        structure=repr((state.neighborhoods,state.bond_sectors,state.physical_basis,state.target_two_j))
        if self.mpo is not mpo or self.structure!=structure:
            self.entries.clear();self.nbytes=0;self.mpo=mpo;self.structure=structure
        signatures=[]
        for blocks in state.tensors:
            digest=hashlib.blake2b()
            for key,a in blocks.items():
                digest.update(repr((key,a.shape,a.dtype.str)).encode())
                digest.update(np.ascontiguousarray(a).view(np.uint8))
            signatures.append(digest.digest())
        return tuple(signatures)

    def nearest(self,signature):
        for cut in range(len(signature)-1,0,-1):
            key=signature[:cut]
            if key in self.entries:
                self.hits+=1;self.entries.move_to_end(key)
                return cut,self.entries[key][0]
        return None

    def store(self,signature,cut,value):
        size=sum(a.nbytes for index in (0,1,3) for a in value[index].values())
        if size>self.max_bytes:return
        key=signature[:cut]
        if key in self.entries:self.nbytes-=self.entries.pop(key)[1]
        while self.entries and self.nbytes+size>self.max_bytes:
            _,(_,old)=self.entries.popitem(last=False);self.nbytes-=old
        self.entries[key]=(value,size);self.nbytes+=size


class ScatterBlock:
    """Immutable exact sparse three-leg tensor; dense conversion is explicit."""
    ndim = 3
    def __init__(self, shape, indices, values):
        self.shape = tuple(shape)
        self.indices = np.asarray(indices)
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.size = int(np.prod(shape))
    @property
    def nbytes(self):
        return self.indices.nbytes+self.values.nbytes
    def copy(self):
        return self
    def conj(self):
        return ScatterBlock(self.shape,self.indices,self.values.conj())
    def __array__(self, dtype=None, copy=None):
        out=np.zeros(self.size,dtype=self.dtype if dtype is None else dtype)
        out[self.indices]=self.values
        return out.reshape(self.shape)
    def reshape(self,*shape):
        return np.asarray(self).reshape(*shape)


def scatter_blocks(embedding, blocks):
    source=embedding.pack_source(blocks)
    result={}
    for key,shape in embedding.target_layout.shapes.items():
        start,stop=embedding.target_layout.offsets[key]
        take=(embedding.target_indices>=start)&(embedding.target_indices<stop)
        result[key]=ScatterBlock(shape,embedding.target_indices[take]-start,
                                source[embedding.source_indices[take]])
    return result


def contract_block(block, left=None, right=None):
    """Preserve the established QR contraction order, one raw block at a time.

    Sparse BLAS changes cancellation in rank-deficient QR frames enough to
    alter capped nonlinear compression. For large compact right contractions,
    omit empty output rows but retain the complete contracted axis and the
    original einsum reduction. Other layouts retain the established path.
    """
    if (isinstance(block,ScatterBlock) and left is None and right is not None
            and block.size>=200000 and block.shape[-1]>0
            and block.dtype in (np.dtype('float64'),np.dtype('complex128'))
            and right.dtype in (np.dtype('float64'),np.dtype('complex128'))
            and np.all(np.isfinite(right))):
        width=block.shape[-1]
        rows,inverse=np.unique(block.indices//width,return_inverse=True)
        count=block.shape[0]*block.shape[1]
        if 2*len(rows)<=count:
            selected=np.zeros((len(rows),1,width),dtype=block.dtype)
            selected.reshape(-1)[inverse*width+block.indices%width]=block.values
            out=np.zeros((count,right.shape[0]),dtype=np.result_type(block.dtype,right.dtype))
            out[rows]=np.einsum('kr,lpr->lpk',right,selected)[:,0,:]
            return out.reshape(block.shape[:2]+(right.shape[0],))
    block=np.asarray(block)
    if left is None:return np.einsum('kr,lpr->lpk',right,block)
    if right is None:return np.einsum('kl,lpr->kpr',left,block)
    return np.einsum('al,lpr,br->apb',left,block,right)


def shallow_site(site):
    """Own the tensor dictionary while sharing immutable input arrays."""
    result=NonabelianTensor({},site.qns,site.dirs,fusion_legs=site.fusion_legs,
                           metadata=site.metadata)
    result.data=dict(site.data)
    return result


def merge_scatter_pair(a, b):
    """Retain the original pair contraction order, one pair block at a time."""
    return np.einsum('apm,mqb->apqb',np.asarray(a),np.asarray(b),optimize=True)


def state_signature(state):
    """Value-based key: in-place tensor changes must invalidate QR reuse."""
    import hashlib
    digest=hashlib.blake2b()
    digest.update(repr((state.neighborhoods,state.bond_sectors)).encode())
    for blocks in state.tensors:
        for key,a in blocks.items():
            digest.update(repr((key,a.shape,a.dtype.str)).encode())
            digest.update(np.ascontiguousarray(a).view(np.uint8))
    return digest.digest()


def factor_adjoint(gradient, block, side):
    """Contract a dense pair gradient against one sparse factor."""
    if not isinstance(block,ScatterBlock):
        return (np.einsum('apqb,mqb->apm',gradient,block.conj(),optimize=True)
                if side==0 else np.einsum('apm,apqb->mqb',block.conj(),gradient,optimize=True))
    l,p,q,r=gradient.shape
    if side==0:
        m=block.shape[0]
        rows,cols=np.divmod(block.indices,q*r)
        matrix=coo_matrix((block.values.conj(),(rows,cols)),shape=(m,q*r)).tocsr()
        return (matrix@gradient.reshape(l*p,q*r).T).T.reshape(l,p,m)
    m=block.shape[2]
    rows,cols=np.divmod(block.indices,m)
    matrix=coo_matrix((block.values.conj(),(rows,cols)),shape=(l*p,m)).tocsr()
    return (matrix.T@gradient.reshape(l*p,q*r)).reshape(m,q,r)


class CanonicalBoundaryCache:
    """Bounded exact QR/Hamiltonian boundary reuse, with value-based invalidation.

    Uses the same reduced QR contractions as the uncached reference (Trefethen
    and Bau, Numerical Linear Algebra, SIAM 1997, doi:10.1137/1.9780898719574).
    No rank cutoff or precision change; the bound evicts cached boundaries only.
    This is an implementation adaptation, not a new variational approximation.
    """
    def __init__(self, max_bytes=256*1024**2):
        from collections import OrderedDict
        self.entries=OrderedDict();self.max_bytes=max_bytes;self.nbytes=0
        self.hits=0;self.advances=0;self.mpo=None

    def signatures(self,sites,mpo):
        import hashlib
        if self.mpo is not mpo:
            self.entries.clear();self.nbytes=0;self.mpo=mpo
        signatures=[]
        for site in sites:
            h=hashlib.blake2b()
            h.update(repr((site.qns,site.dirs)).encode())
            for key,a in site.data.items():
                h.update(repr((key,a.shape,a.dtype.str)).encode())
                if isinstance(a,ScatterBlock):
                    h.update(np.ascontiguousarray(a.indices).view(np.uint8))
                    h.update(np.ascontiguousarray(a.values).view(np.uint8))
                else:h.update(np.ascontiguousarray(a).view(np.uint8))
            signatures.append(h.digest())
        return tuple(signatures)

    def key(self,direction,cut,signatures):
        return direction,cut,signatures[:cut] if direction=='lr' else signatures[cut:]

    def nearest(self,direction,cut,signatures):
        cuts=range(cut,0,-1) if direction=='lr' else range(cut,len(signatures))
        for i in cuts:
            key=self.key(direction,i,signatures)
            if key in self.entries:
                self.entries.move_to_end(key);self.hits+=1
                return i,self.entries[key][0]
        return None

    def store(self,direction,cut,signatures,factor,environment,scale):
        self.advances+=1
        size=sum(a.nbytes for a in factor.values())+sum(a.nbytes for a in environment.values())
        if size>self.max_bytes:return
        key=self.key(direction,cut,signatures)
        if key in self.entries:self.nbytes-=self.entries.pop(key)[1]
        while self.entries and self.nbytes+size>self.max_bytes:
            _,(_,old)=self.entries.popitem(last=False);self.nbytes-=old
        self.entries[key]=((factor,environment,scale),size);self.nbytes+=size


def canonical_environment(sites, center, mpo, *, last_center=None, cache=None, transform_active=True):
    """Consume canonical outer cores immediately into the active boundaries.

    No canonical outer chain is retained; each QR core is contracted once.
    The active interval and original compact outer cores remain available.
    """
    from .reduced_coordinates import _canonical_sites
    from .reduced_environment import (_boundary,cached_transfers,advance_left,
                                     advance_right,ReducedEnvironmentChain)
    from .reduced_norm import _normalize
    last_center=center if last_center is None else last_center
    boundaries={'lr':(_boundary(sites[0],mpo.sites[0],0),0.),
                'rl':(_boundary(sites[-1],mpo.sites[-1],2),0.)}
    seeds=None;signatures=None
    if cache is not None:
        signatures=cache.signatures(sites,mpo)
        from collections import Counter
        seeds=[(0,{q:np.eye(d) for q,d in Counter(sites[0].qns[0]).items()}),
               (len(sites),{q:np.eye(d) for q,d in Counter(sites[-1].qns[2]).items()})]
        for j,(direction,cut) in enumerate((('lr',center),('rl',last_center+1))):
            found=cache.nearest(direction,cut,signatures)
            if found is not None:
                i,(factor,environment,scale)=found
                seeds[j]=(i,factor);boundaries[direction]=(environment,scale)
    def store(direction,cut,factor):
        if cache is not None:cache.store(direction,cut,signatures,factor,*boundaries[direction])
    def consume(direction,i,site):
        environment,scale=boundaries[direction]
        advance=advance_left if direction=='lr' else advance_right
        boundaries[direction]=_normalize(advance(environment,site,cached_transfers(mpo,i,site)),scale)
    canonical,left,right=_canonical_sites(sites,center,last_center=last_center,
        consume=consume,keep_indices=set(range(center,last_center+1)),
        boundary_seeds=seeds,boundary_callback=store,transform_active=transform_active)
    chain=ReducedEnvironmentChain.build(canonical,mpo,lazy=True)
    chain.left[center],chain.left_log_scales[center]=boundaries['lr']
    chain.right[last_center+1],chain.right_log_scales[last_center+1]=boundaries['rl']
    chain.left_valid,chain.right_valid=center,last_center+1
    return canonical,left,right,chain


def raw_pair_environments(sites, center, mpo):
    """Build only the raw boundaries needed by one pair, releasing intermediates.

    The contraction sequence is the same as the uncompressed environment path.
    Compact tensor blocks are materialized by cached_einsum one transfer at a time.
    """
    from .reduced_environment import ReducedEnvironmentChain, _boundary, advance_left, advance_right, cached_transfers
    from .reduced_norm import ReducedNormChain, _normalize, _boundary as nb, advance_norm_left, advance_norm_right
    h=ReducedEnvironmentChain.build(sites,mpo,lazy=True)
    n=ReducedNormChain.build(sites,lazy=True)
    for direction,indices,cut in [('lr',range(center),center),('rl',range(len(sites)-1,center+1,-1),center+2)]:
        axis=0 if direction=='lr' else 2
        end=0 if direction=='lr' else len(sites)-1
        he,ne=_boundary(sites[end],mpo.sites[end],axis),nb(sites[end],axis)
        hs=ns=0.
        ha=advance_left if direction=='lr' else advance_right
        na=advance_norm_left if direction=='lr' else advance_norm_right
        for i in indices:
            site=shallow_site(sites[i])
            site.data={k:np.asarray(a) for k,a in site.data.items()}
            he,hs=_normalize(ha(he,site,cached_transfers(mpo,i,site)),hs)
            ne,ns=_normalize(na(ne,site),ns)
            del site
        name='left' if direction=='lr' else 'right'
        getattr(h,name)[cut]=he;getattr(n,name)[cut]=ne
        getattr(h,name+'_log_scales')[cut]=hs;getattr(n,name+'_log_scales')[cut]=ns
        setattr(h,name+'_valid',cut);setattr(n,name+'_valid',cut)
    return h,n

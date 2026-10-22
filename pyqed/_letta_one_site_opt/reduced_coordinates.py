"""Stable reduced one-site overlap and Hamiltonian coordinates."""
from collections import defaultdict, Counter
import numpy as np
from .reduced_symmetry import _sector_irrep
from .frontier_contraction import contract_block, shallow_site, scatter_blocks

def _canonical_sites(sites,center, *, last_center=None, consume=None, keep_indices=None, transform_active=True, boundary_seeds=None, boundary_callback=None, boundary_matmul=False, structured=False):
    from .frontier_contraction import contract_boundary_block
    from .boundary_structure import support_qr
    factorize = support_qr if structured else lambda a: np.linalg.qr(a, mode="reduced")
    boundary_contract=contract_boundary_block if boundary_matmul else contract_block
    last_center = center if last_center is None else last_center
    if not 0 <= center <= last_center < len(sites):
        raise ValueError("invalid active interval")
    original=sites
    sites=[shallow_site(a) for a in sites]
    left={q:np.eye(d) for q,d in Counter(sites[0].qns[0]).items()}
    left_start=0
    if boundary_seeds is not None:left_start,left=boundary_seeds[0]
    for i in range(left_start,center):
        tensor=sites[i];groups=defaultdict(list)
        for key,a in tensor.data.items():
            b=boundary_contract(a,left=left[key[0]])
            groups[key[2]].append((key,b))
        # Declared sectors with no incoming path have an exact zero-row factor.
        next_left={q:np.zeros((0,d)) for q,d in Counter(tensor.qns[2]).items()};data={}
        for qr,entries in groups.items():
            Q,R=factorize(np.concatenate([b.reshape(-1,b.shape[2]) for _,b in entries]))
            next_left[qr]=R;offset=0
            for key,b in entries:
                rows=b.shape[0]*b.shape[1]
                data[key]=Q[offset:offset+rows].reshape(b.shape[:2]+(R.shape[0],));offset+=rows
        tensor.data=data
        tensor.qns[0]=[q for q,R in left.items() for _ in range(R.shape[0])]
        tensor.qns[2]=[q for q,R in next_left.items() for _ in range(R.shape[0])]
        left=next_left
        if consume is not None:consume('lr',i,tensor)
        if boundary_callback is not None:boundary_callback('lr',i+1,left)
        if keep_indices is not None and i not in keep_indices:sites[i]=original[i]
    right={q:np.eye(d) for q,d in Counter(sites[-1].qns[2]).items()}
    right_start=len(sites)
    if boundary_seeds is not None:right_start,right=boundary_seeds[1]
    for i in range(right_start-1,last_center,-1):
        tensor=sites[i];groups=defaultdict(list)
        for key,a in tensor.data.items():
            b=boundary_contract(a,right=right[key[2]])
            weight=np.sqrt(_sector_irrep(key[2]).dim/_sector_irrep(key[0]).dim)
            groups[key[0]].append((key,b,weight))
        next_right={q:np.zeros((0,d)) for q,d in Counter(tensor.qns[0]).items()};data={}
        for ql,entries in groups.items():
            Q,R=factorize(np.concatenate([weight*b.transpose(2,1,0).reshape(-1,b.shape[0]) for _,b,weight in entries]))
            next_right[ql]=R;offset=0
            for key,b,weight in entries:
                rows=b.shape[1]*b.shape[2]
                data[key]=(Q[offset:offset+rows].reshape(b.shape[2],b.shape[1],R.shape[0]).transpose(2,1,0)/weight);offset+=rows
        tensor.data=data
        tensor.qns[0]=[q for q,R in next_right.items() for _ in range(R.shape[0])]
        tensor.qns[2]=[q for q,R in right.items() for _ in range(R.shape[0])]
        right=next_right
        if consume is not None:consume('rl',i,tensor)
        if boundary_callback is not None:boundary_callback('rl',i,right)
        if keep_indices is not None and i not in keep_indices:sites[i]=original[i]
    if not transform_active:return tuple(sites),left,right
    tensor=sites[center]
    if last_center == center:
        tensor.data={key:contract_block(a,left[key[0]],right[key[2]]) for key,a in tensor.data.items()}
    else:
        tensor.data={key:np.einsum('al,lpr->apr',left[key[0]],np.asarray(a)) for key,a in tensor.data.items()}
        last=sites[last_center]
        last.data={key:np.einsum('br,lpr->lpb',right[key[2]],np.asarray(a)) for key,a in last.data.items()}
    tensor.qns[0]=[q for q,R in left.items() for _ in range(R.shape[0])]
    sites[last_center].qns[2]=[q for q,R in right.items() for _ in range(R.shape[0])]
    return tuple(sites),left,right


class _OverlapFactor:
    """Exact scatter/tensor-product overlap map without its dense Jacobian."""
    def __init__(self, left, right, source, lmap, pmap, rmap, nphysical, size, weight):
        self.left, self.right = left, right
        self.source, self.lmap, self.pmap, self.rmap = source, lmap, pmap, rmap
        self.nphysical, self.size, self.weight = nphysical, size, weight

    def apply(self, vector):
        local = np.zeros((self.left.shape[1], self.nphysical, self.right.shape[1]),
                         dtype=np.result_type(vector, self.left, self.right))
        np.add.at(local, (self.lmap, self.pmap, self.rmap), vector[self.source])
        return self.weight*np.einsum('al,lpr,br->apb', self.left, local, self.right, optimize=True).ravel()

    def adjoint(self, vector):
        local = np.asarray(vector).reshape(self.left.shape[0], self.nphysical, self.right.shape[0])
        pulled = self.weight*np.einsum('al,apb,br->lpr', self.left.conj(), local,
                                       self.right.conj(), optimize=True)
        out = np.zeros(self.size, dtype=pulled.dtype)
        np.add.at(out, self.source, pulled[self.lmap, self.pmap, self.rmap])
        return out

    def dense(self):
        from .overlap_factor import overlap_factor_matrix
        return overlap_factor_matrix(self.left,self.right,self.source,self.lmap,
            self.pmap,self.rmap,self.nphysical,self.size,self.weight)

    @property
    def stored_elements(self):
        return sum(a.size for a in (self.left,self.right,self.source,self.lmap,self.pmap,self.rmap))


def _ordered_product(left, middle, right):
    """Exact three-matrix product with the smaller dense arithmetic count."""
    a, l = left.shape
    r, b = right.shape
    if a*l*r + a*r*b <= l*r*b + a*l*b:
        return (left @ middle) @ right
    return left @ (middle @ right)


class ReducedLocalCoordinates:
    """Orthonormal local coordinates from reduced QR overlap factors.

    Adapts standard QR/SVD least-squares coordinates (L. N. Trefethen and
    D. Bau III, Numerical Linear Algebra, SIAM, 1997,
    https://doi.org/10.1137/1.9780898719574) to SU(2) multiplicity blocks and
    disjoint physical-label configurations. QR preserves the auxiliary MPS;
    SVD removes directions below the configured relative metric cutoff.
    This is not a reproduction of a published LETTA optimizer and does not
    guarantee a global variational minimum. Neither a determinant-space frame
    nor a dense many-body Hamiltonian is constructed.

    The overlap map is stored as exact scatter indices and QR factors. Dense
    factors are materialized one block at a time for the same full SVD and
    global relative cutoff. Whitening uses the retained SVD factors directly;
    no extra rank truncation, lower precision, or Gram squaring is introduced.
    Fixed QR/SVD image maps are composed once per local problem when they
    fit a 16 MiB cache; otherwise the exact factored action is retained.
    This changes floating-point operation order, so equivalence is numerical,
    not bitwise. Fixed adjoints use a separate bounded 64 MiB cache, and
    exact three-matrix products choose the smaller dense MAC count. QR factors
    of identical sector/subset selections are shared within one local problem;
    no factor survives a tensor or gauge update. Retained singular-vector
    storage can still grow rapidly with D. Optional PYQED_QR_ABSORBED_MB
    enables exact absorption of these QR factors into fixed Hamiltonian
    boundaries within that storage budget; zero (default) retains the current
    route. This is an experimental local-action optimization, not a new rank
    truncation. The budget excludes original operators and packing temporaries.
    """
    def __init__(self,sites,embedding,site, *, mpo=None, retain_outer=True, boundary_cache=None, action_only=False, structured=False):
        self.site=site;self.embedding=embedding
        self._prepared_mpo=None
        self._source_sites=sites
        if mpo is not None:
            from .frontier_contraction import canonical_environment
            canonical,self.left,self.right,chain=canonical_environment(
                sites,site,mpo,cache=boundary_cache,transform_active=not action_only,structured=structured)
            shapes={key:(self.left[key[0]].shape[0],a.shape[1],self.right[key[2]].shape[0])
                    for key,a in sites[site].data.items()}
            self.hamiltonian_action=chain.prepare_local_action(site,shapes=shapes)
            # An action needs boundary values and active shapes, not the current
            # active tensor. Do not expose the untransformed chain as canonical.
            self.sites=None if action_only else canonical
            self.hamiltonian=None if action_only else chain
            self._prepared_mpo=mpo
        else:
            self.sites,self.left,self.right=_canonical_sites(sites,site,
                keep_indices=None if retain_outer else {site-1,site,site+1})
        emb=embedding
        self.dimension=emb.source_size;self.blocks=[];self.images=[]
        left_factors={};right_factors={}
        self.factorization_counts=dict(routes=len(emb.coordinate_routes),left=0,right=0)
        for key,indices,source,ul,lmap,up,pmap,ur,rmap,nphysical in emb.coordinate_routes:
            # Only geometry is cached. QR factors must follow the current state.
            left_key=(key[0],tuple(ul));right_key=(key[2],tuple(ur))
            if left_key not in left_factors:
                left_factors[left_key]=np.linalg.qr(self.left[key[0]][:,ul],mode='reduced')
                self.factorization_counts['left']+=1
            if right_key not in right_factors:
                right_factors[right_key]=np.linalg.qr(self.right[key[2]][:,ur],mode='reduced')
                self.factorization_counts['right']+=1
            ql,left=left_factors[left_key]
            qr,right=right_factors[right_key]
            factor_map=_OverlapFactor(left,right,source,lmap,pmap,rmap,len(up),len(indices),
                np.sqrt(_sector_irrep(key[2]).dim))
            self.blocks.append((indices,factor_map))
            self.images.append((key,up,ql,qr,nphysical))
        self.stored_elements=sum(F.stored_elements for _,F in self.blocks)
    def metric(self,v):
        out=np.zeros(self.dimension,dtype=np.result_type(v,complex))
        for sl,F in self.blocks:out[sl]=F.adjoint(F.apply(v[sl]))
        return out
    def restrict(self,active):
        """Restrict the source domain before forming any dense SVD blocks."""
        blocks,images=[],[]
        for (indices,factor),image in zip(self.blocks,self.images):
            take=np.flatnonzero(np.isin(indices,active))
            if not len(take):continue
            inverse=np.full(len(indices),-1,dtype=int);inverse[take]=np.arange(len(take))
            keep=inverse[factor.source]>=0
            blocks.append((indices[take],_OverlapFactor(factor.left,factor.right,
                inverse[factor.source[keep]],factor.lmap[keep],factor.pmap[keep],
                factor.rmap[keep],factor.nphysical,len(take),factor.weight)))
            images.append(image)
        self.blocks,self.images=blocks,images
        self.stored_elements=sum(f.stored_elements for _,f in blocks)
        self.tolerance=None

    def prepare(self,tolerance, *, singular_tolerance=None,max_workspace_mb=None):
        settings=(tolerance,singular_tolerance,max_workspace_mb)
        if getattr(self,'tolerance',None)==settings:
            return
        if max_workspace_mb is not None:
            # Conservative bound for dense blocks, retained SVD factors, and
            # LAPACK work arrays; this is not a bound on total solver memory.
            elements=sum(f.left.shape[0]*f.nphysical*f.right.shape[0]*f.size
                         for _,f in self.blocks)
            if 8*16*elements>max_workspace_mb*1024**2:
                raise MemoryError('factor SVD workspace exceeds max_workspace_mb')
        self.tolerance=settings
        self._absorbed_attempted=False
        self._absorbed_action=None
        self._block_preconditioner=None
        self._correction_maps=None
        factors=[];largest=0.
        for sl,factor_map in self.blocks:
            F=factor_map.dense()
            roots=np.linalg.norm(F,axis=0)
            inv=np.divide(1.,roots,out=np.zeros_like(roots),where=roots>0.)
            F*=inv[None,:]
            normalized=F
            del F
            if not np.all(np.isfinite(normalized)):
                raise FloatingPointError('Nonfinite normalized QR norm factor')
            try:
                u,s,vh=np.linalg.svd(normalized,full_matrices=False)
            except np.linalg.LinAlgError:
                from scipy.linalg import svd
                u,s,vh=svd(normalized,full_matrices=False,lapack_driver='gesvd')
            del normalized
            largest=max(largest,float(s.max(initial=0.)))
            factors.append((sl,roots,inv,s,vh,u))
        cutoff=largest*np.sqrt(max(tolerance,np.finfo(float).eps*self.dimension))
        self.maps=[];self.spectral_images=[];self.support=[];offset=0
        for i in range(len(factors)):
            sl,roots,inv,s,vh,u=factors[i]
            factors[i]=None
            threshold=cutoff if singular_tolerance is None else singular_tolerance*s.max(initial=0.)
            keep=s>threshold;s=s[keep];vh=vh[keep]
            self.maps.append((sl,slice(offset,offset+len(s)),inv,s,vh,roots));offset+=len(s)
            self.spectral_images.append(u[:,keep])
            self.support.append((sl,vh))
        self.rank=offset
        # Reuse the exact QR/SVD composition across Davidson matvecs. Bound
        # extra storage; wider frontier blocks keep the factored action.
        self.composed_images=[]
        remaining=16*1024**2
        for U,(key,up,left,right,np_) in zip(self.spectral_images,self.images):
            needed=left.shape[0]*len(up)*right.shape[0]*U.shape[1]*np.dtype(complex).itemsize
            if needed <= remaining:
                image=np.einsum('al,lprk,br->apbk',left,
                    U.reshape(left.shape[1],len(up),right.shape[1],U.shape[1]),
                    right,optimize=True).reshape(left.shape[0]*len(up)*right.shape[0],U.shape[1])
                self.composed_images.append(image)
                remaining-=needed
            else:
                self.composed_images.append(None)
        # Fixed adjoints belong to this prepared local problem, not a matvec.
        # Bound additional storage independently of Hamiltonian product caches.
        self.image_adjoints=[]
        remaining=64*1024**2
        for U,(_,_,left,right,_),image in zip(self.spectral_images,self.images,self.composed_images):
            arrays=(image,) if image is not None else (left,right,U)
            needed=sum(a.nbytes for a in arrays)
            if needed<=remaining:
                self.image_adjoints.append((image.conj().T,) if image is not None else
                    (left.conj().T,right.conj(),U.conj().T))
                remaining-=needed
            else:self.image_adjoints.append(None)
        # C maps raw parameters to physical norm coordinates. Its adjoint
        # bounds amplification of the transformed residual in raw coordinates.
        self.residual_amplification=max((float(np.sqrt(sum(sj*sj*np.vdot(row*roots,row*roots).real
            for sj,row in zip(s,vh)))) for _,_,inv,s,vh,roots in self.maps),default=1.)
    def expand(self,z):
        out=np.zeros(self.dimension,dtype=np.result_type(z,complex))
        for sl,small,inv,s,vh,roots in self.maps:out[sl]=inv*(vh.conj().T@(z[small]/s))
        return out
    def least_squares_correction(self, z):
        """Minimum Euclidean source correction on the retained physical support.

        QR of the coordinate map adjoint avoids forming normal equations and
        preserves the incumbent's source nullspace in a warm ALS solve.
        """
        from scipy.linalg import solve_triangular
        out = np.zeros(self.dimension, dtype=np.result_type(z, complex))
        if self._correction_maps is None:
            self._correction_maps=[(indices,small,s,*np.linalg.qr((vh*roots).conj().T,mode='reduced'))
                for indices,small,inv,s,vh,roots in self.maps if len(s)]
        for indices,small,s,q,r in self._correction_maps:
            out[indices] = q @ solve_triangular(r.conj().T, z[small] / s, lower=True)
        return out

    def adjoint(self,v):
        out=np.zeros(self.rank,dtype=np.result_type(v,complex))
        for sl,small,inv,s,vh,roots in self.maps:out[small]=(vh@(inv*v[sl]))/s
        return out
    def coordinates(self,v):
        out=np.zeros(self.rank,dtype=np.result_type(v,complex))
        for sl,small,inv,s,vh,roots in self.maps:out[small]=s*(vh@(roots*v[sl]))
        return out

    def orthogonal_expand(self,z, *, out=None):
        out={} if out is None else out
        for (_,small,_,_,_,_),U,(key,up,left,right,np_),image in zip(
                self.maps,self.spectral_images,self.images,self.composed_images):
            if key not in out:out[key]=np.zeros((left.shape[0],np_,right.shape[0]),dtype=complex)
            if image is not None:
                out[key][:,up,:]+=(image@z[small]).reshape(left.shape[0],len(up),right.shape[0])
            else:
                local=(U@z[small]).reshape(left.shape[1],len(up),right.shape[1])
                for j,p in enumerate(up):out[key][:,p,:]+=_ordered_product(left,local[:,j,:],right.T)
        return out

    def orthogonal_adjoint(self,blocks):
        out=np.zeros(self.rank,dtype=complex)
        for (_,small,_,_,_,_),U,(key,up,left,right,np_),image,fixed in zip(
                self.maps,self.spectral_images,self.images,self.composed_images,self.image_adjoints):
            if image is not None:
                adjoint=fixed[0] if fixed is not None else image.conj().T
                out[small]=adjoint@blocks[key][:,up,:].ravel()
            else:
                lh,rc,uh=fixed if fixed is not None and len(fixed)==3 else (left.conj().T,right.conj(),U.conj().T)
                local=np.empty((left.shape[1],len(up),right.shape[1]),dtype=complex)
                for j,p in enumerate(up):
                    local[:,j,:]=_ordered_product(lh,blocks[key][:,p,:],rc)
                out[small]=uh@local.ravel()
        return out

    def set_hamiltonian(self,mpo):
        if self._prepared_mpo is mpo:return
        from .reduced_environment import ReducedEnvironmentChain
        if self.sites is None:
            self.sites,_,_=_canonical_sites(self._source_sites,self.site)
        self.hamiltonian=ReducedEnvironmentChain.build(self.sites,mpo,lazy=True)
        self.hamiltonian_action=self.hamiltonian.prepare_local_action(self.site)

    def norm(self,vector):
        vector = np.asarray(vector).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError('source vector has incompatible size')
        return float(sum(np.linalg.norm(F.apply(vector[sl]))**2 for sl,F in self.blocks))

    def source_blocks(self,vector):
        blocks=scatter_blocks(self.embedding,self.embedding.unpack_source(vector))
        from .numpy_contractions import cached_einsum
        return {k:cached_einsum('al,lpr,br->apb',self.left[k[0]],np.asarray(a),self.right[k[2]])
                for k,a in blocks.items()}

    def energy(self,vector):
        center=self.source_blocks(vector)
        acted=self.hamiltonian_action(center)
        norm=sum(_sector_irrep(k[2]).dim*np.linalg.norm(a)**2 for k,a in center.items())
        if not np.isfinite(norm) or norm<=np.finfo(float).tiny:
            raise FloatingPointError('Null or nonfinite QR local state')
        energy=sum(np.vdot(center[k],a) for k,a in acted.items())/norm
        if not np.isfinite(energy) or abs(energy.imag)>1e-10*max(1.,abs(energy)):
            raise FloatingPointError('Complex or nonfinite QR local energy')
        return float(energy.real)

    def source_action_data(self,vector):
        """One H action supplies the energy, norm, and coefficient H-vector."""
        center=self.source_blocks(vector)
        acted=self.hamiltonian_action(center)
        norm=sum(_sector_irrep(k[2]).dim*np.linalg.norm(a)**2 for k,a in center.items())
        if not np.isfinite(norm) or norm<=np.finfo(float).tiny:
            raise FloatingPointError('Null or nonfinite QR local state')
        energy=sum(np.vdot(center[k],a) for k,a in acted.items())/norm
        if not np.isfinite(energy) or abs(energy.imag)>1e-10*max(1.,abs(energy)):
            raise FloatingPointError('Complex or nonfinite QR local energy')
        return self.pull_hamiltonian(acted),float(energy.real),float(norm)

    def pull_hamiltonian(self,acted):
        from .numpy_contractions import cached_einsum
        pulled={k:cached_einsum('al,apb,br->lpr',self.left[k[0]].conj(),a,self.right[k[2]].conj()) for k,a in acted.items()}
        return self.embedding.adjoint(self.embedding.pack_target(pulled))

    def source_hamiltonian(self,vector):
        return self.pull_hamiltonian(self.hamiltonian_action(self.source_blocks(vector)))

    def orthogonal_hamiltonian(self,blocks):
        unscaled={k:a/np.sqrt(_sector_irrep(k[2]).dim) for k,a in blocks.items()}
        return {k:a/np.sqrt(_sector_irrep(k[2]).dim) for k,a in self.hamiltonian_action(unscaled).items()}

    def apply(self,vector):
        import os
        budget=int(os.environ.get('PYQED_QR_ABSORBED_MB','0'))*1024**2
        if budget<0:raise ValueError('PYQED_QR_ABSORBED_MB must be nonnegative')
        if budget and not getattr(self,'_absorbed_attempted',False):
            from .qr_absorbed_action import QRAbsorbedAction,UnprofitableQRAbsorption
            self._absorbed_attempted=True
            try:self._absorbed_action=QRAbsorbedAction(self,budget=budget,require_reduction=True)
            except (MemoryError,UnprofitableQRAbsorption):self._absorbed_action=None
        if budget and getattr(self,'_absorbed_action',None) is not None:
            return self._absorbed_action(vector)
        action=self.hamiltonian_action
        if not hasattr(action,'apply_packed'):
            return self.orthogonal_adjoint(self.orthogonal_hamiltonian(self.orthogonal_expand(vector)))
        # Expand directly into the operator's input storage. Output owns its
        # memory; later calls cannot overwrite vectors retained by Davidson.
        packed=np.zeros(action.layout.size,dtype=complex)
        blocks=action.layout.unpack(packed,copy=False)
        self.orthogonal_expand(vector,out=blocks)
        for key,a in blocks.items():a/=np.sqrt(_sector_irrep(key[2]).dim)
        acted=action.layout.unpack(action.apply_packed(packed),copy=False)
        for key,a in acted.items():a/=np.sqrt(_sector_irrep(key[2]).dim)
        return self.orthogonal_adjoint(acted)

    def preconditioned_residual(self,residual,energy):
        action=getattr(self,'_absorbed_action',None)
        if action is None:return residual
        if self._block_preconditioner is None:
            from .local_preconditioner import coordinate_blocks
            try:self._block_preconditioner,self.preconditioner_report=coordinate_blocks(action)
            except MemoryError:self._block_preconditioner=False
        if self._block_preconditioner is False:return residual
        return self._block_preconditioner(residual,energy)

    def projector(self,tolerance):
        """Orthogonal support projector in column-equilibrated coordinates."""
        self.prepare(tolerance)
        support=tuple(self.support)
        def project(vector):
            out=np.zeros(self.dimension,dtype=np.result_type(vector,complex))
            for sl,vh in support:out[sl]=vh.conj().T@(vh@vector[sl])
            return out
        return project

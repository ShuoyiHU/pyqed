"""Stable reduced one-site overlap and Hamiltonian coordinates."""
from collections import defaultdict, Counter
from types import SimpleNamespace
import numpy as np
from .reduced_symmetry import _sector_irrep

def _canonical_sites(sites,center):
    sites=[a.copy() for a in sites]
    left={q:np.eye(d) for q,d in Counter(sites[0].qns[0]).items()}
    for i in range(center):
        tensor=sites[i];groups=defaultdict(list)
        for key,a in tensor.data.items():
            b=np.einsum('kl,lpr->kpr',left[key[0]],a)
            groups[key[2]].append((key,b))
        next_left={};data={}
        for qr,entries in groups.items():
            Q,R=np.linalg.qr(np.concatenate([b.reshape(-1,b.shape[2]) for _,b in entries]),mode='reduced')
            next_left[qr]=R;offset=0
            for key,b in entries:
                rows=b.shape[0]*b.shape[1]
                data[key]=Q[offset:offset+rows].reshape(b.shape[:2]+(R.shape[0],));offset+=rows
        tensor.data=data
        tensor.qns[0]=[q for q,R in left.items() for _ in range(R.shape[0])]
        tensor.qns[2]=[q for q,R in next_left.items() for _ in range(R.shape[0])]
        left=next_left
    right={q:np.eye(d) for q,d in Counter(sites[-1].qns[2]).items()}
    for i in range(len(sites)-1,center,-1):
        tensor=sites[i];groups=defaultdict(list)
        for key,a in tensor.data.items():
            b=np.einsum('kr,lpr->lpk',right[key[2]],a)
            weight=np.sqrt(_sector_irrep(key[2]).dim/_sector_irrep(key[0]).dim)
            groups[key[0]].append((key,b,weight))
        next_right={};data={}
        for ql,entries in groups.items():
            Q,R=np.linalg.qr(np.concatenate([weight*b.transpose(2,1,0).reshape(-1,b.shape[0]) for _,b,weight in entries]),mode='reduced')
            next_right[ql]=R;offset=0
            for key,b,weight in entries:
                rows=b.shape[1]*b.shape[2]
                data[key]=(Q[offset:offset+rows].reshape(b.shape[2],b.shape[1],R.shape[0]).transpose(2,1,0)/weight);offset+=rows
        tensor.data=data
        tensor.qns[0]=[q for q,R in next_right.items() for _ in range(R.shape[0])]
        tensor.qns[2]=[q for q,R in right.items() for _ in range(R.shape[0])]
        right=next_right
    tensor=sites[center]
    tensor.data={key:np.einsum('al,lpr,br->apb',left[key[0]],a,right[key[2]]) for key,a in tensor.data.items()}
    tensor.qns[0]=[q for q,R in left.items() for _ in range(R.shape[0])]
    tensor.qns[2]=[q for q,R in right.items() for _ in range(R.shape[0])]
    return tuple(sites),left,right


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
    """
    def __init__(self,sites,embedding,site):
        self.site=site;self.embedding=embedding
        self.sites,self.left,self.right=_canonical_sites(sites,site)
        factor=SimpleNamespace(embedding=embedding,left=self.left,right=self.right)
        emb=embedding
        self.dimension=emb.source_size;self.blocks=[];self.images=[]
        for key in emb.source_layout.keys:
            lo,hi=emb.source_layout.offsets[key]
            tlo,thi=emb.target_layout.offsets[key]
            source_shape=emb.source_layout.shapes[key]
            source_grid=np.arange(lo,hi).reshape(source_shape)
            for physical in np.ndindex(source_shape[1:-1]):
                indices=source_grid[(slice(None),)+physical+(slice(None),)].ravel()
                mask=np.isin(emb.source_indices,indices)
                source=np.searchsorted(indices,emb.source_indices[mask])
                target=emb.target_indices[mask]-tlo
                if np.any(target<0) or np.any(target>=thi-tlo):
                    raise AssertionError('Norm embedding couples different reduced keys')
                shape=emb.target_layout.shapes[key]
                li,pi,ri=np.unravel_index(target,shape)
                ul,lmap=np.unique(li,return_inverse=True)
                up,pmap=np.unique(pi,return_inverse=True)
                ur,rmap=np.unique(ri,return_inverse=True)
                # Restrict to this physical configuration before QR so unused
                # conditional rows never enter the local norm factor.
                ql,left=np.linalg.qr(factor.left[key[0]][:,ul],mode='reduced')
                qr,right=np.linalg.qr(factor.right[key[2]][:,ur],mode='reduced')
                F=np.zeros((left.shape[0],len(up),right.shape[0],len(indices)),dtype=np.result_type(left,right))
                weight=np.sqrt(_sector_irrep(key[2]).dim)
                for col,l,p,r in zip(source,lmap,pmap,rmap):
                    F[:,p,:,col]+=weight*left[:,l,None]*right[None,:,r]
                self.blocks.append((indices,F.reshape(-1,len(indices))))
                self.images.append((key,up,ql,qr,shape[1]))
        self.stored_elements=sum(F.size for _,F in self.blocks)
    def metric(self,v):
        out=np.zeros(self.dimension,dtype=np.result_type(v,complex))
        for sl,F in self.blocks:out[sl]=F.conj().T@(F@v[sl])
        return out
    def prepare(self,tolerance):
        if getattr(self,'tolerance',None)==tolerance:
            return
        self.tolerance=tolerance
        factors=[];largest=0.
        for sl,F in self.blocks:
            roots=np.linalg.norm(F,axis=0)
            inv=np.divide(1.,roots,out=np.zeros_like(roots),where=roots>0.)
            normalized=F*inv[None,:]
            if not np.all(np.isfinite(normalized)):
                raise FloatingPointError('Nonfinite normalized QR norm factor')
            try:
                u,s,vh=np.linalg.svd(normalized,full_matrices=False)
            except np.linalg.LinAlgError:
                from scipy.linalg import svd
                u,s,vh=svd(normalized,full_matrices=False,lapack_driver='gesvd')
            largest=max(largest,float(s.max(initial=0.)))
            factors.append((sl,roots,inv,s,vh,u))
        cutoff=largest*np.sqrt(max(tolerance,np.finfo(float).eps*self.dimension))
        self.maps=[];self.spectral_images=[];self.support=[];offset=0
        for sl,roots,inv,s,vh,u in factors:
            keep=s>cutoff;s=s[keep];vh=vh[keep]
            W=inv[:,None]*vh.conj().T/s[None,:]
            C=s[:,None]*vh*roots[None,:]
            self.maps.append((sl,slice(offset,offset+len(s)),W,C));offset+=len(s)
            self.spectral_images.append(u[:,keep])
            self.support.append((sl,vh))
        self.rank=offset
        # C maps raw parameters to physical norm coordinates. Its adjoint
        # bounds amplification of the transformed residual in raw coordinates.
        self.residual_amplification=max((float(np.linalg.norm(C)) for _,_,_,C in self.maps),default=1.)
    def expand(self,z):
        out=np.zeros(self.dimension,dtype=np.result_type(z,complex))
        for sl,small,W,C in self.maps:out[sl]=W@z[small]
        return out
    def adjoint(self,v):
        out=np.zeros(self.rank,dtype=np.result_type(v,complex))
        for sl,small,W,C in self.maps:out[small]=W.conj().T@v[sl]
        return out
    def coordinates(self,v):
        out=np.zeros(self.rank,dtype=np.result_type(v,complex))
        for sl,small,W,C in self.maps:out[small]=C@v[sl]
        return out

    def orthogonal_expand(self,z):
        out={}
        for (_,small,_,_),U,(key,up,left,right,np_) in zip(self.maps,self.spectral_images,self.images):
            if key not in out:out[key]=np.zeros((left.shape[0],np_,right.shape[0]),dtype=complex)
            local=(U@z[small]).reshape(left.shape[1],len(up),right.shape[1])
            for j,p in enumerate(up):out[key][:,p,:]+=left@local[:,j,:]@right.T
        return out
    def orthogonal_adjoint(self,blocks):
        out=np.zeros(self.rank,dtype=complex)
        for (_,small,_,_),U,(key,up,left,right,np_) in zip(self.maps,self.spectral_images,self.images):
            local=np.stack([left.conj().T@blocks[key][:,p,:]@right.conj() for p in up],axis=1)
            out[small]=U.conj().T@local.ravel()
        return out

    def set_hamiltonian(self,mpo):
        from .reduced_environment import ReducedEnvironmentChain
        self.hamiltonian=ReducedEnvironmentChain.build(self.sites,mpo)

    def norm(self,vector):
        return float(sum(np.linalg.norm(F@vector[sl])**2 for sl,F in self.blocks))

    def source_blocks(self,vector):
        blocks=self.embedding.unpack_target(self.embedding.apply(vector))
        return {k:np.einsum('al,lpr,br->apb',self.left[k[0]],a,self.right[k[2]]) for k,a in blocks.items()}

    def energy(self,vector):
        center=self.source_blocks(vector)
        acted=self.hamiltonian.local_action(self.site,center)
        norm=sum(_sector_irrep(k[2]).dim*np.linalg.norm(a)**2 for k,a in center.items())
        if not np.isfinite(norm) or norm<=np.finfo(float).tiny:
            raise FloatingPointError('Null or nonfinite QR local state')
        energy=sum(np.vdot(center[k],a) for k,a in acted.items())/norm
        if not np.isfinite(energy) or abs(energy.imag)>1e-10*max(1.,abs(energy)):
            raise FloatingPointError('Complex or nonfinite QR local energy')
        return float(energy.real)

    def source_hamiltonian(self,vector):
        center=self.source_blocks(vector)
        acted=self.hamiltonian.local_action(self.site,center)
        pulled={k:np.einsum('al,apb,br->lpr',self.left[k[0]].conj(),a,self.right[k[2]].conj()) for k,a in acted.items()}
        return self.embedding.adjoint(self.embedding.pack_target(pulled))

    def orthogonal_hamiltonian(self,blocks):
        unscaled={k:a/np.sqrt(_sector_irrep(k[2]).dim) for k,a in blocks.items()}
        return {k:a/np.sqrt(_sector_irrep(k[2]).dim) for k,a in self.hamiltonian.local_action(self.site,unscaled).items()}

    def apply(self,vector):
        return self.orthogonal_adjoint(self.orthogonal_hamiltonian(self.orthogonal_expand(vector)))

    def projector(self,tolerance):
        """Orthogonal support projector in column-equilibrated coordinates."""
        self.prepare(tolerance)
        support=tuple(self.support)
        def project(vector):
            out=np.zeros(self.dimension,dtype=np.result_type(vector,complex))
            for sl,vh in support:out[sl]=vh.conj().T@(vh@vector[sl])
            return out
        return project

"""Fermionic orbital RDMs from MPS transfers; no full-state expansion.

I(i,j)=S(i)+S(j)-S(i,j), natural logarithms (no factor 1/2).
The parity string implements fermionic reordering of nonadjacent orbitals.
"""
from operator import index
import numpy as np


def orbital_information(sites):
    """Return single entropies, MI and transfer-work metadata for a parity eigenstate.

    Local basis: empty, up, down, double. Natural logarithms; no half-MI factor.
    Fermionic transfer implementation of the orbital-information diagnostic;
    see Legeza and Solyom, Phys. Rev. B 68, 195116 (2003),
    https://doi.org/10.1103/PhysRevB.68.195116. No full 4**n vector is formed.
    Inputs must have definite total fermion parity (as fixed-N QC states do).
    """
    sites=tuple(np.asarray(a) for a in sites)
    if not sites or any(a.ndim!=3 or a.shape[1]!=4 or not np.all(np.isfinite(a)) for a in sites):
        raise ValueError('sites must be finite (left,4,right) spatial-orbital tensors')
    if sites[0].shape[0]!=1 or sites[-1].shape[2]!=1 or any(a.shape[2]!=b.shape[0] for a,b in zip(sites,sites[1:])):
        raise ValueError('sites must form an open-boundary MPS with scalar endpoints')
    n=len(sites);parity=np.array([0,1,1,0]);right=[None]*(n+1);right[n]=np.ones((1,1))
    for i in range(n-1,-1,-1):
        a=sites[i];right[i]=np.einsum('asc,cd,bsd->ab',a.conj(),right[i+1],a,optimize=True)
    norm=float(right[0][0,0].real)
    if not np.isfinite(norm) or norm<=0:
        raise ValueError('MPS must have positive finite norm')
    def entropy(rho):
        rho=(rho+rho.conj().T)/(2*norm)
        if abs(np.trace(rho)-1)>1e-9:
            raise ValueError('orbital RDM is not normalized')
        ev=np.linalg.eigvalsh(rho)
        if ev.min() < -1e-9:
            raise ValueError('orbital RDM is not positive semidefinite')
        ev=ev[ev>1e-14]
        return float(-np.sum(ev*np.log(ev)))
    single=np.zeros(n);pair=np.zeros((n,n));left=np.ones((1,1));peak=0;calls=0;temporary_peak=0
    weights=(-1.)**((parity[:,None]^parity[None,:])[:,:,None]*parity[None,None,:])
    for i,a in enumerate(sites):
        opened=np.einsum('xy,xar,ybs->abrs',left,a.conj(),a,optimize=True)
        peak=max(peak,opened.nbytes)
        single[i]=entropy(np.einsum('abrs,rs->ab',opened,right[i+1],optimize=True))
        for j in range(i+1,n):
            b=sites[j]
            rho=np.einsum('abxy,xcv,ydw,vw->acbd',opened,b.conj(),b,right[j+1],optimize=True).reshape(16,16)
            # Fixed total fermion parity cancels the prefix string. The open
            # physical pair must have the same total parity on bra and ket.
            pp=(parity[:,None]+parity[None,:]).reshape(16)%2
            rho=np.where(pp[:,None]==pp[None,:],rho,0)
            pair[i,j]=pair[j,i]=entropy(rho);calls+=1
            if j<n-1:
                # Explicit matrix-product ordering avoids the default four-way
                # einsum's O(M^4) simultaneous contraction. No rank truncation.
                temp=np.einsum('abxy,xsv->abysv',opened,b.conj(),optimize=True)
                temp*=weights[:,:,None,:,None]
                temporary_peak=max(temporary_peak,temp.nbytes)
                opened=np.einsum('abysv,ysw->abvw',temp,b,optimize=True)
                peak=max(peak,opened.nbytes)
        left=np.einsum('xy,xar,yas->rs',left,a.conj(),a,optimize=True)
    mi=single[:,None]+single[None,:]-pair;np.fill_diagonal(mi,0)
    if mi.min() < -1e-9:
        raise ValueError('negative mutual information beyond numerical tolerance')
    return single,np.maximum(mi,0),dict(pair_rdms=calls,largest_open_transfer_bytes=peak,
        magnetic_mps_shapes=[list(a.shape) for a in sites],largest_explicit_temporary_bytes=temporary_peak,norm=norm)


def strongest_disjoint(mi,count=2):
    """Greedy positive-MI disjoint pairs; not a maximum-weight matching."""
    from .orbital_ordering import _affinity
    mi=_affinity(mi)
    count=index(count)
    if count<0:
        raise ValueError('count must be nonnegative')
    ranked=sorted([(float(mi[i,j]),i,j) for i in range(len(mi)) for j in range(i+1,len(mi))],key=lambda x:(-x[0],x[1],x[2]))
    if count==0:return [],ranked
    selected=[];used=set()
    for value,i,j in ranked:
        if value<=1e-12:break
        if i in used or j in used:continue
        selected.append((i,j));used.update([i,j])
        if len(selected)==count:break
    return selected,ranked


def conditional_entropies(single, mutual_information):
    """Matrix [i,j] = S(i|j) = S_i - I_ij in nats; diagonal is undefined (NaN)."""
    from .orbital_ordering import _affinity
    mi=_affinity(mutual_information)
    single=np.asarray(single,dtype=float)
    if single.shape!=(len(mi),) or not np.all(np.isfinite(single)) or np.any(single < -1e-12):
        raise ValueError('single must contain one finite nonnegative entropy per orbital')
    result=single[:,None]-mi
    np.fill_diagonal(result,np.nan)
    return result


def negative_conditional_ties(single, mutual_information):
    """All pairs with S(i|j)<0 OR S(j|i)<0, returned as forward edges.

    Ranked by the more negative direction, then indices. This entanglement-
    witness heuristic does not optimize tie orientation or impose a disjointness
    or frontier budget. Values are in nats and the threshold is strictly zero.
    """
    conditional=conditional_entropies(single,mutual_information)
    ranked=sorted((min(conditional[i,j],conditional[j,i]),i,j)
                  for i in range(len(conditional)) for j in range(i+1,len(conditional))
                  if min(conditional[i,j],conditional[j,i])<0.)
    return [(i,j) for _,i,j in ranked]

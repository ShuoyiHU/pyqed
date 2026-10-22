"""Local-coordinate Davidson preconditioners; full H remains unchanged.

Adaptation of the shifted diagonal denominator used by PySCF Davidson:
https://github.com/pyscf/pyscf/blob/master/pyscf/lib/linalg_helper.py
The block variant uses exact diagonal coordinate blocks instead of scalar
entries. Inter-block couplings are omitted only in the inverse approximation,
never in the Hamiltonian or convergence residual. This does not guarantee
fewer iterations or selection of the globally lowest root for every start.
"""
import numpy as np

def boundary_diagonal(action):
    op=action.op;c=action
    offsets={v[0]:k for k,v in op.layout.offsets.items()}
    diagonals={i:np.zeros(shape,complex) for i,shape in op.layout.shapes.items()}
    for row,coefficient in zip(op.routes,op.coefficients):
        lo,ro,xo,yo,l,r,a,b,p,o,np_,no=map(int,row)
        if xo!=yo or p!=o:continue
        i=offsets[xo]
        left=op.left[lo:lo+a*l].reshape(a,l)
        right=op.right[ro:ro+b*r].reshape(b,r)
        diagonals[i][:,p,:]+=coefficient*np.diag(left)[:,None]*np.diag(right)[None,:]*op.scale
    diagonal=np.zeros(c.rank)
    for i,(U,m) in enumerate(zip(c.spectral_images,c.maps)):
        diagonal[m[1]]=np.einsum('ik,i,ik->k',U.conj(),diagonals[i].ravel(),U,optimize=True).real
    def precondition(residual,energy):
        # Same shifted denominator convention as PySCF make_diag_precond.
        denominator=diagonal-(energy-1e-3)
        denominator[np.abs(denominator)<1e-8]=1e-8
        return residual/denominator
    return precondition,diagonal

def coordinate_blocks(action, max_elements=1024**2):
    """Exact diagonal blocks of H in retained coordinates; no full-H probes."""
    op=action.op;c=action
    offsets={v[0]:k for k,v in op.layout.offsets.items()}
    elements=sum(U.shape[1]**2 for U in c.spectral_images)
    if elements>max_elements:raise MemoryError('Local block-preconditioner budget exceeded')
    blocks=[np.zeros((U.shape[1],U.shape[1]),complex) for U in c.spectral_images]
    route_count=0;estimated_mac=0
    for row,coefficient in zip(op.routes,op.coefficients):
        lo,ro,xo,yo,l,r,a,b,p,o,np_,no=map(int,row)
        if xo!=yo:continue
        i=offsets[xo];U=c.spectral_images[i];rank=U.shape[1]
        estimated_mac+=rank*min(a*l*r+a*r*b,l*r*b+a*l*b)+a*b*rank*rank
        left=op.left[lo:lo+a*l].reshape(a,l);right=op.right[ro:ro+b*r].reshape(b,r)
        source=U.reshape(l,np_,r,rank)[:,p,:,:]
        bra=U.reshape(a,no,b,rank)[:,o,:,:].reshape(a*b,rank).conj().T
        for start in range(0,rank,8):
            stop=min(start+8,rank)
            transformed=np.einsum('al,lrk,br->abk',left,source[:,:,start:stop],right,optimize=True)
            blocks[i][:,start:stop]+=coefficient*op.scale*(bra@transformed.reshape(a*b,stop-start))
        route_count+=1
    factors=[]
    for block in blocks:
        if not np.allclose(block,block.conj().T,rtol=2e-11,atol=2e-11):
            raise FloatingPointError('Non-Hermitian local preconditioner block')
        factors.append(np.linalg.eigh((block+block.conj().T)*.5))
    def precondition(residual,energy):
        result=np.empty_like(residual)
        for m,(values,vectors) in zip(c.maps,factors):
            denominator=values-(energy-1e-3)
            denominator[np.abs(denominator)<1e-8]=1e-8
            result[m[1]]=vectors@((vectors.conj().T@residual[m[1]])/denominator)
        return result
    return precondition,dict(diagonal_routes=route_count,stored_elements=sum(B.size for B in blocks),estimated_MAC=estimated_mac,blocks=blocks)

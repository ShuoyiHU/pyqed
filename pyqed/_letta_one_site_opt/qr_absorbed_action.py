"""Exact QR-boundary absorption prototype; no new physical truncation."""
from collections import defaultdict
import numpy as np
from pyqed._letta_two_site_opt.sector_operator import SectorOperator
from pyqed._letta_one_site_opt.reduced_symmetry import _sector_irrep

class UnprofitableQRAbsorption(RuntimeError):
    """Exact regrouping fails the conservative pre-reuse arithmetic screen."""


class QRAbsorbedAction:
    """Exact local-coordinate boundary absorption for repeated H actions.

    Uses associativity of BLAS matrix products (https://www.netlib.org/blas/)
    within the existing QR/SVD coordinates. This is an implementation-specific
    exact regrouping, not an additional variational approximation. Every
    coordinate-block coupling and SU(2) weight is retained. Only the original
    coordinate preparation determines retained rank. No dense many-body or
    dense effective Hamiltonian is formed.

    ``budget`` bounds the single packed cache of transformed boundary values; it
    does not bound total RSS. Original and transformed operators coexist.
    ``require_reduction`` screens raw two-product MAC counts before product
    reuse. It is a conservative admission heuristic, not a runtime guarantee:
    setup, cache reuse and removed coordinate transforms are not included.
    Intended for one fixed local problem; reconstruct after coordinates change.
    """
    def __init__(self, coordinates, budget=512*1024**2, *, require_reduction=False):
        self.maps=tuple(coordinates.maps)
        self.spectral_images=tuple(coordinates.spectral_images)
        self.rank=coordinates.rank
        op=coordinates.hamiltonian_action
        by_slice=defaultdict(list)
        shapes={}
        for i,(key,up,L,R,nphysical) in enumerate(coordinates.images):
            shapes[i]=(L.shape[1],len(up),R.shape[1])
            for j,p in enumerate(up):
                by_slice[op.layout.offsets[key][0],int(p)].append((i,j))
        specs=[];left_specs={};right_specs={};owners=[{},{}]
        old_mac=new_mac=0
        for row,coefficient in zip(op.routes,op.coefficients):
            lo,ro,xo,yo,l,r,a,b,ip,oq,np_,no=map(int,row)
            old_mac+=min(a*l*r+a*r*b,l*r*b+a*l*b)
            for oi,p in by_slice[yo,oq]:
                for ii,q in by_slice[xo,ip]:
                    lk=(id(coordinates.images[oi][2]),id(coordinates.images[ii][2]),lo,a,l)
                    rk=(id(coordinates.images[oi][3]),id(coordinates.images[ii][3]),ro,b,r)
                    owners[0].setdefault(lk,(oi,ii));owners[1].setdefault(rk,(oi,ii))
                    aa,_,bb=shapes[oi];ll,_,rr=shapes[ii]
                    left_specs[lk]=aa*ll;right_specs[rk]=bb*rr
                    new_mac+=min(aa*ll*rr+aa*rr*bb,ll*rr*bb+aa*ll*bb)
                    specs.append((oi,ii,p,q,lk,rk,coefficient))
        size=16*(sum(left_specs.values())+sum(right_specs.values()))
        self.report=dict(original_routes=len(op.routes),fused_routes=len(specs),original_MAC=int(old_mac),fused_MAC=int(new_mac),boundary_bytes=size,original_vector_elements=op.layout.size,fused_vector_elements=sum(np.prod(v).item() for v in shapes.values()))
        if require_reduction and new_mac>=old_mac:
            raise UnprofitableQRAbsorption(self.report)
        if size>budget:raise MemoryError(str(self.report))
        # Allocate final packed storage once; never retain a second full copy.
        from .reduced_frontier import _BlockVectorLayout
        layout=_BlockVectorLayout(shapes)
        offsets=[{},{}];packed=[]
        for side,src in enumerate((left_specs,right_specs)):
            total=sum(src.values());values=np.empty(total,dtype=np.complex128)
            offset=0
            for k,count in src.items():
                offsets[side][k]=offset
                _,_,original_offset,a,l=k
                oi,ii=owners[side][k]
                Qout=coordinates.images[oi][2+side];Qin=coordinates.images[ii][2+side]
                matrix=(op.left if side==0 else op.right)[original_offset:original_offset+a*l].reshape(a,l)
                destination=values[offset:offset+count].reshape(Qout.shape[1],Qin.shape[1])
                # Choose the smaller arithmetic chain; write its last product in place.
                left_cost=Qout.shape[1]*a*l+Qout.shape[1]*l*Qin.shape[1]
                right_cost=a*l*Qin.shape[1]+Qout.shape[1]*a*Qin.shape[1]
                if left_cost<=right_cost:
                    temporary=Qout.conj().T@matrix
                    np.matmul(temporary,Qin,out=destination)
                else:
                    temporary=matrix@Qin
                    np.matmul(Qout.conj().T,temporary,out=destination)
                del temporary
                offset+=count
            packed.append(values)
        routes=[];coefficients=[]
        for oi,ii,p,q,lk,rk,coef in specs:
            a,no,b=shapes[oi];l,np_,r=shapes[ii]
            routes.append((offsets[0][lk],offsets[1][rk],layout.offsets[ii][0],layout.offsets[oi][0],l,r,a,b,q,p,np_,no))
            ko=coordinates.images[oi][0];ki=coordinates.images[ii][0]
            coefficients.append(coef/np.sqrt(_sector_irrep(ko[2]).dim*_sector_irrep(ki[2]).dim))
        self.op=SectorOperator.from_packed(shapes,*packed,routes,coefficients,scale=op.scale)
        self.report['allocated_boundary_bytes']=self.op.left.nbytes+self.op.right.nbytes
    def __call__(self,z):
        blocks={i:(U@z[m[1]]).reshape(self.op.layout.shapes[i]) for i,(U,m) in enumerate(zip(self.spectral_images,self.maps))}
        acted=self.op.layout.unpack(self.op.apply_packed(self.op.layout.pack(blocks),shared_products=False),copy=False)
        result=np.zeros(self.rank,complex)
        for i,(U,m) in enumerate(zip(self.spectral_images,self.maps)):
            result[m[1]]=U.conj().T@acted[i].ravel()
        return result

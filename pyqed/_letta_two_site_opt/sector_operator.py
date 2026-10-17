"""Packed reduced-sector Hamiltonian actions, without a dense operator matrix."""
import math
import numpy as np
from .pair_metric import _engine
from .._letta_one_site_opt.reduced_frontier import _BlockVectorLayout


class SectorOperator:
    """Apply sum L[x,a,l] W[x,y,o,p] R[y,b,r] X[l,p,r] in C.

    All symmetry/recoupling coefficients are supplied by the existing reduced
    transfer tensors. Zero coefficients are skipped exactly, with no threshold.
    This is a contraction implementation, not an additional physical projection.
    """
    def __init__(self, shapes, terms, scale=1.):
        self.layout=_BlockVectorLayout(shapes)
        self.scale=scale
        terms=list(terms)
        dtype=np.result_type(float,*[np.result_type(l,w,r) for _,_,l,w,r in terms])
        self.dtype=np.dtype(np.complex128 if np.issubdtype(dtype,np.complexfloating) else np.float64)
        matrices=[[],[]]; offsets=[{},{}]; sizes=[0,0]
        routes=[];coefficients=[];workspace=0
        def matrix_offset(side, array, index):
            key=(id(array),index)
            if key not in offsets[side]:
                offsets[side][key]=sizes[side]
                matrix=np.ascontiguousarray(array[index],dtype=self.dtype).ravel()
                matrices[side].append(matrix);sizes[side]+=matrix.size
            return offsets[side][key]
        for bra,ket,left,kernel,right in terms:
            a,b=shapes[bra][0],shapes[bra][-1]
            l,r=shapes[ket][0],shapes[ket][-1]
            no,np_=math.prod(shapes[bra][1:-1]),math.prod(shapes[ket][1:-1])
            if left.ndim!=3 or right.ndim!=3 or left.shape[1:]!=(a,l) or right.shape[1:]!=(b,r):
                raise ValueError('operator boundary shape mismatch')
            if kernel.shape!=(left.shape[0],right.shape[0],no,np_):
                raise ValueError('operator kernel shape mismatch')
            if max(a,b,l,r,no,np_)>np.iinfo(np.int32).max:
                raise OverflowError('operator block exceeds BLAS integer range')
            for x,y,o,p in np.argwhere(kernel!=0):
                lo=matrix_offset(0,left,int(x));ro=matrix_offset(1,right,int(y))
                routes.append((lo,ro,self.layout.offsets[ket][0],self.layout.offsets[bra][0],l,r,a,b,p,o,np_,no))
                coefficients.append(kernel[x,y,o,p])
            middle=a*r if a*l*r+a*r*b<=l*r*b+a*l*b else l*b
            workspace=max(workspace,l*r+middle+a*b)
        self.routes=np.asarray(routes,dtype=np.int64).reshape(-1,12)
        self.coefficients=np.asarray(coefficients,dtype=self.dtype)
        self.left,self.right=[np.concatenate(parts) if parts else np.empty(0,dtype=self.dtype) for parts in matrices]
        self.workspace_size=workspace
        self._complex_data=None
        # Reuse first products, not accumulated outputs. Route order and BLAS
        # grouping stay unchanged; buffers belong to each action invocation.
        from collections import Counter
        keys=[];sizes={};costs={}
        for lo,ro,xo,yo,l,r,a,b,ip,op,np_,no in routes:
            first_left=a*l*r+a*r*b<=l*r*b+a*l*b
            key=(first_left,lo if first_left else ro,xo,l,r,a if first_left else b,ip,np_)
            keys.append(key);sizes[key]=a*r if first_left else l*b
            costs[key]=a*l*r if first_left else l*r*b
        counts=Counter(keys);offset=0;selected={}
        for key in sorted(counts,key=lambda k:(counts[k]-1)*costs[k]/max(1,sizes[k]),reverse=True):
            if counts[key]<2 or sizes[key]==0:continue
            # Budget for complex128 too, even when the base operator is real.
            if (offset+sizes[key])*16>64*1024**2:continue
            selected[key]=(len(selected),offset);offset+=sizes[key]
        self.reuse_plan=np.asarray([selected.get(k,(-1,-1)) for k in keys],dtype=np.int64).reshape(-1,2)
        self.reuse_elements=offset
        self.reuse_count=len(selected)

    def __call__(self,blocks):
        vector=self.layout.pack(blocks)
        dtype=np.complex128 if self.dtype.kind=='c' or np.iscomplexobj(vector) else np.float64
        vector=np.ascontiguousarray(vector,dtype=dtype)
        left,right,coefficients=self.left,self.right,self.coefficients
        if dtype==np.complex128 and self.dtype.kind!='c':
            if self._complex_data is None:
                self._complex_data=tuple(a.astype(dtype) for a in (left,right,coefficients))
            left,right,coefficients=self._complex_data
        result=np.zeros(self.layout.size,dtype=dtype)
        work=np.empty(self.workspace_size,dtype=dtype)
        if self.reuse_count:
            reusable=np.empty(self.reuse_elements,dtype=dtype)
            ready=np.zeros(self.reuse_count,dtype=np.uint8)
            _engine().sector_action_reuse(self.routes,coefficients,left,right,vector,result,work,
                                         self.reuse_plan,reusable,ready)
        else:
            _engine().sector_action(self.routes,coefficients,left,right,vector,result,work)
        result*=self.scale
        return self.layout.unpack(result)

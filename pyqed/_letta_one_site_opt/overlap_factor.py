"""Exact reduced overlap-factor assembly; no singular-value truncation."""
import numpy as np
from .._letta_two_site_opt.pair_metric import _engine


def overlap_factor_matrix(left, right, source, lmap, pmap, rmap, nphysical, size, weight):
    """Build F[a,p,b,col] = sum weight L[a,l] R[b,r] from scatter routes.

    Repeated routes accumulate in their original order. This only replaces
    tensor-product assembly; the caller retains its QR/SVD and rank policy.
    """
    left,right=np.asarray(left),np.asarray(right)
    if left.ndim!=2 or right.ndim!=2 or nphysical<0 or size<0:
        raise ValueError('invalid overlap-factor dimensions')
    arrays=[np.asarray(v) for v in (source,lmap,pmap,rmap)]
    if any(v.ndim!=1 or v.dtype.kind not in 'iu' for v in arrays):
        raise ValueError('overlap routes must be one-dimensional integer arrays')
    if len({len(v) for v in arrays})!=1:
        raise ValueError('overlap routes must have equal lengths')
    for v,limit in zip(arrays,(size,left.shape[1],nphysical,right.shape[1])):
        if np.any(v<0) or np.any(v>=limit):
            raise ValueError('overlap route outside factor dimensions')
    dtype=np.complex128 if np.iscomplexobj(left) or np.iscomplexobj(right) else np.float64
    left,right=(np.ascontiguousarray(v,dtype=dtype) for v in (left,right))
    routes=np.ascontiguousarray(np.column_stack(arrays),dtype=np.int64)
    out=np.zeros((left.shape[0],nphysical,right.shape[0],size),dtype=dtype)
    _engine().overlap_matrix(left,right,routes,float(weight),out)
    return out.reshape(left.shape[0]*nphysical*right.shape[0],size)

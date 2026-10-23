"""Column-equilibrated SVD through a bounded-row TSQR tree.

Adapts Demmel et al., SIAM J. Sci. Comput. 34, A206 (2012),
doi:10.1137/080731992. This serial, recomputed-leaf variant avoids storing
the input factor and never forms its Gram matrix. It retains all singular
values; the caller applies its unchanged cutoff. Output singular vectors
and small tree factors are additional storage, outside the row workspace.
"""
import numpy as np


def streamed_svd(shape, rows, workspace_mb=32.):
    m,n=shape
    if n==0 or m<n:
        raise ValueError('TSQR requires a nonempty tall factor')
    itemsize=np.dtype(complex).itemsize
    chunk=max(8*n,int(workspace_mb*1024**2/(4*itemsize*n)))
    if 4*itemsize*chunk*n>workspace_mb*1024**2:
        raise MemoryError('TSQR row workspace is smaller than eight column widths')
    intervals=[(a,min(a+chunk,m)) for a in range(0,m,chunk)]
    roots=np.zeros(n)
    for a,b in intervals:
        roots=np.hypot(roots,np.linalg.norm(rows(a,b),axis=0))
    inv=np.divide(1.,roots,out=np.zeros_like(roots),where=roots>0.)
    def block(a,b):
        out=rows(a,b)*inv[None,:]
        if not np.all(np.isfinite(out)):
            raise FloatingPointError('Nonfinite normalized overlap factor')
        return out
    nodes=[]
    for a,b in intervals:
        _,r=np.linalg.qr(block(a,b),mode='reduced')
        nodes.append((r,(a,b)))
    while len(nodes)>1:
        next_nodes=[]
        for j in range(0,len(nodes),2):
            if j+1==len(nodes):
                next_nodes.append(nodes[j]);continue
            r1,t1=nodes[j];r2,t2=nodes[j+1]
            q,r=np.linalg.qr(np.concatenate((r1,r2)),mode='reduced')
            next_nodes.append((r,(q,r1.shape[0],t1,t2)))
        nodes=next_nodes
    r,tree=nodes[0]
    u,s,vh=np.linalg.svd(r,full_matrices=False)
    output=np.empty((m,len(s)),dtype=np.result_type(r,u))
    def emit(node,transform):
        if len(node)==2:
            a,b=node
            q,_=np.linalg.qr(block(a,b),mode='reduced')
            output[a:b]=q@transform
        else:
            q,split,left,right=node
            emit(left,q[:split]@transform)
            emit(right,q[split:]@transform)
    emit(tree,u)
    return roots,output,s,vh

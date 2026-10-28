"""Exact bra/ket-sector and MPO-channel reachability before contraction.

This Boolean graph removes operator routes that cannot connect both open
boundaries. It uses structural tensor sectors, never incumbent coefficients.
Pending-label sparsity remains in the frontier scatter representation; this
graph does not assume that a QR-rotated multiplicity index is a physical tie.
The graph is deliberately a superset of pending-label compatibility, so it
cannot remove physical directions or free-pair sector growth.
"""
from dataclasses import replace
import numpy as np


def prune_conditioned_transfers(sites,transfers,left_keys,right_keys,*,max_edges=100000):
    """Use pending physical assignments as well as bra/ket/MPO sectors.

    Only whole (MPO-copy,physical-copy) kernel routes with no compatible path
    are removed. Surviving multiplicity matrices remain intact. In a free
    two-site problem both active factors get Cartesian bridges, so legal
    factor support cannot accidentally restrict the larger pair domain.
    If the graph budget is exceeded the unpruned input is returned exactly.
    """
    support=[a.metadata.get('letta_operator_support') for a in sites]
    if any(s is None for s in support):return transfers,dict(conditional_fallback='no pending-label graph')
    nodes={(key,0,(),()) for key in left_keys}
    layers=[];count=0
    for site,local,data in zip(sites,transfers,support):
        edges,left_basis,right_basis,free=data
        routes={}
        if free:
            for key,block in site.data.items():
                for p in range(block.shape[1]):
                    for assignment in left_basis[key[0]]:
                        routes[key,p,assignment]=right_basis[key[2]]
        else:
            for ql,qp,qr,p,la,ra,_ in edges:
                routes.setdefault(((ql,qp,qr),p,la),set()).add(ra)
        by_input={}
        for number,t in enumerate(local):
            for x,y,o,p in np.argwhere(t.kernel!=0):
                by_input.setdefault((t.left_key,int(x)),[]).append((number,int(x),int(y),int(o),int(p)))
        following=set();layer=[]
        for incoming in nodes:
            key,x,ba,ka=incoming
            for number,x,y,o,p in by_input.get((key,x),()):
                t=local[number]
                for br in routes.get((t.bra_key,o,ba),()):
                    for kr in routes.get((t.ket_key,p,ka),()):
                        outgoing=(t.right_key,y,br,kr)
                        layer.append((number,x,y,o,p,incoming,outgoing));following.add(outgoing)
                        count+=1
                        if count>max_edges:
                            return transfers,dict(conditional_fallback='edge budget exceeded',conditional_edges=count)
        nodes=following;layers.append(layer)
    nodes.intersection_update({(key,0,(),()) for key in right_keys})
    output=[None]*len(layers);before=after=0
    for i in range(len(layers)-1,-1,-1):
        selected={};previous=set()
        for number,x,y,o,p,incoming,outgoing in layers[i]:
            if outgoing in nodes:
                selected.setdefault(number,set()).add((x,y,o,p));previous.add(incoming)
        row=[]
        for number,t in enumerate(transfers[i]):
            before+=np.count_nonzero(t.kernel)
            if number not in selected:continue
            live=selected[number];after+=len(live)
            if len(live)==np.count_nonzero(t.kernel):row.append(t);continue
            kernel=np.zeros_like(t.kernel)
            for index in live:kernel[index]=t.kernel[index]
            row.append(replace(t,kernel=kernel))
        output[i]=tuple(row);nodes=previous
    return tuple(output),dict(conditional_edges=count,conditional_routes_removed=int(before-after),conditional_fallback=None)


def _channel_reachability_values(keys,index,width,source):
    values=np.zeros((len(keys),width),dtype=bool)
    for key,array in source.items():
        j=index.get(key)
        if j is not None:
            # Other keys in the previous layer can require wider padded rows.
            count=min(width,len(array))
            values[j,:count]=array[:count]
    return values

def prune_transfers(transfers,left_keys,right_keys):
    """Exact channel reachability in bounded batches of equal kernel shape.

    Propagate only structural nonzero channels. All branches at a cut are
    combined by Boolean OR, then intersected with backward reachability. No
    incumbent amplitudes or coefficient thresholds enter this graph. Packing
    changes only the Boolean work; surviving transfer order is unchanged.
    """
    reachable={key:np.ones(1,dtype=bool) for key in left_keys}
    layers=[];before=0
    for local in transfers:
        lk=list(dict.fromkeys(t.left_key for t in local));rk=list(dict.fromkeys(t.right_key for t in local))
        li={k:j for j,k in enumerate(lk)};ri={k:j for j,k in enumerate(rk)}
        mx=max((t.kernel.shape[0] for t in local),default=1);my=max((t.kernel.shape[1] for t in local),default=1)
        left=_channel_reachability_values(lk,li,mx,reachable);following=np.zeros((len(rk),my),dtype=bool)
        groups={}
        for index,t in enumerate(local):groups.setdefault((t.kernel.shape,t.kernel.dtype.str),[]).append(index)
        candidates=[]
        for (shape,dtype),indices in groups.items():
            nx,ny=shape[:2]
            # Limit stacked kernel and comparison temporaries; one term may exceed it.
            size=int(np.prod(shape))*np.dtype(dtype).itemsize
            chunk=max(1,min(256,16*1024**2//max(1,2*size+3*nx*ny)))
            for start in range(0,len(indices),chunk):
                selected=np.asarray(indices[start:start+chunk]);ts=[local[j] for j in selected]
                live=np.any(np.stack([t.kernel for t in ts])!=0,axis=(3,4))
                counts=np.count_nonzero(live,axis=(1,2));before+=int(counts.sum())
                a=np.asarray([li[t.left_key] for t in ts]);b=np.asarray([ri[t.right_key] for t in ts])
                mask=live & left[a,:nx,None]
                right=np.any(mask,axis=1)
                np.logical_or.at(following[:,:ny],b,right)
                active=np.flatnonzero(np.any(right,axis=1))
                if active.size:
                    candidates.append((selected[active],mask[active],counts[active],a[active],b[active],nx,ny))
        layers.append((local,lk,rk,li,ri,mx,my,candidates))
        reachable={rk[j]:following[j] for j in np.flatnonzero(np.any(following,axis=1))}
    productive={key:np.ones(1,dtype=bool) for key in right_keys};output=[None]*len(layers);after=0
    for i in range(len(layers)-1,-1,-1):
        local,lk,rk,li,ri,mx,my,candidates=layers[i]
        right=_channel_reachability_values(rk,ri,my,productive);previous=np.zeros((len(lk),mx),dtype=bool)
        row=[None]*len(local)
        for selected,mask,counts,a,b,nx,ny in candidates:
            keep=mask & right[b,None,:ny]
            kept=np.count_nonzero(keep,axis=(1,2));after+=int(kept.sum())
            np.logical_or.at(previous[:,:nx],a,np.any(keep,axis=2))
            for j in np.flatnonzero(kept):
                index=selected[j];t=local[index]
                row[index]=t if kept[j]==counts[j] else replace(t,kernel=t.kernel*keep[j,:,:,None,None])
        output[i]=tuple(t for t in row if t is not None)
        productive={lk[j]:previous[j] for j in np.flatnonzero(np.any(previous,axis=1))}
    return tuple(output),dict(channel_routes_before=before,channel_routes_after=after,removed_channel_routes=before-after)

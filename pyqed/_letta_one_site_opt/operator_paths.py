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


def prune_transfers(transfers,left_keys,right_keys):
    reachable=[{(key,0) for key in left_keys}]
    edges=[]
    for local in transfers:
        step=[];following=set()
        for i,t in enumerate(local):
            for x,y in np.argwhere(np.any(t.kernel!=0,axis=(2,3))):
                a=(t.left_key,int(x));b=(t.right_key,int(y))
                if a in reachable[-1]:
                    step.append((i,int(x),int(y),a,b));following.add(b)
        edges.append(step);reachable.append(following)
    reachable[-1].intersection_update({(key,0) for key in right_keys})
    keep=[None]*len(edges)
    for i in range(len(edges)-1,-1,-1):
        keep[i]=[e for e in edges[i] if e[-1] in reachable[i+1]]
        reachable[i]={e[-2] for e in keep[i]}
    output=[];before=after=0
    for local,selected in zip(transfers,keep):
        masks={}
        for i,x,y,*_ in selected:
            masks.setdefault(i,np.zeros(local[i].kernel.shape[:2],dtype=bool))[x,y]=True
        row=[]
        for i,t in enumerate(local):
            live=np.any(t.kernel!=0,axis=(2,3));before+=int(live.sum())
            mask=masks.get(i)
            if mask is None:continue
            after+=int(mask.sum())
            row.append(t if np.array_equal(mask,live) else
                       replace(t,kernel=t.kernel*mask[:,:,None,None]))
        output.append(tuple(row))
    return tuple(output),dict(channel_routes_before=before,channel_routes_after=after,
                              removed_channel_routes=before-after)

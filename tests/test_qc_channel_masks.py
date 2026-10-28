"""Exact route masks compared against an explicit graph reference."""
from dataclasses import replace
import numpy as np
from pyqed._letta_one_site_opt.operator_paths import prune_transfers
from pyqed._letta_one_site_opt.reduced_environment import ReducedTransfer

def reference_graph_pruning(transfers,left_keys,right_keys):
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
            if i not in masks:
                masks[i]=np.zeros(local[i].kernel.shape[:2],dtype=bool)
            masks[i][x,y]=True
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


def test_channel_masks_match_explicit_multichannel_graphs():
    rng=np.random.default_rng(19)
    for trial in range(100):
        layers=[]
        for site in range(4):
            local=[]
            for a in range(2):
                for b in range(2):
                    kernel=rng.integers(-1,2,(3,3,1,1)).astype(float)
                    kernel *= rng.random(kernel.shape) < (0.1+0.8*(trial%5)/4)
                    local.append(ReducedTransfer((a,), (b,), (site,a), (site+1,b),kernel))
            layers.append(tuple(local))
        left={(0,0):None};right={(4,1):None}
        expected,expected_counts=reference_graph_pruning(tuple(layers),left,right)
        actual,counts=prune_transfers(tuple(layers),left,right)
        assert counts==expected_counts
        for xs,ys in zip(expected,actual):
            assert len(xs)==len(ys)
            for x,y in zip(xs,ys):
                assert (x.left_key,x.right_key)==(y.left_key,y.right_key)
                np.testing.assert_array_equal(x.kernel,y.kernel)


import pytest
from pyqed._letta_one_site_opt.reduced_environment import CompiledReducedTransfer


def assert_pruning_matches_reference(layers,left,right):
    expected,counts=reference_graph_pruning(layers,left,right)
    actual,actual_counts=prune_transfers(layers,left,right)
    assert counts==actual_counts
    for xs,ys in zip(expected,actual):
        assert len(xs)==len(ys)
        for x,y in zip(xs,ys):
            assert (x.bra_key,x.ket_key,x.left_key,x.right_key)==(y.bra_key,y.ket_key,y.left_key,y.right_key)
            np.testing.assert_array_equal(x.kernel,y.kernel)


@pytest.mark.parametrize('widths',[(1,9,17,65,1),(1,64,3,8,1),(1,0,9,1)])
@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('compiled',[False,True])
def test_batched_channel_masks_wide_and_disconnected(widths,complex_values,compiled):
    rng=np.random.default_rng(134)
    transfer=CompiledReducedTransfer if compiled else ReducedTransfer
    layers=[]
    for site,(nx,ny) in enumerate(zip(widths,widths[1:])):
        local=[]
        for a in range(2):
            for b in range(2):
                kernel=rng.normal(size=(nx,ny,2,3))
                if complex_values:kernel=kernel+1j*rng.normal(size=kernel.shape)
                kernel*=rng.random(kernel.shape)<.06
                # Tiny finite coefficients must remain structural nonzeros.
                kernel*=1e-250
                local.append(transfer((a,),(b,),(site,a),(site+1,b),kernel))
        layers.append(tuple(local))
    assert_pruning_matches_reference(tuple(layers),{(0,0)},{(len(layers),1)})


def test_batched_channel_masks_multiple_chunks_preserve_transfer_order():
    layers=[tuple(ReducedTransfer((j,),(),(0,0),(1,0),
                np.array([[[[1.]],[[float(j%3)]],[[2.]]]])) for j in range(517)),
            (ReducedTransfer((),(),(1,0),(2,0),np.array([0.,1.,0.]).reshape(3,1,1,1)),)]
    assert_pruning_matches_reference(tuple(layers),{(0,0)},{(2,0)})

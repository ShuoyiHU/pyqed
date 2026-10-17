from itertools import product
import numpy as np
import pytest
from pyqed._letta_one_site_opt.frontier_indices import block_indices


def reference(shape, lv, rv, neighbors, physical, site, nsites, radix):
    source, target = [], []
    lc, rc = radix**len(lv), radix**len(rv)
    for l, left in enumerate(product(range(radix), repeat=len(lv))):
        for p, value in enumerate(physical):
            for r, right in enumerate(product(range(radix), repeat=len(rv))):
                assignments = list(zip(lv,left))+[(site,value)]+list(zip(rv,right))
                a = dict(assignments)
                if any(a[k]!=v for k,v in assignments) or any(k not in a for k in neighbors):
                    continue
                for i in range(shape[0]):
                    for j in range(shape[-1]):
                        source.append(np.ravel_multi_index((i,p)+tuple(a[k] for k in neighbors[1:])+(j,),shape))
                        target.append(np.ravel_multi_index((l*shape[0]+i,p,r*shape[-1]+j),(lc*shape[0],shape[1],rc*shape[-1])))
    return np.array(source,dtype=np.int64),np.array(target,dtype=np.int64)


@pytest.mark.parametrize('radix',[1,3,10])
@pytest.mark.parametrize('lv,rv,neighbors',[
    ((),(),(1,)), ((0,1),(0,2),(1,0,2)),
    ((1,2),(2,3),(1,2)), ((3,0),(0,3),(1,0,3)),
    ((),(0,2),(1,0,2)), ((0,2),(),(1,0,2)),
    ((),(),(1,2))])
@pytest.mark.parametrize('dims',[(1,1),(2,3),(0,2)])
def test_exact_frontier_routes(radix,lv,rv,neighbors,dims):
    shape=(dims[0],min(radix,2))+(radix,)*(len(neighbors)-1)+(dims[1],)
    physical=list(range(min(radix,2)))
    args=shape,lv,rv,neighbors,physical,1,4,radix
    actual=block_indices(*args)
    expected=reference(*args)
    for a,e in zip(actual,expected):np.testing.assert_array_equal(a,e)


def test_invalid_indices():
    with pytest.raises(ValueError):block_indices((2,1,2),(5,),(),(0,),[0],0,2,3)
    with pytest.raises(OverflowError):block_indices((2**62,2,3),(),(),(0,),[0,1],0,2,3)


def test_empty_block_cannot_hide_memory_index_overflow():
    with pytest.raises(OverflowError):
        block_indices((0,1,1),tuple(range(40)),(),(0,),[0],0,40,3)

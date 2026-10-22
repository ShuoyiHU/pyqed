from collections import Counter
import numpy as np
from pyqed._letta_one_site_opt import ReducedFrontier
from pyqed._letta_two_site_opt.reduced_solver import _active_source_indices
from pyqed._letta_two_site_opt.reduced_compression import reduced_factor_blocks
from test_letta_frontier_support import state

def test_allocated_indices_are_compact_and_retain_all_virtual_copies():
    s=state(copies=2)
    f=ReducedFrontier.from_state(s)
    old=ReducedFrontier.from_state(s,prune=False)
    retained=Counter(s.bond_sectors[0])
    for i,side in ((0,'left'),(1,'right')):
        e=f.site_embedding(s,i)
        raw=_active_source_indices(old.site_embedding(s,i),retained,side)
        expected=np.flatnonzero(np.isin(e.supported_source_indices,raw))
        np.testing.assert_array_equal(_active_source_indices(e,retained,side),expected)

def test_factor_gauge_groups_use_compact_source_coordinates():
    s=state(copies=2)
    f=ReducedFrontier.from_state(s)
    le,re=(f.site_embedding(s,i) for i in (0,1))
    groups=reduced_factor_blocks(s,0,le,re,Counter(s.bond_sectors[0]))
    assert groups
    for a,b in groups:
        assert a.min()>=0 and a.max()<le.source_size
        assert b.min()>=0 and b.max()<re.source_size
        assert a.shape[1]==b.shape[0]

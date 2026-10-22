import numpy as np
import pytest
from pyqed._letta_one_site_opt import reduced_cbe
from test_letta_pair_support import electronic_state
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem

@pytest.mark.parametrize('complex_data', [False, True])
def test_cbe_seeds_preserve_raw_coordinate_random_stream(complex_data):
    state, electronic = electronic_state((2,1), ((0,1,2),(1,0,2),(2,0,1)), copies=2)
    embeddings=[]
    for prune in (False,True):
        pair=reduced_pair_problem(state,electronic.su2_mpo(),0,prune=prune,matrix_free=True,dense_solver_threshold=0)
        embeddings.append([pair.frontier.site_embedding(state,i) for i in (0,1)])
    assert hasattr(reduced_cbe, '_source_seeds'), 'CBE starts currently use different random streams after pruning'
    old,new=[reduced_cbe._source_seeds(e,2817,complex_data=complex_data) for e in embeddings]
    for before,after in zip(old,new):
        for side in (0,1):
            np.testing.assert_array_equal(after[side], before[side][embeddings[1][side].supported_source_indices])

@pytest.mark.parametrize('raw_size', [1,65535,65536,65539,131081])
def test_bounded_source_random_gather(raw_size):
    from types import SimpleNamespace
    selected=np.unique(np.array([0,raw_size//2,raw_size-1]))
    embedding=SimpleNamespace(source_layout=SimpleNamespace(size=raw_size),supported_source_indices=selected)
    assert hasattr(reduced_cbe, '_source_normal')
    rng=np.random.default_rng(19)
    expected=rng.normal(size=raw_size)[selected]
    expected_next=rng.normal(size=5)
    rng=np.random.default_rng(19)
    actual=reduced_cbe._source_normal(rng,embedding,complex)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(rng.normal(size=5),expected_next)

"""Ring allocation must preserve independent amplitudes and cyclic pair maps."""
from collections import Counter

import numpy as np
import pytest

from pyqed._letta_two_site_opt.reduced_ring_allocation import (
    expand_ring_pair_space, grow_ring_bond, install_ring_factors)
from pyqed._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from test_letta_ring_sweeps import direct_conditional_ring
from test_letta_ring_compression import problem_and_factors
from test_letta_qchem import integrals


@pytest.mark.parametrize('edge', [0, 1, 2])
@pytest.mark.parametrize('tied', [False, True])
def test_ring_growth_preserves_state_metric_and_shrinks_back(edge, tied):
    state, old, a, b, ranks = problem_and_factors(edge, tied=tied)
    original = direct_conditional_ring(state)
    reachable = expand_ring_pair_space(state, edge, 3)
    expanded = grow_ring_bond(reachable, edge,
        {q: n+1 for q,n in Counter(reachable.bond_sectors[(edge+1) % 3]).items()})
    np.testing.assert_allclose(direct_conditional_ring(expanded), original, atol=3e-12)
    np.testing.assert_array_equal(direct_conditional_ring(state), original)
    middle = (edge+1) % 3
    assert sum(map(len, expanded.bond_sectors)) > sum(map(len, state.bond_sectors))
    for i in range(3):
        if i != middle:
            assert expanded.bond_sectors[i] == state.bond_sectors[i]
    problem = CyclicPairProblem(expanded, ElectronicProblem(*integrals(2), (1,1)).su2_mpo(), edge)
    assert problem.layout.shapes == old.layout.shapes
    np.testing.assert_allclose(problem.old_vector, old.old_vector, atol=2e-12)
    x = np.random.default_rng(42).normal(size=problem.local_dimension)
    np.testing.assert_allclose(problem.apply_metric(x), old.apply_metric(x), atol=3e-12)
    np.testing.assert_allclose(problem.apply_hamiltonian(x), old.apply_hamiltonian(x), atol=3e-12)
    aa = problem.left_embedding.pack_source(expanded.site_blocks(edge))
    bb = problem.right_embedding.pack_source(expanded.site_blocks(problem.right_site))
    restored = install_ring_factors(expanded, problem, aa, bb, ranks)
    assert restored.bond_sectors == state.bond_sectors
    np.testing.assert_allclose(direct_conditional_ring(restored), original, atol=3e-12)
    for key, block in state.site_blocks(edge).items():
        np.testing.assert_array_equal(restored.site_blocks(edge)[key], block)


def test_invalid_growth_is_transactional():
    state, p, a, b, ranks = problem_and_factors()
    before = direct_conditional_ring(state)
    with pytest.raises(ValueError, match='discard'):
        grow_ring_bond(state, 0, {})
    with pytest.raises(ValueError, match='positive'):
        expand_ring_pair_space(state, 0, 0)
    with pytest.raises(ValueError, match='retained'):
        install_ring_factors(state, p, a, b, {q: n+1 for q,n in ranks.items()})
    np.testing.assert_array_equal(direct_conditional_ring(state), before)


def test_ring_growth_opens_missing_internal_spin_sector():
    from pyqed.mps.symmetry import Sector
    from pyqed.mps.su2 import SU2Irrep
    from pyqed._letta_one_site_opt import ReducedRingLETTA
    state, _, _, _, _ = problem_and_factors(tied=True)
    middle = Sector(('charge','su2'),(1,SU2Irrep(0)))
    cores=[{k:a for k,a in state.tensors[0].items() if k[2]==middle},
           {k:a for k,a in state.tensors[1].items() if k[0]==middle}]
    restricted=ReducedRingLETTA(state.physical_basis,state.symmetry.sector,cores,
        bond_sectors=(state.bond_sectors[0],(middle,),state.bond_sectors[2]),
        closure=state.closure,neighborhoods=state.neighborhoods)
    original=direct_conditional_ring(restricted)
    grown=expand_ring_pair_space(restricted,0,4)
    assert any(q.components[-1]==SU2Irrep(2) for q in grown.bond_sectors[1])
    np.testing.assert_allclose(direct_conditional_ring(grown),original,atol=2e-12)

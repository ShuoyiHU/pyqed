import numpy as np
import pytest

from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from test_letta_qchem import integrals


@pytest.mark.parametrize('two_s', [0, 2])
@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_conditional_gauge_preserves_multiplets_and_whitens_supported_grams(two_s, direction):
    from pyqed._letta_one_site_opt.reduced_gauge import (
        reduced_frontier_grams, shift_reduced_frontier_gauge)
    p = ElectronicProblem(*integrals(3), (1, 1))
    state = ReducedLatticeLETTA.random((1, 3), symmetry=p.symmetry('su2', two_s=two_s),
        multiplets_per_sector=2, real=False, seed=737)
    original = {m: state.state_vector(target_two_m=m) for m in range(-two_s, two_s+1, 2)}
    cuts = [1, 2] if direction == 'lr' else [2, 1]
    for cut in cuts:
        shift_reduced_frontier_gauge(state, cut, direction)
        for gram in reduced_frontier_grams(state, cut, direction).values():
            values = np.linalg.eigvalsh(gram)
            np.testing.assert_allclose(values[values > 1e-8], 1., atol=2e-9)
        for m, v in original.items():
            np.testing.assert_allclose(state.state_vector(target_two_m=m), v, atol=2e-11)


def test_direct_long_tie_rejects_unavailable_conditional_gauge_before_mutation():
    from pyqed._letta_one_site_opt.reduced_gauge import canonicalize_reduced_frontier
    p = ElectronicProblem(*integrals(4), (1, 1))
    state = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        neighborhoods=tie_neighborhoods(4, [(0, 3)], nearest=True), seed=73)
    v = state.state_vector()
    with pytest.raises(ValueError, match='shared frontier'):
        canonicalize_reduced_frontier(state, 1)
    np.testing.assert_array_equal(state.state_vector(), v)


def test_carried_long_tie_has_an_exact_spin_preserving_gauge():
    from pyqed._letta_one_site_opt.reduced_gauge import canonicalize_reduced_frontier
    p = ElectronicProblem(*integrals(4), (1, 1))
    state = ReducedLatticeLETTA.random((1, 4), symmetry=p.symmetry('su2'),
        neighborhoods=tie_neighborhoods(4, [(0, 3)], nearest=True, carry=True), seed=75)
    v = state.state_vector()
    canonicalize_reduced_frontier(state, 2)
    np.testing.assert_allclose(state.state_vector(), v, atol=1e-12)

"""Large direct contractions should improve without changing local operators."""
import random
import numpy as np
import opt_einsum as oe
import pytest
from pyqed._letta_one_site_opt import LatticeLETTA
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt import contractions as c


@pytest.mark.parametrize('active', [0, 4])
def test_large_direct_operator_path_reduces_work_and_matches_physical_frame(active):
    model = build_model('ising', '2d', (3, 3))
    state = LatticeLETTA.random((3, 3), bond_dim=2, real=False, seed=1723)
    operands, labels, output = c._network_specification(state, model.mpo, active)
    signature = c._canonical_signature(operands, labels, output)
    equation = ','.join(''.join(oe.get_symbol(i) for i in x) for x in signature[1])
    equation += '->' + ''.join(oe.get_symbol(i) for i in signature[2])
    _, baseline = oe.contract_path(equation, *signature[0], shapes=True, optimize='greedy')
    saved_random = random.getstate()
    expression = c._compiled_contraction(signature)
    assert random.getstate() == saved_random
    _, improved = oe.contract_path(equation, *signature[0], shapes=True,
                                  optimize=[step[0] for step in expression.contraction_list])
    if active == 4:
        assert float(improved.opt_cost) < 0.25 * float(baseline.opt_cost)
    else:
        # A lower estimated FLOP count was slower for this small intermediate.
        # Keep the original grouping for this measured regression case.
        assert [sorted(step) for step in improved.path] == [
            sorted(step) for step in baseline.path
        ]
    assert improved.largest_intermediate <= baseline.largest_intermediate
    actual = c.network_operator_matrix(state, model.mpo, active)
    frame = state.local_frame(active)
    expected = frame.conj().T @ model.mpo.to_dense() @ frame
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=2e-12)


def test_prepared_actions_keep_reuse_aware_greedy_path(monkeypatch):
    from pyqed._letta_one_site_opt import contraction_paths
    def forbidden(*args, **kwargs):
        raise AssertionError('single-evaluation search used for a prepared matvec')
    monkeypatch.setattr(contraction_paths, 'large_contraction_path', forbidden)
    rng = np.random.default_rng(1724)
    operands = [rng.normal(size=(64, 64)) / 8 for _ in range(9)]
    labels = [(i, i+1) for i in range(9)]
    expression = c._prepare_contraction(operands, labels, (0, 9), variable=4)
    expected = operands[0]
    for operand in operands[1:]:
        expected = expected @ operand
    np.testing.assert_allclose(expression(operands[4]), expected, atol=2e-12, rtol=2e-12)

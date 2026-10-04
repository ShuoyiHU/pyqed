"""Prepared solves reuse plans while keeping their fixed tensors independent."""

import gc
import weakref

import numpy as np
import opt_einsum as oe
import pytest

from pyqed._letta_one_site_opt import contractions as c


@pytest.mark.parametrize("shapes,labels,output,variable", [
    ([(2, 3), (3, 4), (4, 2)], [(0, 1), (1, 2), (2, 3)], (0, 3), 1),
    ([(2, 3, 2), (3, 2), (2, 4)], [(0, 1, 2), (1, 2), (2, 3)], (0, 3), 0),
    ([(3, 3, 2), (2, 4)], [(0, 0, 1), (1, 2)], (2,), 0),
    ([(1, 3), (4, 3), (4, 2)], [(0, 1), (0, 1), (0, 2)], (1, 2), 1),
    ([(3, 3)], [(0, 0)], (), 0),
    ([(2,), (3,)], [(0,), (1,)], (1, 0), 1),
    ([(2, 3), (3, 4)], [(0, 1), (1, 2)], (2, 0), 1),
    ([(2, 0), (0, 3)], [(0, 1), (1, 2)], (0, 2), 1),
])
def test_prepared_binding_reuses_plan_and_preserves_operations(
    shapes, labels, output, variable, monkeypatch
):
    rng = np.random.default_rng(1725)
    operands = [rng.normal(size=shape) + 1j * rng.normal(size=shape) for shape in shapes]
    signature = c._canonical_signature(operands, labels, output)
    template = c._compiled_contraction(signature)
    original_steps = list(template.contraction_list)
    arguments = list(operands)
    arguments[variable] = shapes[variable]
    reference = oe.contract_expression(
        template.contraction, *arguments,
        constants=[i for i in range(len(operands)) if i != variable],
        optimize=[step[0] for step in original_steps],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("an already compiled shape plan was compiled again")

    with monkeypatch.context() as m:
        m.setattr(oe, "contract_expression", forbidden)
        action = c._prepare_contraction(operands, labels, output, variable)
        for _ in range(3):
            value = rng.normal(size=shapes[variable]) + 1j * rng.normal(size=shapes[variable])
            np.testing.assert_array_equal(action(value), reference(value))
            # Noncontiguous inputs and a different dtype use the same plan.
            value = np.asfortranarray(value.real)
            np.testing.assert_array_equal(action(value), reference(value))
    assert template.contraction_list == original_steps
    assert template.num_args == len(operands)


def test_new_solve_binds_new_constants_without_altering_existing_solve():
    rng = np.random.default_rng(1726)
    labels, output = [(0, 1), (1, 2), (2, 3)], (0, 3)
    operands = [rng.normal(size=(5, 5)) for _ in labels]
    action = c._prepare_contraction(operands, labels, output, variable=1)
    first = action(operands[1]).copy()
    # A different solve can use the same shape plan with different values and
    # dtype, while the previous solve still owns its constant intermediates.
    changed = [operands[0] + 1j * rng.normal(size=(5, 5)), operands[1], 2 * operands[2]]
    fresh = c._prepare_contraction(changed, labels, output, variable=1)
    expected = changed[0] @ changed[1] @ changed[2]
    np.testing.assert_allclose(fresh(changed[1]), expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(action(operands[1]), first)
    np.testing.assert_allclose(first, operands[0] @ operands[1] @ operands[2], atol=2e-14)


def test_prepared_constant_prefix_is_lazy_and_reused(monkeypatch):
    from pyqed._letta_one_site_opt import numpy_contractions as execution
    rng = np.random.default_rng(1727)
    operands = [rng.normal(size=shape) for shape in [(2, 5), (5, 2), (2, 7)]]
    calls = []
    original = execution._cached_tensordot

    def counted(a, b, *args, **kwargs):
        if (a is operands[0] and b is operands[1]) or (a is operands[1] and b is operands[0]):
            calls.append('constant')
        return original(a, b, *args, **kwargs)

    # Compilation may retain the matrix-product helper in a cached instruction.
    from pyqed._letta_one_site_opt.numpy_contractions import _numpy_steps
    _numpy_steps.cache_clear()
    with monkeypatch.context() as m:
        m.setattr(execution, '_cached_tensordot', counted)
        action = c._prepare_contraction(operands, [(0, 1), (1, 2), (2, 3)], (0, 3), 2)
        assert calls == []
        for scale in [1., 2., -0.5]:
            value = scale * operands[2]
            np.testing.assert_allclose(action(value), operands[0] @ operands[1] @ value, atol=1e-14)
        assert calls == ['constant']
    _numpy_steps.cache_clear()


def test_prepared_global_plans_do_not_keep_tensor_values_alive():
    operands = [np.ones((3, 3)) for _ in range(3)]
    references = [weakref.ref(operand) for operand in operands]
    action = c._prepare_contraction(operands, [(0, 1), (1, 2), (2, 3)], (0, 3), 1)
    action(operands[1])
    del operands, action
    gc.collect()
    assert all(reference() is None for reference in references)

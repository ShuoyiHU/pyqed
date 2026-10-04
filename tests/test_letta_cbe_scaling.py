import numpy as np
import pytest

from pyqed._letta_one_site_opt.benchmarks.cbe_scaling import (
    _profile_call,
    run_scaling_profile,
)


def test_profile_counts_prepared_constants_once_and_each_action():
    import opt_einsum as oe
    a, b, x = np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5))
    action = oe.contract_expression(
        "ab,bc,cd->ad", a, b, x.shape, constants=[0, 1],
        optimize=[(0, 1), (0, 1)],
    )
    def twice():
        action(x)
        return action(x)
    result, report = _profile_call(twice)
    np.testing.assert_allclose(result, a @ b @ x)
    assert report["opt_cost"] == 2 * 2 * 3 * 4 + 2 * (2 * 2 * 4 * 5)
    assert report["contractions"] == 3


@pytest.mark.parametrize("warm", [False, True])
def test_profile_counts_prepared_numpy_actions_and_reused_constants(warm):
    import opt_einsum as oe
    from pyqed._letta_one_site_opt.numpy_contractions import prepare_numpy_contraction

    a, b, x = np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5))
    expression = oe.contract_expression("ab,bc,cd->ad", a.shape, b.shape, x.shape,
                                        optimize=[(0, 1), (0, 1)])
    # Prepare before profiling, including already cached kernel callables.
    action = prepare_numpy_contraction(expression.contraction_list, [a, b, x], 2)
    if warm:
        action(x)

    def twice():
        action(x)
        return action(x)

    result, report = _profile_call(twice)
    np.testing.assert_array_equal(result, a @ b @ x)
    assert report["opt_cost"] == (0 if warm else 2 * 2 * 3 * 4) + 2 * (2 * 2 * 4 * 5)
    assert report["contractions"] == (2 if warm else 3)


def test_profile_restores_observers_when_the_call_raises():
    import sys

    original = sys.getprofile()
    svd, eigh = np.linalg.svd, np.linalg.eigh

    def fails():
        raise RuntimeError("profiled failure")

    with pytest.raises(RuntimeError, match="profiled failure"):
        _profile_call(fails)
    assert sys.getprofile() is original
    assert np.linalg.svd is svd and np.linalg.eigh is eigh


@pytest.mark.parametrize("equation,shapes,cost", [
    ("ab,ab->ab", ((2, 3), (2, 3)), 6),
    ("a,b->ab", ((2,), (3,)), 6),
    ("ab,bc->ca", ((2, 3), (3, 4)), 48),
])
def test_profile_counts_prepared_numpy_kernel_kinds(equation, shapes, cost):
    import opt_einsum as oe
    from pyqed._letta_one_site_opt.numpy_contractions import prepare_numpy_contraction

    a, b = (np.ones(shape) for shape in shapes)
    expression = oe.contract_expression(equation, *shapes, optimize=[(0, 1)])
    action = prepare_numpy_contraction(expression.contraction_list, [a, b], 1)
    result, report = _profile_call(lambda: action(b.tolist()))
    np.testing.assert_array_equal(result, np.einsum(equation, a, b))
    assert report["opt_cost"] == cost
    assert report["contractions"] == 1


@pytest.mark.parametrize("direction", ["lr", "rl"])
def test_general_selector_profile_includes_physical_metric_and_actual_paths(direction):
    report = run_scaling_profile(
        bond_dimensions=(2, 4, 8),
        physical_dimensions=(2, 3, 4),
        mpo_widths=(8, 16, 32),
        direction=direction,
    )
    assert report["proof"]["scope"] == "selector_and_single_actions_not_full_updates"
    assert report["proof"]["pair_actions"] == 0
    assert report["proof"]["pair_metrics"] == 0
    assert report["proof"]["merged_pairs"] == 0
    assert not report["proof"]["universal_one_site_cost"]
    for axis in ("bond", "physical", "mpo"):
        assert all(np.isfinite(value) for value in report["exponents"][axis].values())
        for point in report[axis + "_profile"]:
            selector = point["strict_selector"]
            assert selector["used_physical_metric"]
            assert selector["contractions"] > 0
            assert selector["opt_cost"] > 0
            assert selector["svd_calls"] > 0
            assert selector["eigh_calls"] > 0
            assert selector["work_proxy"] >= selector["opt_cost"]
            assert selector["largest_live_tensor"] >= selector["output_size"]
            assert selector["connector_dimension"] >= point["bond_dimension"]

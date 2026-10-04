"""Actual selector contraction/decomposition audit, not a full-update proof.

The general physical selector includes connector routing, one-site cross
Grams, and supported metric fitting. Contraction costs and decomposition
work proxies are recorded from calls, not inferred from legacy MPS shapes.
The work proxy excludes uninstrumented NumPy matrix products and is not a
complete FLOP count. Full-update timing requires the convergence benchmarks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from functools import partial
from types import CodeType

import numpy as np
import opt_einsum as oe

from ..._letta_two_site_opt import (
    IdentityPairEnvironmentCache,
    LETTAPairEnvironmentCache,
    LETTAPairLayout,
)
from .. import cbe as cbe_module
from ..operators import LatticeMPO
from ..numpy_contractions import _cached_tensordot, prepare_numpy_contraction
from ..state import LatticeLETTA


def _profile_mpo(nsites, physical_dim, width, seed):
    """Return a sparse-width MPO with exactly ``width`` paths per bulk site."""

    rng = np.random.default_rng(seed)
    factors = []
    for site in range(nsites):
        left_width = 1 if site == 0 else width
        right_width = 1 if site == nsites - 1 else width
        factor = np.zeros(
            (left_width, right_width, physical_dim, physical_dim)
        )
        if site == 0:
            transitions = ((0, channel) for channel in range(width))
        elif site == nsites - 1:
            transitions = ((channel, 0) for channel in range(width))
        else:
            transitions = ((channel, channel) for channel in range(width))
        for left_channel, right_channel in transitions:
            factor[left_channel, right_channel] = (
                rng.normal(size=(physical_dim, physical_dim)) / physical_dim
            )
        factors.append(factor)
    return LatticeMPO(factors, lattice_shape=(1, nsites))


def _size(value):
    if hasattr(value, "size"):
        return int(value.size)
    if hasattr(value, "shape"):
        return int(np.prod(value.shape))
    return 1


def _equation_metrics(shapes, equation, removed):
    inputs, output = equation.split("->")
    dimensions = {}
    for labels, shape in zip(inputs.split(","), shapes):
        for label, dimension in zip(labels, shape):
            dimensions[label] = max(dimensions.get(label, 1), dimension)
    result_shape = tuple(dimensions[label] for label in output)
    cost = oe.helpers.flop_count(set(dimensions), bool(removed), len(shapes), dimensions)
    largest = max(int(np.prod(shape)) for shape in shapes + [result_shape])
    return int(cost), largest, result_shape


def _executed_path_metrics(operands, contraction_list, evaluate_constants):
    """Count executed steps, including the once-only constant preparation."""
    shapes = [None if value is None else value.shape for value in operands]
    for indices, removed, equation, _remaining, _blas in contraction_list:
        if evaluate_constants and any(shapes[index] is None for index in indices):
            break
        selected = [shapes.pop(index) for index in indices]
        cost, largest, result_shape = _equation_metrics(selected, equation, removed)
        yield cost, largest
        shapes.append(result_shape)


def _executed_numpy_metrics(values):
    """Read an executing prepared action without modifying its cached kernels."""
    shapes = [np.shape(values["value"] if a is None else a) for a in values["fixed"]]
    steps = values["steps"] if values["remaining"] is None else values["remaining"]
    for positions, kernel in steps:
        selected = [shapes.pop(i) for i in positions]
        if isinstance(kernel, partial) and kernel.func is np.einsum:
            equation = kernel.args[0]
            inputs, output = equation.split("->")
            removed = set(inputs.replace(",", "")) - set(output)
            cost, largest, result_shape = _equation_metrics(selected, equation, removed)
        else:
            if isinstance(kernel, partial) and kernel.func in (np.tensordot, _cached_tensordot):
                axes, transpose = kernel.keywords["axes"], None
            else:
                # The only other prepared kernel is tensordot followed by a
                # transpose; its two immutable defaults specify those axes.
                axes, transpose = kernel.__defaults__
            left, right = selected
            result_shape = tuple(d for i, d in enumerate(left) if i not in axes[0])
            result_shape += tuple(d for i, d in enumerate(right) if i not in axes[1])
            cost = int(np.prod(result_shape)) * int(np.prod([left[i] for i in axes[0]]))
            cost *= 2 if axes[0] else 1
            if transpose is not None:
                result_shape = tuple(result_shape[i] for i in transpose)
            largest = max(int(np.prod(shape)) for shape in selected + [result_shape])
        yield cost, largest
        shapes.append(result_shape)


def _profile_call(function, *, live_tensors=()):
    # Observe execution rather than the LETTA wrappers: prepared expressions
    # bypass those wrappers and reuse their constant contractions across calls.
    contract_module = importlib.import_module("opt_einsum.contract")
    original_contract = contract_module._core_contract
    contractions = []
    svds, eighs = [], []
    original_svd, original_eigh = np.linalg.svd, np.linalg.eigh
    original_profile = sys.getprofile()
    action_code = next(code for code in prepare_numpy_contraction.__code__.co_consts
                       if isinstance(code, CodeType) and code.co_name == "action")

    def recorded_execution(frame, event, _argument):
        if event == "call" and frame.f_code is action_code:
            contractions.extend(_executed_numpy_metrics(frame.f_locals))

    def recorded_contract(operands, contraction_list, backend="auto",
                          evaluate_constants=False, out=None, **kwargs):
        contractions.extend(_executed_path_metrics(
            operands, contraction_list, evaluate_constants
        ))
        return original_contract(
            operands, contraction_list, backend=backend,
            evaluate_constants=evaluate_constants, out=out, **kwargs
        )

    def recorded_svd(a, *args, **kwargs):
        svds.append(a.shape)
        return original_svd(a, *args, **kwargs)

    def recorded_eigh(a, *args, **kwargs):
        eighs.append(a.shape)
        return original_eigh(a, *args, **kwargs)

    contract_module._core_contract = recorded_contract
    np.linalg.svd, np.linalg.eigh = recorded_svd, recorded_eigh
    sys.setprofile(recorded_execution)
    try:
        result = function()
    finally:
        sys.setprofile(original_profile)
        contract_module._core_contract = original_contract
        np.linalg.svd, np.linalg.eigh = original_svd, original_eigh
    live_sizes = [int(size) for size in live_tensors]
    live_sizes.append(_size(result))
    live_sizes.extend(size for _cost, size in contractions)
    live_sizes.extend(int(np.prod(shape)) for shape in svds + eighs)
    return result, {
        "opt_cost": int(sum(cost for cost, _size_ in contractions)),
        "largest_live_tensor": int(max(live_sizes, default=1)),
        "contractions": len(contractions),
        "output_size": _size(result),
        "svd_calls": len(svds),
        "eigh_calls": len(eighs),
        "svd_work_proxy": sum(_svd_work(shape) for shape in svds),
        "eigh_work_proxy": sum(shape[-1] ** 3 for shape in eighs),
    }


def _svd_work(shape):
    rows, columns = (int(dimension) for dimension in shape)
    return rows * columns * min(rows, columns)


def _profile_point(
    bond_dimension,
    physical_dimension,
    mpo_width,
    direction,
    *,
    seed,
):
    nsites = 6
    state = LatticeLETTA.random(
        (1, nsites),
        physical_dim=physical_dimension,
        bond_dim=bond_dimension,
        seed=seed,
    )
    mpo = _profile_mpo(
        nsites,
        physical_dimension,
        mpo_width,
        seed + 1,
    )
    cache = LETTAPairEnvironmentCache(
        state, mpo, use_sparse_mpo=True
    )
    left_environments = cache.build_left_environments()
    right_environments = cache.build_right_environments()
    left_site = 2
    right_site = left_site + 1
    layout = LETTAPairLayout.from_state(state, left_site)
    left_tensor = state.tensors[left_site]
    right_tensor = state.tensors[right_site]
    metric_cache = IdentityPairEnvironmentCache(state)
    metric_left = metric_cache.build_left_environments()[left_site]
    metric_right = metric_cache.build_right_environments()[right_site + 1]

    if direction == "lr":
        active_site = left_site
        active_tensor = left_tensor
        active_left = left_environments[left_site]
        active_right = right_environments[left_site + 1]
    else:
        active_site = right_site
        active_tensor = right_tensor
        active_left = left_environments[right_site]
        active_right = right_environments[right_site + 1]

    _one_site_result, one_site = _profile_call(
        lambda: cache.effective_action(
            active_left,
            active_right,
            active_site,
            active_tensor.reshape(-1),
        ),
        live_tensors=(
            _size(active_left),
            _size(active_right),
            active_tensor.size,
        ),
    )

    strict_live = (
        _size(left_environments[left_site]),
        _size(right_environments[right_site + 1]),
    )
    preselection_dimension = (
        (bond_dimension + mpo_width - 1) // mpo_width
    ) * mpo_width
    selection, strict_selector = _profile_call(
        lambda: cbe_module.streamed_shrewd_cbe_selection(
            cache,
            left_environments[left_site],
            right_environments[right_site + 1],
            layout,
            left_tensor,
            right_tensor,
            expansion_dimension=1,
            preselection_dimension=preselection_dimension,
            direction=direction,
            metric_cache=metric_cache, metric_left=metric_left,
            metric_right=metric_right, energy=0.,
        ),
        live_tensors=strict_live,
    )
    strict_selector["largest_live_tensor"] = max(
        strict_selector["largest_live_tensor"],
        selection.preselection_output_size or 0,
        selection.final_output_size or 0,
    )
    strict_selector["output_size"] = max(
        selection.preselection_output_size or 0,
        selection.final_output_size or 0,
    )
    strict_selector["work_proxy"] = int(
        strict_selector["opt_cost"] + strict_selector["svd_work_proxy"]
        + strict_selector["eigh_work_proxy"]
    )
    strict_selector["used_physical_metric"] = True
    strict_selector["connector_dimension"] = selection.connector_dimension
    strict_selector["pair_actions"] = selection.pair_action_count
    strict_selector["pair_metrics"] = selection.pair_metric_count
    strict_selector["merged_pairs"] = selection.merged_pair_count

    pair_tensor = layout.merge(left_tensor, right_tensor)
    _pair_result, pair_action = _profile_call(
        lambda: cache.effective_pair_action(
            left_environments[left_site],
            right_environments[right_site + 1],
            layout,
            pair_tensor.reshape(-1),
        ),
        live_tensors=strict_live + (pair_tensor.size,),
    )
    return {
        "bond_dimension": int(bond_dimension),
        "physical_dimension": int(physical_dimension),
        "mpo_width": int(mpo_width),
        "one_site_action": one_site,
        "strict_selector": strict_selector,
        "pair_action": pair_action,
    }


def _scaling_exponent(points, independent, method, metric="opt_cost"):
    coordinates = np.asarray(
        [point[independent] for point in points], dtype=float
    )
    measurements = np.asarray(
        [point[method][metric] for point in points], dtype=float
    )
    return float(
        np.polyfit(np.log(coordinates), np.log(measurements), 1)[0]
    )


def run_scaling_profile(
    *,
    bond_dimensions=(2, 4, 8, 16),
    physical_dimensions=(2, 3, 4),
    mpo_widths=(8, 16, 32),
    direction="lr",
    seed=610,
):
    """Profile actual contraction graphs along the ``D``, ``d``, and ``w`` axes."""

    direction = str(direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'.")
    bond_dimensions = tuple(int(value) for value in bond_dimensions)
    physical_dimensions = tuple(int(value) for value in physical_dimensions)
    mpo_widths = tuple(int(value) for value in mpo_widths)
    if any(len(values) < 2 for values in (
        bond_dimensions,
        physical_dimensions,
        mpo_widths,
    )):
        raise ValueError("each scaling axis requires at least two values.")
    if any(
        value <= 0
        for values in (
            bond_dimensions,
            physical_dimensions,
            mpo_widths,
        )
        for value in values
    ):
        raise ValueError("scaling dimensions must be positive.")

    fixed_bond = 4
    fixed_physical = 2
    fixed_width = 4
    bond_profile = [
        _profile_point(
            dimension,
            fixed_physical,
            fixed_width,
            direction,
            seed=seed + index,
        )
        for index, dimension in enumerate(bond_dimensions)
    ]
    physical_profile = [
        _profile_point(
            fixed_bond,
            dimension,
            fixed_width,
            direction,
            seed=seed + 100 + index,
        )
        for index, dimension in enumerate(physical_dimensions)
    ]
    mpo_profile = [
        _profile_point(
            fixed_bond,
            fixed_physical,
            width,
            direction,
            seed=seed + 200 + index,
        )
        for index, width in enumerate(mpo_widths)
    ]
    methods = ("one_site_action", "strict_selector", "pair_action")
    exponents = {
        "bond": {
            method: _scaling_exponent(
                bond_profile, "bond_dimension", method
            )
            for method in methods
        },
        "physical": {
            method: _scaling_exponent(
                physical_profile, "physical_dimension", method
            )
            for method in methods
        },
        "mpo": {
            method: _scaling_exponent(mpo_profile, "mpo_width", method)
            for method in methods
        },
    }
    exponents["bond"]["strict_selector_with_svd"] = _scaling_exponent(
        bond_profile,
        "bond_dimension",
        "strict_selector",
        "work_proxy",
    )
    exponents["physical"]["strict_selector_with_svd"] = (
        _scaling_exponent(
            physical_profile,
            "physical_dimension",
            "strict_selector",
            "work_proxy",
        )
    )
    exponents["mpo"]["strict_selector_with_svd"] = _scaling_exponent(
        mpo_profile,
        "mpo_width",
        "strict_selector",
        "work_proxy",
    )
    return {
        "direction": direction,
        "bond_profile": bond_profile,
        "physical_profile": physical_profile,
        "mpo_profile": mpo_profile,
        "exponents": exponents,
        "proof": {
            "scope": "selector_and_single_actions_not_full_updates",
            "universal_one_site_cost": False,
            **{name: sum(p["strict_selector"][name]
                         for profile in (bond_profile, physical_profile, mpo_profile)
                         for p in profile)
               for name in ("pair_actions", "pair_metrics", "merged_pairs")},
            "path_optimizer": "opt_einsum greedy",
            "timing_used_as_proof": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("lr", "rl"), default="lr")
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run_scaling_profile(direction=arguments.direction),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

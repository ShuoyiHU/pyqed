"""Deterministic contraction-graph scaling audit for strict LETTA-CBE.

The audit profiles the actual sparse one-site, streamed selector, and pair
contraction functions with ``opt_einsum``'s greedy path.  Wall time is not used
as scaling evidence.  A banded synthetic MPO makes the number of nonzero
transitions proportional to its width, so the MPO-width exponent is explicit.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import opt_einsum as oe

from ..._letta_two_site_opt import (
    LETTAPairEnvironmentCache,
    LETTAPairLayout,
)
from ..._letta_two_site_opt import contractions as pair_contractions
from .. import cbe as cbe_module
from .. import contractions as one_site_contractions
from ..operators import LatticeMPO
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


def _path_metrics(operands, labels, output):
    shapes, canonical_labels, canonical_output = (
        one_site_contractions._canonical_signature(
            operands, labels, output
        )
    )
    inputs = [
        "".join(oe.get_symbol(label) for label in indices)
        for indices in canonical_labels
    ]
    equation = ",".join(inputs) + "->" + "".join(
        oe.get_symbol(label) for label in canonical_output
    )
    _path, information = oe.contract_path(
        equation,
        *shapes,
        shapes=True,
        optimize="greedy",
    )
    largest_operand = max(
        (int(np.prod(operand.shape)) for operand in operands), default=1
    )
    return int(information.opt_cost), max(
        largest_operand, int(information.largest_intermediate)
    )


def _profile_call(module, function, *, live_tensors=()):
    original = module._contract_operands
    contractions = []

    def recorded(operands, labels, output):
        contractions.append(_path_metrics(operands, labels, output))
        return original(operands, labels, output)

    module._contract_operands = recorded
    try:
        result = function()
    finally:
        module._contract_operands = original
    live_sizes = [int(size) for size in live_tensors]
    live_sizes.append(_size(result))
    live_sizes.extend(size for _cost, size in contractions)
    return result, {
        "opt_cost": int(sum(cost for cost, _size_ in contractions)),
        "largest_live_tensor": int(max(live_sizes, default=1)),
        "contractions": len(contractions),
        "output_size": _size(result),
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
        one_site_contractions,
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
        cbe_module,
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
    left_parent = int(np.prod(left_tensor.shape[:-1]))
    right_parent = int(np.prod(right_tensor.shape[1:]))
    if direction == "lr":
        opposite_shape = (left_parent, bond_dimension * mpo_width)
        opposite_rank = min(opposite_shape)
        preselection_shape = (opposite_rank, right_parent)
        retained_preselection = min(
            preselection_dimension, *preselection_shape
        )
        final_shape = (left_parent, retained_preselection)
    else:
        opposite_shape = (bond_dimension * mpo_width, right_parent)
        opposite_rank = min(opposite_shape)
        preselection_shape = (left_parent, opposite_rank)
        retained_preselection = min(
            preselection_dimension, *preselection_shape
        )
        final_shape = (retained_preselection, right_parent)
    svd_work = (
        _svd_work(opposite_shape)
        + _svd_work(preselection_shape)
        + _svd_work(final_shape)
    )
    strict_selector["svd_work_proxy"] = int(svd_work)
    strict_selector["work_proxy"] = int(
        strict_selector["opt_cost"] + svd_work
    )

    pair_tensor = layout.merge(left_tensor, right_tensor)
    _pair_result, pair_action = _profile_call(
        pair_contractions,
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
            "pair_actions": 0,
            "pair_metrics": 0,
            "merged_pairs": 0,
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

"""Bounded exact path search for large, unprepared LETTA contractions."""

from math import exp
from random import Random

import opt_einsum as oe


def large_contraction_path(equation, signature):
    """Try randomized greedy paths without enlarging the largest intermediate.

    Small frontier and prepared matvec contractions keep their existing paths.
    Optimizing the cost of one full evaluation is inappropriate for a matvec
    whose fixed sub-contractions are already evaluated once and reused.
    """
    shapes, labels, output = signature
    best_path, baseline = oe.contract_path(
        equation, *shapes, shapes=True, optimize="greedy",
    )
    if (baseline.opt_cost < 1_000_000
            or baseline.largest_intermediate < 65_536):
        return best_path
    generic_cost = max(
        (oe.helpers.flop_count(
            set(step_equation.split("->")[0].replace(",", "")),
            bool(removed), len(positions), baseline.size_dict,
        ) for positions, removed, step_equation, _remaining, blas
         in baseline.contraction_list if not blas or "EINSUM" in blas),
        default=0,
    )
    if generic_cost < 1_000_000:
        return best_path

    # Integer labels and a private RNG make the search independent of Python's
    # string hash seed and leave the application's random state untouched.
    inputs = [frozenset(indices) for indices in labels]
    output = frozenset(output)
    dimensions = {
        index: baseline.size_dict[oe.get_symbol(index)]
        for indices in labels
        for index in indices
    }
    rng = Random(0)
    best_cost = baseline.opt_cost
    for trial in range(32):
        strength = 0.1 + 0.3 * (trial % 4)

        def randomized_cost(size12, size1, size2, *_indices):
            return (size12 - size1 - size2) * exp(rng.gauss(0.0, strength))

        path = oe.paths.greedy(
            inputs, output, dimensions, cost_fn=randomized_cost,
        )
        _, info = oe.contract_path(equation, *shapes, shapes=True, optimize=path)
        # Small FLOP savings can lose optimized BLAS kernels and cost more in
        # practice. Require a substantial work and memory reduction instead.
        if (info.opt_cost < best_cost
                and 4 * info.opt_cost <= baseline.opt_cost
                and 2 * info.largest_intermediate <= baseline.largest_intermediate):
            best_path, best_cost = path, info.opt_cost
    return best_path

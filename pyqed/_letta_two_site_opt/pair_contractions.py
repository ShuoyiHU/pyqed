"""Cache shape-only execution plans for two-tensor pair contractions."""

from functools import lru_cache
from math import prod

import numpy as np


@lru_cache(maxsize=512)
def _pair_contraction_plan(equation, left_shape, right_shape):
    inputs, output = equation.split("->")
    # Match NumPy's optimized two-operand order, including singleton axes and
    # the output view, so downstream matrix products see the same layout.
    a_term, b_term = inputs.split(",")[::-1]
    a_shape, b_shape = right_shape, left_shape
    if len(set(a_term)) != len(a_term) or len(set(b_term)) != len(b_term):
        return None
    sizes = dict(zip(a_term, a_shape))
    sizes.update(zip(b_term, b_shape))
    if any(sizes[label] != size for label, size in zip(a_term, a_shape)):
        return None
    a = [label for label in a_term if sizes[label] != 1]
    b = [label for label in b_term if sizes[label] != 1]
    batch = [label for label in a if label in b and label in output]
    summed = [label for label in a if label in b and label not in output]
    if not summed:
        return None
    kept_a = [label for label in a if label not in b and label in output]
    kept_b = [label for label in b if label not in a and label in output]
    singletons = [label for label in output if sizes[label] == 1]
    a_groups = (batch, kept_a, summed) if batch else (kept_a, summed)
    b_groups = (batch, summed, kept_b) if batch else (summed, kept_b)
    out_groups = (batch, kept_a, kept_b) if batch else (kept_a, kept_b)

    def preparation(term, groups):
        desired = "".join(label for group in groups for label in group)
        pre_equation = f"{term}->{desired}" if term != desired else None
        shape = tuple(prod(sizes[label] for label in group) for group in groups)
        return pre_equation, shape if any(len(group) != 1 for group in groups) else None

    a_equation, a_shape = preparation(a_term, a_groups)
    b_equation, b_shape = preparation(b_term, b_groups)
    out_shape = ((1,) * len(singletons)
                 + tuple(sizes[label] for group in out_groups for label in group))
    if not singletons and all(len(group) == 1 for group in out_groups):
        out_shape = None
    natural = "".join(singletons + batch + kept_a + kept_b)
    permutation = tuple(natural.index(label) for label in output) if natural != output else None
    return a_equation, a_shape, b_equation, b_shape, out_shape, permutation


def contract_pair(equation, left, right):
    """Execute a pair map with NumPy's matrix-product order and output strides.

    LETTA pair equations have unique labels on each input and matching shared
    dimensions. Other cases and pure multiplication retain NumPy's planner.
    The cache contains only strings, dimensions and axis permutations.
    """
    metadata = _pair_contraction_plan(equation, left.shape, right.shape)
    if metadata is None:
        return np.einsum(equation, left, right, optimize=True)
    a_equation, a_shape, b_equation, b_shape, out_shape, permutation = metadata
    a, b = right, left
    if a_equation is not None:
        a = np.einsum(a_equation, a)
    if a_shape is not None:
        a = a.reshape(a_shape)
    if b_equation is not None:
        b = np.einsum(b_equation, b)
    if b_shape is not None:
        b = b.reshape(b_shape)
    result = np.matmul(a, b)
    if out_shape is not None:
        result = result.reshape(out_shape)
    if permutation is not None:
        result = result.transpose(permutation)
    return result

"""Pre-parsed NumPy execution of fixed opt_einsum plans for local actions."""

from functools import lru_cache, partial
from math import prod

import numpy as np
from opt_einsum import contract_expression, parser


@lru_cache(maxsize=512)
def _tensordot_plan(shape_a, shape_b, axes):
    """Cache only NumPy's matrix shapes and axis order for a prepared step."""
    axes_a, axes_b = axes
    if (len(axes_a) != len(axes_b) or len(set(axes_a)) != len(axes_a)
            or len(set(axes_b)) != len(axes_b)
            or any(i < 0 or i >= len(shape_a) for i in axes_a)
            or any(i < 0 or i >= len(shape_b) for i in axes_b)
            or any(shape_a[i] != shape_b[j] for i, j in zip(axes_a, axes_b))):
        return None
    free_a = tuple(i for i in range(len(shape_a)) if i not in axes_a)
    free_b = tuple(i for i in range(len(shape_b)) if i not in axes_b)
    shared = prod(shape_a[i] for i in axes_a)
    matrix_a = (prod(shape_a[i] for i in free_a), shared)
    matrix_b = (shared, prod(shape_b[i] for i in free_b))
    output = tuple(shape_a[i] for i in free_a) + tuple(shape_b[i] for i in free_b)
    return free_a + axes_a, matrix_a, axes_b + free_b, matrix_b, output


def _cached_tensordot(a, b, axes):
    """Execute NumPy's transpose/reshape/dot sequence with cached shape setup."""
    a, b = np.asarray(a), np.asarray(b)
    plan = _tensordot_plan(a.shape, b.shape, axes)
    if plan is None:
        return np.tensordot(a, b, axes=axes)
    order_a, shape_a, order_b, shape_b, output = plan
    left = a.transpose(order_a).reshape(shape_a)
    right = b.transpose(order_b).reshape(shape_b)
    return np.dot(left, right).reshape(output)


@lru_cache(maxsize=512)
def _numpy_steps(key):
    """Cache only operand positions, equations and axes, never tensor values."""
    steps = []
    for positions, removed, equation, blas in key:
        if blas and "EINSUM" not in blas:
            inputs, output = equation.split("->")
            left, right = inputs.split(",")
            natural = "".join(index for index in left + right if index not in removed)
            axes = (
                tuple(zip(*sorted((left.index(i), right.index(i)) for i in removed)))
                if removed else ((), ())
            )
            if natural != output:
                transpose = tuple(map(natural.index, output))

                def kernel(a, b, axes=axes, transpose=transpose):
                    return _cached_tensordot(a, b, axes=axes).transpose(transpose)

            else:
                kernel = partial(_cached_tensordot, axes=axes)
        else:
            if not parser.has_valid_einsum_chars_only(equation):
                equation = parser.convert_to_valid_einsum_chars(equation)
            kernel = partial(np.einsum, equation)
        steps.append((positions, kernel))
    return tuple(steps)


@lru_cache(maxsize=512)
def _einsum_steps(equation, shapes):
    expression = contract_expression(equation, *shapes, optimize='greedy')
    key = tuple((tuple(step[0]), frozenset(step[1]), step[2], step[-1])
                for step in expression.contraction_list)
    return _numpy_steps(key)


def cached_einsum(equation, *operands):
    """Execute a small repeated contraction without replanning its path.

    Cache shapes, axis orders, and kernels only. All tensor values are supplied
    afresh, including after gauges, in-place updates, or dtype changes.
    """
    arrays = [np.asarray(a) for a in operands]
    steps = _einsum_steps(equation, tuple(a.shape for a in arrays))
    for positions, kernel in steps:
        arguments = [arrays.pop(i) for i in positions]
        arrays.append(kernel(*arguments))
    return arrays[0]


def prepare_numpy_contraction(contraction_list, operands, variable):
    """Bind one variable NumPy operand without changing the compiled operations.

    Fixed tensors belong to this solve and must not change during it. As in
    opt_einsum, evaluate the constant prefix lazily on the first call, stopping
    at the first operation involving the variable operand. Even later constant
    steps retain their original position so floating-point order is unchanged.
    """
    key = tuple(
        (tuple(step[0]), frozenset(step[1]), step[2], step[-1])
        for step in contraction_list
    )
    steps = _numpy_steps(key)
    fixed = [None if i == variable else array for i, array in enumerate(operands)]
    remaining = None
    variable_position = variable

    def action(value):
        nonlocal fixed, remaining, variable_position
        if remaining is None:
            fixed = list(fixed)
            count = 0
            for positions, kernel in steps:
                if any(fixed[i] is None for i in positions):
                    break
                arguments = [fixed.pop(i) for i in positions]
                fixed.append(kernel(*arguments))
                count += 1
            remaining = steps[count:]
            variable_position = next(i for i, array in enumerate(fixed) if array is None)
        arrays = list(fixed)
        arrays[variable_position] = value
        for positions, kernel in remaining:
            arguments = [arrays.pop(i) for i in positions]
            arrays.append(kernel(*arguments))
        return arrays[0]

    return action

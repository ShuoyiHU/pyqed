"""Pre-parsed NumPy execution of fixed opt_einsum plans for local actions.

Shape-plan caches retain up to 8192 entries: a width-three SU(2) sweep
uses thousands of distinct block shapes. Entries hold no tensor values;
eviction only repeats planning and never changes the numerical operation.
"""

from functools import lru_cache, partial
from math import prod

import numpy as np
from opt_einsum import contract_expression, parser


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=8192)
def _accumulation_plan(equation, shapes):
    expression=contract_expression(equation,*shapes,optimize='greedy')
    key=tuple((tuple(step[0]),frozenset(step[1]),step[2],step[-1])
              for step in expression.contraction_list)
    positions,removed,eq,blas=key[-1]
    if not blas or 'EINSUM' in blas:return None
    inputs,output=eq.split('->')
    if len(inputs.split(','))!=2:return None
    left,right=inputs.split(',')
    natural=''.join(i for i in left+right if i not in removed)
    axes=tuple(zip(*sorted((left.index(i),right.index(i)) for i in removed))) if removed else ((),())
    permutation=tuple(natural.index(i) for i in output)
    return _numpy_steps(key[:-1]),positions,axes,permutation


def accumulate_einsum(equation, operands, *, alpha=1., out=None):
    """Accumulate a contraction with GEMM alpha/beta at its final product.

    Keeps the greedy contraction order. BLAS combines final scaling and addition;
    unsupported final steps retain the reference contraction. No rank truncation
    is introduced. Floating-point rounding may differ from separate scaling.
    """
    from scipy.linalg.blas import get_blas_funcs
    arrays=[np.asarray(a) for a in operands]
    plan=_accumulation_plan(equation,tuple(a.shape for a in arrays))
    if plan is None:
        value=cached_einsum(equation,*arrays)*alpha
        return value if out is None else out+value
    steps,positions,axes,permutation=plan
    for positions0,kernel in steps:
        arguments=[arrays.pop(i) for i in positions0]
        arrays.append(kernel(*arguments))
    a,b=[arrays.pop(i) for i in positions]
    layout=_tensordot_plan(a.shape,b.shape,axes)
    if layout is None:
        value=cached_einsum(equation,*operands)*alpha
        return value if out is None else out+value
    oa,sa,ob,sb,natural=layout
    left=a.transpose(oa).reshape(sa);right=b.transpose(ob).reshape(sb)
    if not all(left.shape+right.shape):
        value=(alpha*np.dot(left,right)).reshape(natural).transpose(permutation)
        return value if out is None else out+value
    existing=None if out is None else out.transpose(tuple(np.argsort(permutation))).reshape(sa[0],sb[1])
    dtype=np.result_type(left,right,alpha) if out is None else np.result_type(left,right,out,alpha)
    gemm=get_blas_funcs('gemm',dtype=dtype)
    beta=0. if existing is None else 1.
    # SciPy BLAS expects Fortran storage. Multiply the transpose when the
    # accumulator is row-major, and pass input transposes without copying.
    transpose=existing is None or not existing.flags.f_contiguous
    a,b=(right.T,left.T) if transpose else (left,right)
    c=None if existing is None else (existing.T if transpose else existing)
    ta=not a.flags.f_contiguous;tb=not b.flags.f_contiguous
    matrix=gemm(alpha,a.T if ta else a,b.T if tb else b,
                trans_a=int(ta),trans_b=int(tb),beta=beta,c=c,overwrite_c=1)
    if transpose:matrix=matrix.T
    return matrix.reshape(natural).transpose(permutation)

"""Metric-aware controlled bond expansion for one-site LETTA.

The exact pair-space selector is retained as a correctness oracle.  The
strict shrewd path uses weighted half-environment preselection, a streamed
metric-projected physical residual, an expanded one-site solve, and
one-site-metric trimming.  It never constructs a merged pair tensor, pair
action, or pair metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .contractions import BlockDiagonalMetric, _contract_operands
from .._letta_two_site_opt.pair import (
    LETTAPairLayout,
    conditional_svd_split,
)
from .._letta_two_site_opt.truncation import (
    _MetricSquareRoot,
    metric_als_refine,
)


def _stable_floating_point(function):
    """Ignore stale BLAS flags locally while retaining explicit finite checks."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return function(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class MetricOrthogonalComplement:
    """A vector projected into the supported metric tangent complement."""

    vector: np.ndarray
    metric_rank: int
    tangent_rank: int | None
    norm: float
    tangent_overlap_norm: float
    iterations: int = 0
    converged: bool = True


@dataclass(frozen=True)
class CBEMissingDirection:
    """Hamiltonian-informed direction missing from a LETTA pair tangent."""

    vector: np.ndarray
    energy: float
    metric_rank: int
    tangent_rank: int | None
    missing_norm: float
    tangent_overlap_norm: float
    selector: str = "exact"
    projection_iterations: int = 0
    projection_converged: bool = True
    pair_action_count: int = 1
    materialized_pair_metric: bool = True
    materialized_tangent_jacobian: bool = True


@dataclass(frozen=True)
class CBESelection:
    """Low-rank factors selected from a missing pair direction."""

    left_direction: np.ndarray
    right_direction: np.ndarray
    loss: float
    captured_weight: float
    sector_ranks: tuple[int, ...]
    refinement_iterations: int
    selector: str = "exact"
    preselection_dimension: int | None = None
    preselection_loss: float | None = None
    missing_norm: float | None = None
    pair_action_count: int = 1
    pair_metric_count: int = 1
    merged_pair_count: int = 1
    preselection_output_size: int | None = None
    final_output_size: int | None = None


@dataclass(frozen=True)
class CBETrim:
    """A metric-aware fixed-bond approximation to an expanded pair."""

    left_tensor: np.ndarray
    right_tensor: np.ndarray
    loss: float
    iterations: int
    norm: float


def _complete_contraction(operands, labels, output, output_shape):
    operands = list(operands)
    labels = [tuple(indices) for indices in labels]
    output = tuple(output)
    output_shape = tuple(int(dimension) for dimension in output_shape)
    if len(output) != len(output_shape):
        raise ValueError("contraction output labels and shape do not match.")
    used = {label for indices in labels for label in indices}
    for label, dimension in zip(output, output_shape):
        if label not in used:
            operands.append(np.ones(dimension))
            labels.append((label,))
    result = _contract_operands(operands, labels, output)
    if result.shape != output_shape:
        raise ValueError("streamed contraction returned an unexpected shape.")
    return result


def _project_out_columns(candidate, kept, tolerance):
    candidate = np.asarray(candidate)
    kept = np.asarray(kept)
    if candidate.ndim != 2 or kept.ndim != 2:
        raise ValueError("column projection expects matrices.")
    if candidate.shape[0] != kept.shape[0]:
        raise ValueError("column spaces have incompatible parent dimensions.")
    coefficients = np.linalg.pinv(kept, rcond=float(tolerance)) @ candidate
    return candidate - kept @ coefficients


def _project_out_rows(candidate, kept, tolerance):
    candidate = np.asarray(candidate)
    kept = np.asarray(kept)
    if candidate.ndim != 2 or kept.ndim != 2:
        raise ValueError("row projection expects matrices.")
    if candidate.shape[1] != kept.shape[1]:
        raise ValueError("row spaces have incompatible parent dimensions.")
    coefficients = candidate @ np.linalg.pinv(kept, rcond=float(tolerance))
    return candidate - coefficients @ kept


def _retained_svd_rank(singular_values, maximum, tolerance):
    singular_values = np.asarray(singular_values)
    if singular_values.size == 0:
        return 0
    cutoff = float(tolerance) * float(singular_values[0])
    return min(int(maximum), int(np.count_nonzero(singular_values > cutoff)))


def _pad_left_direction(direction, shape, width):
    result = np.zeros(shape[:-1] + (width,), dtype=direction.dtype)
    result[..., : direction.shape[-1]] = direction
    return result


def _pad_right_direction(direction, shape, width):
    result = np.zeros((width,) + shape[1:], dtype=direction.dtype)
    result[: direction.shape[0]] = direction
    return result


def _zero_streamed_selection(
    left_tensor,
    right_tensor,
    expansion_dimension,
    preselection_output_size,
):
    dtype = np.result_type(left_tensor, right_tensor)
    return CBESelection(
        left_direction=np.zeros(
            left_tensor.shape[:-1] + (expansion_dimension,), dtype=dtype
        ),
        right_direction=np.zeros(
            (expansion_dimension,) + right_tensor.shape[1:], dtype=dtype
        ),
        loss=0.0,
        captured_weight=0.0,
        sector_ranks=(0,),
        refinement_iterations=0,
        selector="shrewd",
        preselection_dimension=0,
        preselection_loss=0.0,
        missing_norm=0.0,
        pair_action_count=0,
        pair_metric_count=0,
        merged_pair_count=0,
        preselection_output_size=int(preselection_output_size),
        final_output_size=0,
    )


def _streamed_shrewd_preselection_tensor(
    hamiltonian_cache,
    hamiltonian_left,
    hamiltonian_right,
    layout,
    left_tensor,
    right_tensor,
    direction,
):
    left_site = layout.left_site
    right_site = left_site + 1
    left_bra, left_ket, left_operator = (
        hamiltonian_cache._group_labels(left_site)
    )
    right_bra, right_ket, right_operator = (
        hamiltonian_cache._group_labels(right_site)
    )
    if direction == "rl":
        output = left_bra[:-1] + (left_ket[-1], left_operator[1])
        output_shape = left_tensor.shape[:-1] + (
            left_tensor.shape[-1],
            hamiltonian_cache.mpo.factors[left_site].shape[1],
        )
        if not hamiltonian_cache.use_sparse_mpo:
            return _complete_contraction(
                [
                    hamiltonian_left,
                    hamiltonian_cache.mpo.factors[left_site],
                    left_tensor,
                ],
                [
                    hamiltonian_cache.frontiers[left_site],
                    left_operator,
                    left_ket,
                ],
                output,
                output_shape,
            )
        result = np.zeros(
            output_shape,
            dtype=np.result_type(
                hamiltonian_left,
                left_tensor,
                hamiltonian_cache.mpo.factors[left_site],
            ),
        )
        channel_label = left_operator[1]
        reduced_output = tuple(
            label for label in output if label != channel_label
        )
        reduced_shape = tuple(
            dimension
            for label, dimension in zip(output, output_shape)
            if label != channel_label
        )
        bra_physical, ket_physical = left_operator[2:]
        for left_channel, right_channel, local_operator in (
            hamiltonian_cache.mpo.transitions[left_site]
        ):
            selected_left, selected_labels = (
                hamiltonian_cache._select_channel(
                    hamiltonian_left,
                    hamiltonian_cache.frontiers[left_site],
                    left_operator[0],
                    left_channel,
                )
            )
            if selected_left is None:
                continue
            value = _complete_contraction(
                [selected_left, local_operator, left_tensor],
                [
                    tuple(selected_labels),
                    (bra_physical, ket_physical),
                    left_ket,
                ],
                reduced_output,
                reduced_shape,
            )
            hamiltonian_cache._add_channel(
                result,
                output,
                channel_label,
                right_channel,
                value,
            )
        return result

    output = (right_ket[0], right_operator[0]) + right_bra[1:]
    output_shape = (
        right_tensor.shape[0],
        hamiltonian_cache.mpo.factors[right_site].shape[0],
    ) + right_tensor.shape[1:]
    if not hamiltonian_cache.use_sparse_mpo:
        return _complete_contraction(
            [
                right_tensor,
                hamiltonian_cache.mpo.factors[right_site],
                hamiltonian_right,
            ],
            [
                right_ket,
                right_operator,
                hamiltonian_cache.frontiers[right_site + 1],
            ],
            output,
            output_shape,
        )
    result = np.zeros(
        output_shape,
        dtype=np.result_type(
            hamiltonian_right,
            right_tensor,
            hamiltonian_cache.mpo.factors[right_site],
        ),
    )
    channel_label = right_operator[0]
    reduced_output = tuple(label for label in output if label != channel_label)
    reduced_shape = tuple(
        dimension
        for label, dimension in zip(output, output_shape)
        if label != channel_label
    )
    bra_physical, ket_physical = right_operator[2:]
    for left_channel, right_channel, local_operator in (
        hamiltonian_cache.mpo.transitions[right_site]
    ):
        selected_right, selected_labels = hamiltonian_cache._select_channel(
            hamiltonian_right,
            hamiltonian_cache.frontiers[right_site + 1],
            right_operator[1],
            right_channel,
        )
        if selected_right is None:
            continue
        value = _complete_contraction(
            [right_tensor, local_operator, selected_right],
            [
                right_ket,
                (bra_physical, ket_physical),
                tuple(selected_labels),
            ],
            reduced_output,
            reduced_shape,
        )
        hamiltonian_cache._add_channel(
            result,
            output,
            channel_label,
            left_channel,
            value,
        )
    return result


def _streamed_shrewd_weighted_preselection_tensor(
    hamiltonian_cache,
    hamiltonian_left,
    hamiltonian_right,
    layout,
    left_tensor,
    right_tensor,
    weighted_half,
    direction,
):
    """Close the candidate half against a weighted opposite half."""

    left_site = layout.left_site
    right_site = left_site + 1
    left_bra, left_ket, left_operator = (
        hamiltonian_cache._group_labels(left_site)
    )
    right_bra, right_ket, right_operator = (
        hamiltonian_cache._group_labels(right_site)
    )
    auxiliary_label = -10_000_002
    weighted_half = np.asarray(weighted_half)
    if direction == "rl":
        expected = (
            left_tensor.shape[-1],
            hamiltonian_cache.mpo.factors[left_site].shape[1],
        )
        if weighted_half.shape[:2] != expected:
            raise ValueError("the weighted right half has incompatible axes.")
        output = left_bra[:-1] + (auxiliary_label,)
        output_shape = left_tensor.shape[:-1] + (weighted_half.shape[-1],)
        weighted_labels = (
            left_ket[-1],
            left_operator[1],
            auxiliary_label,
        )
        if not hamiltonian_cache.use_sparse_mpo:
            return _complete_contraction(
                [
                    hamiltonian_left,
                    hamiltonian_cache.mpo.factors[left_site],
                    left_tensor,
                    weighted_half,
                ],
                [
                    hamiltonian_cache.frontiers[left_site],
                    left_operator,
                    left_ket,
                    weighted_labels,
                ],
                output,
                output_shape,
            )
        result = np.zeros(
            output_shape,
            dtype=np.result_type(
                hamiltonian_left, left_tensor, weighted_half
            ),
        )
        bra_physical, ket_physical = left_operator[2:]
        for left_channel, right_channel, local_operator in (
            hamiltonian_cache.mpo.transitions[left_site]
        ):
            selected_left, selected_labels = (
                hamiltonian_cache._select_channel(
                    hamiltonian_left,
                    hamiltonian_cache.frontiers[left_site],
                    left_operator[0],
                    left_channel,
                )
            )
            if selected_left is None:
                continue
            result += _complete_contraction(
                [
                    selected_left,
                    local_operator,
                    left_tensor,
                    weighted_half[:, right_channel, :],
                ],
                [
                    tuple(selected_labels),
                    (bra_physical, ket_physical),
                    left_ket,
                    (left_ket[-1], auxiliary_label),
                ],
                output,
                output_shape,
            )
        return result

    expected = (
        right_tensor.shape[0],
        hamiltonian_cache.mpo.factors[right_site].shape[0],
    )
    if weighted_half.shape[1:] != expected:
        raise ValueError("the weighted left half has incompatible axes.")
    output = (auxiliary_label,) + right_bra[1:]
    output_shape = (weighted_half.shape[0],) + right_tensor.shape[1:]
    weighted_labels = (
        auxiliary_label,
        right_ket[0],
        right_operator[0],
    )
    if not hamiltonian_cache.use_sparse_mpo:
        return _complete_contraction(
            [
                weighted_half,
                right_tensor,
                hamiltonian_cache.mpo.factors[right_site],
                hamiltonian_right,
            ],
            [
                weighted_labels,
                right_ket,
                right_operator,
                hamiltonian_cache.frontiers[right_site + 1],
            ],
            output,
            output_shape,
        )
    result = np.zeros(
        output_shape,
        dtype=np.result_type(
            hamiltonian_right, right_tensor, weighted_half
        ),
    )
    bra_physical, ket_physical = right_operator[2:]
    for left_channel, right_channel, local_operator in (
        hamiltonian_cache.mpo.transitions[right_site]
    ):
        selected_right, selected_labels = hamiltonian_cache._select_channel(
            hamiltonian_right,
            hamiltonian_cache.frontiers[right_site + 1],
            right_operator[1],
            right_channel,
        )
        if selected_right is None:
            continue
        result += _complete_contraction(
            [
                weighted_half[:, :, left_channel],
                right_tensor,
                local_operator,
                selected_right,
            ],
            [
                (auxiliary_label, right_ket[0]),
                right_ket,
                (bra_physical, ket_physical),
                tuple(selected_labels),
            ],
            output,
            output_shape,
        )
    return result


def _streamed_shrewd_final_tensor(
    hamiltonian_cache,
    hamiltonian_left,
    hamiltonian_right,
    layout,
    left_tensor,
    right_tensor,
    preselected,
    direction,
):
    left_site = layout.left_site
    right_site = left_site + 1
    left_bra, left_ket, left_operator = (
        hamiltonian_cache._group_labels(left_site)
    )
    right_bra, right_ket, right_operator = (
        hamiltonian_cache._group_labels(right_site)
    )
    candidate_label = -10_000_001
    if direction == "rl":
        candidate_labels = left_bra[:-1] + (candidate_label,)
        output = (candidate_label,) + right_bra[1:]
        output_shape = (preselected.shape[-1],) + right_tensor.shape[1:]
    else:
        candidate_labels = (candidate_label,) + right_bra[1:]
        output = left_bra[:-1] + (candidate_label,)
        output_shape = left_tensor.shape[:-1] + (preselected.shape[0],)

    if not hamiltonian_cache.use_sparse_mpo:
        return _complete_contraction(
            [
                hamiltonian_left,
                hamiltonian_cache.mpo.factors[left_site],
                hamiltonian_cache.mpo.factors[right_site],
                hamiltonian_right,
                left_tensor,
                right_tensor,
                preselected.conj(),
            ],
            [
                hamiltonian_cache.frontiers[left_site],
                left_operator,
                right_operator,
                hamiltonian_cache.frontiers[right_site + 1],
                left_ket,
                right_ket,
                candidate_labels,
            ],
            output,
            output_shape,
        )

    result = np.zeros(
        output_shape,
        dtype=np.result_type(
            hamiltonian_left,
            hamiltonian_right,
            left_tensor,
            right_tensor,
            preselected,
        ),
    )
    first_physical = left_operator[2:]
    second_physical = right_operator[2:]
    for left_channel, middle_channel, first_operator in (
        hamiltonian_cache.mpo.transitions[left_site]
    ):
        selected_left, selected_left_labels = (
            hamiltonian_cache._select_channel(
                hamiltonian_left,
                hamiltonian_cache.frontiers[left_site],
                left_operator[0],
                left_channel,
            )
        )
        if selected_left is None:
            continue
        for second_middle, right_channel, second_operator in (
            hamiltonian_cache.mpo.transitions[right_site]
        ):
            if second_middle != middle_channel:
                continue
            selected_right, selected_right_labels = (
                hamiltonian_cache._select_channel(
                    hamiltonian_right,
                    hamiltonian_cache.frontiers[right_site + 1],
                    right_operator[1],
                    right_channel,
                )
            )
            if selected_right is None:
                continue
            result += _complete_contraction(
                [
                    selected_left,
                    first_operator,
                    second_operator,
                    selected_right,
                    left_tensor,
                    right_tensor,
                    preselected.conj(),
                ],
                [
                    tuple(selected_left_labels),
                    first_physical,
                    second_physical,
                    tuple(selected_right_labels),
                    left_ket,
                    right_ket,
                    candidate_labels,
                ],
                output,
                output_shape,
            )
    return result


def _streamed_identity_final_tensor(
    metric_cache,
    metric_left,
    metric_right,
    layout,
    left_tensor,
    right_tensor,
    preselected,
    direction,
):
    """Contract a restricted pair overlap without forming the pair tensor."""

    left_site = layout.left_site
    right_site = left_site + 1
    left_bra, left_ket = metric_cache._group_labels(left_site)
    right_bra, right_ket = metric_cache._group_labels(right_site)
    candidate_label = -10_000_001
    if direction == "rl":
        candidate_labels = left_bra[:-1] + (candidate_label,)
        output = (candidate_label,) + right_bra[1:]
        output_shape = (preselected.shape[-1],) + right_tensor.shape[1:]
    elif direction == "lr":
        candidate_labels = (candidate_label,) + right_bra[1:]
        output = left_bra[:-1] + (candidate_label,)
        output_shape = left_tensor.shape[:-1] + (preselected.shape[0],)
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")
    return _complete_contraction(
        [
            metric_left,
            metric_right,
            left_tensor,
            right_tensor,
            preselected.conj(),
        ],
        [
            metric_cache.frontiers[left_site],
            metric_cache.frontiers[right_site + 1],
            left_ket,
            right_ket,
            candidate_labels,
        ],
        output,
        output_shape,
    )


@_stable_floating_point
def streamed_shrewd_cbe_selection(
    hamiltonian_cache,
    hamiltonian_left,
    hamiltonian_right,
    layout,
    left_tensor,
    right_tensor,
    *,
    expansion_dimension,
    preselection_dimension,
    direction,
    tolerance=1.0e-12,
    metric_cache=None,
    metric_left=None,
    metric_right=None,
    energy=None,
    metric_tolerance=1.0e-12,
):
    """Select CBE factors through one-site-sized streamed contractions.

    Supplying the metric environments changes the restricted final stage from
    the legacy Hamiltonian covector to the raised physical residual
    ``N^+ (H - E N) psi`` with its current one-site tangent removed in ``N``.
    """

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    expansion_dimension = int(expansion_dimension)
    preselection_dimension = int(preselection_dimension)
    tolerance = float(tolerance)
    metric_tolerance = float(metric_tolerance)
    if expansion_dimension <= 0 or preselection_dimension <= 0:
        raise ValueError("selection dimensions must be positive.")
    if preselection_dimension < expansion_dimension:
        raise ValueError(
            "preselection_dimension must be at least expansion_dimension."
        )
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("selection and metric tolerances must be positive.")
    residual_inputs = (metric_cache, metric_left, metric_right, energy)
    if any(value is not None for value in residual_inputs) and not all(
        value is not None for value in residual_inputs
    ):
        raise ValueError(
            "metric_cache, metric_left, metric_right, and energy must be "
            "provided together."
        )
    direction = str(direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'.")

    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)

    if direction == "rl":
        opposite_half = _streamed_shrewd_preselection_tensor(
            hamiltonian_cache,
            hamiltonian_left,
            hamiltonian_right,
            layout,
            left_tensor,
            right_tensor,
            "lr",
        )
        right_parent = int(np.prod(right_tensor.shape[1:]))
        opposite_matrix = _project_out_rows(
            opposite_half.reshape(-1, right_parent),
            right_tensor.reshape(right_tensor.shape[0], right_parent),
            tolerance,
        )
        opposite_left, opposite_values, _opposite_right = np.linalg.svd(
            opposite_matrix, full_matrices=False
        )
        opposite_rank = _retained_svd_rank(
            opposite_values, opposite_values.size, tolerance
        )
        if opposite_rank == 0:
            return _zero_streamed_selection(
                left_tensor,
                right_tensor,
                expansion_dimension,
                opposite_half.size,
            )
        weighted_half = (
            opposite_left[:, :opposite_rank]
            * opposite_values[:opposite_rank]
        ).reshape(
            left_tensor.shape[-1],
            opposite_half.shape[1],
            opposite_rank,
        )
        preselection_tensor = (
            _streamed_shrewd_weighted_preselection_tensor(
                hamiltonian_cache,
                hamiltonian_left,
                hamiltonian_right,
                layout,
                left_tensor,
                right_tensor,
                weighted_half,
                direction,
            )
        )
        left_parent = int(np.prod(left_tensor.shape[:-1]))
        preselection_matrix = _project_out_columns(
            preselection_tensor.reshape(left_parent, -1),
            left_tensor.reshape(left_parent, left_tensor.shape[-1]),
            tolerance,
        )
        pre_left, pre_values, _pre_right = np.linalg.svd(
            preselection_matrix, full_matrices=False
        )
        pre_rank = _retained_svd_rank(
            pre_values, preselection_dimension, tolerance
        )
        if pre_rank == 0:
            return _zero_streamed_selection(
                left_tensor,
                right_tensor,
                expansion_dimension,
                max(opposite_half.size, preselection_tensor.size),
            )
        preselected = pre_left[:, :pre_rank].reshape(
            left_tensor.shape[:-1] + (pre_rank,)
        )
        if metric_cache is not None:
            final_matrix, metric_missing_norm, restricted_output_size = (
                _metric_projected_streamed_final(
                    hamiltonian_cache,
                    hamiltonian_left,
                    hamiltonian_right,
                    metric_cache,
                    metric_left,
                    metric_right,
                    layout,
                    left_tensor,
                    right_tensor,
                    preselected,
                    direction,
                    energy=energy,
                    metric_tolerance=metric_tolerance,
                )
            )
            final_output_size = restricted_output_size
        else:
            final_tensor = _streamed_shrewd_final_tensor(
                hamiltonian_cache,
                hamiltonian_left,
                hamiltonian_right,
                layout,
                left_tensor,
                right_tensor,
                preselected,
                direction,
            )
            final_matrix = _project_out_rows(
                final_tensor.reshape(pre_rank, -1),
                right_tensor.reshape(right_tensor.shape[0], -1),
                tolerance,
            )
            metric_missing_norm = None
            final_output_size = final_tensor.size
        final_left, final_values, final_right = np.linalg.svd(
            final_matrix, full_matrices=False
        )
        selected_rank = _retained_svd_rank(
            final_values, expansion_dimension, tolerance
        )
        selected_left = (
            preselected.reshape(left_parent, pre_rank)
            @ final_left[:, :selected_rank]
        ).reshape(left_tensor.shape[:-1] + (selected_rank,))
        selected_right = (
            final_values[:selected_rank, None]
            * final_right[:selected_rank]
        ).reshape((selected_rank,) + right_tensor.shape[1:])
    else:
        opposite_half = _streamed_shrewd_preselection_tensor(
            hamiltonian_cache,
            hamiltonian_left,
            hamiltonian_right,
            layout,
            left_tensor,
            right_tensor,
            "rl",
        )
        left_parent = int(np.prod(left_tensor.shape[:-1]))
        opposite_matrix = _project_out_columns(
            opposite_half.reshape(left_parent, -1),
            left_tensor.reshape(left_parent, left_tensor.shape[-1]),
            tolerance,
        )
        _opposite_left, opposite_values, opposite_right = np.linalg.svd(
            opposite_matrix, full_matrices=False
        )
        opposite_rank = _retained_svd_rank(
            opposite_values, opposite_values.size, tolerance
        )
        if opposite_rank == 0:
            return _zero_streamed_selection(
                left_tensor,
                right_tensor,
                expansion_dimension,
                opposite_half.size,
            )
        weighted_half = (
            opposite_values[:opposite_rank, None]
            * opposite_right[:opposite_rank]
        ).reshape(
            opposite_rank,
            right_tensor.shape[0],
            opposite_half.shape[-1],
        )
        preselection_tensor = (
            _streamed_shrewd_weighted_preselection_tensor(
                hamiltonian_cache,
                hamiltonian_left,
                hamiltonian_right,
                layout,
                left_tensor,
                right_tensor,
                weighted_half,
                direction,
            )
        )
        right_parent = int(np.prod(right_tensor.shape[1:]))
        preselection_matrix = _project_out_rows(
            preselection_tensor.reshape(-1, right_parent),
            right_tensor.reshape(right_tensor.shape[0], right_parent),
            tolerance,
        )
        _pre_left, pre_values, pre_right = np.linalg.svd(
            preselection_matrix, full_matrices=False
        )
        pre_rank = _retained_svd_rank(
            pre_values, preselection_dimension, tolerance
        )
        if pre_rank == 0:
            return _zero_streamed_selection(
                left_tensor,
                right_tensor,
                expansion_dimension,
                max(opposite_half.size, preselection_tensor.size),
            )
        preselected = pre_right[:pre_rank].reshape(
            (pre_rank,) + right_tensor.shape[1:]
        )
        if metric_cache is not None:
            final_matrix, metric_missing_norm, restricted_output_size = (
                _metric_projected_streamed_final(
                    hamiltonian_cache,
                    hamiltonian_left,
                    hamiltonian_right,
                    metric_cache,
                    metric_left,
                    metric_right,
                    layout,
                    left_tensor,
                    right_tensor,
                    preselected,
                    direction,
                    energy=energy,
                    metric_tolerance=metric_tolerance,
                )
            )
            final_output_size = restricted_output_size
        else:
            final_tensor = _streamed_shrewd_final_tensor(
                hamiltonian_cache,
                hamiltonian_left,
                hamiltonian_right,
                layout,
                left_tensor,
                right_tensor,
                preselected,
                direction,
            )
            final_matrix = _project_out_columns(
                final_tensor.reshape(-1, pre_rank),
                left_tensor.reshape(-1, left_tensor.shape[-1]),
                tolerance,
            )
            metric_missing_norm = None
            final_output_size = final_tensor.size
        final_left, final_values, final_right = np.linalg.svd(
            final_matrix, full_matrices=False
        )
        selected_rank = _retained_svd_rank(
            final_values, expansion_dimension, tolerance
        )
        selected_left = (
            final_left[:, :selected_rank]
            * final_values[:selected_rank]
        ).reshape(left_tensor.shape[:-1] + (selected_rank,))
        selected_right = (
            final_right[:selected_rank]
            @ preselected.reshape(pre_rank, right_parent)
        ).reshape((selected_rank,) + right_tensor.shape[1:])

    left_direction = _pad_left_direction(
        selected_left, left_tensor.shape, expansion_dimension
    )
    right_direction = _pad_right_direction(
        selected_right, right_tensor.shape, expansion_dimension
    )
    total_weight = float(np.sum(final_values**2))
    selected_weight = float(np.sum(final_values[:selected_rank] ** 2))
    captured_weight = (
        selected_weight / total_weight
        if total_weight > np.finfo(float).tiny
        else 0.0
    )
    preselection_loss = float(np.sum(pre_values[pre_rank:] ** 2))
    return CBESelection(
        left_direction=left_direction,
        right_direction=right_direction,
        loss=float(np.sum(final_values[selected_rank:] ** 2)),
        captured_weight=float(np.clip(captured_weight, 0.0, 1.0)),
        sector_ranks=(selected_rank,),
        refinement_iterations=0,
        selector="shrewd",
        preselection_dimension=pre_rank,
        preselection_loss=preselection_loss,
        missing_norm=(
            float(np.sqrt(total_weight))
            if metric_missing_norm is None
            else metric_missing_norm
        ),
        pair_action_count=0,
        pair_metric_count=0,
        merged_pair_count=0,
        preselection_output_size=max(
            opposite_half.size, preselection_tensor.size
        ),
        final_output_size=final_output_size,
    )


def _dense_metric(metric):
    if hasattr(metric, "to_dense"):
        return np.asarray(metric.to_dense())
    return np.asarray(metric)


def _supported_eigendecomposition(metric, tolerance):
    dense = _dense_metric(metric)
    dense = 0.5 * (dense + dense.conj().T)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("metric must be a square matrix.")
    values, vectors = np.linalg.eigh(dense)
    scale = max(float(values[-1]), 0.0) if values.size else 0.0
    cutoff = max(
        float(tolerance),
        np.finfo(float).eps * dense.shape[0],
    ) * scale
    retained = values > cutoff
    if not np.any(retained):
        raise ValueError("the pair LETTA overlap metric has zero rank.")
    return dense, values[retained], vectors[:, retained]


@_stable_floating_point
def _metric_support(metric, tolerance):
    dense, values, vectors = _supported_eigendecomposition(metric, tolerance)
    support = vectors @ vectors.conj().T
    if not np.all(np.isfinite(support)):
        raise FloatingPointError("the pair-metric support projector is nonfinite.")
    return dense, values, vectors, support, int(values.size)


@_stable_floating_point
def _hermitian_pseudoinverse(matrix, tolerance):
    matrix = np.asarray(matrix)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    if hermitian.size == 0:
        return np.zeros_like(hermitian), 0
    values, vectors = np.linalg.eigh(hermitian)
    scale = max(float(values[-1]), 0.0)
    cutoff = max(
        float(tolerance),
        np.finfo(float).eps * hermitian.shape[0],
    ) * scale
    retained = values > cutoff
    if not np.any(retained):
        return np.zeros_like(hermitian), 0
    pseudoinverse = (
        vectors[:, retained] / values[retained][None, :]
    ) @ vectors[:, retained].conj().T
    if not np.all(np.isfinite(pseudoinverse)):
        raise FloatingPointError("a tangent pseudoinverse is nonfinite.")
    return pseudoinverse, int(np.count_nonzero(retained))


@_stable_floating_point
def metric_orthogonal_complement(
    vector,
    jacobian,
    metric,
    *,
    tolerance=1.0e-12,
):
    """Project a vector out of a Jacobian range in a PSD metric.

    Null-metric coordinates are removed before the tangent projection.  This
    makes the result invariant to unsupported LETTA parameter directions.
    """

    vector = np.asarray(vector)
    jacobian = np.asarray(jacobian)
    dense, _values, _vectors, support, metric_rank = _metric_support(
        metric, tolerance
    )
    if vector.shape != (dense.shape[0],):
        raise ValueError("vector and metric dimensions do not match.")
    if jacobian.ndim != 2 or jacobian.shape[0] != dense.shape[0]:
        raise ValueError("jacobian and metric dimensions do not match.")

    supported = support @ vector
    gram = jacobian.conj().T @ dense @ jacobian
    gram_pseudoinverse, tangent_rank = _hermitian_pseudoinverse(
        gram, tolerance
    )
    coefficients = (
        gram_pseudoinverse @ jacobian.conj().T @ dense @ supported
    )
    complement = support @ (supported - jacobian @ coefficients)
    metric_norm_squared = float(
        max(0.0, np.real(np.vdot(complement, dense @ complement)))
    )
    tangent_overlap = jacobian.conj().T @ dense @ complement
    if (
        not np.all(np.isfinite(complement))
        or not np.isfinite(metric_norm_squared)
        or not np.all(np.isfinite(tangent_overlap))
    ):
        raise FloatingPointError("the metric tangent complement is nonfinite.")
    return MetricOrthogonalComplement(
        vector=complement,
        metric_rank=metric_rank,
        tangent_rank=tangent_rank,
        norm=float(np.sqrt(metric_norm_squared)),
        tangent_overlap_norm=float(np.linalg.norm(tangent_overlap)),
    )


def _merge_jacobian(layout, left_tensor, right_tensor):
    dtype = np.result_type(left_tensor, right_tensor)
    columns = []
    for position in range(left_tensor.size):
        basis = np.zeros(left_tensor.size, dtype=dtype)
        basis[position] = 1.0
        columns.append(
            layout.merge(basis.reshape(left_tensor.shape), right_tensor).reshape(-1)
        )
    for position in range(right_tensor.size):
        basis = np.zeros(right_tensor.size, dtype=dtype)
        basis[position] = 1.0
        columns.append(
            layout.merge(left_tensor, basis.reshape(right_tensor.shape)).reshape(-1)
        )
    if not columns:
        return np.zeros((int(np.prod(layout.merged_shape)), 0), dtype=dtype)
    return np.column_stack(columns)


class _BlockMetricSpectralOperator:
    """Supported block-metric functions without assembling the full matrix."""

    def __init__(self, metric, tolerance):
        if not all(
            hasattr(metric, attribute)
            for attribute in ("blocks", "indices", "size")
        ):
            raise TypeError(
                "the shrewd selector requires a block-diagonal pair metric."
            )
        self.metric = metric
        self.size = int(metric.size)
        decompositions = []
        scale = 0.0
        for block, indices in zip(metric.blocks, metric.indices):
            block = np.asarray(block)
            indices = np.asarray(indices, dtype=int)
            hermitian = 0.5 * (block + block.conj().T)
            values, vectors = np.linalg.eigh(hermitian)
            decompositions.append((indices, values, vectors))
            if values.size:
                scale = max(scale, float(values[-1]))
        if scale <= 0.0:
            raise ValueError("the pair LETTA overlap metric has zero rank.")
        cutoff = max(
            float(tolerance),
            np.finfo(float).eps * self.size,
        ) * scale
        self.decompositions = tuple(
            (
                indices,
                values[values > cutoff],
                vectors[:, values > cutoff],
            )
            for indices, values, vectors in decompositions
        )
        self.rank = int(
            sum(values.size for _indices, values, _vectors in self.decompositions)
        )
        if self.rank == 0:
            raise ValueError("the pair LETTA overlap metric has zero rank.")

    def _apply(self, vector, function):
        vector = np.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError("the pair-metric operand has an incompatible shape.")
        result = np.zeros(
            self.size,
            dtype=np.result_type(vector, self.metric.dtype),
        )
        for indices, values, vectors in self.decompositions:
            if values.size:
                coefficients = vectors.conj().T @ vector[indices]
                result[indices] = vectors @ (function(values) * coefficients)
        return result

    def square_root(self, vector):
        return self._apply(vector, np.sqrt)

    def pseudoinverse(self, vector):
        return self._apply(vector, lambda values: 1.0 / values)

    def support(self, vector):
        return self._apply(vector, lambda values: np.ones_like(values))


def _metric_projected_restricted_residual(
    covector,
    metric,
    tangent_indices,
    *,
    metric_tolerance,
):
    """Raise a residual covector and remove a coordinate tangent in its metric."""

    covector = np.asarray(covector).reshape(-1)
    tangent_indices = np.asarray(tangent_indices, dtype=int)
    if covector.shape != (metric.size,):
        raise ValueError("the residual and restricted metric dimensions differ.")
    if (
        tangent_indices.ndim != 1
        or np.any(tangent_indices < 0)
        or np.any(tangent_indices >= metric.size)
        or np.unique(tangent_indices).size != tangent_indices.size
    ):
        raise ValueError("tangent_indices are invalid.")
    spectral = _BlockMetricSpectralOperator(metric, metric_tolerance)
    raised = spectral.pseudoinverse(covector)
    if tangent_indices.size:
        tangent_metric = metric.restrict(tangent_indices)
        tangent_spectral = _BlockMetricSpectralOperator(
            tangent_metric, metric_tolerance
        )
        supported_covector = metric @ raised
        coefficients = tangent_spectral.pseudoinverse(
            supported_covector[tangent_indices]
        )
        raised[tangent_indices] -= coefficients
    metric_norm_squared = float(
        max(0.0, np.real(np.vdot(raised, metric @ raised)))
    )
    if not np.all(np.isfinite(raised)) or not np.isfinite(metric_norm_squared):
        raise FloatingPointError("the metric-projected residual is nonfinite.")
    return raised, float(np.sqrt(metric_norm_squared))


def _extend_identity_environment_with_tensor(
    metric_cache,
    environment,
    site,
    tensor,
    direction,
):
    """Extend an overlap environment with a supplied one-site factor."""

    bra_labels, ket_labels = metric_cache._group_labels(site)
    if direction == "lr":
        input_labels = metric_cache.frontiers[site]
        output_labels = metric_cache.frontiers[site + 1]
    elif direction == "rl":
        input_labels = metric_cache.frontiers[site + 1]
        output_labels = metric_cache.frontiers[site]
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")
    return _complete_contraction(
        [environment, tensor.conj(), tensor],
        [input_labels, bra_labels, ket_labels],
        output_labels,
        tuple(
            (
                tensor.shape[0]
                if label in (
                    metric_cache.bra_virtual[site],
                    metric_cache.ket_virtual[site],
                )
                else tensor.shape[-1]
                if label in (
                    metric_cache.bra_virtual[site + 1],
                    metric_cache.ket_virtual[site + 1],
                )
                else metric_cache.label_dimensions[label]
            )
            for label in output_labels
        ),
    )


def _effective_identity_metric_for_shape(
    metric_cache,
    left,
    right,
    site,
    tensor_shape,
):
    """Build one active-site overlap blocks for a temporary tensor shape."""

    tensor_shape = tuple(int(dimension) for dimension in tensor_shape)
    physical_shape = tensor_shape[1:-1]
    expected_physical = (metric_cache.state.physical_dim,) * len(
        metric_cache.state.site_neighborhood(site)
    )
    if physical_shape != expected_physical:
        raise ValueError("the temporary tensor has incompatible physical axes.")
    physical = tuple(
        metric_cache.physical[index]
        for index in metric_cache.state.site_neighborhood(site)
    )
    output = (
        (metric_cache.bra_virtual[site],)
        + physical
        + (
            metric_cache.bra_virtual[site + 1],
            metric_cache.ket_virtual[site],
            metric_cache.ket_virtual[site + 1],
        )
    )
    reduced_shape = (
        (tensor_shape[0],)
        + physical_shape
        + (tensor_shape[-1], tensor_shape[0], tensor_shape[-1])
    )
    reduced = _complete_contraction(
        [left, right],
        [metric_cache.frontiers[site], metric_cache.frontiers[site + 1]],
        output,
        reduced_shape,
    )
    flat_indices = np.arange(np.prod(tensor_shape)).reshape(tensor_shape)
    blocks = []
    indices = []
    for configuration in np.ndindex(*physical_shape):
        source = (slice(None),) + configuration + (slice(None),) * 3
        blocks.append(
            reduced[source].reshape(
                tensor_shape[0] * tensor_shape[-1],
                tensor_shape[0] * tensor_shape[-1],
            )
        )
        indices.append(
            flat_indices[
                (slice(None),) + configuration + (slice(None),)
            ].reshape(-1)
        )
    return BlockDiagonalMetric(int(np.prod(tensor_shape)), blocks, indices)


def _metric_projected_streamed_final(
    hamiltonian_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_cache,
    metric_left,
    metric_right,
    layout,
    left_tensor,
    right_tensor,
    preselected,
    direction,
    *,
    energy,
    metric_tolerance,
):
    """Return the restricted, metric-projected physical pair residual."""

    def residual_covector(bra_factor):
        hamiltonian = _streamed_shrewd_final_tensor(
            hamiltonian_cache,
            hamiltonian_left,
            hamiltonian_right,
            layout,
            left_tensor,
            right_tensor,
            bra_factor,
            direction,
        )
        overlap = _streamed_identity_final_tensor(
            metric_cache,
            metric_left,
            metric_right,
            layout,
            left_tensor,
            right_tensor,
            bra_factor,
            direction,
        )
        return hamiltonian - float(energy) * overlap

    direction = str(direction).lower()
    if direction == "rl":
        tangent_width = left_tensor.shape[-1]
        expanded_left = np.concatenate([left_tensor, preselected], axis=-1)
        local_metric_left = _extend_identity_environment_with_tensor(
            metric_cache,
            metric_left,
            layout.left_site,
            expanded_left,
            "lr",
        )
        covector = np.concatenate(
            [residual_covector(left_tensor), residual_covector(preselected)],
            axis=0,
        )
        metric = _effective_identity_metric_for_shape(
            metric_cache,
            local_metric_left,
            metric_right,
            layout.left_site + 1,
            covector.shape,
        )
        tangent_mask = np.zeros(covector.shape, dtype=bool)
        tangent_mask[:tangent_width] = True
    elif direction == "lr":
        tangent_width = right_tensor.shape[0]
        expanded_right = np.concatenate([right_tensor, preselected], axis=0)
        local_metric_right = _extend_identity_environment_with_tensor(
            metric_cache,
            metric_right,
            layout.left_site + 1,
            expanded_right,
            "rl",
        )
        covector = np.concatenate(
            [residual_covector(right_tensor), residual_covector(preselected)],
            axis=-1,
        )
        metric = _effective_identity_metric_for_shape(
            metric_cache,
            metric_left,
            local_metric_right,
            layout.left_site,
            covector.shape,
        )
        tangent_mask = np.zeros(covector.shape, dtype=bool)
        tangent_mask[..., :tangent_width] = True
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")

    projected, missing_norm = _metric_projected_restricted_residual(
        covector,
        metric,
        np.flatnonzero(tangent_mask.reshape(-1)),
        metric_tolerance=metric_tolerance,
    )
    projected = projected.reshape(covector.shape)
    if direction == "rl":
        final_matrix = projected[tangent_width:].reshape(
            preselected.shape[-1], -1
        )
    else:
        final_matrix = projected[..., tangent_width:].reshape(
            -1, preselected.shape[0]
        )
    return final_matrix, missing_norm, metric.size


def _merge_tangent_products(layout, left_tensor, right_tensor):
    """Return matrix-free products with the pair merge Jacobian and adjoint."""

    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    left_size = left_tensor.size
    parameter_size = left_size + right_tensor.size

    def tangent_product(parameters):
        parameters = np.asarray(parameters)
        if parameters.shape != (parameter_size,):
            raise ValueError("tangent parameters have an incompatible shape.")
        left_delta = parameters[:left_size].reshape(left_tensor.shape)
        right_delta = parameters[left_size:].reshape(right_tensor.shape)
        return (
            layout.merge(left_delta, right_tensor)
            + layout.merge(left_tensor, right_delta)
        ).reshape(-1)

    def tangent_adjoint(vector):
        vector = np.asarray(vector)
        if vector.shape != (int(np.prod(layout.merged_shape)),):
            raise ValueError("the pair tangent operand has an incompatible shape.")
        merged = vector.reshape(layout.merged_shape)
        left_gradient = layout.left_adjoint(merged, right_tensor).reshape(-1)
        right_gradient = layout.right_adjoint(
            left_tensor, merged
        ).reshape(-1)
        return np.concatenate([left_gradient, right_gradient])

    return tangent_product, tangent_adjoint, parameter_size


@_stable_floating_point
def matrix_free_metric_orthogonal_complement(
    vector,
    layout,
    left_tensor,
    right_tensor,
    metric,
    *,
    tolerance=1.0e-10,
    max_iterations=100,
    metric_tolerance=1.0e-12,
    _spectral=None,
):
    """Project out the LETTA pair tangent using operator-only least squares."""

    tolerance = float(tolerance)
    max_iterations = int(max_iterations)
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("projection and metric tolerances must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    spectral = (
        _BlockMetricSpectralOperator(metric, metric_tolerance)
        if _spectral is None
        else _spectral
    )
    tangent_product, tangent_adjoint, parameter_size = (
        _merge_tangent_products(layout, left_tensor, right_tensor)
    )
    pair_size = spectral.size
    dtype = np.result_type(vector, left_tensor, right_tensor, metric.dtype)

    def matvec(parameters):
        return spectral.square_root(tangent_product(parameters))

    def rmatvec(pair_vector):
        return tangent_adjoint(spectral.square_root(pair_vector))

    operator = LinearOperator(
        shape=(pair_size, parameter_size),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=dtype,
    )
    supported = spectral.support(np.asarray(vector))
    right_hand_side = spectral.square_root(supported)
    solution = lsmr(
        operator,
        right_hand_side,
        atol=tolerance,
        btol=tolerance,
        maxiter=max_iterations,
    )
    coefficients = solution[0]
    stop_code = int(solution[1])
    iterations = int(solution[2])
    complement = spectral.support(
        supported - tangent_product(coefficients)
    )
    metric_complement = metric @ complement
    metric_norm_squared = float(
        max(0.0, np.real(np.vdot(complement, metric_complement)))
    )
    tangent_overlap = tangent_adjoint(metric_complement)
    if (
        not np.all(np.isfinite(complement))
        or not np.isfinite(metric_norm_squared)
        or not np.all(np.isfinite(tangent_overlap))
    ):
        raise FloatingPointError("the matrix-free tangent complement is nonfinite.")
    return MetricOrthogonalComplement(
        vector=complement,
        metric_rank=spectral.rank,
        tangent_rank=None,
        norm=float(np.sqrt(metric_norm_squared)),
        tangent_overlap_norm=float(np.linalg.norm(tangent_overlap)),
        iterations=iterations,
        converged=stop_code in {0, 1, 2, 4, 5},
    )


@_stable_floating_point
def matrix_free_missing_pair_direction(
    layout,
    left_tensor,
    right_tensor,
    pair_action,
    pair_metric,
    *,
    metric_tolerance=1.0e-12,
    projection_tolerance=1.0e-10,
    projection_max_iterations=100,
):
    """Return a metric-aware missing direction without dense pair geometry."""

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    theta = layout.merge(left_tensor, right_tensor).reshape(-1)
    spectral = _BlockMetricSpectralOperator(pair_metric, metric_tolerance)
    applied = np.asarray(pair_action(theta))
    if applied.shape != theta.shape:
        raise ValueError("pair_action returned an incompatible vector.")
    metric_theta = pair_metric @ theta
    denominator = np.vdot(theta, metric_theta)
    if np.real(denominator) <= np.finfo(float).tiny:
        raise ValueError("the represented LETTA pair has zero metric norm.")
    energy = float(np.real(np.vdot(theta, applied) / denominator))
    covector = applied - energy * metric_theta
    raised_residual = spectral.pseudoinverse(covector)
    if not np.isfinite(energy) or not np.all(np.isfinite(raised_residual)):
        raise FloatingPointError("the supported pair residual is nonfinite.")
    complement = matrix_free_metric_orthogonal_complement(
        raised_residual,
        layout,
        left_tensor,
        right_tensor,
        pair_metric,
        tolerance=projection_tolerance,
        max_iterations=projection_max_iterations,
        metric_tolerance=metric_tolerance,
        _spectral=spectral,
    )
    return CBEMissingDirection(
        vector=complement.vector,
        energy=energy,
        metric_rank=complement.metric_rank,
        tangent_rank=None,
        missing_norm=complement.norm,
        tangent_overlap_norm=complement.tangent_overlap_norm,
        selector="shrewd",
        projection_iterations=complement.iterations,
        projection_converged=complement.converged,
        pair_action_count=1,
        materialized_pair_metric=False,
        materialized_tangent_jacobian=False,
    )


@_stable_floating_point
def exact_missing_pair_direction(
    layout,
    left_tensor,
    right_tensor,
    pair_action,
    pair_metric,
    *,
    metric_tolerance=1.0e-12,
):
    """Return the exact metric-supported pair residual outside one-site tangents."""

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    theta = layout.merge(left_tensor, right_tensor).reshape(-1)
    dense, metric_values, metric_vectors, _support, _rank = _metric_support(
        pair_metric, metric_tolerance
    )
    applied = np.asarray(pair_action(theta))
    denominator = np.vdot(theta, dense @ theta)
    if np.real(denominator) <= np.finfo(float).tiny:
        raise ValueError("the represented LETTA pair has zero metric norm.")
    energy = float(np.real(np.vdot(theta, applied) / denominator))
    covector = applied - energy * (dense @ theta)
    metric_scale = float(metric_values[-1])
    scaled_values = metric_values / metric_scale
    scaled_covector = covector / metric_scale
    raised_residual = metric_vectors @ (
        (metric_vectors.conj().T @ scaled_covector) / scaled_values
    )
    if not np.isfinite(energy) or not np.all(np.isfinite(raised_residual)):
        raise FloatingPointError("the supported pair residual is nonfinite.")
    jacobian = _merge_jacobian(layout, left_tensor, right_tensor)
    complement = metric_orthogonal_complement(
        raised_residual,
        jacobian,
        dense,
        tolerance=metric_tolerance,
    )
    return CBEMissingDirection(
        vector=complement.vector,
        energy=energy,
        metric_rank=complement.metric_rank,
        tangent_rank=complement.tangent_rank,
        missing_norm=complement.norm,
        tangent_overlap_norm=complement.tangent_overlap_norm,
    )


def _metric_loss(target, approximation, metric):
    difference = np.asarray(target).reshape(-1) - np.asarray(
        approximation
    ).reshape(-1)
    return float(max(0.0, np.real(np.vdot(difference, metric @ difference))))


def select_cbe_directions(
    missing_direction,
    layout,
    pair_metric,
    *,
    expansion_dimension,
    direction,
    tolerance=1.0e-10,
    max_iterations=4,
    metric_tolerance=1.0e-12,
):
    """Select a controlled low-rank factorization of a missing pair direction."""

    if not isinstance(missing_direction, CBEMissingDirection):
        raise TypeError("missing_direction must be a CBEMissingDirection.")
    expansion_dimension = int(expansion_dimension)
    max_iterations = int(max_iterations)
    if expansion_dimension <= 0:
        raise ValueError("expansion_dimension must be positive.")
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("selection and metric tolerances must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    target = missing_direction.vector.reshape(layout.merged_shape)
    split = conditional_svd_split(
        target,
        layout,
        max_bond_dim=expansion_dimension,
        direction=direction,
    )
    refinement = metric_als_refine(
        target,
        layout,
        split,
        pair_metric,
        tolerance=tolerance,
        max_iterations=max_iterations,
        metric_tolerance=metric_tolerance,
    )
    approximation = layout.merge(
        refinement.left_tensor, refinement.right_tensor
    )
    loss = _metric_loss(target, approximation, pair_metric)
    norm_squared = missing_direction.missing_norm**2
    if norm_squared <= np.finfo(float).tiny:
        captured_weight = 0.0
    else:
        captured_weight = float(np.clip(1.0 - loss / norm_squared, 0.0, 1.0))
    return CBESelection(
        left_direction=refinement.left_tensor,
        right_direction=refinement.right_tensor,
        loss=loss,
        captured_weight=captured_weight,
        sector_ranks=split.sector_ranks,
        refinement_iterations=refinement.iterations,
        selector="exact",
    )


def select_shrewd_cbe_directions(
    missing_direction,
    layout,
    pair_metric,
    *,
    expansion_dimension,
    preselection_dimension,
    direction,
    tolerance=1.0e-10,
    max_iterations=4,
    metric_tolerance=1.0e-12,
):
    """Select expansion factors through preselection and metric final selection.

    The first conditional SVD limits the candidate complement before the
    expansion-rank split.  The final ALS closes the full LETTA metric and is
    therefore the optimization-relevant selection step.
    """

    if not isinstance(missing_direction, CBEMissingDirection):
        raise TypeError("missing_direction must be a CBEMissingDirection.")
    expansion_dimension = int(expansion_dimension)
    preselection_dimension = int(preselection_dimension)
    max_iterations = int(max_iterations)
    if expansion_dimension <= 0:
        raise ValueError("expansion_dimension must be positive.")
    if preselection_dimension < expansion_dimension:
        raise ValueError(
            "preselection_dimension must be at least expansion_dimension."
        )
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("selection and metric tolerances must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    target = missing_direction.vector.reshape(layout.merged_shape)
    preselection = conditional_svd_split(
        target,
        layout,
        max_bond_dim=preselection_dimension,
        direction=direction,
    )
    preselected_target = layout.merge(
        preselection.left_tensor, preselection.right_tensor
    )
    preselection_loss = _metric_loss(
        target, preselected_target, pair_metric
    )
    final_split = conditional_svd_split(
        preselected_target,
        layout,
        max_bond_dim=expansion_dimension,
        direction=direction,
    )
    refinement = metric_als_refine(
        target,
        layout,
        final_split,
        pair_metric,
        tolerance=tolerance,
        max_iterations=max_iterations,
        metric_tolerance=metric_tolerance,
    )
    approximation = layout.merge(
        refinement.left_tensor, refinement.right_tensor
    )
    loss = _metric_loss(target, approximation, pair_metric)
    norm_squared = missing_direction.missing_norm**2
    if norm_squared <= np.finfo(float).tiny:
        captured_weight = 0.0
    else:
        captured_weight = float(
            np.clip(1.0 - loss / norm_squared, 0.0, 1.0)
        )
    return CBESelection(
        left_direction=refinement.left_tensor,
        right_direction=refinement.right_tensor,
        loss=loss,
        captured_weight=captured_weight,
        sector_ranks=final_split.sector_ranks,
        refinement_iterations=refinement.iterations,
        selector="shrewd",
        preselection_dimension=int(max(preselection.sector_ranks, default=0)),
        preselection_loss=preselection_loss,
    )


def metric_trim_pair(
    target,
    layout,
    pair_metric,
    *,
    bond_dimension,
    direction,
    tolerance=1.0e-10,
    max_iterations=4,
    metric_tolerance=1.0e-12,
):
    """Trim an expanded pair in the full LETTA norm metric."""

    target = np.asarray(target)
    split = conditional_svd_split(
        target,
        layout,
        max_bond_dim=int(bond_dimension),
        direction=direction,
    )
    refinement = metric_als_refine(
        target,
        layout,
        split,
        pair_metric,
        tolerance=tolerance,
        max_iterations=int(max_iterations),
        metric_tolerance=metric_tolerance,
    )
    left_tensor = refinement.left_tensor.copy()
    right_tensor = refinement.right_tensor.copy()
    reconstructed = layout.merge(left_tensor, right_tensor)
    loss = _metric_loss(target, reconstructed, pair_metric)
    vector = reconstructed.reshape(-1)
    norm_squared = float(np.real(np.vdot(vector, pair_metric @ vector)))
    norm = float(np.sqrt(max(0.0, norm_squared)))
    if norm > np.finfo(float).tiny:
        if direction == "lr":
            right_tensor /= norm
        elif direction == "rl":
            left_tensor /= norm
        else:
            raise ValueError("direction must be 'lr' or 'rl'.")
    return CBETrim(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        loss=loss,
        iterations=refinement.iterations,
        norm=norm,
    )


def embed_cbe_pair(
    left_tensor,
    right_tensor,
    left_direction,
    right_direction,
    *,
    direction,
):
    """Enlarge a bond without changing the represented pair tensor."""

    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    left_direction = np.asarray(left_direction)
    right_direction = np.asarray(right_direction)
    if left_tensor.shape[-1] != right_tensor.shape[0]:
        raise ValueError("the original pair has incompatible virtual dimensions.")
    if left_direction.shape[:-1] != left_tensor.shape[:-1]:
        raise ValueError("left expansion direction has incompatible outer axes.")
    if right_direction.shape[1:] != right_tensor.shape[1:]:
        raise ValueError("right expansion direction has incompatible outer axes.")
    expansion_dimension = left_direction.shape[-1]
    if expansion_dimension <= 0 or right_direction.shape[0] != expansion_dimension:
        raise ValueError("expansion directions must have the same positive width.")
    direction = str(direction).lower()
    dtype = np.result_type(
        left_tensor, right_tensor, left_direction, right_direction
    )
    if direction == "lr":
        expanded_left = np.concatenate(
            [
                left_tensor.astype(dtype, copy=False),
                np.zeros(left_direction.shape, dtype=dtype),
            ],
            axis=-1,
        )
        expanded_right = np.concatenate(
            [
                right_tensor.astype(dtype, copy=False),
                right_direction.astype(dtype, copy=False),
            ],
            axis=0,
        )
    elif direction == "rl":
        expanded_left = np.concatenate(
            [
                left_tensor.astype(dtype, copy=False),
                left_direction.astype(dtype, copy=False),
            ],
            axis=-1,
        )
        expanded_right = np.concatenate(
            [
                right_tensor.astype(dtype, copy=False),
                np.zeros(right_direction.shape, dtype=dtype),
            ],
            axis=0,
        )
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")
    return expanded_left, expanded_right


def _pair_rayleigh(vector, action, metric):
    vector = np.asarray(vector)
    denominator = np.vdot(vector, metric @ vector)
    if np.real(denominator) <= np.finfo(float).tiny:
        return np.inf
    return float(np.real(np.vdot(vector, action(vector)) / denominator))


def _cbe_baseline_allowance(old_energy, baseline_energy, options):
    """Return the permitted exploratory loss relative to one-site descent."""

    baseline_gain = max(float(old_energy) - float(baseline_energy), 0.0)
    return options.cbe_baseline_guard_fraction * baseline_gain


def _ordinary_bond_fallback(
    state,
    layout,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    direction,
    options,
):
    from .solver import _update_from_cached_environments

    if direction == "lr":
        active_site = layout.left_site
        local_hamiltonian_right = hamiltonian_cache.extend_right(
            hamiltonian_right, layout.left_site + 1
        )
        local_metric_right = metric_cache.extend_right(
            metric_right, layout.left_site + 1
        )
        return _update_from_cached_environments(
            state,
            active_site,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            local_hamiltonian_right,
            metric_left,
            local_metric_right,
            options,
        )
    active_site = layout.left_site + 1
    local_hamiltonian_left = hamiltonian_cache.extend_left(
        hamiltonian_left, layout.left_site
    )
    local_metric_left = metric_cache.extend_left(
        metric_left, layout.left_site
    )
    return _update_from_cached_environments(
        state,
        active_site,
        hamiltonian_cache,
        metric_cache,
        local_hamiltonian_left,
        hamiltonian_right,
        local_metric_left,
        metric_right,
        options,
    )


def _close_streamed_bond_environment(cache, left, right, layout):
    environment = cache.extend_left(left, layout.left_site)
    environment = cache.extend_left(environment, layout.left_site + 1)
    frontier = cache.frontiers[layout.left_site + 2]
    return _contract_operands(
        [environment, right],
        [frontier, frontier],
        (),
    )


def _streamed_bond_energy(
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    layout,
):
    numerator = _close_streamed_bond_environment(
        hamiltonian_cache,
        hamiltonian_left,
        hamiltonian_right,
        layout,
    )
    denominator = _close_streamed_bond_environment(
        metric_cache,
        metric_left,
        metric_right,
        layout,
    )
    denominator = float(np.real(denominator))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("the streamed LETTA bond has zero metric norm.")
    energy = float(np.real(numerator / denominator))
    if not np.isfinite(energy):
        raise FloatingPointError("the streamed LETTA bond energy is nonfinite.")
    return energy, float(np.sqrt(denominator))


def _directional_one_site_trim(
    left_tensor,
    right_tensor,
    *,
    bond_dimension,
    direction,
):
    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    bond_dimension = int(bond_dimension)
    direction = str(direction).lower()
    if bond_dimension <= 0:
        raise ValueError("bond_dimension must be positive.")
    if left_tensor.shape[-1] != right_tensor.shape[0]:
        raise ValueError("expanded LETTA tensors have incompatible bonds.")

    if direction == "lr":
        matrix = left_tensor.reshape(-1, left_tensor.shape[-1])
        vectors, values, adjoint = np.linalg.svd(matrix, full_matrices=False)
        keep = min(bond_dimension, values.size)
        trimmed_left_matrix = np.zeros(
            (matrix.shape[0], bond_dimension), dtype=left_tensor.dtype
        )
        transfer = np.zeros(
            (bond_dimension, matrix.shape[1]),
            dtype=np.result_type(left_tensor, right_tensor),
        )
        trimmed_left_matrix[:, :keep] = vectors[:, :keep]
        transfer[:keep] = values[:keep, None] * adjoint[:keep]
        trimmed_left = trimmed_left_matrix.reshape(
            left_tensor.shape[:-1] + (bond_dimension,)
        )
        trimmed_right = np.tensordot(
            transfer, right_tensor, axes=([1], [0])
        )
    elif direction == "rl":
        matrix = right_tensor.reshape(right_tensor.shape[0], -1)
        vectors, values, adjoint = np.linalg.svd(matrix, full_matrices=False)
        keep = min(bond_dimension, values.size)
        transfer = np.zeros(
            (matrix.shape[0], bond_dimension),
            dtype=np.result_type(left_tensor, right_tensor),
        )
        trimmed_right_matrix = np.zeros(
            (bond_dimension, matrix.shape[1]), dtype=right_tensor.dtype
        )
        transfer[:, :keep] = vectors[:, :keep] * values[:keep]
        trimmed_right_matrix[:keep] = adjoint[:keep]
        trimmed_left = np.tensordot(
            left_tensor, transfer, axes=([-1], [0])
        )
        trimmed_right = trimmed_right_matrix.reshape(
            (bond_dimension,) + right_tensor.shape[1:]
        )
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")

    discarded_weight = float(np.sum(values[keep:] ** 2))
    return CBETrim(
        left_tensor=trimmed_left,
        right_tensor=trimmed_right,
        loss=discarded_weight,
        iterations=0,
        norm=float(np.linalg.norm(values[:keep])),
    )


def _one_site_factorization_loss(target, left, right, metric):
    difference = np.asarray(target) - np.asarray(left) @ np.asarray(right)
    vector = difference.reshape(-1)
    return float(max(0.0, np.real(np.vdot(vector, metric @ vector))))


def _metric_low_rank_factorization(
    target,
    metric,
    rank,
    *,
    tolerance,
    max_iterations,
    metric_tolerance,
):
    target = np.asarray(target)
    rows, columns = target.shape
    vectors, values, adjoint = np.linalg.svd(target, full_matrices=False)
    keep = min(int(rank), values.size)
    left = np.zeros((rows, rank), dtype=target.dtype)
    right = np.zeros((rank, columns), dtype=target.dtype)
    left[:, :keep] = vectors[:, :keep]
    right[:keep] = values[:keep, None] * adjoint[:keep]
    square_root = _MetricSquareRoot(metric, metric_tolerance)
    if square_root.size != target.size:
        raise ValueError("one-site metric and trim target dimensions differ.")
    weighted_target = square_root.apply(target.reshape(-1))

    def optimize_left(current_left, current_right):
        def forward(vector):
            candidate = vector.reshape(rows, rank) @ current_right
            return square_root.apply(candidate.reshape(-1))

        def adjoint_action(vector):
            weighted = square_root.apply(vector).reshape(rows, columns)
            return (weighted @ current_right.conj().T).reshape(-1)

        operator = LinearOperator(
            (target.size, rows * rank),
            matvec=forward,
            rmatvec=adjoint_action,
            dtype=np.result_type(target, metric.dtype),
        )
        solution = lsmr(
            operator,
            weighted_target,
            atol=tolerance,
            btol=tolerance,
            maxiter=40,
            x0=current_left.reshape(-1),
        )[0]
        return solution.reshape(rows, rank)

    def optimize_right(current_left, current_right):
        def forward(vector):
            candidate = current_left @ vector.reshape(rank, columns)
            return square_root.apply(candidate.reshape(-1))

        def adjoint_action(vector):
            weighted = square_root.apply(vector).reshape(rows, columns)
            return (current_left.conj().T @ weighted).reshape(-1)

        operator = LinearOperator(
            (target.size, rank * columns),
            matvec=forward,
            rmatvec=adjoint_action,
            dtype=np.result_type(target, metric.dtype),
        )
        solution = lsmr(
            operator,
            weighted_target,
            atol=tolerance,
            btol=tolerance,
            maxiter=40,
            x0=current_right.reshape(-1),
        )[0]
        return solution.reshape(rank, columns)

    loss = _one_site_factorization_loss(target, left, right, metric)
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        previous = loss
        proposed_left = optimize_left(left, right)
        proposed_loss = _one_site_factorization_loss(
            target, proposed_left, right, metric
        )
        if np.isfinite(proposed_loss) and proposed_loss <= loss:
            left = proposed_left
            loss = proposed_loss
        proposed_right = optimize_right(left, right)
        proposed_loss = _one_site_factorization_loss(
            target, left, proposed_right, metric
        )
        if np.isfinite(proposed_loss) and proposed_loss <= loss:
            right = proposed_right
            loss = proposed_loss
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if (
            left_norm > np.finfo(float).tiny
            and right_norm > np.finfo(float).tiny
        ):
            scale = np.sqrt(right_norm / left_norm)
            left *= scale
            right /= scale
        iterations = iteration
        if previous - loss <= tolerance * max(1.0, previous):
            break
    return left, right, loss, iterations


def _directional_one_site_metric_trim(
    left_tensor,
    right_tensor,
    effective_metric,
    *,
    bond_dimension,
    direction,
    tolerance,
    max_iterations,
    metric_tolerance,
):
    """Trim an expanded bond in the active one-site LETTA norm."""

    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    bond_dimension = int(bond_dimension)
    direction = str(direction).lower()
    if direction == "lr":
        target = left_tensor.reshape(-1, left_tensor.shape[-1])
        active, transfer, loss, iterations = _metric_low_rank_factorization(
            target,
            effective_metric,
            bond_dimension,
            tolerance=tolerance,
            max_iterations=max_iterations,
            metric_tolerance=metric_tolerance,
        )
        trimmed_left = active.reshape(
            left_tensor.shape[:-1] + (bond_dimension,)
        )
        trimmed_right = np.tensordot(
            transfer, right_tensor, axes=([1], [0])
        )
        approximation = active @ transfer
    elif direction == "rl":
        target = right_tensor.reshape(right_tensor.shape[0], -1)
        transfer, active, loss, iterations = _metric_low_rank_factorization(
            target,
            effective_metric,
            bond_dimension,
            tolerance=tolerance,
            max_iterations=max_iterations,
            metric_tolerance=metric_tolerance,
        )
        trimmed_left = np.tensordot(
            left_tensor, transfer, axes=([-1], [0])
        )
        trimmed_right = active.reshape(
            (bond_dimension,) + right_tensor.shape[1:]
        )
        approximation = transfer @ active
    else:
        raise ValueError("direction must be 'lr' or 'rl'.")
    approximation_vector = approximation.reshape(-1)
    norm_squared = float(
        max(
            0.0,
            np.real(
                np.vdot(
                    approximation_vector,
                    effective_metric @ approximation_vector,
                )
            ),
        )
    )
    return CBETrim(
        left_tensor=trimmed_left,
        right_tensor=trimmed_right,
        loss=loss,
        iterations=iterations,
        norm=float(np.sqrt(norm_squared)),
    )


def _strict_shrewd_cbe_bond_update(
    state,
    layout,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    direction,
    options,
):
    from dataclasses import replace

    from .solver import _update_from_cached_environments

    left_site = layout.left_site
    right_site = left_site + 1
    original_left = state.tensors[left_site].copy()
    original_right = state.tensors[right_site].copy()
    original_bond_dimension = original_left.shape[-1]
    old_energy, _old_norm = _streamed_bond_energy(
        hamiltonian_cache,
        metric_cache,
        hamiltonian_left,
        hamiltonian_right,
        metric_left,
        metric_right,
        layout,
    )
    preselection_dimension = options.cbe_preselection_dimension
    if preselection_dimension is None:
        preselection_dimension = max(
            options.cbe_expansion_dimension,
            min(
                original_bond_dimension
                + 2 * options.cbe_expansion_dimension,
                int(np.prod(original_left.shape[:-1])),
                int(np.prod(original_right.shape[1:])),
            ),
        )
    selection = streamed_shrewd_cbe_selection(
        hamiltonian_cache,
        hamiltonian_left,
        hamiltonian_right,
        layout,
        original_left,
        original_right,
        expansion_dimension=options.cbe_expansion_dimension,
        preselection_dimension=preselection_dimension,
        direction=direction,
        tolerance=options.cbe_selection_tolerance,
        metric_cache=metric_cache,
        metric_left=metric_left,
        metric_right=metric_right,
        energy=old_energy,
        metric_tolerance=options.metric_tolerance,
    )
    common_diagnostics = dict(
        cbe_expansion_dimension=options.cbe_expansion_dimension,
        cbe_selector="shrewd",
        cbe_preselection_dimension=selection.preselection_dimension,
        cbe_pair_dimension=int(np.prod(layout.merged_shape)),
        cbe_pair_metric_rank=None,
        cbe_tangent_rank=None,
        cbe_projection_iterations=0,
        cbe_projection_converged=True,
        cbe_selector_pair_action_count=0,
        cbe_selector_pair_metric_count=selection.pair_metric_count,
        cbe_selector_merged_pair_count=selection.merged_pair_count,
        cbe_preselection_output_size=selection.preselection_output_size,
        cbe_final_output_size=selection.final_output_size,
        cbe_materialized_pair_tensor=False,
        cbe_materialized_pair_metric=False,
        cbe_materialized_tangent_jacobian=False,
        cbe_trim_method="one-site-metric-als",
        cbe_missing_norm=selection.missing_norm,
        cbe_old_energy=old_energy,
    )
    if (
        selection.missing_norm is None
        or selection.missing_norm <= options.cbe_selection_tolerance
        or selection.sector_ranks == (0,)
    ):
        fallback = _ordinary_bond_fallback(
            state,
            layout,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            hamiltonian_right,
            metric_left,
            metric_right,
            direction,
            options,
        )
        return replace(
            fallback,
            **common_diagnostics,
            cbe_captured_weight=0.0,
            cbe_selection_loss=selection.loss,
            cbe_trim_loss=0.0,
            cbe_fallback=True,
        )

    expanded_left, expanded_right = embed_cbe_pair(
        original_left,
        original_right,
        selection.left_direction,
        selection.right_direction,
        direction=direction,
    )
    state.tensors[left_site] = expanded_left
    state.tensors[right_site] = expanded_right
    if direction == "lr":
        active_site = left_site
        local_hamiltonian_right = hamiltonian_cache.extend_right(
            hamiltonian_right, right_site
        )
        local_metric_right = metric_cache.extend_right(
            metric_right, right_site
        )
        effective_trim_metric = metric_cache.effective_metric(
            metric_left,
            local_metric_right,
            active_site,
        )
        site_update = _update_from_cached_environments(
            state,
            active_site,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            local_hamiltonian_right,
            metric_left,
            local_metric_right,
            options,
            effective_metric=effective_trim_metric,
        )
    else:
        active_site = right_site
        local_hamiltonian_left = hamiltonian_cache.extend_left(
            hamiltonian_left, left_site
        )
        local_metric_left = metric_cache.extend_left(metric_left, left_site)
        effective_trim_metric = metric_cache.effective_metric(
            local_metric_left,
            metric_right,
            active_site,
        )
        site_update = _update_from_cached_environments(
            state,
            active_site,
            hamiltonian_cache,
            metric_cache,
            local_hamiltonian_left,
            hamiltonian_right,
            local_metric_left,
            metric_right,
            options,
            effective_metric=effective_trim_metric,
        )

    trim = _directional_one_site_metric_trim(
        state.tensors[left_site],
        state.tensors[right_site],
        effective_trim_metric,
        bond_dimension=original_bond_dimension,
        direction=direction,
        tolerance=options.cbe_selection_tolerance,
        max_iterations=options.cbe_refinement_max_iterations,
        metric_tolerance=options.metric_tolerance,
    )
    state.tensors[left_site] = trim.left_tensor
    state.tensors[right_site] = trim.right_tensor
    trimmed_energy, trimmed_norm = _streamed_bond_energy(
        hamiltonian_cache,
        metric_cache,
        hamiltonian_left,
        hamiltonian_right,
        metric_left,
        metric_right,
        layout,
    )
    candidate_is_safe = (
        site_update.accepted
        and trimmed_norm > np.finfo(float).tiny
        and np.isfinite(trimmed_energy)
        and trimmed_energy
        <= old_energy + options.energy_increase_tolerance
    )
    candidate_left = trim.left_tensor.copy()
    candidate_right = trim.right_tensor.copy()
    state.tensors[left_site] = original_left
    state.tensors[right_site] = original_right
    baseline = _ordinary_bond_fallback(
        state,
        layout,
        hamiltonian_cache,
        metric_cache,
        hamiltonian_left,
        hamiltonian_right,
        metric_left,
        metric_right,
        direction,
        options,
    )
    baseline_allowance = _cbe_baseline_allowance(
        old_energy, baseline.energy, options
    )
    accepted = (
        candidate_is_safe
        and trimmed_energy
        <= baseline.energy
        + baseline_allowance
        + options.energy_increase_tolerance
    )
    if accepted:
        state.tensors[left_site] = candidate_left
        state.tensors[right_site] = candidate_right
        normalization = np.sqrt(trimmed_norm**2)
        if direction == "lr":
            state.tensors[right_site] /= normalization
        else:
            state.tensors[left_site] /= normalization
        return replace(
            site_update,
            energy=trimmed_energy,
            accepted=True,
            **common_diagnostics,
            cbe_captured_weight=selection.captured_weight,
            cbe_selection_loss=selection.loss,
            cbe_trim_loss=trim.loss,
            cbe_expanded_energy=site_update.energy,
            cbe_trimmed_energy=trimmed_energy,
            cbe_baseline_energy=baseline.energy,
            cbe_baseline_allowance=baseline_allowance,
            cbe_baseline_selected=False,
            cbe_fallback=False,
        )
    return replace(
        baseline,
        **common_diagnostics,
        cbe_captured_weight=selection.captured_weight,
        cbe_selection_loss=selection.loss,
        cbe_trim_loss=trim.loss,
        cbe_expanded_energy=site_update.energy,
        cbe_trimmed_energy=trimmed_energy,
        cbe_baseline_energy=baseline.energy,
        cbe_baseline_allowance=baseline_allowance,
        cbe_baseline_selected=True,
        cbe_fallback=True,
    )


def _cbe_bond_update(
    state,
    layout,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    direction,
    options,
):
    if options.cbe_selector == "shrewd":
        return _strict_shrewd_cbe_bond_update(
            state,
            layout,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            hamiltonian_right,
            metric_left,
            metric_right,
            direction,
            options,
        )

    from dataclasses import replace

    from .solver import _update_from_cached_environments

    left_site = layout.left_site
    right_site = left_site + 1
    original_left = state.tensors[left_site].copy()
    original_right = state.tensors[right_site].copy()
    original_bond_dimension = original_left.shape[-1]
    pair_metric = metric_cache.effective_pair_metric(
        metric_left, metric_right, layout
    )

    def pair_action(vector):
        return hamiltonian_cache.effective_pair_action(
            hamiltonian_left,
            hamiltonian_right,
            layout,
            vector,
        )

    missing = exact_missing_pair_direction(
        layout,
        original_left,
        original_right,
        pair_action,
        pair_metric,
        metric_tolerance=options.metric_tolerance,
    )
    pair_dimension = int(np.prod(layout.merged_shape))
    common_diagnostics = dict(
        cbe_expansion_dimension=options.cbe_expansion_dimension,
        cbe_selector=missing.selector,
        cbe_pair_dimension=pair_dimension,
        cbe_pair_metric_rank=missing.metric_rank,
        cbe_tangent_rank=missing.tangent_rank,
        cbe_projection_iterations=missing.projection_iterations,
        cbe_projection_converged=missing.projection_converged,
        cbe_selector_pair_action_count=missing.pair_action_count,
        cbe_selector_pair_metric_count=1,
        cbe_selector_merged_pair_count=None,
        cbe_materialized_pair_tensor=True,
        cbe_materialized_pair_metric=missing.materialized_pair_metric,
        cbe_materialized_tangent_jacobian=(
            missing.materialized_tangent_jacobian
        ),
        cbe_trim_method="pair-metric-als",
        cbe_missing_norm=missing.missing_norm,
        cbe_old_energy=missing.energy,
    )
    if missing.missing_norm <= options.cbe_selection_tolerance:
        fallback = _ordinary_bond_fallback(
            state,
            layout,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            hamiltonian_right,
            metric_left,
            metric_right,
            direction,
            options,
        )
        return replace(
            fallback,
            **common_diagnostics,
            cbe_captured_weight=0.0,
            cbe_selection_loss=0.0,
            cbe_trim_loss=0.0,
            cbe_fallback=True,
        )

    selection = select_cbe_directions(
        missing,
        layout,
        pair_metric,
        expansion_dimension=options.cbe_expansion_dimension,
        direction=direction,
        tolerance=options.cbe_selection_tolerance,
        max_iterations=options.cbe_refinement_max_iterations,
        metric_tolerance=options.metric_tolerance,
    )
    expanded_left, expanded_right = embed_cbe_pair(
        original_left,
        original_right,
        selection.left_direction,
        selection.right_direction,
        direction=direction,
    )
    state.tensors[left_site] = expanded_left
    state.tensors[right_site] = expanded_right

    if direction == "lr":
        active_site = left_site
        local_hamiltonian_right = hamiltonian_cache.extend_right(
            hamiltonian_right, right_site
        )
        local_metric_right = metric_cache.extend_right(
            metric_right, right_site
        )
        site_update = _update_from_cached_environments(
            state,
            active_site,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_left,
            local_hamiltonian_right,
            metric_left,
            local_metric_right,
            options,
        )
    else:
        active_site = right_site
        local_hamiltonian_left = hamiltonian_cache.extend_left(
            hamiltonian_left, left_site
        )
        local_metric_left = metric_cache.extend_left(metric_left, left_site)
        site_update = _update_from_cached_environments(
            state,
            active_site,
            hamiltonian_cache,
            metric_cache,
            local_hamiltonian_left,
            hamiltonian_right,
            local_metric_left,
            metric_right,
            options,
        )

    expanded_pair = layout.merge(
        state.tensors[left_site], state.tensors[right_site]
    )
    expanded_energy = _pair_rayleigh(
        expanded_pair.reshape(-1), pair_action, pair_metric
    )
    trim = metric_trim_pair(
        expanded_pair,
        layout,
        pair_metric,
        bond_dimension=original_bond_dimension,
        direction=direction,
        tolerance=options.cbe_selection_tolerance,
        max_iterations=options.cbe_refinement_max_iterations,
        metric_tolerance=options.metric_tolerance,
    )
    trimmed_pair = layout.merge(trim.left_tensor, trim.right_tensor).reshape(-1)
    trimmed_energy = _pair_rayleigh(trimmed_pair, pair_action, pair_metric)
    candidate_is_safe = (
        site_update.accepted
        and trim.norm > np.finfo(float).tiny
        and np.isfinite(trimmed_energy)
        and trimmed_energy
        <= missing.energy + options.energy_increase_tolerance
    )
    candidate_left = trim.left_tensor.copy()
    candidate_right = trim.right_tensor.copy()
    state.tensors[left_site] = original_left
    state.tensors[right_site] = original_right
    baseline = _ordinary_bond_fallback(
        state,
        layout,
        hamiltonian_cache,
        metric_cache,
        hamiltonian_left,
        hamiltonian_right,
        metric_left,
        metric_right,
        direction,
        options,
    )
    baseline_allowance = _cbe_baseline_allowance(
        missing.energy, baseline.energy, options
    )
    accepted = (
        candidate_is_safe
        and trimmed_energy
        <= baseline.energy
        + baseline_allowance
        + options.energy_increase_tolerance
    )
    if accepted:
        state.tensors[left_site] = candidate_left
        state.tensors[right_site] = candidate_right
        return replace(
            site_update,
            energy=trimmed_energy,
            accepted=True,
            **common_diagnostics,
            cbe_captured_weight=selection.captured_weight,
            cbe_selection_loss=selection.loss,
            cbe_trim_loss=trim.loss,
            cbe_expanded_energy=expanded_energy,
            cbe_trimmed_energy=trimmed_energy,
            cbe_baseline_energy=baseline.energy,
            cbe_baseline_allowance=baseline_allowance,
            cbe_baseline_selected=False,
            cbe_fallback=False,
        )
    return replace(
        baseline,
        **common_diagnostics,
        cbe_captured_weight=selection.captured_weight,
        cbe_selection_loss=selection.loss,
        cbe_trim_loss=trim.loss,
        cbe_expanded_energy=expanded_energy,
        cbe_trimmed_energy=trimmed_energy,
        cbe_baseline_energy=baseline.energy,
        cbe_baseline_allowance=baseline_allowance,
        cbe_baseline_selected=True,
        cbe_fallback=True,
    )


@_stable_floating_point
def cbe_cached_mpo_sweep(
    state,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_environments,
    metric_environments,
    direction,
    options,
):
    """Perform one exact-or-strict-shrewd LETTA-CBE sweep."""

    from .solver import _shift_virtual_gauge, _update_from_cached_environments

    updates = []
    if direction == "lr":
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        hamiltonian_environments[0] = hamiltonian_boundary
        metric_environments[0] = metric_boundary
        for site in range(state.nsites - 1):
            layout = LETTAPairLayout.from_state(state, site)
            updates.append(
                _cbe_bond_update(
                    state,
                    layout,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_boundary,
                    hamiltonian_environments[site + 2],
                    metric_boundary,
                    metric_environments[site + 2],
                    direction,
                    options,
                )
            )
            _shift_virtual_gauge(state, site, direction, options.gauge_mode)
            hamiltonian_boundary = hamiltonian_cache.extend_left(
                hamiltonian_boundary, site
            )
            metric_boundary = metric_cache.extend_left(metric_boundary, site)
            hamiltonian_environments[site + 1] = hamiltonian_boundary
            metric_environments[site + 1] = metric_boundary
        terminal = state.nsites - 1
        updates.append(
            _update_from_cached_environments(
                state,
                terminal,
                hamiltonian_cache,
                metric_cache,
                hamiltonian_boundary,
                hamiltonian_environments[state.nsites],
                metric_boundary,
                metric_environments[state.nsites],
                options,
            )
        )
        _shift_virtual_gauge(state, terminal, direction, options.gauge_mode)
        hamiltonian_boundary = hamiltonian_cache.extend_left(
            hamiltonian_boundary, terminal
        )
        metric_boundary = metric_cache.extend_left(metric_boundary, terminal)
        hamiltonian_environments[state.nsites] = hamiltonian_boundary
        metric_environments[state.nsites] = metric_boundary
    else:
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        hamiltonian_environments[state.nsites] = hamiltonian_boundary
        metric_environments[state.nsites] = metric_boundary
        for site in range(state.nsites - 2, -1, -1):
            layout = LETTAPairLayout.from_state(state, site)
            updates.append(
                _cbe_bond_update(
                    state,
                    layout,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_environments[site],
                    hamiltonian_boundary,
                    metric_environments[site],
                    metric_boundary,
                    direction,
                    options,
                )
            )
            active_site = site + 1
            _shift_virtual_gauge(
                state, active_site, direction, options.gauge_mode
            )
            hamiltonian_boundary = hamiltonian_cache.extend_right(
                hamiltonian_boundary, active_site
            )
            metric_boundary = metric_cache.extend_right(
                metric_boundary, active_site
            )
            hamiltonian_environments[active_site] = hamiltonian_boundary
            metric_environments[active_site] = metric_boundary
        updates.append(
            _update_from_cached_environments(
                state,
                0,
                hamiltonian_cache,
                metric_cache,
                hamiltonian_environments[0],
                hamiltonian_boundary,
                metric_environments[0],
                metric_boundary,
                options,
            )
        )
        _shift_virtual_gauge(state, 0, direction, options.gauge_mode)
        hamiltonian_boundary = hamiltonian_cache.extend_right(
            hamiltonian_boundary, 0
        )
        metric_boundary = metric_cache.extend_right(metric_boundary, 0)
        hamiltonian_environments[0] = hamiltonian_boundary
        metric_environments[0] = metric_boundary

    energy = float(np.real(hamiltonian_boundary / metric_boundary))
    return (
        tuple(updates),
        energy,
        hamiltonian_environments,
        metric_environments,
    )

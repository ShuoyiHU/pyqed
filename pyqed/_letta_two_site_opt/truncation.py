"""Environment-weighted projection of merged tensors back into LETTA form."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .._letta_one_site_opt.contractions import (
    BlockDiagonalMetric, DiagonalMetric, _equilibrated_metric_factors,
)
from .._letta_compression import MetricCompressionOptions, compress_factors
from .pair import LETTAPairLayout, LETTASplit


@dataclass(frozen=True)
class LETTAMetricRefinement:
    left_tensor: np.ndarray
    right_tensor: np.ndarray
    loss: float
    iterations: int
    diagnostics: dict | None = None


class _MetricSquareRoot:
    """Blockwise factor S with S^H S equal to the supported overlap metric.

    S need not be Hermitian. Its rank is resolved after column equilibration,
    so a small coordinate weight alone does not erase a physical fit error.
    """

    def __init__(self, metric, tolerance):
        if not isinstance(metric, BlockDiagonalMetric):
            raise TypeError("metric must be a BlockDiagonalMetric.")
        grouped = {}
        rank = 0
        for block, indices in zip(metric.blocks, metric.indices):
            if isinstance(metric, DiagonalMetric) and metric.support is not None:
                keep = metric.support[indices]
                block, indices = block[np.ix_(keep, keep)], indices[keep]
            _, factor = _equilibrated_metric_factors(block, tolerance)
            width = factor.shape[0]
            if not width:
                continue
            grouped.setdefault((factor.shape, factor.dtype), []).append(
                (indices, indices[:width], factor))
            rank += width
        if not rank:
            raise ValueError("the two-site LETTA overlap metric has zero rank.")
        self.size, self.dtype = metric.size, metric.dtype
        self.groups = tuple(tuple(map(np.stack, zip(*pieces)))
                            for pieces in grouped.values())

    def apply(self, vector):
        vector = np.asarray(vector)
        result = np.zeros(self.size, dtype=np.result_type(self.dtype, vector))
        for indices, target, factors in self.groups:
            result[target] = (factors @ vector[indices][..., None])[..., 0]
        return result

    def adjoint(self, vector):
        vector = np.asarray(vector)
        result = np.zeros(self.size, dtype=np.result_type(self.dtype, vector))
        for indices, source, factors in self.groups:
            result[indices] = (factors.conj().swapaxes(1, 2)
                               @ vector[source][..., None])[..., 0]
        return result


def _metric_loss(target, layout, left_tensor, right_tensor, metric):
    difference = target - layout.merge(left_tensor, right_tensor).reshape(-1)
    return float(max(0.0, np.real(np.vdot(difference, metric @ difference))))


def _lsmr_iterations(variable_size):
    return max(50, min(1000, 5 * int(variable_size)))


def _embedded(vector, indices, size, dtype):
    if indices is None:
        return np.asarray(vector, dtype=dtype)
    result = np.zeros(size, dtype=np.result_type(dtype, vector))
    result[indices] = vector
    return result


def _optimize_left(
    target,
    layout,
    left_tensor,
    right_tensor,
    square_root,
    tolerance,
    indices,
    lsmr_max_iterations=None,
):
    target_shape = layout.merged_shape
    variable_shape = left_tensor.shape

    def forward(vector):
        full = _embedded(vector, indices, left_tensor.size, left_tensor.dtype)
        merged = layout.merge(full.reshape(variable_shape), right_tensor)
        return square_root.apply(merged.reshape(-1))

    def adjoint(vector):
        weighted = square_root.adjoint(vector).reshape(target_shape)
        result = layout.left_adjoint(weighted, right_tensor).reshape(-1)
        return result if indices is None else result[indices]

    variable_size = left_tensor.size if indices is None else len(indices)

    operator = LinearOperator(
        (square_root.size, variable_size),
        matvec=forward,
        rmatvec=adjoint,
        dtype=np.result_type(target, left_tensor, right_tensor),
    )
    solution = lsmr(
        operator,
        square_root.apply(target),
        atol=tolerance,
        btol=tolerance,
        maxiter=lsmr_max_iterations or _lsmr_iterations(variable_size),
        x0=(
            left_tensor.reshape(-1)
            if indices is None
            else left_tensor.reshape(-1)[indices]
        ),
    )[0]
    return _embedded(
        solution, indices, left_tensor.size, left_tensor.dtype
    ).reshape(variable_shape)


def _optimize_right(
    target,
    layout,
    left_tensor,
    right_tensor,
    square_root,
    tolerance,
    indices,
    lsmr_max_iterations=None,
):
    target_shape = layout.merged_shape
    variable_shape = right_tensor.shape

    def forward(vector):
        full = _embedded(vector, indices, right_tensor.size, right_tensor.dtype)
        merged = layout.merge(left_tensor, full.reshape(variable_shape))
        return square_root.apply(merged.reshape(-1))

    def adjoint(vector):
        weighted = square_root.adjoint(vector).reshape(target_shape)
        result = layout.right_adjoint(left_tensor, weighted).reshape(-1)
        return result if indices is None else result[indices]

    variable_size = right_tensor.size if indices is None else len(indices)

    operator = LinearOperator(
        (square_root.size, variable_size),
        matvec=forward,
        rmatvec=adjoint,
        dtype=np.result_type(target, left_tensor, right_tensor),
    )
    solution = lsmr(
        operator,
        square_root.apply(target),
        atol=tolerance,
        btol=tolerance,
        maxiter=lsmr_max_iterations or _lsmr_iterations(variable_size),
        x0=(
            right_tensor.reshape(-1)
            if indices is None
            else right_tensor.reshape(-1)[indices]
        ),
    )[0]
    return _embedded(
        solution, indices, right_tensor.size, right_tensor.dtype
    ).reshape(variable_shape)


def _balance_pair(left_tensor, right_tensor):
    left_norm = np.linalg.norm(left_tensor)
    right_norm = np.linalg.norm(right_tensor)
    if left_norm <= np.finfo(float).tiny or right_norm <= np.finfo(float).tiny:
        return left_tensor, right_tensor
    scale = np.sqrt(right_norm / left_norm)
    return left_tensor * scale, right_tensor / scale


def metric_als_refine(
    target,
    layout,
    initial,
    metric,
    *,
    tolerance=1.0e-10,
    max_iterations=8,
    metric_tolerance=1.0e-12,
    left_indices=None,
    right_indices=None,
    lsmr_max_iterations=None,
):
    """Improve a conditional split in the full LETTA wavefunction norm."""

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    if not isinstance(initial, LETTASplit):
        raise TypeError("initial must be a LETTASplit.")
    target = np.asarray(target)
    if tuple(target.shape) != layout.merged_shape:
        raise ValueError("target tensor shape does not match the pair layout.")
    tolerance = float(tolerance)
    metric_tolerance = float(metric_tolerance)
    max_iterations = int(max_iterations)
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("truncation tolerances must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    if lsmr_max_iterations is not None and (not isinstance(lsmr_max_iterations, (int, np.integer)) or lsmr_max_iterations <= 0):
        raise ValueError("lsmr_max_iterations must be a positive integer.")

    target_vector = target.reshape(-1)
    square_root = _MetricSquareRoot(metric, metric_tolerance)
    left_tensor = initial.left_tensor.copy()
    right_tensor = initial.right_tensor.copy()
    loss = _metric_loss(
        target_vector, layout, left_tensor, right_tensor, metric
    )
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        previous = loss
        proposed_left = _optimize_left(
            target_vector,
            layout,
            left_tensor,
            right_tensor,
            square_root,
            tolerance,
            left_indices,
            lsmr_max_iterations,
        )
        proposed_loss = _metric_loss(
            target_vector, layout, proposed_left, right_tensor, metric
        )
        if proposed_loss <= loss + 10.0 * np.finfo(float).eps:
            left_tensor = proposed_left
            loss = proposed_loss

        proposed_right = _optimize_right(
            target_vector,
            layout,
            left_tensor,
            right_tensor,
            square_root,
            tolerance,
            right_indices,
            lsmr_max_iterations,
        )
        proposed_loss = _metric_loss(
            target_vector, layout, left_tensor, proposed_right, metric
        )
        if proposed_loss <= loss + 10.0 * np.finfo(float).eps:
            right_tensor = proposed_right
            loss = proposed_loss
        left_tensor, right_tensor = _balance_pair(left_tensor, right_tensor)
        iterations = iteration
        improvement = previous - loss
        if improvement <= tolerance * max(1.0, previous):
            break
    return LETTAMetricRefinement(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        loss=loss,
        iterations=iterations,
    )


def metric_refine(target, layout, initial, metric, *, compression=None,
                  tolerance=1e-10, max_iterations=8, metric_tolerance=1e-12,
                  left_indices=None, right_indices=None):
    """Selectable norm-compression solver; energy refinement is separate."""
    if not isinstance(layout, LETTAPairLayout) or not isinstance(initial, LETTASplit):
        raise TypeError("metric_refine requires LETTAPairLayout and LETTASplit.")
    if np.shape(target) != layout.merged_shape:
        raise ValueError("target tensor shape does not match the pair layout.")
    if layout.symmetry is not None:
        if left_indices is None:
            left_indices = np.flatnonzero(layout.factor_mask("left"))
        if right_indices is None:
            right_indices = np.flatnonzero(layout.factor_mask("right"))
    compression = compression or MetricCompressionOptions()
    if not isinstance(compression, MetricCompressionOptions):
        raise TypeError("compression must be MetricCompressionOptions.")
    def fallback():
        fit = metric_als_refine(
            target, layout, initial, metric, tolerance=tolerance,
            max_iterations=compression.als_max_iterations or max_iterations,
            metric_tolerance=metric_tolerance, left_indices=left_indices,
            right_indices=right_indices,
            lsmr_max_iterations=compression.lsmr_max_iterations)
        return fit.left_tensor, fit.right_tensor, fit.loss, fit.iterations
    fit = compress_factors(
        target, metric, initial.left_tensor, initial.right_tensor,
        options=compression, fallback=fallback, layout=layout,
        metric_tolerance=metric_tolerance, left_indices=left_indices,
        right_indices=right_indices)
    return LETTAMetricRefinement(fit.left, fit.right, fit.loss, fit.iterations, fit.diagnostics)

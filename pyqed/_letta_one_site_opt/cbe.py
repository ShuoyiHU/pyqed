"""Metric-aware controlled bond expansion for one-site LETTA.

The exact pair-space selector is retained as a correctness oracle.  The
strict shrewd path uses weighted half-environment preselection, a streamed
metric-projected physical residual, an expanded one-site solve, and
one-site-metric trimming, and fixed-rank alternating energy relaxation.
It never constructs a merged pair tensor, pair action, or pair metric.
"""

from __future__ import annotations

from .._letta_compression import MetricCompressionOptions, compress_factors


from dataclasses import dataclass
from functools import wraps
import time

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
    metric_refine,
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
    metric_kinds: tuple[str, ...] = ()
    tangent_iterations: int = 0
    tangent_relative_residual: float = 0.0
    overlap_applications: int = 0
    connector_dimension: int = 0
    tangent_block_shapes: tuple[tuple[int, int], ...] = ()
    selection_timings: dict[str, float] | None = None


@dataclass(frozen=True)
class CBETrim:
    """A metric-aware fixed-bond approximation to an expanded pair."""

    left_tensor: np.ndarray
    right_tensor: np.ndarray
    loss: float
    iterations: int
    norm: float
    metric_kinds: tuple[str, ...] = ()
    diagnostics: tuple[dict, ...] = ()


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
    hamiltonian_cache, hamiltonian_left, hamiltonian_right, layout,
    left_tensor, right_tensor, *, expansion_dimension, preselection_dimension,
    direction, tolerance=1e-12, metric_cache=None, metric_left=None,
    metric_right=None, energy=None, metric_tolerance=1e-12,
):
    """Dependency-aware conditional CBE with a restricted physical fit."""
    from .cbe_general import general_cbe_selection
    return general_cbe_selection(
        hamiltonian_cache, hamiltonian_left, hamiltonian_right, layout,
        left_tensor, right_tensor, expansion_dimension=expansion_dimension,
        preselection_dimension=preselection_dimension, direction=direction,
        tolerance=tolerance, metric_cache=metric_cache, metric_left=metric_left,
        metric_right=metric_right, energy=energy, metric_tolerance=metric_tolerance,
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
    compression=None,
):
    """Trim an expanded pair in the full LETTA norm metric."""

    target = np.asarray(target)
    split = conditional_svd_split(
        target,
        layout,
        max_bond_dim=int(bond_dimension),
        direction=direction,
    )
    refinement = metric_refine(
        target,
        layout,
        split,
        pair_metric,
        compression=compression,
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
        diagnostics=(refinement.diagnostics,),
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



def _cbe_candidate_is_preferred(candidate_energy, old_energy, baseline_energy, options):
    """Choose the acceptance reference explicitly; ties use the baseline.

    Zero allowance requires strictly more descent than the ordinary update.
    Full allowance requires descent from the pre-update state. Intermediate
    fractions retain the exploratory allowance for controlled comparisons.
    """
    if not np.isfinite(candidate_energy):
        return False
    fraction = options.cbe_baseline_guard_fraction
    if fraction == 0.0:
        return candidate_energy < min(old_energy, baseline_energy)
    if fraction == 1.0:
        return candidate_energy < old_energy
    return candidate_energy <= (
        baseline_energy + _cbe_baseline_allowance(old_energy, baseline_energy, options)
        + options.energy_increase_tolerance
    )


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
    *,
    local_environments=None,
):
    from .solver import _update_from_cached_environments

    if direction == "lr":
        active_site = layout.left_site
        if local_environments is None:
            local_environments = (
                hamiltonian_cache.extend_right(hamiltonian_right, layout.left_site + 1),
                metric_cache.extend_right(metric_right, layout.left_site + 1))
        local_hamiltonian_right, local_metric_right = local_environments
        if local_metric_right is None:
            local_metric_right = metric_cache.extend_right(metric_right, layout.left_site + 1)
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
    if local_environments is None:
        local_environments = (
            hamiltonian_cache.extend_left(hamiltonian_left, layout.left_site),
            metric_cache.extend_left(metric_left, layout.left_site))
    local_hamiltonian_left, local_metric_left = local_environments
    if local_metric_left is None:
        local_metric_left = metric_cache.extend_left(metric_left, layout.left_site)
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
    layout, *, reject_invalid=False,
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
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        if reject_invalid:
            return float("inf"), 0.0
        raise ValueError("the streamed LETTA bond has zero metric norm.")
    energy = float(np.real(numerator / denominator))
    if not np.isfinite(energy):
        raise FloatingPointError("the streamed LETTA bond energy is nonfinite.")
    return energy, float(np.sqrt(denominator))


@dataclass(frozen=True)
class CBEEnergyRefinement:
    left_tensor: np.ndarray
    right_tensor: np.ndarray
    initial_energy: float
    energy: float
    norm: float
    iterations: int
    accepted_substeps: int
    hamiltonian_applications: int
    coupled_iterations: int = 0
    coupled_accepted_steps: int = 0
    coupled_hamiltonian_applications: int = 0


def _refine_cbe_pair_energy(
    state, layout, hamiltonian_cache, metric_cache,
    hamiltonian_left, hamiltonian_right, metric_left, metric_right,
    direction, options, left_tensor, right_tensor, *, norm_limit=None,
):
    """Relax fixed-rank factors using fresh one-site environments only.

    The outer environments remain fixed. Restore the live pair even if a
    local solve fails, so competing initializations see identical surroundings.
    """
    from .solver import _update_from_cached_environments

    i, j = layout.left_site, layout.left_site + 1
    saved = state.tensors[i], state.tensors[j]
    state.tensors[i], state.tensors[j] = left_tensor.copy(), right_tensor.copy()

    def energy(*, reject_invalid=False):
        return _streamed_bond_energy(
            hamiltonian_cache, metric_cache, hamiltonian_left, hamiltonian_right,
            metric_left, metric_right, layout, reject_invalid=reject_invalid,
        )

    try:
        initial_energy, norm = energy()
        current_energy = initial_energy
        # The budget belongs to this start, before any relaxation changes its gauge.
        start_limit = 100.0 * np.sqrt(
            np.linalg.norm(left_tensor) * np.linalg.norm(right_tensor) / norm
        )
        norm_limit = start_limit if norm_limit is None else min(norm_limit, start_limit)
        accepted_substeps = applications = iterations = 0
        order = (i, j) if direction == "lr" else (j, i)
        for iteration in range(options.cbe_energy_refinement_max_iterations):
            previous = current_energy
            for site in order:
                old_tensor = state.tensors[site].copy()
                if site == i:
                    local = (
                        hamiltonian_left, hamiltonian_cache.extend_right(hamiltonian_right, j),
                        metric_left, metric_cache.extend_right(metric_right, j),
                    )
                else:
                    local = (
                        hamiltonian_cache.extend_left(hamiltonian_left, i), hamiltonian_right,
                        metric_cache.extend_left(metric_left, i), metric_right,
                    )
                update = _update_from_cached_environments(
                    state, site, hamiltonian_cache, metric_cache, *local, options,
                )
                applications += update.hamiltonian_applications
                proposed, proposed_norm = energy(reject_invalid=True)
                # Apply the per-start conditioning budget to alternating
                # updates too, not only to the later coupled correction.
                # Measure after scalar balancing so harmless A/B rescaling
                # does not change the decision.
                balanced_norm = np.sqrt(
                    np.linalg.norm(state.tensors[i])
                    * np.linalg.norm(state.tensors[j]) / proposed_norm
                ) if proposed_norm > np.finfo(float).tiny else float("inf")
                if (update.accepted and proposed <= current_energy
                        and np.isfinite(balanced_norm) and balanced_norm <= norm_limit):
                    current_energy, norm = proposed, proposed_norm
                    accepted_substeps += 1
                else:
                    state.tensors[site] = old_tensor
            iterations = iteration + 1
            if previous - current_energy <= options.cbe_energy_refinement_tolerance:
                break
        checks = coupled_steps = coupled_applications = 0
        if (options.cbe_energy_refinement_max_iterations
                and options.cbe_coupled_max_iterations
                and state.tensors[i].size + state.tensors[j].size
                    <= options.cbe_coupled_max_parameters
                and initial_energy - current_energy
                    <= options.cbe_coupled_energy_threshold * max(1.0, abs(current_energy))):
            from .cbe_coupled import refine_coupled_factors
            coupled = refine_coupled_factors(
                state, layout, hamiltonian_cache, metric_cache,
                hamiltonian_left, hamiltonian_right, metric_left, metric_right,
                options=options, norm_limit=norm_limit,
            )
            checks, coupled_steps = coupled.iterations, coupled.accepted_steps
            coupled_applications = coupled.hamiltonian_applications
            if coupled_steps and coupled.energy <= current_energy:
                state.tensors[i], state.tensors[j] = coupled.left, coupled.right
                current_energy, norm = coupled.energy, coupled.norm
        return CBEEnergyRefinement(
            state.tensors[i].copy(), state.tensors[j].copy(), initial_energy,
            current_energy, norm, iterations, accepted_substeps,
            applications + coupled_applications, checks, coupled_steps,
            coupled_applications,
        )
    finally:
        state.tensors[i], state.tensors[j] = saved


def _refine_cbe_candidates(
    state, layout, hamiltonian_cache, metric_cache,
    hamiltonian_left, hamiltonian_right, metric_left, metric_right,
    direction, options, trim, original_left, original_right, diagnostics,
    *, trim_is_valid=True, norm_limit=None,
):
    """Compare the trim and incumbent *after* independent energy relaxation."""
    started = time.perf_counter()
    arguments = (
        state, layout, hamiltonian_cache, metric_cache,
        hamiltonian_left, hamiltonian_right, metric_left, metric_right,
        direction, options,
    )
    refined = (_refine_cbe_pair_energy(*arguments, trim.left_tensor, trim.right_tensor,
                                     norm_limit=norm_limit) if trim_is_valid else None)
    incumbent = _refine_cbe_pair_energy(*arguments, original_left, original_right,
                                       norm_limit=norm_limit)
    if refined is None or incumbent.energy < refined.energy:
        selected, start = incumbent, "incumbent"
    else:
        selected, start = refined, "trim"
    diagnostics.update(
        cbe_refined_energy=refined.energy if refined is not None else None,
        cbe_incumbent_refined_energy=incumbent.energy,
        cbe_energy_refinement_start=start,
        cbe_energy_refinement_iterations=sum(r.iterations for r in (refined, incumbent) if r is not None),
        cbe_energy_refinement_substeps=sum(r.accepted_substeps for r in (refined, incumbent) if r is not None),
        cbe_coupled_iterations=sum(r.coupled_iterations for r in (refined, incumbent) if r is not None),
        cbe_coupled_accepted_steps=sum(r.coupled_accepted_steps for r in (refined, incumbent) if r is not None),
        cbe_coupled_hamiltonian_applications=sum(r.coupled_hamiltonian_applications
                                                for r in (refined, incumbent) if r is not None),
    )
    diagnostics["cbe_timings"]["energy_refinement"] = time.perf_counter() - started
    return selected, sum(r.hamiltonian_applications for r in (refined, incumbent) if r is not None)


def _within_factor_budget(left, right, norm, limit):
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        return False
    balanced = np.sqrt(np.linalg.norm(left) * np.linalg.norm(right) / norm)
    return bool(np.isfinite(balanced) and balanced <= limit)


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
    initial_factors=None,
    lsmr_max_iterations=40,
):
    target = np.asarray(target, dtype=np.result_type(target, metric.dtype))
    rows, columns = target.shape
    if initial_factors is None:
        vectors, values, adjoint = np.linalg.svd(target, full_matrices=False)
        keep = min(int(rank), values.size)
        left = np.zeros((rows, rank), dtype=target.dtype)
        right = np.zeros((rank, columns), dtype=target.dtype)
        left[:, :keep] = vectors[:, :keep]
        right[:keep] = values[:keep, None] * adjoint[:keep]
    else:
        left, right = (np.array(factor, dtype=target.dtype, copy=True)
                       for factor in initial_factors)
        if left.shape != (rows, rank) or right.shape != (rank, columns):
            raise ValueError("initial factors must have shapes (rows, rank) and (rank, columns).")
    square_root = _MetricSquareRoot(metric, metric_tolerance)
    if square_root.size != target.size:
        raise ValueError("one-site metric and trim target dimensions differ.")
    weighted_target = square_root.apply(target.reshape(-1))

    def optimize_left(current_left, current_right):
        def forward(vector):
            candidate = vector.reshape(rows, rank) @ current_right
            return square_root.apply(candidate.reshape(-1))

        def adjoint_action(vector):
            weighted = square_root.adjoint(vector).reshape(rows, columns)
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
            maxiter=lsmr_max_iterations,
            x0=current_left.reshape(-1),
        )[0]
        return solution.reshape(rows, rank)

    def optimize_right(current_left, current_right):
        def forward(vector):
            candidate = current_left @ vector.reshape(rank, columns)
            return square_root.apply(candidate.reshape(-1))

        def adjoint_action(vector):
            weighted = square_root.adjoint(vector).reshape(rows, columns)
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
            maxiter=lsmr_max_iterations,
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


def _metric_trim_factorization(target, metric, rank, *, tolerance, max_iterations,
                               metric_tolerance, compression=None, diagnostics=None):
    """Use the physical metric's exact separable fit, or general-metric ALS."""
    from .cbe_general import _separable_factors, fit_low_rank_quadratic

    target = np.asarray(target)
    if _separable_factors(metric, target.shape, metric_tolerance * 10) is None:
        # Here the target tensor is already known. Do not replace it by
        # N+ N target: that removes null-space entries which can provide a
        # useful low-rank completion, changing the local ALS initialization.
        config = compression or MetricCompressionOptions()
        u, values, vh = np.linalg.svd(target, full_matrices=False)
        keep = min(rank, len(values))
        left = np.zeros((target.shape[0], rank), dtype=target.dtype)
        right = np.zeros((rank, target.shape[1]), dtype=target.dtype)
        left[:, :keep], right[:keep] = u[:, :keep], values[:keep, None]*vh[:keep]
        def fallback():
            return _metric_low_rank_factorization(
                target, metric, rank, tolerance=tolerance,
                max_iterations=config.als_max_iterations or max_iterations,
                metric_tolerance=metric_tolerance,
                lsmr_max_iterations=config.lsmr_max_iterations or 40)
        fit = compress_factors(target, metric, left, right, options=config,
                               fallback=fallback, metric_tolerance=metric_tolerance)
        if diagnostics is not None:
            diagnostics.append(fit.diagnostics)
        return fit.left, fit.right, fit.loss, fit.iterations, "general"
    fit = fit_low_rank_quadratic(
        metric @ target.reshape(-1), metric, target.shape, rank=rank,
        tolerance=metric_tolerance, max_iterations=max_iterations)
    # Recompute against the original target, avoiding cancellation of nearly
    # equal captured/available weights when the compression loss is tiny.
    loss = _one_site_factorization_loss(target, fit.left, fit.right, metric)
    if diagnostics is not None:
        diagnostics.append(dict(requested_solver=(compression or MetricCompressionOptions()).solver,
                                used_solver="separable-svd", final_loss=loss))
    return fit.left, fit.right, loss, fit.iterations, fit.metric_kind


def _conditional_one_site_metric_trim(
    left_tensor, right_tensor, metric, layout, *, bond_dimension,
    direction, tolerance, max_iterations, metric_tolerance, compression=None,
):
    """Factor each shared-physical sector without forming a merged pair.

    The transfer may depend on every physical index common to both factors.
    Identity environments are block diagonal in these indices, so each sector
    has an independent norm-minimization problem in the active-site metric.
    """
    dtype = np.result_type(left_tensor, right_tensor, metric.dtype)
    active = np.asarray(left_tensor if direction == "lr" else right_tensor, dtype=dtype)
    approximation = np.zeros_like(active)
    left = np.zeros(left_tensor.shape[:-1] + (bond_dimension,), dtype=dtype)
    right = np.zeros((bond_dimension,) + right_tensor.shape[1:], dtype=dtype)
    coordinates = np.arange(active.size).reshape(active.shape)
    loss = 0.0
    iterations = 0
    metric_kinds = []
    diagnostics = []
    for configuration in np.ndindex(*((layout.physical_dim,) * len(layout.shared))):
        left_section = [slice(None)] * left_tensor.ndim
        right_section = [slice(None)] * right_tensor.ndim
        for site, value in zip(layout.shared, configuration):
            left_section[1 + layout.left_neighborhood.index(site)] = value
            right_section[1 + layout.right_neighborhood.index(site)] = value
        left_section, right_section = tuple(left_section), tuple(right_section)
        section = left_section if direction == "lr" else right_section
        sector = active[section]
        target = (sector.reshape(-1, sector.shape[-1]) if direction == "lr"
                  else sector.reshape(sector.shape[0], -1))
        sector_metric = metric.restrict(coordinates[section].reshape(-1))
        # An exact or sufficiently accurate SVD needs no ALS. This also
        # handles zero-norm sectors without a pseudoinverse.
        u, s, vh = np.linalg.svd(target, full_matrices=False)
        keep = min(bond_dimension, s.size)
        a = np.zeros((target.shape[0], bond_dimension), dtype=target.dtype)
        b = np.zeros((bond_dimension, target.shape[1]), dtype=target.dtype)
        a[:, :keep] = u[:, :keep]
        b[:keep] = s[:keep, None] * vh[:keep]
        sector_loss = _one_site_factorization_loss(target, a, b, sector_metric)
        target_norm = float(np.real(np.vdot(target.reshape(-1), sector_metric @ target.reshape(-1))))
        count = 0
        kind = "negligible"
        if sector_loss > tolerance * max(target_norm, np.finfo(float).tiny):
            a, b, sector_loss, count, kind = _metric_trim_factorization(
                target, sector_metric, bond_dimension,
                tolerance=tolerance, max_iterations=max_iterations,
                metric_tolerance=metric_tolerance,
                compression=compression, diagnostics=diagnostics,
            )
        metric_kinds.append(kind)
        if direction == "lr":
            left[left_section] = a.reshape(left[left_section].shape)
            right[right_section] = np.tensordot(b, right_tensor[right_section], axes=([1], [0]))
        else:
            left[left_section] = np.tensordot(left_tensor[left_section], a, axes=([-1], [0]))
            right[right_section] = b.reshape(right[right_section].shape)
        approximation[section] = (a @ b).reshape(sector.shape)
        loss += sector_loss
        iterations = max(iterations, count)
    vector = approximation.reshape(-1)
    norm = float(np.sqrt(max(0.0, np.real(np.vdot(vector, metric @ vector)))))
    return CBETrim(left, right, loss, iterations, norm, tuple(metric_kinds), tuple(diagnostics))


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
    layout=None,
    compression=None,
):
    """Trim an expanded bond in the active one-site LETTA norm."""

    diagnostics = []
    left_tensor = np.asarray(left_tensor)
    right_tensor = np.asarray(right_tensor)
    bond_dimension = int(bond_dimension)
    direction = str(direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'.")
    if layout is not None and layout.shared:
        return _conditional_one_site_metric_trim(
            left_tensor, right_tensor, effective_metric, layout,
            bond_dimension=bond_dimension, direction=direction,
            tolerance=tolerance, max_iterations=max_iterations,
            metric_tolerance=metric_tolerance, compression=compression,
        )
    if direction == "lr":
        target = left_tensor.reshape(-1, left_tensor.shape[-1])
        active, transfer, loss, iterations, kind = _metric_trim_factorization(
            target,
            effective_metric,
            bond_dimension,
            tolerance=tolerance,
            max_iterations=max_iterations,
            metric_tolerance=metric_tolerance,
            compression=compression, diagnostics=diagnostics,
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
        transfer, active, loss, iterations, kind = _metric_trim_factorization(
            target,
            effective_metric,
            bond_dimension,
            tolerance=tolerance,
            max_iterations=max_iterations,
            metric_tolerance=metric_tolerance,
            compression=compression, diagnostics=diagnostics,
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
        metric_kinds=(kind,),
        diagnostics=tuple(diagnostics),
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
    *,
    local_environments=None,
):
    from dataclasses import replace

    from .solver import _update_from_cached_environments

    started = time.perf_counter()
    timings = {}
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
    timings["selection"] = time.perf_counter() - started
    common_diagnostics = dict(
        cbe_timings=timings,
        cbe_selection_diagnostics={
            "stage_seconds": selection.selection_timings,
            "metric_kinds": selection.metric_kinds,
            "tangent_block_shapes": selection.tangent_block_shapes,
            "tangent_relative_residual": selection.tangent_relative_residual,
            "overlap_applications": selection.overlap_applications,
            "connector_dimension": selection.connector_dimension,
            "fit_iterations": selection.refinement_iterations,
        },
        cbe_expansion_dimension=options.cbe_expansion_dimension,
        cbe_selector="shrewd",
        cbe_preselection_dimension=selection.preselection_dimension,
        cbe_pair_dimension=int(np.prod(layout.merged_shape)),
        cbe_pair_metric_rank=None,
        cbe_tangent_rank=None,
        cbe_projection_iterations=selection.tangent_iterations,
        cbe_projection_converged=(
            selection.tangent_relative_residual <= options.cbe_projection_tolerance),
        cbe_selector_pair_action_count=0,
        cbe_selector_pair_metric_count=selection.pair_metric_count,
        cbe_selector_merged_pair_count=selection.merged_pair_count,
        cbe_preselection_output_size=selection.preselection_output_size,
        cbe_final_output_size=selection.final_output_size,
        cbe_materialized_pair_tensor=False,
        cbe_materialized_pair_metric=False,
        cbe_materialized_tangent_jacobian=False,
        cbe_trim_method="one-site-metric-svd/als",
        cbe_missing_norm=selection.missing_norm,
        cbe_old_energy=old_energy,
    )
    if (
        selection.missing_norm is None
        or selection.missing_norm <= options.cbe_selection_tolerance
        or not any(selection.sector_ranks)
    ):
        started = time.perf_counter()
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
            local_environments=local_environments,
        )
        timings["baseline"] = time.perf_counter() - started
        return replace(
            fallback,
            **common_diagnostics,
            cbe_captured_weight=0.0,
            cbe_selection_loss=selection.loss,
            cbe_trim_loss=0.0,
            cbe_fallback=True,
        )

    started = time.perf_counter()
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

    timings["expanded_solve"] = time.perf_counter() - started
    started = time.perf_counter()
    trim = _directional_one_site_metric_trim(
        state.tensors[left_site],
        state.tensors[right_site],
        effective_trim_metric,
        bond_dimension=original_bond_dimension,
        direction=direction,
        tolerance=options.cbe_selection_tolerance,
        max_iterations=options.cbe_refinement_max_iterations,
        metric_tolerance=options.metric_tolerance,
        layout=layout if options.cbe_conditional_trim else None,
        compression=options.compression,
    )
    state.tensors[left_site] = trim.left_tensor
    state.tensors[right_site] = trim.right_tensor
    common_diagnostics["cbe_trim_metric_kinds"] = trim.metric_kinds
    common_diagnostics["cbe_compression_diagnostics"] = trim.diagnostics
    # Compression may amplify null components before relaxation starts. Check
    # both starts against the incumbent's budget before contracting huge factors.
    norm_limit = 100.0 * np.sqrt(
        np.linalg.norm(original_left) * np.linalg.norm(original_right) / _old_norm)
    trim_is_valid = _within_factor_budget(trim.left_tensor, trim.right_tensor, trim.norm, norm_limit)
    trimmed_energy, trimmed_norm = float("inf"), 0.0
    if trim_is_valid:
        trimmed_energy, trimmed_norm = _streamed_bond_energy(
            hamiltonian_cache, metric_cache, hamiltonian_left, hamiltonian_right,
            metric_left, metric_right, layout, reject_invalid=True,
        )
        trim_is_valid = _within_factor_budget(
            trim.left_tensor, trim.right_tensor, trimmed_norm, norm_limit)
    candidate_left = trim.left_tensor.copy()
    candidate_right = trim.right_tensor.copy()
    candidate_energy, candidate_norm = trimmed_energy, trimmed_norm
    timings["trim"] = time.perf_counter() - started
    refinement_applications = 0
    if options.cbe_energy_refinement_max_iterations:
        refined, refinement_applications = _refine_cbe_candidates(
            state, layout, hamiltonian_cache, metric_cache,
            hamiltonian_left, hamiltonian_right, metric_left, metric_right,
            direction, options, trim, original_left, original_right, common_diagnostics,
            trim_is_valid=trim_is_valid, norm_limit=norm_limit,
        )
        candidate_left, candidate_right = refined.left_tensor, refined.right_tensor
        candidate_energy, candidate_norm = refined.energy, refined.norm
    candidate_is_safe = (
        (site_update.accepted or common_diagnostics.get("cbe_energy_refinement_start") == "incumbent")
        and _within_factor_budget(candidate_left, candidate_right, candidate_norm, norm_limit)
        and np.isfinite(candidate_energy)
        and candidate_energy <= old_energy + options.energy_increase_tolerance
    )
    state.tensors[left_site] = original_left
    state.tensors[right_site] = original_right
    started = time.perf_counter()
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
        local_environments=local_environments,
    )
    timings["baseline"] = time.perf_counter() - started
    common_diagnostics["hamiltonian_applications"] = (
        site_update.hamiltonian_applications + baseline.hamiltonian_applications
        + refinement_applications
    )
    baseline_allowance = _cbe_baseline_allowance(
        old_energy, baseline.energy, options
    )
    accepted = (
        candidate_is_safe
        and _cbe_candidate_is_preferred(
            candidate_energy, old_energy, baseline.energy, options
        )
    )
    if accepted:
        state.tensors[left_site] = candidate_left
        state.tensors[right_site] = candidate_right
        normalization = candidate_norm
        if direction == "lr":
            state.tensors[right_site] /= normalization
        else:
            state.tensors[left_site] /= normalization
        return replace(
            site_update,
            energy=candidate_energy,
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
    *,
    local_environments=None,
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
            local_environments=local_environments,
        )

    from dataclasses import replace

    from .solver import _update_from_cached_environments

    started = time.perf_counter()
    timings = {}
    left_site = layout.left_site
    right_site = left_site + 1
    original_left = state.tensors[left_site].copy()
    original_right = state.tensors[right_site].copy()
    original_bond_dimension = original_left.shape[-1]
    pair_metric = metric_cache.effective_pair_metric(
        metric_left, metric_right, layout
    )
    old_pair = layout.merge(original_left, original_right).reshape(-1)
    old_norm = np.sqrt(float(np.vdot(old_pair, pair_metric @ old_pair).real))
    norm_limit = 100.0 * np.sqrt(
        np.linalg.norm(original_left) * np.linalg.norm(original_right) / old_norm)

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
    timings["selection"] = time.perf_counter() - started
    common_diagnostics = dict(
        cbe_timings=timings,
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
        started = time.perf_counter()
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
            local_environments=local_environments,
        )
        timings["baseline"] = time.perf_counter() - started
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
    timings["selection"] = time.perf_counter() - started
    started = time.perf_counter()
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

    timings["expanded_solve"] = time.perf_counter() - started
    started = time.perf_counter()
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
        compression=options.compression,
    )
    common_diagnostics["cbe_compression_diagnostics"] = trim.diagnostics
    trimmed_pair = layout.merge(trim.left_tensor, trim.right_tensor).reshape(-1)
    normalized_trim_norm = 1.0 if trim.norm > np.finfo(float).tiny else 0.0
    trim_is_valid = _within_factor_budget(
        trim.left_tensor, trim.right_tensor, normalized_trim_norm, norm_limit)
    trimmed_energy = (_pair_rayleigh(trimmed_pair, pair_action, pair_metric)
                      if trim_is_valid else float("inf"))
    trim_is_valid = trim_is_valid and np.isfinite(trimmed_energy)
    candidate_left = trim.left_tensor.copy()
    candidate_right = trim.right_tensor.copy()
    # metric_trim_pair already normalizes its returned factors; trim.norm is
    # the pre-normalization norm of the compression initializer.
    candidate_energy = trimmed_energy
    candidate_norm = normalized_trim_norm
    timings["trim"] = time.perf_counter() - started
    refinement_applications = 0
    if options.cbe_energy_refinement_max_iterations:
        refined, refinement_applications = _refine_cbe_candidates(
            state, layout, hamiltonian_cache, metric_cache,
            hamiltonian_left, hamiltonian_right, metric_left, metric_right,
            direction, options, trim, original_left, original_right, common_diagnostics,
            trim_is_valid=trim_is_valid, norm_limit=norm_limit,
        )
        candidate_left, candidate_right = refined.left_tensor, refined.right_tensor
        candidate_energy, candidate_norm = refined.energy, refined.norm
    candidate_is_safe = (
        (site_update.accepted or common_diagnostics.get("cbe_energy_refinement_start") == "incumbent")
        and _within_factor_budget(candidate_left, candidate_right, candidate_norm, norm_limit)
        and np.isfinite(candidate_energy)
        and candidate_energy <= missing.energy + options.energy_increase_tolerance
    )
    state.tensors[left_site] = original_left
    state.tensors[right_site] = original_right
    started = time.perf_counter()
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
        local_environments=local_environments,
    )
    timings["baseline"] = time.perf_counter() - started
    common_diagnostics["hamiltonian_applications"] = (
        site_update.hamiltonian_applications + baseline.hamiltonian_applications
        + refinement_applications
    )
    baseline_allowance = _cbe_baseline_allowance(
        missing.energy, baseline.energy, options
    )
    accepted = (
        candidate_is_safe
        and _cbe_candidate_is_preferred(
            candidate_energy, missing.energy, baseline.energy, options
        )
    )
    if accepted:
        state.tensors[left_site] = candidate_left
        state.tensors[right_site] = candidate_right
        if direction == "lr":
            state.tensors[right_site] /= candidate_norm
        else:
            state.tensors[left_site] /= candidate_norm
        return replace(
            site_update,
            energy=candidate_energy,
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


def _baseline_metric_environment(cache, environment, cut, direction):
    # A certified diagonal is a rounded representation of the contracted
    # Gram. Replacing the baseline's raw Gram by it can alter near-null solves.
    record = cache.saved_canonical(cut, direction)
    return None if record is not None and environment is record.environment else environment


def _is_cbe_numerical_failure(error):
    """Do not hide invalid options, tensor shapes, or other programming errors."""
    if isinstance(error, (FloatingPointError, np.linalg.LinAlgError, OverflowError,
                          ZeroDivisionError)):
        return True
    return isinstance(error, ValueError) and str(error) in {
        "cannot evaluate an operator on a zero LETTA state.",
        "cannot shift the gauge of a zero LETTA tensor.",
        "the local LETTA overlap metric has zero rank.",
        "the two-site LETTA overlap metric has zero rank.",
        "the optimized local LETTA tensor has zero norm.",
        "the represented LETTA pair has zero metric norm.",
        "the streamed LETTA bond has zero metric norm.",
        "a fixed-rank LETTA pair has zero physical norm.",
        "the initial fixed-rank LETTA pair has zero norm.",
        "array must not contain infs or NaNs",
    }


def _checked_cbe_energy(state, mpo, ceiling, tolerance):
    if any(not np.all(np.isfinite(a)) for a in state.tensors):
        raise FloatingPointError("nonfinite CBE tensor")
    value = np.real_if_close(state.expectation(mpo))
    if np.iscomplexobj(value) or not np.isfinite(value):
        raise FloatingPointError("nonfinite or complex fresh CBE energy")
    energy = float(value)
    if energy > ceiling + tolerance:
        raise FloatingPointError(
            f"fresh CBE energy increased: {ceiling:.17g} -> {energy:.17g}")
    return energy


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
    """Check each complete step; retry failed CBE steps with ordinary one-site.

    A transaction includes the gauge shift, which also changes a neighboring
    tensor. Recovery rebuilds both environments from the restored tensors.
    Subsequent bonds continue to use CBE. No dense physical state is formed.
    """
    from dataclasses import replace
    from .solver import LETTASiteUpdate, _shift_gauge_and_extend_metric, _update_from_cached_environments

    hcache, mcache = hamiltonian_cache, metric_cache
    henv, menv = hamiltonian_environments, metric_environments
    lr = direction == "lr"
    sites = range(state.nsites) if lr else range(state.nsites - 1, -1, -1)
    terminal = state.nsites - 1 if lr else 0
    boundary_cut = 0 if lr else state.nsites
    henv[boundary_cut], menv[boundary_cut] = hcache.scalar_boundary(), mcache.scalar_boundary()
    energy = _checked_cbe_energy(state, hcache.mpo, np.inf, 0.)
    # A per-step tolerance must not accumulate into a sweep-size energy rise.
    sweep_start_energy = energy
    updates = []

    def restore(tensors):
        state.tensors = [a.copy() for a in tensors]
        # Discard certificates made by the rejected trial, including entries
        # that a nested solve may have removed or replaced.
        state._canonical_norm_environments = {}

    def rebuild(site):
        hl, hr = hcache.build_left_environments(), hcache.build_right_environments()
        ml, mr = mcache.build_left_environments(), mcache.build_right_environments()
        henv[:] = hl[:site + 1] + hr[site + 1:]
        menv[:] = ml[:site + 1] + mr[site + 1:]

    def ordinary(site):
        return _update_from_cached_environments(
            state, site, hcache, mcache, henv[site], henv[site + 1],
            menv[site], menv[site + 1], options)

    def advance(site, *, gauge=True):
        incoming = menv[site if lr else site + 1]
        if gauge:
            outgoing_metric = _shift_gauge_and_extend_metric(
                state, site, direction, options, mcache, incoming)
        else:
            extend = mcache.extend_left if lr else mcache.extend_right
            outgoing_metric = extend(incoming, site)
        extend = hcache.extend_left if lr else hcache.extend_right
        outgoing_hamiltonian = extend(henv[site if lr else site + 1], site)
        return outgoing_hamiltonian, outgoing_metric

    for site in sites:
        before = [a.copy() for a in state.tensors]
        ceiling = min(energy, sweep_start_energy)
        try:
            if site == terminal:
                update = ordinary(site)
            else:
                pair_site = site if lr else site - 1
                layout = LETTAPairLayout.from_state(state, pair_site)
                local_cut = pair_site + 1
                update = _cbe_bond_update(
                    state, layout, hcache, mcache,
                    henv[pair_site], henv[pair_site + 2],
                    menv[pair_site], menv[pair_site + 2], direction, options,
                    local_environments=(henv[local_cut], _baseline_metric_environment(
                        mcache, menv[local_cut], local_cut, "rl" if lr else "lr")))
            outgoing_h, outgoing_m = advance(site)
            energy = _checked_cbe_energy(state, hcache.mpo, ceiling,
                                         options.energy_increase_tolerance)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            restore(before)
            if not _is_cbe_numerical_failure(error):
                raise
            reason = f"{type(error).__name__}: {error}"
            try:
                rebuild(site)
                update = ordinary(site)
                ordinary_energy = _checked_cbe_energy(
                    state, hcache.mpo, ceiling, options.energy_increase_tolerance)
                # If it is the gauge itself that is unstable, retain the
                # validated ordinary update and advance without that gauge.
                ordinary_tensors = [a.copy() for a in state.tensors]
                try:
                    outgoing_h, outgoing_m = advance(site)
                    energy = _checked_cbe_energy(
                        state, hcache.mpo, min(ceiling, ordinary_energy),
                        options.energy_increase_tolerance)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as gauge_error:
                    restore(ordinary_tensors)
                    if not _is_cbe_numerical_failure(gauge_error):
                        raise
                    rebuild(site)
                    outgoing_h, outgoing_m = advance(site, gauge=False)
                    energy = _checked_cbe_energy(
                        state, hcache.mpo, ceiling, options.energy_increase_tolerance)
                    reason += f"; skipped unstable gauge: {gauge_error}"
                update = replace(update, energy=energy, cbe_fallback=True,
                                 cbe_baseline_selected=True, cbe_baseline_energy=energy,
                                 cbe_recovery_reason=reason)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as fallback_error:
                restore(before)
                if not _is_cbe_numerical_failure(fallback_error):
                    raise
                # Even the ordinary solve can be ill-conditioned. Keep the
                # last valid tensors, advance without a gauge, and mark this
                # step unresolved so it cannot cause false convergence.
                rebuild(site)
                outgoing_h, outgoing_m = advance(site, gauge=False)
                energy = _checked_cbe_energy(state, hcache.mpo, ceiling,
                                             options.energy_increase_tolerance)
                reason += f"; one-site also rejected: {fallback_error}"
                update = LETTASiteUpdate(
                    site=site, local_energy=energy, energy=energy,
                    metric_rank=0, local_dimension=state.tensors[site].size,
                    residual_norm=float('inf'), accepted=False, cbe_fallback=True,
                    cbe_recovery_reason=reason, cbe_recovery_rejected=True)
            if options.verbosity:
                action = "kept previous state" if update.cbe_recovery_rejected else "recovered with one-site"
                print(f"CBE site {site} ({direction}): {action}; {reason}",
                      flush=True)
        cut = site + 1 if lr else site
        henv[cut], menv[cut] = outgoing_h, outgoing_m
        updates.append(update)

    # Return the validated energy also to benchmark observers, which run before
    # the driver's sweep guard. They must never see the rejected trial state.
    return tuple(updates), energy, henv, menv

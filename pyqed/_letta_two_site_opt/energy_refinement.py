"""Fixed-rank pair-energy refinement for split LETTA tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .._letta_one_site_opt.contractions import _equilibrated_metric_factors
from .._letta_one_site_opt.solver import _lowest_generalized_eigenpair
from .pair import LETTAPairLayout, LETTASplit


@dataclass(frozen=True)
class LETTAEnergyRefinement:
    left_tensor: np.ndarray
    right_tensor: np.ndarray
    initial_energy: float
    energy: float
    iterations: int
    accepted_substeps: int
    max_factor_norm: float
    coupled_iterations: int = 0
    coupled_accepted_steps: int = 0


def _rayleigh(vector, action, metric):
    vector = np.asarray(vector)
    denominator = np.vdot(vector, metric @ vector)
    if np.real(denominator) <= np.finfo(float).tiny:
        raise ValueError("a fixed-rank LETTA pair has zero physical norm.")
    return float(np.real(np.vdot(vector, action(vector)) / denominator))


def _balance_pair(left_tensor, right_tensor):
    left_norm = np.linalg.norm(left_tensor)
    right_norm = np.linalg.norm(right_tensor)
    if left_norm <= np.finfo(float).tiny or right_norm <= np.finfo(float).tiny:
        return left_tensor, right_tensor
    scale = np.sqrt(right_norm / left_norm)
    return left_tensor * scale, right_tensor / scale


def _factor_frame(layout, left_tensor, right_tensor, side, indices=None):
    if side == "left":
        shape = left_tensor.shape
    elif side == "right":
        shape = right_tensor.shape
    else:
        raise ValueError("side must be 'left' or 'right'.")
    layout._validate_tensor_shapes(left_tensor, right_tensor)
    size = int(np.prod(shape))
    dtype = np.result_type(left_tensor, right_tensor)
    if indices is None:
        indices = np.arange(size)
        basis = np.eye(size, dtype=dtype)
    else:
        indices = np.asarray(indices, dtype=int)
        basis = np.zeros((size, indices.size), dtype=dtype)
        basis[indices] = np.eye(indices.size, dtype=dtype)
    basis = basis.reshape(shape + (indices.size,))
    left_labels, right_labels, output_labels = layout._contraction_labels()
    batch = max(left_labels + right_labels + output_labels) + 1
    # Contract every basis vector together; each column is the same merge as
    # before. Keep the original contiguous frame layout for subsequent BLAS.
    if side == "left":
        frame = np.einsum(
            basis, left_labels + [batch], right_tensor, right_labels,
            output_labels + [batch], optimize=True,
        )
    else:
        frame = np.einsum(
            left_tensor, left_labels, basis, right_labels + [batch],
            output_labels + [batch], optimize=True,
        )
    return np.ascontiguousarray(frame.reshape(-1, indices.size)), indices


def _lowest_factor(frame, action, metric, metric_tolerance):
    applied = action(frame)
    hamiltonian = frame.conj().T @ applied
    overlap = frame.conj().T @ (metric @ frame)
    _energy, vector, _rank, _residual = _lowest_generalized_eigenpair(
        hamiltonian,
        overlap,
        metric_tolerance,
    )
    return vector


def _spectral_metric_descent(metric, gradient, relative_cutoff):
    """Solve an already equilibrated metric, retaining its numerical nullspace."""
    try:
        values, vectors = np.linalg.eigh(metric)
    except np.linalg.LinAlgError:
        values, vectors = linalg.eigh(metric, driver="evr", check_finite=True)
    floor = max(relative_cutoff, np.finfo(metric.real.dtype).eps * len(gradient))
    keep = values > floor * max(float(values[-1]) if values.size else 0., 1.)
    coordinates = vectors[:, keep].conj().T @ gradient
    direction = -(vectors[:, keep] @ (coordinates / values[keep]))
    return direction, float(np.sum(np.abs(coordinates)**2 / values[keep]).real)


def _metric_descent(metric, gradient, relative_cutoff, *, overlap=None):
    """Return the supported natural direction and its squared residual."""
    basis, _ = _equilibrated_metric_factors(metric, relative_cutoff)
    coordinates = basis.conj().T @ gradient
    if overlap is None:
        direction = -(basis @ coordinates)
        residual_squared = float(np.vdot(coordinates, coordinates).real)
    else:
        # Remove normalization only after scaling the raw Gram matrix. A tiny
        # diagonal created by subtraction can be roundoff, not a new coordinate.
        reduced = basis.conj().T @ metric @ basis
        projected = basis.conj().T @ overlap
        reduced -= np.outer(projected, projected.conj())
        reduced = 0.5 * (reduced + reduced.conj().T)
        delta, residual_squared = _spectral_metric_descent(reduced, coordinates, relative_cutoff)
        direction = basis @ delta
    return direction, residual_squared


def _coupled_factor_descent(
    layout, left, right, action, metric, *, max_iterations, metric_tolerance,
    coupling_ratio, norm_limit, left_indices=None, right_indices=None,
):
    """Backtrack simultaneous factor changes in the existing pair environment.

    The tangent metric removes the normalization direction. Its small positive
    eigenvalues reveal directions hidden by separate A/B minimizations. Only
    actual fixed-rank factors are evaluated and returned, never a tangent-state
    projection or an enlarged physical state.
    """
    left, right = left.copy(), right.copy()
    theta = layout.merge(left, right).reshape(-1)
    right /= np.sqrt(np.vdot(theta, metric @ theta).real)
    left, right = _balance_pair(left, right)
    energy = _rayleigh(layout.merge(left, right).reshape(-1), action, metric)
    iterations = accepted_steps = 0
    for iteration in range(max_iterations):
        theta = layout.merge(left, right).reshape(-1)
        n_theta, h_theta = metric @ theta, action(theta)
        norm = float(np.vdot(theta, n_theta).real)
        frame_a, indices_a = _factor_frame(layout, left, right, "left", left_indices)
        frame_b, indices_b = _factor_frame(layout, left, right, "right", right_indices)
        frame = np.column_stack((frame_a, frame_b))
        overlap = frame.conj().T @ n_theta / norm
        tangent_metric = frame.conj().T @ (metric @ frame) / norm
        tangent_metric = 0.5 * (tangent_metric + tangent_metric.conj().T)
        gradient = frame.conj().T @ (h_theta - energy * n_theta) / norm
        delta, residual_squared = _metric_descent(
            tangent_metric, gradient, metric_tolerance, overlap=overlap)
        iterations = iteration + 1
        noise_floor = max(1e-12, np.finfo(tangent_metric.real.dtype).eps * gradient.size)
        noise_floor *= max(1.0, abs(energy))
        if residual_squared <= noise_floor**2:
            break
        boundary = indices_a.size
        if iteration == 0 and coupling_ratio > 0.0:
            independent_squared = 0.0
            for selected in (slice(0, boundary), slice(boundary, None)):
                _, squared = _metric_descent(
                    tangent_metric[selected, selected], gradient[selected], metric_tolerance,
                    overlap=overlap[selected],
                )
                independent_squared += squared
            if residual_squared <= coupling_ratio**2 * max(independent_squared, noise_floor**2):
                break
        delta_a = np.zeros(left.size, dtype=delta.dtype)
        delta_b = np.zeros(right.size, dtype=delta.dtype)
        delta_a[indices_a], delta_b[indices_b] = delta[:boundary], delta[boundary:]
        delta_a, delta_b = delta_a.reshape(left.shape), delta_b.reshape(right.shape)
        slope = float(2 * np.vdot(gradient, delta).real)
        if not np.isfinite(slope) or slope >= 0.0:
            break
        relative_step = max(np.linalg.norm(delta_a) / np.linalg.norm(left),
                            np.linalg.norm(delta_b) / np.linalg.norm(right))
        step = min(1.0, 0.25 / max(relative_step, np.finfo(float).tiny))
        accepted = False
        for _ in range(50):
            proposed_a, proposed_b = left + step * delta_a, right + step * delta_b
            proposed = layout.merge(proposed_a, proposed_b).reshape(-1)
            proposed_norm = float(np.vdot(proposed, metric @ proposed).real)
            if np.isfinite(proposed_norm) and proposed_norm > np.finfo(float).tiny:
                proposed_b /= np.sqrt(proposed_norm)
                proposed_a, proposed_b = _balance_pair(proposed_a, proposed_b)
                max_norm = max(np.linalg.norm(proposed_a), np.linalg.norm(proposed_b))
                if np.isfinite(max_norm) and max_norm <= norm_limit:
                    proposed = layout.merge(proposed_a, proposed_b).reshape(-1)
                    proposed_energy = _rayleigh(proposed, action, metric)
                    if (np.isfinite(proposed_energy) and proposed_energy < energy
                            and proposed_energy <= energy + 1e-4 * step * slope):
                        accepted = True
                        break
            step *= 0.5
        if not accepted:
            break
        left, right, energy = proposed_a, proposed_b, proposed_energy
        accepted_steps += 1
    return left, right, energy, iterations, accepted_steps


def energy_refine_split(
    layout,
    initial,
    action,
    metric,
    *,
    max_iterations=1,
    tolerance=1.0e-10,
    metric_tolerance=1.0e-12,
    energy_increase_tolerance=1.0e-10,
    max_factor_norm_growth=100.0,
    left_indices=None,
    right_indices=None,
    coupled_max_iterations=0,
    coupled_metric_tolerance=1.0e-12,
    coupled_activation_ratio=5.0,
):
    """Minimize fixed-rank pair energy, optionally correcting coupled directions."""

    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    if not isinstance(initial, LETTASplit):
        raise TypeError("initial must be a LETTASplit.")
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    metric_tolerance = float(metric_tolerance)
    energy_increase_tolerance = float(energy_increase_tolerance)
    max_factor_norm_growth = float(max_factor_norm_growth)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if tolerance <= 0.0 or metric_tolerance <= 0.0:
        raise ValueError("energy-refinement tolerances must be positive.")
    if energy_increase_tolerance < 0.0:
        raise ValueError("energy_increase_tolerance must be nonnegative.")
    if max_factor_norm_growth < 1.0:
        raise ValueError("max_factor_norm_growth must be at least one.")
    if not isinstance(coupled_max_iterations, (int, np.integer)) or coupled_max_iterations < 0:
        raise ValueError("coupled_max_iterations must be a nonnegative integer.")
    if not np.isfinite(coupled_metric_tolerance) or not 0 < coupled_metric_tolerance < 1:
        raise ValueError("coupled_metric_tolerance must be finite and between zero and one.")
    if not np.isfinite(coupled_activation_ratio) or coupled_activation_ratio < 0:
        raise ValueError("coupled_activation_ratio must be finite and nonnegative.")

    left_tensor = initial.left_tensor.copy()
    right_tensor = initial.right_tensor.copy()
    merged = layout.merge(left_tensor, right_tensor).reshape(-1)
    norm = float(np.real(np.vdot(merged, metric @ merged)))
    if norm <= np.finfo(float).tiny:
        raise ValueError("the initial fixed-rank LETTA pair has zero norm.")
    right_tensor /= np.sqrt(norm)
    left_tensor, right_tensor = _balance_pair(left_tensor, right_tensor)
    merged = layout.merge(left_tensor, right_tensor).reshape(-1)
    initial_energy = _rayleigh(merged, action, metric)
    energy = initial_energy
    baseline_norm = max(
        float(np.linalg.norm(left_tensor)),
        float(np.linalg.norm(right_tensor)),
    )
    norm_limit = max_factor_norm_growth * baseline_norm
    accepted_substeps = 0
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iteration_energy = energy
        for side in ("left", "right"):
            factor_indices = left_indices if side == "left" else right_indices
            frame, factor_indices = _factor_frame(
                layout,
                left_tensor,
                right_tensor,
                side,
                factor_indices,
            )
            candidate = _lowest_factor(
                frame,
                action,
                metric,
                metric_tolerance,
            )
            if side == "left":
                proposed_left = np.zeros_like(left_tensor).reshape(-1)
                proposed_left[factor_indices] = candidate
                proposed_left = proposed_left.reshape(left_tensor.shape)
                proposed_right = right_tensor.copy()
            else:
                proposed_left = left_tensor.copy()
                proposed_right = np.zeros_like(right_tensor).reshape(-1)
                proposed_right[factor_indices] = candidate
                proposed_right = proposed_right.reshape(right_tensor.shape)
            proposed_left, proposed_right = _balance_pair(
                proposed_left,
                proposed_right,
            )
            proposed_max_norm = max(
                float(np.linalg.norm(proposed_left)),
                float(np.linalg.norm(proposed_right)),
            )
            if not np.isfinite(proposed_max_norm) or proposed_max_norm > norm_limit:
                continue
            proposed = layout.merge(
                proposed_left, proposed_right
            ).reshape(-1)
            proposed_norm = float(np.vdot(proposed, metric @ proposed).real)
            if not np.isfinite(proposed_norm) or proposed_norm <= np.finfo(float).tiny:
                # The current pair remains valid even if a numerically
                # singular factor solve proposes an unusable tensor.
                continue
            proposed_energy = _rayleigh(proposed, action, metric)
            if proposed_energy <= energy + energy_increase_tolerance:
                left_tensor = proposed_left
                right_tensor = proposed_right
                energy = proposed_energy
                accepted_substeps += 1
        iterations = iteration
        improvement = iteration_energy - energy
        if improvement <= tolerance * max(1.0, abs(iteration_energy)):
            break

    coupled_iterations = coupled_accepted_steps = 0
    if coupled_max_iterations:
        proposed_left, proposed_right, proposed_energy, coupled_iterations, coupled_accepted_steps = (
            _coupled_factor_descent(
                layout, left_tensor, right_tensor, action, metric,
                max_iterations=coupled_max_iterations,
                metric_tolerance=coupled_metric_tolerance,
                coupling_ratio=coupled_activation_ratio,
                norm_limit=norm_limit,
                left_indices=left_indices, right_indices=right_indices,
            )
        )
        # Normalization may change the last bit. Preserve the complete lower
        # state, and never compound the original per-start norm-growth budget.
        if coupled_accepted_steps and proposed_energy <= energy:
            left_tensor, right_tensor, energy = proposed_left, proposed_right, proposed_energy
        else:
            coupled_accepted_steps = 0

    max_factor_norm = max(
        float(np.linalg.norm(left_tensor)),
        float(np.linalg.norm(right_tensor)),
    )
    return LETTAEnergyRefinement(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        initial_energy=initial_energy,
        energy=energy,
        iterations=iterations + coupled_iterations,
        accepted_substeps=accepted_substeps + coupled_accepted_steps,
        max_factor_norm=max_factor_norm,
        coupled_iterations=coupled_iterations,
        coupled_accepted_steps=coupled_accepted_steps,
    )

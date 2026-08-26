"""Fixed-rank pair-energy refinement for split LETTA tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

        def merge(vector):
            return layout.merge(vector.reshape(shape), right_tensor)

    elif side == "right":
        shape = right_tensor.shape

        def merge(vector):
            return layout.merge(left_tensor, vector.reshape(shape))

    else:
        raise ValueError("side must be 'left' or 'right'.")
    size = int(np.prod(shape))
    if indices is None:
        indices = np.arange(size)
    else:
        indices = np.asarray(indices, dtype=int)
    identity = np.eye(
        indices.size,
        dtype=np.result_type(left_tensor, right_tensor),
    )
    columns = []
    for column in range(identity.shape[1]):
        full = np.zeros(size, dtype=identity.dtype)
        full[indices] = identity[:, column]
        columns.append(merge(full).reshape(-1))
    return np.column_stack(columns), indices


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
):
    """Alternately minimize pair energy over two fixed-rank factors."""

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

    max_factor_norm = max(
        float(np.linalg.norm(left_tensor)),
        float(np.linalg.norm(right_tensor)),
    )
    return LETTAEnergyRefinement(
        left_tensor=left_tensor,
        right_tensor=right_tensor,
        initial_energy=initial_energy,
        energy=energy,
        iterations=iterations,
        accepted_substeps=accepted_substeps,
        max_factor_norm=max_factor_norm,
    )

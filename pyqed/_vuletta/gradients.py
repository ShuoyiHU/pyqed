"""Analytic transfer-response gradients for uniform LETTA states."""

from __future__ import annotations

import numpy as np

from .operators import (
    _as_two_site_operator,
    _dense_transfer_matrix,
    transfer_data,
)
from .state import UniformLETTA


def _structured_direction(direction):
    bond_dim, physical_dim, _physical_dim, _right_bond = direction.shape
    effective_bond = bond_dim * physical_dim
    structured = np.zeros(
        (effective_bond, physical_dim, effective_bond),
        dtype=direction.dtype,
    )
    for left in range(bond_dim):
        for previous in range(physical_dim):
            left_combined = left * physical_dim + previous
            for current in range(physical_dim):
                for right in range(bond_dim):
                    right_combined = right * physical_dim + current
                    structured[left_combined, current, right_combined] = (
                        direction[left, previous, current, right]
                    )
    return structured


def _transfer_direction(tensor, direction, value):
    return np.einsum(
        "aib,bc,dic->ad",
        direction,
        value,
        tensor.conj(),
        optimize=True,
    ) + np.einsum(
        "aib,bc,dic->ad",
        tensor,
        value,
        direction.conj(),
        optimize=True,
    )


def _ket_direction(tensor, direction, value):
    return np.einsum(
        "aib,bc,dic->ad",
        direction,
        value,
        tensor.conj(),
        optimize=True,
    )


def _bra_direction(tensor, direction, value):
    return np.einsum(
        "aib,bc,dic->ad",
        tensor,
        value,
        direction.conj(),
        optimize=True,
    )


def _double_direction(bra_direction, ket_direction, value):
    return np.einsum(
        "aib,bc,dic->ad",
        ket_direction,
        value,
        bra_direction.conj(),
        optimize=True,
    )


def _two_site_action(tensor, hamiltonian, value, pair=None):
    if pair is None:
        pair = np.einsum(
            "aib,bjc->aijc",
            tensor,
            tensor,
            optimize=True,
        )
    return np.einsum(
        "ijkl,aklc,cd,bijd->ab",
        hamiltonian,
        pair,
        value,
        pair.conj(),
        optimize=True,
    )


def _two_site_direction(
    tensor,
    direction,
    hamiltonian,
    value,
    pair=None,
):
    if pair is None:
        pair = np.einsum(
            "aib,bjc->aijc",
            tensor,
            tensor,
            optimize=True,
        )
    pair_direction = np.einsum(
        "aib,bjc->aijc",
        direction,
        tensor,
        optimize=True,
    ) + np.einsum(
        "aib,bjc->aijc",
        tensor,
        direction,
        optimize=True,
    )
    return np.einsum(
        "ijkl,aklc,cd,bijd->ab",
        hamiltonian,
        pair_direction,
        value,
        pair.conj(),
        optimize=True,
    ) + np.einsum(
        "ijkl,aklc,cd,bijd->ab",
        hamiltonian,
        pair,
        value,
        pair_direction.conj(),
        optimize=True,
    )


def _reduced_resolvent(tensor, left, right):
    transfer = _dense_transfer_matrix(tensor)
    left_vector = left.reshape(-1)
    right_vector = right.reshape(-1)
    overlap = np.vdot(left_vector, right_vector)
    if not np.allclose(overlap, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("transfer fixed points must have unit overlap.")
    projector = np.outer(right_vector, np.conj(left_vector))
    identity = np.eye(transfer.shape[0], dtype=transfer.dtype)
    return np.linalg.inv(identity - transfer + projector) - projector


def _parameter_directions(state, data, real):
    tensor = state.tensor
    norm = np.linalg.norm(tensor)
    normalized_tensor = tensor / norm
    coordinate_count = tensor.size if real else 2 * tensor.size
    packed_tensor = (
        np.real(normalized_tensor).reshape(-1)
        if real
        else np.concatenate(
            [
                np.real(normalized_tensor).reshape(-1),
                np.imag(normalized_tensor).reshape(-1),
            ]
        )
    )
    scale = np.sqrt(data.original_eigenvalue)
    directions = []
    for coordinate in range(coordinate_count):
        direction = np.zeros(tensor.shape, dtype=tensor.dtype)
        flat_coordinate = coordinate % tensor.size
        direction.reshape(-1)[flat_coordinate] = (
            1.0 if coordinate < tensor.size else 1j
        )
        direction = (direction - packed_tensor[coordinate] * normalized_tensor) / norm
        structured_direction = _structured_direction(direction) / scale
        eigenvalue_change = np.vdot(
            data.left_fixed_point.reshape(-1),
            _transfer_direction(
                data.structured_tensor,
                structured_direction,
                data.right_fixed_point,
            ).reshape(-1),
        )
        structured_direction -= (
            0.5 * float(np.real(eigenvalue_change)) * data.structured_tensor
        )
        directions.append(structured_direction)
    return tuple(directions)


def tangent_gram_matrix(state, *, real=None):
    """Return the connected uniform-state tangent metric in real coordinates.

    Radial normalization and LETTA gauge directions are null directions.  For
    a complex tensor, coordinates are ordered as all real parts followed by
    all imaginary parts, matching the solver's parameter packing.
    """

    if not isinstance(state, UniformLETTA):
        raise TypeError("state must be a UniformLETTA.")
    if real is None:
        real = not np.iscomplexobj(state.tensor)
    real = bool(real)
    if real and np.max(np.abs(np.imag(state.tensor))) > 1.0e-12:
        raise ValueError("a complex state cannot use a real tangent metric.")

    data = transfer_data(state.normalized_parameters())
    tensor = data.structured_tensor
    left = data.left_fixed_point
    right = data.right_fixed_point
    resolvent = _reduced_resolvent(tensor, left, right)
    directions = _parameter_directions(state, data, real)
    coordinate_count = len(directions)
    metric = np.empty((coordinate_count, coordinate_count), dtype=float)

    ket_sources = [
        _ket_direction(tensor, direction, right) for direction in directions
    ]
    bra_sources = [
        _bra_direction(tensor, direction, right) for direction in directions
    ]
    ket_parallel = [np.vdot(left, source) for source in ket_sources]
    bra_parallel = [np.vdot(left, source) for source in bra_sources]
    ket_responses = [
        (resolvent @ source.reshape(-1)).reshape(right.shape)
        for source in ket_sources
    ]
    bra_responses = [
        (resolvent @ source.reshape(-1)).reshape(right.shape)
        for source in bra_sources
    ]

    for bra_index, bra_variation in enumerate(directions):
        for ket_index, ket_variation in enumerate(directions):
            overlap = np.vdot(
                left,
                _double_direction(bra_variation, ket_variation, right),
            )
            overlap -= bra_parallel[bra_index] * ket_parallel[ket_index]
            overlap += np.vdot(
                left,
                _bra_direction(
                    tensor,
                    bra_variation,
                    ket_responses[ket_index],
                ),
            )
            overlap += np.vdot(
                left,
                _ket_direction(
                    tensor,
                    ket_variation,
                    bra_responses[bra_index],
                ),
            )
            metric[bra_index, ket_index] = float(np.real(overlap))

    return 0.5 * (metric + metric.T)


def natural_gradient(gradient, metric, *, rcond=1.0e-10):
    """Solve the singular tangent-metric equation with a pseudoinverse."""

    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    metric = np.asarray(metric, dtype=float)
    if metric.shape != (gradient.size, gradient.size):
        raise ValueError("metric and gradient dimensions do not agree.")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    largest = max(float(eigenvalues[-1]), 0.0)
    cutoff = max(rcond * largest, np.finfo(float).eps)
    retained = eigenvalues > cutoff
    if not np.any(retained):
        raise ValueError("the LETTA tangent metric has zero numerical rank.")
    projected = eigenvectors[:, retained].T @ gradient
    direction = eigenvectors[:, retained] @ (projected / eigenvalues[retained])
    residual = float(np.sqrt(max(np.dot(gradient, direction), 0.0)))
    return direction, residual, int(np.count_nonzero(retained))


def _directional_energy_derivative(
    tensor,
    direction,
    hamiltonian,
    left,
    right,
    resolvent,
    local_action,
    pair,
):
    transfer_change_on_right = _transfer_direction(
        tensor,
        direction,
        right,
    )
    response = _two_site_direction(
        tensor,
        direction,
        hamiltonian,
        right,
        pair,
    )
    left_environment = (
        resolvent @ local_action.reshape(-1)
    ).reshape(right.shape)
    right_environment = (
        resolvent @ transfer_change_on_right.reshape(-1)
    ).reshape(right.shape)
    response += _transfer_direction(
        tensor,
        direction,
        left_environment,
    )
    response += _two_site_action(
        tensor,
        hamiltonian,
        right_environment,
        pair,
    )
    derivative = np.vdot(left.reshape(-1), response.reshape(-1))
    if abs(np.imag(derivative)) > 1.0e-8:
        raise ValueError("the Hermitian energy derivative acquired an imaginary part.")
    return float(np.real(derivative))


def energy_and_gradient(state, hamiltonian):
    """Return thermodynamic energy and its analytic LETTA parameter gradient.

    For a complex tensor, the returned complex gradient g uses the convention
    delta_e = real(vdot(g, delta_T)).
    """

    if not isinstance(state, UniformLETTA):
        raise TypeError("state must be a UniformLETTA.")
    hamiltonian = _as_two_site_operator(hamiltonian, state.physical_dim)
    matrix = hamiltonian.reshape(state.physical_dim**2, state.physical_dim**2)
    if not np.allclose(matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("the Hamiltonian must be Hermitian.")

    data = transfer_data(state)
    tensor = data.structured_tensor
    left = data.left_fixed_point
    right = data.right_fixed_point
    pair = np.einsum(
        "aib,bjc->aijc",
        tensor,
        tensor,
        optimize=True,
    )
    local_action = _two_site_action(
        tensor,
        hamiltonian,
        right,
        pair,
    )
    energy = np.vdot(left.reshape(-1), local_action.reshape(-1))
    if abs(np.imag(energy)) > 1.0e-9:
        raise ValueError("a Hermitian LETTA energy acquired a nonreal value.")

    resolvent = _reduced_resolvent(tensor, left, right)
    scale = np.sqrt(data.original_eigenvalue)
    gradient = np.zeros(
        state.tensor.shape,
        dtype=complex if np.iscomplexobj(state.tensor) else float,
    )
    flat_gradient = gradient.reshape(-1)
    for flat_index in range(state.tensor.size):
        direction = np.zeros(state.tensor.shape, dtype=state.tensor.dtype)
        direction.reshape(-1)[flat_index] = 1.0
        structured_direction = _structured_direction(direction) / scale
        relative_eigenvalue_change = np.vdot(
            left.reshape(-1),
            _transfer_direction(tensor, structured_direction, right).reshape(-1),
        )
        if abs(np.imag(relative_eigenvalue_change)) > 1.0e-9:
            raise ValueError(
                "the transfer spectral-radius derivative acquired "
                "an imaginary part."
            )
        normalized_direction = (
            structured_direction
            - 0.5 * float(np.real(relative_eigenvalue_change)) * tensor
        )
        real_derivative = _directional_energy_derivative(
            tensor,
            normalized_direction,
            hamiltonian,
            left,
            right,
            resolvent,
            local_action,
            pair,
        )
        if not np.iscomplexobj(state.tensor):
            flat_gradient[flat_index] = real_derivative
            continue

        imaginary_direction = 1j * _structured_direction(direction) / scale
        imaginary_eigenvalue_change = np.vdot(
            left.reshape(-1),
            _transfer_direction(tensor, imaginary_direction, right).reshape(-1),
        )
        normalized_imaginary_direction = (
            imaginary_direction
            - 0.5 * float(np.real(imaginary_eigenvalue_change)) * tensor
        )
        imaginary_derivative = _directional_energy_derivative(
            tensor,
            normalized_imaginary_direction,
            hamiltonian,
            left,
            right,
            resolvent,
            local_action,
            pair,
        )
        flat_gradient[flat_index] = real_derivative + 1j * imaginary_derivative

    return float(np.real(energy)), gradient


def energy_gradient(state, hamiltonian):
    """Return the analytic gradient of the thermodynamic energy density."""

    _energy, gradient = energy_and_gradient(state, hamiltonian)
    return gradient

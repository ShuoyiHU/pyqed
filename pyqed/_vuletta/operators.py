"""Thermodynamic contractions for a uniform LETTA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import UniformLETTA


def _as_two_site_operator(operator, physical_dim):
    array = np.asarray(operator)
    if array.ndim == 2:
        expected = physical_dim * physical_dim
        if array.shape != (expected, expected):
            raise ValueError(f"a two-site operator must have shape ({expected}, {expected}).")
        return array.reshape(
            physical_dim,
            physical_dim,
            physical_dim,
            physical_dim,
        )
    expected = (physical_dim,) * 4
    if array.shape != expected:
        raise ValueError(f"a rank-four two-site operator must have shape {expected}.")
    return array


def _positive_eigenmatrix(vector, dimension):
    matrix = vector.reshape(dimension, dimension)
    trace = np.trace(matrix)
    if abs(trace) > 1.0e-14:
        matrix = matrix * (np.conj(trace) / abs(trace))
    matrix = 0.5 * (matrix + matrix.conj().T)
    if np.real(np.trace(matrix)) < 0.0:
        matrix = -matrix
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-9 * scale:
        raise ValueError("the dominant transfer eigenmatrix is not positive.")
    return matrix


def _dense_transfer_matrix(tensor, *, adjoint=False):
    matrix = np.einsum(
        "aib,dic->adbc",
        tensor,
        tensor.conj(),
        optimize=True,
    ).reshape(tensor.shape[0] ** 2, tensor.shape[0] ** 2)
    return matrix.conj().T if adjoint else matrix


@dataclass(frozen=True)
class LETTATransferData:
    """Normalized structured transfer tensor and its fixed points."""

    structured_tensor: np.ndarray
    left_fixed_point: np.ndarray
    right_fixed_point: np.ndarray
    original_eigenvalue: float
    spectral_gap: float


def transfer_data(state, *, injectivity_tolerance=1.0e-10):
    """Construct normalized dominant transfer data for a uniform LETTA."""

    if not isinstance(state, UniformLETTA):
        raise TypeError("state must be a UniformLETTA.")
    injectivity_tolerance = float(injectivity_tolerance)
    if not np.isfinite(injectivity_tolerance) or injectivity_tolerance < 0.0:
        raise ValueError("injectivity_tolerance must be finite and nonnegative.")
    structured = state.structured_mps_tensor()
    dimension = structured.shape[0]
    transfer = _dense_transfer_matrix(structured)
    values, right_vectors = np.linalg.eig(transfer)
    order = np.argsort(np.abs(values))[::-1]
    dominant_index = int(order[0])
    dominant = values[dominant_index]
    eigenvalue = float(abs(dominant))
    if eigenvalue <= np.finfo(float).tiny:
        raise ValueError("the LETTA transfer operator has zero spectral radius.")
    if len(order) == 1:
        gap = 1.0
    else:
        gap = 1.0 - float(abs(values[order[1]]) / eigenvalue)
    if gap <= injectivity_tolerance:
        raise ValueError(
            "the uniform LETTA transfer operator is noninjective: "
            "its dominant eigenvalue is not unique."
        )
    right = _positive_eigenmatrix(right_vectors[:, dominant_index], dimension)

    adjoint = _dense_transfer_matrix(structured, adjoint=True)
    left_values, left_vectors = np.linalg.eig(adjoint)
    left_index = int(np.argmin(np.abs(left_values - np.conj(dominant))))
    left = _positive_eigenmatrix(left_vectors[:, left_index], dimension)

    overlap = np.real(np.trace(left @ right))
    if overlap <= np.finfo(float).tiny:
        raise ValueError("left and right LETTA transfer fixed points do not overlap.")
    right_trace = np.real(np.trace(right))
    if right_trace <= np.finfo(float).tiny:
        raise ValueError("the LETTA right fixed point has zero trace.")
    right = right / right_trace
    left = left / np.trace(left @ right)
    structured = structured / np.sqrt(eigenvalue)

    return LETTATransferData(
        structured_tensor=structured,
        left_fixed_point=left,
        right_fixed_point=right,
        original_eigenvalue=eigenvalue,
        spectral_gap=gap,
    )


def one_site_expectation(state, operator):
    """Return an exact thermodynamic one-site expectation value."""

    operator = np.asarray(operator)
    expected = (state.physical_dim, state.physical_dim)
    if operator.shape != expected:
        raise ValueError(f"a one-site operator must have shape {expected}.")
    data = transfer_data(state)
    tensor = data.structured_tensor
    left = data.left_fixed_point
    right = data.right_fixed_point
    value = np.einsum(
        "ab,bkc,cd,ajd,jk->",
        left,
        tensor,
        right,
        tensor.conj(),
        operator,
        optimize=True,
    )
    return np.real_if_close(value).item()


def two_site_expectation(state, operator):
    """Return an exact thermodynamic nearest-neighbor expectation value."""

    h = _as_two_site_operator(operator, state.physical_dim)
    data = transfer_data(state)
    tensor = data.structured_tensor
    left = data.left_fixed_point
    right = data.right_fixed_point
    pair = np.einsum("aib,bjc->aijc", tensor, tensor, optimize=True)
    value = np.einsum(
        "ab,bklc,cd,aijd,ijkl->",
        left,
        pair,
        right,
        pair.conj(),
        h,
        optimize=True,
    )
    return np.real_if_close(value).item()


def energy_density(state, hamiltonian):
    """Return the real energy density of a Hermitian two-site Hamiltonian."""

    h = _as_two_site_operator(hamiltonian, state.physical_dim)
    matrix = h.reshape(state.physical_dim**2, state.physical_dim**2)
    if not np.allclose(matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("the Hamiltonian must be Hermitian.")
    value = two_site_expectation(state, h)
    if abs(np.imag(value)) > 1.0e-9:
        raise ValueError("a Hermitian LETTA energy acquired a nonreal value.")
    return float(np.real(value))

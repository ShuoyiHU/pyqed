"""Canonical one-site uniform MPS data for VUMPS.

All site tensors use the project-wide ``(left, physical, right)`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs


def _as_site_tensor(tensor):
    array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError("a uniform MPS tensor must have shape (left, physical, right).")
    if array.shape[0] != array.shape[2]:
        raise ValueError("a one-site uniform MPS requires equal left and right bond dimensions.")
    if array.shape[1] <= 0:
        raise ValueError("the physical dimension must be positive.")
    return array


def apply_right_transfer(tensor, matrix):
    """Apply ``X -> sum_s A[s] X A[s].H``."""

    tensor = _as_site_tensor(tensor)
    matrix = np.asarray(matrix)
    dtype = np.result_type(tensor.dtype, matrix.dtype, np.complex128)
    out = np.zeros(matrix.shape, dtype=dtype)
    for physical in range(tensor.shape[1]):
        site = tensor[:, physical, :]
        out += site @ matrix @ site.conj().T
    return out


def apply_left_transfer(tensor, matrix):
    """Apply ``X -> sum_s A[s].H X A[s]``."""

    tensor = _as_site_tensor(tensor)
    matrix = np.asarray(matrix)
    dtype = np.result_type(tensor.dtype, matrix.dtype, np.complex128)
    out = np.zeros(matrix.shape, dtype=dtype)
    for physical in range(tensor.shape[1]):
        site = tensor[:, physical, :]
        out += site.conj().T @ matrix @ site
    return out


def _dense_superoperator(action, dimension, dtype):
    size = dimension * dimension
    matrix = np.zeros((size, size), dtype=dtype)
    for column in range(size):
        basis = np.zeros(size, dtype=dtype)
        basis[column] = 1.0
        matrix[:, column] = action(basis.reshape(dimension, dimension)).reshape(-1)
    return matrix


def _dominant_eigenmatrix(action, dimension, dtype, dense_threshold=256):
    size = dimension * dimension
    if size <= dense_threshold:
        operator = _dense_superoperator(action, dimension, dtype)
        values, vectors = np.linalg.eig(operator)
        index = int(np.argmax(np.abs(values)))
        value = values[index]
        vector = vectors[:, index]
    else:
        operator = LinearOperator(
            (size, size),
            matvec=lambda vector: action(vector.reshape(dimension, dimension)).reshape(-1),
            dtype=dtype,
        )
        values, vectors = eigs(operator, k=1, which="LM")
        value = values[0]
        vector = vectors[:, 0]
    matrix = vector.reshape(dimension, dimension)
    trace = np.trace(matrix)
    if abs(trace) > 1.0e-14:
        matrix *= np.conj(trace) / abs(trace)
    matrix = 0.5 * (matrix + matrix.conj().T)
    if np.real(np.trace(matrix)) < 0.0:
        matrix = -matrix
    return value, matrix


def _positive_matrix_sqrt(matrix, rcond):
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).conj().T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), 1.0)
    cutoff = float(rcond) * scale
    if float(np.min(values)) < -100.0 * cutoff:
        raise ValueError("the transfer fixed point is not positive semidefinite.")
    values = np.clip(np.real(values), 0.0, None)
    if np.any(values <= cutoff):
        raise ValueError("the transfer fixed point is rank deficient.")
    roots = np.sqrt(values)
    sqrt = (vectors * roots) @ vectors.conj().T
    inverse = (vectors * (1.0 / roots)) @ vectors.conj().T
    return sqrt, inverse


@dataclass(frozen=True)
class CanonicalMPS:
    """One-site mixed-canonical uniform MPS.

    ``AL`` and ``AR`` have shape ``(bond, physical, bond)`` and ``C`` has
    shape ``(bond, bond)``. At a converged VUMPS fixed point they satisfy
    ``AC[s] = AL[s] C = C AR[s]``. During an iteration, ``center_tensor``
    stores the independently optimized ``AC``.
    """

    AL: np.ndarray
    C: np.ndarray
    AR: np.ndarray
    center_tensor: np.ndarray | None = None

    def __post_init__(self):
        left = _as_site_tensor(self.AL)
        right = _as_site_tensor(self.AR)
        center = np.asarray(self.C)
        if left.shape != right.shape:
            raise ValueError("AL and AR must have the same shape.")
        if center.shape != (left.shape[0], left.shape[2]):
            raise ValueError("C has an incompatible bond dimension.")
        center_tensor = self.center_tensor
        if center_tensor is not None:
            center_tensor = _as_site_tensor(center_tensor)
            if center_tensor.shape != left.shape:
                raise ValueError("AC has an incompatible shape.")
        object.__setattr__(self, "AL", left)
        object.__setattr__(self, "AR", right)
        object.__setattr__(self, "C", center)
        object.__setattr__(self, "center_tensor", center_tensor)
        self.validate()

    @property
    def bond_dim(self):
        return int(self.C.shape[0])

    @property
    def physical_dim(self):
        return int(self.AL.shape[1])

    @property
    def AC(self):
        if self.center_tensor is not None:
            return self.center_tensor
        return np.stack(
            [self.AL[:, physical, :] @ self.C for physical in range(self.physical_dim)],
            axis=1,
        )

    @property
    def rho_left(self):
        density = self.C @ self.C.conj().T
        return density / np.trace(density)

    @property
    def rho_right(self):
        density = self.C.conj().T @ self.C
        return density / np.trace(density)

    def validate(self, tolerance=1.0e-8):
        """Validate the normalized canonical-leg invariants."""

        arrays = (self.AL, self.C, self.AR)
        if self.center_tensor is not None:
            arrays = arrays + (self.center_tensor,)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("canonical MPS tensors must contain only finite values.")

        if self.left_isometry_error() > tolerance:
            raise ValueError("AL is not left-canonical.")
        if self.right_isometry_error() > tolerance:
            raise ValueError("AR is not right-canonical.")
        if not np.isclose(np.linalg.norm(self.C), 1.0, rtol=tolerance, atol=tolerance):
            raise ValueError("C must have unit Frobenius norm.")
        if self.center_tensor is not None:
            if not np.isclose(
                np.linalg.norm(self.center_tensor),
                1.0,
                rtol=tolerance,
                atol=tolerance,
            ):
                raise ValueError("AC must have unit Frobenius norm.")
        elif self.center_error() > tolerance:
            raise ValueError("AL, C, and AR do not satisfy the center relation.")
        return self

    def left_isometry_error(self):
        identity = np.eye(self.bond_dim, dtype=self.AL.dtype)
        value = apply_left_transfer(self.AL, identity)
        return float(np.linalg.norm(value - identity))

    def right_isometry_error(self):
        identity = np.eye(self.bond_dim, dtype=self.AR.dtype)
        value = apply_right_transfer(self.AR, identity)
        return float(np.linalg.norm(value - identity))

    def center_error(self):
        left_center = np.stack(
            [self.AL[:, physical, :] @ self.C for physical in range(self.physical_dim)],
            axis=1,
        )
        right_center = np.stack(
            [self.C @ self.AR[:, physical, :] for physical in range(self.physical_dim)],
            axis=1,
        )
        return max(
            float(np.linalg.norm(self.AC - left_center)),
            float(np.linalg.norm(self.AC - right_center)),
        )

    def to_uniform_mps(self):
        """Return the existing physical-first ``UniformMPS`` representation."""

        from pyqed.mps import UniformMPS

        return UniformMPS(self.AL.transpose(1, 0, 2))


def right_fixed_point(tensor, *, dense_threshold=256):
    """Return the normalized positive right fixed point of a site transfer map."""

    tensor = _as_site_tensor(tensor)
    dimension = tensor.shape[0]
    dtype = np.result_type(tensor.dtype, np.complex128)
    _value, matrix = _dominant_eigenmatrix(
        lambda value: apply_right_transfer(tensor, value),
        dimension,
        dtype,
        dense_threshold=dense_threshold,
    )
    trace = np.trace(matrix)
    if abs(trace) <= np.finfo(float).tiny:
        raise ValueError("the right transfer fixed point has zero trace.")
    return matrix / trace


def canonicalize(tensor, *, rcond=1.0e-12, dense_threshold=256):
    """Return a mixed-canonical form gauge-equivalent to ``tensor``."""

    tensor = _as_site_tensor(tensor)
    bond = tensor.shape[0]
    dtype = np.result_type(tensor.dtype, np.complex128)
    work = tensor.astype(dtype, copy=False)

    right_value, right_fixed = _dominant_eigenmatrix(
        lambda matrix: apply_right_transfer(work, matrix),
        bond,
        dtype,
        dense_threshold=dense_threshold,
    )
    left_value, left_fixed = _dominant_eigenmatrix(
        lambda matrix: apply_left_transfer(work, matrix),
        bond,
        dtype,
        dense_threshold=dense_threshold,
    )
    radius = 0.5 * (abs(right_value) + abs(left_value))
    if radius <= np.finfo(float).tiny:
        raise ValueError("cannot canonicalize a tensor with zero transfer radius.")
    work = work / np.sqrt(radius)

    sqrt_left, inv_sqrt_left = _positive_matrix_sqrt(left_fixed, rcond)
    sqrt_right, inv_sqrt_right = _positive_matrix_sqrt(right_fixed, rcond)

    left = np.empty_like(work)
    right = np.empty_like(work)
    for physical in range(work.shape[1]):
        site = work[:, physical, :]
        left[:, physical, :] = sqrt_left @ site @ inv_sqrt_left
        right[:, physical, :] = inv_sqrt_right @ site @ sqrt_right

    center = sqrt_left @ sqrt_right
    norm = np.linalg.norm(center)
    if norm <= np.finfo(float).tiny:
        raise ValueError("canonicalization produced a zero center matrix.")
    center = center / norm
    return CanonicalMPS(AL=left, C=center, AR=right)


def random_canonical_mps(physical_dim, bond_dim, *, seed=None, real=False):
    """Create a random injective one-site canonical MPS."""

    physical_dim = int(physical_dim)
    bond_dim = int(bond_dim)
    if physical_dim <= 0 or bond_dim <= 0:
        raise ValueError("physical_dim and bond_dim must be positive.")
    rng = np.random.default_rng(seed)
    tensor = rng.normal(size=(bond_dim, physical_dim, bond_dim))
    if not real:
        tensor = tensor + 1j * rng.normal(size=tensor.shape)
    tensor /= np.sqrt(max(physical_dim * bond_dim, 1))
    return canonicalize(tensor)

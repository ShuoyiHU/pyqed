"""Nearest-neighbor VUMPS environments and effective Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .state import (
    CanonicalMPS,
    apply_left_transfer,
    apply_right_transfer,
    right_fixed_point,
)


def as_two_site_operator(operator, physical_dim=None):
    """Return ``h[bra_left, bra_right, ket_left, ket_right]``."""

    array = np.asarray(operator)
    if array.ndim == 2:
        dimension = int(round(np.sqrt(array.shape[0])))
        if array.shape != (dimension * dimension, dimension * dimension):
            raise ValueError("a two-site operator matrix must have shape (d**2, d**2).")
        if physical_dim is not None and int(physical_dim) != dimension:
            raise ValueError("physical_dim is inconsistent with the operator.")
        return array.reshape(dimension, dimension, dimension, dimension)
    if array.ndim == 4:
        if len(set(array.shape)) != 1:
            raise ValueError("a rank-4 two-site operator must have shape (d, d, d, d).")
        if physical_dim is not None and int(physical_dim) != array.shape[0]:
            raise ValueError("physical_dim is inconsistent with the operator.")
        return array
    raise ValueError("the nearest-neighbor operator must be rank 2 or rank 4.")


def nearest_neighbor_energy(state, operator):
    """Evaluate the energy density of the physical state represented by ``AL``."""

    if not isinstance(state, CanonicalMPS):
        raise TypeError("state must be a CanonicalMPS.")
    state.validate()
    h = as_two_site_operator(operator, state.physical_dim)
    density = right_fixed_point(state.AL)
    energy = np.trace(density @ _left_source(state, h))
    return float(np.real(np.real_if_close(energy)))


def one_site_expectation(state, operator):
    """Evaluate a one-site operator in the physical state represented by ``AL``."""

    if not isinstance(state, CanonicalMPS):
        raise TypeError("state must be a CanonicalMPS.")
    state.validate()
    operator = np.asarray(operator)
    expected_shape = (state.physical_dim, state.physical_dim)
    if operator.shape != expected_shape:
        raise ValueError(
            f"a one-site operator must have shape {expected_shape}, "
            f"not {operator.shape}."
        )

    density = right_fixed_point(state.AL)
    value = 0.0j
    for bra in range(state.physical_dim):
        bra_tensor = state.AL[:, bra, :]
        for ket in range(state.physical_dim):
            ket_tensor = state.AL[:, ket, :]
            value += operator[bra, ket] * np.trace(
                density @ bra_tensor.conj().T @ ket_tensor
            )
    return np.real_if_close(value).item()


def _left_source(state, h):
    out = np.zeros(
        (state.bond_dim, state.bond_dim),
        dtype=np.result_type(state.AL.dtype, h.dtype, np.complex128),
    )
    for bra_left in range(state.physical_dim):
        a_bra_left = state.AL[:, bra_left, :]
        for bra_right in range(state.physical_dim):
            a_bra_right = state.AL[:, bra_right, :]
            for ket_left in range(state.physical_dim):
                a_ket_left = state.AL[:, ket_left, :]
                for ket_right in range(state.physical_dim):
                    coefficient = h[bra_left, bra_right, ket_left, ket_right]
                    a_ket_right = state.AL[:, ket_right, :]
                    out += coefficient * (
                        a_bra_right.conj().T
                        @ a_bra_left.conj().T
                        @ a_ket_left
                        @ a_ket_right
                    )
    return 0.5 * (out + out.conj().T)


def _right_source(state, h):
    out = np.zeros(
        (state.bond_dim, state.bond_dim),
        dtype=np.result_type(state.AR.dtype, h.dtype, np.complex128),
    )
    for bra_left in range(state.physical_dim):
        a_bra_left = state.AR[:, bra_left, :]
        for bra_right in range(state.physical_dim):
            a_bra_right = state.AR[:, bra_right, :]
            for ket_left in range(state.physical_dim):
                a_ket_left = state.AR[:, ket_left, :]
                for ket_right in range(state.physical_dim):
                    coefficient = h[bra_left, bra_right, ket_left, ket_right]
                    a_ket_right = state.AR[:, ket_right, :]
                    out += coefficient * (
                        a_ket_left
                        @ a_ket_right
                        @ a_bra_right.conj().T
                        @ a_bra_left.conj().T
                    )
    return 0.5 * (out + out.conj().T)


def _solve_environment(
    transfer,
    source,
    density,
    *,
    tolerance=1.0e-12,
    max_iterations=None,
    dense_threshold=256,
):
    dimension = source.shape[0]
    size = dimension * dimension
    identity = np.eye(dimension, dtype=np.result_type(source.dtype, density.dtype))

    def gauge_overlap(matrix):
        return np.trace(density @ matrix)

    centered = source - identity * gauge_overlap(source)

    def action(matrix):
        return matrix - transfer(matrix) + identity * gauge_overlap(matrix)

    if size <= dense_threshold:
        matrix = np.zeros((size, size), dtype=np.result_type(centered.dtype, np.complex128))
        for column in range(size):
            basis = np.zeros(size, dtype=matrix.dtype)
            basis[column] = 1.0
            matrix[:, column] = action(basis.reshape(dimension, dimension)).reshape(-1)
        solution = np.linalg.solve(matrix, centered.reshape(-1)).reshape(dimension, dimension)
    else:
        operator = LinearOperator(
            (size, size),
            matvec=lambda vector: action(vector.reshape(dimension, dimension)).reshape(-1),
            dtype=np.result_type(centered.dtype, np.complex128),
        )
        solution, info = gmres(
            operator,
            centered.reshape(-1),
            rtol=float(tolerance),
            atol=0.0,
            maxiter=max_iterations,
        )
        if info != 0:
            raise RuntimeError(f"VUMPS environment GMRES did not converge (info={info}).")
        solution = solution.reshape(dimension, dimension)

    solution = 0.5 * (solution + solution.conj().T)
    solution -= identity * gauge_overlap(solution)
    return solution


@dataclass(frozen=True)
class EffectiveHamiltonians:
    """Matrix-free center-site and center-bond Hamiltonians."""

    state: CanonicalMPS
    h: np.ndarray
    energy: float
    HL: np.ndarray
    HR: np.ndarray

    def apply_center(self, tensor):
        tensor = np.asarray(tensor)
        if tensor.shape != self.state.AC.shape:
            raise ValueError("center tensor has an incompatible shape.")
        out = np.zeros_like(
            tensor,
            dtype=np.result_type(tensor.dtype, self.HL.dtype, self.HR.dtype, self.h.dtype),
        )
        for physical in range(self.state.physical_dim):
            out[:, physical, :] += (
                self.HL @ tensor[:, physical, :]
                + tensor[:, physical, :] @ self.HR
            )

        for bra_left in range(self.state.physical_dim):
            al_bra = self.state.AL[:, bra_left, :]
            for bra_right in range(self.state.physical_dim):
                ar_bra = self.state.AR[:, bra_right, :]
                for ket_left in range(self.state.physical_dim):
                    al_ket = self.state.AL[:, ket_left, :]
                    for ket_right in range(self.state.physical_dim):
                        coefficient = self.h[bra_left, bra_right, ket_left, ket_right]
                        out[:, bra_right, :] += coefficient * (
                            al_bra.conj().T
                            @ al_ket
                            @ tensor[:, ket_right, :]
                        )
                        out[:, bra_left, :] += coefficient * (
                            tensor[:, ket_left, :]
                            @ self.state.AR[:, ket_right, :]
                            @ ar_bra.conj().T
                        )
        return out

    def apply_bond(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.shape != self.state.C.shape:
            raise ValueError("center matrix has an incompatible shape.")
        out = self.HL @ matrix + matrix @ self.HR
        for bra_left in range(self.state.physical_dim):
            al_bra = self.state.AL[:, bra_left, :]
            for bra_right in range(self.state.physical_dim):
                ar_bra = self.state.AR[:, bra_right, :]
                for ket_left in range(self.state.physical_dim):
                    al_ket = self.state.AL[:, ket_left, :]
                    for ket_right in range(self.state.physical_dim):
                        coefficient = self.h[bra_left, bra_right, ket_left, ket_right]
                        out += coefficient * (
                            al_bra.conj().T
                            @ al_ket
                            @ matrix
                            @ self.state.AR[:, ket_right, :]
                            @ ar_bra.conj().T
                        )
        return out


def build_effective_hamiltonians(
    state,
    operator,
    *,
    environment_tolerance=1.0e-12,
    environment_max_iterations=None,
    dense_environment_threshold=256,
):
    """Construct regularized infinite Hamiltonian environments."""

    if not isinstance(state, CanonicalMPS):
        raise TypeError("state must be a CanonicalMPS.")
    state.validate()
    h = as_two_site_operator(operator, state.physical_dim)
    energy = nearest_neighbor_energy(state, h)
    identity = np.eye(state.bond_dim, dtype=np.result_type(state.C.dtype, h.dtype))
    left_source = _left_source(state, h) - energy * identity
    right_source = _right_source(state, h) - energy * identity
    left = _solve_environment(
        lambda matrix: apply_left_transfer(state.AL, matrix),
        left_source,
        state.rho_left,
        tolerance=environment_tolerance,
        max_iterations=environment_max_iterations,
        dense_threshold=dense_environment_threshold,
    )
    right = _solve_environment(
        lambda matrix: apply_right_transfer(state.AR, matrix),
        right_source,
        state.rho_right,
        tolerance=environment_tolerance,
        max_iterations=environment_max_iterations,
        dense_threshold=dense_environment_threshold,
    )
    return EffectiveHamiltonians(state=state, h=h, energy=energy, HL=left, HR=right)

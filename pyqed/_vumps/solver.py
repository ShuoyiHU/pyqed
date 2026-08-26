"""High-level one-site VUMPS solver for nearest-neighbor Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from pyqed._vumps.operators import (
    as_two_site_operator,
    build_effective_hamiltonians,
    nearest_neighbor_energy,
)
from pyqed._vumps.state import CanonicalMPS, canonicalize, random_canonical_mps


@dataclass(frozen=True)
class VUMPSOptions:
    max_iterations: int = 100
    tolerance: float = 1.0e-8
    eigensolver_tolerance: float = 1.0e-10
    eigensolver_max_iterations: int | None = None
    dense_eigensolver_threshold: int = 256
    environment_tolerance: float = 1.0e-12
    environment_max_iterations: int | None = None
    dense_environment_threshold: int = 256
    verbosity: int = 0


@dataclass(frozen=True)
class VUMPSIteration:
    iteration: int
    energy: float
    residual_norm: float
    canonical_residual_norm: float
    fixed_point_residual_norm: float
    center_eigenvalue: float
    bond_eigenvalue: float


@dataclass(frozen=True)
class VUMPSResult:
    state: CanonicalMPS
    energy: float
    converged: bool
    iterations: int
    residual_norm: float
    canonical_residual_norm: float
    fixed_point_residual_norm: float
    history: tuple[VUMPSIteration, ...]
    message: str

    def to_uniform_mps(self):
        return self.state.to_uniform_mps()


def _phase_align(vector, reference):
    overlap = np.vdot(np.asarray(reference).reshape(-1), np.asarray(vector).reshape(-1))
    if abs(overlap) > 1.0e-14:
        vector = vector * (np.conj(overlap) / abs(overlap))
    return vector


def _lowest_eigenpair(action, shape, reference, options):
    size = int(np.prod(shape))
    dtype = np.result_type(np.asarray(reference).dtype, np.complex128)

    if size <= int(options.dense_eigensolver_threshold):
        matrix = np.zeros((size, size), dtype=dtype)
        for column in range(size):
            basis = np.zeros(size, dtype=dtype)
            basis[column] = 1.0
            matrix[:, column] = action(basis.reshape(shape)).reshape(-1)
        matrix = 0.5 * (matrix + matrix.conj().T)
        values, vectors = np.linalg.eigh(matrix)
        index = int(np.argmin(np.real(values)))
        value = values[index]
        vector = vectors[:, index]
    elif size == 1:
        vector = np.ones(1, dtype=dtype)
        value = action(vector.reshape(shape)).reshape(-1)[0]
    else:
        operator = LinearOperator(
            (size, size),
            matvec=lambda vector: action(vector.reshape(shape)).reshape(-1),
            dtype=dtype,
        )
        values, vectors = eigsh(
            operator,
            k=1,
            which="SA",
            v0=np.asarray(reference).reshape(-1),
            tol=float(options.eigensolver_tolerance),
            maxiter=options.eigensolver_max_iterations,
        )
        value = values[0]
        vector = vectors[:, 0]

    vector = _phase_align(vector.reshape(shape), reference)
    norm = np.linalg.norm(vector)
    if norm <= np.finfo(float).tiny:
        raise RuntimeError("the VUMPS eigensolver returned a zero vector.")
    return float(np.real(np.real_if_close(value))), vector / norm


def _polar_isometry(matrix):
    left, _singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def _gauge_match(center_tensor, center_matrix):
    bond, physical, _ = center_tensor.shape
    left_center = center_tensor.reshape(bond * physical, bond)
    left_isometry = _polar_isometry(left_center)
    center_left_isometry = _polar_isometry(center_matrix)
    AL = (left_isometry @ center_left_isometry.conj().T).reshape(
        bond, physical, bond
    )

    right_center = center_tensor.reshape(bond, physical * bond)
    right_isometry = _polar_isometry(right_center)
    center_right_isometry = _polar_isometry(center_matrix)
    AR = (center_right_isometry.conj().T @ right_isometry).reshape(
        bond, physical, bond
    )
    return AL, AR


def _initial_state(initial, physical_dim, bond_dim, seed, real):
    if initial is None:
        return random_canonical_mps(physical_dim, bond_dim, seed=seed, real=real)
    if isinstance(initial, CanonicalMPS):
        if initial.physical_dim != physical_dim:
            raise ValueError("the initial state has an incompatible physical dimension.")
        # A solver iterate may carry an independently optimized, provisional
        # center. Restart from the definite physical uMPS represented by AL.
        return canonicalize(initial.AL)

    try:
        from pyqed.mps import UniformMPS
    except ImportError:  # pragma: no cover
        UniformMPS = ()
    if UniformMPS and isinstance(initial, UniformMPS):
        if initial.unit_cell_size != 1:
            raise ValueError("VUMPS currently supports a one-site initial UniformMPS.")
        tensor = initial.tensor.transpose(1, 0, 2)
    else:
        tensor = np.asarray(initial)
    if (
        tensor.ndim != 3
        or tensor.shape[0] != tensor.shape[2]
        or tensor.shape[1] != physical_dim
    ):
        raise ValueError(
            "the initial tensor must have shape "
            "(bond_dim, physical_dim, bond_dim)."
        )
    return canonicalize(tensor)


def vumps(
    hamiltonian,
    *,
    physical_dim=None,
    bond_dim=4,
    initial=None,
    seed=None,
    real=False,
    options=None,
):
    """Optimize a one-site uniform MPS for a nearest-neighbor Hamiltonian.

    Parameters
    ----------
    hamiltonian
        Hermitian two-site operator with shape ``(d**2, d**2)`` or
        ``(d, d, d, d)``. Rank-four indices are ordered as
        ``(bra_left, bra_right, ket_left, ket_right)``.
    physical_dim
        Optional physical dimension used to validate ``hamiltonian``.
    bond_dim
        Bond dimension of a randomly generated initial state. An explicit
        ``initial`` state supplies its own bond dimension.
    initial
        A :class:`CanonicalMPS`, one-site
        :class:`pyqed.mps.UniformMPS`, or tensor with shape
        ``(bond, physical, bond)``.
    seed
        Seed for random initialization.
    real
        Generate a real random initial tensor. Iterations may still become
        complex when the Hamiltonian is complex.
    options
        Numerical tolerances, iteration limits, and verbosity.
    """

    options = VUMPSOptions() if options is None else options
    if not isinstance(options, VUMPSOptions):
        raise TypeError("options must be a VUMPSOptions instance.")
    h = as_two_site_operator(hamiltonian, physical_dim)
    physical_dim = int(h.shape[0])
    h_matrix = h.reshape(physical_dim**2, physical_dim**2)
    if not np.all(np.isfinite(h_matrix)):
        raise ValueError("the Hamiltonian must contain only finite values.")
    if not np.allclose(h_matrix, h_matrix.conj().T):
        raise ValueError("the Hamiltonian must be Hermitian.")
    bond_dim = int(bond_dim)
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive.")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    state = _initial_state(initial, physical_dim, bond_dim, seed, real)
    history = []
    converged = False
    residual = float("inf")
    canonical_residual = float("inf")
    fixed_point_residual = float("inf")

    for iteration in range(1, int(options.max_iterations) + 1):
        previous_center_tensor = state.AC
        previous_center_matrix = state.C
        effective = build_effective_hamiltonians(
            state,
            h,
            environment_tolerance=options.environment_tolerance,
            environment_max_iterations=options.environment_max_iterations,
            dense_environment_threshold=options.dense_environment_threshold,
        )
        center_value, center_tensor = _lowest_eigenpair(
            effective.apply_center,
            state.AC.shape,
            state.AC,
            options,
        )
        bond_value, center_matrix = _lowest_eigenpair(
            effective.apply_bond,
            state.C.shape,
            state.C,
            options,
        )

        AL, AR = _gauge_match(center_tensor, center_matrix)
        left_difference = center_tensor - np.stack(
            [
                AL[:, physical, :] @ center_matrix
                for physical in range(physical_dim)
            ],
            axis=1,
        )
        right_difference = center_tensor - np.stack(
            [
                center_matrix @ AR[:, physical, :]
                for physical in range(physical_dim)
            ],
            axis=1,
        )
        canonical_residual = max(
            float(np.linalg.norm(left_difference)),
            float(np.linalg.norm(right_difference)),
        )
        fixed_point_residual = max(
            float(np.linalg.norm(center_tensor - previous_center_tensor)),
            float(np.linalg.norm(center_matrix - previous_center_matrix)),
        )
        residual = max(canonical_residual, fixed_point_residual)

        state = CanonicalMPS(
            AL=AL,
            C=center_matrix,
            AR=AR,
            center_tensor=center_tensor,
        )
        energy = nearest_neighbor_energy(state, h)
        record = VUMPSIteration(
            iteration=iteration,
            energy=energy,
            residual_norm=residual,
            canonical_residual_norm=canonical_residual,
            fixed_point_residual_norm=fixed_point_residual,
            center_eigenvalue=center_value,
            bond_eigenvalue=bond_value,
        )
        history.append(record)
        if options.verbosity:
            print(
                f"VUMPS {iteration:4d}  energy={energy: .14f}  "
                f"residual={residual:.3e}  canonical={canonical_residual:.3e}  "
                f"fixed-point={fixed_point_residual:.3e}"
            )
        if residual <= float(options.tolerance):
            converged = True
            break

    message = "converged" if converged else "maximum iterations reached"
    return VUMPSResult(
        state=state,
        energy=nearest_neighbor_energy(state, h),
        converged=converged,
        iterations=len(history),
        residual_norm=residual,
        canonical_residual_norm=canonical_residual,
        fixed_point_residual_norm=fixed_point_residual,
        history=tuple(history),
        message=message,
    )

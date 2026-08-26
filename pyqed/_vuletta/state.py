"""Uniform nearest-neighbor leg-tied tensor states."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np


def _as_letta_tensor(tensor):
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError(
            "a uniform LETTA tensor must have shape "
            "(left, previous_physical, current_physical, right)."
        )
    if array.shape[0] != array.shape[3]:
        raise ValueError("left and right LETTA bond dimensions must agree.")
    if array.shape[1] != array.shape[2]:
        raise ValueError("the two shared physical dimensions must agree.")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("bond and physical dimensions must be positive.")
    if not np.all(np.isfinite(array)):
        raise ValueError("a LETTA tensor must contain only finite values.")
    if np.linalg.norm(array) <= np.finfo(float).tiny:
        raise ValueError("a LETTA tensor cannot be numerically zero.")
    return array


@dataclass(frozen=True)
class UniformLETTA:
    """One-site uniform nearest-neighbor LETTA state.

    The tensor order is ``(left, previous_physical, current_physical, right)``.
    The periodic amplitude is ``trace(prod_n T[s_n, s_(n+1)]))``.
    """

    tensor: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "tensor", _as_letta_tensor(self.tensor))

    @property
    def bond_dim(self):
        return int(self.tensor.shape[0])

    @property
    def physical_dim(self):
        return int(self.tensor.shape[1])

    @property
    def effective_bond_dim(self):
        return self.bond_dim * self.physical_dim

    def normalized_parameters(self):
        """Return a copy with unit Frobenius norm."""

        return UniformLETTA(self.tensor / np.linalg.norm(self.tensor))

    def periodic_amplitude(self, configuration):
        """Return the amplitude of a finite periodic configuration."""

        configuration = tuple(int(value) for value in configuration)
        if not configuration:
            raise ValueError("a periodic configuration must contain at least one site.")
        if any(value < 0 or value >= self.physical_dim for value in configuration):
            raise ValueError("a physical configuration index is out of range.")
        product = np.eye(self.bond_dim, dtype=self.tensor.dtype)
        for site, physical in enumerate(configuration):
            neighbor = configuration[(site + 1) % len(configuration)]
            product = product @ self.tensor[:, physical, neighbor, :]
        return np.trace(product)

    def structured_mps_tensor(self):
        """Return the exact sparse MPS contraction identity for this LETTA."""

        bond = self.bond_dim
        physical_dim = self.physical_dim
        effective_bond = self.effective_bond_dim
        structured = np.zeros(
            (effective_bond, physical_dim, effective_bond),
            dtype=self.tensor.dtype,
        )
        for left in range(bond):
            for previous in range(physical_dim):
                left_combined = left * physical_dim + previous
                for current in range(physical_dim):
                    for right in range(bond):
                        right_combined = right * physical_dim + current
                        structured[left_combined, current, right_combined] = (
                            self.tensor[left, previous, current, right]
                        )
        return structured

    def shifted_structured_mps_tensor(self):
        """Return the equivalent MPS with the copy constraint on the left leg."""

        bond = self.bond_dim
        physical_dim = self.physical_dim
        effective_bond = self.effective_bond_dim
        structured = np.zeros(
            (effective_bond, physical_dim, effective_bond),
            dtype=self.tensor.dtype,
        )
        for left in range(bond):
            for previous in range(physical_dim):
                left_combined = left * physical_dim + previous
                for current in range(physical_dim):
                    for right in range(bond):
                        right_combined = right * physical_dim + current
                        structured[left_combined, previous, right_combined] = (
                            self.tensor[left, previous, current, right]
                        )
        return structured

    def gauge_transform(self, gauges):
        """Apply ``T[p,s] -> inv(G[p]) T[p,s] G[s]``."""

        gauges = np.asarray(gauges)
        expected = (self.physical_dim, self.bond_dim, self.bond_dim)
        if gauges.shape != expected:
            raise ValueError(f"gauges must have shape {expected}.")
        transformed = np.empty(
            self.tensor.shape,
            dtype=np.result_type(self.tensor.dtype, gauges.dtype),
        )
        for previous in range(self.physical_dim):
            for current in range(self.physical_dim):
                transformed[:, previous, current, :] = np.linalg.solve(
                    gauges[previous],
                    self.tensor[:, previous, current, :] @ gauges[current],
                )
        return UniformLETTA(transformed)


def _conditioned_positive_sqrt(matrix, rcond):
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    cutoff = float(rcond) * scale
    if float(np.min(eigenvalues)) < -100.0 * cutoff:
        raise ValueError("a conditioned transfer fixed point is not positive semidefinite.")
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
    retained = eigenvalues > cutoff
    roots = np.sqrt(eigenvalues)
    inverse_roots = np.zeros_like(roots)
    inverse_roots[retained] = 1.0 / roots[retained]
    sqrt = (eigenvectors * roots) @ eigenvectors.conj().T
    inverse = (eigenvectors * inverse_roots) @ eigenvectors.conj().T
    return sqrt, inverse, int(np.count_nonzero(retained))


def _physical_diagonal_blocks(matrix, bond_dim, physical_dim, *, tolerance):
    blocked = np.asarray(matrix).reshape(
        bond_dim,
        physical_dim,
        bond_dim,
        physical_dim,
    )
    diagonal = np.stack(
        [blocked[:, physical, :, physical] for physical in range(physical_dim)]
    )
    remainder = blocked.copy()
    for physical in range(physical_dim):
        remainder[:, physical, :, physical] = 0.0
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    if float(np.linalg.norm(remainder)) > float(tolerance) * scale:
        raise ValueError(
            "the conditioned transfer fixed point is not block diagonal in "
            "the shared physical state."
        )
    return diagonal


def _dominant_right_fixed_point(tensor, injectivity_tolerance):
    from .operators import _dense_transfer_matrix, _positive_eigenmatrix

    transfer = _dense_transfer_matrix(tensor)
    eigenvalues, eigenvectors = np.linalg.eig(transfer)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    dominant_index = int(order[0])
    radius = float(abs(eigenvalues[dominant_index]))
    if radius <= np.finfo(float).tiny:
        raise ValueError("the shifted LETTA transfer operator has zero spectral radius.")
    gap = (
        1.0
        if len(order) == 1
        else 1.0 - float(abs(eigenvalues[order[1]]) / radius)
    )
    if gap <= float(injectivity_tolerance):
        raise ValueError(
            "the shifted uniform LETTA transfer operator is noninjective."
        )
    dimension = tensor.shape[0]
    fixed = _positive_eigenmatrix(eigenvectors[:, dominant_index], dimension)
    trace = np.real(np.trace(fixed))
    if trace <= np.finfo(float).tiny:
        raise ValueError("the shifted LETTA right fixed point has zero trace.")
    return fixed / trace, radius


@dataclass(frozen=True)
class ConditionalCanonicalLETTA:
    """Physical-state-conditioned mixed-canonical nearest-neighbor LETTA."""

    TL: np.ndarray
    C: np.ndarray
    TR: np.ndarray
    TC: np.ndarray
    transfer_eigenvalue: float
    left_conditioned_ranks: tuple[int, ...]
    right_conditioned_ranks: tuple[int, ...]

    def __post_init__(self):
        left = _as_letta_tensor(self.TL)
        right = _as_letta_tensor(self.TR)
        center_tensor = _as_letta_tensor(self.TC)
        centers = np.asarray(self.C)
        if left.shape != right.shape or left.shape != center_tensor.shape:
            raise ValueError("TL, TC, and TR must have identical LETTA shapes.")
        expected_centers = (left.shape[1], left.shape[0], left.shape[0])
        if centers.shape != expected_centers:
            raise ValueError(f"C must have shape {expected_centers}.")
        if not np.all(np.isfinite(centers)):
            raise ValueError("conditional center matrices must contain finite values.")
        object.__setattr__(self, "TL", left)
        object.__setattr__(self, "TR", right)
        object.__setattr__(self, "TC", center_tensor)
        object.__setattr__(self, "C", centers)
        self.validate()

    @property
    def bond_dim(self):
        return int(self.TL.shape[0])

    @property
    def physical_dim(self):
        return int(self.TL.shape[1])

    @property
    def state(self):
        """Return the normalized thermodynamic state represented by ``TL``."""

        return UniformLETTA(self.TL)

    def amplitude_scale(self, num_sites):
        """Return the finite-periodic amplitude scale relative to the input."""

        try:
            num_sites = index(num_sites)
        except TypeError as error:
            raise ValueError("num_sites must be an integer.") from error
        if num_sites <= 0:
            raise ValueError("num_sites must be positive.")
        return float(self.transfer_eigenvalue) ** (-0.5 * num_sites)

    def left_isometry_error(self):
        identity = np.eye(self.bond_dim, dtype=self.TL.dtype)
        error = 0.0
        for current in range(self.physical_dim):
            value = np.zeros_like(identity)
            for previous in range(self.physical_dim):
                block = self.TL[:, previous, current, :]
                value += block.conj().T @ block
            error = max(error, float(np.linalg.norm(value - identity)))
        return error

    def right_isometry_error(self):
        identity = np.eye(self.bond_dim, dtype=self.TR.dtype)
        error = 0.0
        for previous in range(self.physical_dim):
            value = np.zeros_like(identity)
            for current in range(self.physical_dim):
                block = self.TR[:, previous, current, :]
                value += block @ block.conj().T
            error = max(error, float(np.linalg.norm(value - identity)))
        return error

    def center_error(self):
        error = 0.0
        for previous in range(self.physical_dim):
            for current in range(self.physical_dim):
                left_center = (
                    self.TL[:, previous, current, :] @ self.C[current]
                )
                right_center = (
                    self.C[previous] @ self.TR[:, previous, current, :]
                )
                error = max(
                    error,
                    float(
                        np.linalg.norm(
                            self.TC[:, previous, current, :] - left_center
                        )
                    ),
                    float(np.linalg.norm(left_center - right_center)),
                )
        return error

    def validate(self, tolerance=1.0e-8):
        """Validate conditional isometries and the blockwise center relation."""

        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if self.left_isometry_error() > tolerance:
            raise ValueError("TL is not conditionally left-canonical.")
        if self.right_isometry_error() > tolerance:
            raise ValueError("TR is not conditionally right-canonical.")
        if self.center_error() > tolerance:
            raise ValueError("TL, TC, C, and TR violate the conditional center relation.")
        if not np.isclose(np.linalg.norm(self.C), 1.0, atol=tolerance, rtol=tolerance):
            raise ValueError("the conditional center matrices must have unit norm.")
        if not np.isclose(np.linalg.norm(self.TC), 1.0, atol=tolerance, rtol=tolerance):
            raise ValueError("TC must have unit norm.")
        return self


def conditional_canonicalize(
    state,
    *,
    rcond=1.0e-12,
    injectivity_tolerance=1.0e-10,
    allow_rank_deficient=False,
):
    """Return the conditional mixed-canonical form of a uniform NN LETTA."""

    from .operators import transfer_data

    if not isinstance(state, UniformLETTA):
        state = UniformLETTA(state)
    rcond = float(rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive.")
    data = transfer_data(
        state,
        injectivity_tolerance=injectivity_tolerance,
    )
    bond_dim = state.bond_dim
    physical_dim = state.physical_dim
    block_tolerance = max(100.0 * rcond, 1.0e-10)
    left_blocks = _physical_diagonal_blocks(
        data.left_fixed_point,
        bond_dim,
        physical_dim,
        tolerance=block_tolerance,
    )

    normalized_tensor = state.tensor / np.sqrt(data.original_eigenvalue)
    normalized_state = UniformLETTA(normalized_tensor)
    shifted = normalized_state.shifted_structured_mps_tensor()
    shifted_right, shifted_radius = _dominant_right_fixed_point(
        shifted,
        injectivity_tolerance,
    )
    if not np.isclose(shifted_radius, 1.0, rtol=1.0e-9, atol=1.0e-11):
        raise ValueError("the two exact LETTA embeddings have inconsistent transfer radii.")
    right_blocks = _physical_diagonal_blocks(
        shifted_right,
        bond_dim,
        physical_dim,
        tolerance=block_tolerance,
    )

    left_sqrts = []
    left_inverse_sqrts = []
    right_sqrts = []
    right_inverse_sqrts = []
    left_ranks = []
    right_ranks = []
    for physical in range(physical_dim):
        sqrt, inverse, rank = _conditioned_positive_sqrt(
            left_blocks[physical],
            rcond,
        )
        left_sqrts.append(sqrt)
        left_inverse_sqrts.append(inverse)
        left_ranks.append(rank)
        sqrt, inverse, rank = _conditioned_positive_sqrt(
            right_blocks[physical],
            rcond,
        )
        right_sqrts.append(sqrt)
        right_inverse_sqrts.append(inverse)
        right_ranks.append(rank)

    if not allow_rank_deficient and (
        min(left_ranks) < bond_dim or min(right_ranks) < bond_dim
    ):
        raise ValueError(
            "conditional canonicalization requires full-rank conditioned "
            "transfer support."
        )

    dtype = np.result_type(normalized_tensor.dtype, np.complex128)
    TL = np.empty(normalized_tensor.shape, dtype=dtype)
    TR = np.empty(normalized_tensor.shape, dtype=dtype)
    centers = np.empty((physical_dim, bond_dim, bond_dim), dtype=dtype)
    for physical in range(physical_dim):
        centers[physical] = left_sqrts[physical] @ right_sqrts[physical]
    center_norm = np.linalg.norm(centers)
    if center_norm <= np.finfo(float).tiny:
        raise ValueError("conditional canonicalization produced a zero center.")
    centers /= center_norm

    TC = np.empty_like(TL)
    for previous in range(physical_dim):
        for current in range(physical_dim):
            block = normalized_tensor[:, previous, current, :]
            TL[:, previous, current, :] = (
                left_sqrts[previous] @ block @ left_inverse_sqrts[current]
            )
            TR[:, previous, current, :] = (
                right_inverse_sqrts[previous] @ block @ right_sqrts[current]
            )
            TC[:, previous, current, :] = (
                TL[:, previous, current, :] @ centers[current]
            )

    if not np.iscomplexobj(state.tensor):
        TL = np.real_if_close(TL, tol=1000)
        TR = np.real_if_close(TR, tol=1000)
        TC = np.real_if_close(TC, tol=1000)
        centers = np.real_if_close(centers, tol=1000)

    return ConditionalCanonicalLETTA(
        TL=TL,
        C=centers,
        TR=TR,
        TC=TC,
        transfer_eigenvalue=data.original_eigenvalue,
        left_conditioned_ranks=tuple(left_ranks),
        right_conditioned_ranks=tuple(right_ranks),
    )


def random_uniform_letta(physical_dim, bond_dim, *, seed=None, real=False):
    """Return a random normalized uniform LETTA tensor."""

    try:
        physical_dim = index(physical_dim)
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("physical_dim and bond_dim must be integers.") from error
    if physical_dim <= 0 or bond_dim <= 0:
        raise ValueError("physical_dim and bond_dim must be positive.")
    rng = np.random.default_rng(seed)
    shape = (bond_dim, physical_dim, physical_dim, bond_dim)
    tensor = rng.normal(size=shape)
    if not real:
        tensor = tensor + 1j * rng.normal(size=shape)
    tensor /= np.linalg.norm(tensor)
    return UniformLETTA(tensor)


def expand_uniform_letta(
    state,
    bond_dim,
    *,
    seed=None,
    relative_noise=3.0e-2,
):
    """Embed a converged LETTA tensor into a larger virtual space.

    A normalized perturbation activates the added virtual sector and avoids a
    rank-deficient transfer operator at the exactly block-padded tensor.
    """

    if not isinstance(state, UniformLETTA):
        raise TypeError("state must be a UniformLETTA.")
    try:
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("bond_dim must be an integer.") from error
    if bond_dim <= state.bond_dim:
        raise ValueError("the expanded bond dimension must be larger.")
    relative_noise = float(relative_noise)
    if not np.isfinite(relative_noise) or relative_noise <= 0.0:
        raise ValueError("relative_noise must be finite and positive.")

    rng = np.random.default_rng(seed)
    shape = (
        bond_dim,
        state.physical_dim,
        state.physical_dim,
        bond_dim,
    )
    perturbation = rng.normal(size=shape)
    if np.iscomplexobj(state.tensor):
        perturbation = perturbation + 1j * rng.normal(size=shape)
    perturbation /= np.linalg.norm(perturbation)
    tensor = relative_noise * perturbation
    old_bond_dim = state.bond_dim
    tensor[:old_bond_dim, :, :, :old_bond_dim] += (
        state.tensor / np.linalg.norm(state.tensor)
    )
    return UniformLETTA(tensor / np.linalg.norm(tensor))

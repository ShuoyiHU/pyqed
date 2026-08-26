"""Boundary-MPS contractions for a uniform LETTA on the infinite plane."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from .plane_state import UniformPlaneLETTA


@dataclass(frozen=True)
class PlaneEnvironmentOptions:
    """Controls the finite-window sequence approaching the infinite plane."""

    window_sizes: tuple[int, ...] = (3, 5, 7)
    boundary_bond_dim: int = 32
    boundary_bond_dims: tuple[int, ...] | None = None
    cutoff: float = 1.0e-10
    convergence_tolerance: float = 1.0e-6
    boundary_convergence_tolerance: float | None = None

    def validated(self):
        try:
            sizes = tuple(index(size) for size in self.window_sizes)
            boundary_bond_dim = index(self.boundary_bond_dim)
        except TypeError as error:
            raise ValueError("environment dimensions must be integers.") from error
        if not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
            raise ValueError(
                "window_sizes must contain odd integers of at least three."
            )
        if any(right <= left for left, right in zip(sizes, sizes[1:])):
            raise ValueError("window_sizes must be strictly increasing.")
        cutoff = float(self.cutoff)
        tolerance = float(self.convergence_tolerance)
        if boundary_bond_dim <= 0:
            raise ValueError("boundary_bond_dim must be positive.")
        if self.boundary_bond_dims is None:
            smaller = max(1, boundary_bond_dim // 2)
            boundary_bond_dims = (
                (boundary_bond_dim,)
                if smaller == boundary_bond_dim
                else (smaller, boundary_bond_dim)
            )
        else:
            try:
                boundary_bond_dims = tuple(
                    index(value) for value in self.boundary_bond_dims
                )
            except TypeError as error:
                raise ValueError(
                    "boundary_bond_dims must contain integers."
                ) from error
            if (
                not boundary_bond_dims
                or any(value <= 0 for value in boundary_bond_dims)
                or any(
                    right <= left
                    for left, right in zip(
                        boundary_bond_dims,
                        boundary_bond_dims[1:],
                    )
                )
            ):
                raise ValueError(
                    "boundary_bond_dims must contain strictly increasing "
                    "positive integers."
                )
            if boundary_bond_dims[-1] != boundary_bond_dim:
                raise ValueError(
                    "boundary_bond_dims must end at boundary_bond_dim."
                )
        if cutoff < 0.0 or not np.isfinite(cutoff):
            raise ValueError("cutoff must be finite and nonnegative.")
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError(
                "convergence_tolerance must be finite and positive."
            )
        boundary_tolerance = self.boundary_convergence_tolerance
        if boundary_tolerance is None:
            boundary_tolerance = tolerance
        boundary_tolerance = float(boundary_tolerance)
        if boundary_tolerance <= 0.0 or not np.isfinite(boundary_tolerance):
            raise ValueError(
                "boundary_convergence_tolerance must be finite and positive."
            )
        return type(self)(
            window_sizes=sizes,
            boundary_bond_dim=boundary_bond_dim,
            boundary_bond_dims=boundary_bond_dims,
            cutoff=cutoff,
            convergence_tolerance=tolerance,
            boundary_convergence_tolerance=boundary_tolerance,
        )


@dataclass(frozen=True)
class BoundaryContraction:
    log_magnitude: float
    phase: complex
    maximum_bond_dimension: int
    discarded_weight: float


def double_layer_cell(state, operator):
    r"""Build one rank-four cell of the LETTA expectation-value network.

    Each directional cell bond contains a ket virtual index, a bra virtual
    index, and the bra/ket pair for a tied physical leg.  Its dimension is
    $D^2d^2$.
    """

    if not isinstance(state, UniformPlaneLETTA):
        raise TypeError("state must be a UniformPlaneLETTA.")
    operator = np.asarray(operator)
    local_dim = state.local_physical_dim
    if operator.shape != (local_dim, local_dim):
        raise ValueError(
            "a one-site operator must match the local physical dimension."
        )
    tensor = state.normalized_parameters().tensor
    identity = np.eye(local_dim, dtype=np.result_type(tensor, operator))
    cell = np.einsum(
        "LRUDCXY,lrudcxy,Cc,CP,cp->LlCcRrXxUuPpDdYy",
        tensor.conj(),
        tensor,
        operator,
        identity,
        identity,
        optimize=True,
    )
    dimension = state.double_layer_bond_dim
    return cell.reshape((dimension,) * 4)


def _boundary_vector(state):
    virtual_identity = np.eye(state.bond_dim)
    physical_identity = np.eye(state.local_physical_dim)
    vector = np.einsum(
        "ab,ij->abij",
        virtual_identity,
        physical_identity,
        optimize=True,
    ).reshape(-1)
    return vector / np.linalg.norm(vector)


def _mps_norm(cores):
    environment = np.ones((1, 1), dtype=np.result_type(*cores))
    for core in cores:
        environment = np.einsum(
            "aA,apb,ApB->bB",
            environment,
            core,
            core.conj(),
            optimize=True,
        )
    value = float(np.real(environment.reshape(())))
    return np.sqrt(max(value, 0.0))


def _compress_mps(cores, max_bond_dim, cutoff):
    cores = [np.asarray(core) for core in cores]
    discarded_weight = 0.0
    maximum_bond = 1
    for site in range(len(cores) - 1):
        left, physical, right = cores[site].shape
        matrix = cores[site].reshape(left * physical, right)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = (
            cutoff * singular_values[0] if singular_values.size else 0.0
        )
        keep = min(
            max_bond_dim,
            max(1, int(np.count_nonzero(singular_values > threshold))),
        )
        total = float(np.sum(singular_values**2))
        if total:
            discarded_weight += float(
                np.sum(singular_values[keep:] ** 2) / total
            )
        cores[site] = u[:, :keep].reshape(left, physical, keep)
        transfer = singular_values[:keep, None] * vh[:keep]
        cores[site + 1] = np.tensordot(
            transfer,
            cores[site + 1],
            axes=([1], [0]),
        )
        maximum_bond = max(maximum_bond, keep)
    return cores, maximum_bond, discarded_weight


def _apply_row(cores, row_cells, boundary, max_bond_dim, cutoff):
    updated = []
    last_column = len(cores) - 1
    for column, (core, cell) in enumerate(zip(cores, row_cells)):
        applied = np.einsum(
            "aub,lrud->aldbr",
            core,
            cell,
            optimize=True,
        )
        if column == 0:
            applied = np.einsum(
                "aldbr,l->adbr",
                applied,
                boundary,
                optimize=True,
            )
        if column == last_column:
            if column == 0:
                applied = np.einsum(
                    "adbr,r->adb",
                    applied,
                    boundary,
                    optimize=True,
                )
            else:
                applied = np.einsum(
                    "aldbr,r->aldb",
                    applied,
                    boundary,
                    optimize=True,
                )
        if column == 0 and column == last_column:
            new_core = applied
        elif column == 0:
            new_core = applied.reshape(
                applied.shape[0],
                applied.shape[1],
                applied.shape[2] * applied.shape[3],
            )
        elif column == last_column:
            new_core = applied.reshape(
                applied.shape[0] * applied.shape[1],
                applied.shape[2],
                applied.shape[3],
            )
        else:
            new_core = applied.reshape(
                applied.shape[0] * applied.shape[1],
                applied.shape[2],
                applied.shape[3] * applied.shape[4],
            )
        updated.append(new_core)
    return _compress_mps(updated, max_bond_dim, cutoff)


def contract_plane_window(
    state,
    base_cell,
    size,
    *,
    replacements=None,
    boundary_bond_dim=32,
    cutoff=1.0e-10,
):
    """Contract one open square window with optional local cell insertions."""

    if not isinstance(state, UniformPlaneLETTA):
        raise TypeError("state must be a UniformPlaneLETTA.")
    try:
        size = index(size)
        boundary_bond_dim = index(boundary_bond_dim)
    except TypeError as error:
        raise ValueError("window and boundary dimensions must be integers.") from error
    if size < 1 or boundary_bond_dim <= 0:
        raise ValueError("window and boundary dimensions must be positive.")
    base_cell = np.asarray(base_cell)
    expected = (state.double_layer_bond_dim,) * 4
    if base_cell.shape != expected:
        raise ValueError(f"a plane cell must have shape {expected}.")
    replacements = {} if replacements is None else dict(replacements)
    if any(
        len(coordinate) != 2
        or coordinate[0] < 0
        or coordinate[0] >= size
        or coordinate[1] < 0
        or coordinate[1] >= size
        for coordinate in replacements
    ):
        raise ValueError("a replacement coordinate lies outside the window.")
    if any(np.asarray(cell).shape != expected for cell in replacements.values()):
        raise ValueError("a replacement cell has an incompatible shape.")

    boundary = _boundary_vector(state)
    cores = [
        boundary.reshape(1, boundary.size, 1).copy()
        for _ in range(size)
    ]
    log_scale = 0.0
    maximum_bond = 1
    discarded_weight = 0.0
    for row in range(size):
        row_cells = [
            replacements.get((row, column), base_cell)
            for column in range(size)
        ]
        cores, row_maximum, row_discarded = _apply_row(
            cores,
            row_cells,
            boundary,
            boundary_bond_dim,
            cutoff,
        )
        norm = _mps_norm(cores)
        if not np.isfinite(norm):
            raise ValueError(
                "the plane boundary contraction became numerically zero."
            )
        if norm <= np.finfo(float).tiny:
            return BoundaryContraction(
                log_magnitude=-np.inf,
                phase=0.0j,
                maximum_bond_dimension=max(maximum_bond, row_maximum),
                discarded_weight=discarded_weight + row_discarded,
            )
        cores[0] = cores[0] / norm
        log_scale += np.log(norm)
        maximum_bond = max(maximum_bond, row_maximum)
        discarded_weight += row_discarded

    value = np.ones((1,), dtype=np.result_type(*cores))
    for core in cores:
        matrix = np.einsum(
            "apb,p->ab",
            core,
            boundary,
            optimize=True,
        )
        value = value @ matrix
    scalar = value.reshape(()).item()
    magnitude = abs(scalar)
    if magnitude <= np.finfo(float).tiny:
        return BoundaryContraction(
            log_magnitude=-np.inf,
            phase=0.0j,
            maximum_bond_dimension=maximum_bond,
            discarded_weight=discarded_weight,
        )
    return BoundaryContraction(
        log_magnitude=float(log_scale + np.log(magnitude)),
        phase=complex(scalar / magnitude),
        maximum_bond_dimension=maximum_bond,
        discarded_weight=discarded_weight,
    )


def contraction_ratio(numerator, denominator):
    """Return a stable ratio of two normalized boundary contractions."""

    if not isinstance(numerator, BoundaryContraction) or not isinstance(
        denominator,
        BoundaryContraction,
    ):
        raise TypeError("contraction_ratio expects BoundaryContraction objects.")
    if not np.isfinite(denominator.log_magnitude):
        raise ValueError("the normalization contraction is numerically zero.")
    if not np.isfinite(numerator.log_magnitude):
        return 0.0
    phase = numerator.phase / denominator.phase
    value = phase * np.exp(
        numerator.log_magnitude - denominator.log_magnitude
    )
    if abs(np.imag(value)) > 1.0e-8 * max(1.0, abs(value)):
        raise ValueError("a Hermitian plane expectation acquired a complex value.")
    return float(np.real(value))

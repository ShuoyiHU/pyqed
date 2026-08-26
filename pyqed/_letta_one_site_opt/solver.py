"""One-site DMRG-like sweeps for finite lattice LETTA states."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import linalg, sparse

from .contractions import (
    BlockDiagonalMetric,
    IdentityEnvironmentCache,
    LETTAEnvironmentCache,
    network_operator_matrix,
)
from .operators import LatticeMPO, identity_mpo
from .state import LatticeLETTA


@dataclass(frozen=True)
class LETTADMROptions:
    max_sweeps: int = 8
    tolerance: float = 1.0e-9
    metric_tolerance: float = 1.0e-12
    energy_increase_tolerance: float = 1.0e-9
    start_direction: str = "lr"
    alternate: bool = True
    gauge_mode: str = "qr"
    environment_granularity: str = "site"
    use_sparse_mpo: bool = True
    matrix_free: bool = False
    boundary_bond_dim: int | None = None
    boundary_cutoff: float = 1.0e-12
    bond_dimension_schedule: tuple[int, ...] | None = None
    bond_schedule_sweeps: tuple[int, ...] | None = None
    bond_expansion_noise: float = 1.0e-3
    verbosity: int = 0


@dataclass(frozen=True)
class LETTASiteUpdate:
    site: int
    local_energy: float
    energy: float
    metric_rank: int
    local_dimension: int
    residual_norm: float
    accepted: bool
    full_local_dimension: int | None = None


@dataclass(frozen=True)
class LETTASweep:
    sweep: int
    direction: str
    energy: float
    energy_change: float
    energy_density_change: float
    bond_dimension: int
    updates: tuple[LETTASiteUpdate, ...]


@dataclass(frozen=True)
class LETTADMRGResult:
    state: LatticeLETTA
    energy: float
    converged: bool
    sweeps: int
    history: tuple[LETTASweep, ...]
    message: str
    max_boundary_discarded_weight: float = 0.0


def _allocate_schedule_sweeps(number_of_stages, max_sweeps):
    if number_of_stages <= 0:
        raise ValueError("a bond schedule must contain at least one stage.")
    if max_sweeps < number_of_stages:
        raise ValueError("max_sweeps must allow at least one sweep per stage.")
    if number_of_stages == 1:
        return (max_sweeps,)
    lower = [
        max(1, int(round(max_sweeps * 0.2 / 2 ** (number_of_stages - 2 - i))))
        for i in range(number_of_stages - 1)
    ]
    while sum(lower) >= max_sweeps:
        largest = max(range(len(lower)), key=lower.__getitem__)
        if lower[largest] == 1:
            raise ValueError("max_sweeps must allow a final target-bond sweep.")
        lower[largest] -= 1
    return tuple(lower + [max_sweeps - sum(lower)])


def automatic_bond_schedule(target_bond_dim, max_sweeps):
    """Return a doubling bond schedule and a cost-aware sweep allocation."""

    target_bond_dim = int(target_bond_dim)
    max_sweeps = int(max_sweeps)
    if target_bond_dim <= 0 or max_sweeps <= 0:
        raise ValueError("target_bond_dim and max_sweeps must be positive.")
    if target_bond_dim == 1:
        dimensions = (1,)
    else:
        values = []
        dimension = 2
        while dimension < target_bond_dim:
            values.append(dimension)
            dimension *= 2
        values.append(target_bond_dim)
        dimensions = tuple(values)
    if len(dimensions) > max_sweeps:
        dimensions = dimensions[-max_sweeps:]
    return dimensions, _allocate_schedule_sweeps(len(dimensions), max_sweeps)


def _lowest_generalized_eigenpair(hamiltonian, metric, metric_tolerance):
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    if isinstance(metric, BlockDiagonalMetric):
        basis, metric_rank = metric.whitening_basis(metric_tolerance)
    else:
        metric = 0.5 * (metric + metric.conj().T)
        metric_values, metric_vectors = np.linalg.eigh(metric)
        scale = max(float(metric_values[-1]), 0.0)
        if scale == 0.0:
            raise ValueError("the local LETTA overlap metric has zero rank.")
        relative_cutoff = max(
            float(metric_tolerance),
            np.finfo(float).eps * metric.shape[0],
        )
        cutoff = relative_cutoff * scale
        retained = metric_values > cutoff
        if not np.any(retained):
            raise ValueError("the local LETTA overlap metric has zero rank.")
        basis = metric_vectors[:, retained] / np.sqrt(
            metric_values[retained]
        )[None, :]
        metric_rank = int(np.count_nonzero(retained))
    reduced_hamiltonian = basis.conj().T @ hamiltonian @ basis
    reduced_hamiltonian = 0.5 * (
        reduced_hamiltonian + reduced_hamiltonian.conj().T
    )
    values, vectors = linalg.eigh(
        reduced_hamiltonian,
        subset_by_index=[0, 0],
        check_finite=False,
    )
    vector = basis @ vectors[:, 0]
    norm = np.vdot(vector, metric @ vector)
    if np.real(norm) <= np.finfo(float).tiny:
        raise ValueError("the optimized local LETTA tensor has zero norm.")
    vector /= np.sqrt(norm)
    energy = float(np.real(values[0]))
    residual = np.linalg.norm(hamiltonian @ vector - energy * (metric @ vector))
    return energy, vector, metric_rank, float(residual)


def _validate_hamiltonian(hamiltonian, state):
    if isinstance(hamiltonian, LatticeMPO):
        if hamiltonian.nsites != state.nsites:
            raise ValueError("Hamiltonian MPO length does not match the LETTA state.")
        if hamiltonian.physical_dim != state.physical_dim:
            raise ValueError(
                "Hamiltonian MPO physical dimension does not match the LETTA state."
            )
        if (
            hamiltonian.lattice_shape is not None
            and hamiltonian.lattice_shape != state.lattice_shape
        ):
            raise ValueError(
                "Hamiltonian MPO lattice shape does not match the LETTA state."
            )
        return hamiltonian
    dimension = state.hilbert_dim
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError(
            "hamiltonian shape does not match the lattice Hilbert dimension."
        )
    if sparse.issparse(hamiltonian):
        difference = hamiltonian - hamiltonian.getH()
        if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-10:
            raise ValueError("hamiltonian must be Hermitian.")
        return hamiltonian
    matrix = np.asarray(hamiltonian)
    if not np.allclose(matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("hamiltonian must be Hermitian.")
    return matrix


def _local_update_vector(old_tensor, effective_hamiltonian, effective_metric, options):
    local_energy, candidate, metric_rank, _residual = (
        _lowest_generalized_eigenpair(
            effective_hamiltonian,
            effective_metric,
            options.metric_tolerance,
        )
    )
    old_vector = old_tensor.reshape(-1)
    trial_basis = np.column_stack([old_vector, candidate])
    trial_hamiltonian = trial_basis.conj().T @ effective_hamiltonian @ trial_basis
    trial_metric = trial_basis.conj().T @ (
        effective_metric @ trial_basis
    )
    local_energy, coefficients, _trial_rank, _trial_residual = (
        _lowest_generalized_eigenpair(
            trial_hamiltonian,
            trial_metric,
            options.metric_tolerance,
        )
    )
    vector = trial_basis @ coefficients
    vector /= np.sqrt(np.vdot(vector, effective_metric @ vector))
    residual = np.linalg.norm(
        effective_hamiltonian @ vector
        - local_energy * (effective_metric @ vector)
    )
    return local_energy, vector, metric_rank, float(residual)


def _active_local_indices(state, site):
    if state.symmetry is None:
        return None
    return state.symmetry_indices(site)


def _restrict_metric(metric, indices):
    if indices is None:
        return metric
    if isinstance(metric, BlockDiagonalMetric):
        return metric.restrict(indices)
    return metric[np.ix_(indices, indices)]


def _embed_local_vector(vector, indices, full_dimension):
    if indices is None:
        return np.asarray(vector)
    result = np.zeros(full_dimension, dtype=np.asarray(vector).dtype)
    result[indices] = vector
    return result


def _optimize_site(state, hamiltonian, site, options):
    old_tensor = state.tensors[site].copy()
    indices = _active_local_indices(state, site)
    old_energy = state.expectation(hamiltonian)
    if isinstance(hamiltonian, LatticeMPO):
        effective_hamiltonian = network_operator_matrix(
            state,
            hamiltonian,
            site,
        )
        effective_metric = network_operator_matrix(
            state,
            identity_mpo(
                state.nsites,
                state.physical_dim,
                lattice_shape=state.lattice_shape,
            ),
            site,
        )
    else:
        frame = state.local_frame(site)
        if indices is not None:
            frame = frame[:, indices]
        applied_frame = hamiltonian @ frame
        effective_hamiltonian = frame.conj().T @ applied_frame
        effective_metric = frame.conj().T @ frame
    if indices is not None and isinstance(hamiltonian, LatticeMPO):
        effective_hamiltonian = effective_hamiltonian[np.ix_(indices, indices)]
        effective_metric = effective_metric[np.ix_(indices, indices)]
    old_local = old_tensor.reshape(-1)
    if indices is not None:
        old_local = old_local[indices]
    local_energy, vector, metric_rank, residual = _local_update_vector(
        old_local,
        effective_hamiltonian,
        effective_metric,
        options,
    )
    state.tensors[site] = _embed_local_vector(
        vector, indices, old_tensor.size
    ).reshape(old_tensor.shape)
    state.normalize()
    energy = state.expectation(hamiltonian)
    accepted = True
    if energy > old_energy + options.energy_increase_tolerance:
        state.tensors[site] = old_tensor
        state.normalize()
        energy = state.expectation(hamiltonian)
        accepted = False
    return LETTASiteUpdate(
        site=site,
        local_energy=local_energy,
        energy=energy,
        metric_rank=metric_rank,
        local_dimension=old_local.size,
        residual_norm=residual,
        accepted=accepted,
        full_local_dimension=old_tensor.size,
    )


def _rayleigh_quotient(vector, hamiltonian, metric):
    numerator = np.vdot(vector, hamiltonian @ vector)
    denominator = np.vdot(vector, metric @ vector)
    return float(np.real(numerator / denominator))


def _optimize_site_from_environments(
    state,
    site,
    effective_hamiltonian,
    effective_metric,
    options,
):
    old_tensor = state.tensors[site].copy()
    indices = _active_local_indices(state, site)
    old_vector = old_tensor.reshape(-1)
    if indices is not None:
        old_vector = old_vector[indices]
        effective_hamiltonian = effective_hamiltonian[np.ix_(indices, indices)]
        effective_metric = _restrict_metric(effective_metric, indices)
    old_energy = _rayleigh_quotient(
        old_vector,
        effective_hamiltonian,
        effective_metric,
    )
    local_energy, vector, metric_rank, residual = _local_update_vector(
        old_vector,
        effective_hamiltonian,
        effective_metric,
        options,
    )
    energy = _rayleigh_quotient(
        vector,
        effective_hamiltonian,
        effective_metric,
    )
    accepted = energy <= old_energy + options.energy_increase_tolerance
    if accepted:
        state.tensors[site] = _embed_local_vector(
            vector, indices, old_tensor.size
        ).reshape(old_tensor.shape)
    else:
        energy = old_energy
    return LETTASiteUpdate(
        site=site,
        local_energy=local_energy,
        energy=energy,
        metric_rank=metric_rank,
        local_dimension=old_vector.size,
        residual_norm=residual,
        accepted=accepted,
        full_local_dimension=old_tensor.size,
    )


def _lowest_matrix_free_eigenpair(action, metric, metric_tolerance):
    if isinstance(metric, BlockDiagonalMetric):
        basis, rank = metric.whitening_basis(metric_tolerance)
        metric_dtype = metric.dtype
    else:
        metric = 0.5 * (metric + metric.conj().T)
        metric_values, metric_vectors = np.linalg.eigh(metric)
        scale = max(float(metric_values[-1]), 0.0)
        if scale == 0.0:
            raise ValueError("the local LETTA overlap metric has zero rank.")
        relative_cutoff = max(
            float(metric_tolerance),
            np.finfo(float).eps * metric.shape[0],
        )
        retained = metric_values > relative_cutoff * scale
        basis = metric_vectors[:, retained] / np.sqrt(
            metric_values[retained]
        )[None, :]
        rank = basis.shape[1]
        metric_dtype = metric.dtype

    def reduced_action(vector):
        return basis.conj().T @ action(basis @ vector)

    if rank == 1:
        reduced_vector = np.ones(1, dtype=metric_dtype)
    else:
        reduced_vector = _lowest_lanczos_vector(
            reduced_action,
            rank,
            metric_dtype,
        )
    vector = basis @ reduced_vector
    vector /= np.sqrt(np.vdot(vector, metric @ vector))
    applied = action(vector)
    energy = float(np.real(np.vdot(vector, applied)))
    residual = np.linalg.norm(applied - energy * (metric @ vector))
    return energy, vector, rank, float(residual)


def _lowest_lanczos_vector(action, dimension, dtype, tolerance=1.0e-11):
    """Return the lowest Ritz vector using stable full reorthogonalization."""

    vector = np.ones(dimension, dtype=dtype)
    vector /= np.linalg.norm(vector)
    previous = np.zeros_like(vector)
    previous_beta = 0.0
    basis = []
    diagonal = []
    off_diagonal = []
    coefficients = np.ones(1, dtype=dtype)

    for iteration in range(dimension):
        basis.append(vector)
        residual = np.asarray(action(vector), dtype=dtype)
        if iteration:
            residual -= previous_beta * previous
        alpha = float(np.real(np.vdot(vector, residual)))
        diagonal.append(alpha)
        residual -= alpha * vector
        for _ in range(2):
            for basis_vector in basis:
                residual -= basis_vector * np.vdot(basis_vector, residual)
        beta = float(np.linalg.norm(residual))

        tridiagonal = np.diag(diagonal)
        if off_diagonal:
            indices = np.arange(len(off_diagonal))
            tridiagonal[indices, indices + 1] = off_diagonal
            tridiagonal[indices + 1, indices] = off_diagonal
        values, vectors = np.linalg.eigh(tridiagonal)
        coefficients = vectors[:, 0]
        ritz_scale = max(1.0, abs(float(values[0])))
        if (
            beta <= np.finfo(float).eps
            or (
                iteration >= 3
                and beta * abs(coefficients[-1]) <= tolerance * ritz_scale
            )
        ):
            break
        off_diagonal.append(beta)
        previous = vector
        previous_beta = beta
        vector = residual / beta

    result = np.column_stack(basis) @ coefficients
    return result / np.linalg.norm(result)


def _optimize_site_matrix_free(
    state,
    site,
    action,
    effective_metric,
    options,
    *,
    indices=None,
):
    old_tensor = state.tensors[site].copy()
    old_vector = old_tensor.reshape(-1)
    if indices is not None:
        old_vector = old_vector[indices]
    old_energy = float(
        np.real(
            np.vdot(old_vector, action(old_vector))
            / np.vdot(old_vector, effective_metric @ old_vector)
        )
    )
    _energy, candidate, metric_rank, _residual = (
        _lowest_matrix_free_eigenpair(
            action,
            effective_metric,
            options.metric_tolerance,
        )
    )
    trial_basis = np.column_stack([old_vector, candidate])
    applied_basis = np.column_stack(
        [action(trial_basis[:, column]) for column in range(2)]
    )
    trial_hamiltonian = trial_basis.conj().T @ applied_basis
    trial_metric = trial_basis.conj().T @ (
        effective_metric @ trial_basis
    )
    local_energy, coefficients, _rank, _residual = _lowest_generalized_eigenpair(
        trial_hamiltonian,
        trial_metric,
        options.metric_tolerance,
    )
    vector = trial_basis @ coefficients
    vector /= np.sqrt(np.vdot(vector, effective_metric @ vector))
    applied = action(vector)
    residual = np.linalg.norm(
        applied - local_energy * (effective_metric @ vector)
    )
    energy = float(np.real(np.vdot(vector, applied)))
    accepted = energy <= old_energy + options.energy_increase_tolerance
    if accepted:
        state.tensors[site] = _embed_local_vector(
            vector, indices, old_tensor.size
        ).reshape(old_tensor.shape)
    else:
        energy = old_energy
    return LETTASiteUpdate(
        site=site,
        local_energy=local_energy,
        energy=energy,
        metric_rank=metric_rank,
        local_dimension=old_vector.size,
        residual_norm=float(residual),
        accepted=accepted,
        full_local_dimension=old_tensor.size,
    )


def _update_from_cached_environments(
    state,
    site,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    options,
):
    compression_error = max(
        hamiltonian_cache.compression_errors
        + metric_cache.compression_errors,
        default=0.0,
    )
    local_options = options
    if compression_error > 0.0:
        adaptive_tolerance = max(
            options.metric_tolerance,
            min(0.1, 1.0e-2 * compression_error),
        )
        local_options = replace(options, metric_tolerance=adaptive_tolerance)
    effective_metric = metric_cache.effective_metric(
        metric_left,
        metric_right,
        site,
    )
    indices = _active_local_indices(state, site)
    if options.matrix_free:
        if indices is None:
            action = lambda vector: hamiltonian_cache.effective_action(
                hamiltonian_left,
                hamiltonian_right,
                site,
                vector,
            )
        else:
            full_dimension = state.tensors[site].size

            def action(vector):
                full = _embed_local_vector(vector, indices, full_dimension)
                applied = hamiltonian_cache.effective_action(
                    hamiltonian_left,
                    hamiltonian_right,
                    site,
                    full,
                )
                return applied[indices]

            effective_metric = _restrict_metric(effective_metric, indices)
        return _optimize_site_matrix_free(
            state,
            site,
            action,
            effective_metric,
            local_options,
            indices=indices,
        )
    effective_hamiltonian = hamiltonian_cache.effective_matrix(
        hamiltonian_left,
        hamiltonian_right,
        site,
    )
    return _optimize_site_from_environments(
        state,
        site,
        effective_hamiltonian,
        effective_metric,
        local_options,
    )


def _shift_scalar_gauge(state, site, direction):
    neighbor = site + 1 if direction == "lr" else site - 1
    if neighbor < 0 or neighbor >= state.nsites:
        return
    tensor_norm = np.linalg.norm(state.tensors[site])
    if tensor_norm <= np.finfo(float).tiny:
        raise ValueError("cannot shift the gauge of a zero LETTA tensor.")
    state.tensors[site] = state.tensors[site] / tensor_norm
    state.tensors[neighbor] = state.tensors[neighbor] * tensor_norm


def _shift_virtual_gauge(state, site, direction, mode="qr"):
    if mode == "none":
        return
    if mode == "scalar":
        _shift_scalar_gauge(state, site, direction)
        return
    if mode != "qr":
        raise ValueError("gauge_mode must be 'qr', 'scalar', or 'none'.")

    if state.symmetry is not None:
        _shift_symmetry_virtual_gauge(state, site, direction)
        return

    if direction == "lr":
        neighbor = site + 1
        if neighbor >= state.nsites:
            return
        tensor = state.tensors[site]
        matrix = tensor.reshape(-1, tensor.shape[-1])
        q, r = np.linalg.qr(matrix, mode="reduced")
        state.tensors[site] = q.reshape(tensor.shape[:-1] + (q.shape[1],))
        state.tensors[neighbor] = np.tensordot(
            r,
            state.tensors[neighbor],
            axes=([1], [0]),
        )
        return

    if direction != "rl":
        raise ValueError("direction must be 'lr' or 'rl'.")
    neighbor = site - 1
    if neighbor < 0:
        return
    tensor = state.tensors[site]
    matrix = tensor.reshape(tensor.shape[0], -1)
    q, r = np.linalg.qr(matrix.T, mode="reduced")
    state.tensors[site] = q.T.reshape((q.shape[1],) + tensor.shape[1:])
    state.tensors[neighbor] = np.tensordot(
        state.tensors[neighbor],
        r.T,
        axes=([-1], [0]),
    )


def _charge_groups(charges):
    groups = {}
    for position, charge in enumerate(charges):
        groups.setdefault(charge, []).append(position)
    return tuple(np.asarray(group, dtype=int) for group in groups.values())


def _shift_symmetry_virtual_gauge(state, site, direction):
    """Apply QR independently in every virtual charge block."""

    if direction == "lr":
        neighbor = site + 1
        if neighbor >= state.nsites:
            return
        tensor = state.tensors[site]
        bond_dim = tensor.shape[-1]
        matrix = tensor.reshape(-1, bond_dim)
        mask = state.symmetry_mask(site).reshape(-1, bond_dim)
        gauged = np.zeros_like(matrix)
        transfer = np.zeros((bond_dim, bond_dim), dtype=tensor.dtype)
        for columns in _charge_groups(state.right_virtual_charges(site)):
            rows = np.flatnonzero(np.any(mask[:, columns], axis=1))
            block = matrix[np.ix_(rows, columns)]
            q, r = np.linalg.qr(block, mode="reduced")
            width = columns.size
            padded_q = np.zeros((rows.size, width), dtype=q.dtype)
            padded_r = np.zeros((width, width), dtype=r.dtype)
            rank = q.shape[1]
            padded_q[:, :rank] = q
            padded_r[:rank] = r
            gauged[np.ix_(rows, columns)] = padded_q
            transfer[np.ix_(columns, columns)] = padded_r
        state.tensors[site] = gauged.reshape(tensor.shape)
        state.tensors[neighbor] = np.tensordot(
            transfer, state.tensors[neighbor], axes=([1], [0])
        )
        state.enforce_symmetry()
        return

    if direction != "rl":
        raise ValueError("direction must be 'lr' or 'rl'.")
    neighbor = site - 1
    if neighbor < 0:
        return
    tensor = state.tensors[site]
    bond_dim = tensor.shape[0]
    matrix = tensor.reshape(bond_dim, -1)
    mask = state.symmetry_mask(site).reshape(bond_dim, -1)
    gauged = np.zeros_like(matrix)
    transfer = np.zeros((bond_dim, bond_dim), dtype=tensor.dtype)
    for rows in _charge_groups(state.left_virtual_charges(site)):
        columns = np.flatnonzero(np.any(mask[rows, :], axis=0))
        block = matrix[np.ix_(rows, columns)]
        q, r = np.linalg.qr(block.T, mode="reduced")
        width = rows.size
        padded_q = np.zeros((columns.size, width), dtype=q.dtype)
        padded_r = np.zeros((width, width), dtype=r.dtype)
        rank = q.shape[1]
        padded_q[:, :rank] = q
        padded_r[:rank] = r
        gauged[np.ix_(rows, columns)] = padded_q.T
        transfer[np.ix_(rows, rows)] = padded_r.T
    state.tensors[site] = gauged.reshape(tensor.shape)
    state.tensors[neighbor] = np.tensordot(
        state.tensors[neighbor], transfer, axes=([-1], [0])
    )
    state.enforce_symmetry()


def _cached_mpo_sweep(
    state,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_environments,
    metric_environments,
    direction,
    options,
):
    if options.environment_granularity == "column":
        return _cached_mpo_column_sweep(
            state,
            hamiltonian_cache,
            metric_cache,
            hamiltonian_environments,
            metric_environments,
            direction,
            options,
        )
    updates = []

    if direction == "lr":
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        hamiltonian_environments[0] = hamiltonian_boundary
        metric_environments[0] = metric_boundary
        sites = range(state.nsites)
        for site in sites:
            updates.append(
                _update_from_cached_environments(
                    state,
                    site,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_boundary,
                    hamiltonian_environments[site + 1],
                    metric_boundary,
                    metric_environments[site + 1],
                    options,
                )
            )
            _shift_virtual_gauge(state, site, direction, options.gauge_mode)
            hamiltonian_boundary = hamiltonian_cache.extend_left(
                hamiltonian_boundary,
                site,
            )
            metric_boundary = metric_cache.extend_left(
                metric_boundary,
                site,
            )
            hamiltonian_environments[site + 1] = hamiltonian_boundary
            metric_environments[site + 1] = metric_boundary
    else:
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        hamiltonian_environments[-1] = hamiltonian_boundary
        metric_environments[-1] = metric_boundary
        sites = range(state.nsites - 1, -1, -1)
        for site in sites:
            updates.append(
                _update_from_cached_environments(
                    state,
                    site,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_environments[site],
                    hamiltonian_boundary,
                    metric_environments[site],
                    metric_boundary,
                    options,
                )
            )
            _shift_virtual_gauge(state, site, direction, options.gauge_mode)
            hamiltonian_boundary = hamiltonian_cache.extend_right(
                hamiltonian_boundary,
                site,
            )
            metric_boundary = metric_cache.extend_right(
                metric_boundary,
                site,
            )
            hamiltonian_environments[site] = hamiltonian_boundary
            metric_environments[site] = metric_boundary

    energy = float(np.real(hamiltonian_boundary / metric_boundary))
    return (
        tuple(updates),
        energy,
        hamiltonian_environments,
        metric_environments,
    )


def _build_environment_checkpoints(cache, direction, block_size):
    nsites = cache.state.nsites
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if direction == "rl":
        environment = cache.scalar_boundary()
        checkpoints = {nsites: environment}
        for site in range(nsites - 1, -1, -1):
            environment = cache.extend_right(environment, site)
            if site % block_size == 0:
                checkpoints[site] = environment
        return checkpoints
    if direction == "lr":
        environment = cache.scalar_boundary()
        checkpoints = {0: environment}
        for site in range(nsites):
            environment = cache.extend_left(environment, site)
            cut = site + 1
            if cut % block_size == 0 or cut == nsites:
                checkpoints[cut] = environment
        return checkpoints
    raise ValueError("direction must be 'lr' or 'rl'.")


def _cached_mpo_column_sweep(
    state,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_checkpoints,
    metric_checkpoints,
    direction,
    options,
):
    block_size = int(np.prod(state.lattice_shape[1:]))
    nsites = state.nsites
    updates = []

    if direction == "lr":
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        next_hamiltonian = {0: hamiltonian_boundary}
        next_metric = {0: metric_boundary}
        for start in range(0, nsites, block_size):
            end = min(start + block_size, nsites)
            local_hamiltonian = {end: hamiltonian_checkpoints[end]}
            local_metric = {end: metric_checkpoints[end]}
            for site in range(end - 1, start, -1):
                local_hamiltonian[site] = hamiltonian_cache.extend_right(
                    local_hamiltonian[site + 1], site
                )
                local_metric[site] = metric_cache.extend_right(
                    local_metric[site + 1], site
                )
            for site in range(start, end):
                updates.append(
                    _update_from_cached_environments(
                        state,
                        site,
                        hamiltonian_cache,
                        metric_cache,
                        hamiltonian_boundary,
                        local_hamiltonian[site + 1],
                        metric_boundary,
                        local_metric[site + 1],
                        options,
                    )
                )
                _shift_virtual_gauge(state, site, direction, options.gauge_mode)
                hamiltonian_boundary = hamiltonian_cache.extend_left(
                    hamiltonian_boundary, site
                )
                metric_boundary = metric_cache.extend_left(metric_boundary, site)
            next_hamiltonian[end] = hamiltonian_boundary
            next_metric[end] = metric_boundary
    else:
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        next_hamiltonian = {nsites: hamiltonian_boundary}
        next_metric = {nsites: metric_boundary}
        block_starts = range(
            ((nsites - 1) // block_size) * block_size,
            -1,
            -block_size,
        )
        for start in block_starts:
            end = min(start + block_size, nsites)
            local_hamiltonian = {start: hamiltonian_checkpoints[start]}
            local_metric = {start: metric_checkpoints[start]}
            for site in range(start, end - 1):
                local_hamiltonian[site + 1] = hamiltonian_cache.extend_left(
                    local_hamiltonian[site], site
                )
                local_metric[site + 1] = metric_cache.extend_left(
                    local_metric[site], site
                )
            for site in range(end - 1, start - 1, -1):
                updates.append(
                    _update_from_cached_environments(
                        state,
                        site,
                        hamiltonian_cache,
                        metric_cache,
                        local_hamiltonian[site],
                        hamiltonian_boundary,
                        local_metric[site],
                        metric_boundary,
                        options,
                    )
                )
                _shift_virtual_gauge(state, site, direction, options.gauge_mode)
                hamiltonian_boundary = hamiltonian_cache.extend_right(
                    hamiltonian_boundary, site
                )
                metric_boundary = metric_cache.extend_right(metric_boundary, site)
            next_hamiltonian[start] = hamiltonian_boundary
            next_metric[start] = metric_boundary

    energy = float(np.real(hamiltonian_boundary / metric_boundary))
    return tuple(updates), energy, next_hamiltonian, next_metric


def _validated_bond_schedule(options, bond_dim, state):
    if options.bond_dimension_schedule is None:
        if options.bond_schedule_sweeps is not None:
            raise ValueError(
                "bond_schedule_sweeps requires bond_dimension_schedule."
            )
        return None
    try:
        dimensions = tuple(int(value) for value in options.bond_dimension_schedule)
    except (TypeError, ValueError) as error:
        raise ValueError("bond dimensions must be integers.") from error
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError("bond dimensions must be positive.")
    if any(right <= left for left, right in zip(dimensions, dimensions[1:])):
        raise ValueError("bond dimensions must be strictly increasing.")
    if state is None:
        if dimensions[-1] != int(bond_dim):
            raise ValueError(
                "the final scheduled bond dimension must equal bond_dim."
            )
    else:
        current = max(state.bond_dimensions, default=1)
        if dimensions[0] < current:
            raise ValueError("a bond schedule cannot shrink the supplied state.")

    if options.bond_schedule_sweeps is None:
        sweeps = _allocate_schedule_sweeps(len(dimensions), options.max_sweeps)
    else:
        try:
            sweeps = tuple(int(value) for value in options.bond_schedule_sweeps)
        except (TypeError, ValueError) as error:
            raise ValueError("bond schedule sweep counts must be integers.") from error
        if len(sweeps) != len(dimensions) or any(value <= 0 for value in sweeps):
            raise ValueError(
                "bond schedule sweep counts must be positive and match the stages."
            )
        if sum(sweeps) != options.max_sweeps:
            raise ValueError(
                "bond schedule sweep counts must sum to max_sweeps."
            )
    return dimensions, sweeps


def _letta_dmrg_with_bond_schedule(
    hamiltonian,
    *,
    state,
    lattice_shape,
    coordinates,
    physical_dim,
    bond_dim,
    seed,
    real,
    symmetry,
    bond_charges,
    options,
    schedule,
):
    dimensions, stage_sweeps = schedule
    current_state = state
    history = []
    max_discarded_weight = 0.0
    start_direction = options.start_direction.lower()
    final_result = None

    for stage, (dimension, sweeps) in enumerate(
        zip(dimensions, stage_sweeps)
    ):
        if current_state is not None:
            current_dimension = max(current_state.bond_dimensions, default=1)
            if dimension > current_dimension:
                expansion_seed = None if seed is None else int(seed) + stage
                current_state = current_state.expand_bond_dimension(
                    dimension,
                    noise=options.bond_expansion_noise,
                    seed=expansion_seed,
                )
        stage_options = replace(
            options,
            max_sweeps=sweeps,
            start_direction=start_direction,
            bond_dimension_schedule=None,
            bond_schedule_sweeps=None,
        )
        final_result = letta_dmrg(
            hamiltonian,
            state=current_state,
            lattice_shape=lattice_shape,
            coordinates=coordinates,
            physical_dim=physical_dim,
            bond_dim=dimension,
            seed=seed,
            real=real,
            symmetry=symmetry,
            bond_charges=bond_charges if current_state is None else None,
            options=stage_options,
        )
        offset = len(history)
        history.extend(
            replace(
                sweep,
                sweep=offset + local_sweep,
                bond_dimension=dimension,
            )
            for local_sweep, sweep in enumerate(final_result.history, start=1)
        )
        current_state = final_result.state
        max_discarded_weight = max(
            max_discarded_weight,
            final_result.max_boundary_discarded_weight,
        )
        if options.alternate and final_result.sweeps % 2:
            start_direction = "rl" if start_direction == "lr" else "lr"

    return LETTADMRGResult(
        state=final_result.state,
        energy=final_result.energy,
        converged=final_result.converged,
        sweeps=len(history),
        history=tuple(history),
        message=final_result.message,
        max_boundary_discarded_weight=max_discarded_weight,
    )


def letta_dmrg(
    hamiltonian,
    *,
    state=None,
    lattice_shape=None,
    coordinates=None,
    physical_dim=2,
    bond_dim=2,
    seed=None,
    real=True,
    symmetry=None,
    bond_charges=None,
    options=None,
):
    """Optimize a finite lattice LETTA by alternating one-site eigensweeps."""

    options = LETTADMROptions() if options is None else options
    if not isinstance(options, LETTADMROptions):
        raise TypeError("options must be a LETTADMROptions instance.")
    if options.max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive.")
    if (
        options.tolerance <= 0.0
        or options.metric_tolerance <= 0.0
        or options.energy_increase_tolerance < 0.0
    ):
        raise ValueError("solver tolerances must be positive.")
    if options.gauge_mode not in {"qr", "scalar", "none"}:
        raise ValueError("gauge_mode must be 'qr', 'scalar', or 'none'.")
    if options.environment_granularity not in {"site", "column"}:
        raise ValueError("environment_granularity must be 'site' or 'column'.")
    if options.boundary_bond_dim is not None and options.boundary_bond_dim <= 0:
        raise ValueError("boundary_bond_dim must be positive when provided.")
    if options.boundary_cutoff < 0.0:
        raise ValueError("boundary_cutoff must be nonnegative.")
    if options.bond_expansion_noise < 0.0:
        raise ValueError("bond_expansion_noise must be nonnegative.")
    schedule = _validated_bond_schedule(options, bond_dim, state)
    if schedule is not None:
        return _letta_dmrg_with_bond_schedule(
            hamiltonian,
            state=state,
            lattice_shape=lattice_shape,
            coordinates=coordinates,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            seed=seed,
            real=real,
            symmetry=symmetry,
            bond_charges=bond_charges,
            options=options,
            schedule=schedule,
        )
    direction = options.start_direction.lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("start_direction must be 'lr' or 'rl'.")
    if state is None:
        if lattice_shape is None:
            raise ValueError("lattice_shape is required when state is omitted.")
        state = LatticeLETTA.random(
            lattice_shape,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            seed=seed,
            real=real,
            coordinates=coordinates,
            symmetry=symmetry,
            bond_charges=bond_charges,
        )
    elif not isinstance(state, LatticeLETTA):
        raise TypeError("state must be a LatticeLETTA.")
    else:
        if symmetry is not None and state.symmetry != symmetry:
            raise ValueError("supplied state and symmetry do not match.")
        if bond_charges is not None and state.bond_charges != tuple(
            tuple(charges) for charges in bond_charges
        ):
            raise ValueError("supplied state and bond_charges do not match.")
        state = state.copy()
    hamiltonian = _validate_hamiltonian(hamiltonian, state)
    nominal_bond_dimension = max(state.bond_dimensions, default=1)

    previous_energy = None
    hamiltonian_cache = None
    metric_cache = None
    hamiltonian_environments = None
    metric_environments = None
    if isinstance(hamiltonian, LatticeMPO):
        hamiltonian_cache = LETTAEnvironmentCache(
            state,
            hamiltonian,
            use_sparse_mpo=options.use_sparse_mpo,
            boundary_bond_dim=options.boundary_bond_dim,
            boundary_cutoff=options.boundary_cutoff,
        )
        metric_cache = IdentityEnvironmentCache(
            state,
            boundary_bond_dim=options.boundary_bond_dim,
            boundary_cutoff=options.boundary_cutoff,
        )
        if options.environment_granularity == "column":
            block_size = int(np.prod(state.lattice_shape[1:]))
            build_direction = "rl" if direction == "lr" else "lr"
            hamiltonian_environments = _build_environment_checkpoints(
                hamiltonian_cache, build_direction, block_size
            )
            metric_environments = _build_environment_checkpoints(
                metric_cache, build_direction, block_size
            )
        elif direction == "lr":
            hamiltonian_environments = hamiltonian_cache.build_right_environments()
            metric_environments = metric_cache.build_right_environments()
        else:
            hamiltonian_environments = hamiltonian_cache.build_left_environments()
            metric_environments = metric_cache.build_left_environments()
        if options.boundary_bond_dim is None:
            full_cut = 0 if direction == "lr" else state.nsites
            previous_energy = float(
                np.real(
                    hamiltonian_environments[full_cut]
                    / metric_environments[full_cut]
                )
            )
        else:
            previous_energy = state.expectation(hamiltonian)
    else:
        previous_energy = state.expectation(hamiltonian)
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    for sweep in range(1, options.max_sweeps + 1):
        compressed_sweep = (
            isinstance(hamiltonian, LatticeMPO)
            and options.boundary_bond_dim is not None
        )
        if compressed_sweep and sweep > 1:
            block_size = int(np.prod(state.lattice_shape[1:]))
            build_direction = "rl" if direction == "lr" else "lr"
            if options.environment_granularity == "column":
                hamiltonian_environments = _build_environment_checkpoints(
                    hamiltonian_cache, build_direction, block_size
                )
                metric_environments = _build_environment_checkpoints(
                    metric_cache, build_direction, block_size
                )
            elif direction == "lr":
                hamiltonian_environments = (
                    hamiltonian_cache.build_right_environments()
                )
                metric_environments = metric_cache.build_right_environments()
            else:
                hamiltonian_environments = (
                    hamiltonian_cache.build_left_environments()
                )
                metric_environments = metric_cache.build_left_environments()
        sweep_start = (
            [tensor.copy() for tensor in state.tensors]
            if compressed_sweep
            else None
        )
        compressed_sweep_rejected = False
        if isinstance(hamiltonian, LatticeMPO):
            (
                updates,
                energy,
                hamiltonian_environments,
                metric_environments,
            ) = _cached_mpo_sweep(
                state,
                hamiltonian_cache,
                metric_cache,
                hamiltonian_environments,
                metric_environments,
                direction,
                options,
            )
            if compressed_sweep:
                exact_sweep_energy = state.expectation(hamiltonian)
                if (
                    exact_sweep_energy
                    > previous_energy + options.energy_increase_tolerance
                ):
                    state.tensors = [
                        tensor.copy() for tensor in sweep_start
                    ]
                    energy = previous_energy
                    compressed_sweep_rejected = True
                else:
                    energy = exact_sweep_energy
        else:
            sites = (
                range(state.nsites)
                if direction == "lr"
                else range(state.nsites - 1, -1, -1)
            )
            updates = tuple(
                _optimize_site(state, hamiltonian, site, options)
                for site in sites
            )
            energy = state.expectation(hamiltonian)
        energy_change = abs(energy - previous_energy)
        energy_density_change = energy_change / state.nsites
        history.append(
            LETTASweep(
                sweep=sweep,
                direction=direction,
                energy=energy,
                energy_change=energy_change,
                energy_density_change=energy_density_change,
                bond_dimension=nominal_bond_dimension,
                updates=updates,
            )
        )
        if options.verbosity:
            print(
                f"lattice LETTA sweep {sweep:3d}  direction={direction}  "
                f"bond={nominal_bond_dimension}  energy={energy:.14f}  "
                f"dE/site={energy_density_change:.3e}"
            )
        if compressed_sweep_rejected:
            message = "STOP: COMPRESSED BOUNDARY SWEEP FAILED EXACT ENERGY CHECK"
            break
        if energy_density_change <= options.tolerance:
            converged = True
            message = "CONVERGENCE: SWEEP ENERGY DENSITY CHANGE <= TOLERANCE"
            break
        previous_energy = energy
        if options.alternate:
            direction = "rl" if direction == "lr" else "lr"
        elif isinstance(hamiltonian, LatticeMPO):
            if options.environment_granularity == "column":
                block_size = int(np.prod(state.lattice_shape[1:]))
                build_direction = "rl" if direction == "lr" else "lr"
                hamiltonian_environments = _build_environment_checkpoints(
                    hamiltonian_cache, build_direction, block_size
                )
                metric_environments = _build_environment_checkpoints(
                    metric_cache, build_direction, block_size
                )
            elif direction == "lr":
                hamiltonian_environments = (
                    hamiltonian_cache.build_right_environments()
                )
                metric_environments = metric_cache.build_right_environments()
            else:
                hamiltonian_environments = (
                    hamiltonian_cache.build_left_environments()
                )
                metric_environments = metric_cache.build_left_environments()

    compression_errors = []
    if hamiltonian_cache is not None:
        compression_errors.extend(hamiltonian_cache.compression_errors)
        compression_errors.extend(metric_cache.compression_errors)
    return LETTADMRGResult(
        state=state,
        energy=float(history[-1].energy),
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
        max_boundary_discarded_weight=max(compression_errors, default=0.0),
    )

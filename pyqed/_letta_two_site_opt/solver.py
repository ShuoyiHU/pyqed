"""Two-site generalized-eigenvalue sweeps for finite lattice LETTA states."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from .._letta_compression import MetricCompressionOptions

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .._letta_one_site_opt.contractions import DiagonalMetric, _equilibrated_metric_factors
from .._letta_one_site_opt.operators import LatticeMPO
from .._letta_one_site_opt.solver import (
    LETTADMROptions,
    _lowest_generalized_eigenpair,
    _shift_gauge_and_extend_metric,
    _shift_virtual_gauge,
    letta_dmrg,
)
from .._letta_one_site_opt.state import LatticeLETTA
from .contractions import (
    IdentityPairEnvironmentCache,
    LETTAPairEnvironmentCache,
)
from .energy_refinement import energy_refine_split
from .pair import LETTAPairLayout, conditional_svd_split
from .truncation import metric_refine


@dataclass(frozen=True)
class LETTATwoSiteOptions:
    compression: MetricCompressionOptions = MetricCompressionOptions()
    max_sweeps: int = 8
    tolerance: float = 1.0e-9
    metric_tolerance: float = 1.0e-10
    energy_increase_tolerance: float = 1.0e-10
    eigensolver_tolerance: float = 1.0e-10
    eigensolver_max_iterations: int = 300
    dense_solver_threshold: int = 64
    split_method: str = "metric-als-energy"
    conditional_svd_cutoff: float = 0.0
    truncation_tolerance: float = 1.0e-10
    truncation_max_iterations: int = 8
    energy_refinement_tolerance: float = 1.0e-10
    energy_refinement_max_iterations: int = 8
    energy_refinement_max_factor_norm_growth: float = 100.0
    coupled_refinement_max_iterations: int = 8
    coupled_refinement_metric_tolerance: float = 1.0e-12
    coupled_refinement_activation_ratio: float = 5.0
    matrix_free: bool = True
    use_sparse_mpo: bool = True
    start_direction: str = "lr"
    alternate: bool = True
    gauge_mode: str = "frontier"
    reduced_sector_growth: bool = False  # open missing SU(2) pair fusion sectors before splitting
    one_site_polish_sweeps: int = 0
    verbosity: int = 0


@dataclass(frozen=True)
class LETTAPairUpdate:
    left_site: int
    right_site: int
    shared_physical_sites: tuple[int, ...]
    old_energy: float
    local_energy: float
    energy: float
    metric_rank: int
    local_dimension: int
    residual_norm: float
    conditional_discarded_weight: float
    metric_truncation_loss: float
    truncation_iterations: int
    energy_refinement_initial_energy: float | None
    energy_refinement_energy: float | None
    energy_refinement_iterations: int
    energy_refinement_accepted_substeps: int
    max_factor_norm: float
    sector_ranks: tuple[int, ...]
    accepted: bool
    full_local_dimension: int | None = None
    hamiltonian_applications: int = 0
    energy_refinement_start: str | None = None
    split_refinement_energy: float | None = None
    incumbent_refinement_energy: float | None = None
    energy_refinement_factor_norm_limit: float | None = None
    split_refinement_within_norm_limit: bool | None = None
    coupled_refinement_iterations: int = 0
    coupled_refinement_accepted_steps: int = 0
    compression_diagnostics: dict | None = None


@dataclass(frozen=True)
class LETTATwoSiteSweep:
    sweep: int
    direction: str
    energy: float
    energy_change: float
    energy_density_change: float
    bond_dimension: int
    updates: tuple[LETTAPairUpdate, ...]
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class LETTATwoSiteResult:
    state: LatticeLETTA
    energy: float
    converged: bool
    sweeps: int
    history: tuple[LETTATwoSiteSweep, ...]
    message: str
    two_site_energy: float
    polish_sweeps: int = 0


class _BlockMetricWhitening:
    """Implicit blockwise map from Euclidean to metric-normalized vectors."""

    def __init__(self, metric, tolerance):
        grouped = {}
        offset = 0
        for block, indices in zip(metric.blocks, metric.indices):
            if isinstance(metric, DiagonalMetric) and metric.support is not None:
                keep = metric.support[indices]
                block, indices = block[np.ix_(keep, keep)], indices[keep]
            basis, coordinates = _equilibrated_metric_factors(block, tolerance)
            width = basis.shape[1]
            if not width:
                continue
            grouped.setdefault((basis.shape, basis.dtype), []).append((
                np.asarray(indices), np.arange(offset, offset + width),
                basis, coordinates,
            ))
            offset += width
        if offset == 0:
            raise ValueError("the two-site LETTA overlap metric has zero rank.")
        self.size = metric.size
        self.rank = offset
        self.dtype = metric.dtype
        groups = []
        for pieces in grouped.values():
            indices, sources, bases, coordinates = map(np.stack, zip(*pieces))
            adjoints = np.ascontiguousarray(bases.conj().swapaxes(1, 2))
            groups.append((indices, sources, bases, adjoints, coordinates))
        self.groups = tuple(groups)

    def to_full(self, reduced):
        reduced = np.asarray(reduced)
        result = np.zeros(self.size, dtype=np.result_type(self.dtype, reduced))
        for indices, source, bases, _, _ in self.groups:
            result[indices] = (bases @ reduced[source][..., None])[..., 0]
        return result

    def adjoint(self, full):
        full = np.asarray(full)
        result = np.empty(self.rank, dtype=np.result_type(self.dtype, full))
        for indices, target, _, adjoints, _ in self.groups:
            result[target] = (adjoints @ full[indices][..., None])[..., 0]
        return result

    def coordinates(self, full):
        full = np.asarray(full)
        result = np.empty(self.rank, dtype=np.result_type(self.dtype, full))
        for indices, target, _, _, coordinates in self.groups:
            result[target] = (coordinates @ full[indices][..., None])[..., 0]
        return result


def _rayleigh(vector, action, metric):
    applied = action(vector)
    denominator = np.vdot(vector, metric @ vector)
    if np.real(denominator) <= np.finfo(float).tiny:
        raise ValueError("a proposed two-site LETTA tensor has zero norm.")
    return float(np.real(np.vdot(vector, applied) / denominator))


def _embed_pair_vector(vector, indices, full_dimension):
    vector = np.asarray(vector)
    if indices is None:
        return vector
    result = np.zeros(
        (full_dimension,) + vector.shape[1:], dtype=vector.dtype
    )
    result[indices] = vector
    return result


def _lowest_pair_vector(action, metric, old_vector, options):
    whitening = _BlockMetricWhitening(metric, options.metric_tolerance)
    old_applied = action(old_vector)

    def reduced_action(vector):
        return whitening.adjoint(action(whitening.to_full(vector)))

    initial = whitening.coordinates(old_vector)
    initial_norm = np.linalg.norm(initial)
    if initial_norm <= np.finfo(float).tiny:
        initial = np.ones(whitening.rank, dtype=whitening.dtype)
        initial_norm = np.linalg.norm(initial)
    initial = initial / initial_norm

    if whitening.rank == 1:
        reduced_vector = np.ones(1, dtype=whitening.dtype)
    elif whitening.rank <= options.dense_solver_threshold:
        identity = np.eye(whitening.rank, dtype=whitening.dtype)
        reduced_hamiltonian = np.column_stack(
            [reduced_action(identity[:, column]) for column in range(whitening.rank)]
        )
        reduced_hamiltonian = 0.5 * (
            reduced_hamiltonian + reduced_hamiltonian.conj().T
        )
        _values, vectors = linalg.eigh(
            reduced_hamiltonian,
            subset_by_index=[0, 0],
            check_finite=False,
        )
        reduced_vector = vectors[:, 0]
    else:
        operator = LinearOperator(
            (whitening.rank, whitening.rank),
            matvec=reduced_action,
            dtype=np.result_type(old_vector, metric.dtype, old_applied),
        )
        try:
            _values, vectors = eigsh(
                operator,
                k=1,
                which="SA",
                v0=initial,
                tol=options.eigensolver_tolerance,
                maxiter=options.eigensolver_max_iterations,
            )
        except ArpackNoConvergence as error:
            if error.eigenvectors is None or error.eigenvectors.shape[1] == 0:
                raise
            vectors = error.eigenvectors
        reduced_vector = vectors[:, 0]

    candidate = whitening.to_full(reduced_vector)
    candidate /= np.sqrt(np.vdot(candidate, metric @ candidate))
    trial_basis = np.column_stack([old_vector, candidate])
    applied_basis = np.column_stack(
        [old_applied, action(candidate)]
    )
    trial_hamiltonian = trial_basis.conj().T @ applied_basis
    trial_metric = trial_basis.conj().T @ (metric @ trial_basis)
    energy, coefficients, _rank, _residual = _lowest_generalized_eigenpair(
        trial_hamiltonian,
        trial_metric,
        options.metric_tolerance,
    )
    vector = trial_basis @ coefficients
    vector /= np.sqrt(np.vdot(vector, metric @ vector))
    applied = action(vector)
    residual = np.linalg.norm(applied - energy * (metric @ vector))
    return energy, vector, whitening.rank, float(residual)


def _optimize_pair(
    state,
    layout,
    hamiltonian_cache,
    metric_cache,
    hamiltonian_left,
    hamiltonian_right,
    metric_left,
    metric_right,
    bond_dim,
    direction,
    options,
):
    old_merged_full = layout.merge(
        state.tensors[layout.left_site],
        state.tensors[layout.left_site + 1],
    ).reshape(-1)
    full_metric = metric_cache.effective_pair_metric(
        metric_left, metric_right, layout
    )
    if options.matrix_free:
        full_action = hamiltonian_cache.prepare_pair_action(
            hamiltonian_left, hamiltonian_right, layout)
    else:
        hamiltonian = hamiltonian_cache.effective_pair_matrix(
            hamiltonian_left, hamiltonian_right, layout
        )
        full_action = lambda vector: hamiltonian @ vector

    hamiltonian_applications = 0
    original_action = full_action

    def full_action(vector):
        nonlocal hamiltonian_applications
        hamiltonian_applications += 1
        return original_action(vector)

    indices = layout.symmetry_indices() if layout.symmetry is not None else None
    if indices is None:
        metric = full_metric
        action = full_action
        old_merged = old_merged_full
    else:
        metric = full_metric.restrict(indices)

        def action(vector):
            full = _embed_pair_vector(vector, indices, old_merged_full.size)
            return full_action(full)[indices]

        old_merged = old_merged_full[indices]

    old_energy = _rayleigh(old_merged, action, metric)
    local_energy, optimized, metric_rank, residual = _lowest_pair_vector(
        action, metric, old_merged, options
    )
    optimized_full = _embed_pair_vector(
        optimized, indices, old_merged_full.size
    )
    split = conditional_svd_split(
        optimized_full.reshape(layout.merged_shape),
        layout,
        max_bond_dim=bond_dim,
        direction=direction,
        cutoff=options.conditional_svd_cutoff,
    )
    truncation_iterations = 0
    compression_diagnostics = None
    energy_refinement_initial_energy = None
    energy_refinement_energy = None
    energy_refinement_iterations = 0
    energy_refinement_accepted_substeps = 0
    energy_refinement_start = None
    split_refinement_energy = None
    incumbent_refinement_energy = None
    energy_refinement_factor_norm_limit = None
    split_refinement_within_norm_limit = None
    coupled_refinement_iterations = 0
    coupled_refinement_accepted_steps = 0
    left_indices = (
        np.flatnonzero(layout.factor_mask("left").reshape(-1))
        if layout.symmetry is not None
        else None
    )
    right_indices = (
        np.flatnonzero(layout.factor_mask("right").reshape(-1))
        if layout.symmetry is not None
        else None
    )
    left_tensor = split.left_tensor.copy()
    right_tensor = split.right_tensor.copy()
    if options.split_method in {"metric", "metric-energy", "metric-als", "metric-als-energy"}:
        refinement = metric_refine(
            optimized_full.reshape(layout.merged_shape),
            layout,
            split,
            full_metric,
            compression=options.compression,
            tolerance=options.truncation_tolerance,
            max_iterations=options.truncation_max_iterations,
            metric_tolerance=options.metric_tolerance,
            left_indices=left_indices,
            right_indices=right_indices,
        )
        left_tensor = refinement.left_tensor.copy()
        right_tensor = refinement.right_tensor.copy()
        truncation_iterations = refinement.iterations
        compression_diagnostics = refinement.diagnostics
    if options.split_method in {"energy-refined", "metric-energy", "metric-als-energy"}:
        # The split can enter a worse fixed-rank basin even when its initial
        # energy is lower. Compare both starts *after* energy relaxation.
        initial = replace(split, left_tensor=left_tensor, right_tensor=right_tensor)
        refinement_options = dict(
            tolerance=options.energy_refinement_tolerance,
            max_iterations=options.energy_refinement_max_iterations,
            metric_tolerance=options.metric_tolerance,
            energy_increase_tolerance=options.energy_increase_tolerance,
            max_factor_norm_growth=(
                options.energy_refinement_max_factor_norm_growth
            ),
            left_indices=left_indices,
            right_indices=right_indices,
        )
        split_refinement = energy_refine_split(
            layout, initial, full_action, full_metric,
            coupled_max_iterations=options.coupled_refinement_max_iterations,
            coupled_metric_tolerance=options.coupled_refinement_metric_tolerance,
            coupled_activation_ratio=options.coupled_refinement_activation_ratio,
            **refinement_options,
        )
        incumbent = replace(
            split,
            left_tensor=state.tensors[layout.left_site],
            right_tensor=state.tensors[layout.left_site + 1],
        )
        incumbent_refinement = energy_refine_split(
            layout, incumbent, full_action, full_metric,
            coupled_max_iterations=options.coupled_refinement_max_iterations,
            coupled_metric_tolerance=options.coupled_refinement_metric_tolerance,
            coupled_activation_ratio=options.coupled_refinement_activation_ratio,
            **refinement_options,
        )
        split_refinement_energy = split_refinement.energy
        incumbent_refinement_energy = incumbent_refinement.energy
        # ALS can already have amplified null-space components before the
        # refinement's per-start growth guard runs. Both candidates need the
        # same budget, measured in the normalized, balanced incumbent gauge.
        incumbent_norm = float(np.real(np.vdot(
            old_merged_full, full_metric @ old_merged_full
        )))
        incumbent_factor_norm = np.sqrt(
            np.linalg.norm(state.tensors[layout.left_site])
            * np.linalg.norm(state.tensors[layout.left_site + 1])
            / np.sqrt(incumbent_norm)
        )
        energy_refinement_factor_norm_limit = float(
            options.energy_refinement_max_factor_norm_growth * incumbent_factor_norm
        )
        split_refinement_within_norm_limit = bool(
            np.isfinite(split_refinement.max_factor_norm)
            and split_refinement.max_factor_norm <= energy_refinement_factor_norm_limit
        )
        if (not split_refinement_within_norm_limit
                or incumbent_refinement.energy < split_refinement.energy):
            refinement = incumbent_refinement
            energy_refinement_start = "incumbent"
        else:
            refinement = split_refinement
            energy_refinement_start = "split"
        left_tensor = refinement.left_tensor.copy()
        right_tensor = refinement.right_tensor.copy()
        energy_refinement_initial_energy = refinement.initial_energy
        energy_refinement_energy = refinement.energy
        energy_refinement_iterations = (
            split_refinement.iterations + incumbent_refinement.iterations
        )
        energy_refinement_accepted_substeps = (
            split_refinement.accepted_substeps + incumbent_refinement.accepted_substeps
        )
        coupled_refinement_iterations = (
            split_refinement.coupled_iterations + incumbent_refinement.coupled_iterations
        )
        coupled_refinement_accepted_steps = (
            split_refinement.coupled_accepted_steps + incumbent_refinement.coupled_accepted_steps
        )
    reconstructed = layout.merge(left_tensor, right_tensor).reshape(-1)
    difference = optimized_full - reconstructed
    truncation_loss = float(
        max(0.0, np.real(np.vdot(difference, full_metric @ difference)))
    )
    norm = np.real(np.vdot(reconstructed, full_metric @ reconstructed))
    accepted = norm > np.finfo(float).tiny
    if accepted:
        right_tensor /= np.sqrt(norm)
        reconstructed = layout.merge(left_tensor, right_tensor).reshape(-1)
        energy = _rayleigh(reconstructed, full_action, full_metric)
        accepted = energy <= old_energy + options.energy_increase_tolerance
    else:
        energy = old_energy

    if accepted:
        state.tensors[layout.left_site] = left_tensor
        state.tensors[layout.left_site + 1] = right_tensor
        state.enforce_symmetry()
    else:
        energy = old_energy
    return LETTAPairUpdate(
        left_site=layout.left_site,
        right_site=layout.left_site + 1,
        shared_physical_sites=layout.shared,
        old_energy=old_energy,
        local_energy=local_energy,
        energy=energy,
        metric_rank=metric_rank,
        local_dimension=old_merged.size,
        residual_norm=residual,
        conditional_discarded_weight=split.discarded_weight,
        metric_truncation_loss=truncation_loss,
        truncation_iterations=truncation_iterations,
        compression_diagnostics=compression_diagnostics,
        energy_refinement_initial_energy=energy_refinement_initial_energy,
        energy_refinement_energy=energy_refinement_energy,
        energy_refinement_iterations=energy_refinement_iterations,
        energy_refinement_accepted_substeps=(
            energy_refinement_accepted_substeps
        ),
        max_factor_norm=max(
            float(np.linalg.norm(left_tensor)),
            float(np.linalg.norm(right_tensor)),
        ),
        sector_ranks=split.sector_ranks,
        accepted=accepted,
        full_local_dimension=old_merged_full.size,
        hamiltonian_applications=hamiltonian_applications,
        energy_refinement_start=energy_refinement_start,
        split_refinement_energy=split_refinement_energy,
        incumbent_refinement_energy=incumbent_refinement_energy,
        energy_refinement_factor_norm_limit=energy_refinement_factor_norm_limit,
        split_refinement_within_norm_limit=split_refinement_within_norm_limit,
        coupled_refinement_iterations=coupled_refinement_iterations,
        coupled_refinement_accepted_steps=coupled_refinement_accepted_steps,
    )


def _shift_fixed_virtual_gauge(state, site, direction, mode):
    """Shift a QR gauge without shrinking the allocated virtual bond."""

    if state.symmetry is not None:
        _shift_virtual_gauge(state, site, direction, mode)
        return
    if mode != "qr":
        _shift_virtual_gauge(state, site, direction, mode)
        return
    if direction == "lr":
        neighbor = site + 1
        if neighbor >= state.nsites:
            return
        tensor = state.tensors[site]
        bond_dim = tensor.shape[-1]
        matrix = tensor.reshape(-1, bond_dim)
        q, r = np.linalg.qr(matrix, mode="reduced")
        if q.shape[1] < bond_dim:
            rank = q.shape[1]
            padded_q = np.zeros(
                (matrix.shape[0], bond_dim), dtype=q.dtype
            )
            padded_r = np.zeros((bond_dim, bond_dim), dtype=r.dtype)
            padded_q[:, :rank] = q
            padded_r[:rank] = r
            q, r = padded_q, padded_r
        state.tensors[site] = q.reshape(tensor.shape)
        state.tensors[neighbor] = np.tensordot(
            r, state.tensors[neighbor], axes=([1], [0])
        )
        return
    if direction != "rl":
        raise ValueError("direction must be 'lr' or 'rl'.")
    neighbor = site - 1
    if neighbor < 0:
        return
    tensor = state.tensors[site]
    bond_dim = tensor.shape[0]
    matrix = tensor.reshape(bond_dim, -1)
    q, r = np.linalg.qr(matrix.T, mode="reduced")
    if q.shape[1] < bond_dim:
        rank = q.shape[1]
        padded_q = np.zeros(
            (matrix.shape[1], bond_dim), dtype=q.dtype
        )
        padded_r = np.zeros((bond_dim, bond_dim), dtype=r.dtype)
        padded_q[:, :rank] = q
        padded_r[:rank] = r
        q, r = padded_q, padded_r
    state.tensors[site] = q.T.reshape(tensor.shape)
    state.tensors[neighbor] = np.tensordot(
        state.tensors[neighbor], r.T, axes=([-1], [0])
    )


def _pair_sweep(state, hamiltonian, bond_dim, direction, options, *, workspace=None):
    if workspace is None and options.gauge_mode == "frontier":
        from .._letta_one_site_opt.gauge import canonicalize_frontier

        # Also prepares the untouched side for non-alternating sweeps. Valid
        # complete-side certificates are reused on alternating sweeps.
        canonicalize_frontier(
            state, 0 if direction == "lr" else state.nsites - 1,
            tolerance=options.metric_tolerance,
        )
    if workspace is None:
        hamiltonian_cache = LETTAPairEnvironmentCache(
            state, hamiltonian, use_sparse_mpo=options.use_sparse_mpo)
        metric_cache = IdentityPairEnvironmentCache(state)
        build = "build_right_environments" if direction == "lr" else "build_left_environments"
        hamiltonian_environments = getattr(hamiltonian_cache, build)()
        metric_environments = getattr(metric_cache, build)()
    else:
        hamiltonian_cache, metric_cache, hamiltonian_environments, metric_environments = workspace
    updates = []
    if direction == "lr":
        hamiltonian_right = hamiltonian_environments
        metric_right = metric_environments
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        for site in range(state.nsites - 1):
            layout = LETTAPairLayout.from_state(state, site)
            updates.append(
                _optimize_pair(
                    state,
                    layout,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_boundary,
                    hamiltonian_right[site + 2],
                    metric_boundary,
                    metric_right[site + 2],
                    bond_dim,
                    direction,
                    options,
                )
            )
            if options.gauge_mode == "frontier":
                metric_boundary = _shift_gauge_and_extend_metric(
                    state, site, direction, options, metric_cache, metric_boundary
                )
            else:
                _shift_fixed_virtual_gauge(state, site, direction, options.gauge_mode)
                metric_boundary = metric_cache.extend_left(metric_boundary, site)
            hamiltonian_boundary = hamiltonian_cache.extend_left(
                hamiltonian_boundary, site
            )
            hamiltonian_environments[site + 1] = hamiltonian_boundary
            metric_environments[site + 1] = metric_boundary
        hamiltonian_boundary = hamiltonian_cache.extend_left(
            hamiltonian_boundary, state.nsites - 1
        )
        metric_boundary = metric_cache.extend_left(
            metric_boundary, state.nsites - 1
        )
    else:
        hamiltonian_left = hamiltonian_environments
        metric_left = metric_environments
        hamiltonian_boundary = hamiltonian_cache.scalar_boundary()
        metric_boundary = metric_cache.scalar_boundary()
        for site in range(state.nsites - 2, -1, -1):
            layout = LETTAPairLayout.from_state(state, site)
            updates.append(
                _optimize_pair(
                    state,
                    layout,
                    hamiltonian_cache,
                    metric_cache,
                    hamiltonian_left[site],
                    hamiltonian_boundary,
                    metric_left[site],
                    metric_boundary,
                    bond_dim,
                    direction,
                    options,
                )
            )
            if options.gauge_mode == "frontier":
                metric_boundary = _shift_gauge_and_extend_metric(
                    state, site + 1, direction, options, metric_cache, metric_boundary
                )
            else:
                _shift_fixed_virtual_gauge(state, site + 1, direction, options.gauge_mode)
                metric_boundary = metric_cache.extend_right(metric_boundary, site + 1)
            hamiltonian_boundary = hamiltonian_cache.extend_right(
                hamiltonian_boundary, site + 1
            )
            hamiltonian_environments[site + 1] = hamiltonian_boundary
            metric_environments[site + 1] = metric_boundary
        hamiltonian_boundary = hamiltonian_cache.extend_right(
            hamiltonian_boundary, 0
        )
        metric_boundary = metric_cache.extend_right(metric_boundary, 0)
    full_cut = state.nsites if direction == "lr" else 0
    hamiltonian_environments[full_cut] = hamiltonian_boundary
    metric_environments[full_cut] = metric_boundary
    # The opposite endpoint remains a scalar, not the old completed sweep.
    empty_cut = 0 if direction == "lr" else state.nsites
    hamiltonian_environments[empty_cut] = hamiltonian_cache.scalar_boundary()
    metric_environments[empty_cut] = metric_cache.scalar_boundary()
    energy = float(np.real(hamiltonian_boundary / metric_boundary))
    return tuple(updates), energy


def _validate_options(options):
    if not isinstance(options, LETTATwoSiteOptions):
        raise TypeError("options must be a LETTATwoSiteOptions instance.")
    if not isinstance(options.compression, MetricCompressionOptions):
        raise TypeError("compression must be MetricCompressionOptions.")
    if options.max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive.")
    if (
        options.tolerance <= 0.0
        or options.metric_tolerance <= 0.0
        or options.energy_increase_tolerance < 0.0
        or options.eigensolver_tolerance <= 0.0
    ):
        raise ValueError("solver tolerances must be positive.")
    if options.eigensolver_max_iterations <= 0:
        raise ValueError("eigensolver_max_iterations must be positive.")
    if options.dense_solver_threshold <= 0:
        raise ValueError("dense_solver_threshold must be positive.")
    if options.split_method not in {
        "metric",
        "metric-energy",
        "conditional-svd",
        "metric-als",
        "metric-als-energy",
        "energy-refined",
    }:
        raise ValueError(
            "split_method must be 'conditional-svd', 'metric', 'metric-energy', "
            "'metric-als', 'metric-als-energy', or 'energy-refined'."
        )
    if options.conditional_svd_cutoff < 0.0:
        raise ValueError("conditional_svd_cutoff must be nonnegative.")
    if options.truncation_tolerance <= 0.0:
        raise ValueError("truncation_tolerance must be positive.")
    if options.truncation_max_iterations <= 0:
        raise ValueError("truncation_max_iterations must be positive.")
    if options.energy_refinement_tolerance <= 0.0:
        raise ValueError("energy_refinement_tolerance must be positive.")
    if options.energy_refinement_max_iterations <= 0:
        raise ValueError("energy_refinement_max_iterations must be positive.")
    if (not isinstance(options.coupled_refinement_max_iterations, (int, np.integer))
            or options.coupled_refinement_max_iterations < 0):
        raise ValueError("coupled_refinement_max_iterations must be a nonnegative integer.")
    if (not np.isfinite(options.coupled_refinement_metric_tolerance)
            or not 0 < options.coupled_refinement_metric_tolerance < 1):
        raise ValueError("coupled_refinement_metric_tolerance must be finite and between zero and one.")
    if (not np.isfinite(options.coupled_refinement_activation_ratio)
            or options.coupled_refinement_activation_ratio < 0):
        raise ValueError("coupled_refinement_activation_ratio must be finite and nonnegative.")
    if options.energy_refinement_max_factor_norm_growth < 1.0:
        raise ValueError(
            "energy_refinement_max_factor_norm_growth must be at least one."
        )
    if options.gauge_mode not in {"qr", "frontier", "scalar", "none"}:
        raise ValueError("gauge_mode must be 'qr', 'frontier', 'scalar', or 'none'.")
    if options.one_site_polish_sweeps < 0:
        raise ValueError("one_site_polish_sweeps must be nonnegative.")


def _refresh_pair_gauge(state, direction, options, workspace):
    """Keep unchanged side environments when preparing the next frontier gauge.

    Non-admissible cuts still need the original QR preparation. Its changes
    invalidate only the prefixes/suffixes containing those changed tensors.
    """
    from .._letta_one_site_opt.canonical import tensor_signature
    from .._letta_one_site_opt.gauge import canonicalize_frontier

    before = [tensor_signature(a) for a in state.tensors]
    canonicalize_frontier(state, 0 if direction == "lr" else state.nsites - 1,
                          tolerance=options.metric_tolerance)
    changed = [i for i, a in enumerate(state.tensors) if tensor_signature(a) != before[i]]
    if not changed:
        return
    hcache, ncache, henv, nenv = workspace
    if direction == "rl":
        for site in range(min(changed), state.nsites):
            henv[site + 1] = hcache.extend_left(henv[site], site)
            saved = ncache.saved_canonical(site + 1, "lr")
            nenv[site + 1] = (saved.environment if saved is not None else
                             ncache.extend_left(nenv[site], site))
    else:
        for site in range(max(changed), -1, -1):
            henv[site] = hcache.extend_right(henv[site + 1], site)
            saved = ncache.saved_canonical(site, "rl")
            nenv[site] = (saved.environment if saved is not None else
                         ncache.extend_right(nenv[site + 1], site))


def letta_two_site_dmrg(
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
    """Optimize adjacent LETTA tensors with shared-index-aware splitting."""

    solve_started = perf_counter()
    options = LETTATwoSiteOptions() if options is None else options
    _validate_options(options)
    from .._letta_one_site_opt.reduced_operators import ReducedMPOHamiltonian
    from .._letta_one_site_opt.reduced_state import ReducedLatticeLETTA

    if isinstance(hamiltonian, ReducedMPOHamiltonian) or isinstance(
        state, ReducedLatticeLETTA
    ):
        if state is None:
            raise ValueError(
                "state is required for reduced two-site optimization so the "
                "user-defined SU(2) basis and multiplet allocation are explicit"
            )
        from .reduced_solver import reduced_two_site_dmrg

        return reduced_two_site_dmrg(
            hamiltonian,
            state=state,
            bond_dim=bond_dim,
            options=options,
        )
    if not isinstance(hamiltonian, LatticeMPO):
        raise TypeError("hamiltonian must be a LatticeMPO.")
    bond_dim = int(bond_dim)
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive.")
    if state is None:
        if lattice_shape is None:
            if hamiltonian.lattice_shape is None:
                raise ValueError("lattice_shape is required when state is omitted.")
            lattice_shape = hamiltonian.lattice_shape
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
        if options.gauge_mode == "frontier":
            from .._letta_one_site_opt.canonical import copy_canonical_state

            state = copy_canonical_state(state)
        else:
            state = state.copy()
    if state.nsites < 2:
        raise ValueError("two-site optimization requires at least two sites.")
    if state.nsites != hamiltonian.nsites:
        raise ValueError("Hamiltonian MPO length does not match the LETTA state.")
    if state.physical_dim != hamiltonian.physical_dim:
        raise ValueError("Hamiltonian physical dimension does not match the state.")
    if any(dimension > bond_dim for dimension in state.bond_dimensions):
        raise ValueError("bond_dim cannot shrink the supplied LETTA state.")
    if any(dimension != bond_dim for dimension in state.bond_dimensions):
        state = state.expand_bond_dimension(bond_dim)

    direction = options.start_direction.lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("start_direction must be 'lr' or 'rl'.")
    if options.gauge_mode == "frontier":
        from .._letta_one_site_opt.gauge import canonicalize_frontier
        canonicalize_frontier(state, 0 if direction == "lr" else state.nsites - 1,
                              tolerance=options.metric_tolerance)
    hamiltonian_cache = LETTAPairEnvironmentCache(
        state, hamiltonian, use_sparse_mpo=options.use_sparse_mpo)
    metric_cache = IdentityPairEnvironmentCache(state)
    build = "build_right_environments" if direction == "lr" else "build_left_environments"
    hamiltonian_environments = getattr(hamiltonian_cache, build)()
    metric_environments = getattr(metric_cache, build)()
    workspace = (hamiltonian_cache, metric_cache, hamiltonian_environments, metric_environments)
    full_cut = 0 if direction == "lr" else state.nsites
    previous_energy = float(np.real(hamiltonian_environments[full_cut] / metric_environments[full_cut]))
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    for sweep in range(1, options.max_sweeps + 1):
        sweep_start = [tensor.copy() for tensor in state.tensors]
        if sweep > 1 and options.alternate and options.gauge_mode == "frontier":
            _refresh_pair_gauge(state, direction, options, workspace)
        updates, energy = _pair_sweep(
            state, hamiltonian, bond_dim, direction, options, workspace=workspace
        )
        # The final local quadratic form is not an independent energy check:
        # small environment errors can be amplified by large null components.
        energy = state.expectation(hamiltonian)
        rejected = (not np.isfinite(energy)
                    or energy > previous_energy + options.energy_increase_tolerance)
        if rejected:
            state.tensors = sweep_start
            energy = previous_energy
        energy_change = abs(energy - previous_energy)
        density_change = energy_change / state.nsites
        history.append(
            LETTATwoSiteSweep(
                sweep=sweep,
                direction=direction,
                energy=energy,
                energy_change=energy_change,
                energy_density_change=density_change,
                bond_dimension=bond_dim,
                updates=updates,
                elapsed_seconds=perf_counter() - solve_started,
            )
        )
        if options.verbosity:
            print(
                f"two-site sweep {sweep} ({direction}): "
                f"energy={energy:.12f}, density change={density_change:.3e}"
            )
        previous_energy = energy
        if rejected:
            message = "STOP: SWEEP FAILED FRESH ENERGY CHECK"
            break
        if density_change < options.tolerance:
            converged = True
            message = "CONVERGED: ENERGY DENSITY CHANGE BELOW TOLERANCE"
            break
        if options.alternate:
            direction = "rl" if direction == "lr" else "lr"
        elif sweep < options.max_sweeps:
            # A repeated direction needs the opposite side rebuilt after edits.
            if options.gauge_mode == "frontier":
                canonicalize_frontier(state, 0 if direction == "lr" else state.nsites - 1,
                                      tolerance=options.metric_tolerance)
            hamiltonian_environments[:] = getattr(hamiltonian_cache, build)()
            metric_environments[:] = getattr(metric_cache, build)()
    two_site_energy = previous_energy
    polish_sweeps = 0
    if options.one_site_polish_sweeps:
        polish_direction = (
            "rl" if history[-1].direction == "lr" else "lr"
        )
        polished = letta_dmrg(
            hamiltonian,
            state=state,
            options=LETTADMROptions(
                max_sweeps=options.one_site_polish_sweeps,
                tolerance=options.tolerance,
                metric_tolerance=options.metric_tolerance,
                energy_increase_tolerance=options.energy_increase_tolerance,
                start_direction=polish_direction,
                alternate=options.alternate,
                gauge_mode=options.gauge_mode,
                use_sparse_mpo=options.use_sparse_mpo,
                matrix_free=options.matrix_free,
                verbosity=options.verbosity,
            ),
        )
        polish_sweeps = polished.sweeps
        if polished.energy <= previous_energy + options.energy_increase_tolerance:
            state = polished.state
            previous_energy = polished.energy
        message = f"{message}; ONE-SITE POLISH: {polished.message}"
    return LETTATwoSiteResult(
        state=state,
        energy=previous_energy,
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
        two_site_energy=two_site_energy,
        polish_sweeps=polish_sweeps,
    )

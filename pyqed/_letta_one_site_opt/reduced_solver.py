"""Exact reduced-SU(2) one-site LETTA local problems and sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.nonabelian.builder import identity_operator
from pyqed.mps.nonabelian.environment import (
    BlockSparseEnvironmentChain,
    _left_reduced_rank_coupled_block,
)
from pyqed.mps.nonabelian.mpo import (
    MPO,
    IrreducibleMPO,
    RankCoupledMPO,
    as_rank_coupled_mpo,
)
from pyqed.mps.nonabelian.tensor import NonabelianTensor

from .reduced_contraction import (
    CanonicalEnvironmentChain,
    expand_reduced_mps_site,
    identity_canonical_factors,
    reduce_expanded_mps_site,
)
from .reduced_frontier import FrontierSiteEmbedding, ReducedFrontier
from .reduced_operators import (
    ReducedMPOHamiltonian,
    physical_leg_from_reduced_basis,
)
from .reduced_state import ReducedLatticeLETTA


@dataclass(frozen=True)
class ReducedLocalProblem:
    site: int
    frame: np.ndarray | None
    embedding: FrontierSiteEmbedding
    hamiltonian: np.ndarray | None = None
    metric: np.ndarray | None = None
    hamiltonian_action: object | None = None
    metric_action: object | None = None

    @property
    def local_dimension(self):
        return self.embedding.source_size

    @property
    def full_local_dimension(self):
        return self.embedding.target_size

    def apply_hamiltonian(self, vector):
        if self.hamiltonian is not None:
            return self.hamiltonian @ np.asarray(vector)
        if self.hamiltonian_action is None:
            raise RuntimeError("local Hamiltonian action is unavailable")
        return np.asarray(self.hamiltonian_action(vector))

    def apply_metric(self, vector):
        if self.metric is not None:
            return self.metric @ np.asarray(vector)
        if self.metric_action is None:
            raise RuntimeError("local metric action is unavailable")
        return np.asarray(self.metric_action(vector))


def _validate_dense_hamiltonian(hamiltonian, state):
    matrix = np.asarray(hamiltonian)
    dimension = state.dense_physical_dim ** state.nsites
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"dense Hamiltonian has shape {matrix.shape}, expected {(dimension, dimension)}"
        )
    if not np.allclose(matrix, matrix.conj().T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("reduced LETTA dense Hamiltonian must be Hermitian")
    return matrix


_REDUCED_MPO_TYPES = (MPO, IrreducibleMPO, RankCoupledMPO)


def _is_reduced_mpo(hamiltonian):
    if isinstance(hamiltonian, ReducedMPOHamiltonian):
        return True
    return (
        isinstance(hamiltonian, (tuple, list))
        and bool(hamiltonian)
        and all(isinstance(core, _REDUCED_MPO_TYPES) for core in hamiltonian)
    )


def _validate_reduced_mpo(hamiltonian, state):
    if not _is_reduced_mpo(hamiltonian):
        raise TypeError(
            "reduced Hamiltonian must be a dense matrix or a sequence of "
            "MPO/IrreducibleMPO/RankCoupledMPO factors"
        )
    wrapped = isinstance(hamiltonian, ReducedMPOHamiltonian)
    factors = (
        hamiltonian.factors
        if wrapped
        else tuple(hamiltonian)
    )
    if len(factors) != state.nsites:
        raise ValueError(
            f"reduced MPO has {len(factors)} factors, expected {state.nsites}"
        )
    validation_key = (state.physical_basis, state.nsites)
    if wrapped and validation_key in getattr(
        hamiltonian, "_letta_validation_keys", ()
    ):
        return factors
    expected = physical_leg_from_reduced_basis(state.physical_basis)
    for site, core in enumerate(factors):
        if core.phys_out_leg != expected or core.phys_in_leg != expected:
            raise ValueError(
                f"reduced MPO physical leg at site {site} does not match the "
                "state's fully reduced physical basis"
            )
    factor_shapes = tuple(np.asarray(core.as_dense()).shape for core in factors)
    for site, shape in enumerate(factor_shapes):
        if len(shape) != 4:
            raise ValueError(f"reduced MPO factor {site} is not rank four")
        if site == 0 and shape[0] != 1:
            raise ValueError("reduced MPO left boundary dimension must be one")
        if site == len(factor_shapes) - 1 and shape[1] != 1:
            raise ValueError("reduced MPO right boundary dimension must be one")
        if site and factor_shapes[site - 1][1] != shape[0]:
            raise ValueError(
                f"reduced MPO virtual dimensions do not match between sites "
                f"{site - 1} and {site}"
            )
    rank_coupled = tuple(isinstance(core, RankCoupledMPO) for core in factors)
    if any(rank_coupled) and not all(rank_coupled):
        raise ValueError(
            "a reduced MPO chain must use either rank-coupled factors at every "
            "site or scalar factors at every site"
        )
    if all(rank_coupled) and not wrapped:
        raise ValueError(
            "rank-coupled factors require ReducedMPOHamiltonian with an exact "
            "canonical_factors view; raw rank-coupled chains are ambiguous "
            "beyond two sites"
        )
    if wrapped:
        canonical_expected = physical_leg_from_reduced_basis(
            state.physical_basis, fully_reduced=False
        )
        canonical_shapes = []
        for site, core in enumerate(hamiltonian.canonical_factors):
            phys_out = getattr(core, "phys_out_leg", None)
            phys_in = getattr(core, "phys_in_leg", None)
            if phys_out is not None or phys_in is not None:
                if phys_out != canonical_expected or phys_in != canonical_expected:
                    raise ValueError(
                        f"canonical MPO physical leg at site {site} does not "
                        "match the state's ordered magnetic-component basis"
                    )
            dense = np.asarray(
                core.as_dense() if hasattr(core, "as_dense") else core
            )
            if dense.ndim != 4:
                raise ValueError(f"canonical MPO factor {site} is not rank four")
            if dense.shape[2:] != (
                state.dense_physical_dim,
                state.dense_physical_dim,
            ):
                raise ValueError(
                    f"canonical MPO physical dimensions at site {site} do not "
                    "match the state's magnetic-component basis"
                )
            canonical_shapes.append(dense.shape)
        for site, shape in enumerate(canonical_shapes):
            if site == 0 and shape[0] != 1:
                raise ValueError("canonical MPO left boundary dimension must be one")
            if site == len(canonical_shapes) - 1 and shape[1] != 1:
                raise ValueError("canonical MPO right boundary dimension must be one")
            if site and canonical_shapes[site - 1][1] != shape[0]:
                raise ValueError(
                    f"canonical MPO virtual dimensions do not match between "
                    f"sites {site - 1} and {site}"
                )
        cached = set(getattr(hamiltonian, "_letta_validation_keys", ()))
        cached.add(validation_key)
        object.__setattr__(hamiltonian, "_letta_validation_keys", frozenset(cached))
    return factors


def _identity_mpo_factors(factors):
    identity_factors = tuple(
        MPO.from_site_operator(identity_operator(core.phys_in_leg, dtype=complex))
        for core in factors
    )
    if all(isinstance(core, RankCoupledMPO) for core in factors):
        return tuple(as_rank_coupled_mpo(core) for core in identity_factors)
    return identity_factors


def _frontier_environment_chains(state, factors):
    frontier = ReducedFrontier.from_state(state)
    sites = frontier.to_mps(state)
    hamiltonian_chain = BlockSparseEnvironmentChain.build(sites, factors)
    metric_chain = BlockSparseEnvironmentChain.build(
        sites, _identity_mpo_factors(factors)
    )
    return frontier, sites, hamiltonian_chain, metric_chain


def _target_local_matrix(chain, embedding, site):
    """Materialize one polynomial-size frontier-MPS local operator."""

    site = int(site)
    left = chain.left_envs[site]
    right = chain.right_envs[site]
    core = chain.mpo_factors[site]
    layout = embedding.target_layout
    dtype = np.result_type(core.dtype, complex)
    matrix = np.zeros((layout.size, layout.size), dtype=dtype)

    for in_key in layout.keys:
        q_lk, q_pk, q_rk = in_key
        in_start, in_stop = layout.offsets[in_key]
        for out_key in layout.keys:
            q_lb, q_pb, q_rb = out_key
            left_blocks = left.get((q_lb, q_lk))
            right_blocks = right.get((q_rb, q_rk))
            if left_blocks is None or right_blocks is None:
                continue
            out_start, out_stop = layout.offsets[out_key]
            if chain.rank_coupled:
                # Project the physical operator between the two fixed fusion
                # trees at the active site.  A raw W.reduced_block omits this
                # six-sector recoupling and even gives a wrong local norm for
                # the identity when neighboring bond spins differ.
                reduced_blocks = _left_reduced_rank_coupled_block(
                    core,
                    q_lb,
                    q_lk,
                    q_pb,
                    q_pk,
                    q_rb,
                    q_rk,
                )
                kernel = None
                for (left_index, right_index), mpo_block in reduced_blocks.items():
                    if (
                        left_index >= len(left_blocks)
                        or right_index >= len(right_blocks)
                    ):
                        continue
                    contribution = np.einsum(
                        "xal,xyop,ybr->aoblpr",
                        np.asarray(left_blocks[left_index]),
                        np.asarray(mpo_block),
                        np.asarray(right_blocks[right_index]),
                        optimize=True,
                    ).reshape(out_stop - out_start, in_stop - in_start)
                    kernel = contribution if kernel is None else kernel + contribution
            else:
                mpo_block = core.block(q_pb, q_pk)
                if mpo_block is None:
                    continue
                kernel = np.einsum(
                    "xal,xyop,ybr->aoblpr",
                    np.asarray(left_blocks),
                    np.asarray(mpo_block),
                    np.asarray(right_blocks),
                    optimize=True,
                ).reshape(out_stop - out_start, in_stop - in_start)
            if kernel is not None:
                matrix[out_start:out_stop, in_start:in_stop] += kernel
    return matrix


def _projected_mpo_local_problem(state, factors, site):
    if isinstance(factors, ReducedMPOHamiltonian):
        return _projected_canonical_local_problem(state, factors, site)
    factors = _validate_reduced_mpo(factors, state)
    frontier, _sites, hamiltonian_chain, metric_chain = (
        _frontier_environment_chains(state, factors)
    )
    embedding = frontier.site_embedding(state, site)
    projection = embedding.dense_matrix()
    target_h = _target_local_matrix(hamiltonian_chain, embedding, site)
    target_metric = _target_local_matrix(metric_chain, embedding, site)
    local_h = projection.conj().T @ target_h @ projection
    metric = projection.conj().T @ target_metric @ projection
    return ReducedLocalProblem(
        site=int(site),
        frame=None,
        hamiltonian=0.5 * (local_h + local_h.conj().T),
        metric=0.5 * (metric + metric.conj().T),
        embedding=embedding,
    )


def _expanded_source_frame(sites, embedding, site):
    template = sites[int(site)]
    columns = []
    for parameter in range(embedding.source_size):
        source = np.zeros(embedding.source_size, dtype=complex)
        source[parameter] = 1.0
        blocks = embedding.unpack_target(embedding.apply(source))
        local = NonabelianTensor(
            data=blocks,
            qns=[leg[:] for leg in template.qns],
            dirs=template.dirs[:],
            fusion_legs=template.fusion_legs[:],
            metadata=template.metadata.copy(),
        )
        columns.append(expand_reduced_mps_site(local).reshape(-1))
    return np.column_stack(columns)


def _canonical_source_action(chain, sites, embedding, site, source):
    template = sites[int(site)]
    blocks = embedding.unpack_target(embedding.apply(source))
    local = NonabelianTensor(
        data=blocks,
        qns=[leg[:] for leg in template.qns],
        dirs=template.dirs[:],
        fusion_legs=template.fusion_legs[:],
        metadata=template.metadata.copy(),
    )
    expanded = expand_reduced_mps_site(local)
    projected = reduce_expanded_mps_site(
        template, chain.local_action(site, expanded)
    )
    return embedding.adjoint(embedding.pack_target(projected))


def _projected_canonical_local_problem(
    state,
    hamiltonian,
    site,
    *,
    matrix_free=False,
    dense_solver_threshold=64,
):
    _validate_reduced_mpo(hamiltonian, state)
    frontier = ReducedFrontier.from_state(state)
    sites = tuple(frontier.to_mps(state))
    embedding = frontier.site_embedding(state, site)
    hamiltonian_chain = CanonicalEnvironmentChain.build(
        sites, hamiltonian.canonical_factors
    )
    metric_chain = CanonicalEnvironmentChain.build(
        sites, identity_canonical_factors(sites)
    )
    hamiltonian_action = lambda vector: _canonical_source_action(
        hamiltonian_chain, sites, embedding, site, vector
    )
    metric_action = lambda vector: _canonical_source_action(
        metric_chain, sites, embedding, site, vector
    )
    if bool(matrix_free) and embedding.source_size > int(dense_solver_threshold):
        return ReducedLocalProblem(
            site=int(site),
            frame=None,
            embedding=embedding,
            hamiltonian_action=hamiltonian_action,
            metric_action=metric_action,
        )

    source_frame = _expanded_source_frame(sites, embedding, site)
    local_h = hamiltonian_chain.local_matrix(site, source_frame)
    metric = metric_chain.local_matrix(site, source_frame)
    return ReducedLocalProblem(
        site=int(site),
        frame=None,
        embedding=embedding,
        hamiltonian=0.5 * (local_h + local_h.conj().T),
        metric=0.5 * (metric + metric.conj().T),
        hamiltonian_action=hamiltonian_action,
        metric_action=metric_action,
    )


def reduced_local_frame(state, site):
    """Linear map from one reduced LETTA core to the selected target multiplet."""

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("reduced_local_frame expects ReducedLatticeLETTA")
    site = int(site)
    frontier = ReducedFrontier.from_state(state)
    embedding = frontier.site_embedding(state, site)
    working = state.copy()
    columns = []
    for parameter in range(embedding.source_size):
        vector = np.zeros(embedding.source_size, dtype=complex)
        vector[parameter] = 1.0
        working.tensors[site] = embedding.unpack_source(vector)
        columns.append(np.asarray(working.state_vector(), dtype=complex))
    working.tensors[site] = {
        key: block.copy() for key, block in state.tensors[site].items()
    }
    return np.column_stack(columns), embedding


def reduced_local_problem(
    state,
    hamiltonian,
    site,
    *,
    matrix_free=False,
    dense_solver_threshold=64,
):
    """Build the exact projected one-site Hamiltonian and norm matrices."""

    if _is_reduced_mpo(hamiltonian):
        if isinstance(hamiltonian, ReducedMPOHamiltonian):
            return _projected_canonical_local_problem(
                state,
                hamiltonian,
                site,
                matrix_free=matrix_free,
                dense_solver_threshold=dense_solver_threshold,
            )
        return _projected_mpo_local_problem(state, hamiltonian, site)
    hamiltonian = _validate_dense_hamiltonian(hamiltonian, state)
    frame, embedding = reduced_local_frame(state, site)
    local_h = frame.conj().T @ hamiltonian @ frame
    metric = frame.conj().T @ frame
    return ReducedLocalProblem(
        site=int(site),
        frame=frame,
        embedding=embedding,
        hamiltonian=0.5 * (local_h + local_h.conj().T),
        metric=0.5 * (metric + metric.conj().T),
    )


def _lowest_generalized(problem, metric_tolerance, *, initial_vector=None):
    metric_values, metric_vectors = np.linalg.eigh(problem.metric)
    scale = max(float(np.max(metric_values, initial=0.0)), 0.0)
    if scale <= np.finfo(float).tiny:
        raise ValueError("reduced one-site metric has zero rank")
    cutoff = max(
        float(metric_tolerance),
        np.finfo(float).eps * problem.local_dimension,
    ) * scale
    keep = metric_values > cutoff
    if not np.any(keep):
        raise ValueError("reduced one-site metric has no retained directions")
    whitening = metric_vectors[:, keep] / np.sqrt(metric_values[keep])[None, :]
    transformed = whitening.conj().T @ problem.hamiltonian @ whitening
    transformed = 0.5 * (transformed + transformed.conj().T)
    values, vectors = np.linalg.eigh(transformed)
    vector = whitening @ vectors[:, 0]
    if initial_vector is not None:
        initial_vector = np.asarray(initial_vector).reshape(-1)
        if initial_vector.size != problem.local_dimension:
            raise ValueError("initial_vector has incompatible local dimension")
        degeneracy_tolerance = max(
            100.0 * np.finfo(float).eps,
            10.0 * float(metric_tolerance),
        ) * max(1.0, abs(float(values[0])))
        degenerate = values <= values[0] + degeneracy_tolerance
        subspace = whitening @ vectors[:, degenerate]
        coefficients = subspace.conj().T @ problem.metric @ initial_vector
        if np.linalg.norm(coefficients) > np.finfo(float).tiny:
            vector = subspace @ coefficients
    norm = np.sqrt(np.real(np.vdot(vector, problem.metric @ vector)))
    vector = vector / norm
    energy = float(
        np.real(np.vdot(vector, problem.hamiltonian @ vector))
        / np.real(np.vdot(vector, problem.metric @ vector))
    )
    residual = problem.hamiltonian @ vector - energy * (problem.metric @ vector)
    return energy, vector, int(np.count_nonzero(keep)), float(
        np.linalg.norm(residual)
    )


def _matrix_free_generalized_davidson(problem, options, initial_vector):
    """Lowest generalized root in an N-orthonormal Davidson space.

    The overlap operator may be positive semidefinite.  Null directions are
    removed during metric orthogonalization, so neither N nor a whitening
    matrix is ever materialized.
    """

    initial_vector = np.asarray(initial_vector, dtype=complex).reshape(-1)
    if initial_vector.size != problem.local_dimension:
        raise ValueError("initial_vector has incompatible local dimension")
    metric_initial = problem.apply_metric(initial_vector)
    norm_squared = float(np.real(np.vdot(initial_vector, metric_initial)))
    if not np.isfinite(norm_squared) or norm_squared <= np.finfo(float).tiny:
        raise ValueError("matrix-free reduced local metric has zero initial norm")
    initial_vector = initial_vector / np.sqrt(norm_squared)
    metric_initial = metric_initial / np.sqrt(norm_squared)
    dimension = problem.local_dimension
    tolerance = float(options.eigensolver_tolerance)
    lindep = max(
        float(options.metric_tolerance), np.finfo(float).eps * dimension
    )
    max_space = min(dimension, 48)
    minimum_explored = min(dimension, max_space, 16)
    basis = initial_vector[:, None]
    metric_basis = metric_initial[:, None]
    hamiltonian_basis = problem.apply_hamiltonian(initial_vector)[:, None]
    seed_cursor = 0

    def metric_orthogonalize(candidate, metric_candidate=None):
        candidate = np.asarray(candidate, dtype=complex).reshape(-1)
        metric_candidate = (
            problem.apply_metric(candidate)
            if metric_candidate is None
            else np.asarray(metric_candidate, dtype=complex).reshape(-1)
        )
        for _ in range(2):
            overlaps = basis.conj().T @ metric_candidate
            candidate = candidate - basis @ overlaps
            metric_candidate = metric_candidate - metric_basis @ overlaps
        metric_norm_squared = float(np.real(np.vdot(candidate, metric_candidate)))
        scale = float(np.linalg.norm(candidate) * np.linalg.norm(metric_candidate))
        if (
            not np.isfinite(metric_norm_squared)
            or metric_norm_squared <= lindep * max(scale, np.finfo(float).tiny)
        ):
            return None, None
        normalization = np.sqrt(metric_norm_squared)
        return candidate / normalization, metric_candidate / normalization

    def next_independent_seed():
        nonlocal seed_cursor
        while seed_cursor < dimension:
            candidate = np.zeros(dimension, dtype=complex)
            candidate[seed_cursor] = 1.0
            seed_cursor += 1
            candidate, metric_candidate = metric_orthogonalize(candidate)
            if candidate is not None:
                return candidate, metric_candidate
        return None, None

    def lowest_projected_root():
        projected = hamiltonian_basis.conj().T @ basis
        projected = 0.5 * (projected + projected.conj().T)
        values, vectors = np.linalg.eigh(projected)
        degeneracy_tolerance = max(
            100.0 * np.finfo(float).eps,
            10.0 * tolerance,
        ) * max(1.0, abs(float(values[0])))
        degenerate = values <= values[0] + degeneracy_tolerance
        subspace = vectors[:, degenerate]
        reference_coefficients = basis.conj().T @ metric_initial
        weights = subspace.conj().T @ reference_coefficients
        if np.linalg.norm(weights) > np.finfo(float).tiny:
            coefficients = subspace @ weights
            coefficients /= np.linalg.norm(coefficients)
        else:
            coefficients = vectors[:, 0]
        return float(values[0]), coefficients, values, vectors

    energy = np.inf
    residual = np.zeros(dimension, dtype=complex)
    coefficients = np.ones(1, dtype=complex)
    for _iteration in range(int(options.eigensolver_max_iterations)):
        energy, coefficients, _values, projected_vectors = lowest_projected_root()
        vector = basis @ coefficients
        hamiltonian_vector = hamiltonian_basis @ coefficients
        metric_vector = metric_basis @ coefficients
        residual = hamiltonian_vector - energy * metric_vector
        residual_scale = max(
            1.0,
            abs(energy),
            float(np.linalg.norm(hamiltonian_vector)),
            abs(energy) * float(np.linalg.norm(metric_vector)),
        )
        converged = np.linalg.norm(residual) <= tolerance * residual_scale
        candidate = None
        metric_candidate = None
        if not converged:
            candidate, metric_candidate = metric_orthogonalize(-residual)
        if candidate is None and (not converged or basis.shape[1] < minimum_explored):
            candidate, metric_candidate = next_independent_seed()
        if candidate is None:
            break
        if basis.shape[1] >= max_space:
            retained = min(4, projected_vectors.shape[1])
            restart = projected_vectors[:, :retained]
            basis = basis @ restart
            metric_basis = metric_basis @ restart
            hamiltonian_basis = hamiltonian_basis @ restart
            candidate, metric_candidate = metric_orthogonalize(
                candidate, metric_candidate
            )
            if candidate is None:
                candidate, metric_candidate = next_independent_seed()
            if candidate is None:
                break
        basis = np.column_stack((basis, candidate))
        metric_basis = np.column_stack((metric_basis, metric_candidate))
        hamiltonian_basis = np.column_stack(
            (hamiltonian_basis, problem.apply_hamiltonian(candidate))
        )

    energy, coefficients, _values, _vectors = lowest_projected_root()
    vector = basis @ coefficients
    return energy, vector, basis.shape[1]


def _lowest_generalized_matrix_free(problem, options, *, initial_vector):
    """Solve a projected generalized problem using only exact matvecs."""

    energy, vector, explored_rank = _matrix_free_generalized_davidson(
        problem, options, initial_vector
    )
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    metric_vector = problem.apply_metric(vector)
    norm_squared = float(np.real(np.vdot(vector, metric_vector)))
    if not np.isfinite(norm_squared) or norm_squared <= np.finfo(float).tiny:
        raise ValueError("matrix-free reduced eigensolver returned a null vector")
    vector /= np.sqrt(norm_squared)
    metric_vector = problem.apply_metric(vector)
    hamiltonian_vector = problem.apply_hamiltonian(vector)
    energy = float(
        np.real(np.vdot(vector, hamiltonian_vector))
        / np.real(np.vdot(vector, metric_vector))
    )
    residual = hamiltonian_vector - energy * metric_vector
    # A full rank determination itself would require metric materialization;
    # report the number of independent N-directions actually explored.
    return energy, vector, explored_rank, float(np.linalg.norm(residual))


def _solve_local_problem(problem, options, *, initial_vector):
    if problem.hamiltonian is None or problem.metric is None:
        return _lowest_generalized_matrix_free(
            problem, options, initial_vector=initial_vector
        )
    return _lowest_generalized(
        problem,
        options.metric_tolerance,
        initial_vector=initial_vector,
    )


def _mpo_expectation(state, factors, *, stable=False):
    if isinstance(factors, ReducedMPOHamiltonian):
        sites = tuple(ReducedFrontier.from_state(state).to_mps(state))
        chain = CanonicalEnvironmentChain.build(
            sites, factors.canonical_factors
        )
        return chain.stable_expectation() if stable else chain.expectation()
    factors = _validate_reduced_mpo(factors, state)
    sites = ReducedFrontier.from_state(state).to_mps(state)
    chain = BlockSparseEnvironmentChain.build(sites, factors)
    final = chain.left_envs[-1].advance(
        chain.mpo_factors[-1],
        sites[-1],
        sites[-1],
        **(
            {}
            if chain.rank_coupled
            else {"phys_slices": chain.site_layouts[-1]["sector_slices"][1]}
        ),
    )
    return final.expectation()


def _energy(state, hamiltonian, *, stable=False):
    if _is_reduced_mpo(hamiltonian):
        numerator = _mpo_expectation(state, hamiltonian, stable=stable)
        if isinstance(hamiltonian, ReducedMPOHamiltonian):
            sites = tuple(ReducedFrontier.from_state(state).to_mps(state))
            metric_chain = CanonicalEnvironmentChain.build(
                sites, identity_canonical_factors(sites)
            )
            denominator = (
                metric_chain.stable_expectation()
                if stable
                else metric_chain.expectation()
            )
        else:
            denominator = _mpo_expectation(
                state, _identity_mpo_factors(tuple(hamiltonian))
            )
        return float(np.real(numerator / denominator))
    vector = np.asarray(state.state_vector(), dtype=complex)
    denominator = np.vdot(vector, vector)
    return float(np.real(np.vdot(vector, hamiltonian @ vector) / denominator))


def optimize_reduced_site(state, hamiltonian, site, options):
    """Solve and accept one exact reduced local generalized eigenproblem."""

    from .solver import LETTASiteUpdate

    problem = reduced_local_problem(
        state,
        hamiltonian,
        site,
        matrix_free=options.matrix_free,
        dense_solver_threshold=options.dense_solver_threshold,
    )
    local_energy, vector, metric_rank, residual = _solve_local_problem(
        problem,
        options,
        initial_vector=problem.embedding.pack_source(state.tensors[int(site)]),
    )
    old_blocks = {
        key: block.copy() for key, block in state.tensors[int(site)].items()
    }
    old_energy = _energy(state, hamiltonian, stable=True)
    state.tensors[int(site)] = problem.embedding.unpack_source(vector)
    new_energy = _energy(state, hamiltonian, stable=True)
    accepted = new_energy <= old_energy + options.energy_increase_tolerance
    if not accepted:
        state.tensors[int(site)] = old_blocks
        new_energy = old_energy
    else:
        state.balance_scalar_gauge()
    return LETTASiteUpdate(
        site=int(site),
        local_energy=local_energy,
        energy=new_energy,
        metric_rank=metric_rank,
        local_dimension=problem.local_dimension,
        residual_norm=residual,
        accepted=accepted,
        full_local_dimension=problem.full_local_dimension,
    )


def reduced_letta_dmrg(hamiltonian, *, state, options):
    """Alternate exact one-site sweeps in reduced LETTA parameter spaces."""

    from .solver import LETTADMRGResult, LETTASweep

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("state must be ReducedLatticeLETTA")
    if _is_reduced_mpo(hamiltonian):
        validated_factors = _validate_reduced_mpo(hamiltonian, state)
        if not isinstance(hamiltonian, ReducedMPOHamiltonian):
            hamiltonian = validated_factors
    else:
        hamiltonian = _validate_dense_hamiltonian(hamiltonian, state)
    state = state.copy()
    direction = str(options.start_direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("start_direction must be 'lr' or 'rl'")
    previous_energy = _energy(state, hamiltonian, stable=True)
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    nominal_bond = max((len(bond) for bond in state.bond_sectors), default=1)
    for sweep in range(1, int(options.max_sweeps) + 1):
        sites = (
            range(state.nsites)
            if direction == "lr"
            else range(state.nsites - 1, -1, -1)
        )
        updates = tuple(
            optimize_reduced_site(state, hamiltonian, site, options)
            for site in sites
        )
        energy = _energy(state, hamiltonian, stable=True)
        change = abs(energy - previous_energy)
        density_change = change / state.nsites
        history.append(
            LETTASweep(
                sweep=sweep,
                direction=direction,
                energy=energy,
                energy_change=change,
                energy_density_change=density_change,
                bond_dimension=nominal_bond,
                updates=updates,
            )
        )
        if options.verbosity:
            print(
                f"reduced SU(2) LETTA sweep {sweep:3d}  direction={direction}  "
                f"energy={energy:.14f}  dE/site={density_change:.3e}"
            )
        if density_change <= options.tolerance:
            converged = True
            message = "CONVERGENCE: SWEEP ENERGY DENSITY CHANGE <= TOLERANCE"
            break
        previous_energy = energy
        if options.alternate:
            direction = "rl" if direction == "lr" else "lr"
    final_energy = _energy(state, hamiltonian, stable=True)
    return LETTADMRGResult(
        state=state,
        energy=float(final_energy),
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
        max_boundary_discarded_weight=0.0,
    )


__all__ = [
    "ReducedLocalProblem",
    "optimize_reduced_site",
    "reduced_letta_dmrg",
    "reduced_local_frame",
    "reduced_local_problem",
]

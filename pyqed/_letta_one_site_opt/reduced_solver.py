"""Exact reduced-SU(2) one-site LETTA local problems and sweeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

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
from .reduced_norm import ReducedNormChain, left_canonical_reduced_sites
from .reduced_environment import ReducedEnvironmentChain


@dataclass(frozen=True)
class ReducedLocalProblem:
    site: int
    frame: np.ndarray | None
    embedding: FrontierSiteEmbedding
    hamiltonian: np.ndarray | None = None
    metric: np.ndarray | None = None
    hamiltonian_action: object | None = None
    metric_action: object | None = None
    equilibrated_projector_factory: object | None = None
    coordinates: object | None = None

    @property
    def local_dimension(self):
        return self.embedding.source_size

    @property
    def full_local_dimension(self):
        return self.embedding.target_size

    def apply_hamiltonian(self, vector):
        if self.coordinates is not None:
            return self.coordinates.source_hamiltonian(np.asarray(vector))
        if self.hamiltonian is not None:
            return self.hamiltonian @ np.asarray(vector)
        if self.hamiltonian_action is None:
            raise RuntimeError("local Hamiltonian action is unavailable")
        return np.asarray(self.hamiltonian_action(vector))

    def apply_metric(self, vector):
        if self.coordinates is not None:
            return self.coordinates.metric(np.asarray(vector))
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
        (hamiltonian.factors or ())
        if wrapped
        else tuple(hamiltonian)
    )
    if (not wrapped or hamiltonian.factors is not None) and len(factors) != state.nsites:
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
    if factors and all(rank_coupled) and not wrapped:
        raise ValueError(
            "rank-coupled factors require ReducedMPOHamiltonian with an exact "
            "canonical_factors view; raw rank-coupled chains are ambiguous "
            "beyond two sites"
        )
    if wrapped:
        if len(hamiltonian.canonical_factors) != state.nsites:
            raise ValueError('canonical MPO must contain one factor per site')
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
            shape = getattr(core, 'component_shape', None)
            if shape is None:
                shape = np.asarray(core.as_dense() if hasattr(core, "as_dense") else core).shape
            if len(shape) != 4:
                raise ValueError(f"canonical MPO factor {site} is not rank four")
            if shape[2:] != (
                state.dense_physical_dim,
                state.dense_physical_dim,
            ):
                raise ValueError(
                    f"canonical MPO physical dimensions at site {site} do not "
                    "match the state's magnetic-component basis"
                )
            canonical_shapes.append(shape)
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
    sites = tuple(frontier.to_mps(state, compact=hamiltonian.contraction_backend == 'reduced'))
    embedding = frontier.site_embedding(state, site)
    if hamiltonian.contraction_backend == 'reduced':
        return _native_local_problem(state, hamiltonian, site, sites, embedding,
                                     matrix_free, dense_solver_threshold)
    hamiltonian_chain = CanonicalEnvironmentChain.build(
        sites, hamiltonian.canonical_factors
    )
    metric_chain = ReducedNormChain.build(sites)
    hamiltonian_action = lambda vector: _canonical_source_action(
        hamiltonian_chain, sites, embedding, site, vector
    )
    metric_action = lambda vector: embedding.adjoint(embedding.pack_target(
        metric_chain.local_action(site, embedding.unpack_target(embedding.apply(vector)))
    ))
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
    metric = np.column_stack([metric_action(v) for v in np.eye(embedding.source_size)])
    return ReducedLocalProblem(
        site=int(site),
        frame=None,
        embedding=embedding,
        hamiltonian=0.5 * (local_h + local_h.conj().T),
        metric=0.5 * (metric + metric.conj().T),
        hamiltonian_action=hamiltonian_action,
        metric_action=metric_action,
    )


def _native_local_problem(state, hamiltonian, site, sites, embedding,
                          matrix_free, dense_solver_threshold, boundary_cache=None, structured=False):
    from .reduced_coordinates import ReducedLocalCoordinates
    coordinates=ReducedLocalCoordinates(sites,embedding,site,mpo=hamiltonian.native_mpo(state.physical_basis),boundary_cache=boundary_cache,action_only=True,structured=structured)
    coordinates.set_hamiltonian(hamiltonian.native_mpo(state.physical_basis))
    h_action=coordinates.source_hamiltonian
    n_action=coordinates.metric
    h,n=None,None
    if not matrix_free or embedding.source_size<=dense_solver_threshold:
        identity=np.eye(embedding.source_size)
        h=np.column_stack([h_action(v) for v in identity])
        n=np.column_stack([n_action(v) for v in identity])
        h=.5*(h+h.conj().T);n=.5*(n+n.conj().T)
    return ReducedLocalProblem(site=int(site),frame=None,embedding=embedding,
        hamiltonian=h,metric=n,hamiltonian_action=h_action,metric_action=n_action,
        equilibrated_projector_factory=coordinates.projector,coordinates=coordinates)


class StableEnergyCache:
    """Reuse QR-checked energy only for identical tensor values and fixed H.

    Holds a scalar signature and bounded independent QR energy prefixes.
    Changed values always trigger a fresh contraction of the changed suffix.
    """
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.key = self.value = None
        from .frontier_contraction import EnergyBoundaryCache
        self.boundaries = EnergyBoundaryCache()

    def __call__(self, state, *, center=None, canonical_boundaries=None):
        from .frontier_contraction import state_signature
        key = state_signature(state)
        if key != self.key:
            value = _energy(state, self.hamiltonian, stable=True, energy_boundaries=self.boundaries,
                            center=center, canonical_boundaries=canonical_boundaries)
            self.key, self.value = key, value
        return self.value


class ReducedSweepContext:
    """Cache supported one-site coordinates independently of the gauge layout.

    Conditional gauges retain their own Cartesian QR boundary. This context
    contracts the resulting tensors on exact symmetry support and rebuilds all
    frontier coordinates when any structural layout changes.
    """

    def __init__(self, state, hamiltonian, *, prune=True):
        self.state, self.hamiltonian = state, hamiltonian
        self.prune = bool(prune)
        self._rebuild_layout()
        from .frontier_contraction import CanonicalBoundaryCache
        self.boundary_cache = CanonicalBoundaryCache()
        self._energy_key = None
        self._energy_value = None
        from .frontier_contraction import EnergyBoundaryCache
        self.energy_boundaries = EnergyBoundaryCache()

    def stable_energy(self, state=None):
        """Reuse an independent contraction only for identical tensor values.

        The Hamiltonian is fixed for the lifetime of this sweep context.
        Normalization, gauges, and in-place changes all change the value key.
        """
        from .frontier_contraction import state_signature
        state=self.state if state is None else state
        key=state_signature(state)
        if key!=self._energy_key:
            value=_energy(state,self.hamiltonian,stable=True,energy_boundaries=self.energy_boundaries)
            self._energy_key,self._energy_value=key,value
        return self._energy_value

    def _layout_signature(self):
        state = self.state
        physical = tuple((q.sector, q.copy) for q in state.physical_basis.reduced_states)
        blocks = tuple(tuple((key, a.shape) for key, a in sorted(tensor.items()))
                       for tensor in state.tensors)
        return (state.neighborhoods, state.bond_sectors, state.symmetry.identity,
                state.symmetry.sector, physical, blocks)

    def _rebuild_layout(self):
        self.frontier = ReducedFrontier.from_state(self.state, prune=self.prune)
        self.embeddings = tuple(self.frontier.site_embedding(self.state, i)
                                for i in range(self.state.nsites))
        self.sites = list(self.frontier.to_mps(self.state, compact=True))
        self.n_chain = ReducedNormChain.build(self.sites, lazy=True)
        self._structure = self._layout_signature()
        self._energy_key = None

    def synchronize(self, indices):
        if self._layout_signature() != self._structure:
            self._rebuild_layout()
            return
        changes = {}
        for i in indices:
            from .frontier_contraction import shallow_site, scatter_blocks
            a = shallow_site(self.sites[i])
            a.data = scatter_blocks(self.embeddings[i],self.state.tensors[i])
            changes[i] = self.sites[i] = a
        self.n_chain.replace_sites(changes)

    def local_problem(self, site, options):
        return _native_local_problem(self.state, self.hamiltonian, site, self.sites,
            self.embeddings[site], options.matrix_free, options.dense_solver_threshold, self.boundary_cache)


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
    from .contractions import _equilibrated_metric_factors

    # Rank belongs to normalized Gram columns, not their arbitrary tensor
    # coordinate scales. Otherwise an invertible gauge can erase a physical
    # direction before the energy minimization even starts.
    whitening, _ = _equilibrated_metric_factors(problem.metric, metric_tolerance)
    metric_rank = whitening.shape[1]
    if not metric_rank:
        raise ValueError("reduced one-site metric has no retained directions")
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
    return energy, vector, metric_rank, float(
        np.linalg.norm(residual)
    )


def _matrix_free_generalized_davidson(problem, options, initial_vector):
    """Equilibrate Gram columns without materializing the full local metric."""
    dimension = problem.local_dimension
    initial_vector = np.asarray(initial_vector, dtype=complex).reshape(-1)
    if initial_vector.size != dimension:
        raise ValueError("initial_vector has incompatible local dimension")
    if problem.metric is not None:
        diagonal = np.diag(problem.metric).real.copy()
    else:
        diagonal = np.empty(dimension)
        unit = np.zeros(dimension, dtype=complex)
        for j in range(dimension):
            unit[j] = 1.
            diagonal[j] = problem.apply_metric(unit)[j].real
            unit[j] = 0.
    if not np.all(np.isfinite(diagonal)):
        raise FloatingPointError('nonfinite reduced metric diagonal')
    scale = max(float(np.max(diagonal, initial=0.)), np.finfo(float).tiny)
    if np.min(diagonal, initial=0.) < -1e-10*scale:
        raise FloatingPointError('negative reduced metric diagonal')
    supported = diagonal > 0.
    roots = np.sqrt(np.maximum(diagonal, 0.))
    inverse = np.zeros(dimension)
    inverse[supported] = 1/roots[supported]
    # The raw-coordinate projector is not valid after this change of basis.
    # Native open-chain adapters provide the conditional Gram projector in
    # equilibrated coordinates; generic correlated metrics retain Davidson's
    # metric-orthogonality/null-direction checks.
    scaled = SimpleNamespace(local_dimension=dimension,
        apply_hamiltonian=lambda x: inverse*problem.apply_hamiltonian(inverse*x),
        apply_metric=lambda x: inverse*problem.apply_metric(inverse*x),
        # A PSD correlation matrix has ||C||_2 <= trace(C).
        metric_scale=float(np.count_nonzero(supported)),
        metric_projector_factory=getattr(problem, 'equilibrated_projector_factory', None))
    energy, vector, rank = _davidson_in_metric_coordinates(
        scaled, options, roots*np.asarray(initial_vector))
    return energy, inverse*vector, rank


def _davidson_in_metric_coordinates(problem, options, initial_vector):
    """Lowest generalized root in an N-orthonormal Davidson space.

    The overlap operator may be positive semidefinite.  Null directions are
    removed during metric orthogonalization, so neither N nor a whitening
    matrix is ever materialized.

    This residual-subspace variant retains full-support probes
    and auxiliary-root exploration for an unchanged incumbent. An optional
    preconditioner changes search directions only, with a raw-residual fallback.
    The unpreconditioned algorithm remains the default. After a resolved
    energy decrease it converges only the lowest Ritz root. Finite exploration
    is a safeguard against a stationary excited start, not a proof of the global
    lowest eigenvalue for every operator and initial vector.
    """

    initial_vector = np.asarray(initial_vector, dtype=complex).reshape(-1)
    if initial_vector.size != problem.local_dimension:
        raise ValueError("initial_vector has incompatible local dimension")
    dimension = problem.local_dimension
    lindep = max(float(options.metric_tolerance), np.finfo(float).eps*dimension)
    factory = getattr(problem, 'metric_projector_factory', None)
    projector = None if factory is None else factory(lindep)
    if projector is not None:
        initial_vector = projector(initial_vector)
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
    metric_scale = problem.metric_scale
    basis = initial_vector[:, None]
    metric_basis = metric_initial[:, None]
    hamiltonian_basis = problem.apply_hamiltonian(initial_vector)[:, None]
    initial_rayleigh = float(np.vdot(initial_vector, hamiltonian_basis[:, 0]).real)
    seed_cursor = 0

    def metric_orthogonalize(candidate, metric_candidate=None):
        candidate = np.asarray(candidate, dtype=complex).reshape(-1)
        if projector is not None:
            candidate = projector(candidate)
            metric_candidate = None
        metric_candidate = (
            problem.apply_metric(candidate)
            if metric_candidate is None
            else np.asarray(metric_candidate, dtype=complex).reshape(-1)
        )
        original_norm = abs(float(np.real(np.vdot(candidate, metric_candidate))))
        for _ in range(2):
            overlaps = basis.conj().T @ metric_candidate
            candidate = candidate - basis @ overlaps
            metric_candidate = metric_candidate - metric_basis @ overlaps
        # Subtracting cached N-vectors can leave cancellation noise after the
        # actual vector has entered the metric nullspace. Reapply N and compare
        # with the *pre-orthogonalization* norm before amplifying that noise.
        metric_candidate = problem.apply_metric(candidate)
        metric_norm_squared = float(np.real(np.vdot(candidate, metric_candidate)))
        scale = float(np.linalg.norm(candidate) * np.linalg.norm(metric_candidate))
        euclidean_norm_squared = float(np.real(np.vdot(candidate, candidate)))
        if (
            not np.isfinite(metric_norm_squared)
            or metric_norm_squared <= lindep * max(original_norm, scale,
                metric_scale*euclidean_norm_squared, np.finfo(float).tiny)
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

    # A warm start can be an exact excited eigenvector. Its residual then
    # supplies no search direction, and a short prefix of coordinate seeds
    # may miss an entire lower invariant subspace. Reproducible full-support
    # probes explore every coordinate before testing residual convergence.
    # They are search vectors only: no random perturbation of the state or
    # Hamiltonian is made, and the incumbent remains in the Ritz space.
    rng = np.random.default_rng(271828)
    for _ in range(min(2, dimension - 1)):
        candidate, metric_candidate = metric_orthogonalize(
            rng.normal(size=dimension) + 1j*rng.normal(size=dimension))
        if candidate is not None:
            basis = np.column_stack((basis, candidate))
            metric_basis = np.column_stack((metric_basis, metric_candidate))
            hamiltonian_basis = np.column_stack(
                (hamiltonian_basis, problem.apply_hamiltonian(candidate)))

    energy = np.inf
    residual = np.zeros(dimension, dtype=complex)
    coefficients = np.ones(basis.shape[1], dtype=complex)
    for _iteration in range(int(options.eigensolver_max_iterations)):
        energy, coefficients, _values, projected_vectors = lowest_projected_root()
        candidate = None
        metric_candidate = None
        converged = True
        # Follow the exploratory Ritz roots as well as the incumbent. Merely
        # adding random vectors is insufficient: their initial Rayleigh values
        # may exceed an already stationary excited incumbent, whose residual
        # would then suppress all further Hamiltonian-driven exploration.
        # Keep the excited-incumbent guard until the lowest Ritz energy has
        # decreased. Thereafter target only the requested ground-state root;
        # converging auxiliary excited roots needlessly consumes matvecs.
        explore = not (energy < initial_rayleigh - tolerance*max(1., abs(initial_rayleigh)))
        roots = [coefficients] + ([projected_vectors[:, j]
            for j in range(1, min(3, projected_vectors.shape[1]))] if explore else [])
        for root in roots:
            hamiltonian_vector = hamiltonian_basis @ root
            metric_vector = metric_basis @ root
            root_energy = float(np.real(np.vdot(basis @ root, hamiltonian_vector)))
            residual = hamiltonian_vector - root_energy * metric_vector
            residual_scale = max(
                1.0, abs(root_energy), float(np.linalg.norm(hamiltonian_vector)),
                abs(root_energy) * float(np.linalg.norm(metric_vector)))
            if np.linalg.norm(residual) <= tolerance * residual_scale:
                continue
            converged = False
            preconditioner=getattr(problem,'preconditioner',None)
            direction=-residual if preconditioner is None else preconditioner(-residual,root_energy)
            candidate, metric_candidate = metric_orthogonalize(direction)
            if candidate is None and preconditioner is not None:
                candidate, metric_candidate = metric_orthogonalize(-residual)
            if candidate is not None:
                break
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


def _lowest_coordinate_root(problem, options, initial_vector):
    coordinates=problem.coordinates
    coordinates.prepare(options.metric_tolerance)
    initial=coordinates.coordinates(initial_vector)
    if coordinates.rank==0:
        raise ValueError('Reduced local metric has no retained directions')
    local_options=replace(options,eigensolver_tolerance=options.eigensolver_tolerance/
                          max(1.,coordinates.residual_amplification))
    if not options.matrix_free or coordinates.rank<=options.dense_solver_threshold:
        h=np.column_stack([coordinates.apply(v) for v in np.eye(coordinates.rank)])
        local=SimpleNamespace(hamiltonian=.5*(h+h.conj().T),metric=np.eye(coordinates.rank),
                              local_dimension=coordinates.rank)
        _,z,_,_=_lowest_generalized(local,options.metric_tolerance,initial_vector=initial)
    else:
        local=SimpleNamespace(local_dimension=coordinates.rank,apply_hamiltonian=coordinates.apply,
                              apply_metric=lambda z:z,metric_scale=1.)
        import os
        preconditioner=os.environ.get('PYQED_LOCAL_PRECONDITIONER','none')
        if preconditioner not in ('none','blocks'):
            raise ValueError('PYQED_LOCAL_PRECONDITIONER must be none or blocks')
        if preconditioner=='blocks':local.preconditioner=coordinates.preconditioned_residual
        _,z,_=_davidson_in_metric_coordinates(local,local_options,initial)
    vector=coordinates.expand(z)
    norm=coordinates.norm(vector)
    if not np.isfinite(norm) or norm<=np.finfo(float).tiny:
        raise FloatingPointError('Null QR-coordinate local solution')
    vector/=np.sqrt(norm)
    from .reduced_updates import local_residual_diagnostics
    diagnostics=local_residual_diagnostics(problem,vector,metric_tolerance=options.metric_tolerance)
    coordinates.solution_diagnostics=(vector.copy(),diagnostics)
    return diagnostics['energy'],vector,coordinates.rank,diagnostics['residual_norm']


def _solve_local_problem(problem, options, *, initial_vector):
    if getattr(problem,'coordinates',None) is not None:
        return _lowest_coordinate_root(problem,options,initial_vector)
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
        if factors.contraction_backend == 'reduced':
            if stable:
                sites = left_canonical_reduced_sites(sites)
            chain = ReducedEnvironmentChain.build(sites, factors.native_mpo(state.physical_basis), lazy=True)
            return chain.expectation()
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


def _energy(state, hamiltonian, *, stable=False, energy_boundaries=None, center=None, canonical_boundaries=None):
    if (isinstance(hamiltonian, ReducedMPOHamiltonian)
            and hamiltonian.contraction_backend == 'reduced'):
        if stable and center is not None:
            from .frontier_contraction import centered_energy
            numerator,denominator=centered_energy(state,hamiltonian.native_mpo(state.physical_basis),
                center,cache=canonical_boundaries)
            return _checked_energy(numerator,denominator)
        if stable:
            from .dependency_environment import fixed_dependency_view
            state,_=fixed_dependency_view(state)
        if stable and ReducedFrontier.from_state(state).requires_compact(state):
            from .frontier_contraction import frontier_energy
            numerator, denominator = frontier_energy(state, hamiltonian.native_mpo(state.physical_basis), cache=energy_boundaries)
            return _checked_energy(numerator, denominator)
        sites = tuple(ReducedFrontier.from_state(state).to_mps(state))
        if stable:
            sites = left_canonical_reduced_sites(sites)
        chain = ReducedEnvironmentChain.build(
            sites, hamiltonian.native_mpo(state.physical_basis), lazy=True)
        numerator = chain.expectation()
        denominator = ReducedNormChain.build(sites).expectation()
        return _checked_energy(numerator, denominator)
    if _is_reduced_mpo(hamiltonian):
        numerator = _mpo_expectation(state, hamiltonian, stable=stable)
        if isinstance(hamiltonian, ReducedMPOHamiltonian):
            sites = tuple(ReducedFrontier.from_state(state).to_mps(state))
            if stable:
                sites = left_canonical_reduced_sites(sites)
            denominator = ReducedNormChain.build(sites).expectation()
        else:
            denominator = _mpo_expectation(
                state, _identity_mpo_factors(tuple(hamiltonian))
            )
        return _checked_energy(numerator, denominator)
    vector = np.asarray(state.state_vector(), dtype=complex)
    denominator = np.vdot(vector, vector)
    return _checked_energy(np.vdot(vector, hamiltonian @ vector), denominator)


def _checked_energy(numerator, denominator):
    if (not np.isfinite(denominator) or np.real(denominator) <= np.finfo(float).tiny
            or abs(np.imag(denominator)) > 1e-10*abs(denominator)):
        raise FloatingPointError("nonpositive, complex or nonfinite reduced state norm")
    value = numerator / denominator
    if not np.isfinite(value) or abs(np.imag(value)) > 1e-10*max(1., abs(value)):
        raise FloatingPointError("complex or nonfinite reduced energy")
    return float(np.real(value))


def _optimize_reduced_site_impl(state, hamiltonian, site, options, *, context=None, energy_cache=None):
    """Propose a normalized update; the caller checks energy and commits it."""

    from .solver import LETTASiteUpdate

    problem = context.local_problem(site, options) if context is not None else reduced_local_problem(
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
    from .reduced_updates import local_residual_diagnostics
    saved=getattr(problem.coordinates,'solution_diagnostics',None)
    diagnostics=(saved[1] if saved is not None and np.array_equal(saved[0],vector) else
                 local_residual_diagnostics(problem,vector,metric_tolerance=options.metric_tolerance))
    residual=diagnostics['residual_norm']
    relative_residual=diagnostics['relative_residual']
    # Normalize the candidate before the transaction's independent energy
    # check. Checking both unnormalized and normalized copies repeats an
    # expensive contraction of the same physical state.
    state.tensors[int(site)] = problem.embedding.unpack_source(vector)
    if context is None:
        state.normalize(center=int(site), balance=options.gauge_mode != 'frontier')
    else:
        norm = diagnostics['norm_squared']
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise FloatingPointError('Null or nonfinite local candidate norm')
        state.tensors[int(site)] = problem.embedding.unpack_source(
            vector*np.sqrt((state.target_two_j+1)/norm))
    if context is not None:
        context.synchronize([site])
    return LETTASiteUpdate(
        site=int(site),
        local_energy=local_energy,
        energy=local_energy,
        metric_rank=metric_rank,
        local_dimension=problem.local_dimension,
        residual_norm=residual,
        relative_residual=relative_residual,
        raw_residual_norm=diagnostics['raw_residual_norm'],
        raw_relative_residual=diagnostics['raw_relative_residual'],
        local_converged=relative_residual <= options.eigensolver_tolerance,
        accepted=True,
        full_local_dimension=problem.full_local_dimension,
    )


def optimize_reduced_site(state, hamiltonian, site, options, *, context=None, energy_cache=None):
    """Commit a finite energy-checked update, or restore all tensors and caches."""
    from .reduced_updates import NUMERICAL_ERRORS
    evaluate = (context.stable_energy if context is not None else energy_cache
                if energy_cache is not None else lambda s: _energy(s, hamiltonian, stable=True))
    snapshot = state.copy()
    old_energy = evaluate(snapshot)
    def restore():
        state.tensors, state.bond_sectors = snapshot.tensors, snapshot.bond_sectors
        if context is not None:
            context.synchronize(list(range(state.nsites)))
    try:
        update = _optimize_reduced_site_impl(state, hamiltonian, site, options, context=context, energy_cache=energy_cache)
        actual = evaluate(state)
        if not np.isfinite(actual) or actual > old_energy+options.energy_increase_tolerance:
            restore()
            return replace(update, accepted=False, energy=old_energy, local_converged=False)
        return replace(update, energy=actual)
    except NUMERICAL_ERRORS:
        restore()
        raise


def reduced_letta_dmrg(hamiltonian, *, state, options, sweep_callback=None):
    """Alternate exact one-site sweeps in reduced LETTA parameter spaces.

    Optional ``sweep_callback(state, sweep)`` observes each completed sweep,
    before the convergence audit. It must not mutate the live state; copy or
    serialize it to retain a checkpoint. Callback exceptions propagate.
    """

    from .solver import LETTADMRGResult, LETTASiteUpdate, LETTASweep
    from .reduced_updates import NUMERICAL_ERRORS

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("state must be ReducedLatticeLETTA")
    if _is_reduced_mpo(hamiltonian):
        validated_factors = _validate_reduced_mpo(hamiltonian, state)
        if not isinstance(hamiltonian, ReducedMPOHamiltonian):
            hamiltonian = validated_factors
    else:
        hamiltonian = _validate_dense_hamiltonian(hamiltonian, state)
    if options.cbe_enabled:
        if not isinstance(hamiltonian, ReducedMPOHamiltonian) or hamiltonian.contraction_backend != 'reduced':
            raise ValueError('reduced CBE requires a native ReducedMPOHamiltonian')
        if options.cbe_selector not in {'exact', 'mps-dd'}:
            raise ValueError('reduced CBE supports exact (LETTA) or mps-dd (untied OBC MPS)')
        if options.cbe_selector == 'mps-dd' and state.neighborhoods != tuple((i,) for i in range(state.nsites)):
            raise ValueError('mps-dd CBE requires an untied open-boundary MPS')
        if options.cbe_baseline_guard_fraction != 0.:
            raise ValueError('reduced CBE requires the strict one-site baseline guard (fraction=0)')
        if options.cbe_preselection_dimension is not None:
            raise ValueError('reduced exact CBE does not use a preselection dimension')
        if not options.cbe_conditional_trim:
            raise ValueError('reduced CBE requires physical-metric pair trimming')
    context_type=ReducedSweepContext
    if options.simplify_fixed_ties:
        if options.gauge_mode!='none' or options.cbe_enabled:
            raise ValueError('simplify_fixed_ties requires gauge_mode="none" and one-site updates without CBE')
        if not isinstance(hamiltonian,ReducedMPOHamiltonian) or hamiltonian.contraction_backend!='reduced':
            raise ValueError('simplify_fixed_ties requires reduced MPO environments')
        from .dependency_environment import DependencySweepContext
        context_type=DependencySweepContext
    nominal_bond = max((len(bond) for bond in state.bond_sectors), default=1)
    state = state.copy()
    direction = str(options.start_direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("start_direction must be 'lr' or 'rl'")
    initial_recovery = None
    if options.gauge_mode == 'frontier':
        from .reduced_gauge import condition_reduced_sweep
        initial_recovery = condition_reduced_sweep(state, hamiltonian, options,
            center=0 if direction == 'lr' else state.nsites-1)
    previous_energy = _energy(state, hamiltonian, stable=True)
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    context = (context_type(state, hamiltonian)
        if isinstance(hamiltonian, ReducedMPOHamiltonian)
        and hamiltonian.contraction_backend == 'reduced'
        and (options.gauge_mode == 'frontier' or options.simplify_fixed_ties) else None)
    end_energy = StableEnergyCache(hamiltonian)
    for sweep in range(1, int(options.max_sweeps) + 1):
        from .reduced_gauge import GaugeQRBoundary
        gauge_boundary=GaugeQRBoundary(state,direction) if options.gauge_mode=='frontier' else None
        sites = (
            range(state.nsites)
            if direction == "lr"
            else range(state.nsites - 1, -1, -1)
        )
        updates = []
        for site in sites:
            if options.cbe_enabled:
                if options.cbe_selector == 'mps-dd':
                    from .reduced_mps_cbe import mps_cbe_site as reduced_cbe_site
                else:
                    from .reduced_cbe import reduced_cbe_site
                updates.append(reduced_cbe_site(state, hamiltonian, site, direction, nominal_bond, options))
                # A new multiplet allocation changes layouts as well as values.
                if context is not None:
                    context = context_type(state, hamiltonian)
            else:
                try:
                    update = optimize_reduced_site(state, hamiltonian, site, options, context=context)
                except NUMERICAL_ERRORS as error:
                    # The local transaction has restored the incumbent. Drop
                    # any partially refreshed environments before continuing.
                    if context is not None:
                        context = context_type(state, hamiltonian)
                    energy = _energy(state, hamiltonian, stable=True)
                    update = LETTASiteUpdate(site=site, local_energy=energy, energy=energy,
                        metric_rank=0, local_dimension=0, residual_norm=np.inf,
                        relative_residual=np.inf, accepted=False, local_converged=False,
                        recovery_reason=f'one-site update failed: {type(error).__name__}: {error}')
                updates.append(update)
            if initial_recovery:
                updates[-1] = replace(updates[-1], recovery_reason=initial_recovery, local_converged=False)
                initial_recovery = None
            if options.gauge_mode == 'frontier':
                cut = site+1 if direction == 'lr' else site
                if 0 < cut < state.nsites:
                    reason = condition_reduced_sweep(state, hamiltonian, options,
                        cut=cut, direction=direction, context=context, qr_boundary=gauge_boundary)
                    if reason:
                        prior = updates[-1].recovery_reason
                        updates[-1] = replace(updates[-1], local_converged=False,
                            recovery_reason=reason if prior is None else prior+'; '+reason)
                        if context is not None:
                            context = context_type(state, hamiltonian)
        updates = tuple(updates)
        energy = context.stable_energy() if context is not None else end_energy(state)
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
        if sweep_callback is not None:
            sweep_callback(state, history[-1])
        if options.verbosity:
            print(
                f"reduced SU(2) LETTA sweep {sweep:3d}  direction={direction}  "
                f"energy={energy:.14f}  dE/site={density_change:.3e}"
            )
        if density_change <= options.tolerance and all(
                u.accepted and u.local_converged and not u.cbe_recovery_reason and not u.recovery_reason
                and all(d.get('optimizer_success', True) for d in u.cbe_compression_diagnostics)
                for u in updates):
            from .reduced_updates import reduced_stationarity
            audit = reduced_stationarity(state, hamiltonian, range(state.nsites), options)
            if max(r['relative_residual'] for r in audit) <= options.eigensolver_tolerance:
                converged = True
                message = "CONVERGENCE: ENERGY STATIONARY AND FRESH LOCAL RESIDUALS <= TOLERANCE"
                break
        previous_energy = energy
        if options.alternate:
            direction = "rl" if direction == "lr" else "lr"
    final_energy = context.stable_energy() if context is not None else end_energy(state)
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

"""Exact reduced-SU(2) adjacent-pair LETTA optimization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr

from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values

from .._letta_one_site_opt.reduced_contraction import (
    CanonicalEnvironmentChain,
    _component_axis_layout,
    expand_reduced_mps_site,
    identity_canonical_factors,
)
from .._letta_one_site_opt.reduced_frontier import (
    ReducedFrontier,
    _BlockVectorLayout,
)
from .._letta_one_site_opt.reduced_operators import ReducedMPOHamiltonian
from .._letta_one_site_opt.reduced_solver import (
    _energy,
    _solve_local_problem,
    _validate_reduced_mpo,
)
from .._letta_one_site_opt.reduced_state import ReducedLatticeLETTA
from .._letta_one_site_opt.reduced_symmetry import _sector_irrep


@dataclass(frozen=True)
class ReducedPairProblem:
    left_site: int
    old_vector: np.ndarray
    layout: _BlockVectorLayout
    frontier: ReducedFrontier
    expanded_dimension: int
    hamiltonian: np.ndarray | None = None
    metric: np.ndarray | None = None
    source_frame: np.ndarray | None = None
    hamiltonian_action: object | None = None
    metric_action: object | None = None

    @property
    def local_dimension(self):
        return self.layout.size

    @property
    def full_local_dimension(self):
        return self.expanded_dimension

    def apply_hamiltonian(self, vector):
        if self.hamiltonian is not None:
            return self.hamiltonian @ np.asarray(vector)
        if self.hamiltonian_action is None:
            raise RuntimeError("pair Hamiltonian action is unavailable")
        return np.asarray(self.hamiltonian_action(vector))

    def apply_metric(self, vector):
        if self.metric is not None:
            return self.metric @ np.asarray(vector)
        if self.metric_action is None:
            raise RuntimeError("pair metric action is unavailable")
        return np.asarray(self.metric_action(vector))


@dataclass(frozen=True)
class ReducedPairSplit:
    left_blocks: dict
    right_blocks: dict
    discarded_weight: float
    sector_ranks: tuple[int, ...]
    retained_multiplicities: tuple[tuple[object, int], ...]


def _merge_pair_blocks(left, right):
    left_data = left if isinstance(left, dict) else left.data
    right_data = right if isinstance(right, dict) else right.data
    merged = {}
    for (q_left, q_phys_left, q_middle), left_block in left_data.items():
        for (middle_2, q_phys_right, q_right), right_block in right_data.items():
            if middle_2 != q_middle:
                continue
            key = (q_left, q_phys_left, q_middle, q_phys_right, q_right)
            merged[key] = np.einsum(
                "apm,mqb->apqb", left_block, right_block, optimize=True
            )
    if not merged:
        raise ValueError("adjacent reduced MPS sites have no compatible bond blocks")
    return merged


def _expand_pair_blocks(blocks, left, right):
    left_layout = _component_axis_layout(left, 0)
    phys_left_layout = _component_axis_layout(left, 1)
    phys_right_layout = _component_axis_layout(right, 1)
    right_layout = _component_axis_layout(right, 2)
    shape = (
        left_layout[2],
        phys_left_layout[2],
        phys_right_layout[2],
        right_layout[2],
    )
    dtype = np.result_type(complex, *[np.asarray(block).dtype for block in blocks.values()])
    dense = np.zeros(shape, dtype=dtype)
    for (q_left, q_p1, q_middle, q_p2, q_right), block in blocks.items():
        block = np.asarray(block)
        irreps = tuple(
            _sector_irrep(sector)
            for sector in (q_left, q_p1, q_middle, q_p2, q_right)
        )
        j_left, j_p1, j_middle, j_p2, j_right = irreps
        offsets = (
            left_layout[0][q_left],
            phys_left_layout[0][q_p1],
            phys_right_layout[0][q_p2],
            right_layout[0][q_right],
        )
        components = tuple(ordered_two_m_values(irrep) for irrep in irreps)
        for a in range(block.shape[0]):
            for p in range(block.shape[1]):
                for q in range(block.shape[2]):
                    for b in range(block.shape[3]):
                        value = block[a, p, q, b]
                        if value == 0:
                            continue
                        for il, ml in enumerate(components[0]):
                            for ip, mp in enumerate(components[1]):
                                for im, mm in enumerate(components[2]):
                                    first = clebsch_gordan(
                                        j_left, j_p1, j_middle, ml, mp, mm
                                    )
                                    if first == 0:
                                        continue
                                    for iq, mq in enumerate(components[3]):
                                        for ir, mr in enumerate(components[4]):
                                            second = clebsch_gordan(
                                                j_middle,
                                                j_p2,
                                                j_right,
                                                mm,
                                                mq,
                                                mr,
                                            )
                                            if second == 0:
                                                continue
                                            dense[
                                                offsets[0] + a * j_left.dim + il,
                                                offsets[1] + p * j_p1.dim + ip,
                                                offsets[2] + q * j_p2.dim + iq,
                                                offsets[3] + b * j_right.dim + ir,
                                            ] += value * first * second
    if np.max(np.abs(dense.imag), initial=0.0) <= 1.0e-14:
        return dense.real
    return dense


def _reduce_expanded_pair_blocks(dense, layout, left, right):
    """Apply the adjoint of :func:`_expand_pair_blocks`."""

    left_layout = _component_axis_layout(left, 0)
    phys_left_layout = _component_axis_layout(left, 1)
    phys_right_layout = _component_axis_layout(right, 1)
    right_layout = _component_axis_layout(right, 2)
    expected = (
        left_layout[2],
        phys_left_layout[2],
        phys_right_layout[2],
        right_layout[2],
    )
    dense = np.asarray(dense)
    if dense.shape != expected:
        raise ValueError(
            f"expanded pair has shape {dense.shape}, expected {expected}"
        )
    blocks = {
        key: np.zeros(shape, dtype=np.result_type(dense, complex))
        for key, shape in layout.shapes.items()
    }
    for (q_left, q_p1, q_middle, q_p2, q_right), block in blocks.items():
        irreps = tuple(
            _sector_irrep(sector)
            for sector in (q_left, q_p1, q_middle, q_p2, q_right)
        )
        j_left, j_p1, j_middle, j_p2, j_right = irreps
        offsets = (
            left_layout[0][q_left],
            phys_left_layout[0][q_p1],
            phys_right_layout[0][q_p2],
            right_layout[0][q_right],
        )
        components = tuple(ordered_two_m_values(irrep) for irrep in irreps)
        for a in range(block.shape[0]):
            for p in range(block.shape[1]):
                for q in range(block.shape[2]):
                    for b in range(block.shape[3]):
                        value = 0.0j
                        for il, ml in enumerate(components[0]):
                            for ip, mp in enumerate(components[1]):
                                for im, mm in enumerate(components[2]):
                                    first = clebsch_gordan(
                                        j_left, j_p1, j_middle, ml, mp, mm
                                    )
                                    if first == 0:
                                        continue
                                    for iq, mq in enumerate(components[3]):
                                        for ir, mr in enumerate(components[4]):
                                            second = clebsch_gordan(
                                                j_middle,
                                                j_p2,
                                                j_right,
                                                mm,
                                                mq,
                                                mr,
                                            )
                                            if second == 0:
                                                continue
                                            value += np.conjugate(first * second) * dense[
                                                offsets[0] + a * j_left.dim + il,
                                                offsets[1] + p * j_p1.dim + ip,
                                                offsets[2] + q * j_p2.dim + iq,
                                                offsets[3] + b * j_right.dim + ir,
                                            ]
                        block[a, p, q, b] = value
    return blocks


def _pair_source_frame(layout, left, right):
    columns = []
    for parameter in range(layout.size):
        vector = np.zeros(layout.size, dtype=complex)
        vector[parameter] = 1.0
        columns.append(
            _expand_pair_blocks(layout.unpack(vector), left, right).reshape(-1)
        )
    return np.column_stack(columns)


def _canonical_pair_action(chain, layout, left, right, left_site, vector):
    expanded = _expand_pair_blocks(layout.unpack(vector), left, right)
    applied = chain.pair_action(left_site, expanded)
    return layout.pack(
        _reduce_expanded_pair_blocks(applied, layout, left, right)
    )


def reduced_pair_problem(
    state,
    hamiltonian,
    left_site,
    *,
    matrix_free=False,
    dense_solver_threshold=64,
):
    """Build an exact polynomial-size reduced two-site generalized problem."""

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("state must be ReducedLatticeLETTA")
    if not isinstance(hamiltonian, ReducedMPOHamiltonian):
        raise TypeError(
            "exact reduced two-site optimization requires ReducedMPOHamiltonian "
            "with canonical local factors"
        )
    _validate_reduced_mpo(hamiltonian, state)
    left_site = int(left_site)
    if not 0 <= left_site < state.nsites - 1:
        raise IndexError("left_site is not an internal pair start")
    frontier = ReducedFrontier.from_state(state)
    sites = tuple(frontier.to_mps(state))
    merged = _merge_pair_blocks(sites[left_site], sites[left_site + 1])
    layout = _BlockVectorLayout({key: block.shape for key, block in merged.items()})
    old_vector = layout.pack(merged)
    hamiltonian_chain = CanonicalEnvironmentChain.build(
        sites, hamiltonian.canonical_factors
    )
    metric_chain = CanonicalEnvironmentChain.build(
        sites, identity_canonical_factors(sites)
    )
    left = sites[left_site]
    right = sites[left_site + 1]
    expanded_shape = (
        _component_axis_layout(left, 0)[2],
        _component_axis_layout(left, 1)[2],
        _component_axis_layout(right, 1)[2],
        _component_axis_layout(right, 2)[2],
    )
    expanded_dimension = int(np.prod(expanded_shape, dtype=int))
    hamiltonian_action = lambda vector: _canonical_pair_action(
        hamiltonian_chain, layout, left, right, left_site, vector
    )
    metric_action = lambda vector: _canonical_pair_action(
        metric_chain, layout, left, right, left_site, vector
    )
    if bool(matrix_free) and layout.size > int(dense_solver_threshold):
        return ReducedPairProblem(
            left_site=left_site,
            old_vector=old_vector,
            layout=layout,
            frontier=frontier,
            expanded_dimension=expanded_dimension,
            hamiltonian_action=hamiltonian_action,
            metric_action=metric_action,
        )

    source_frame = _pair_source_frame(layout, left, right)
    local_h = hamiltonian_chain.pair_local_matrix(left_site, source_frame)
    metric = metric_chain.pair_local_matrix(left_site, source_frame)
    return ReducedPairProblem(
        left_site=left_site,
        old_vector=old_vector,
        layout=layout,
        frontier=frontier,
        expanded_dimension=expanded_dimension,
        hamiltonian=0.5 * (local_h + local_h.conj().T),
        metric=0.5 * (metric + metric.conj().T),
        source_frame=source_frame,
        hamiltonian_action=hamiltonian_action,
        metric_action=metric_action,
    )


def _select_multiplet_ranks(decompositions, bond_dim):
    """Maximize retained weighted norm under a reduced-multiplet budget."""

    try:
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("bond_dim must be an integer") from error
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive")
    items = []
    for q_middle, decomposition in decompositions.items():
        singular_values = decomposition["singular_values"]
        available = decomposition["available"]
        irrep_dimension = _sector_irrep(q_middle).dim
        for position in range(available):
            items.append(
                (
                    q_middle,
                    position,
                    1,
                    irrep_dimension * float(singular_values[position] ** 2),
                )
            )
    states = {0: (0.0, ())}
    for item_index, (_sector, _position, cost, value) in enumerate(items):
        updated = dict(states)
        for used, (retained, selected) in states.items():
            proposed_cost = used + cost
            if proposed_cost > bond_dim:
                continue
            proposal = (retained + value, selected + (item_index,))
            current = updated.get(proposed_cost)
            if current is None or proposal[0] > current[0] + 1.0e-15:
                updated[proposed_cost] = proposal
        states = updated
    feasible = [
        (value, -cost, selected)
        for cost, (value, selected) in states.items()
        if selected
    ]
    if not feasible:
        minimum = min((item[2] for item in items), default=None)
        raise ValueError(
            "bond_dim cannot retain any complete multiplet"
            + ("" if minimum is None else f"; minimum required is {minimum}")
        )
    selected = max(feasible)[2]
    ranks = {sector: 0 for sector in decompositions}
    for item_index in selected:
        sector = items[item_index][0]
        ranks[sector] += 1
    return ranks


def _split_reduced_pair(
    blocks,
    left_template,
    right_template,
    *,
    bond_dim,
    sector_capacities,
    direction,
    cutoff,
):
    direction = str(direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'")
    dtype = np.result_type(*[np.asarray(block).dtype for block in blocks.values()])
    middle_sectors = tuple(sorted({key[2] for key in blocks}))
    decompositions = {}
    for q_middle in middle_sectors:
        sector_keys = tuple(key for key in blocks if key[2] == q_middle)
        row_pairs = tuple(sorted({(key[0], key[1]) for key in sector_keys}))
        col_pairs = tuple(sorted({(key[3], key[4]) for key in sector_keys}))
        row_shapes = {
            pair: next(
                blocks[key].shape[:2]
                for key in sector_keys
                if key[:2] == pair
            )
            for pair in row_pairs
        }
        col_shapes = {
            pair: next(
                blocks[key].shape[2:]
                for key in sector_keys
                if key[3:] == pair
            )
            for pair in col_pairs
        }
        row_offsets = {}
        cursor = 0
        for pair in row_pairs:
            size = int(np.prod(row_shapes[pair], dtype=int))
            row_offsets[pair] = (cursor, cursor + size)
            cursor += size
        col_offsets = {}
        col_cursor = 0
        for pair in col_pairs:
            size = int(np.prod(col_shapes[pair], dtype=int))
            col_offsets[pair] = (col_cursor, col_cursor + size)
            col_cursor += size
        matrix = np.zeros((cursor, col_cursor), dtype=np.result_type(*blocks.values()))
        for key in sector_keys:
            row = row_offsets[key[:2]]
            col = col_offsets[key[3:]]
            matrix[row[0] : row[1], col[0] : col[1]] = blocks[key].reshape(
                row[1] - row[0], col[1] - col[0]
            )
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        capacity = int(sector_capacities.get(q_middle, 0))
        threshold = float(cutoff) * (
            singular_values[0] if singular_values.size else 0.0
        )
        available = int(np.count_nonzero(singular_values > threshold))
        decompositions[q_middle] = {
            "row_pairs": row_pairs,
            "col_pairs": col_pairs,
            "row_shapes": row_shapes,
            "col_shapes": col_shapes,
            "row_offsets": row_offsets,
            "col_offsets": col_offsets,
            "u": u,
            "singular_values": singular_values,
            "vh": vh,
            "available": min(capacity, available),
        }
    retained = _select_multiplet_ranks(decompositions, bond_dim)
    left_blocks = {
        key: np.zeros(np.asarray(block).shape, dtype=dtype)
        for key, block in left_template.data.items()
    }
    right_blocks = {
        key: np.zeros(np.asarray(block).shape, dtype=dtype)
        for key, block in right_template.data.items()
    }
    total = 0.0
    discarded = 0.0
    for q_middle in middle_sectors:
        decomposition = decompositions[q_middle]
        row_pairs = decomposition["row_pairs"]
        col_pairs = decomposition["col_pairs"]
        row_shapes = decomposition["row_shapes"]
        col_shapes = decomposition["col_shapes"]
        row_offsets = decomposition["row_offsets"]
        col_offsets = decomposition["col_offsets"]
        u = decomposition["u"]
        singular_values = decomposition["singular_values"]
        vh = decomposition["vh"]
        keep = retained[q_middle]
        multiplet_weight = float(_sector_irrep(q_middle).dim)
        total += multiplet_weight * float(np.sum(singular_values**2))
        discarded += multiplet_weight * float(np.sum(singular_values[keep:] ** 2))
        if keep == 0:
            continue
        left_factor = u[:, :keep]
        right_factor = vh[:keep]
        if direction == "lr":
            right_factor = singular_values[:keep, None] * right_factor
        else:
            left_factor = left_factor * singular_values[None, :keep]
        for pair in row_pairs:
            start, stop = row_offsets[pair]
            key = (pair[0], pair[1], q_middle)
            left_blocks[key][..., :keep] = left_factor[start:stop].reshape(
                row_shapes[pair] + (keep,)
            )
        for pair in col_pairs:
            start, stop = col_offsets[pair]
            key = (q_middle, pair[0], pair[1])
            right_blocks[key][:keep, ...] = right_factor[:, start:stop].reshape(
                (keep,) + col_shapes[pair]
            )
    return ReducedPairSplit(
        left_blocks=left_blocks,
        right_blocks=right_blocks,
        discarded_weight=discarded / total if total > 0.0 else 0.0,
        sector_ranks=tuple(retained[sector] for sector in middle_sectors),
        retained_multiplicities=tuple(
            (sector, retained[sector]) for sector in middle_sectors
        ),
    )


def _project_frontier_blocks(embedding, blocks):
    target = embedding.pack_target(blocks)
    counts = embedding.adjoint(np.ones(embedding.target_size, dtype=float))
    projected = embedding.adjoint(target)
    nonzero = counts > 0
    projected[nonzero] /= counts[nonzero]
    projected[~nonzero] = 0.0
    return embedding.unpack_source(projected)


class _DenseMetricSquareRoot:
    def __init__(self, metric, tolerance):
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).conj().T)
        values, vectors = np.linalg.eigh(metric)
        scale = float(values[-1]) if values.size else 0.0
        if scale <= 0.0:
            raise ValueError("the reduced pair metric has zero rank")
        cutoff = max(
            float(tolerance), np.finfo(float).eps * metric.shape[0]
        ) * scale
        retained = values > cutoff
        if not np.any(retained):
            raise ValueError("the reduced pair metric has no retained directions")
        self.factor = (
            np.sqrt(values[retained])[:, None] * vectors[:, retained].conj().T
        )

    @property
    def rank(self):
        return self.factor.shape[0]

    def apply(self, vector):
        return self.factor @ np.asarray(vector)

    def adjoint(self, vector):
        return self.factor.conj().T @ np.asarray(vector)


@dataclass(frozen=True)
class _ReducedMetricProjection:
    left_vector: np.ndarray
    right_vector: np.ndarray
    loss: float
    norm_squared: float
    iterations: int


def _expanded_source_blocks(embedding, source_vector):
    return embedding.unpack_target(embedding.apply(source_vector))


def _pair_vector_from_sources(
    layout, left_embedding, right_embedding, left_vector, right_vector
):
    left_blocks = _expanded_source_blocks(left_embedding, left_vector)
    right_blocks = _expanded_source_blocks(right_embedding, right_vector)
    return layout.pack(_merge_pair_blocks(left_blocks, right_blocks))


def _left_source_adjoint(
    layout, gradient_vector, right_blocks, left_embedding
):
    gradients = {
        key: np.zeros(shape, dtype=np.result_type(gradient_vector, complex))
        for key, shape in left_embedding.target_layout.shapes.items()
    }
    for key, gradient in layout.unpack(gradient_vector).items():
        q_left, q_phys_left, q_middle, q_phys_right, q_right = key
        right = right_blocks.get((q_middle, q_phys_right, q_right))
        if right is None:
            continue
        gradients[(q_left, q_phys_left, q_middle)] += np.einsum(
            "apqb,mqb->apm", gradient, np.asarray(right).conj(), optimize=True
        )
    return left_embedding.adjoint(left_embedding.pack_target(gradients))


def _right_source_adjoint(
    layout, left_blocks, gradient_vector, right_embedding
):
    gradients = {
        key: np.zeros(shape, dtype=np.result_type(gradient_vector, complex))
        for key, shape in right_embedding.target_layout.shapes.items()
    }
    for key, gradient in layout.unpack(gradient_vector).items():
        q_left, q_phys_left, q_middle, q_phys_right, q_right = key
        left = left_blocks.get((q_left, q_phys_left, q_middle))
        if left is None:
            continue
        gradients[(q_middle, q_phys_right, q_right)] += np.einsum(
            "apm,apqb->mqb", np.asarray(left).conj(), gradient, optimize=True
        )
    return right_embedding.adjoint(right_embedding.pack_target(gradients))


def _active_source_indices(embedding, retained, side):
    indices = []
    for key in embedding.source_layout.keys:
        shape = embedding.source_layout.shapes[key]
        mask = np.zeros(shape, dtype=bool)
        if side == "left":
            rank = retained.get(key[2], 0)
            mask[..., :rank] = True
        elif side == "right":
            rank = retained.get(key[0], 0)
            mask[:rank, ...] = True
        else:
            raise ValueError("side must be 'left' or 'right'")
        offset = embedding.source_layout.offsets[key][0]
        indices.extend(offset + np.flatnonzero(mask.reshape(-1)))
    result = np.asarray(indices, dtype=int)
    if not result.size:
        raise ValueError(f"retained multiplets leave no active {side} parameters")
    return result


def _embedded_active(vector, indices, size, dtype):
    result = np.zeros(size, dtype=np.result_type(dtype, vector))
    result[indices] = vector
    return result


def _metric_inner(problem, left, right):
    return np.vdot(np.asarray(left), problem.apply_metric(right))


def _pair_metric_loss(target, candidate, problem):
    difference = np.asarray(target) - np.asarray(candidate)
    return float(max(0.0, np.real(_metric_inner(problem, difference, difference))))


def _optimize_left_source(
    target,
    problem,
    left_embedding,
    right_embedding,
    left_vector,
    right_vector,
    square_root,
    active_indices,
    tolerance,
):
    right_blocks = _expanded_source_blocks(right_embedding, right_vector)

    def pair_forward(active):
        full = _embedded_active(
            active,
            active_indices,
            left_embedding.source_size,
            left_vector.dtype,
        )
        merged = problem.layout.pack(
            _merge_pair_blocks(
                _expanded_source_blocks(left_embedding, full), right_blocks
            )
        )
        return merged

    def pair_adjoint(gradient):
        return _left_source_adjoint(
            problem.layout, gradient, right_blocks, left_embedding
        )

    dtype = np.result_type(target, left_vector, right_vector, complex)
    initial = np.asarray(left_vector[active_indices], dtype=dtype)
    max_iterations = max(50, min(1000, 5 * active_indices.size))
    if square_root is not None:
        operator = LinearOperator(
            (square_root.rank, active_indices.size),
            matvec=lambda active: square_root.apply(pair_forward(active)),
            rmatvec=lambda weighted: pair_adjoint(
                square_root.adjoint(weighted)
            )[active_indices],
            dtype=dtype,
        )
        solution = lsmr(
            operator,
            square_root.apply(target),
            atol=tolerance,
            btol=tolerance,
            maxiter=max_iterations,
            x0=initial,
        )[0]
    else:
        normal = LinearOperator(
            (active_indices.size, active_indices.size),
            matvec=lambda active: pair_adjoint(
                problem.apply_metric(pair_forward(active))
            )[active_indices],
            dtype=dtype,
        )
        rhs = pair_adjoint(problem.apply_metric(target))[active_indices]
        solution, info = cg(
            normal,
            rhs,
            x0=initial,
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
        )
        if info < 0:
            raise RuntimeError("matrix-free left metric projection failed")
    return _embedded_active(
        solution,
        active_indices,
        left_embedding.source_size,
        left_vector.dtype,
    )


def _optimize_right_source(
    target,
    problem,
    left_embedding,
    right_embedding,
    left_vector,
    right_vector,
    square_root,
    active_indices,
    tolerance,
):
    left_blocks = _expanded_source_blocks(left_embedding, left_vector)

    def pair_forward(active):
        full = _embedded_active(
            active,
            active_indices,
            right_embedding.source_size,
            right_vector.dtype,
        )
        merged = problem.layout.pack(
            _merge_pair_blocks(
                left_blocks, _expanded_source_blocks(right_embedding, full)
            )
        )
        return merged

    def pair_adjoint(gradient):
        return _right_source_adjoint(
            problem.layout, left_blocks, gradient, right_embedding
        )

    dtype = np.result_type(target, left_vector, right_vector, complex)
    initial = np.asarray(right_vector[active_indices], dtype=dtype)
    max_iterations = max(50, min(1000, 5 * active_indices.size))
    if square_root is not None:
        operator = LinearOperator(
            (square_root.rank, active_indices.size),
            matvec=lambda active: square_root.apply(pair_forward(active)),
            rmatvec=lambda weighted: pair_adjoint(
                square_root.adjoint(weighted)
            )[active_indices],
            dtype=dtype,
        )
        solution = lsmr(
            operator,
            square_root.apply(target),
            atol=tolerance,
            btol=tolerance,
            maxiter=max_iterations,
            x0=initial,
        )[0]
    else:
        normal = LinearOperator(
            (active_indices.size, active_indices.size),
            matvec=lambda active: pair_adjoint(
                problem.apply_metric(pair_forward(active))
            )[active_indices],
            dtype=dtype,
        )
        rhs = pair_adjoint(problem.apply_metric(target))[active_indices]
        solution, info = cg(
            normal,
            rhs,
            x0=initial,
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
        )
        if info < 0:
            raise RuntimeError("matrix-free right metric projection failed")
    return _embedded_active(
        solution,
        active_indices,
        right_embedding.source_size,
        right_vector.dtype,
    )


def _balance_source_pair(left_vector, right_vector):
    left_norm = np.linalg.norm(left_vector)
    right_norm = np.linalg.norm(right_vector)
    if left_norm <= np.finfo(float).tiny or right_norm <= np.finfo(float).tiny:
        return left_vector, right_vector
    scale = np.sqrt(right_norm / left_norm)
    return left_vector * scale, right_vector / scale


def _metric_project_sources(
    target,
    problem,
    left_embedding,
    right_embedding,
    left_vector,
    right_vector,
    left_indices,
    right_indices,
    *,
    tolerance,
    max_iterations,
    metric_tolerance,
):
    square_root = (
        _DenseMetricSquareRoot(problem.metric, metric_tolerance)
        if problem.metric is not None
        else None
    )
    candidate = _pair_vector_from_sources(
        problem.layout,
        left_embedding,
        right_embedding,
        left_vector,
        right_vector,
    )
    loss = _pair_metric_loss(target, candidate, problem)
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        previous = loss
        proposed_left = _optimize_left_source(
            target,
            problem,
            left_embedding,
            right_embedding,
            left_vector,
            right_vector,
            square_root,
            left_indices,
            tolerance,
        )
        proposed = _pair_vector_from_sources(
            problem.layout,
            left_embedding,
            right_embedding,
            proposed_left,
            right_vector,
        )
        proposed_loss = _pair_metric_loss(target, proposed, problem)
        if proposed_loss <= loss + 10.0 * np.finfo(float).eps:
            left_vector = proposed_left
            candidate = proposed
            loss = proposed_loss

        proposed_right = _optimize_right_source(
            target,
            problem,
            left_embedding,
            right_embedding,
            left_vector,
            right_vector,
            square_root,
            right_indices,
            tolerance,
        )
        proposed = _pair_vector_from_sources(
            problem.layout,
            left_embedding,
            right_embedding,
            left_vector,
            proposed_right,
        )
        proposed_loss = _pair_metric_loss(target, proposed, problem)
        if proposed_loss <= loss + 10.0 * np.finfo(float).eps:
            right_vector = proposed_right
            candidate = proposed
            loss = proposed_loss
        left_vector, right_vector = _balance_source_pair(
            left_vector, right_vector
        )
        iterations = iteration
        if previous - loss <= tolerance * max(1.0, previous):
            break
    candidate = _pair_vector_from_sources(
        problem.layout,
        left_embedding,
        right_embedding,
        left_vector,
        right_vector,
    )
    norm_squared = float(np.real(_metric_inner(problem, candidate, candidate)))
    return _ReducedMetricProjection(
        left_vector=left_vector,
        right_vector=right_vector,
        loss=_pair_metric_loss(target, candidate, problem),
        norm_squared=norm_squared,
        iterations=iterations,
    )


def _masked_source(vector, indices):
    result = np.zeros_like(np.asarray(vector))
    result[indices] = np.asarray(vector)[indices]
    return result


def _choose_projection_start(
    target,
    problem,
    left_embedding,
    right_embedding,
    candidates,
):
    choices = []
    target_norm = float(np.real(_metric_inner(problem, target, target)))
    threshold = np.finfo(float).eps * max(1.0, target_norm)
    for left_vector, right_vector in candidates:
        merged = _pair_vector_from_sources(
            problem.layout,
            left_embedding,
            right_embedding,
            left_vector,
            right_vector,
        )
        norm_squared = float(np.real(_metric_inner(problem, merged, merged)))
        if not np.isfinite(norm_squared) or norm_squared <= threshold:
            continue
        choices.append(
            (
                _pair_metric_loss(target, merged, problem),
                left_vector,
                right_vector,
            )
        )
    if not choices:
        raise ValueError("no nonzero LETTA-compatible initialization for pair projection")
    _loss, left_vector, right_vector = min(choices, key=lambda item: item[0])
    return left_vector, right_vector


def _shrink_source_blocks(blocks, retained, side):
    result = {}
    for key, block in blocks.items():
        if side == "left":
            rank = retained.get(key[2], 0)
            if rank:
                result[key] = np.asarray(block)[..., :rank].copy()
        elif side == "right":
            rank = retained.get(key[0], 0)
            if rank:
                result[key] = np.asarray(block)[:rank, ...].copy()
        else:
            raise ValueError("side must be 'left' or 'right'")
    if not result:
        raise ValueError(f"multiplet truncation removed every {side} fusion block")
    return result


def _retained_bond_sectors(old_bond, retained):
    used = {sector: 0 for sector in retained}
    new_bond = []
    for sector in old_bond:
        if used.get(sector, 0) < retained.get(sector, 0):
            new_bond.append(sector)
            used[sector] = used.get(sector, 0) + 1
    if any(used.get(sector, 0) != rank for sector, rank in retained.items()):
        raise RuntimeError("retained multiplets are inconsistent with bond allocation")
    if not new_bond:
        raise ValueError("multiplet truncation removed the entire virtual bond")
    return tuple(new_bond)


def _optimize_reduced_pair(
    state, hamiltonian, left_site, direction, bond_dim, options
):
    from .solver import LETTAPairUpdate

    problem = reduced_pair_problem(
        state,
        hamiltonian,
        left_site,
        matrix_free=options.matrix_free,
        dense_solver_threshold=options.dense_solver_threshold,
    )
    local_energy, vector, metric_rank, residual = _solve_local_problem(
        problem, options, initial_vector=problem.old_vector
    )
    optimized = problem.layout.unpack(vector)
    sites = tuple(problem.frontier.to_mps(state))
    split = _split_reduced_pair(
        optimized,
        sites[left_site],
        sites[left_site + 1],
        bond_dim=bond_dim,
        sector_capacities=Counter(state.bond_sectors[left_site]),
        direction=direction,
        cutoff=options.conditional_svd_cutoff,
    )
    left_embedding = problem.frontier.site_embedding(state, left_site)
    right_embedding = problem.frontier.site_embedding(state, left_site + 1)
    retained = dict(split.retained_multiplicities)
    left_indices = _active_source_indices(left_embedding, retained, "left")
    right_indices = _active_source_indices(right_embedding, retained, "right")
    old_left_vector = left_embedding.pack_source(state.tensors[left_site])
    old_right_vector = right_embedding.pack_source(state.tensors[left_site + 1])
    projected_left = left_embedding.pack_source(
        _project_frontier_blocks(left_embedding, split.left_blocks)
    )
    projected_right = right_embedding.pack_source(
        _project_frontier_blocks(right_embedding, split.right_blocks)
    )
    initial_left, initial_right = _choose_projection_start(
        vector,
        problem,
        left_embedding,
        right_embedding,
        (
            (
                _masked_source(old_left_vector, left_indices),
                _masked_source(old_right_vector, right_indices),
            ),
            (
                _masked_source(projected_left, left_indices),
                _masked_source(projected_right, right_indices),
            ),
        ),
    )
    refinement = _metric_project_sources(
        vector,
        problem,
        left_embedding,
        right_embedding,
        initial_left,
        initial_right,
        left_indices,
        right_indices,
        tolerance=options.truncation_tolerance,
        max_iterations=options.truncation_max_iterations,
        metric_tolerance=options.metric_tolerance,
    )
    norm_threshold = np.finfo(float).eps * max(
        1.0, float(np.real(_metric_inner(problem, vector, vector)))
    )
    if (
        not np.isfinite(refinement.norm_squared)
        or refinement.norm_squared <= norm_threshold
    ):
        raise ValueError("reduced pair projection produced a zero or non-finite state")
    normalized_left = refinement.left_vector / np.sqrt(refinement.norm_squared)
    normalized_right = refinement.right_vector
    normalized_pair = _pair_vector_from_sources(
        problem.layout,
        left_embedding,
        right_embedding,
        normalized_left,
        normalized_right,
    )
    projection_loss = _pair_metric_loss(
        vector, normalized_pair, problem
    )
    left_source = _shrink_source_blocks(
        left_embedding.unpack_source(normalized_left), retained, "left"
    )
    right_source = _shrink_source_blocks(
        right_embedding.unpack_source(normalized_right), retained, "right"
    )
    new_bond = _retained_bond_sectors(
        state.bond_sectors[left_site], retained
    )

    old_energy = _energy(state, hamiltonian, stable=True)
    old_left = state.tensors[left_site]
    old_right = state.tensors[left_site + 1]
    old_bonds = state.bond_sectors
    state.tensors[left_site] = left_source
    state.tensors[left_site + 1] = right_source
    bonds = list(state.bond_sectors)
    bonds[left_site] = new_bond
    state.bond_sectors = tuple(bonds)
    state.tensors = state._validate_tensors(state.tensors)
    new_energy = _energy(state, hamiltonian, stable=True)
    accepted = new_energy <= old_energy + options.energy_increase_tolerance
    if not accepted:
        state.tensors[left_site] = old_left
        state.tensors[left_site + 1] = old_right
        state.bond_sectors = old_bonds
        new_energy = old_energy
    else:
        state.balance_scalar_gauge()
    return LETTAPairUpdate(
        left_site=left_site,
        right_site=left_site + 1,
        shared_physical_sites=tuple(
            sorted(
                set(state.site_neighborhood(left_site))
                & set(state.site_neighborhood(left_site + 1))
            )
        ),
        old_energy=old_energy,
        local_energy=local_energy,
        energy=new_energy,
        metric_rank=metric_rank,
        local_dimension=problem.local_dimension,
        residual_norm=residual,
        conditional_discarded_weight=split.discarded_weight,
        metric_truncation_loss=projection_loss,
        truncation_iterations=refinement.iterations,
        energy_refinement_initial_energy=None,
        energy_refinement_energy=None,
        energy_refinement_iterations=0,
        energy_refinement_accepted_substeps=0,
        max_factor_norm=max(
            max(np.linalg.norm(block) for block in left_source.values()),
            max(np.linalg.norm(block) for block in right_source.values()),
        ),
        sector_ranks=split.sector_ranks,
        accepted=accepted,
        full_local_dimension=problem.full_local_dimension,
    )


def reduced_two_site_dmrg(hamiltonian, *, state, bond_dim, options):
    from .solver import LETTATwoSiteResult, LETTATwoSiteSweep

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("state must be ReducedLatticeLETTA")
    if not isinstance(hamiltonian, ReducedMPOHamiltonian):
        raise TypeError("hamiltonian must be ReducedMPOHamiltonian")
    _validate_reduced_mpo(hamiltonian, state)
    if options.split_method != "conditional-svd":
        raise NotImplementedError(
            "exact reduced SU(2) two-site optimization currently supports only "
            "split_method='conditional-svd'; metric-ALS and energy refinement "
            "have not yet been formulated in irrep-multiplicity space"
        )
    if state.nsites < 2:
        raise ValueError("two-site optimization requires at least two sites")
    try:
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("bond_dim must be an integer") from error
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive")
    state = state.copy()
    direction = str(options.start_direction).lower()
    if direction not in {"lr", "rl"}:
        raise ValueError("start_direction must be 'lr' or 'rl'")
    previous_energy = _energy(state, hamiltonian, stable=True)
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    for sweep in range(1, int(options.max_sweeps) + 1):
        pair_sites = (
            range(state.nsites - 1)
            if direction == "lr"
            else range(state.nsites - 2, -1, -1)
        )
        updates = tuple(
            _optimize_reduced_pair(
                state, hamiltonian, site, direction, bond_dim, options
            )
            for site in pair_sites
        )
        energy = _energy(state, hamiltonian, stable=True)
        change = abs(energy - previous_energy)
        density_change = change / state.nsites
        history.append(
            LETTATwoSiteSweep(
                sweep=sweep,
                direction=direction,
                energy=energy,
                energy_change=change,
                energy_density_change=density_change,
                bond_dimension=bond_dim,
                updates=updates,
            )
        )
        if options.verbosity:
            print(
                f"reduced SU(2) two-site sweep {sweep:3d} "
                f"direction={direction} energy={energy:.14f} "
                f"dE/site={density_change:.3e}"
            )
        if density_change <= options.tolerance:
            converged = True
            message = "CONVERGENCE: SWEEP ENERGY DENSITY CHANGE <= TOLERANCE"
            break
        previous_energy = energy
        if options.alternate:
            direction = "rl" if direction == "lr" else "lr"

    two_site_energy = float(history[-1].energy)
    polish_sweeps = 0
    if options.one_site_polish_sweeps:
        from .._letta_one_site_opt import LETTADMROptions, letta_dmrg

        polished = letta_dmrg(
            hamiltonian,
            state=state,
            options=LETTADMROptions(
                max_sweeps=options.one_site_polish_sweeps,
                tolerance=options.tolerance,
                metric_tolerance=options.metric_tolerance,
                energy_increase_tolerance=options.energy_increase_tolerance,
                eigensolver_tolerance=options.eigensolver_tolerance,
                eigensolver_max_iterations=options.eigensolver_max_iterations,
                dense_solver_threshold=options.dense_solver_threshold,
                matrix_free=options.matrix_free,
                gauge_mode=options.gauge_mode,
            ),
        )
        state = polished.state
        polish_sweeps = polished.sweeps
    final_energy = _energy(state, hamiltonian, stable=True)
    return LETTATwoSiteResult(
        state=state,
        energy=final_energy,
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
        two_site_energy=two_site_energy,
        polish_sweeps=polish_sweeps,
    )


__all__ = [
    "ReducedPairProblem",
    "ReducedPairSplit",
    "reduced_pair_problem",
    "reduced_two_site_dmrg",
]

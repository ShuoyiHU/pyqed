"""Exact reduced-SU(2) adjacent-pair LETTA optimization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from operator import index

import numpy as np

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
from .._letta_one_site_opt.reduced_environment import ReducedEnvironmentChain
from .._letta_one_site_opt.reduced_norm import ReducedNormChain
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
    metric_scale: float | None = None
    metric_projector_factory: object | None = None

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
    left = sites[left_site]
    right = sites[left_site + 1]
    expanded_shape = (
        _component_axis_layout(left, 0)[2],
        _component_axis_layout(left, 1)[2],
        _component_axis_layout(right, 1)[2],
        _component_axis_layout(right, 2)[2],
    )
    expanded_dimension = int(np.prod(expanded_shape, dtype=int))
    if hamiltonian.contraction_backend == 'reduced':
        h_chain = ReducedEnvironmentChain.build(sites, hamiltonian.native_mpo(state.physical_basis))
        n_chain = ReducedNormChain.build(sites)
        h_action = lambda v: layout.pack(h_chain.pair_action(left_site, layout.unpack(v)))
        n_action = lambda v: layout.pack(n_chain.pair_action(left_site, layout.unpack(v)))
        h, n = None, None
        if not matrix_free or layout.size <= dense_solver_threshold:
            identity = np.eye(layout.size)
            h = np.column_stack([h_action(v) for v in identity])
            n = np.column_stack([n_action(v) for v in identity])
            h, n = .5*(h+h.conj().T), .5*(n+n.conj().T)
        return ReducedPairProblem(left_site=left_site, old_vector=old_vector,
            layout=layout, frontier=frontier, expanded_dimension=expanded_dimension,
            hamiltonian=h, metric=n, hamiltonian_action=h_action, metric_action=n_action,
            metric_scale=n_chain.metric_scale(left_site, layout.keys, width=2),
            metric_projector_factory=lambda tol: n_chain.pair_projector(left_site, layout, tol))
    hamiltonian_chain = CanonicalEnvironmentChain.build(sites, hamiltonian.canonical_factors)
    metric_chain = CanonicalEnvironmentChain.build(sites, identity_canonical_factors(sites))
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


@dataclass(frozen=True)
class _ReducedMetricProjection:
    left_vector: np.ndarray
    right_vector: np.ndarray
    loss: float
    norm_squared: float
    iterations: int
    diagnostics: dict


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


def _metric_inner(problem, left, right):
    return np.vdot(np.asarray(left), problem.apply_metric(right))


def _pair_metric_loss(target, candidate, problem):
    difference = np.asarray(target) - np.asarray(candidate)
    return float(max(0.0, np.real(_metric_inner(problem, difference, difference))))


def _masked_source(vector, indices):
    result = np.zeros_like(np.asarray(vector))
    result[indices] = np.asarray(vector)[indices]
    return result


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
    from .._letta_one_site_opt.reduced_updates import NUMERICAL_ERRORS, one_site_options
    from .._letta_one_site_opt.reduced_solver import optimize_reduced_site
    from .solver import LETTAPairUpdate
    before = _energy(state, hamiltonian, stable=True)
    try:
        candidate = (_expand_reduced_pair_space(state, left_site, bond_dim)
                     if options.reduced_sector_growth else state.copy())
        update = _optimize_allocated_reduced_pair(
            candidate, hamiltonian, left_site, direction, bond_dim, options)
        actual = _energy(candidate, hamiltonian, stable=True)
        valid = update.accepted and actual <= before+options.energy_increase_tolerance
        # Compression must not cost the progress of an ordinary one-site step
        # from the same incumbent. Only compare feasible bond allocations.
        baseline_error = None
        if len(state.bond_sectors[left_site]) <= bond_dim:
            baseline = state.copy()
            site = left_site if direction == 'lr' else left_site+1
            try:
                ordinary = optimize_reduced_site(baseline, hamiltonian, site, one_site_options(options))
                baseline_energy = _energy(baseline, hamiltonian, stable=True)
                update = replace(update, baseline_energy=baseline_energy)
                if ordinary.accepted and (not valid or baseline_energy <= actual):
                    state.tensors, state.bond_sectors = baseline.tensors, baseline.bond_sectors
                    return replace(update, energy=baseline_energy, accepted=True, baseline_selected=True)
            except NUMERICAL_ERRORS as error:
                baseline_error = f'one-site baseline failed: {type(error).__name__}: {error}'
        if valid:
            state.tensors, state.bond_sectors = candidate.tensors, candidate.bond_sectors
            return replace(update, energy=actual, recovery_reason=baseline_error)
        return replace(update, energy=before, accepted=False, recovery_reason=baseline_error)
    except NUMERICAL_ERRORS as error:
        # The failed candidate owns its allocations and arrays. The incumbent
        # is still intact when the ordinary one-site fallback starts.
        reason = f'{type(error).__name__}: {error}'
        fallback = state.copy()
        site = left_site if direction == 'lr' else left_site+1
        try:
            ordinary = optimize_reduced_site(fallback, hamiltonian, site, one_site_options(options))
            energy = _energy(fallback, hamiltonian, stable=True)
            accepted = ordinary.accepted and np.isfinite(energy) and energy <= before+options.energy_increase_tolerance
            if accepted:
                state.tensors, state.bond_sectors = fallback.tensors, fallback.bond_sectors
        except NUMERICAL_ERRORS as second:
            ordinary, accepted = None, False
            reason += f'; one-site fallback failed: {type(second).__name__}: {second}'
        return LETTAPairUpdate(left_site=left_site, right_site=left_site+1,
            shared_physical_sites=tuple(sorted(set(state.site_neighborhood(left_site)) &
                set(state.site_neighborhood(left_site+1)))), old_energy=before,
            energy=_energy(state, hamiltonian, stable=True),
            local_energy=before if ordinary is None else ordinary.local_energy,
            metric_rank=0 if ordinary is None else ordinary.metric_rank,
            local_dimension=0 if ordinary is None else ordinary.local_dimension,
            residual_norm=np.inf if ordinary is None else ordinary.residual_norm,
            conditional_discarded_weight=0., metric_truncation_loss=0., truncation_iterations=0,
            energy_refinement_initial_energy=None, energy_refinement_energy=None,
            energy_refinement_iterations=0, energy_refinement_accepted_substeps=0,
            max_factor_norm=max(np.linalg.norm(a) for tensor in state.tensors for a in tensor.values()),
            sector_ranks=tuple(Counter(state.bond_sectors[left_site]).values()),
            accepted=accepted, fallback=True, recovery_reason=reason)


def _expand_reduced_pair_space(state, left_site, bond_dim):
    """Enlarge a bond with locally reachable whole multiplets, preserving Psi.

    New left columns are zero, while matching right rows are seeded so ALS
    factorization can activate them without a bilinear zero-start trap. Temporary
    per-sector capacities can exceed the final total multiplet budget; the
    existing metric-aware pair split enforces that budget on acceptance.
    """
    candidate = state.copy()
    left = Counter(state.left_virtual_sectors(left_site))
    right = Counter(state.right_virtual_sectors(left_site+1))
    physical = dict(zip(state.physical_basis.sectors, state.physical_basis.multiplicities))
    left_capacity, right_capacity = Counter(), Counter()
    for ql, dl in left.items():
        for qp, dp in physical.items():
            for qm in state.symmetry.fuse(ql, qp):
                left_capacity[qm] += dl*dp
    for qm in left_capacity:
        for qp, dp in physical.items():
            for qr in state.symmetry.fuse(qm, qp):
                if qr in right:
                    right_capacity[qm] += dp*right[qr]
    old = Counter(state.bond_sectors[left_site])
    memory_left = state.physical_dim**(len(state.site_neighborhood(left_site))-1)
    memory_right = state.physical_dim**(len(state.site_neighborhood(left_site+1))-1)
    capacities = dict(old)
    for qm in left_capacity.keys() & right_capacity.keys():
        capacities[qm] = max(old[qm], min(bond_dim,
            left_capacity[qm]*memory_left, right_capacity[qm]*memory_right))
    bonds = list(state.bond_sectors)
    bonds[left_site] = tuple(q for q in sorted(capacities) for _ in range(capacities[q]))
    candidate.bond_sectors = tuple(bonds)
    dtype = np.result_type(*[a.dtype for i in (left_site, left_site+1)
                            for a in state.tensors[i].values()])
    rng = np.random.default_rng(1701+left_site)
    for i in (left_site, left_site+1):
        dl = Counter(candidate.left_virtual_sectors(i))
        dr = Counter(candidate.right_virtual_sectors(i))
        dependencies = (state.physical_dim,)*(len(state.site_neighborhood(i))-1)
        blocks = {}
        for ql in dl:
            for qp in physical:
                for qr in state.symmetry.fuse(ql, qp):
                    if qr not in dr:
                        continue
                    key = (ql, qp, qr)
                    block = np.zeros((dl[ql], physical[qp])+dependencies+(dr[qr],), dtype=dtype)
                    if key in state.tensors[i]:
                        original = state.tensors[i][key]
                        block[tuple(slice(0, n) for n in original.shape)] = original
                    if i == left_site+1 and dl[ql] > old[ql]:
                        section = block[old[ql]:]
                        section[...] = rng.normal(size=section.shape)/np.sqrt(max(1, section.size))
                    blocks[key] = block
        candidate.tensors[i] = blocks
    candidate.tensors = candidate._validate_tensors(candidate.tensors)
    return candidate


def _optimize_allocated_reduced_pair(
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
    if all(len(state.site_neighborhood(i)) == 1 for i in range(state.nsites)):
        return _optimize_untied_split(state, hamiltonian, problem, vector,
            local_energy, metric_rank, residual, direction, bond_dim, options)
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
    starts = (
            (
                _masked_source(old_left_vector, left_indices),
                _masked_source(old_right_vector, right_indices),
            ),
            (
                _masked_source(projected_left, left_indices),
                _masked_source(projected_right, right_indices),
            ),
    )
    norm_threshold = np.finfo(float).eps * max(
        1.0, float(np.real(_metric_inner(problem, vector, vector)))
    )
    # Projection can erase complementary factors and create an ALS stationary
    # point, even when its initial loss beats the incumbent. Compare *refined*
    # starts so the seeded growth directions get a chance to enter the state.
    refinements = []
    for initial_left, initial_right in starts:
        merged = _pair_vector_from_sources(problem.layout, left_embedding,
            right_embedding, initial_left, initial_right)
        if float(np.real(_metric_inner(problem, merged, merged))) <= norm_threshold:
            continue
        from .reduced_compression import compress_reduced_pair
        fit = compress_reduced_pair(vector, problem, state, initial_left, initial_right,
            retained, options=options.compression,
            als_max_iterations=options.truncation_max_iterations,
            metric_tolerance=options.metric_tolerance)
        merged = _pair_vector_from_sources(problem.layout, left_embedding,
            right_embedding, fit.left, fit.right)
        norm_squared = float(np.real(_metric_inner(problem, merged, merged)))
        if np.isfinite(norm_squared) and norm_squared > norm_threshold:
            refinements.append(_ReducedMetricProjection(fit.left, fit.right,
                fit.loss, norm_squared, fit.iterations, fit.diagnostics))
    if not refinements:
        raise ValueError("reduced pair projection produced a zero or non-finite state")
    refinement = min(refinements, key=lambda trial: trial.loss)
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
    energy_fit = _refine_pair_if_requested(state, hamiltonian, left_site, direction, options)
    if energy_fit is not None:
        state.tensors, state.bond_sectors = energy_fit.state.tensors, energy_fit.state.bond_sectors
        new_energy = energy_fit.energy
    accepted = np.isfinite(new_energy) and new_energy <= old_energy + options.energy_increase_tolerance
    if not accepted:
        state.tensors[left_site] = old_left
        state.tensors[left_site + 1] = old_right
        state.bond_sectors = old_bonds
        new_energy = old_energy
    else:
        state.normalize(center=left_site, balance=options.gauge_mode != 'frontier')
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
        compression_diagnostics=refinement.diagnostics,
        energy_refinement_initial_energy=None if energy_fit is None else energy_fit.initial_energy,
        energy_refinement_energy=None if energy_fit is None else energy_fit.energy,
        energy_refinement_iterations=0 if energy_fit is None else energy_fit.iterations,
        energy_refinement_accepted_substeps=0 if energy_fit is None else energy_fit.accepted_substeps,
        energy_refinement_diagnostics=None if energy_fit is None else energy_fit.diagnostics,
        max_factor_norm=max(
            max(np.linalg.norm(block) for block in state.tensors[left_site].values()),
            max(np.linalg.norm(block) for block in state.tensors[left_site+1].values()),
        ),
        sector_ranks=split.sector_ranks,
        accepted=accepted,
        full_local_dimension=problem.full_local_dimension,
    )


def _refine_pair_if_requested(state, hamiltonian, site, direction, options):
    if options.split_method not in {'metric-als-energy', 'metric-energy', 'energy-refined'}:
        return None
    from .._letta_one_site_opt.reduced_updates import refine_reduced_pair_energy
    return refine_reduced_pair_energy(state, hamiltonian, site, options,
        max_iterations=options.energy_refinement_max_iterations,
        tolerance=options.energy_refinement_tolerance, direction=direction)


def _gram_roots(gram, tolerance):
    values, vectors = np.linalg.eigh(.5*(gram+gram.conj().T))
    scale = float(np.max(values, initial=0.))
    if np.min(values, initial=0.) < -10*tolerance*max(scale, np.finfo(float).tiny):
        raise FloatingPointError('MPS boundary Gram is not positive semidefinite')
    keep = values > tolerance*scale
    u, s = vectors[:, keep], np.sqrt(values[keep])
    return (u*s)@u.conj().T, (u/s)@u.conj().T


def _schmidt_split_untied(problem, blocks, sites, state, bond_dim, direction, options):
    """Optimal whole-multiplet Schmidt truncation in the physical pair norm.

    For middle spin J, assemble sqrt(d_right/d_J) G_L^(1/2) Theta
    (G_R^(1/2))^T. Its singular values carry full-multiplet weight d_J*s^2.
    Undo the boundary roots and spin weight after the reduced SVD. No magnetic
    variational tensor or determinant-space projection enters this operation.
    """
    i = problem.left_site
    chain = ReducedNormChain.build(sites)
    left = {q: _gram_roots(g, options.metric_tolerance) for q, g in chain.left[i].items()}
    right = {q: _gram_roots(g, options.metric_tolerance) for q, g in chain.right[i+2].items()}
    weighted = {key: np.sqrt(_sector_irrep(key[-1]).dim/_sector_irrep(key[2]).dim)
        *np.einsum('al,lpqr,br->apqb', left[key[0]][0], a, right[key[-1]][0], optimize=True)
        for key, a in blocks.items()}
    split = _split_reduced_pair(weighted, sites[i], sites[i+1], bond_dim=bond_dim,
        sector_capacities=Counter(state.bond_sectors[i]), direction=direction,
        cutoff=options.conditional_svd_cutoff)
    a = {key: np.einsum('al,lpm->apm', left[key[0]][1], value, optimize=True)
         for key, value in split.left_blocks.items()}
    b = {key: np.sqrt(_sector_irrep(key[0]).dim/_sector_irrep(key[-1]).dim)
         *np.einsum('mpb,rb->mpr', value, right[key[-1]][1], optimize=True)
         for key, value in split.right_blocks.items()}
    return ReducedPairSplit(a, b, split.discarded_weight, split.sector_ranks,
                            split.retained_multiplicities)


def _optimize_untied_split(state, hamiltonian, problem, vector, local_energy,
                          metric_rank, residual, direction, bond_dim, options):
    from .solver import LETTAPairUpdate
    i = problem.left_site
    sites = tuple(problem.frontier.to_mps(state))
    split = _schmidt_split_untied(problem, problem.layout.unpack(vector), sites,
                                state, bond_dim, direction, options)
    pair = problem.layout.pack(_merge_pair_blocks(split.left_blocks, split.right_blocks))
    norm = float(np.real(_metric_inner(problem, pair, pair)))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError('Schmidt truncation produced a null state')
    a = {key: value/np.sqrt(norm) for key, value in split.left_blocks.items()}
    retained = dict(split.retained_multiplicities)
    a = _shrink_source_blocks(a, retained, 'left')
    b = _shrink_source_blocks(split.right_blocks, retained, 'right')
    old_energy = _energy(state, hamiltonian, stable=True)
    old_a, old_b, old_bonds = state.tensors[i], state.tensors[i+1], state.bond_sectors
    bonds = list(old_bonds)
    bonds[i] = _retained_bond_sectors(old_bonds[i], retained)
    state.bond_sectors = tuple(bonds)
    state.tensors[i], state.tensors[i+1] = a, b
    state.tensors = state._validate_tensors(state.tensors)
    energy = _energy(state, hamiltonian, stable=True)
    energy_fit = _refine_pair_if_requested(state, hamiltonian, i, direction, options)
    if energy_fit is not None:
        state.tensors, state.bond_sectors = energy_fit.state.tensors, energy_fit.state.bond_sectors
        energy = energy_fit.energy
    accepted = np.isfinite(energy) and energy <= old_energy+options.energy_increase_tolerance
    if not accepted:
        state.tensors[i], state.tensors[i+1], state.bond_sectors = old_a, old_b, old_bonds
        energy = old_energy
    else:
        state.normalize(center=i, balance=options.gauge_mode != 'frontier')
    return LETTAPairUpdate(left_site=i, right_site=i+1, shared_physical_sites=(),
        old_energy=old_energy, local_energy=local_energy, energy=energy,
        metric_rank=metric_rank, local_dimension=problem.local_dimension,
        residual_norm=residual, conditional_discarded_weight=split.discarded_weight,
        metric_truncation_loss=_pair_metric_loss(vector, pair/np.sqrt(norm), problem),
        truncation_iterations=0,
        energy_refinement_initial_energy=None if energy_fit is None else energy_fit.initial_energy,
        energy_refinement_energy=None if energy_fit is None else energy_fit.energy,
        energy_refinement_iterations=0 if energy_fit is None else energy_fit.iterations,
        energy_refinement_accepted_substeps=0 if energy_fit is None else energy_fit.accepted_substeps,
        energy_refinement_diagnostics=None if energy_fit is None else energy_fit.diagnostics,
        max_factor_norm=max(np.linalg.norm(x) for core in (a, b) for x in core.values()),
        sector_ranks=split.sector_ranks, accepted=accepted,
        full_local_dimension=problem.full_local_dimension)


def reduced_two_site_dmrg(hamiltonian, *, state, bond_dim, options):
    from .solver import LETTATwoSiteResult, LETTATwoSiteSweep

    if not isinstance(state, ReducedLatticeLETTA):
        raise TypeError("state must be ReducedLatticeLETTA")
    if not isinstance(hamiltonian, ReducedMPOHamiltonian):
        raise TypeError("hamiltonian must be ReducedMPOHamiltonian")
    _validate_reduced_mpo(hamiltonian, state)
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
    if options.gauge_mode == 'frontier':
        from .._letta_one_site_opt.reduced_gauge import (
            canonicalize_reduced_frontier, shift_reduced_frontier_gauge)
        canonicalize_reduced_frontier(state, 0 if direction == 'lr' else state.nsites-1,
                                      tolerance=options.metric_tolerance)
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
        updates = []
        for site in pair_sites:
            updates.append(_optimize_reduced_pair(
                state, hamiltonian, site, direction, bond_dim, options
            ))
            if options.gauge_mode == 'frontier':
                shift_reduced_frontier_gauge(state, site+1, direction,
                                            tolerance=options.metric_tolerance)
        updates = tuple(updates)
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
        if density_change <= options.tolerance and all(
                u.accepted and not u.fallback and not u.recovery_reason and
                (u.compression_diagnostics is None or u.compression_diagnostics.get('optimizer_success', False))
                for u in updates):
            from .._letta_one_site_opt.reduced_updates import reduced_stationarity
            audit = reduced_stationarity(state, hamiltonian, range(state.nsites), options)
            if max(r['relative_residual'] for r in audit) <= options.eigensolver_tolerance:
                converged = True
                message = "CONVERGENCE: ENERGY STATIONARY AND FRESH LOCAL RESIDUALS <= TOLERANCE"
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

"""Dependency-aware building blocks for general LETTA bond expansion.

Contraction connectors, legal factor arguments, and physical metrics are
different inventories. This module never infers one from an SVD reshape.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from time import perf_counter

import numpy as np

from .contractions import (BlockDiagonalMetric, DiagonalMetric, _contract_operands,
                           _equilibrated_metric_factors)


@dataclass(frozen=True)
class PhysicalIndex:
    site: int
    dimension: int
    home: int
    home_region: str
    dependent_sites: tuple[int, ...]
    regions: tuple[str, ...]
    role: str

    @property
    def category(self):
        # Unlike sorted incidence, this also names backward dependencies
        # without lying about which region contains the physical operator.
        return "".join(r for r in self.regions if r != self.home_region) + self.home_region


@dataclass(frozen=True)
class PhysicalIndexInventory:
    indices: tuple[PhysicalIndex, ...]
    left_site: int

    @classmethod
    def from_dependencies(cls, neighborhoods, *, homes, dimensions, left_site):
        neighborhoods = tuple(tuple(map(index, ns)) for ns in neighborhoods)
        homes, dimensions = tuple(map(index, homes)), tuple(map(index, dimensions))
        left_site = index(left_site)
        if not 0 <= left_site < len(neighborhoods) - 1:
            raise ValueError("the active pair is outside the tensor chain")
        if len(homes) != len(dimensions) or any(d <= 0 for d in dimensions):
            raise ValueError("physical homes and positive dimensions must match")
        if any(len(set(ns)) != len(ns) or any(p < 0 or p >= len(homes) for p in ns)
               for ns in neighborhoods):
            raise ValueError("invalid physical dependency list")

        def region(site):
            return ("L" if site < left_site else "A" if site == left_site else
                    "B" if site == left_site + 1 else "R")

        records = []
        for p, (home, dimension) in enumerate(zip(homes, dimensions)):
            dependent = tuple(k for k, ns in enumerate(neighborhoods) if p in ns)
            if home not in dependent:
                raise ValueError("each physical home must depend on its index")
            present = {region(k) for k in dependent}
            regions = tuple(r for r in "LABR" if r in present)
            role = ("shared" if "A" in present and "B" in present else
                    "row" if "A" in present else
                    "column" if "B" in present else "environment")
            records.append(PhysicalIndex(p, dimension, home, region(home),
                                         dependent, regions, role))
        return cls(tuple(records), left_site)

    @classmethod
    def from_state(cls, state, left_site):
        return cls.from_dependencies(
            tuple(state.site_neighborhood(k) for k in range(state.nsites)),
            homes=tuple(range(state.nsites)),
            dimensions=(state.physical_dim,) * state.nsites,
            left_site=left_site,
        )

    @property
    def categories(self):
        result = {}
        for item in self.indices:
            result.setdefault(item.category, []).append(item.site)
        return {key: tuple(value) for key, value in result.items()}


@dataclass(frozen=True)
class ResponseHalves:
    left: np.ndarray
    right: np.ndarray
    connectors: tuple[int, ...]
    connector_shape: tuple[int, ...]


@dataclass(frozen=True)
class ResponseRouting:
    """Exact bipartite routing with explicit output COPY constraints.

    Each operand is an (array, labels) pair. Shared labels are output/bra
    values only; their ket copies must have distinct labels. Missing size-one
    boundary legs may be supplied, but missing physical dependence is an error.
    """

    left_operands: tuple
    right_operands: tuple
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    shared: tuple[int, ...]
    dimensions: dict
    connectors: tuple[int, ...]

    @classmethod
    def from_cache(cls, cache, left, right, layout, a, b):
        site = layout.left_site
        ab, ak, aw = cache._group_labels(site)
        bb, bk, bw = cache._group_labels(site + 1)
        inventory = PhysicalIndexInventory.from_state(cache.state, site)
        shared_sites = {p.site for p in inventory.indices if p.role == "shared"}
        shared = tuple(cache.bra_physical[p] for p in layout.shared)
        if shared_sites != set(layout.shared):
            raise ValueError("pair layout and physical incidence disagree")
        return cls.from_operands(
            [(left, cache.frontiers[site]), (cache.mpo.factors[site], aw), (a, ak)],
            [(b, bk), (cache.mpo.factors[site + 1], bw), (right, cache.frontiers[site + 2])],
            rows=tuple(z for z in ab[:-1] if z not in shared),
            columns=tuple(z for z in bb[1:] if z not in shared),
            shared=shared, dimensions=cache.label_dimensions,
        )

    @classmethod
    def from_operands(cls, left, right, *, rows, columns, shared, dimensions):
        rows, columns, shared = tuple(rows), tuple(columns), tuple(shared)
        output = rows + columns + shared
        if len(set(output)) != len(output):
            raise ValueError("row, column and shared output labels must be disjoint")
        dimensions = {label: index(d) for label, d in dimensions.items()}
        groups = []
        for group in (left, right):
            normalized = []
            for operand, labels in group:
                operand, labels = np.asarray(operand), tuple(labels)
                if operand.shape != tuple(dimensions[z] for z in labels):
                    raise ValueError("operand shape disagrees with label dimensions")
                normalized.append((operand, labels))
            groups.append(normalized)

        def used(group):
            return {z for _, labels in group for z in labels}

        all_used = used(groups[0]) | used(groups[1])
        for z in output:
            if z not in all_used:
                if dimensions[z] != 1:
                    raise ValueError(f"missing nontrivial output label {z}")
                groups[1 if z in columns else 0].append((np.ones(1), (z,)))
        fresh = min((0, *dimensions)) - 1
        for own, coordinates in ((0, rows), (1, columns)):
            other = 1 - own
            for z in coordinates:
                if z not in used(groups[other]):
                    continue
                copy_label = fresh
                fresh -= 1
                dimensions[copy_label] = dimensions[z]
                groups[other] = [
                    (array, tuple(copy_label if label == z else label for label in labels))
                    for array, labels in groups[other]
                ]
                # This enforces equality without giving the other adjustable
                # tensor an extra argument. The copy is an intermediate only.
                groups[own].append((np.eye(dimensions[z]), (z, copy_label)))
        connectors = tuple(sorted((used(groups[0]) & used(groups[1])) - set(output)))
        return cls(tuple(groups[0]), tuple(groups[1]), rows, columns, shared,
                   dimensions, connectors)

    def half_matrices(self, configuration=()):
        configuration = tuple(map(index, configuration))
        if len(configuration) != len(self.shared) or any(
            value < 0 or value >= self.dimensions[label]
            for label, value in zip(self.shared, configuration)
        ):
            raise ValueError("invalid shared output configuration")
        fixed = dict(zip(self.shared, configuration))

        def contract(group, output):
            arrays, labels = [], []
            for array, indices in group:
                arrays.append(array[tuple(fixed.get(z, slice(None)) for z in indices)])
                labels.append(tuple(z for z in indices if z not in fixed))
            return _contract_operands(arrays, labels, output)

        connector_shape = tuple(self.dimensions[z] for z in self.connectors)
        middle = prod(connector_shape)
        left = contract(self.left_operands, self.rows + self.connectors)
        right = contract(self.right_operands, self.connectors + self.columns)
        return ResponseHalves(
            left.reshape(prod(self.dimensions[z] for z in self.rows), middle),
            right.reshape(middle, prod(self.dimensions[z] for z in self.columns)),
            self.connectors, connector_shape,
        )


@dataclass(frozen=True)
class ConditionalCandidates:
    tensor: np.ndarray
    sector_ranks: tuple[int, ...]
    connector_dimension: int
    largest_half: int
    largest_weighted: int
    discarded_coordinate_weight: float
    timings: dict[str, float]


def _column_complement(matrix, tolerance):
    u, values, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = (int(np.count_nonzero(values > tolerance * values[0]))
            if values.size and values[0] > 0 else 0)
    return u[:, rank:]


def _sector_sections(layout, configuration):
    left = [slice(None)] * len(layout.left_shape)
    right = [slice(None)] * len(layout.right_shape)
    for p, q in zip(layout.shared, configuration):
        left[1 + layout.left_neighborhood.index(p)] = q
        right[1 + layout.right_neighborhood.index(p)] = q
    return tuple(left), tuple(right)


def conditional_candidates(cache, left, right, layout, a, b, *, direction,
                           width, tolerance=1e-12):
    """Coordinate proposals with exact routing and legal shared blocks.

    The full opposite response rank is retained before contraction with the
    candidate half. Thus the candidate singular space equals that of the
    doubly coordinate-projected joined response without forming that response.
    These are proposals, not a claim of physical-metric optimality.
    """
    width = index(width)
    if direction not in ("lr", "rl") or width <= 0 or tolerance <= 0:
        raise ValueError("invalid candidate direction, width or tolerance")
    started = perf_counter()
    route = ResponseRouting.from_cache(cache, left, right, layout, a, b)
    timings = {"routing": perf_counter() - started}
    started = perf_counter()
    shape = a.shape[:-1] + (width,) if direction == "rl" else (width,) + b.shape[1:]
    tensor = np.zeros(shape, dtype=np.result_type(a, b, left, right, *cache.mpo.factors))
    ranks, largest_half, largest_weighted, discarded = [], 0, 0, 0.
    for block in np.ndindex(*((layout.physical_dim,) * len(layout.shared))):
        halves = route.half_matrices(block)
        asection, bsection = _sector_sections(layout, block)
        am = a[asection].reshape(-1, a.shape[-1])
        bm = b[bsection].reshape(b.shape[0], -1)
        ca = _column_complement(am, tolerance)
        cb = _column_complement(bm.conj().T, tolerance)
        largest_half = max(largest_half, halves.left.size, halves.right.size)
        if ca.shape[1] == 0 or cb.shape[1] == 0:
            ranks.append(0)
            continue
        lh, rh = ca.conj().T @ halves.left, halves.right @ cb
        # Transpose swaps the desired row/column candidate space, preserving
        # complex algebra (a conjugate transpose would change the response).
        if direction == "lr":
            lh, rh = rh.T, lh.T
        u, values, _ = np.linalg.svd(rh, full_matrices=False)
        weighted = lh @ (u * values)
        largest_weighted = max(largest_weighted, weighted.size)
        candidate, singular, _ = np.linalg.svd(weighted, full_matrices=False)
        rank = min(width, int(np.count_nonzero(singular > tolerance * singular[0]))) if singular.size and singular[0] > 0 else 0
        ranks.append(rank)
        discarded += float(np.sum(singular[rank:] ** 2))
        if direction == "rl":
            result = np.zeros((am.shape[0], width), dtype=tensor.dtype)
            result[:, :rank] = ca @ candidate[:, :rank]
            tensor[asection] = result.reshape(tensor[asection].shape)
        else:
            result = np.zeros((width, bm.shape[1]), dtype=tensor.dtype)
            result[:rank] = candidate[:, :rank].T @ cb.conj().T
            tensor[bsection] = result.reshape(tensor[bsection].shape)
    timings["preselection"] = perf_counter() - started
    return ConditionalCandidates(tensor, tuple(ranks),
                                 prod(route.dimensions[z] for z in route.connectors),
                                 largest_half, largest_weighted, discarded, timings)


def _join_halves(left, right, output, shape, *, variable=None):
    """Contract each local half before joining; never merge the two ket sites."""
    groups = [list(left), list(right)]
    used = [{z for _, labels in g for z in labels} for g in groups]
    for z, d in zip(output, shape):
        if z not in used[0] | used[1]:
            if d != 1:
                raise ValueError(f"missing nontrivial output label {z}")
            groups[0].append((np.ones(1), (z,)))
            used[0].add(z)
    connector = used[0] & used[1]
    arrays, labels = [], []
    dynamic = None
    for g, labels_used in zip(groups, used):
        retained = tuple(sorted(labels_used & (connector | set(output))))
        operands, indices = zip(*g)
        if variable is not None and variable[0] == len(arrays):
            from .contractions import _prepare_contraction
            dynamic = _prepare_contraction(operands, indices, retained, variable[1])
            dimensions = {z: d for a, ix in g for z, d in zip(ix, a.shape)}
            arrays.append(np.empty(tuple(dimensions[z] for z in retained)))
        else:
            arrays.append(_contract_operands(operands, indices, retained))
        labels.append(retained)
    if variable is not None:
        joined = _prepare_contraction(arrays, labels, output, variable[0])
        return lambda value: joined(dynamic(value)).reshape(shape)
    return _contract_operands(arrays, labels, output).reshape(shape)


@dataclass(frozen=True, eq=False)
class OneSiteFrame:
    side: str
    fixed: np.ndarray
    shape: tuple[int, ...]


class FixedPairFrames:
    """Actions of cross Grams between old and candidate one-site frames."""

    def __init__(self, cache, left, right, layout):
        self.cache, self.left, self.right, self.layout = cache, left, right, layout
        self._metrics = {}
        self._overlap_actions = {}
        self.overlap_applications = 0

    def frame(self, side, fixed):
        fixed = np.asarray(fixed)
        if side == "A":
            if fixed.shape[1:] != self.layout.right_shape[1:]:
                raise ValueError("fixed B has incompatible physical arguments")
            shape = self.layout.left_shape[:-1] + (fixed.shape[0],)
        elif side == "B":
            if fixed.shape[:-1] != self.layout.left_shape[:-1]:
                raise ValueError("fixed A has incompatible physical arguments")
            shape = (fixed.shape[-1],) + self.layout.right_shape[1:]
        else:
            raise ValueError("a frame varies either A or B")
        return OneSiteFrame(side, fixed, shape)

    def overlap(self, bra, ket, vector):
        self.overlap_applications += 1
        vector = np.asarray(vector)
        batched = vector.ndim == 2
        batch_shape = (vector.shape[1],) if batched else ()
        key = (bra, ket, batch_shape)
        value = vector.reshape(ket.shape + batch_shape)
        if key in self._overlap_actions:
            return self._overlap_actions[key](value).reshape((prod(bra.shape),) + batch_shape)
        site = self.layout.left_site
        ab, ak = self.cache._group_labels(site)
        bb, bk = self.cache._group_labels(site + 1)
        left = [(self.left, self.cache.frontiers[site])]
        right = [(self.right, self.cache.frontiers[site + 2])]
        batch_label = -100_000_003
        left.append((value if ket.side == "A" else ket.fixed,
                     ak + ((batch_label,) if batched and ket.side == "A" else ())))
        right.append((ket.fixed if ket.side == "A" else value,
                      bk + ((batch_label,) if batched and ket.side == "B" else ())))
        if bra.side == "A":
            right.append((bra.fixed.conj(), bb))
            output = ab
        else:
            left.append((bra.fixed.conj(), ab))
            output = bb
        action = _join_halves(left, right, output + ((batch_label,) if batched else ()),
                              bra.shape + batch_shape, variable=(0 if ket.side == "A" else 1, 1))
        # Frames and their fixed factors live only for this selection. Bound
        # batch variants without retaining tensor data in a global cache.
        if len(self._overlap_actions) >= 16:
            del self._overlap_actions[next(iter(self._overlap_actions))]
        self._overlap_actions[key] = action
        result = action(value)
        return result.reshape((prod(bra.shape),) + batch_shape)

    def metric(self, frame):
        if frame in self._metrics:
            return self._metrics[frame]
        # These operations form only the variable site's physical blocks.
        from .cbe import (_extend_identity_environment_with_tensor,
                          _effective_identity_metric_for_shape)
        site = self.layout.left_site
        left, right = self.left, self.right
        if frame.side == "A":
            right = _extend_identity_environment_with_tensor(
                self.cache, right, site + 1, frame.fixed, "rl")
        else:
            left = _extend_identity_environment_with_tensor(
                self.cache, left, site, frame.fixed, "lr")
            site += 1
        metric = _effective_identity_metric_for_shape(self.cache, left, right,
                                                     site, frame.shape)
        self._metrics[frame] = metric
        return metric

    def response(self, frame, hcache, hleft, hright, a, b):
        site = self.layout.left_site
        ab, ak, aw = hcache._group_labels(site)
        bb, bk, bw = hcache._group_labels(site + 1)
        left = [(hleft, hcache.frontiers[site]), (hcache.mpo.factors[site], aw), (a, ak)]
        right = [(b, bk), (hcache.mpo.factors[site + 1], bw), (hright, hcache.frontiers[site + 2])]
        if frame.side == "A":
            right.append((frame.fixed.conj(), bb))
            output = ab
        else:
            left.append((frame.fixed.conj(), ab))
            output = bb
        return _join_halves(left, right, output, frame.shape).reshape(-1)


class _BlockWhitener:
    """Compact supported one-site whitening; no dense n-by-rank basis."""

    def __init__(self, metric, tolerance):
        self.pieces, self.rank, self.size, self.dtype = [], 0, metric.size, metric.dtype
        for block, indices in zip(metric.blocks, metric.indices):
            if isinstance(metric, DiagonalMetric) and metric.support is not None:
                keep = metric.support[indices]
                block, indices = block[np.ix_(keep, keep)], indices[keep]
            basis, _ = _equilibrated_metric_factors(block, tolerance)
            count = basis.shape[1]
            if count:
                self.pieces.append((indices, slice(self.rank, self.rank + count),
                                    basis))
                self.rank += count

    def lift(self, value):
        result = np.zeros((self.size,) + value.shape[1:], dtype=np.result_type(value, self.dtype))
        for indices, section, matrix in self.pieces:
            result[indices] = matrix @ value[section]
        return result

    def adjoint(self, value):
        result = np.zeros((self.rank,) + value.shape[1:], dtype=np.result_type(value, self.dtype))
        for indices, section, matrix in self.pieces:
            result[section] = matrix.conj().T @ value[indices]
        return result

    def pseudoinverse(self, value):
        return self.lift(self.adjoint(value))

    def sector_columns(self, coordinate_sectors):
        result = {}
        for indices, section, _ in self.pieces:
            sector = int(coordinate_sectors[indices[0]])
            if np.any(coordinate_sectors[indices] != sector):
                raise ValueError("a metric block mixes independent shared outputs")
            result.setdefault(sector, []).extend(range(section.start, section.stop))
        return {sector: np.asarray(columns, dtype=int) for sector, columns in result.items()}


def _supported_schur_complement(nn, oo, on, tolerance):
    """Project the old span and reject numerically unresolved candidate modes.

    The candidate support is a stability-limited proposal space, not the
    physical norm used for compression. Its cutoff must reference the original
    candidate Gram, not amplify a tiny remainder after projection.
    """
    basis, _ = _equilibrated_metric_factors(oo, tolerance)
    projected = basis.conj().T @ on
    residual = nn - projected.conj().T @ projected
    values, vectors = np.linalg.eigh((residual + residual.conj().T) * .5)
    scale = max(0., float(np.linalg.eigvalsh((nn + nn.conj().T) * .5)[-1]))
    floor = max(tolerance, np.finfo(nn.real.dtype).eps * (len(nn) + len(oo)))
    supported = values > floor * scale
    return (vectors[:, supported] * values[supported]) @ vectors[:, supported].conj().T


def _schur_candidate_metric(network, old, candidate, tolerance):
    axis = 0 if old.side == "A" else -1
    enlarged = network.frame(old.side, np.concatenate([old.fixed, candidate.fixed], axis=axis))
    full = network.metric(enlarged)
    coordinates = np.arange(full.size).reshape(enlarged.shape)
    section = [slice(None)] * len(enlarged.shape)
    section[-1 if old.side == "A" else 0] = slice(old.shape[-1] if old.side == "A" else old.shape[0], None)
    new_indices = coordinates[tuple(section)].reshape(-1)
    new_positions = np.full(full.size, -1, dtype=int)
    new_positions[new_indices] = np.arange(new_indices.size)
    blocks, block_indices = [], []
    for block, indices in zip(full.blocks, full.indices):
        new = new_positions[indices] >= 0
        if not np.any(new):
            continue
        nn, oo, on = block[np.ix_(new, new)], block[np.ix_(~new, ~new)], block[np.ix_(~new, new)]
        blocks.append(_supported_schur_complement(nn, oo, on, tolerance))
        block_indices.append(new_positions[indices[new]])
    return BlockDiagonalMetric(new_indices.size, blocks, block_indices)


@dataclass(frozen=True)
class RestrictedPhysicalProblem:
    rhs: np.ndarray
    metric: BlockDiagonalMetric
    shape: tuple[int, ...]
    tangent_iterations: int
    tangent_relative_residual: float
    overlap_applications: int
    tangent_block_shapes: tuple[tuple[int, int], ...]
    timings: dict[str, float]


def restricted_physical_problem(hcache, hleft, hright, network, a, b, preselected,
                                *, direction, energy, tolerance=1e-11,
                                max_iterations=300):
    """Evaluate the full-old-tangent target and old-active Schur norm.

    This is the restricted quadratic in equations (89)-(92) of the category
    note. Both old tangents are removed from the RHS. Only the old *active*
    coefficients are eliminated from the candidate norm. No pair frame,
    pair metric, merged tensor or tangent Jacobian is materialized.
    """
    if direction not in ("lr", "rl"):
        raise ValueError("direction must be lr or rl")
    started = perf_counter()
    fa, fb = network.frame("A", b), network.frame("B", a)
    candidate = network.frame("B" if direction == "rl" else "A", preselected)
    wa = _BlockWhitener(network.metric(fa), tolerance)
    wb = _BlockWhitener(network.metric(fb), tolerance)
    timings = {"old_metrics": perf_counter() - started}

    def residual(frame):
        return (network.response(frame, hcache, hleft, hright, a, b)
                - energy * network.overlap(frame, fa, a.reshape(-1)))

    started = perf_counter()
    ra, rb, rp = residual(fa), residual(fb), residual(candidate)
    rhs = np.concatenate([wa.adjoint(ra), wb.adjoint(rb)])
    timings["response"] = perf_counter() - started

    started = perf_counter()
    iterations = 0
    block_shapes = []
    norm = float(np.linalg.norm(rhs))
    state_norm = np.sqrt(max(0., float(np.real(np.vdot(a.reshape(-1), network.metric(fa) @ a.reshape(-1))))))
    absolute_floor = max(100 * np.finfo(float).eps, tolerance) * max(norm, abs(energy) * state_norm, 1.)
    if norm > absolute_floor:
        # The overlap of two orthonormal frames determines their joint
        # projector. Singular values near one identify shared tangent modes.
        # Resolve this support explicitly instead of iterating on a singular
        # normal equation. This forms a ONE-SITE cross Gram, not a pair metric
        # or tangent Jacobian; its cost must be included in benchmarks.
        ar, br = rhs[:wa.rank], rhs[wa.rank:]
        sa, sb = np.zeros_like(ar), np.zeros_like(br)
        asectors, bsectors = np.empty(a.shape, dtype=int), np.empty(b.shape, dtype=int)
        configurations = tuple(np.ndindex(*((network.layout.physical_dim,) * len(network.layout.shared))))
        for sector, block in enumerate(configurations):
            asection, bsection = _sector_sections(network.layout, block)
            asectors[asection], bsectors[bsection] = sector, sector
        acols = wa.sector_columns(asectors.reshape(-1))
        bcols = wb.sector_columns(bsectors.reshape(-1))
        error_squared = 0.
        for sector in range(len(configurations)):
            ia, ib = acols.get(sector, np.array([], dtype=int)), bcols.get(sector, np.array([], dtype=int))
            block_shapes.append((ia.size, ib.size))
            if ia.size and ib.size:
                probes = np.zeros((wb.rank, ib.size), dtype=rhs.dtype)
                probes[ib, np.arange(ib.size)] = 1.
                cross = wa.adjoint(network.overlap(fa, fb, wb.lift(probes)))[ia]
            else:
                cross = np.zeros((ia.size, ib.size), dtype=rhs.dtype)
            u, singular, vh = np.linalg.svd(cross, full_matrices=False)
            v = vh.conj().T
            qa, qb = u.conj().T @ ar[ia], v.conj().T @ br[ib]
            plus = (qa + qb) / (2. * (1. + singular))
            minus = np.zeros_like(plus)
            keep = 1. - singular > tolerance * 2.
            minus[keep] = (qa[keep] - qb[keep]) / (2. * (1. - singular[keep]))
            sa[ia], sb[ib] = ar[ia] + u @ (plus + minus - qa), br[ib] + v @ (plus - minus - qb)
            error_squared += np.linalg.norm(sa[ia] + cross @ sb[ib] - ar[ia]) ** 2
            error_squared += np.linalg.norm(sb[ib] + cross.conj().T @ sa[ia] - br[ib]) ** 2
        relative_residual = float(np.sqrt(error_squared) / norm)
        if not np.all(np.isfinite(sa)) or not np.all(np.isfinite(sb)):
            raise FloatingPointError("nonfinite supported old-tangent projection")
        ca, cb = wa.lift(sa), wb.lift(sb)
        rp = rp - network.overlap(candidate, fa, ca) - network.overlap(candidate, fb, cb)
    else:
        relative_residual = 0.
    timings["tangent_projection"] = perf_counter() - started
    started = perf_counter()
    metric = _schur_candidate_metric(network, fb if direction == "rl" else fa,
                                     candidate, tolerance)
    timings["schur_metric"] = perf_counter() - started
    return RestrictedPhysicalProblem(rp, metric, candidate.shape, iterations,
                                     relative_residual, network.overlap_applications,
                                     tuple(block_shapes), timings)


def _separable_factors(metric, shape, tolerance):
    """Verify a Kronecker metric, including entries absent from stored blocks.

    Partial traces propose the factors. Verification streams rows, avoiding
    a dense vector-space metric or its rank-destroying general square root.
    """
    rows, columns = shape
    gl = np.zeros((rows, rows), dtype=metric.dtype)
    gr = np.zeros((columns, columns), dtype=metric.dtype)
    for block, indices in zip(metric.blocks, metric.indices):
        r, c = indices // columns, indices % columns
        for i in range(indices.size):
            same_c, same_r = c == c[i], r == r[i]
            np.add.at(gl[r[i]], r[same_c], block[i, same_c])
            np.add.at(gr[c[i]], c[same_r], block[i, same_r])
    trace = float(np.real(np.trace(gl)))
    if trace <= 0.:
        return None
    gl = (gl + gl.conj().T) / (2. * trace)
    gr = (gr + gr.conj().T) * .5
    scale = np.sqrt(sum(float(np.linalg.norm(b) ** 2) for b in metric.blocks))
    error = 0.
    rr, cc = np.arange(metric.size) // columns, np.arange(metric.size) % columns
    for block, indices in zip(metric.blocks, metric.indices):
        for i, coordinate in enumerate(indices):
            predicted = gl[coordinate // columns, rr] * gr[coordinate % columns, cc]
            predicted[indices] -= block[i]
            error += float(np.linalg.norm(predicted) ** 2)
            if error > (tolerance * scale) ** 2:
                return None
    return gl, gr


@dataclass(frozen=True)
class LowRankPhysicalFit:
    left: np.ndarray
    right: np.ndarray
    loss: float
    captured_weight: float
    available_weight: float
    metric_kind: str
    iterations: int


def fit_low_rank_quadratic(rhs, metric, shape, *, rank, tolerance=1e-11,
                          max_iterations=8):
    """Minimize z* N z - 2 Re(z* b) with a legal matrix rank constraint.

    A separable supported metric admits rank-preserving whitening and an
    optimal SVD. For other metrics a monotone ALS fit is local, not globally
    optimal. Returned weights are measured in the supplied physical metric.
    """
    rows, columns = map(index, shape)
    rank = index(rank)
    rhs = np.asarray(rhs).reshape(-1)
    if rank <= 0 or rows * columns != metric.size or rhs.size != metric.size:
        raise ValueError("incompatible low-rank physical fit dimensions")
    whitener = _BlockWhitener(metric, tolerance)
    dtype = np.result_type(rhs, metric.dtype)
    if whitener.rank == 0:
        return LowRankPhysicalFit(np.zeros((rows, rank), dtype=dtype),
                                  np.zeros((rank, columns), dtype=dtype),
                                  0., 0., 0., "zero", 0)
    target = whitener.pseudoinverse(rhs).reshape(rows, columns)
    supported_rhs = (metric @ target.reshape(-1)).reshape(rows, columns)
    available = max(0., float(np.real(np.vdot(target, supported_rhs))))
    factors = _separable_factors(metric, (rows, columns), tolerance * 10)
    if factors is not None:
        gl, gr = factors
        bl, _ = _equilibrated_metric_factors(gl, tolerance)
        br, _ = _equilibrated_metric_factors(gr, tolerance)
        white_rhs = bl.conj().T @ supported_rhs @ br.conj()
        u, s, vh = np.linalg.svd(white_rhs, full_matrices=False)
        keep = min(rank, s.size)
        left, right = np.zeros((rows, rank), dtype=dtype), np.zeros((rank, columns), dtype=dtype)
        left[:, :keep] = bl @ u[:, :keep]
        right[:keep] = (s[:keep, None] * vh[:keep]) @ br.T
        iterations, kind = 0, "separable"
    else:
        from .cbe import _metric_low_rank_factorization
        # Normalize the ALS problem so its stopping rule does not confuse
        # a small residual amplitude with convergence of the low-rank fit.
        target_scale = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        metric_scale = max(float(np.linalg.norm(b)) for b in metric.blocks)
        normalized = BlockDiagonalMetric(metric.size,
                                         [b / metric_scale for b in metric.blocks],
                                         metric.indices)
        left, right, _, iterations = _metric_low_rank_factorization(
            target / target_scale, normalized, rank, tolerance=tolerance,
            max_iterations=max_iterations, metric_tolerance=tolerance,
        )
        right *= target_scale
        kind = "general"
    approximation = (left @ right).reshape(-1)
    captured = float(np.real(2 * np.vdot(approximation, supported_rhs.reshape(-1))
                             - np.vdot(approximation, metric @ approximation)))
    if not np.isfinite(captured):
        raise FloatingPointError("nonfinite physical low-rank objective")
    return LowRankPhysicalFit(left, right, max(0., available - captured),
                              captured, available, kind, iterations)


def general_cbe_selection(cache, left, right, layout, a, b, *,
                          expansion_dimension, preselection_dimension, direction,
                          tolerance=1e-12, metric_cache=None, metric_left=None,
                          metric_right=None, energy=None, metric_tolerance=1e-12):
    """Complete conditional selector; physical geometry when metrics supplied."""
    from .cbe import CBESelection
    from .._letta_two_site_opt.pair import LETTAPairLayout
    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout")
    expansion_dimension, preselection_dimension = index(expansion_dimension), index(preselection_dimension)
    if expansion_dimension <= 0 or preselection_dimension < expansion_dimension:
        raise ValueError("preselection_dimension must be at least expansion_dimension > 0")
    if metric_tolerance <= 0 or tolerance <= 0:
        raise ValueError("selection and metric tolerances must be positive")
    inputs = (metric_cache, metric_left, metric_right, energy)
    if any(x is not None for x in inputs) and not all(x is not None for x in inputs):
        raise ValueError("metric_cache, metric_left, metric_right, and energy must be provided together")
    a, b = np.asarray(a), np.asarray(b)
    proposal = conditional_candidates(cache, left, right, layout, a, b,
                                      direction=direction, width=preselection_dimension,
                                      tolerance=tolerance)
    timings = dict(proposal.timings)
    timings.update({phase: 0. for phase in (
        "old_metrics", "response", "tangent_projection", "schur_metric")})
    dtype = proposal.tensor.dtype
    ld = np.zeros(a.shape[:-1] + (expansion_dimension,), dtype=dtype)
    rd = np.zeros((expansion_dimension,) + b.shape[1:], dtype=dtype)
    physical = None
    if any(proposal.sector_ranks) and metric_cache is not None:
        network = FixedPairFrames(metric_cache, metric_left, metric_right, layout)
        physical = restricted_physical_problem(
            cache, left, right, network, a, b, proposal.tensor,
            direction=direction, energy=energy, tolerance=metric_tolerance,
        )
        timings.update(physical.timings)
        grid = np.arange(physical.metric.size).reshape(physical.shape)
    elif any(proposal.sector_ranks):
        route = ResponseRouting.from_cache(cache, left, right, layout, a, b)
    started = perf_counter()
    ranks, kinds, total, captured, loss, iterations, final_size = [], [], 0., 0., 0., 0, 0
    for block, proposal_rank in zip(np.ndindex(*((layout.physical_dim,) * len(layout.shared))),
                                    proposal.sector_ranks):
        asec, bsec = _sector_sections(layout, block)
        if proposal_rank == 0:
            ranks.append(0)
            kinds.append("zero")
            continue
        if direction == "rl":
            pre = proposal.tensor[asec].reshape(-1, preselection_dimension)
            matrix_shape = (preselection_dimension, int(np.prod(b[bsec].shape[1:])))
            section = bsec
        else:
            pre = proposal.tensor[bsec].reshape(preselection_dimension, -1)
            matrix_shape = (int(np.prod(a[asec].shape[:-1])), preselection_dimension)
            section = asec
        if physical is not None:
            coordinates = grid[section].reshape(-1)
            metric = physical.metric.restrict(coordinates)
            fit = fit_low_rank_quadratic(physical.rhs[coordinates], metric,
                                         matrix_shape, rank=expansion_dimension,
                                         tolerance=metric_tolerance)
            x, y = fit.left, fit.right
            total += fit.available_weight
            captured += max(0., fit.captured_weight)
            loss += fit.loss
            kinds.append(fit.metric_kind)
            iterations += fit.iterations
            if fit.captured_weight <= 0.:
                ranks.append(0)
                continue
        else:
            halves = route.half_matrices(block)
            if direction == "rl":
                matrix = (pre.conj().T @ halves.left) @ halves.right
                bm = b[bsec].reshape(b.shape[0], -1)
                matrix -= (matrix @ np.linalg.pinv(bm, rcond=tolerance)) @ bm
            else:
                matrix = halves.left @ (halves.right @ pre.conj().T)
                am = a[asec].reshape(-1, a.shape[-1])
                matrix -= am @ (np.linalg.pinv(am, rcond=tolerance) @ matrix)
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = min(expansion_dimension, s.size)
            x, y = u[:, :keep], s[:keep, None] * vh[:keep]
            total += float(np.sum(s ** 2))
            captured += float(np.sum(s[:keep] ** 2))
            loss += float(np.sum(s[keep:] ** 2))
            kinds.append("coordinate")
        final_size = max(final_size, prod(matrix_shape))
        if direction == "rl":
            candidate = pre @ x
        else:
            candidate = y @ pre
        u, s, vh = np.linalg.svd(candidate, full_matrices=False)
        rank = min(expansion_dimension, int(np.count_nonzero(s > tolerance * s[0]))) if s.size and s[0] > 0 else 0
        ranks.append(rank)
        lm = np.zeros((int(np.prod(a[asec].shape[:-1])), expansion_dimension), dtype=dtype)
        rm = np.zeros((expansion_dimension, int(np.prod(b[bsec].shape[1:]))), dtype=dtype)
        if direction == "rl":
            lm[:, :rank] = u[:, :rank]
            rm[:rank] = (s[:rank, None] * vh[:rank]) @ y
        else:
            lm[:, :rank] = (x @ u[:, :rank]) * s[:rank]
            rm[:rank] = vh[:rank]
        ld[asec], rd[bsec] = lm.reshape(ld[asec].shape), rm.reshape(rd[bsec].shape)
    timings["restricted_fit"] = perf_counter() - started
    return CBESelection(
        left_direction=ld, right_direction=rd, loss=loss,
        captured_weight=float(np.clip(captured / total, 0., 1.)) if total > 0 else 0.,
        sector_ranks=tuple(ranks), refinement_iterations=iterations, selector="shrewd",
        preselection_dimension=max(proposal.sector_ranks, default=0),
        preselection_loss=proposal.discarded_coordinate_weight,
        missing_norm=float(np.sqrt(total)), pair_action_count=0,
        pair_metric_count=0, merged_pair_count=0,
        preselection_output_size=max(proposal.largest_half, proposal.largest_weighted),
        final_output_size=physical.metric.size if physical is not None else final_size,
        metric_kinds=tuple(kinds),
        tangent_iterations=physical.tangent_iterations if physical is not None else 0,
        tangent_relative_residual=physical.tangent_relative_residual if physical is not None else 0.,
        overlap_applications=physical.overlap_applications if physical is not None else 0,
        connector_dimension=proposal.connector_dimension,
        tangent_block_shapes=physical.tangent_block_shapes if physical is not None else (),
        selection_timings=timings,
    )

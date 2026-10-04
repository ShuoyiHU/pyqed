"""Conditional gauges of complete overlap frontiers, including internal ties.

See ``letta_frontier_gauge.tex`` for the admissibility and rank conditions.
All transformations are invertible: small metric eigenvalues are not deleted.
"""

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.linalg import solve_triangular

from .contractions import IdentityEnvironmentCache
from .canonical import get_canonical, save_canonical


@dataclass(frozen=True)
class FrontierGaugeCut:
    cut: int
    physical_indices: tuple[int, ...]
    shared_indices: tuple[int, ...]

    @property
    def admissible(self):
        """Whether every frontier index is available on both adjacent tensors."""
        return set(self.physical_indices) <= set(self.shared_indices)


@dataclass(frozen=True)
class FrontierGaugeReport:
    cut: FrontierGaugeCut
    direction: str
    applied: bool
    ranks: tuple[int, ...] = ()
    bond_dimension: int = 0
    projector_residual: float | None = None
    tolerance: float = 1.e-12

    @property
    def full_rank(self):
        return self.applied and all(r == self.bond_dimension for r in self.ranks)

    @property
    def full_identity(self):
        """Whether full rank and the measured unit-Gram residual both pass."""
        return (self.full_rank and self.projector_residual is not None
                and self.projector_residual <= max(10 * self.tolerance,
                                                  100 * np.finfo(float).eps * self.bond_dimension))


def frontier_gauge_cuts(state):
    """Classify each internal cut from actual physical-index incidence.

    This is a sufficient structural test for arbitrary tensor values, not a claim
    that every rejected state lacks an accidental, more restricted gauge.
    """
    neighborhoods = [set(state.site_neighborhood(i)) for i in range(state.nsites)]
    suffix = [set() for _ in range(state.nsites + 1)]
    for i in range(state.nsites - 1, -1, -1):
        suffix[i] = suffix[i + 1] | neighborhoods[i]
    prefix = set()
    cuts = []
    for cut in range(1, state.nsites):
        prefix |= neighborhoods[cut - 1]
        cuts.append(FrontierGaugeCut(
            cut, tuple(sorted(prefix & suffix[cut])),
            tuple(sorted(neighborhoods[cut - 1] & neighborhoods[cut])),
        ))
    return tuple(cuts)


def _row_preserving_qr(matrix):
    """Retain small physical rows when large, possibly null rows dominate QR.

    Householder Q is normwise accurate, but can lose an entire small row.
    A well-conditioned R lets us recover Q row by row without mixing those
    scales. If that solve is unsafe, a scalar gauge preserves the input instead
    of imposing inaccurate orthogonality.
    """
    q, r = np.linalg.qr(matrix, mode="reduced")
    epsilon = np.finfo(matrix.real.dtype).eps
    row_scale = np.max(np.abs(matrix), axis=1, initial=0.)
    allowance = 32 * epsilon * max(1, matrix.shape[1]) * row_scale

    def preserves_rows(factor):
        error = np.max(np.abs(factor @ r - matrix), axis=1, initial=0.)
        return np.all(error <= allowance)

    if preserves_rows(q):
        return q, r
    if (r.shape[0] == r.shape[1]
            and np.linalg.cond(r) < 1. / np.sqrt(epsilon)):
        corrected = solve_triangular(r.T, matrix.T, lower=True).T
        if preserves_rows(corrected):
            return corrected, r
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.:
        return q, r
    return matrix / scale, np.eye(matrix.shape[1], dtype=matrix.dtype) * scale


def _fixed_size_qr(state, site, direction):
    """Ordinary QR fallback without changing cached bond dimensions."""
    if state.symmetry is not None:
        from .solver import _shift_symmetry_virtual_gauge
        _shift_symmetry_virtual_gauge(state, site, direction)
        return
    a = state.tensors[site]
    matrix = a.reshape(-1, a.shape[-1]) if direction == "lr" else a.reshape(a.shape[0], -1).T
    q, r = _row_preserving_qr(matrix)
    dimension = matrix.shape[1]
    padded_q = np.zeros_like(matrix)
    padded_r = np.zeros((dimension, dimension), dtype=r.dtype)
    padded_q[:, :q.shape[1]] = q
    padded_r[:r.shape[0]] = r
    if direction == "lr":
        state.tensors[site] = padded_q.reshape(a.shape)
        state.tensors[site + 1] = np.tensordot(padded_r, state.tensors[site + 1], axes=(1, 0))
    else:
        state.tensors[site] = padded_q.T.reshape(a.shape)
        state.tensors[site - 1] = np.tensordot(state.tensors[site - 1], padded_r.T, axes=(-1, 0))


def _gram_blocks(cache, environment, cut):
    if cut in {0, cache.state.nsites}:
        return np.asarray(environment).reshape(1, 1), ()
    physical = tuple(p for p, label in enumerate(cache.physical)
                     if label in cache.frontiers[cut])
    labels = tuple(cache.physical[p] for p in physical)
    order = tuple(cache.frontiers[cut].index(label) for label in
                  labels + (cache.bra_virtual[cut], cache.ket_virtual[cut]))
    return np.asarray(environment).transpose(order), physical


def _incoming_roots(cache, incoming, cut, tolerance, direction):
    record = get_canonical(cache.state, cut, direction, incoming)
    grams, physical = _gram_blocks(cache, incoming, cut)
    roots = np.empty_like(grams)
    if record is not None or cut in {0, cache.state.nsites}:
        diagonal = (record.diagonal if record is not None else
                    np.asarray(incoming).reshape(1))
        roots.fill(0.)
        indices = np.arange(grams.shape[-1])
        roots[..., indices, indices] = np.sqrt(diagonal)
        return roots, physical, True
    for configuration in np.ndindex(grams.shape[:-2]):
        gram = grams[configuration]
        if not np.all(np.isfinite(gram)):
            raise FloatingPointError("nonfinite frontier Gram matrix")
        values, vectors = np.linalg.eigh(.5 * (gram + gram.conj().T))
        scale = float(np.max(np.abs(values)))
        if values[0] < -10 * max(tolerance, np.finfo(float).eps * len(values)) * scale:
            raise FloatingPointError("frontier Gram matrix is not positive semidefinite")
        roots[configuration] = np.sqrt(np.maximum(values, 0.))[:, None] * vectors.conj().T
    return roots, physical, False


def _weighted_frame(state, site, direction, outgoing, configuration, roots, physical):
    neighborhood = state.site_neighborhood(site)
    fixed = dict(zip(outgoing, configuration))
    summed = tuple(p for p in neighborhood if p not in fixed)
    dimensions = tuple(state.tensors[site].shape[1 + neighborhood.index(p)] for p in summed)
    rows = []
    for values in np.ndindex(dimensions):
        assignment = fixed | dict(zip(summed, values))
        root = roots[tuple(assignment[p] for p in physical)]
        section = (slice(None),) + tuple(assignment[p] for p in neighborhood) + (slice(None),)
        a = state.tensors[site][section]
        rows.append(root @ (a if direction == "lr" else a.T))
    return np.concatenate(rows, axis=0)


def _whiten_frame(frame, groups, tolerance):
    """SVD of a square-root environment avoids squaring the output condition."""
    if not np.all(np.isfinite(frame)):
        raise FloatingPointError("nonfinite weighted frontier frame")
    decompositions = []
    for group in groups:
        matrix = frame[:, group]
        # Complete the virtual null space without allocating a rows-by-rows U.
        if matrix.shape[0] < len(group):
            matrix = np.pad(matrix, ((0, len(group) - matrix.shape[0]), (0, 0)))
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        decompositions.append((singular, vh.conj().T, u[:frame.shape[0]]))
    scale = max(float(singular[0]) for singular, _, _ in decompositions)
    dimension = frame.shape[1]
    transform = np.eye(dimension, dtype=frame.dtype)
    inverse = transform.copy()
    supported = np.zeros(dimension, dtype=bool)
    canonical_frame = np.zeros_like(frame)
    diagonal = np.zeros(dimension)
    if scale == 0.:
        return transform, inverse, supported, canonical_frame, diagonal
    cutoff = np.sqrt(max(tolerance, np.finfo(frame.real.dtype).eps * dimension)) * scale
    for group, (singular, vectors, u) in zip(groups, decompositions):
        keep = singular > cutoff
        roots = np.where(keep, singular, scale)
        ix = np.ix_(group, group)
        transform[ix] = vectors / roots
        inverse[ix] = roots[:, None] * vectors.conj().T
        supported[group] = keep
        canonical_frame[:, group] = u * (singular / roots)
        diagonal[group] = (singular / roots) ** 2
    return transform, inverse, supported, canonical_frame, diagonal


def _assign_canonical_rows(tensor, neighborhood, direction, outgoing, configuration,
                           roots, physical, frame, tolerance):
    """Use SVD orthogonal factors directly on rows with nonzero input norm.

    Null input rows retain the invertible bond transformation already applied
    to the tensor. No bond coordinates or small singular values are deleted.
    """
    fixed = dict(zip(outgoing, configuration))
    summed = tuple(p for p in neighborhood if p not in fixed)
    dimensions = tuple(tensor.shape[1 + neighborhood.index(p)] for p in summed)
    offset = 0
    for values in np.ndindex(dimensions):
        assignment = fixed | dict(zip(summed, values))
        weights = np.diag(roots[tuple(assignment[p] for p in physical)])
        count = len(weights)
        # Dividing through numerical-null input weights can amplify roundoff
        # into enormous tensor entries. Those rows keep A @ W instead.
        cutoff = np.sqrt(max(tolerance, np.finfo(float).eps * count)) * np.max(np.abs(weights))
        nonzero = np.abs(weights) > cutoff
        section = (slice(None),) + tuple(assignment[p] for p in neighborhood) + (slice(None),)
        matrix = tensor[section] if direction == "lr" else tensor[section].T
        matrix[nonzero, :] = frame[offset:offset + count][nonzero] / weights[nonzero, None]
        offset += count


def shift_frontier_gauge(state, site, direction, *, cache=None, incoming=None,
                         tolerance=1.e-12):
    """Gauge the completed side and return ``(outgoing_environment, report)``.

    ``incoming`` excludes ``site`` and must be exact and current. Passing the
    sweep's cache and incoming boundary avoids rebuilding the completed side.
    If omitted, only that side is contracted. Inadmissible cuts use ordinary
    QR. Shapes and physical dependencies are preserved in both cases.
    """
    site = index(site)
    if direction not in {"lr", "rl"}:
        raise ValueError("direction must be 'lr' or 'rl'.")
    if not 0 <= site < state.nsites:
        raise IndexError("site index out of range.")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("gauge tolerance must lie strictly between zero and one.")
    cache = IdentityEnvironmentCache(state) if cache is None else cache
    if cache.state is not state or cache.boundary_bond_dim is not None:
        raise ValueError("frontier gauges require an exact cache of the same state.")
    extend = cache.extend_left if direction == "lr" else cache.extend_right
    if incoming is None:
        incoming = cache.scalar_boundary()
        sites = range(site) if direction == "lr" else range(state.nsites - 1, site, -1)
        for i in sites:
            incoming = extend(incoming, i)
    cut = site + 1 if direction == "lr" else site
    if cut in {0, state.nsites}:
        return extend(incoming, site), None
    if not hasattr(cache, "_gauge_cuts"):
        cache._gauge_cuts = frontier_gauge_cuts(state)
    classification = cache._gauge_cuts[cut - 1]
    dimension = state.tensors[cut - 1].shape[-1]
    if not classification.admissible:
        _fixed_size_qr(state, site, direction)
        return extend(incoming, site), FrontierGaugeReport(
            classification, direction, False, bond_dimension=dimension, tolerance=tolerance)
    incoming_cut = site if direction == "lr" else site + 1
    roots, incoming_physical, diagonal_input = _incoming_roots(
        cache, incoming, incoming_cut, tolerance, direction)
    physical_labels = tuple(cache.physical[p] for p in classification.physical_indices)
    order = tuple(cache.frontiers[cut].index(label) for label in
                  physical_labels + (cache.bra_virtual[cut], cache.ket_virtual[cut]))
    physical_shape = tuple(state.physical_dim for _ in physical_labels)
    a, b = state.tensors[cut - 1].copy(), state.tensors[cut].copy()
    if state.symmetry is None:
        groups = (np.arange(dimension),)
    else:
        charges = state.bond_charges[cut - 1]
        groups = tuple(np.array([i for i, q in enumerate(charges) if q == charge])
                       for charge in dict.fromkeys(charges))
    ranks, residual = [], 0.
    supports = []
    diagonals = np.empty(physical_shape + (dimension,))
    for configuration in np.ndindex(physical_shape):
        frame = _weighted_frame(state, site, direction, classification.physical_indices,
                                configuration, roots, incoming_physical)
        w, inverse, supported, canonical_frame, diagonal = _whiten_frame(frame, groups, tolerance)
        left_section, right_section = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        for p, value in zip(classification.physical_indices, configuration):
            left_section[1 + state.site_neighborhood(cut - 1).index(p)] = value
            right_section[1 + state.site_neighborhood(cut).index(p)] = value
        left_section, right_section = tuple(left_section), tuple(right_section)
        if direction == "lr":
            a[left_section] = np.tensordot(a[left_section], w, axes=(-1, 0))
            b[right_section] = np.tensordot(inverse, b[right_section], axes=(1, 0))
        else:
            # Environments use (bra, ket) order on BOTH sides. Hence transpose,
            # not adjoint, when the ket gauge acts on the left axis of B.
            a[left_section] = np.tensordot(a[left_section], inverse.T, axes=(-1, 0))
            b[right_section] = np.tensordot(w.T, b[right_section], axes=(1, 0))
        if diagonal_input:
            _assign_canonical_rows(a if direction == "lr" else b,
                                   state.site_neighborhood(site), direction,
                                   classification.physical_indices, configuration,
                                   roots, incoming_physical, canonical_frame, tolerance)
        ranks.append(int(np.count_nonzero(supported)))
        supports.append(supported)
        diagonals[configuration] = diagonal
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise FloatingPointError("nonfinite frontier gauge transformation")
    state.tensors[cut - 1], state.tensors[cut] = a, b
    # Contract the transformed tensor, rather than a congruence of a previously
    # rounded Gram. The latter amplifies roundoff at small singular values.
    outgoing = np.asarray(extend(incoming, site))
    if not np.all(np.isfinite(outgoing)):
        raise FloatingPointError("nonfinite gauged frontier environment")
    transformed = outgoing.transpose(order)
    for configuration, supported in zip(np.ndindex(physical_shape), supports):
        residual = max(residual, float(np.linalg.norm(
            transformed[configuration] - np.diag(supported.astype(float)), ord=np.inf)))
    report = FrontierGaugeReport(
        classification, direction, True, tuple(ranks), dimension, residual, tolerance)
    exact = np.zeros_like(transformed)
    indices = np.arange(dimension)
    exact[..., indices, indices] = diagonals
    error = max(float(np.linalg.norm((transformed - exact)[q], ord=np.inf))
                for q in np.ndindex(physical_shape))
    threshold = max(.1 * tolerance, 64 * np.finfo(float).eps * dimension)
    if error <= threshold:
        record = save_canonical(state, cut, direction, classification.physical_indices,
                                diagonals, exact.transpose(np.argsort(order)), report, error)
        return record.environment, report
    return outgoing, report


def canonicalize_frontier(state, center=0, *, tolerance=1.e-12):
    """Move all admissible complete-side gauges toward ``center``, in place.

    No state normalization or truncation is performed. A subsequent global
    tensor rebalance would undo the unit normalization of the environments.
    Reports identify where only a supported projector (or QR) is available.
    """
    center = index(center)
    if not 0 <= center < state.nsites:
        raise IndexError("center index out of range.")
    cache = IdentityEnvironmentCache(state)
    reports = []
    for direction, sites in (("lr", range(center)),
                             ("rl", range(state.nsites - 1, center, -1))):
        boundary = cache.scalar_boundary()
        for site in sites:
            cut = site + 1 if direction == "lr" else site
            saved = cache.saved_canonical(cut, direction)
            if saved is not None and saved.report.tolerance == tolerance:
                boundary, report = saved.environment, saved.report
            else:
                boundary, report = shift_frontier_gauge(
                    state, site, direction, cache=cache, incoming=boundary, tolerance=tolerance)
            reports.append(report)
    return tuple(reports)

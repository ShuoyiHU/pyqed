"""Symmetry-equivariant gauges on full or marginal scalar-label frontiers.

The invariant boundary metric is G[q,xi] tensor I_(2S+1). Only multiplicity
coordinates are transformed. The metric is contracted directly in reduced
multiplicity spaces; no magnetic components are restored.
"""
from collections import Counter
from operator import index

import numpy as np

from .gauge import frontier_gauge_cuts
from .reduced_norm import ReducedNormChain
from .reduced_frontier import ReducedFrontier


def _validate_cut(state, cut, direction, *, strict=False):
    cut = index(cut)
    if not 0 < cut < state.nsites:
        raise ValueError('cut must be an internal bond')
    if direction not in {'lr', 'rl'}:
        raise ValueError('direction must be lr or rl')
    descriptor = frontier_gauge_cuts(state)[cut-1]
    if strict and not descriptor.admissible:
        raise ValueError(f'cut {cut} does not have a shared frontier; use a scalar gauge')
    return descriptor


def reduced_gauge_variables(state, cut):
    """Labels available to both factors for an invertible conditional gauge."""
    descriptor = _validate_cut(state, cut, 'lr')
    return tuple(v for v in descriptor.physical_indices if v in descriptor.shared_indices)


def reduced_frontier_grams(state, cut, direction, *, environment=None, strict=False):
    """Return legal conditional Grams keyed by (shared configuration, sector).

    If some frontier labels are unavailable on either neighboring tensor, sum
    their diagonal environment blocks. This partial trace supplies an invertible
    bond preconditioner; it does not eliminate off-diagonal memory correlations
    or replace the full physical overlap metric. ``strict=True`` requires the
    complete frontier to be shared and rejects the marginal fallback.

    The right boundary sums the entire target multiplet, so this metric is
    invariant also for non-singlets. Its physical norm convention is a sum
    over target M; ``state.norm()`` divides that sum by 2S+1.
    """
    descriptor = _validate_cut(state, cut, direction, strict=strict)
    variables = reduced_gauge_variables(state, cut)
    if environment is None:
        frontier = ReducedFrontier.from_state(state)
        chain = ReducedNormChain.build(frontier.to_mps(state))
    else:
        chain, frontier = environment
    if direction == 'lr':
        chain.ensure(cut, state.nsites)
        environment = chain.left[cut]
        scale = np.exp(chain.left_log_scales[cut])
    else:
        chain.ensure(0, cut)
        environment = chain.right[cut]
        scale = np.exp(chain.right_log_scales[cut])
    multiplicities = Counter(state.bond_sectors[cut-1])
    configurations = frontier._assignments(descriptor.physical_indices)
    grams = {}
    for memory, configuration in enumerate(configurations):
        assignment = dict(zip(descriptor.physical_indices, configuration))
        shared = tuple(assignment[v] for v in variables)
        for q, r in multiplicities.items():
            start = memory*r
            gram = environment[q][start:start+r, start:start+r]*scale
            key = (shared, q)
            grams[key] = grams.get(key, 0.) + .5*(gram+gram.conj().T)
    return grams


def reduced_frontier_factors(state, cut, direction, *, environment=None, strict=False):
    """Square-root factors of the legal marginal Grams, without Gram products.

    Adapts reduced sector QR and standard SVD whitening (Trefethen and Bau,
    Numerical Linear Algebra, SIAM 1997, doi:10.1137/1.9780898719574).
    Stacking conditional column slices implements the marginal partial trace.
    This conditions legal bond gauges only, not the full correlated overlap.
    """
    from .reduced_coordinates import _canonical_sites
    descriptor = _validate_cut(state, cut, direction, strict=strict)
    variables = reduced_gauge_variables(state, cut)
    if environment is None:
        frontier = ReducedFrontier.from_state(state)
        sites = frontier.to_mps(state)
    else:
        chain, frontier = environment
        sites = chain.sites
    _, left, right = _canonical_sites(sites, cut if direction == 'lr' else cut-1)
    boundary = left if direction == 'lr' else right
    groups = {}
    for memory, configuration in enumerate(frontier._assignments(descriptor.physical_indices)):
        assignment = dict(zip(descriptor.physical_indices, configuration))
        shared = tuple(assignment[v] for v in variables)
        for q, r in Counter(state.bond_sectors[cut-1]).items():
            groups.setdefault((shared, q), []).append(boundary[q][:, memory*r:(memory+1)*r])
    return {key: np.linalg.qr(np.concatenate(parts), mode='r') for key, parts in groups.items()}


def _apply_conditional(state, site, axis, variables, matrices):
    basis = state.physical_basis
    labels = {(p.sector, p.copy): i for i, p in enumerate(basis.reduced_states)}
    neighborhood = state.site_neighborhood(site)
    for key, block in state.tensors[site].items():
        out = np.empty_like(block, dtype=np.result_type(block, *[a.dtype for a in matrices.values()]))
        q = key[0] if axis == 0 else key[2]
        for physical in np.ndindex(block.shape[1:-1]):
            assignment = dict(zip(neighborhood, (labels[(key[1], physical[0])],)+physical[1:]))
            matrix = matrices[(tuple(assignment[v] for v in variables), q)]
            section = (slice(None),)+physical+(slice(None),)
            out[section] = matrix@block[section] if axis == 0 else block[section]@matrix
        state.tensors[site][key] = np.real_if_close(out)


def shift_reduced_frontier_gauge(state, cut, direction, *, tolerance=1e-12, environment=None, strict=False):
    """Whiten one completed side, absorbing its inverse into the neighbor.

    Small squared singular values retain an invertible unit gauge rather than
    deleting physical directions. Bond dimensions and tied dependencies are
    unchanged. The support threshold uses the largest singular value across
    all conditional blocks, so negligible-weight blocks are not independently
    amplified. On a non-shared frontier only its legal marginal is whitened;
    the full local metric remains correlated. Returns the supported rank of each
    (shared-label configuration, sector) block. Numerical failures restore both
    cores. ``strict=True`` disables the marginal fallback.
    """
    _validate_cut(state, cut, direction, strict=strict)
    variables = reduced_gauge_variables(state, cut)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('gauge tolerance must lie between zero and one')
    factors = reduced_frontier_factors(state, cut, direction, environment=environment, strict=strict)
    transforms, inverses, ranks = {}, {}, {}
    decompositions = {}
    for key, factor in factors.items():
        if not np.all(np.isfinite(factor)):
            raise FloatingPointError('nonfinite reduced frontier factor')
        from scipy.linalg import svd
        _, singular, vh = svd(factor, full_matrices=True, lapack_driver='gesvd')
        decompositions[key] = singular, vh
    largest = max((float(s.max(initial=0.)) for s, _ in decompositions.values()), default=0.)
    for key, (singular, vh) in decompositions.items():
        weights = np.ones(vh.shape[0])
        threshold = np.sqrt(max(tolerance, np.finfo(float).eps*len(weights)))*largest
        keep = singular > threshold
        supported = np.flatnonzero(keep)
        weights[supported] = singular[keep]
        vectors = vh.conj().T
        transforms[key] = (vectors/weights)@vectors.conj().T
        inverses[key] = (vectors*weights)@vectors.conj().T
        ranks[key] = int(np.count_nonzero(keep))
    original = [{key: a.copy() for key, a in state.tensors[i].items()} for i in (cut-1, cut)]
    try:
        if direction == 'lr':
            _apply_conditional(state, cut-1, 2, variables, transforms)
            _apply_conditional(state, cut, 0, variables, inverses)
        else:
            # Right environments use bra/ket ordering: transpose, not adjoint.
            _apply_conditional(state, cut, 0, variables,
                               {key: a.T for key, a in transforms.items()})
            _apply_conditional(state, cut-1, 2, variables,
                               {key: a.T for key, a in inverses.items()})
    except Exception:
        state.tensors[cut-1], state.tensors[cut] = original
        raise
    return ranks


def canonicalize_reduced_frontier(state, center, *, tolerance=1e-12, strict=False):
    """Condition both sides using complete shared or legal marginal Grams.

    Packed reduced center coordinates still carry CG norm weights; the local
    solver must retain its generalized metric rather than assume Euclidean I.
    """
    center = index(center)
    if not 0 <= center < state.nsites:
        raise ValueError('center must be a valid site')
    for cut in range(1, state.nsites):
        _validate_cut(state, cut, 'lr', strict=strict)
    original = [{key: a.copy() for key, a in core.items()} for core in state.tensors]
    reports = []
    try:
        for cut in range(1, center+1):
            reports.append(shift_reduced_frontier_gauge(state, cut, 'lr',
                tolerance=tolerance, strict=strict))
        for cut in range(state.nsites-1, center, -1):
            reports.append(shift_reduced_frontier_gauge(state, cut, 'rl',
                tolerance=tolerance, strict=strict))
    except Exception:
        state.tensors = original
        raise
    return tuple(reports)


def condition_reduced_sweep(state, hamiltonian, options, *, center=None,
                            cut=None, direction='lr', context=None):
    """Condition a sweep transactionally, returning a recovery reason or None.

    A caller with moving environments must rebuild that context on recovery.
    Restoring tensors alone cannot undo partially advanced cached boundaries.
    """
    from .reduced_solver import _energy
    from .reduced_updates import NUMERICAL_ERRORS
    snapshot = state.copy()
    before = _energy(snapshot, hamiltonian, stable=True)
    try:
        if center is not None:
            canonicalize_reduced_frontier(state, center, tolerance=options.metric_tolerance)
            changed = list(range(state.nsites))
        else:
            shift_reduced_frontier_gauge(state, cut, direction,
                tolerance=options.metric_tolerance,
                environment=None if context is None else (context.n_chain, context.frontier))
            changed = [cut-1, cut]
        checked = _energy(state, hamiltonian, stable=True)
        allowance = max(options.energy_increase_tolerance, 1e-10*max(1., abs(before)))
        if not np.isfinite(checked) or abs(checked-before) > allowance:
            raise FloatingPointError('reduced gauge changed the physical energy')
        if context is not None:
            context.synchronize(changed)
    except NUMERICAL_ERRORS as error:
        state.tensors, state.bond_sectors = snapshot.tensors, snapshot.bond_sectors
        where = 'initial gauge' if center is not None else 'gauge'
        return f'{where}: {type(error).__name__}: {error}'
    return None

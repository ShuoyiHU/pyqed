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
        frontier = ReducedFrontier.from_state(state, prune=False)
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


class GaugeQRBoundary:
    """One moving, value-validated QR boundary for conditional gauge shifts.

    Adaptation of sector QR (Trefethen and Bau, Numerical Linear Algebra,
    SIAM 1997, doi:10.1137/1.9780898719574). Only the requested side is
    contracted. After a gauge changes its last site, the known transformation
    is applied directly to the retained factor. Boundary contractions use BLAS
    matrix multiplication. No rank cutoff or approximation is added; rounding
    and the boundary row basis can differ from a full rebuild. Only R is retained; no Q is
    needed for the marginal gauge factors. This does not remove the dense
    QR cost of an individual wide frontier.
    """
    def __init__(self, state, direction):
        if direction not in ('lr','rl'):raise ValueError('direction must be lr or rl')
        self.state=state;self.direction=direction;self.site_contractions=0
        self.reset()

    def reset(self):
        self.frontier=ReducedFrontier.from_state(self.state, prune=False)
        self.cut=0 if self.direction=='lr' else self.state.nsites
        sectors=(self.state.left_virtual_sectors(0) if self.direction=='lr'
                 else self.state.right_virtual_sectors(self.state.nsites-1))
        self.boundary={q:np.eye(d) for q,d in Counter(sectors).items()}
        self.incoming=None;self.last_site=None
        self.signature=self._signature()

    def _signature(self):
        import hashlib
        digest=hashlib.blake2b()
        bonds=self.state.bond_sectors[:self.cut] if self.direction=='lr' else self.state.bond_sectors[max(0,self.cut-1):]
        digest.update(repr((self.state.neighborhoods,bonds)).encode())
        indices=range(self.cut) if self.direction=='lr' else range(self.cut,self.state.nsites)
        for i in indices:
            for key,a in self.state.tensors[i].items():
                digest.update(repr((key,a.shape,a.dtype.str)).encode())
                digest.update(np.ascontiguousarray(a).view(np.uint8))
        return digest.digest()

    def _advance(self, site, incoming):
        from collections import defaultdict
        from .frontier_contraction import scatter_blocks
        from .reduced_symmetry import _sector_irrep
        embedding=self.frontier.site_embedding(self.state,site)
        blocks=scatter_blocks(embedding,self.state.tensors[site])
        groups=defaultdict(list)
        lr=self.direction=='lr'
        for key,a in blocks.items():
            if lr:
                dense=np.asarray(a)
                b=(incoming[key[0]]@dense.reshape(dense.shape[0],-1)).reshape(
                    incoming[key[0]].shape[0],dense.shape[1],dense.shape[2])
                groups[key[2]].append(b.reshape(-1,b.shape[2]))
            else:
                width=a.shape[2]
                rows,inverse=np.unique(a.indices//width,return_inverse=True)
                factor=incoming[key[2]]
                if 2*len(rows)<=a.shape[0]*a.shape[1]:
                    selected=np.zeros((len(rows),width),dtype=a.dtype)
                    selected[inverse,a.indices%width]=a.values
                    b=np.zeros((a.shape[0]*a.shape[1],factor.shape[0]),
                               dtype=np.result_type(a.dtype,factor.dtype))
                    b[rows]=selected@factor.T
                    b=b.reshape(a.shape[:2]+(factor.shape[0],))
                else:
                    dense=np.asarray(a)
                    b=(dense.reshape(-1,width)@factor.T).reshape(
                        dense.shape[0],dense.shape[1],factor.shape[0])
                weight=np.sqrt(_sector_irrep(key[2]).dim/_sector_irrep(key[0]).dim)
                groups[key[0]].append(weight*b.transpose(2,1,0).reshape(-1,b.shape[0]))
        sectors=(self.state.right_virtual_sectors(site) if lr else self.state.left_virtual_sectors(site))
        labels=(self.frontier.right_variables(site) if lr else self.frontier.left_variables(site))
        result={q:np.zeros((0,d*self.frontier.reduced_dim**len(labels))) for q,d in Counter(sectors).items()}
        for q,entries in groups.items():result[q]=np.linalg.qr(np.concatenate(entries),mode='r')
        self.site_contractions+=1
        return result

    def at(self, cut):
        cut=index(cut)
        if not 0<=cut<=self.state.nsites:raise IndexError('invalid boundary cut')
        reversed_request=cut<self.cut if self.direction=='lr' else cut>self.cut
        if reversed_request or self.signature!=self._signature():self.reset()
        while self.cut!=cut:
            site=self.cut if self.direction=='lr' else self.cut-1
            self.incoming=self.boundary;self.last_site=site
            self.boundary=self._advance(site,self.incoming)
            self.cut+=1 if self.direction=='lr' else -1
        self.signature=self._signature()
        return self.boundary

    def refresh(self, variables, matrices):
        """Apply the known conditional gauge to the existing square-root factor.

        If the contracted side is Q R and the gauge is G, Q (R G) represents
        the new side exactly. R G need not be triangular: subsequent contractions
        and marginal QR only require its Gram matrix. No rank truncation occurs.
        """
        labels=(self.frontier.right_variables(self.last_site) if self.direction=='lr'
                else self.frontier.left_variables(self.last_site))
        multiplicities=Counter(self.state.bond_sectors[self.cut-1])
        updated={q:np.empty_like(a,dtype=np.result_type(a,*matrices.values()))
                 for q,a in self.boundary.items()}
        for memory,configuration in enumerate(self.frontier._assignments(labels)):
            assignment=dict(zip(labels,configuration))
            shared=tuple(assignment[v] for v in variables)
            for q,d in multiplicities.items():
                section=slice(memory*d,(memory+1)*d)
                updated[q][:,section]=self.boundary[q][:,section]@matrices[(shared,q)]
        self.boundary=updated
        self.signature=self._signature()


def reduced_frontier_factors(state, cut, direction, *, environment=None, strict=False, qr_boundary=None):
    """Square-root factors of the legal marginal Grams, without Gram products.

    Adapts reduced sector QR and standard SVD whitening (Trefethen and Bau,
    Numerical Linear Algebra, SIAM 1997, doi:10.1137/1.9780898719574).
    Stacking conditional column slices implements the marginal partial trace.
    This conditions legal bond gauges only, not the full correlated overlap.
    """
    descriptor = _validate_cut(state, cut, direction, strict=strict)
    variables = reduced_gauge_variables(state, cut)
    boundary_cache=GaugeQRBoundary(state,direction) if qr_boundary is None else qr_boundary
    if boundary_cache.state is not state or boundary_cache.direction!=direction:
        raise ValueError('QR boundary belongs to another state or direction')
    boundary=boundary_cache.at(cut)
    frontier=boundary_cache.frontier
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


def shift_reduced_frontier_gauge(state, cut, direction, *, tolerance=1e-12, environment=None, strict=False, qr_boundary=None):
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
    factors = reduced_frontier_factors(state, cut, direction, environment=environment, strict=strict, qr_boundary=qr_boundary)
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
        if qr_boundary is not None:qr_boundary.refresh(variables, transforms)
    except Exception:
        state.tensors[cut-1], state.tensors[cut] = original
        if qr_boundary is not None:qr_boundary.reset()
        raise
    return ranks


def canonicalize_reduced_frontier(state, center, *, tolerance=1e-12, strict=False, verbosity=0):
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
        left_boundary=GaugeQRBoundary(state,'lr')
        right_boundary=GaugeQRBoundary(state,'rl')
        import time
        for direction,cuts,boundary in [('lr',range(1,center+1),left_boundary),
                                       ('rl',range(state.nsites-1,center,-1),right_boundary)]:
            for cut in cuts:
                start=time.perf_counter()
                if verbosity:print(f'reduced SU(2) initial gauge cut={cut} direction={direction} begin',flush=True)
                reports.append(shift_reduced_frontier_gauge(state,cut,direction,
                    tolerance=tolerance,strict=strict,qr_boundary=boundary))
                if verbosity:print(f'reduced SU(2) initial gauge cut={cut} direction={direction} done seconds={time.perf_counter()-start:.3f}',flush=True)
    except Exception:
        state.tensors = original
        raise
    return tuple(reports)


def condition_reduced_sweep(state, hamiltonian, options, *, center=None,
                            cut=None, direction='lr', context=None, qr_boundary=None):
    """Condition a sweep transactionally, returning a recovery reason or None.

    A caller with moving environments must rebuild that context on recovery.
    Restoring tensors alone cannot undo partially advanced cached boundaries.
    """
    from .reduced_solver import _energy
    from .reduced_updates import NUMERICAL_ERRORS
    snapshot = state.copy()
    before = (context.stable_energy(snapshot) if context is not None else _energy(snapshot, hamiltonian, stable=True))
    try:
        if center is not None:
            canonicalize_reduced_frontier(state, center, tolerance=options.metric_tolerance,
                                          verbosity=options.verbosity)
            changed = list(range(state.nsites))
        else:
            boundary=qr_boundary
            if boundary is None and context is not None:boundary=getattr(context,'gauge_boundary',None)
            if boundary is None or boundary.state is not state or boundary.direction!=direction:
                boundary=GaugeQRBoundary(state,direction)
            shift_reduced_frontier_gauge(state, cut, direction,
                tolerance=options.metric_tolerance, qr_boundary=boundary)
            if context is not None:context.gauge_boundary=boundary
            changed = [cut-1, cut]
        certify = getattr(context, 'certify_gauge', None)
        if center is None and certify is not None and certify(snapshot, cut):
            # The physical state was checked directly in orthonormal exterior
            # coordinates; a second Hamiltonian contraction is redundant.
            context.remember_source_energy(state, before)
            checked = before
        else:
            checked = (context.stable_energy() if context is not None else _energy(state, hamiltonian, stable=True))
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

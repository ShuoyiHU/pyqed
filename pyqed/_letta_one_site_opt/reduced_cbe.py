"""Controlled expansion with native reduced pair residuals and legal factors.

The selector removes the incumbent one-site tangent space in the physical norm.
Independent sector fits allocate complete multiplets to the residual.
Cached rank profiles reuse the shared two-site metric compression routine.
Only an expanded *one-site* Hamiltonian is diagonalized. Pair actions and pair
coefficient vectors are materialized. The OBC adapter avoids a dense pair
metric; the shared selector accepts other topology-specific metric roots.
A reduced tangent matrix is materialized only within the configured workspace
budget; larger projections remain matrix-free.
"""
from collections import Counter
from dataclasses import dataclass, replace
from operator import index

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .reduced_updates import NUMERICAL_ERRORS, one_site_options, reduced_stationarity
from .._letta_two_site_opt.reduced_compression import ReducedPairMetricRoot, compress_reduced_pair


@dataclass(frozen=True)
class ReducedCBESelection:
    scaffold: object
    problem: object
    multiplicities: dict
    left: dict
    right: dict
    missing_norm: float
    tangent_overlap: float
    projection_iterations: int
    projection_converged: bool
    captured_weight: float
    loss: float
    diagnostics: tuple
    projection_preconditioned: bool = False
    projection_solver: str = 'lsmr'
    projection_rank: int | None = None
    projection_materialized: bool = False
    projection_fallback_reason: str | None = None
    factor_fit_attempts: int = 0
    pruned_allocations: int = 0


def select_reduced_cbe(state, hamiltonian, left_site, options):
    """Select residual factors without a two-site energy minimization.

    ``exact`` denotes native, unapproximated pair actions, not a globally
    optimal low-rank factorization. The latter is a nonconvex metric fit; its
    solver, residual and iteration status are returned explicitly.

    Residual expansion adapts Gleis, Li, and von Delft, PRL 130, 246402
    (2023), doi:10.1103/PhysRevLett.130.246402; this is not their shrewd
    selector. Orthogonal-sector rank profiles cache the shared two-site
    compression solves and retain per-sector convergence diagnostics.

    The open-boundary tangent projection adapts reduced QR/SVD orthogonal
    projection (Trefethen and Bau, Numerical Linear Algebra, SIAM 1997,
    doi:10.1137/1.9780898719574). A bounded direct SVD avoids near-dependent
    tangent iterations; larger maps use physical-coordinate LSMR. Numerical
    rank cutoffs remain explicit. When no expansion is selected, alternating
    fixed-bond Rayleigh minimization may relax the neighboring site, under
    the same energy guard and configured refinement budget. This hybrid is
    a LETTA adaptation, not an exact reproduction of a published CBE method
    or a guarantee of global variational convergence.
    """
    from .._letta_two_site_opt.reduced_solver import (
        reduced_pair_problem, _expand_reduced_pair_space,
        _pair_vector_from_sources, _left_source_adjoint, _right_source_adjoint,
        _expanded_source_blocks)
    budget = index(options.cbe_expansion_dimension)
    if budget < 1 or options.cbe_selector != 'exact':
        raise ValueError('reduced CBE requires exact selector and a positive expansion budget')
    old = Counter(state.bond_sectors[left_site])
    scaffold = _expand_reduced_pair_space(state, left_site, sum(old.values())+budget)
    problem = reduced_pair_problem(scaffold, hamiltonian, left_site,
                                   matrix_free=True, dense_solver_threshold=0)
    root = ReducedPairMetricRoot(problem, scaffold, options.metric_tolerance)
    le, re = (problem.frontier.site_embedding(scaffold, k) for k in (left_site, left_site+1))
    a, b = le.pack_source(scaffold.tensors[left_site]), re.pack_source(scaffold.tensors[left_site+1])
    ablocks, bblocks = _expanded_source_blocks(le, a), _expanded_source_blocks(re, b)
    def adjoint(side, vector):
        return (_left_source_adjoint(problem.layout, vector, bblocks, le) if side == 0 else
                _right_source_adjoint(problem.layout, ablocks, vector, re))
    def compress(target, u, v, allocation):
        return compress_reduced_pair(target, problem, scaffold, u, v, allocation,
            options=options.compression, metric_tolerance=options.metric_tolerance,
            als_max_iterations=options.cbe_refinement_max_iterations, root=root)
    return select_metric_cbe(scaffold, problem, root, old, Counter(scaffold.bond_sectors[left_site]),
        (le, re), (a, b), options=options,
        merge=lambda u, v: _pair_vector_from_sources(problem.layout, le, re, u, v),
        factor_adjoint=adjoint, compress=compress, seed=2817+left_site,
        complex_data=any(np.iscomplexobj(v) for tensor in state.tensors for v in tensor.values()),
        tangent_coordinates=lambda: _tangent_coordinates(scaffold, problem, (le, re), old, options),
        physical_tangent=True,
        residual_fit=lambda target, available: _fit_residual_pair(
            scaffold, problem, target, available, (le, re), options),
        weighted_residual=(lambda: _canonical_pair_residual(problem, scaffold, hamiltonian, root))
            if hamiltonian.contraction_backend == "reduced" else None,
        allocation_bounds=lambda target, allocations: _sector_loss_bounds(root, target, allocations))


def _fit_residual_pair(state, problem, target, capacities, embeddings, options):
    """Cache orthogonal-sector rank fits instead of refitting full allocations.

    Pair metric blocks preserve the middle irrep. Consequently the residual
    loss is a sum over sectors and previously fitted sectors do not change
    when one sector gains a multiplet. Greedy rank allocation uses measured
    metric losses, not raw singular weights. Fits remain nonconvex; a warm
    start supplements two independent starts at each new rank.
    """
    from .._letta_two_site_opt.reduced_solver import _active_source_indices
    le, re = embeddings
    rng = np.random.default_rng(2817+problem.left_site)
    seeds = [(rng.normal(size=le.source_size).astype(complex),
              rng.normal(size=re.source_size).astype(complex)) for _ in range(2)]
    if any(np.iscomplexobj(v) for tensor in state.tensors for v in tensor.values()):
        seeds = [(a+1j*rng.normal(size=a.size), b+1j*rng.normal(size=b.size))
                 for a,b in seeds]
    root = ReducedPairMetricRoot(problem,state,options.metric_tolerance)
    weights = Counter()
    for key, block in root.physical_layout.unpack(root.physical_apply(target)).items():
        weights[key[2]] += float(np.linalg.norm(block)**2)
    total = sum(weights.values())
    target_blocks = problem.layout.unpack(target)
    sector_problems = {}
    cache = {}
    attempts = 0
    def profile(q, rank):
        nonlocal attempts
        key = (q, rank)
        if key in cache:
            return cache[key]
        allocation = {q: rank}
        if q not in sector_problems:
            local_root = root.restrict_middle_sector(q)
            local_target = local_root.input_layout.pack({k:target_blocks[k]
                                                        for k in local_root.input_layout.shapes})
            local_problem = replace(problem,layout=local_root.input_layout,
                old_vector=np.zeros(local_root.input_layout.size,dtype=target.dtype),
                hamiltonian=None,metric=None,source_frame=None,hamiltonian_action=None,
                metric_action=None,equilibrated_projector_factory=None)
            sector_problems[q]=(local_problem,local_root,local_target)
        local_problem,local_root,local_target=sector_problems[q]
        starts = list(seeds)
        previous = cache.get((q, rank-1))
        if previous is not None:
            a, b = previous.left.copy(), previous.right.copy()
            old = _active_source_indices(re, {q:rank-1}, 'right')
            new = _active_source_indices(re, allocation, 'right')
            extra = np.setdiff1d(new, old)
            b[extra] = seeds[0][1][extra]
            starts.insert(0, (a, b))
        fits = []
        for a, b in starts:
            fit = compress_reduced_pair(local_target,local_problem,state,a,b,allocation,
                options=options.compression,metric_tolerance=options.metric_tolerance,
                als_max_iterations=options.cbe_refinement_max_iterations,root=local_root)
            fit = replace(fit,loss=fit.loss+total-weights[q])
            attempts += 1
            if np.isfinite(fit.loss):
                fits.append(fit)
        cache[key] = min(fits, key=lambda fit:fit.loss) if fits else None
        return cache[key]
    retained = {}
    chosen = {}
    for _ in range(options.cbe_expansion_dimension):
        def upper_bound(q):
            return (chosen[q].loss-total+weights[q]) if q in chosen else weights[q]
        best = None
        for q in sorted(capacities,key=upper_bound,reverse=True):
            rank = retained.get(q, 0)
            if rank >= capacities[q]:continue
            bound = upper_bound(q)
            if bound <= (0. if best is None else best[0])-100*np.finfo(float).eps:
                continue
            if bound <= 0.:continue
            fit = profile(q, rank+1)
            if fit is None:continue
            old_loss = chosen[q].loss if q in chosen else total
            gain = old_loss-fit.loss
            if best is None or gain > best[0]:best=(gain,q,fit)
        if best is None or best[0] <= 0.:break
        gain,q,fit = best
        retained[q] = retained.get(q,0)+1
        chosen[q] = fit
    if not chosen:return retained,None,attempts
    left = sum((fit.left for fit in chosen.values()), np.zeros(le.source_size,complex))
    right = sum((fit.right for fit in chosen.values()), np.zeros(re.source_size,complex))
    # Recompute the joint physical loss rather than subtract nearly equal
    # omitted-sector constants from independently reported losses.
    from .._letta_two_site_opt.reduced_solver import _pair_vector_from_sources
    loss=float(np.linalg.norm(root.physical_apply(
        _pair_vector_from_sources(problem.layout,le,re,left,right)-target))**2)
    from .._letta_compression import CompressionResult
    reports=[dict(fit.diagnostics,sector=str(q),rank=retained[q],iterations=fit.iterations)
             for q,fit in chosen.items()]
    info=dict(requested_solver=options.compression.solver,used_solver=options.compression.solver,
        allocation_method='cached-sector-metric-fits',sector_fits=reports,
        optimizer_success=all(fit.diagnostics.get('optimizer_success',False) for fit in chosen.values()),
        final_loss=loss,rank_profiles=len(cache))
    return retained,CompressionResult(left,right,loss,max(fit.iterations for fit in chosen.values()),info),attempts


def _sector_loss_bounds(root, target, allocations):
    """Exact lower bounds from mutually orthogonal intermediate sectors.

    A product retaining only sectors in an allocation is zero in all other
    intermediate-sector blocks. Their physical target norm is therefore an
    unavoidable loss, irrespective of the nonlinear factor fit's quality.
    The open reduced QR metric preserves this block separation.
    """
    from collections import defaultdict
    weights = defaultdict(float)
    for key, block in root.physical_layout.unpack(root.physical_apply(target)).items():
        weights[key[2]] += float(np.linalg.norm(block)**2)
    return [sum(weight for q, weight in weights.items() if not allocation.get(q, 0))
            for allocation in allocations]


def _canonical_pair_residual(problem, state, hamiltonian, root):
    """Evaluate the reduced Hamiltonian residual directly in QR coordinates.

    This uses the same reduced MPO action as the raw pair problem, with its
    external blocks orthonormalized first. No determinant projection is used.
    Applying H in raw coordinates and then multiplying by inverse overlap
    factors can amplify contraction roundoff into spurious CBE directions.
    """
    from .reduced_environment import ReducedEnvironmentChain
    from .reduced_solver import _checked_energy
    from .._letta_two_site_opt.reduced_solver import _merge_pair_blocks
    i = problem.left_site
    pair = _merge_pair_blocks(root.canonical_sites[i], root.canonical_sites[i+1])
    y = root.physical_layout.pack({key: value*root.physical_factors[key][2]
                                   for key, value in pair.items()})
    blocks = root.physical_layout.unpack(y)
    chain = getattr(root,'environment',None)
    if chain is None:
        chain = ReducedEnvironmentChain.build(root.canonical_sites,
            hamiltonian.native_mpo(state.physical_basis), lazy=True)
    applied = chain.pair_action(problem.left_site,
        {key: value/root.physical_factors[key][2] for key, value in blocks.items()})
    hy = root.physical_layout.pack({key: value/root.physical_factors[key][2]
                                   for key, value in applied.items()})
    energy = _checked_energy(np.vdot(y, hy), np.vdot(y, y))
    return root.project_physical(hy-energy*y)


def _tangent_coordinates(scaffold, problem, embeddings, old, options):
    """Whiten each incumbent tangent factor using the one-site metric cutoff.

    This is QR/SVD preconditioning of a least-squares projection, not an
    expansion of its tangent space. New bond entries remain masked out.
    """
    from .reduced_coordinates import ReducedLocalCoordinates
    from .._letta_two_site_opt.reduced_solver import _active_source_indices
    sites = problem.frontier.to_mps(scaffold, compact=True)
    result = []
    for side, embedding in enumerate(embeddings):
        active = _active_source_indices(embedding, old, 'left' if side == 0 else 'right')
        coordinates = ReducedLocalCoordinates(sites, embedding, problem.left_site+side, retain_outer=False)
        coordinates.restrict(active)
        coordinates.prepare(options.metric_tolerance)
        result.append(coordinates)
    return result


def select_metric_cbe(scaffold, problem, root, old, capacities, embeddings, factors, *,
                      merge, factor_adjoint, compress, options, seed, complex_data, tangent_coordinates=None,
                      weighted_residual=None, allocation_bounds=None, physical_tangent=False, residual_fit=None):
    """Common physical-metric residual projection and whole-sector selection.

    Topology-specific adapters supply the true overlap root and source maps.
    The pair eigensolver is deliberately absent from this interface.
    """
    from .._letta_two_site_opt.reduced_solver import _active_source_indices
    budget = index(options.cbe_expansion_dimension)
    if budget < 1 or options.cbe_selector != 'exact':
        raise ValueError('reduced CBE requires exact selector and a positive expansion budget')
    x = problem.old_vector
    if weighted_residual is None:
        hx = problem.apply_hamiltonian(x)
        physical = getattr(root, "physical_apply", root.apply)
        physical_adjoint = getattr(root, "physical_adjoint", root.adjoint)
        weighted_x = physical(x)
        nx = physical_adjoint(weighted_x)
        norm = float(np.vdot(weighted_x, weighted_x).real)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError('invalid reduced CBE norm')
        energy = np.vdot(x, hx).real/norm
        weighted_gradient = root.unwhiten_adjoint(hx-energy*nx)
    else:
        weighted_gradient = weighted_residual()
    le, re = embeddings
    a, b = factors
    li, ri = _active_source_indices(le, old, 'left'), _active_source_indices(re, old, 'right')
    dtype = np.result_type(a, b, weighted_gradient, complex)

    def tangent(v):
        da, db = np.zeros(a.size, dtype=dtype), np.zeros(b.size, dtype=dtype)
        da[li], db[ri] = v[:len(li)], v[len(li):]
        return root.apply(merge(da, b)+merge(a, db))

    def adjoint(w):
        v = root.adjoint(w)
        return np.concatenate((factor_adjoint(0, v)[li],
                               factor_adjoint(1, v)[ri]))

    jacobian = LinearOperator((root.size, len(li)+len(ri)), matvec=tangent,
                             rmatvec=adjoint, dtype=dtype)
    preconditioned = False
    direct_projection = None
    projection_rank = None
    projection_materialized = False
    projection_info = {}
    if tangent_coordinates is not None:
        maps = tangent_coordinates()
        rank = maps[0].rank
        def expand(z):
            return np.concatenate((maps[0].expand(z[:rank])[li], maps[1].expand(z[rank:])[ri]))
        def pullback(v):
            u, w = np.zeros_like(a, dtype=dtype), np.zeros_like(b, dtype=dtype)
            u[li], w[ri] = v[:len(li)], v[len(li):]
            return np.concatenate((maps[0].adjoint(u), maps[1].adjoint(w)))
        preconditioner = LinearOperator((len(li)+len(ri),sum(m.rank for m in maps)),
            matvec=expand, rmatvec=pullback, dtype=dtype)
        jacobian = jacobian@preconditioner
        if physical_tangent:
            from .._letta_two_site_opt.reduced_compression import _physical_factor_map
            physical_maps=[_physical_factor_map(c,root,problem.left_site,side) for side,c in enumerate(maps)]
            def forward(z):
                return root.project_physical(physical_maps[0][0](z[:rank])+physical_maps[1][0](z[rank:]))
            def reverse(w):
                full=root.embed_physical(w)
                return np.concatenate([m[1](full) for m in physical_maps])
            jacobian=LinearOperator(jacobian.shape,matvec=forward,rmatvec=reverse,dtype=dtype)
            from .tangent_projection import project_tangent
            direct_projection, projection_info = project_tangent(maps, root,
                problem.left_site, weighted_gradient,
                workspace_mb=options.cbe_projection_max_workspace_mb,
                rank_tolerance=options.compression.pinv_tolerance,
                normal_tolerance=options.cbe_projection_tolerance)
            projection_rank = projection_info['rank']
            projection_materialized = projection_info['materialized']
        preconditioned = True
    if direct_projection is None:
        projection = lsmr(jacobian, weighted_gradient.astype(dtype,copy=False),
            atol=options.cbe_projection_tolerance, btol=options.cbe_projection_tolerance,
            maxiter=options.cbe_projection_max_iterations)
        iterations = int(projection[2])
        missing = weighted_gradient-jacobian@projection[0]
        converged = projection[1] in (0, 1, 2, 4, 5)
    else:
        missing=direct_projection
        iterations=0
        converged=True
    overlap = float(np.linalg.norm(adjoint(missing)))
    if not converged or not np.all(np.isfinite(missing)):
        raise FloatingPointError(f'reduced CBE tangent projection did not converge (LSMR {projection[1]})')
    missing_norm = float(np.linalg.norm(missing))
    empty = dict(scaffold=scaffold, problem=problem, missing_norm=missing_norm,
        tangent_overlap=overlap, projection_iterations=iterations,
        projection_converged=converged, projection_preconditioned=preconditioned,
        projection_solver='svd' if direct_projection is not None else 'lsmr',
        projection_rank=projection_rank, projection_materialized=projection_materialized,
        projection_fallback_reason=projection_info.get('fallback_reason'))
    if missing_norm <= options.cbe_selection_tolerance:
        return ReducedCBESelection(**empty, multiplicities={}, left={}, right={},
                                   captured_weight=0., loss=missing_norm**2, diagnostics=())
    # Separate residual magnitude from its direction so absolute least-squares
    # stopping tolerances do not change the selected subspace near convergence.
    target = root.unwhiten(missing/missing_norm)
    available = {q: capacities[q]-old[q] for q in capacities if capacities[q] > old[q]}
    if residual_fit is not None:
        retained, best, attempts = residual_fit(target, available)
        if best is None or 1.-best.loss <= (options.cbe_selection_tolerance/missing_norm)**2:
            return ReducedCBESelection(**empty, multiplicities={}, left={}, right={},
                captured_weight=0., loss=missing_norm**2, diagnostics=(),
                factor_fit_attempts=attempts)
        report = {**best.diagnostics, 'iterations': best.iterations,
            'normalized_residual_objective': True, 'residual_scale': missing_norm,
            'factor_fit_attempts': attempts}
        return ReducedCBESelection(**empty, multiplicities=retained,
            left=le.unpack_source(best.left*np.sqrt(missing_norm)),
            right=re.unpack_source(best.right*np.sqrt(missing_norm)),
            captured_weight=missing_norm**2*max(0., 1.-best.loss),
            loss=missing_norm**2*best.loss, diagnostics=(report,),
            factor_fit_attempts=attempts)
    retained, best, reports = {}, None, []
    best_loss = 1.
    rng = np.random.default_rng(seed)
    # Legal source-coordinate starts include every tie label. Projecting a
    # frontier SVD alone can erase those labels and trap ALS at zero factors.
    seeds = [(rng.normal(size=a.size).astype(dtype), rng.normal(size=b.size).astype(dtype))
             for _ in range(2)]
    if complex_data:
        seeds = [(u+1j*rng.normal(size=u.size), v+1j*rng.normal(size=v.size)) for u, v in seeds]
    attempts, pruned = 0, 0
    for _ in range(budget):
        trials = []
        allocations = [{**retained, q: retained.get(q, 0)+1} for q in available
                       if retained.get(q, 0) < available[q]]
        bounds = ([0.]*len(allocations) if allocation_bounds is None
                  else allocation_bounds(target, allocations))
        trial_best = best_loss
        for trial_index in sorted(range(len(allocations)), key=lambda j: bounds[j]):
            allocation, bound = allocations[trial_index], bounds[trial_index]
            if bound > trial_best+100*np.finfo(float).eps*max(1., abs(trial_best)):
                pruned += 1
                continue
            for u, v in seeds:
                attempts += 1
                fit = compress(target, u, v, allocation)
                if np.isfinite(fit.loss):
                    trials.append((fit.loss, allocation, fit))
                    trial_best = min(trial_best, fit.loss)
        if not trials:
            break
        loss, allocation, fit = min(trials, key=lambda trial: trial[0])
        if best_loss-loss <= (options.cbe_selection_tolerance/missing_norm)**2:
            break
        retained, best, best_loss = allocation, fit, loss
        reports.append({**fit.diagnostics, 'iterations': fit.iterations,
                        'normalized_residual_objective': True, 'residual_scale': missing_norm})
    return ReducedCBESelection(**empty, multiplicities=retained,
        left={} if best is None else le.unpack_source(best.left*np.sqrt(missing_norm)),
        right={} if best is None else re.unpack_source(best.right*np.sqrt(missing_norm)),
        captured_weight=missing_norm**2*max(0., 1.-best_loss), loss=missing_norm**2*best_loss,
        diagnostics=tuple(reports), factor_fit_attempts=attempts, pruned_allocations=pruned)


def expand_reduced_cbe(state, selection, direction):
    """Append complete multiplets and a zero partner, preserving the state."""
    if direction not in {'lr', 'rl'}:
        raise ValueError('CBE direction must be lr or rl')
    if not selection.multiplicities:
        return state.copy()
    i = selection.problem.left_site
    old = Counter(state.bond_sectors[i])
    new = old+Counter(selection.multiplicities)
    result = state.copy()
    bonds = list(result.bond_sectors)
    bonds[i] = tuple(q for q in sorted(new) for _ in range(new[q]))
    result.bond_sectors = tuple(bonds)
    for side, source in enumerate((selection.left, selection.right)):
        tensors = {}
        for key, template in selection.scaffold.tensors[i+side].items():
            q, axis = (key[2], -1) if side == 0 else (key[0], 0)
            if not new[q]:
                continue
            shape = list(template.shape); shape[axis] = new[q]
            block = np.zeros(shape, dtype=np.result_type(template, source.get(key, 0.), complex))
            if key in state.tensors[i+side]:
                section = [slice(None)]*block.ndim; section[axis] = slice(0, old[q])
                block[tuple(section)] = state.tensors[i+side][key]
            if (side == 1 and direction == 'lr') or (side == 0 and direction == 'rl'):
                count = selection.multiplicities.get(q, 0)
                if count:
                    dst = [slice(None)]*block.ndim; dst[axis] = slice(old[q], new[q])
                    src = [slice(None)]*block.ndim; src[axis] = slice(0, count)
                    block[tuple(dst)] = source[key][tuple(src)]
            tensors[key] = block
        result.tensors[i+side] = tensors
    result.tensors = result._validate_tensors(result.tensors)
    return result


def _trim_options(options):
    from .._letta_two_site_opt.solver import LETTATwoSiteOptions
    names = ('compression', 'metric_tolerance', 'energy_increase_tolerance',
             'eigensolver_tolerance', 'eigensolver_max_iterations',
             'dense_solver_threshold', 'matrix_free', 'gauge_mode')
    return LETTATwoSiteOptions(**{k: getattr(options, k) for k in names},
        truncation_max_iterations=options.cbe_refinement_max_iterations,
        energy_refinement_max_iterations=options.cbe_energy_refinement_max_iterations,
        energy_refinement_tolerance=options.cbe_energy_refinement_tolerance)


def reduced_cbe_site(state, hamiltonian, site, direction, bond_dim, options):
    """Ordinary baseline → residual expansion → one-site solve → guarded trim.

    A numerical failure discards the entire expansion candidate and commits the
    independently computed ordinary step. The next site can attempt CBE again.
    Trimming preserves at least one multiplet of each incumbent bond sector.
    This constrained LETTA adaptation prevents norm-ranked compression from
    deleting a sector needed by later one-site updates; multiplicities may still
    change within the fixed total budget. It is not globally optimal compression
    or a guarantee of convergence to the best fixed-D energy.
    """
    from .reduced_solver import StableEnergyCache, optimize_reduced_site
    from .._letta_two_site_opt.reduced_solver import reduced_pair_problem, compress_reduced_pair_vector
    if direction not in {'lr', 'rl'}:
        raise ValueError('CBE direction must be lr or rl')
    energy_cache = StableEnergyCache(hamiltonian)
    before = energy_cache(state)
    baseline = state.copy()
    try:
        ordinary = optimize_reduced_site(baseline, hamiltonian, site, one_site_options(options), energy_cache=energy_cache)
        base_energy = energy_cache(baseline)
    except NUMERICAL_ERRORS as error:
        from .solver import LETTASiteUpdate
        return LETTASiteUpdate(site=site, local_energy=before, energy=before,
            metric_rank=0, local_dimension=0, residual_norm=np.inf, accepted=False,
            local_converged=False, cbe_selector='exact', cbe_old_energy=before,
            cbe_fallback=True, cbe_recovery_rejected=True,
            cbe_recovery_reason=f'one-site baseline failed: {type(error).__name__}: {error}')
    update = replace(ordinary, cbe_selector='exact', cbe_old_energy=before,
        cbe_baseline_energy=base_energy, cbe_baseline_allowance=0., cbe_baseline_selected=True)
    i = site if direction == 'lr' else site-1
    chosen = baseline
    if 0 <= i < state.nsites-1:
        try:
            selection = select_reduced_cbe(baseline.copy(), hamiltonian, i, options)
            update = replace(update, cbe_expansion_dimension=sum(selection.multiplicities.values()),
                cbe_projection_iterations=selection.projection_iterations,
                cbe_projection_converged=selection.projection_converged,
                cbe_pair_dimension=selection.problem.local_dimension,
                cbe_materialized_pair_tensor=True, cbe_materialized_pair_metric=False,
                cbe_materialized_tangent_jacobian=selection.projection_materialized,
                cbe_missing_norm=selection.missing_norm, cbe_captured_weight=selection.captured_weight,
                cbe_selection_loss=selection.loss,
                cbe_selection_diagnostics=dict(tangent_overlap=selection.tangent_overlap,
                    projection_preconditioned=selection.projection_preconditioned,
                    projection_solver=selection.projection_solver,
                    projection_rank=selection.projection_rank,
                    projection_fallback_reason=selection.projection_fallback_reason,
                    factor_fits=selection.diagnostics,
                    factor_fit_attempts=selection.factor_fit_attempts,
                    pruned_allocations=selection.pruned_allocations,
                    allocation=tuple((str(q), n) for q, n in selection.multiplicities.items())))
            if selection.multiplicities:
                candidate = expand_reduced_cbe(baseline, selection, direction)
                expanded = optimize_reduced_site(candidate, hamiltonian, site, one_site_options(options), energy_cache=energy_cache)
                expanded_energy = energy_cache(candidate)
                update = replace(update, cbe_expanded_energy=expanded_energy)
                expansion_guard = 100*np.finfo(float).eps*max(1., abs(base_energy), abs(expanded_energy))
                if expanded_energy < base_energy-expansion_guard:
                    problem = reduced_pair_problem(candidate, hamiltonian, i,
                        matrix_free=True, dense_solver_threshold=0)
                    trim = compress_reduced_pair_vector(candidate, hamiltonian, problem,
                        problem.old_vector, expanded.local_energy, expanded.metric_rank,
                        expanded.residual_norm, direction, bond_dim, _trim_options(options),
                        acceptance_energy=base_energy,
                        minimum_multiplicities={q: 1 for q in baseline.bond_sectors[i]})
                    energy = energy_cache(candidate)
                    update = replace(update, cbe_trimmed_energy=trim.energy_refinement_initial_energy,
                        cbe_refined_energy=trim.energy_refinement_energy,
                        cbe_trim_loss=trim.metric_truncation_loss,
                        cbe_trim_method='reduced-physical-metric',
                        cbe_compression_diagnostics=(() if trim.compression_diagnostics is None
                                                     else ({**trim.compression_diagnostics, 'iterations': trim.truncation_iterations},)),
                        cbe_energy_refinement_iterations=trim.energy_refinement_iterations,
                        cbe_energy_refinement_substeps=trim.energy_refinement_accepted_substeps)
                    guard = 100*np.finfo(float).eps*max(1., abs(base_energy), abs(energy))
                    if trim.accepted and energy < base_energy-guard and len(candidate.bond_sectors[i]) <= bond_dim:
                        audit = reduced_stationarity(candidate, hamiltonian, (site,), options)[0]
                        update = replace(update, energy=energy, accepted=True, cbe_baseline_selected=False,
                            residual_norm=audit['residual_norm'], relative_residual=audit['relative_residual'],
                            local_converged=audit['relative_residual'] <= options.eigensolver_tolerance)
                        chosen = candidate
            else:
                neighbor = i+1 if direction == 'lr' else i
                stationarity = reduced_stationarity(baseline, hamiltonian, (neighbor,), options)[0]
                if stationarity['relative_residual'] > options.eigensolver_tolerance:
                    from .reduced_updates import refine_reduced_pair_energy
                    refinement = refine_reduced_pair_energy(baseline, hamiltonian, i, options,
                        max_iterations=options.cbe_energy_refinement_max_iterations,
                        tolerance=options.cbe_energy_refinement_tolerance, direction=direction)
                    energy = energy_cache(refinement.state)
                    update = replace(update, cbe_refined_energy=energy,
                        cbe_trim_method='fixed-bond-energy-relaxation',
                        cbe_energy_refinement_iterations=refinement.iterations,
                        cbe_energy_refinement_substeps=refinement.accepted_substeps)
                    guard = 100*np.finfo(float).eps*max(1., abs(base_energy), abs(energy))
                    if energy < base_energy-guard:
                        audit = reduced_stationarity(refinement.state, hamiltonian, (site,), options)[0]
                        update = replace(update, energy=energy, accepted=True, cbe_baseline_selected=False,
                            residual_norm=audit['residual_norm'], relative_residual=audit['relative_residual'],
                            local_converged=audit['relative_residual'] <= options.eigensolver_tolerance)
                        chosen = refinement.state
        except NUMERICAL_ERRORS as error:
            update = replace(update, energy=base_energy, accepted=ordinary.accepted,
                local_converged=False, cbe_baseline_selected=True, cbe_fallback=True,
                cbe_recovery_reason=f'{type(error).__name__}: {error}')
    state.tensors, state.bond_sectors = chosen.tensors, chosen.bond_sectors
    return update

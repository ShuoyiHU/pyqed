"""Controlled expansion with native reduced pair residuals and legal factors.

The selector removes the incumbent one-site tangent space in the physical norm.
A greedy metric fit allocates complete multiplets to the remaining residual.
Only an expanded *one-site* Hamiltonian is diagonalized. Pair actions and pair
coefficient vectors are materialized. The OBC adapter avoids a dense pair
metric; the shared selector accepts other topology-specific metric roots.
No tangent Jacobian is materialized.
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


def select_reduced_cbe(state, hamiltonian, left_site, options):
    """Select residual factors without a two-site energy minimization.

    ``exact`` denotes native, unapproximated pair actions, not a globally
    optimal low-rank factorization. The latter is a nonconvex metric fit; its
    solver, residual and iteration status are returned explicitly.
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
            als_max_iterations=options.cbe_refinement_max_iterations)
    return select_metric_cbe(scaffold, problem, root, old, Counter(scaffold.bond_sectors[left_site]),
        (le, re), (a, b), options=options,
        merge=lambda u, v: _pair_vector_from_sources(problem.layout, le, re, u, v),
        factor_adjoint=adjoint, compress=compress, seed=2817+left_site,
        complex_data=any(np.iscomplexobj(v) for tensor in state.tensors for v in tensor.values()))


def select_metric_cbe(scaffold, problem, root, old, capacities, embeddings, factors, *,
                      merge, factor_adjoint, compress, options, seed, complex_data):
    """Common physical-metric residual projection and whole-sector selection.

    Topology-specific adapters supply the true overlap root and source maps.
    The pair eigensolver is deliberately absent from this interface.
    """
    from .._letta_two_site_opt.reduced_solver import _active_source_indices
    budget = index(options.cbe_expansion_dimension)
    if budget < 1 or options.cbe_selector != 'exact':
        raise ValueError('reduced CBE requires exact selector and a positive expansion budget')
    x = problem.old_vector
    hx, nx = problem.apply_hamiltonian(x), problem.apply_metric(x)
    norm = np.vdot(x, nx).real
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('invalid reduced CBE norm')
    energy = np.vdot(x, hx).real/norm
    weighted_gradient = root.unwhiten_adjoint(hx-energy*nx)
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
    projection = lsmr(jacobian, weighted_gradient.astype(dtype, copy=False), atol=options.cbe_projection_tolerance,
        btol=options.cbe_projection_tolerance, maxiter=options.cbe_projection_max_iterations)
    missing = weighted_gradient-jacobian@projection[0]
    overlap = float(np.linalg.norm(adjoint(missing)))
    converged = projection[1] in (0, 1, 2, 4, 5)
    if not converged or not np.all(np.isfinite(missing)):
        raise FloatingPointError(f'reduced CBE tangent projection did not converge (LSMR {projection[1]})')
    missing_norm = float(np.linalg.norm(missing))
    empty = dict(scaffold=scaffold, problem=problem, missing_norm=missing_norm,
        tangent_overlap=overlap, projection_iterations=int(projection[2]),
        projection_converged=converged)
    if missing_norm <= options.cbe_selection_tolerance:
        return ReducedCBESelection(**empty, multiplicities={}, left={}, right={},
                                   captured_weight=0., loss=missing_norm**2, diagnostics=())
    target = root.unwhiten(missing)
    available = {q: capacities[q]-old[q] for q in capacities if capacities[q] > old[q]}
    retained, best, reports = {}, None, []
    best_loss = missing_norm**2
    rng = np.random.default_rng(seed)
    # Legal source-coordinate starts include every tie label. Projecting a
    # frontier SVD alone can erase those labels and trap ALS at zero factors.
    seeds = [(rng.normal(size=a.size).astype(dtype), rng.normal(size=b.size).astype(dtype))
             for _ in range(2)]
    if complex_data:
        seeds = [(u+1j*rng.normal(size=u.size), v+1j*rng.normal(size=v.size)) for u, v in seeds]
    for _ in range(budget):
        trials = []
        for q in available:
            if retained.get(q, 0) >= available[q]:
                continue
            allocation = {**retained, q: retained.get(q, 0)+1}
            for u, v in seeds:
                fit = compress(target, u, v, allocation)
                if np.isfinite(fit.loss):
                    trials.append((fit.loss, allocation, fit))
        if not trials:
            break
        loss, allocation, fit = min(trials, key=lambda trial: trial[0])
        if best_loss-loss <= options.cbe_selection_tolerance**2:
            break
        retained, best, best_loss = allocation, fit, loss
        reports.append({**fit.diagnostics, 'iterations': fit.iterations})
    return ReducedCBESelection(**empty, multiplicities=retained,
        left={} if best is None else le.unpack_source(best.left),
        right={} if best is None else re.unpack_source(best.right),
        captured_weight=max(0., missing_norm**2-best_loss), loss=best_loss,
        diagnostics=tuple(reports))


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
    """
    from .reduced_solver import _energy, optimize_reduced_site
    from .._letta_two_site_opt.reduced_solver import reduced_pair_problem, compress_reduced_pair_vector
    if direction not in {'lr', 'rl'}:
        raise ValueError('CBE direction must be lr or rl')
    before = _energy(state, hamiltonian, stable=True)
    baseline = state.copy()
    try:
        ordinary = optimize_reduced_site(baseline, hamiltonian, site, one_site_options(options))
        base_energy = _energy(baseline, hamiltonian, stable=True)
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
                cbe_materialized_tangent_jacobian=False,
                cbe_missing_norm=selection.missing_norm, cbe_captured_weight=selection.captured_weight,
                cbe_selection_loss=selection.loss,
                cbe_selection_diagnostics=dict(tangent_overlap=selection.tangent_overlap,
                    factor_fits=selection.diagnostics,
                    allocation=tuple((str(q), n) for q, n in selection.multiplicities.items())))
            if selection.multiplicities:
                candidate = expand_reduced_cbe(baseline, selection, direction)
                expanded = optimize_reduced_site(candidate, hamiltonian, site, one_site_options(options))
                expanded_energy = _energy(candidate, hamiltonian, stable=True)
                update = replace(update, cbe_expanded_energy=expanded_energy)
                problem = reduced_pair_problem(candidate, hamiltonian, i,
                    matrix_free=True, dense_solver_threshold=0)
                trim = compress_reduced_pair_vector(candidate, hamiltonian, problem,
                    problem.old_vector, expanded.local_energy, expanded.metric_rank,
                    expanded.residual_norm, direction, bond_dim, _trim_options(options),
                    acceptance_energy=base_energy)
                energy = _energy(candidate, hamiltonian, stable=True)
                update = replace(update, cbe_trimmed_energy=trim.energy_refinement_initial_energy,
                    cbe_refined_energy=trim.energy_refinement_energy,
                    cbe_trim_loss=trim.metric_truncation_loss,
                    cbe_trim_method='reduced-physical-metric',
                    cbe_compression_diagnostics=(() if trim.compression_diagnostics is None
                                                 else ({**trim.compression_diagnostics, 'iterations': trim.truncation_iterations},)),
                    cbe_energy_refinement_iterations=trim.energy_refinement_iterations,
                    cbe_energy_refinement_substeps=trim.energy_refinement_accepted_substeps)
                if trim.accepted and energy < base_energy and len(candidate.bond_sectors[i]) <= bond_dim:
                    audit = reduced_stationarity(candidate, hamiltonian, (site,), options)[0]
                    update = replace(update, energy=energy, accepted=True, cbe_baseline_selected=False,
                        residual_norm=audit['residual_norm'], relative_residual=audit['relative_residual'],
                        local_converged=audit['relative_residual'] <= options.eigensolver_tolerance)
                    chosen = candidate
        except NUMERICAL_ERRORS as error:
            update = replace(update, energy=base_energy, accepted=ordinary.accepted,
                local_converged=False, cbe_baseline_selected=True, cbe_fallback=True,
                cbe_recovery_reason=f'{type(error).__name__}: {error}')
    state.tensors, state.bond_sectors = chosen.tensors, chosen.bond_sectors
    return update

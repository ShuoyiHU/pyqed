"""Residual-controlled whole-multiplet expansion on a closed LETTA ring."""
from collections import Counter
from dataclasses import replace
from operator import index

import numpy as np

from .reduced_cbe import select_metric_cbe, _trim_options
from .reduced_updates import one_site_options
from .reduced_ring_solver import ring_energy, optimize_ring_site, RingSiteUpdate
from .._letta_two_site_opt.reduced_ring_allocation import (
    ring_edge, expand_ring_pair_space, grow_ring_bond)
from .._letta_two_site_opt.reduced_ring_pair import CyclicPairProblem, CyclicPairMetricRoot
from .._letta_two_site_opt.reduced_ring_compression import compress_ring_pair, ring_factor_adjoint
from .._letta_two_site_opt.reduced_ring_solver import (
    fit_ring_pair_target, ring_stationarity, RING_NUMERICAL_ERRORS)


def select_ring_cbe(state, hamiltonian, left_site, options):
    """Project the physical residual off the incumbent tangent, then fit it.

    The full cyclic Gram supplies the root. Its supported coordinates can mix
    different middle fusion paths; independent left/right boundary roots would
    give a different, generally incorrect projection.
    """
    i, j = ring_edge(state, left_site)
    budget = index(options.cbe_expansion_dimension)
    if budget < 1 or options.cbe_selector != 'exact':
        raise ValueError('ring CBE requires exact selector and a positive expansion budget')
    old = Counter(state.bond_sectors[j])
    scaffold = expand_ring_pair_space(state, i, sum(old.values())+budget)
    p = CyclicPairProblem(scaffold, hamiltonian, i)
    root = CyclicPairMetricRoot(p, options.metric_tolerance,
                               max_workspace_mb=options.compression.max_workspace_mb)
    le, re = p.left_embedding, p.right_embedding
    a, b = le.pack_source(scaffold.site_blocks(i)), re.pack_source(scaffold.site_blocks(j))
    def compress(target, u, v, allocation):
        return compress_ring_pair(target, p, scaffold, u, v, allocation,
            options=options.compression, metric_tolerance=options.metric_tolerance,
            als_max_iterations=options.cbe_refinement_max_iterations)
    return select_metric_cbe(scaffold, p, root, old, Counter(scaffold.bond_sectors[j]),
        (le, re), (a, b), options=options, merge=p.merge,
        factor_adjoint=lambda side, v: ring_factor_adjoint(p, side, a, b, v),
        compress=compress, seed=2817+i,
        complex_data=any(np.iscomplexobj(v) for site in range(state.nsites+1)
                         for v in state.site_blocks(site).values()))


def expand_ring_cbe(state, selection, direction):
    """Insert selected factors with a zero partner; preserve every amplitude."""
    if direction not in {'lr', 'rl'}:
        raise ValueError('ring CBE direction must be lr or rl')
    if not selection.multiplicities:
        return state.copy()
    i, j = ring_edge(state, selection.problem.left_site)
    old = Counter(state.bond_sectors[j])
    new = old+Counter(selection.multiplicities)
    candidate = grow_ring_bond(state, i, new)
    for side, site in enumerate((i, j)):
        source = selection.left if side == 0 else selection.right
        data = {}
        for key, a in candidate.site_blocks(site).items():
            q, axis = (key[2], -1) if side == 0 else (key[0], 0)
            block = np.array(a, dtype=np.result_type(a, source.get(key, 0.), complex), copy=True)
            count = selection.multiplicities.get(q, 0)
            if count:
                dst = [slice(None)]*block.ndim
                dst[axis] = slice(old[q], new[q])
                block[tuple(dst)] = 0.
                if (side == 1 and direction == 'lr') or (side == 0 and direction == 'rl'):
                    src = [slice(None)]*block.ndim
                    src[axis] = slice(0, count)
                    block[tuple(dst)] = source[key][tuple(src)]
            data[key] = block
        candidate.set_site_blocks(site, data)
    return candidate


def ring_cbe_site(state, hamiltonian, site, direction, bond_dim, options):
    """Ordinary baseline, selected expansion, one-site solve and guarded fit."""
    if direction not in {'lr', 'rl'} or not 0 <= site <= state.nsites or index(bond_dim) < 1:
        raise ValueError('invalid ring CBE site, direction or cap')
    i = site if direction == 'lr' else (site-1) % (state.nsites+1)
    j = (i+1) % (state.nsites+1)
    if len(state.bond_sectors[j]) > bond_dim:
        raise ValueError('CBE cap cannot be smaller than the incumbent ring bond')
    before = ring_energy(state, hamiltonian)
    baseline = state.copy()
    local_options = one_site_options(options)
    try:
        ordinary = optimize_ring_site(baseline, hamiltonian, site, local_options)
        if ordinary.recovery_reason or not ordinary.accepted:
            raise FloatingPointError(ordinary.recovery_reason or 'ordinary update rejected')
        base_energy = ring_energy(baseline, hamiltonian)
    except RING_NUMERICAL_ERRORS as error:
        return RingSiteUpdate(site=site, local_energy=before, energy=before,
            metric_rank=0, local_dimension=0, residual_norm=np.inf, relative_residual=np.inf,
            accepted=False, local_converged=False, is_target_closure=site == state.nsites,
            cbe_selector='exact', cbe_old_energy=before, cbe_fallback=True,
            cbe_recovery_rejected=True,
            cbe_recovery_reason=f'one-site baseline failed: {type(error).__name__}: {error}')
    update = replace(ordinary, cbe_selector='exact', cbe_old_energy=before,
        cbe_baseline_energy=base_energy, cbe_baseline_allowance=0., cbe_baseline_selected=True)
    chosen = baseline
    try:
        selection = select_ring_cbe(baseline.copy(), hamiltonian, i, options)
        update = replace(update, cbe_expansion_dimension=sum(selection.multiplicities.values()),
            cbe_projection_iterations=selection.projection_iterations,
            cbe_projection_converged=selection.projection_converged,
            cbe_pair_dimension=selection.problem.local_dimension,
            cbe_materialized_pair_tensor=True, cbe_materialized_pair_metric=True,
            cbe_materialized_tangent_jacobian=False, cbe_missing_norm=selection.missing_norm,
            cbe_captured_weight=selection.captured_weight, cbe_selection_loss=selection.loss,
            cbe_selection_diagnostics=dict(metric_kind='full-cyclic-reduced',
                tangent_overlap=selection.tangent_overlap, factor_fits=selection.diagnostics,
                allocation=tuple((str(q), n) for q, n in selection.multiplicities.items())))
        if selection.multiplicities:
            candidate = expand_ring_cbe(baseline, selection, direction)
            expanded = optimize_ring_site(candidate, hamiltonian, site, local_options)
            if expanded.recovery_reason or not expanded.accepted:
                raise FloatingPointError(expanded.recovery_reason or 'expanded one-site update rejected')
            update = replace(update, cbe_expanded_energy=ring_energy(candidate, hamiltonian))
            p = CyclicPairProblem(candidate, hamiltonian, i)
            trimmed, fit, split, refinement, diagnostics = fit_ring_pair_target(
                candidate, hamiltonian, p, p.old_vector, bond_dim, _trim_options(options), direction=direction)
            energy = ring_energy(trimmed, hamiltonian)
            update = replace(update,
                cbe_trimmed_energy=energy if refinement is None else refinement.initial_energy,
                cbe_refined_energy=None if refinement is None else refinement.energy,
                cbe_trim_loss=fit.loss, cbe_trim_method='full-cyclic-physical-metric',
                cbe_compression_diagnostics=({**diagnostics, 'iterations': fit.iterations},),
                cbe_energy_refinement_iterations=0 if refinement is None else refinement.iterations,
                cbe_energy_refinement_substeps=0 if refinement is None else refinement.accepted_substeps)
            if energy < base_energy and len(trimmed.bond_sectors[j]) <= bond_dim:
                audit = ring_stationarity(trimmed, hamiltonian, (site,), options)[0]
                update = replace(update, energy=energy, accepted=True, cbe_baseline_selected=False,
                    residual_norm=audit['residual_norm'], relative_residual=audit['relative_residual'],
                    local_converged=audit['relative_residual'] <= options.eigensolver_tolerance)
                chosen = trimmed
            # Incomplete compression is a usable best iterate, not convergence.
            reports = selection.diagnostics+(diagnostics,)
            if not all(r.get('optimizer_success', False) for r in reports):
                update = replace(update, local_converged=False)
    except RING_NUMERICAL_ERRORS as error:
        update = replace(update, energy=base_energy, accepted=ordinary.accepted,
            local_converged=False, cbe_baseline_selected=True, cbe_fallback=True,
            cbe_recovery_reason=f'{type(error).__name__}: {error}')
        chosen = baseline
    state.restore(chosen)
    return update

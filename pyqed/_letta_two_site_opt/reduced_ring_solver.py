"""Transactional cyclic two-site updates in the full reduced overlap metric."""
from collections import Counter
from dataclasses import replace
from operator import index

import numpy as np

from .._letta_one_site_opt.reduced_ring_solver import (
    ring_energy, ring_local_problem, ring_gauge_shift, optimize_ring_site)
from .._letta_one_site_opt.reduced_solver import _solve_local_problem
from .._letta_one_site_opt.reduced_updates import (
    NUMERICAL_ERRORS, one_site_options, local_residual, ReducedEnergyRefinement)
from .reduced_ring_pair import CyclicPairProblem
from .reduced_ring_compression import compress_ring_pair
from .reduced_ring_allocation import ring_edge, expand_ring_pair_space, install_ring_factors
from .reduced_solver import _split_reduced_pair, _project_frontier_blocks


RING_NUMERICAL_ERRORS = NUMERICAL_ERRORS


def ring_stationarity(state, hamiltonian, sites, options):
    result = []
    for site in sites:
        p = ring_local_problem(state, hamiltonian, site, matrix_free=options.matrix_free,
                               dense_solver_threshold=options.dense_solver_threshold)
        absolute, relative = local_residual(p, p.embedding.pack_source(state.site_blocks(site)))
        result.append(dict(site=site, residual_norm=absolute, relative_residual=relative))
    return tuple(result)


def refine_ring_pair_energy(state, hamiltonian, left_site, options, *, direction='lr'):
    """Alternate both graph vertices, retaining the full cyclic local metric."""
    sites = ring_edge(state, left_site)
    if direction not in {'lr', 'rl'}:
        raise ValueError('invalid ring pair direction')
    if direction == 'rl':
        sites = sites[::-1]
    candidate = state.copy()
    local = one_site_options(options)
    initial = ring_energy(candidate, hamiltonian)
    energies, accepted, iteration = [initial], 0, 0
    status = 'maximum energy-refinement rounds reached'
    stationary = ()
    for iteration in range(1, options.energy_refinement_max_iterations+1):
        previous = energies[-1]
        updates = []
        for site in sites:
            u = optimize_ring_site(candidate, hamiltonian, site, local)
            if u.recovery_reason:
                raise FloatingPointError(u.recovery_reason)
            updates.append(u)
            accepted += int(u.accepted)
            energies.append(u.energy)
        if abs(energies[-1]-previous) <= options.energy_refinement_tolerance:
            stationary = ring_stationarity(candidate, hamiltonian, sites, local)
            if (all(u.accepted and u.local_converged for u in updates) and
                    max(r['relative_residual'] for r in stationary) <= options.eigensolver_tolerance):
                status = 'converged'
                break
    if not stationary or status != 'converged':
        stationary = ring_stationarity(candidate, hamiltonian, sites, local)
    return ReducedEnergyRefinement(candidate, initial, ring_energy(candidate, hamiltonian),
        iteration, accepted, tuple(energies), status == 'converged', status, stationary)


def fit_ring_pair_target(state, hamiltonian, problem, vector, bond_dim, options, *, direction='lr'):
    """Compress an externally supplied pair vector; never diagonalize H here.

    The coefficient SVD only initializes ranks and factors. The final fit uses
    the correlated physical metric, followed by optional A/B energy alternation.
    CBE can supply its expanded one-site state to this same routine.
    """
    i, j = problem.left_site, problem.right_site
    split = _split_reduced_pair(problem.layout.unpack(vector), problem.sites[i], problem.sites[j],
        bond_dim=bond_dim, sector_capacities=Counter(state.bond_sectors[j]),
        direction=direction, cutoff=options.conditional_svd_cutoff)
    ranks = dict(split.retained_multiplicities)
    le, re = problem.left_embedding, problem.right_embedding
    starts = ((le.pack_source(state.site_blocks(i)), re.pack_source(state.site_blocks(j))),
              (le.pack_source(_project_frontier_blocks(le, split.left_blocks)),
               re.pack_source(_project_frontier_blocks(re, split.right_blocks))))
    fits, failures = [], []
    for a, b in starts:
        try:
            fit = compress_ring_pair(vector, problem, state, a, b, ranks,
                options=options.compression, metric_tolerance=options.metric_tolerance,
                als_max_iterations=options.truncation_max_iterations)
            merged = problem.merge(fit.left, fit.right)
            norm = float(np.vdot(merged, problem.apply_metric(merged)).real)
            if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
                raise FloatingPointError('null or nonfinite compressed ring state')
            candidate = install_ring_factors(state, problem, fit.left/np.sqrt(norm), fit.right, ranks)
            candidate.normalize(center=i, balance=False)
            fits.append((fit, candidate))
        except NUMERICAL_ERRORS as error:
            failures.append(f'{type(error).__name__}: {error}')
    if not fits:
        raise FloatingPointError('all cyclic compression starts failed: '+'; '.join(failures))
    fit, candidate = min(fits, key=lambda pair: pair[0].loss)
    refinement = None
    if options.split_method in {'metric-als-energy', 'metric-energy', 'energy-refined'}:
        refinement = refine_ring_pair_energy(candidate, hamiltonian, i, options, direction=direction)
        candidate = refinement.state
    diagnostics = {**fit.diagnostics, 'initializations': len(starts),
                   'successful_initializations': len(fits), 'failed_initializations': tuple(failures)}
    return candidate, fit, split, refinement, diagnostics


def optimize_ring_pair(state, hamiltonian, left_site, direction, bond_dim, options):
    """Compare the compressed/refined trial to a same-start ordinary update."""
    from .solver import LETTAPairUpdate
    i, j = ring_edge(state, left_site)
    if direction not in {'lr', 'rl'} or index(bond_dim) < 1:
        raise ValueError('invalid ring pair direction or cap')
    before = ring_energy(state, hamiltonian)
    common = dict(left_site=i, right_site=j,
        shared_physical_sites=tuple(sorted(set(state.site_neighborhood(i) if i < state.nsites else ()) &
            set(state.site_neighborhood(j) if j < state.nsites else ()))),
        old_energy=before, local_energy=before, energy=before, metric_rank=0, local_dimension=0,
        residual_norm=float('inf'), conditional_discarded_weight=0., metric_truncation_loss=0.,
        truncation_iterations=0, energy_refinement_initial_energy=None, energy_refinement_energy=None,
        energy_refinement_iterations=0, energy_refinement_accepted_substeps=0,
        max_factor_norm=max(np.linalg.norm(a) for site in (i,j) for a in state.site_blocks(site).values()),
        sector_ranks=tuple(Counter(state.bond_sectors[j]).values()), accepted=False)
    update = LETTAPairUpdate(**common)
    candidate = None
    try:
        expanded = expand_ring_pair_space(state, i, bond_dim) if options.reduced_sector_growth else state.copy()
        p = CyclicPairProblem(expanded, hamiltonian, i)
        if not options.matrix_free or p.local_dimension <= options.dense_solver_threshold:
            p.materialize(max_workspace_mb=options.compression.max_workspace_mb)
        energy, vector, rank, residual = _solve_local_problem(p, options, initial_vector=p.old_vector)
        candidate, fit, split, refinement, diagnostic = fit_ring_pair_target(
            expanded, hamiltonian, p, vector, bond_dim, options, direction=direction)
        actual = ring_energy(candidate, hamiltonian)
        update = replace(update, local_energy=energy, energy=actual, metric_rank=rank,
            local_dimension=p.local_dimension, full_local_dimension=p.full_local_dimension,
            residual_norm=residual, metric_truncation_loss=fit.loss,
            # This coefficient-SVD statistic is not a physical discarded norm.
            conditional_discarded_weight=split.discarded_weight, sector_ranks=split.sector_ranks,
            truncation_iterations=fit.iterations, compression_diagnostics=diagnostic,
            energy_refinement_initial_energy=None if refinement is None else refinement.initial_energy,
            energy_refinement_energy=None if refinement is None else refinement.energy,
            energy_refinement_iterations=0 if refinement is None else refinement.iterations,
            energy_refinement_accepted_substeps=0 if refinement is None else refinement.accepted_substeps,
            energy_refinement_diagnostics=None if refinement is None else refinement.diagnostics,
            max_factor_norm=max(np.linalg.norm(a) for site in (i,j)
                                for a in candidate.site_blocks(site).values()),
            accepted=np.isfinite(actual) and actual <= before+options.energy_increase_tolerance)
    except RING_NUMERICAL_ERRORS as error:
        update = replace(update, fallback=True, recovery_reason=f'{type(error).__name__}: {error}')
    # The baseline is always computed from the untouched incumbent. It is a
    # feasible competing candidate only when the current bond meets the cap.
    baseline = state.copy()
    try:
        ordinary = optimize_ring_site(baseline, hamiltonian, i if direction == 'lr' else j,
                                      one_site_options(options))
        baseline_energy = ring_energy(baseline, hamiltonian)
        feasible = len(state.bond_sectors[j]) <= bond_dim
        update = replace(update, baseline_energy=baseline_energy)
        if (feasible and ordinary.accepted and not ordinary.recovery_reason and
                (not update.accepted or baseline_energy <= update.energy)):
            state.restore(baseline)
            return replace(update, energy=baseline_energy, accepted=True, baseline_selected=True)
        if ordinary.recovery_reason and not update.recovery_reason:
            update = replace(update, recovery_reason='baseline: '+ordinary.recovery_reason)
    except RING_NUMERICAL_ERRORS as error:
        update = replace(update, recovery_reason=(update.recovery_reason or '')+
                         f'; baseline: {type(error).__name__}: {error}')
    if update.accepted:
        state.restore(candidate)
        return update
    return replace(update, energy=before, accepted=False)


def ring_two_site_dmrg(hamiltonian, *, state, bond_dim, options):
    """Sweep every ring graph edge, including both covariant-closure edges."""
    from .solver import LETTATwoSiteSweep, LETTATwoSiteResult
    if index(bond_dim) < 1:
        raise ValueError('ring bond cap must be positive')
    if options.start_direction not in {'lr', 'rl'} or options.max_sweeps < 1:
        raise ValueError('invalid ring sweep direction or count')
    if options.gauge_mode not in {'frontier', 'scalar', 'none'}:
        raise ValueError('ring gauge mode must be frontier, scalar, or none')
    state = state.copy()
    state.normalize()
    direction = options.start_direction
    previous = ring_energy(state, hamiltonian)
    history, converged = [], False
    for sweep in range(1, options.max_sweeps+1):
        edges = range(state.nsites+1) if direction == 'lr' else range(state.nsites, -1, -1)
        updates = []
        for edge in edges:
            update = optimize_ring_pair(state, hamiltonian, edge, direction, bond_dim, options)
            if options.gauge_mode != 'none':
                snapshot = state.copy()
                try:
                    site = edge if direction == 'lr' else (edge+1) % (state.nsites+1)
                    ring_gauge_shift(state, site, direction, tolerance=options.metric_tolerance, mode=options.gauge_mode)
                    checked = ring_energy(state, hamiltonian)
                    if abs(checked-update.energy) > max(options.energy_increase_tolerance, 1e-10*max(1., abs(checked))):
                        raise FloatingPointError('ring gauge changed the physical energy')
                except RING_NUMERICAL_ERRORS as error:
                    state.restore(snapshot)
                    update = replace(update, recovery_reason=f'gauge: {type(error).__name__}: {error}')
            updates.append(update)
        energy = ring_energy(state, hamiltonian)
        change = abs(energy-previous)
        history.append(LETTATwoSiteSweep(sweep, direction, energy, change, change/state.nsites,
                                        max(state.bond_dimensions), tuple(updates)))
        if options.verbosity:
            print(f'reduced ring two-site sweep {sweep}: energy={energy:.14f}, dE/site={change/state.nsites:.3e}')
        if (change/state.nsites <= options.tolerance and max(state.bond_dimensions) <= bond_dim and
                all(u.accepted and not u.recovery_reason and not u.fallback and
                    (u.compression_diagnostics or {}).get('optimizer_success', False)
                    for u in updates)):
            stationary = ring_stationarity(state, hamiltonian, range(state.nsites+1), options)
            if max(r['relative_residual'] for r in stationary) <= options.eigensolver_tolerance:
                converged = True
                break
        previous = energy
        if options.alternate:
            direction = 'rl' if direction == 'lr' else 'lr'
    two_site_energy = energy
    polish = 0
    if options.one_site_polish_sweeps:
        from .._letta_one_site_opt.reduced_ring_solver import ring_dmrg
        polished = ring_dmrg(hamiltonian, state=state,
            options=replace(one_site_options(options), max_sweeps=options.one_site_polish_sweeps))
        state, energy, polish = polished.state, polished.energy, polished.sweeps
        converged = converged and polished.converged
    return LETTATwoSiteResult(state, energy, converged, len(history), tuple(history),
        'CONVERGENCE: ENERGY AND FRESH LOCAL RESIDUALS' if converged else 'STOP: MAXIMUM SWEEPS REACHED',
        two_site_energy, polish)

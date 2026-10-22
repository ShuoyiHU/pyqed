"""Native reduced energy refinement and local stationarity diagnostics."""
from dataclasses import dataclass
from operator import index

import numpy as np


NUMERICAL_ERRORS = (FloatingPointError, ArithmeticError, np.linalg.LinAlgError, ValueError, MemoryError)


def one_site_options(options):
    """Carry numerical controls into an ordinary, non-CBE local update."""
    from .solver import LETTADMROptions
    names = ('tolerance', 'metric_tolerance', 'energy_increase_tolerance',
             'eigensolver_tolerance', 'eigensolver_max_iterations',
             'dense_solver_threshold', 'matrix_free', 'gauge_mode')
    return LETTADMROptions(max_sweeps=1, **{k: getattr(options, k) for k in names})


def local_residual_diagnostics(problem, vector, *, metric_tolerance=1e-12):
    """Physical-coordinate stationarity plus unprojected coefficient residuals.

    QR coordinates use the retained overlap support. Their normalized residual
    tests stationarity in that physical subspace, not a global energy minimum.
    Raw residuals remain available to diagnose cutoff and conditioning effects.
    Problems without QR coordinates retain the raw-coordinate convention.
    """
    vector=np.asarray(vector)
    coordinates=getattr(problem,'coordinates',None)
    if coordinates is not None:
        h,energy,norm=coordinates.source_action_data(vector)
        n=problem.apply_metric(vector)
    else:
        h,n=problem.apply_hamiltonian(vector),problem.apply_metric(vector)
        norm=float(np.vdot(vector,n).real)
        energy=float(np.vdot(vector,h).real/norm) if norm>0 else np.nan
    if not np.isfinite(norm) or norm<=np.finfo(float).tiny:
        raise FloatingPointError('Null or nonfinite state in local residual check')
    raw=float(np.linalg.norm(h-energy*n))
    raw_scale=max(float(np.linalg.norm(h)),abs(energy)*float(np.linalg.norm(n)),np.finfo(float).tiny)
    if coordinates is not None:
        coordinates.prepare(metric_tolerance)
        h,n=coordinates.adjoint(h),coordinates.adjoint(n)
    absolute=float(np.linalg.norm(h-energy*n))
    scale=max(float(np.linalg.norm(h)),abs(energy)*float(np.linalg.norm(n)),np.finfo(float).tiny)
    if not np.isfinite(energy) or not np.isfinite(absolute) or not np.isfinite(raw):
        raise FloatingPointError('Nonfinite reduced local residual')
    return dict(energy=energy,norm_squared=norm,residual_norm=absolute,relative_residual=absolute/scale,
                raw_residual_norm=raw,raw_relative_residual=raw/raw_scale)


def local_residual(problem, vector, *, metric_tolerance=1e-12):
    """Return the absolute and relative local stationarity residuals."""
    report=local_residual_diagnostics(problem,vector,metric_tolerance=metric_tolerance)
    return report['residual_norm'],report['relative_residual']


def reduced_stationarity(state, hamiltonian, sites, options):
    """Rebuild local actions at the final state, after all neighboring changes."""
    from .reduced_solver import reduced_local_problem
    context = None
    if getattr(options, 'simplify_fixed_ties', False):
        from .dependency_environment import DependencySweepContext
        context = DependencySweepContext(state, hamiltonian)
    reports = []
    for site in sites:
        problem = (context.local_problem(site, options) if context is not None else
            reduced_local_problem(state, hamiltonian, site,
                matrix_free=options.matrix_free, dense_solver_threshold=options.dense_solver_threshold))
        vector = problem.embedding.pack_source(state.tensors[site])
        absolute, relative = local_residual(problem, vector, metric_tolerance=options.metric_tolerance)
        reports.append(dict(site=site, residual_norm=absolute, relative_residual=relative))
    return tuple(reports)


@dataclass(frozen=True)
class ReducedEnergyRefinement:
    state: object
    initial_energy: float
    energy: float
    iterations: int
    accepted_substeps: int
    energies: tuple
    converged: bool
    status: str
    stationarity: tuple

    @property
    def diagnostics(self):
        return dict(converged=self.converged, status=self.status,
                    stationarity=self.stationarity, energies=self.energies)


def refine_reduced_pair_energy(state, hamiltonian, left_site, options, *,
                               max_iterations, tolerance, direction='lr'):
    """Alternate exact one-site energy minimizations with a fixed bond layout.

    Work on a copy. Each local update performs a fresh physical energy check
    and is transactional. Flat energy alone is not called convergence.
    """
    from .reduced_solver import StableEnergyCache, optimize_reduced_site
    limit = index(max_iterations)
    if limit < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid energy-refinement budget or tolerance')
    if not 0 <= left_site < state.nsites-1 or direction not in {'lr', 'rl'}:
        raise ValueError('invalid pair or sweep direction')
    candidate = state.copy()
    local_options = one_site_options(options)
    sites = (left_site, left_site+1) if direction == 'lr' else (left_site+1, left_site)
    energy_cache = StableEnergyCache(hamiltonian)
    initial = energy_cache(candidate)
    energies = [initial]
    accepted = 0
    status = 'maximum energy-refinement rounds reached'
    stationary = ()
    iteration = 0
    for iteration in range(1, limit+1):
        before = energies[-1]
        updates = []
        for site in sites:
            update = optimize_reduced_site(candidate, hamiltonian, site, local_options, energy_cache=energy_cache)
            updates.append(update)
            accepted += int(update.accepted)
            energies.append(update.energy)
        if abs(energies[-1]-before) <= tolerance:
            stationary = reduced_stationarity(candidate, hamiltonian, sites, local_options)
            if (all(u.accepted and u.local_converged for u in updates) and
                    max(r['relative_residual'] for r in stationary) <= options.eigensolver_tolerance):
                status = 'converged'
                break
    if not stationary or status != 'converged':
        stationary = reduced_stationarity(candidate, hamiltonian, sites, local_options)
    return ReducedEnergyRefinement(candidate, initial,
        energy_cache(candidate), iteration, accepted, tuple(energies),
        status == 'converged', status, stationary)

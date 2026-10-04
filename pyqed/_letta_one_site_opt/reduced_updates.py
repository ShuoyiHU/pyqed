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


def local_residual(problem, vector):
    """Report the unprojected Hx-E Nx residual in local coordinates.

    The relative value is scaled by the two terms in the generalized equation.
    This is a local stationarity check, not a certificate of a global minimum.
    """
    vector = np.asarray(vector)
    h, n = problem.apply_hamiltonian(vector), problem.apply_metric(vector)
    norm = float(np.vdot(vector, n).real)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise FloatingPointError('null or nonfinite state in local residual check')
    energy = float(np.vdot(vector, h).real/norm)
    residual = float(np.linalg.norm(h-energy*n))
    scale = max(float(np.linalg.norm(h)), abs(energy)*float(np.linalg.norm(n)),
                np.finfo(float).tiny)
    if not np.isfinite(energy) or not np.isfinite(residual):
        raise FloatingPointError('nonfinite reduced local residual')
    return residual, residual/scale


def reduced_stationarity(state, hamiltonian, sites, options):
    """Rebuild local actions at the final state, after all neighboring changes."""
    from .reduced_solver import reduced_local_problem
    reports = []
    for site in sites:
        problem = reduced_local_problem(state, hamiltonian, site,
            matrix_free=options.matrix_free, dense_solver_threshold=options.dense_solver_threshold)
        vector = problem.embedding.pack_source(state.tensors[site])
        absolute, relative = local_residual(problem, vector)
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
    from .reduced_solver import _energy, optimize_reduced_site
    limit = index(max_iterations)
    if limit < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid energy-refinement budget or tolerance')
    if not 0 <= left_site < state.nsites-1 or direction not in {'lr', 'rl'}:
        raise ValueError('invalid pair or sweep direction')
    candidate = state.copy()
    local_options = one_site_options(options)
    sites = (left_site, left_site+1) if direction == 'lr' else (left_site+1, left_site)
    initial = _energy(candidate, hamiltonian, stable=True)
    energies = [initial]
    accepted = 0
    status = 'maximum energy-refinement rounds reached'
    stationary = ()
    iteration = 0
    for iteration in range(1, limit+1):
        before = energies[-1]
        updates = []
        for site in sites:
            update = optimize_reduced_site(candidate, hamiltonian, site, local_options)
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
        _energy(candidate, hamiltonian, stable=True), iteration, accepted, tuple(energies),
        status == 'converged', status, stationary)

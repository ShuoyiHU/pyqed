"""Full-metric one-site optimization for reduced conditional virtual rings."""
from dataclasses import dataclass, replace

import numpy as np

from .reduced_frontier import ReducedFrontier, FrontierSiteEmbedding, _BlockVectorLayout
from .reduced_ring_state import ReducedRingLETTA
from .reduced_ring_target import CyclicReducedMPO
from .reduced_ring_contraction import CyclicReducedNorm, CyclicReducedOperator
from .reduced_solver import ReducedLocalProblem, _solve_local_problem, _checked_energy
from .reduced_updates import NUMERICAL_ERRORS, local_residual
from .solver import LETTASiteUpdate, LETTASweep, LETTADMRGResult


@dataclass(frozen=True)
class RingSiteUpdate(LETTASiteUpdate):
    recovery_reason: str | None = None
    is_target_closure: bool = False


def _native_hamiltonian(state, hamiltonian):
    if hasattr(hamiltonian, 'native_mpo'):
        hamiltonian = hamiltonian.native_mpo(state.physical_basis)
    from .operators import LatticeMPO
    from .reduced_mpo_compile import SpinTensorMPO
    if isinstance(hamiltonian, LatticeMPO):
        hamiltonian = SpinTensorMPO.compile(hamiltonian.factors, state.physical_basis)
    native = CyclicReducedMPO.from_native(hamiltonian)
    return state.to_target_ring().extend_hamiltonian(native)


def ring_energy(state, hamiltonian):
    ring = state.to_target_ring()
    native = _native_hamiltonian(state, hamiltonian)
    return _checked_energy(CyclicReducedOperator(ring.sites, native).overlap(),
                           CyclicReducedNorm(ring.sites).overlap())


def ring_site_embedding(state, site):
    state.site_blocks(site)
    if site < state.nsites:
        return ReducedFrontier.from_state(state).site_embedding(state, site)
    layout = _BlockVectorLayout({key: a.shape for key, a in state.closure.data.items()})
    indices = np.arange(layout.size)
    return FrontierSiteEmbedding(layout, layout, indices, indices, (), ())


def ring_local_problem(state, hamiltonian, site, *, matrix_free=False, dense_solver_threshold=96):
    """Compose the exact tie embedding with native cyclic H/N actions."""
    if not isinstance(state, ReducedRingLETTA):
        raise TypeError('expected ReducedRingLETTA')
    state.site_blocks(site)  # Check the physical/closure core index.
    ring = state.to_target_ring()
    embedding = ring_site_embedding(state, site)
    h_chain = CyclicReducedOperator(ring.sites, _native_hamiltonian(state, hamiltonian))
    n_chain = CyclicReducedNorm(ring.sites)
    def action(chain, vector):
        target = embedding.unpack_target(embedding.apply(vector))
        return embedding.adjoint(embedding.pack_target(chain.local_action(site, target)))
    h_action = lambda v: action(h_chain, v)
    n_action = lambda v: action(n_chain, v)
    h, n = None, None
    if not matrix_free or embedding.source_size <= dense_solver_threshold:
        basis = np.eye(embedding.source_size)
        h = np.column_stack([h_action(v) for v in basis])
        n = np.column_stack([n_action(v) for v in basis])
        for matrix in (h, n):
            scale = max(np.linalg.norm(matrix), np.finfo(float).tiny)
            if not np.all(np.isfinite(matrix)) or np.linalg.norm(matrix-matrix.conj().T) > 1e-9*scale:
                raise FloatingPointError('nonfinite or non-Hermitian cyclic local operator')
        h, n = (h+h.conj().T)/2, (n+n.conj().T)/2
        eigenvalues = np.linalg.eigvalsh(n)
        if eigenvalues[0] < -1e-10*max(abs(eigenvalues[-1]), np.finfo(float).tiny):
            raise FloatingPointError('cyclic local metric is not positive semidefinite')
    return ReducedLocalProblem(site, None, embedding, h, n, h_action, n_action)


def ring_gauge_shift(state, site, direction, *, tolerance=1e-12, mode='frontier'):
    """Invertible sector-multiplicity balancing on any edge, including closure.

    Uses a local Gram to condition the bond. It neither assumes nor replaces
    the full cyclic environment metric. Unconditional gauges are legal for
    every physical tie pattern. Null directions keep unit gauge weights.
    """
    if direction not in {'lr', 'rl'} or mode not in {'frontier', 'scalar'} or not 0 < tolerance < 1:
        raise ValueError('invalid ring gauge direction or tolerance')
    state.site_blocks(site)
    neighbor = (site+(1 if direction == 'lr' else -1)) % (state.nsites+1)
    source, other = state.site_blocks(site), state.site_blocks(neighbor)
    if mode == 'scalar':
        scale = max(float(np.max(np.abs(a), initial=0.)) for a in source.values())
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            raise FloatingPointError('null or nonfinite scalar ring gauge')
        updated_source = {key: a/scale for key, a in source.items()}
        updated_other = {key: a*scale for key, a in other.items()}
        if any(not np.all(np.isfinite(a)) for a in (*updated_source.values(), *updated_other.values())):
            raise FloatingPointError('nonfinite scalar ring gauge result')
        state.set_site_blocks(site, updated_source)
        state.set_site_blocks(neighbor, updated_other)
        return
    axis = 2 if direction == 'lr' else 0
    grams = {}
    for key, block in source.items():
        matrix = block.reshape(-1, block.shape[-1]) if axis == 2 else block.reshape(block.shape[0], -1)
        gram = matrix.conj().T@matrix if axis == 2 else matrix@matrix.conj().T
        grams[key[axis]] = grams.get(key[axis], 0.)+gram
    transforms, inverses = {}, {}
    for q, gram in grams.items():
        if not np.all(np.isfinite(gram)):
            raise FloatingPointError('nonfinite ring gauge Gram')
        values, vectors = np.linalg.eigh((gram+gram.conj().T)/2)
        threshold = max(tolerance, np.finfo(float).eps*len(values))*max(float(values[-1]), 0.)
        weights = np.ones_like(values)
        keep = values > threshold
        weights[keep] = np.sqrt(values[keep])
        transforms[q] = (vectors/weights)@vectors.conj().T
        inverses[q] = (vectors*weights)@vectors.conj().T
    updated_source, updated_other = {}, {}
    for key, a in source.items():
        g = transforms[key[axis]]
        updated_source[key] = a@g if axis == 2 else np.einsum('ab,b...->a...', g, a)
    for key, a in other.items():
        q = key[0] if axis == 2 else key[2]
        if q not in inverses:
            updated_other[key] = a.copy()
        else:
            g = inverses[q]
            updated_other[key] = np.einsum('ab,b...->a...', g, a) if axis == 2 else a@g
    if any(not np.all(np.isfinite(a)) for a in (*updated_source.values(), *updated_other.values())):
        raise FloatingPointError('nonfinite ring gauge result')
    state.set_site_blocks(site, updated_source)
    state.set_site_blocks(neighbor, updated_other)


def optimize_ring_site(state, hamiltonian, site, options):
    snapshot = state.copy()
    old_energy = ring_energy(snapshot, hamiltonian)
    problem = None
    try:
        problem = ring_local_problem(state, hamiltonian, site, matrix_free=options.matrix_free,
                                     dense_solver_threshold=options.dense_solver_threshold)
        local_energy, vector, rank, residual = _solve_local_problem(problem, options,
            initial_vector=problem.embedding.pack_source(state.site_blocks(site)))
        _, relative = local_residual(problem, vector)
        state.set_site_blocks(site, problem.embedding.unpack_source(vector))
        state.normalize(center=site, balance=False)
        energy = ring_energy(state, hamiltonian)
        accepted = energy <= old_energy+options.energy_increase_tolerance
        if not accepted:
            state.restore(snapshot)
            energy = old_energy
        return RingSiteUpdate(site=site, local_energy=local_energy, energy=energy,
            metric_rank=rank, local_dimension=problem.local_dimension,
            full_local_dimension=problem.full_local_dimension, residual_norm=residual,
            relative_residual=relative, local_converged=accepted and relative <= options.eigensolver_tolerance,
            accepted=accepted, is_target_closure=site == state.nsites)
    except NUMERICAL_ERRORS as error:
        state.restore(snapshot)
        return RingSiteUpdate(site=site, local_energy=old_energy, energy=old_energy,
            metric_rank=0, local_dimension=0 if problem is None else problem.local_dimension,
            residual_norm=float('inf'), relative_residual=float('inf'),
            local_converged=False, accepted=False, is_target_closure=site == state.nsites,
            recovery_reason=f'{type(error).__name__}: {error}')


def ring_dmrg(hamiltonian, *, state, options):
    """Sweep physical cores and the target closure with exact cyclic metrics."""
    if not isinstance(state, ReducedRingLETTA):
        raise TypeError('expected ReducedRingLETTA')
    if options.cbe_enabled and options.cbe_selector != 'exact':
        raise ValueError('ring CBE currently requires the exact residual selector')
    if options.cbe_enabled and options.cbe_baseline_guard_fraction != 0:
        raise ValueError('ring CBE uses a strict ordinary one-site energy baseline')
    if options.cbe_enabled and options.cbe_preselection_dimension is not None:
        raise ValueError('ring exact CBE does not use a preselection dimension')
    if options.cbe_enabled and not options.cbe_conditional_trim:
        raise ValueError('ring CBE requires physical-metric pair trimming')
    if options.environment_granularity != 'site':
        raise ValueError('ring sweeps require site-granularity environments')
    if options.bond_dimension_schedule is not None or options.boundary_bond_dim is not None:
        raise ValueError('ring sweeps require a fixed bond budget and exact environments')
    if options.gauge_mode not in {'frontier', 'scalar', 'none'}:
        raise ValueError('ring gauge mode must be frontier, scalar, or none')
    if options.start_direction not in {'lr', 'rl'} or options.max_sweeps < 1:
        raise ValueError('invalid sweep direction or count')
    state = state.copy()
    state.normalize()
    direction = options.start_direction
    bond_budget = max(state.bond_dimensions)
    previous = ring_energy(state, hamiltonian)
    history, converged = [], False
    for sweep in range(1, options.max_sweeps+1):
        sites = range(state.nsites+1) if direction == 'lr' else range(state.nsites, -1, -1)
        updates = []
        for site in sites:
            if options.cbe_enabled:
                from .reduced_ring_cbe import ring_cbe_site
                update = ring_cbe_site(state, hamiltonian, site, direction, bond_budget, options)
            else:
                update = optimize_ring_site(state, hamiltonian, site, options)
            if options.gauge_mode != 'none':
                snapshot = state.copy()
                try:
                    ring_gauge_shift(state, site, direction, tolerance=options.metric_tolerance, mode=options.gauge_mode)
                    checked = ring_energy(state, hamiltonian)
                    if abs(checked-update.energy) > max(options.energy_increase_tolerance, 1e-10*max(1., abs(checked))):
                        raise FloatingPointError('ring gauge changed the physical energy')
                except NUMERICAL_ERRORS as error:
                    state.restore(snapshot)
                    update = replace(update, local_converged=False,
                                     recovery_reason=f'gauge: {type(error).__name__}: {error}')
            updates.append(update)
        energy = ring_energy(state, hamiltonian)
        change = abs(energy-previous)
        history.append(LETTASweep(sweep, direction, energy, change, change/state.nsites,
                                 max(state.bond_dimensions), tuple(updates)))
        if options.verbosity:
            print(f'reduced ring sweep {sweep}: energy={energy:.14f}, dE/site={change/state.nsites:.3e}')
        if change/state.nsites <= options.tolerance and all(u.accepted and u.local_converged and not u.recovery_reason and
                not u.cbe_recovery_reason for u in updates):
            residuals = []
            for site in range(state.nsites+1):
                p = ring_local_problem(state, hamiltonian, site, matrix_free=options.matrix_free,
                                       dense_solver_threshold=options.dense_solver_threshold)
                residuals.append(local_residual(p, p.embedding.pack_source(state.site_blocks(site)))[1])
            if max(residuals) <= options.eigensolver_tolerance:
                converged = True
                break
        previous = energy
        if options.alternate:
            direction = 'rl' if direction == 'lr' else 'lr'
    return LETTADMRGResult(state, energy, converged, len(history), tuple(history),
        'CONVERGENCE: ENERGY AND FRESH LOCAL RESIDUALS' if converged else 'STOP: MAXIMUM SWEEPS REACHED')

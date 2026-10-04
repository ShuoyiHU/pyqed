"""One validated user interface for the shared symmetry-preserving solvers."""
from dataclasses import dataclass, field, replace

import numpy as np

from .._letta_compression import MetricCompressionOptions
from .._letta_one_site_opt import LETTADMROptions, letta_dmrg
from .._letta_one_site_opt.reduced_state import ReducedLatticeLETTA
from .._letta_one_site_opt.reduced_ring_state import ReducedRingLETTA
from .._letta_one_site_opt.reduced_ring_target import signed_physical_basis, signed_sector
from .._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from .models import LETTAProblem, _integer, _real


@dataclass(frozen=True)
class OptimizationOptions:
    """A sweep is one directional pass; an energy round visits both pair cores.

    Compression ALS cycles and inner LSMR iterations are independent. Explicit
    defaults also replace None budgets in a supplied MetricCompressionOptions.
    Nonlinear compression retains that object's evaluation/iteration semantics.
    """
    max_sweeps: int = 100
    energy_tolerance: float = 1e-10
    metric_tolerance: float = 1e-12
    acceptance_tolerance: float = 1e-10
    eigensolver_tolerance: float = 1e-10
    eigensolver_max_iterations: int = 300
    dense_solver_threshold: int = 64
    matrix_free: bool = True
    gauge_mode: str = 'frontier'
    start_direction: str = 'lr'
    alternate: bool = True
    compression: MetricCompressionOptions = field(default_factory=lambda:
        MetricCompressionOptions(als_max_iterations=32, lsmr_max_iterations=512))
    energy_refinement_rounds: int = 32
    energy_refinement_tolerance: float = 1e-10
    cbe_expansion_dimension: int = 1
    cbe_projection_max_iterations: int = 512
    cbe_projection_tolerance: float = 1e-10
    verbosity: int = 0

    def __post_init__(self):
        for name in ('max_sweeps', 'eigensolver_max_iterations', 'cbe_expansion_dimension',
                     'cbe_projection_max_iterations'):
            _integer(getattr(self, name), name, 1)
        for name in ('dense_solver_threshold', 'energy_refinement_rounds', 'verbosity'):
            _integer(getattr(self, name), name)
        for name in ('energy_tolerance', 'metric_tolerance', 'eigensolver_tolerance',
                     'energy_refinement_tolerance', 'cbe_projection_tolerance'):
            value = _real(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.metric_tolerance >= 1:
            raise ValueError('metric_tolerance must be less than one')
        if _real(self.acceptance_tolerance, 'acceptance_tolerance') < 0:
            raise ValueError('acceptance_tolerance must be finite and nonnegative')
        if self.gauge_mode not in {'frontier', 'scalar', 'none'}:
            raise ValueError('gauge_mode must be frontier, scalar or none')
        for name in ('matrix_free', 'alternate'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f'{name} must be a boolean')
        if self.start_direction not in {'lr', 'rl'}:
            raise ValueError('start_direction must be lr or rl')
        if not isinstance(self.compression, MetricCompressionOptions):
            raise TypeError('compression must be MetricCompressionOptions')
        object.__setattr__(self, 'compression', replace(self.compression,
            als_max_iterations=self.compression.als_max_iterations or 32,
            lsmr_max_iterations=self.compression.lsmr_max_iterations or 512))

    def backend_options(self, method):
        common = dict(max_sweeps=self.max_sweeps, tolerance=self.energy_tolerance,
            metric_tolerance=self.metric_tolerance, energy_increase_tolerance=self.acceptance_tolerance,
            eigensolver_tolerance=self.eigensolver_tolerance,
            eigensolver_max_iterations=self.eigensolver_max_iterations,
            dense_solver_threshold=self.dense_solver_threshold, matrix_free=self.matrix_free,
            gauge_mode=self.gauge_mode, start_direction=self.start_direction,
            alternate=self.alternate, compression=self.compression, verbosity=self.verbosity)
        if method == 'two-site':
            return LETTATwoSiteOptions(**common, reduced_sector_growth=True,
                split_method='metric-als-energy' if self.energy_refinement_rounds else 'metric-als',
                truncation_max_iterations=self.compression.als_max_iterations,
                truncation_tolerance=self.compression.tolerance,
                energy_refinement_max_iterations=self.energy_refinement_rounds,
                energy_refinement_tolerance=self.energy_refinement_tolerance)
        if method not in {'one-site', 'cbe'}:
            raise ValueError('method must be one-site, cbe or two-site')
        return LETTADMROptions(**common, cbe_enabled=method == 'cbe',
            cbe_expansion_dimension=self.cbe_expansion_dimension,
            cbe_refinement_max_iterations=self.compression.als_max_iterations,
            cbe_energy_refinement_max_iterations=self.energy_refinement_rounds,
            cbe_energy_refinement_tolerance=self.energy_refinement_tolerance,
            cbe_projection_max_iterations=self.cbe_projection_max_iterations,
            cbe_projection_tolerance=self.cbe_projection_tolerance)


def random_state(problem, *, topology='open', ties='nn', multiplets_per_sector=1,
                 seed=None, real=False, anchor_sector=None):
    """Copies per sector are NOT a total bond cap; inspect state.bond_dimensions.

    nn has no wrap dependency regardless of the virtual topology. nn-periodic
    explicitly includes the last-first tie. An explicit neighborhood sequence
    may contain arbitrary forward/backward dependencies; ownership comes first.
    """
    if not isinstance(problem, LETTAProblem):
        raise TypeError('problem must be LETTAProblem')
    if not isinstance(real, (bool, np.bool_)):
        raise ValueError('real must be a boolean')
    copies = _integer(multiplets_per_sector, 'multiplets_per_sector', 1)
    n = problem.nsites
    if isinstance(ties, str):
        if ties not in {'none', 'nn', 'nn-periodic'}:
            raise ValueError('ties must be none, nn, nn-periodic or explicit neighborhoods')
        neighborhoods = tuple((i,) if ties == 'none' or n == 1 else
            (i, i+1) if i < n-1 else (i, 0) if ties == 'nn-periodic' else (i,)
            for i in range(n))
    else:
        neighborhoods = ties
    if topology == 'ring':
        return ReducedRingLETTA.random(n, problem.physical_basis, problem.symmetry.sector,
            neighborhoods=neighborhoods, multiplets_per_sector=copies,
            anchor_sector=anchor_sector, seed=seed, real=real)
    if topology != 'open' or anchor_sector is not None:
        raise ValueError('topology must be open or ring; an anchor is only valid for a ring')
    return ReducedLatticeLETTA.random((1, n), symmetry=problem.symmetry,
        neighborhoods=neighborhoods, multiplets_per_sector=copies, seed=seed, real=real)


def solve(problem, *, state, method='one-site', options=None, bond_dim=None):
    """Optimize a copy; return native detailed histories and actual allocations.

    bond_dim is the retained total multiplet cap for two-site. One-site and CBE
    inherit the initial maximum allocation; a conflicting cap is an input error.
    Supplied state sites must follow the orbital/site ordering of the model.
    """
    if not isinstance(problem, LETTAProblem):
        raise TypeError('problem must be LETTAProblem')
    options = OptimizationOptions() if options is None else options
    if not isinstance(options, OptimizationOptions):
        raise TypeError('options must be OptimizationOptions')
    controls = options.backend_options(method)
    if not isinstance(state, (ReducedLatticeLETTA, ReducedRingLETTA)):
        raise TypeError('state must be a reduced LETTA open chain or ring')
    if (state.nsites != problem.nsites or
            signed_physical_basis(state.physical_basis) != signed_physical_basis(problem.physical_basis) or
            signed_sector(state.symmetry.sector) != signed_sector(problem.symmetry.sector)):
        raise ValueError('state site count, physical basis or target sector differs from the model')
    initial_cap = max(map(len, state.bond_sectors), default=1)
    cap = initial_cap if bond_dim is None else _integer(bond_dim, 'bond_dim', 1)
    if method == 'two-site':
        return letta_two_site_dmrg(problem.hamiltonian, state=state, bond_dim=cap, options=controls)
    if cap != initial_cap:
        raise ValueError('one-site/CBE cap must equal the maximum initial multiplet allocation')
    return letta_dmrg(problem.hamiltonian, state=state, options=controls)

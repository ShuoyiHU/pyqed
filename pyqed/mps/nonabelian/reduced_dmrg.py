"""Reduced SU(2) two-site DMRG using persistent environments and sweep state.

Integration of upstream/bg 8204ae6a (engine snapshot f0a36f86), retaining
PyQED's independent reduced expectation and LETTA-importable tensor format.
Two-site DMRG: S. R. White, Phys. Rev. Lett. 69, 2863 (1992),
https://doi.org/10.1103/PhysRevLett.69.2863; U. Schollwoeck, Ann. Phys.
326, 96 (2011), https://doi.org/10.1016/j.aop.2010.09.012.
This adapter supports real spatial-orbital integrals and ordinary SU(2) MPS,
not tied LETTA tensors, point-group irreps, CBE, or complex Hamiltonians.
"""
import time
import numpy as np
from .canonical import right_canonicalize_sites, left_canonicalize_sites
from .sweep_engine import load_engine
from .tensor import NonabelianTensor
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep


def _export_sites(owner):
    sites = []
    for record in owner.export_owned_split_sites():
        tensor = record['tensor']
        def sector(q):
            return SpinChargeSector(int(q[0]), SU2Irrep(int(q[1])))
        blocks = {}
        for j, labels in enumerate(tensor['labels']):
            key = tuple(sector(q) for q in labels.reshape(3, 2))
            shape = tensor['shapes'][tensor['shape_offsets'][j]:tensor['shape_offsets'][j+1]]
            blocks[key] = tensor['values'][tensor['offsets'][j]:tensor['offsets'][j+1]].reshape(shape).copy()
        qns = []
        for axis in range(3):
            start, stop = record['leg_sector_offsets'][axis:axis+2]
            qns.append([sector(q) for q, d in zip(record['leg_sector_labels'][start:stop],
                                                record['leg_sector_dims'][start:stop]) for _ in range(d)])
        sites.append(NonabelianTensor(data=blocks, qns=qns, dirs=[-1,1,1],
                                     metadata={'physical_basis': 'fully_reduced_su2'}))
    return sites


class _DirectionalCycle:
    """Compare each endpoint with its own preceding visit, not the other end."""
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.energies = {}
        self.changes = {}

    def observe(self, direction, energy):
        if direction in self.energies:
            self.changes[direction] = energy-self.energies[direction]
        self.energies[direction] = energy
        if (self.tolerance > 0 and len(self.changes) == 2
                and all(abs(x) <= self.tolerance for x in self.changes.values())
                and abs(self.energies['lr']-self.energies['rl']) > self.tolerance):
            return min(self.energies, key=self.energies.get)
        return None


class _PolishPlateau(Exception):
    pass


class ReducedDMRG:
    """Two-site solver; ``max_bond`` caps total SU(2) multiplets, not each sector.

    See module references and scope. ``max_sweeps`` counts directional passes.
    Local residual tolerance is absolute; unconverged local roots raise.
    Convergence requires two consecutive small independently evaluated energy
    changes. It certifies a sweep plateau, not a global minimum or exact state.
    A stable LR/RL cycle switches to one-site optimization from the lower
    endpoint. Each direction must change by at most ``cycle_tolerance`` on
    its own consecutive visits, while the endpoints differ by more than it.
    ``one_site_max_sweeps`` is a separate directional-pass budget; polishing
    keeps the selected sectors and uses the same total-energy tolerance.
    Set ``polish_on_cycle=False`` for two-site-only benchmarks; zero energy
    tolerance also disables this automatic handoff.
    Initial tensors are copied. The initial gauge never truncates them.
    """
    def __init__(self, problem, sites, *, max_bond=30, threads=1):
        from pyqed._letta_one_site_opt import ReducedLatticeLETTA
        if max_bond < 1 or threads < 1 or len(sites) < 2:
            raise ValueError('Positive max_bond/threads and at least two sites required')
        if len(sites) != problem.norb:
            raise ValueError('Site count does not match Hamiltonian')
        if np.iscomplexobj(problem.h1) or np.iscomplexobj(problem.eri):
            raise ValueError('Only real integrals are supported')
        # Also checks the fully reduced convention before any engine conversion.
        self.problem = problem
        self.symmetry = problem.symmetry('su2')
        self.neighborhoods = tuple((i,) for i in range(problem.norb))
        ReducedLatticeLETTA.from_mps(sites, symmetry=self.symmetry, neighborhoods=self.neighborhoods)
        self.max_bond = int(max_bond)
        self.threads = int(threads)
        self.sites = [s.copy() for s in sites]
        self.history = []
        self.success = False
        self.message = 'Not run'

    def _energy(self):
        from pyqed._letta_one_site_opt import ReducedLatticeLETTA
        from pyqed._letta_one_site_opt.reduced_solver import _energy
        self.state = ReducedLatticeLETTA.from_mps(self.sites, symmetry=self.symmetry,
                                                neighborhoods=self.neighborhoods)
        return float(_energy(self.state, self.hamiltonian, stable=True))

    def _polish(self, max_sweeps, energy_tolerance, residual_tolerance,
                max_iterations, callback, direction, start):
        from pyqed._letta_one_site_opt import ReducedFrontier, LETTADMROptions
        from pyqed._letta_one_site_opt.reduced_solver import reduced_letta_dmrg
        sectors = self.state.bond_sectors
        stable = 0
        tick = time.perf_counter()
        def observe(state, sweep):
            nonlocal stable, tick
            if state.bond_sectors != sectors:
                raise RuntimeError('One-site polishing changed the selected multiplet sectors')
            residual = max(u.residual_norm for u in sweep.updates)
            if not all(u.accepted and u.local_converged and not u.recovery_reason
                       for u in sweep.updates):
                raise RuntimeError('One-site polishing did not converge every local solve')
            energy = float(sweep.energy)
            delta = energy-self.energy
            if not np.isfinite(energy) or delta > 1e-10:
                raise RuntimeError('One-site polishing increased energy or returned nonfinite energy')
            self.state = state
            self.sites = ReducedFrontier.from_state(state).to_mps(state)
            self.energy = energy
            row = dict(sweep=len(self.history)+1, stage='one_site', direction=sweep.direction,
                       energy=energy, delta_energy=delta, preparation_seconds=0.,
                       solve_seconds=time.perf_counter()-tick, verification_seconds=0.,
                       max_residual=float(residual), elapsed_seconds=time.perf_counter()-start,
                       bonds=[], cycle_handoff=self.cycle_handoff)
            self.history.append(row)
            if callback is not None:
                callback(row)
            stable = stable+1 if abs(delta) <= energy_tolerance else 0
            tick = time.perf_counter()
            if stable >= 2:
                self.success = True
                raise _PolishPlateau()
        try:
            result = reduced_letta_dmrg(self.hamiltonian, state=self.state,
                options=LETTADMROptions(max_sweeps=max_sweeps, tolerance=0.,
                    eigensolver_tolerance=residual_tolerance,
                    eigensolver_max_iterations=max_iterations, metric_tolerance=1e-12,
                    energy_increase_tolerance=1e-11, gauge_mode='frontier',
                    dense_solver_threshold=32, start_direction=direction),
                sweep_callback=observe)
            self.success = bool(result.converged)
        except _PolishPlateau:
            pass
        self.message = ('One-site energy plateau after directional cycle' if self.success
                        else 'One-site polishing pass cap reached')
        self.elapsed_seconds = time.perf_counter()-start
        return self

    def run(self, *, max_sweeps=100, energy_tolerance=1e-9,
            residual_tolerance=1e-10, max_iterations=300, max_space=96, callback=None, start_direction="lr",
            polish_on_cycle=True, cycle_tolerance=1e-6, one_site_max_sweeps=100,
            start_stage="two_site"):
        if max_sweeps < 1 or energy_tolerance < 0 or residual_tolerance <= 0:
            raise ValueError('Invalid stopping parameters')
        if start_direction not in {'lr', 'rl'}:
            raise ValueError('start_direction must be lr or rl')
        if not np.isfinite(cycle_tolerance) or cycle_tolerance <= 0 or one_site_max_sweeps < 1 or start_stage not in {'two_site', 'one_site'}:
            raise ValueError('Invalid cycle/polishing options')
        if start_stage == 'one_site' and energy_tolerance <= 0:
            raise ValueError('One-site polishing requires a positive energy tolerance')
        if start_stage == 'two_site' and any(np.iscomplexobj(b) for s in self.sites for b in s.data.values()):
            raise ValueError('Two-site optimization requires real initial tensors')
        start = time.perf_counter()
        self.cycle_handoff = None
        self.success = False
        self.history = []
        self.hamiltonian = self.problem.su2_mpo()
        self.energy = self._energy()
        self.initial_energy = self.energy
        self.sites = (right_canonicalize_sites if start_direction == "lr" else left_canonicalize_sites)(self.sites)
        if abs(self._energy()-self.energy) > 1e-10*max(1.,abs(self.energy)):
            raise RuntimeError('Initial exact gauge changed energy')
        if start_stage == 'one_site':
            self.setup_seconds = time.perf_counter()-start
            return self._polish(one_site_max_sweeps, energy_tolerance, residual_tolerance,
                                max_iterations, callback, start_direction, start)
        nelec = sum(self.problem.nelec)
        two_s = abs(self.problem.nelec[0]-self.problem.nelec[1])
        owner = load_engine().SU2MovingEnvironment(self.problem.h1, self.problem.eri,
                 nelec, two_s=two_s, ecore=self.problem.ecore, cutoff=0., include_half=True)
        owner.set_num_threads(self.threads)
        owner.install_mps(self.sites)
        self.setup_seconds = time.perf_counter()-start
        self.history = []
        self.success = False
        stable = 0
        cycle = _DirectionalCycle(cycle_tolerance)
        endpoints = {}
        for sweep in range(max_sweeps):
            direction = start_direction if sweep % 2 == 0 else ('rl' if start_direction == 'lr' else 'lr')
            tick = time.perf_counter()
            owner.begin_half_sweep(direction, self.problem.norb)
            owner.prepare_owned_half_sweep()
            prepared = time.perf_counter()
            records = owner.execute_owned_half_sweep(max_bond=self.max_bond, max_bond_mode='reduced',
                cutoff=0., davidson_tolerance=residual_tolerance, max_iterations=max_iterations,
                max_space=max_space, accept_unconverged=False)
            owner.finish_half_sweep()
            solved = time.perf_counter()
            self.sites = _export_sites(owner)
            if any(len(s.qns[2]) > self.max_bond for s in self.sites[:-1]):
                raise RuntimeError('Engine exceeded total multiplet cap')
            residual = max(float(r['solve']['residual_norm']) for r in records)
            if not np.isfinite(residual) or residual > residual_tolerance*1.01 or not all(r['solve']['converged'] for r in records):
                raise RuntimeError('Engine did not meet requested local residual tolerance')
            energy = self._energy()
            if not np.isfinite(energy):
                raise RuntimeError('Nonfinite independent energy')
            delta = energy-self.energy
            self.energy = energy
            selected = cycle.observe(direction, energy) if polish_on_cycle and energy_tolerance > 0 else None
            if polish_on_cycle and energy_tolerance > 0:
                endpoints[direction] = (energy, [s.copy() for s in self.sites], sweep+1)
            if selected is not None:
                self.cycle_handoff = dict(selected_direction=selected,
                    selected_energy=endpoints[selected][0], selected_pass=endpoints[selected][2],
                    endpoint_energies=dict(cycle.energies), same_direction_changes=dict(cycle.changes),
                    endpoint_gap=abs(cycle.energies['lr']-cycle.energies['rl']),
                    tolerance=cycle_tolerance)
            row = dict(sweep=sweep+1, stage='two_site', direction=direction, energy=energy, delta_energy=delta,
                       same_direction_delta=cycle.changes.get(direction), cycle_handoff=self.cycle_handoff,
                       preparation_seconds=prepared-tick, solve_seconds=solved-prepared,
                       verification_seconds=time.perf_counter()-solved, max_residual=residual,
                       elapsed_seconds=time.perf_counter()-start, bonds=records)
            self.history.append(row)
            if callback is not None:
                callback(row)
            if selected is not None:
                saved_energy, self.sites, _ = endpoints[selected]
                self.energy = self._energy()
                if abs(self.energy-saved_energy) > 1e-9:
                    raise RuntimeError('Restored cycle endpoint energy mismatch')
                del owner, endpoints
                return self._polish(one_site_max_sweeps, energy_tolerance, residual_tolerance,
                    max_iterations, callback, 'rl' if selected == 'lr' else 'lr', start)
            stable = stable+1 if abs(delta) <= energy_tolerance else 0
            if stable >= 2:
                self.success = True
                break
        self.message = 'Energy plateau with converged local roots' if self.success else 'Directional sweep cap reached'
        self.elapsed_seconds = time.perf_counter()-start
        return self

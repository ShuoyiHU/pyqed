"""Small-system accuracy audit with independent physical-state energies.

Dense Hilbert-space contractions here are validation only, never solver inputs.
Run with PYTHONPATH=. and one BLAS thread. Outputs include final tensors so a
different method can restart from exactly the same endpoint.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
    make_shared_initial_state, _hash_arrays,
)


def run(args):
    model = build_model(args.model, '2d', tuple(args.shape))
    if model.physical_dim ** model.nsites > 4096:
        raise ValueError('This independent dense validation is limited to 4096 states.')
    initial = make_shared_initial_state(model, bond_dim=args.bond, seed=args.seed).letta
    if args.resume:
        with np.load(args.resume) as arrays:
            initial.tensors = [arrays[f'tensor_{i}'].copy() for i in range(initial.nsites)]
    hamiltonian = model.mpo.to_dense()
    exact = float(np.linalg.eigvalsh(hamiltonian)[0])

    def physical_energy(state):
        vector = state.state_vector()
        return float((np.vdot(vector, hamiltonian @ vector) / np.vdot(vector, vector)).real)

    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(model=args.model, shape=args.shape, bond=args.bond, seed=args.seed,
                  initial_hash=_hash_arrays(initial.tensors), initial_energy=physical_energy(initial),
                  exact_ground_energy=exact, results=[])
    for method in args.methods:
        common = dict(max_sweeps=args.sweeps, tolerance=1e-12, metric_tolerance=1e-10,
                      eigensolver_tolerance=1e-11)
        if method == 'two-site':
            options = LETTATwoSiteOptions(**common, energy_refinement_max_iterations=args.inner)
            solve = lambda: letta_two_site_dmrg(model.mpo, state=initial, bond_dim=args.bond,
                                                options=options)
        else:
            options = LETTADMROptions(**common, cbe_enabled=method == 'cbe', cbe_selector='shrewd',
                                     cbe_energy_refinement_max_iterations=args.inner)
            solve = lambda: letta_dmrg(model.mpo, state=initial, options=options)
        started = perf_counter()
        result = solve()
        elapsed = perf_counter() - started
        energies = [report['initial_energy']] + [s.energy for s in result.history]
        record = dict(method=method, options=asdict(options), energy=result.energy,
                      physical_energy=physical_energy(result.state), sweeps=result.sweeps,
                      energy_change_stopping=result.converged, elapsed_seconds=elapsed,
                      stopping_message=result.message,
                      max_sweep_energy_increase=float(max(np.diff(energies), default=0.)),
                      history=energies)
        # A fresh complete one-site pass is a descent diagnostic, not a proof
        # of joint-factor stationarity or a globally minimal fixed-D energy.
        probe = letta_dmrg(model.mpo, state=result.state, options=LETTADMROptions(
            max_sweeps=2, tolerance=1e-14, metric_tolerance=1e-12,
            eigensolver_tolerance=1e-12))
        record['two_pass_polish_energy'] = physical_energy(probe.state)
        record['two_pass_polish_gain'] = record['physical_energy'] - record['two_pass_polish_energy']
        np.savez_compressed(args.output / f'{method}_final.npz',
                            **{f'tensor_{i}': a for i, a in enumerate(result.state.tensors)})
        report['results'].append(record)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k not in {'options', 'history'}}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='bose_hubbard')
    parser.add_argument('--shape', type=int, nargs=2, default=[2, 2])
    parser.add_argument('--bond', type=int, default=2)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--sweeps', type=int, default=100)
    parser.add_argument('--inner', type=int, default=32)
    parser.add_argument('--methods', nargs='+', choices=['one-site', 'cbe', 'two-site'],
                        default=['one-site', 'cbe', 'two-site'])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

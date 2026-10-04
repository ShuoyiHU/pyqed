"""Small public-API LETTA example; no dense many-body solver in the active path.

Run from the repository root with PYTHONPATH=. and one BLAS/OpenMP thread.
"""
import argparse
from collections import Counter
import json

from pyqed.letta import (MetricCompressionOptions, OptimizationOptions,
                        hubbard, bose_hubbard, heisenberg, random_state, solve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['hubbard', 'bose', 'heisenberg'], default='hubbard')
    parser.add_argument('--sites', type=int, default=4)
    parser.add_argument('--symmetry', choices=['u1', 'su2'])
    parser.add_argument('--periodic-hamiltonian', action='store_true')
    parser.add_argument('--topology', choices=['open', 'ring'], default='open')
    parser.add_argument('--ties', choices=['none', 'nn', 'nn-periodic'], default='nn')
    parser.add_argument('--method', choices=['one-site', 'cbe', 'two-site'], default='cbe')
    parser.add_argument('--copies-per-sector', type=int, default=1)
    parser.add_argument('--bond-cap', type=int)
    parser.add_argument('--sweeps', type=int, default=4)
    parser.add_argument('--compression', choices=['als', 'variable-projection', 'joint-ls', 'grassmann-newton'], default='als')
    parser.add_argument('--als-rounds', type=int, default=32)
    parser.add_argument('--lsmr-iterations', type=int, default=512)
    parser.add_argument('--nonlinear-iterations', type=int, default=100)
    parser.add_argument('--energy-rounds', type=int, default=32)
    parser.add_argument('--seed', type=int, default=71)
    args = parser.parse_args()
    symmetry = args.symmetry or ('u1' if args.model == 'bose' else 'su2')
    common = dict(periodic=args.periodic_hamiltonian)
    if args.model == 'hubbard':
        model = hubbard(args.sites, nelec=((args.sites+1)//2, args.sites//2),
                        symmetry=symmetry, **common)
    elif args.model == 'heisenberg':
        model = heisenberg(args.sites, symmetry=symmetry, **common)
    else:
        if symmetry != 'u1':
            parser.error('the spinless Bose-Hubbard model uses U(1) particle number')
        model = bose_hubbard(args.sites, particles=args.sites, max_occupancy=2, **common)
    state = random_state(model, topology=args.topology, ties=args.ties,
                         multiplets_per_sector=args.copies_per_sector, seed=args.seed)
    options = OptimizationOptions(max_sweeps=args.sweeps, energy_refinement_rounds=args.energy_rounds,
        compression=MetricCompressionOptions(solver=args.compression,
            als_max_iterations=args.als_rounds, lsmr_max_iterations=args.lsmr_iterations,
            max_iterations=args.nonlinear_iterations))
    result = solve(model, state=state, method=args.method, options=options, bond_dim=args.bond_cap)
    updates = [u for sweep in result.history for u in sweep.updates]
    print(json.dumps(dict(model=model.name, symmetry=symmetry, topology=args.topology,
        periodic_hamiltonian=args.periodic_hamiltonian, ties=args.ties, method=args.method,
        energy=result.energy, converged=result.converged, message=result.message,
        sweeps=result.sweeps, sweep_energies=[s.energy for s in result.history],
        initial_bond_multiplets=[len(b) for b in state.bond_sectors],
        final_bond_multiplets=[len(b) for b in result.state.bond_sectors],
        final_sectors=[{str(q): count for q, count in Counter(b).items()} for b in result.state.bond_sectors],
        numerical_recoveries=sum(bool(getattr(u, 'recovery_reason', None) or
                                      getattr(u, 'cbe_recovery_reason', None)) for u in updates)), indent=2))


if __name__ == '__main__':
    main()

"""Local Bose-Hubbard compression experiments; all dispatch is scoped here.

Run with PYTHONPATH=. and single-thread BLAS. Data go to --output, normally
/private/tmp. Each case/method is a separate resumable command.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from .. import cbe
from ..contractions import BlockDiagonalMetric
from ..solver import LETTADMROptions, letta_dmrg
from .condensed_models import build_model
from .condensed_runner import make_shared_initial_state, _letta_one_site_record, _hash_arrays
from .metric_compression_solvers import METHODS, solve, WeightedProblem


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(args):
    shape = tuple(map(int, args.shape.split(',')))
    name = f'bose{shape[0]}{shape[1]}_d{args.bond}_{args.method}'
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (name+'.json')
    if destination.exists():
        raise FileExistsError(destination)
    model = build_model('bose_hubbard', '2d', shape)
    initial = make_shared_initial_state(model, bond_dim=args.bond, seed=args.seed)
    initial_hash = _hash_arrays(initial.letta.tensors)
    dense = model.mpo.to_dense(max_sites=model.nsites)
    exact = float(np.linalg.eigvalsh(dense)[0])
    options = LETTADMROptions(
        max_sweeps=args.sweeps, tolerance=1e-12, metric_tolerance=1e-10,
        cbe_enabled=args.method != 'one', cbe_selector='shrewd',
        cbe_baseline_guard_fraction=args.acceptance_fraction,
        cbe_energy_refinement_max_iterations=args.energy_passes,
        verbosity=1)
    traces, samples = [], []
    rng = np.random.default_rng(6081)
    original_update = cbe._strict_shrewd_cbe_bond_update
    updates = []

    def fit(target, metric, rank, **kwargs):
        nonlocal samples
        fit_id = len(traces)
        if args.method == 'baseline':
            sample = (np.array(target), metric.to_dense(), rank, fit_id)
            if len(samples) < 12:
                samples.append(sample)
            else:
                index = int(rng.integers(fit_id+1))
                if index < len(samples):
                    samples[index] = sample
        result = solve(target, metric, rank, args.method,
                       metric_tolerance=kwargs['metric_tolerance'])
        traces.append(dict(fit=fit_id, shape=list(target.shape), rank=rank,
                           loss=result.loss, iterations=result.iterations,
                           **result.diagnostics))
        return result.left, result.right, result.loss, result.iterations

    def update(*a, **kw):
        result = original_update(*a, **kw)
        fields = ('cbe_old_energy', 'cbe_expanded_energy', 'cbe_trimmed_energy',
                  'cbe_refined_energy', 'cbe_incumbent_refined_energy',
                  'cbe_baseline_energy', 'cbe_energy_refinement_start', 'cbe_fallback',
                  'cbe_trim_loss', 'cbe_timings')
        updates.append({key: getattr(result, key) for key in fields})
        return result

    print('START', name, flush=True)
    start = perf_counter()
    with patch.object(cbe, '_metric_low_rank_factorization', fit), \
         patch.object(cbe, '_strict_shrewd_cbe_bond_update', update):
        result = letta_dmrg(model.mpo, state=initial.letta, options=options)
    elapsed = perf_counter()-start
    assert _hash_arrays(initial.letta.tensors) == initial_hash
    record = _letta_one_site_record(result, args.method, elapsed, initial.fingerprint, exact)
    physical = float(result.state.expectation(model.mpo))
    assert abs(physical - result.energy) < 1e-7
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__), Path(cbe.__file__),
             Path(__file__).with_name('metric_compression_solvers.py'),
             Path(__file__).with_name('grassmann_compression.py'),
             Path(cbe.__file__).with_name('solver.py')]
    output = dict(config=vars(args), parameters=dict(model.parameters),
                  options=asdict(options), exact_energy=exact,
                  physical_energy=physical, initial_energy=initial.energy,
                  record=record, fits=traces, updates=updates,
                  source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files})
    write_json(destination, output)
    snapshots = folder / 'snapshots'
    snapshots.mkdir(exist_ok=True)
    for index, (target, metric, rank, fit_id) in enumerate(samples):
        np.savez_compressed(snapshots / f'{name}_{index:02d}.npz', target=target,
                            metric=metric, rank=rank, fit_id=fit_id)
    print('DONE', name, result.energy, elapsed, 'fits', len(traces), flush=True)


def replay(args):
    source = Path(args.source)
    records = []
    for snapshot in sorted(source.glob('*.npz')):
        data = np.load(snapshot)
        target, dense = data['target'], data['metric']
        metric = BlockDiagonalMetric(target.size, [dense], [np.arange(target.size)])
        rank = int(data['rank'])
        p = WeightedProblem(target, metric, rank)
        methods = METHODS + ('als200_2000', 'als40_dense', 'varpro_trf_tight', 'varpro_lm_tight',
                             'joint_trf_tight', 'grassmann_newton_tight',
                             'varpro_trf_small_tight')
        for method in methods:
            if method.endswith('_tight'):
                fit = solve(target, metric, rank, method.removesuffix('_tight'),
                            max_nfev=1000, tolerance=1e-13)
            else:
                fit = solve(target, metric, rank, method)
            q, transfer = np.linalg.qr(fit.left, mode='reduced')
            residual = p.residual(q, transfer @ fit.right)
            gradient = p.joint_jacobian(q, transfer @ fit.right).T @ residual
            approximation = fit.left @ fit.right
            u, _, vh = np.linalg.svd(approximation, full_matrices=False)
            u, v = u[:, :rank], vh[:rank].conj().T
            physical_gradient = (metric @ (approximation-target).ravel()).reshape(target.shape)
            projected = (u @ (u.conj().T @ physical_gradient)
                         + physical_gradient @ v @ v.conj().T
                         - u @ (u.conj().T @ physical_gradient @ v) @ v.conj().T)
            records.append(dict(snapshot=snapshot.name, fit_id=int(data['fit_id']),
                shape=list(target.shape), rank=rank, method=method,
                loss=fit.loss, iterations=fit.iterations,
                relative_loss=fit.loss/max(float(p.b@p.b), 1e-300),
                projected_gradient_relative=float(np.linalg.norm(projected)
                    / max(np.linalg.norm(metric @ target.ravel()), 1e-300)),
                **dict(fit.diagnostics, stationarity=float(np.linalg.norm(gradient)))))
            print(snapshot.name, method, fit.loss, flush=True)
        write_json(Path(args.output)/'replay.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('run')
    p.add_argument('--shape', default='2,2')
    p.add_argument('--bond', type=int, default=2)
    p.add_argument('--method', choices=('one',)+METHODS, default='baseline')
    p.add_argument('--sweeps', type=int, default=50)
    p.add_argument('--seed', type=int, default=731)
    p.add_argument('--energy-passes', type=int, default=3)
    p.add_argument('--acceptance-fraction', type=float, default=0.0)
    p.add_argument('--output', required=True)
    p = subs.add_parser('replay')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else replay)(args)


if __name__ == '__main__':
    main()

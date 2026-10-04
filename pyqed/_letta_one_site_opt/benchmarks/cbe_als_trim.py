"""Click-run comparison of weighted SVD and physical-norm ALS truncation.

Only non-negligible conditional D-truncations are replaced. Selection,
expanded solves, acceptance guards, and production defaults are unchanged.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import json
import time

import numpy as np

from pyqed._letta_one_site_opt import cbe
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'output/benchmarks/cbe_harder_cases/fermi_hubbard_6_6_L6_D6_energy_difference.json'


def run(source=SOURCE, output=None):
    source = Path(source)
    case = json.loads(source.read_text())
    output = Path(output) if output else ROOT / 'output/benchmarks/cbe_als_trim' / datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    original_trim = cbe._metric_trim_factorization
    original_update = cbe._strict_shrewd_cbe_bond_update
    options = {key: case[key] for key in (
        'dimension', 'bond_dim', 'expansion_dimension', 'cbe_baseline_guard_fraction',
        'max_sweeps', 'seed', 'tolerance', 'eigensolver_tolerance',
        'eigensolver_max_iterations', 'cbe_conditional_trim')}
    options.update(size=case['nsites'], model_parameters=case['parameters'],
                   exact_max_dimension=0, solvers=['letta_cbe_strict'], raise_on_failure=True)
    results = {}
    # Four iterations matches the existing ALS budget; 20 checks budget sensitivity.
    for mode, budget in [('weighted_svd', 4), ('als_weighted', 4), ('als_simple', 4), ('als_simple', 20)]:
        name = f'{mode}_{budget}' if mode != 'weighted_svd' else mode
        trims, updates = [], []

        def trim(target, metric, rank, **kwargs):
            started = time.perf_counter()
            if mode == 'weighted_svd':
                result = original_trim(target, metric, rank, **kwargs)
                initial_loss = result[2]
            else:
                if mode == 'als_weighted':
                    initial = original_trim(target, metric, rank, **kwargs)[:2]
                else:
                    u, s, vh = np.linalg.svd(target, full_matrices=False)
                    keep = min(rank, len(s))
                    left = np.zeros((target.shape[0], rank), dtype=target.dtype)
                    right = np.zeros((rank, target.shape[1]), dtype=target.dtype)
                    left[:, :keep], right[:keep] = u[:, :keep], s[:keep, None] * vh[:keep]
                    initial = left, right
                initial_loss = cbe._one_site_factorization_loss(target, *initial, metric)
                als_options = dict(kwargs, max_iterations=budget)
                result = (*cbe._metric_low_rank_factorization(
                    target, metric, rank, initial_factors=initial, **als_options), mode)
            trims.append(dict(initial_loss=float(initial_loss), final_loss=float(result[2]),
                              iterations=int(result[3]), seconds=time.perf_counter()-started))
            return result

        def update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            if not result.cbe_fallback:
                reason = 'accepted'
            elif result.cbe_expanded_energy is None:
                reason = 'early_fallback'
            elif result.cbe_trimmed_energy > result.cbe_old_energy + 1.e-9:
                reason = 'trim_above_old'
            else:
                reason = 'other_late_fallback'
            updates.append(dict(reason=reason, old=result.cbe_old_energy,
                                expanded=result.cbe_expanded_energy,
                                trimmed=result.cbe_trimmed_energy,
                                baseline=result.cbe_baseline_energy))
            return result

        print('START', name, flush=True)
        with patch.object(cbe, '_metric_trim_factorization', trim), patch.object(cbe, '_strict_shrewd_cbe_bond_update', update):
            result = run_benchmark(case['model'], **options)
        record = result['records'][0]
        counts = dict(Counter(u['reason'] for u in updates))
        assert len(updates) == record['cbe_updates']
        assert counts.get('accepted', 0) == record['cbe_accepted']
        assert all(t['final_loss'] <= t['initial_loss'] + 1.e-12 for t in trims)
        results[name] = dict(record=record, counts=counts, trims=trims, updates=updates)
        (output / f'{name}.json').write_text(json.dumps(results[name], indent=2) + '\n')
        print('DONE', name, 'energy', record['energy'], counts, flush=True)
    saved = next(r for r in case['records'] if r['solver'] == 'letta_cbe_strict')
    baseline = results['weighted_svd']['record']
    delta = float(np.max(np.abs(np.array(saved['sweep_energies']) - baseline['sweep_energies'])))
    assert delta < 1.e-8
    assert baseline['sweep_cbe_accepted'] == saved['sweep_cbe_accepted']
    assert len({r['record']['initial_state_fingerprint'] for r in results.values()}) == 1
    summary = dict(source=str(source), case=case, baseline_max_energy_difference=delta,
                   results=results)
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Physical-norm ALS truncation experiment', '',
             f"Fermi–Hubbard, L={case['nsites']}, D={case['bond_dim']}, seed={case['seed']}; t=1, U=4, mu=2.", '',
             'Same initial state, 50-sweep cap, expansion dimension 1, and acceptance guard. Exact CBE remains off.',
             'ALS minimizes physical-norm compression error, not energy. Negligible-loss SVD shortcuts remain enabled.',
             f'Baseline replay maximum sweep-energy difference: {delta:.3g}.', '',
             '| Truncation | Sweeps | CBE accepted/attempts | Early fallback | Trim above old | Other late | Final energy | Error vs exact | Seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, data in results.items():
        r, c = data['record'], data['counts']
        lines.append(f"| {name} | {r['sweeps']} | {r['cbe_accepted']}/{r['cbe_updates']} | {c.get('early_fallback',0)} | {c.get('trim_above_old',0)} | {c.get('other_late_fallback',0)} | {r['energy']:.12f} | {r['energy']-case['exact_energy']:.6g} | {r['elapsed_seconds']:.2f} |")
    lines += ['', '## ALS stopping diagnostics', '']
    for name, data in results.items():
        if name == 'weighted_svd':
            continue
        trims = data['trims']
        lines.append(f"- {name}: {len(trims)} non-negligible sector fits; ALS iterations {min(t['iterations'] for t in trims)}–{max(t['iterations'] for t in trims)}; summed initial/final physical loss {sum(t['initial_loss'] for t in trims):.12g}/{sum(t['final_loss'] for t in trims):.12g}.")
    lines += ['', 'Times are single instrumented runs, not repeated timing measurements.',
              'Per-truncation initial/final losses, iteration counts, update energies, and per-sweep acceptance counts are in the JSON files.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('OUTPUT', output, flush=True)
    return output


if __name__ == '__main__':
    run()

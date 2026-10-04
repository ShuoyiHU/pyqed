"""Click-run fixed-D energy ALS after CBE's weighted-SVD initialization.

Benchmark-only hooks; no production dispatch changes. The original-pair arm
controls for extra local energy solves without using the expanded candidate.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import inspect
import json
import time
import numpy as np

from pyqed._letta_one_site_opt import cbe, solver
from pyqed._letta_one_site_opt.benchmarks.cbe_als_trim import SOURCE, ROOT
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark


def energy_polish(context, left, right, passes):
    """Alternating generalized eigenproblems; retain only energy-decreasing steps."""
    state = context['state']
    layout = context['layout']
    i, j = layout.left_site, layout.left_site + 1
    hc, nc = context['hamiltonian_cache'], context['metric_cache']
    hl, hr = context['hamiltonian_left'], context['hamiltonian_right']
    nl, nr = context['metric_left'], context['metric_right']
    options = context['options']
    saved = state.tensors[i], state.tensors[j]
    state.tensors[i], state.tensors[j] = left.copy(), right.copy()
    def energy():
        return cbe._streamed_bond_energy(hc, nc, hl, hr, nl, nr, layout)
    initial_energy, _ = energy()
    current_energy = initial_energy
    steps = []
    try:
        order = (i, j) if context['direction'] == 'lr' else (j, i)
        for _ in range(passes):
            previous = current_energy
            for site in order:
                old_tensor = state.tensors[site].copy()
                if site == i:
                    local = hl, hc.extend_right(hr, j), nl, nc.extend_right(nr, j)
                else:
                    local = hc.extend_left(hl, i), hr, nc.extend_left(nl, i), nr
                update = solver._update_from_cached_environments(
                    state, site, hc, nc, *local, options)
                proposed, norm = energy()
                accepted = bool(np.isfinite(proposed) and norm > 0 and proposed <= current_energy)
                if accepted:
                    current_energy = proposed
                else:
                    state.tensors[site] = old_tensor
                steps.append(dict(site=site, proposed_energy=proposed, accepted=accepted,
                                  energy=current_energy, hamiltonian_applications=update.hamiltonian_applications))
            if previous - current_energy <= 1.e-10:
                break
        final_energy, norm = energy()
        assert final_energy <= initial_energy + 1.e-12
        assert state.tensors[i].shape == left.shape and state.tensors[j].shape == right.shape
        return (state.tensors[i].copy(), state.tensors[j].copy(), norm,
                dict(initial_energy=initial_energy, final_energy=final_energy, steps=steps))
    finally:
        state.tensors[i], state.tensors[j] = saved


def run(source=SOURCE, output=None):
    case = json.loads(Path(source).read_text())
    output = Path(output) if output else ROOT / 'output/benchmarks/cbe_energy_trim' / datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    original_trim, original_update = cbe._directional_one_site_metric_trim, cbe._strict_shrewd_cbe_bond_update
    signature = inspect.signature(original_update)
    options = {k: case[k] for k in ('dimension', 'bond_dim', 'expansion_dimension',
        'cbe_baseline_guard_fraction', 'max_sweeps', 'seed', 'tolerance',
        'eigensolver_tolerance', 'eigensolver_max_iterations', 'cbe_conditional_trim')}
    options.update(size=case['nsites'], model_parameters=case['parameters'],
                   cbe_energy_refinement_max_iterations=0,
                   exact_max_dimension=0, solvers=['letta_cbe_strict'], raise_on_failure=True)
    results = {}
    for name, passes, initialization in [('weighted_svd', 0, 'svd'),
            ('energy_svd_1', 1, 'svd'), ('energy_svd_3', 3, 'svd'),
            ('energy_original_1', 1, 'original')]:
        traces, polishes = [], []
        context = None
        def trim(*args, **kwargs):
            seed = original_trim(*args, **kwargs)
            if not passes:
                return seed
            left, right = (seed.left_tensor, seed.right_tensor) if initialization == 'svd' else context['original_pair']
            started = time.perf_counter()
            left, right, norm, details = energy_polish(context, left, right, passes)
            details.update(seconds=time.perf_counter()-started, seed_metric_loss=seed.loss)
            polishes.append(details)
            # The original norm-compression loss is not the energy-polished loss.
            return replace(seed, left_tensor=left, right_tensor=right, norm=norm,
                           loss=float('nan'), iterations=len(details['steps']),
                           metric_kinds=('energy-als',))
        def update(*args, **kwargs):
            nonlocal context
            context = signature.bind(*args, **kwargs).arguments
            state, layout = context['state'], context['layout']
            i = layout.left_site
            context['original_pair'] = state.tensors[i].copy(), state.tensors[i+1].copy()
            result = original_update(*args, **kwargs)
            if not result.cbe_fallback:
                reason = 'accepted'
            elif result.cbe_expanded_energy is None:
                reason = 'early_fallback'
            elif result.cbe_trimmed_energy > result.cbe_old_energy + 1.e-9:
                reason = 'trim_above_old'
            else:
                reason = 'other_late_fallback'
            traces.append(dict(reason=reason, old=result.cbe_old_energy,
                expanded=result.cbe_expanded_energy, trimmed=result.cbe_trimmed_energy,
                baseline=result.cbe_baseline_energy))
            return result
        print('START', name, flush=True)
        with patch.object(cbe, '_directional_one_site_metric_trim', trim), patch.object(cbe, '_strict_shrewd_cbe_bond_update', update):
            result = run_benchmark(case['model'], **options)
        record = result['records'][0]
        counts = dict(Counter(t['reason'] for t in traces))
        assert counts.get('accepted', 0) == record['cbe_accepted']
        assert len(traces) == record['cbe_updates']
        results[name] = dict(record=record, counts=counts, polishes=polishes, updates=traces)
        # The runner may aggregate the unavailable polished norm-loss as NaN.
        def clean(value):
            if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list): return [clean(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value): return None
            return value
        results[name] = clean(results[name])
        (output / f'{name}.json').write_text(json.dumps(results[name], indent=2, allow_nan=False)+'\n')
        print('DONE', name, record['energy'], record['sweeps'], counts, flush=True)
    saved = next(r for r in case['records'] if r['solver']=='letta_cbe_strict')
    base = results['weighted_svd']['record']
    np.testing.assert_allclose(base['sweep_energies'], saved['sweep_energies'], rtol=0, atol=1.e-10)
    assert base['sweep_cbe_accepted'] == saved['sweep_cbe_accepted']
    assert len({v['record']['initial_state_fingerprint'] for v in results.values()}) == 1
    (output/'results.json').write_text(json.dumps(dict(source=str(source),case=case,results=results), indent=2, allow_nan=False)+'\n')
    lines = ['# Fixed-D energy-aware CBE truncation', '',
        'Fermi–Hubbard L=6, D=6, seed 731; t=1, U=4, mu=2. Expansion dimension 1, sweep cap 50.', '',
        'Energy ALS alternates one-site generalized eigenproblems at fixed D with the outer environment fixed. Each step is retained only if the streamed physical energy does not increase. The original CBE safety and baseline guards still apply.', '',
        'The original-pair control uses the same extra optimization when CBE selection succeeds, but initializes from the pre-expansion pair. Its accepted count measures acceptance of this control candidate, not benefit from expansion.', '',
        '| Method | Sweeps | Accepted/attempts | Early fallback | Above old | Other late | Final energy | Error vs exact | Seconds | Extra local solves |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, data in results.items():
        r,c = data['record'], data['counts']
        lines.append(f"| {name} | {r['sweeps']} | {r['cbe_accepted']}/{r['cbe_updates']} | {c.get('early_fallback',0)} | {c.get('trim_above_old',0)} | {c.get('other_late_fallback',0)} | {r['energy']:.12f} | {r['energy']-case['exact_energy']:.6g} | {r['elapsed_seconds']:.2f} | {sum(len(p['steps']) for p in data['polishes'])} |")
    lines += ['', 'Single-run timings are indicative. Extra ALS Hamiltonian applications are recorded in polishes.steps and are not included in the production runner application counter.',
        'Polished physical-norm compression loss is unavailable (null); seed_metric_loss records the initializer loss. Production defaults and existing benchmark figures are unchanged.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('OUTPUT', output, flush=True)
    return output


if __name__ == '__main__':
    run()

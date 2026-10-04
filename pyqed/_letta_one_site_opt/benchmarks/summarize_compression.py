"""Verify and summarize completed strict-acceptance local compression runs."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .metric_compression_solvers import METHODS

LABELS = dict(one='One-site', baseline='ALS 4 / LSMR 40', als40_40='ALS 40 / LSMR 40',
              als4_400='ALS 4 / LSMR 400', als40_400='ALS 40 / LSMR 400',
              varpro_trf='Variable projection · TRF', varpro_lm='Variable projection · LM',
              joint_trf='Joint factors · TRF', grassmann_newton='Grassmann chart · Newton',
              varpro_trf_small='Variable projection · smaller side',
              grassmann_newton_balanced='Grassmann Newton · balanced factors',
              varpro_trf_balanced='Variable projection TRF · balanced',
              varpro_lm_balanced='Variable projection LM · balanced',
              varpro_trf_small_balanced='Smaller-side variable projection · balanced')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [('22', 2), ('22', 3), ('22', 4), ('32', 3)]
    records, all_data, failures = [], [], []
    for shape, bond in cases:
        fingerprints = set()
        for method in ('one',)+METHODS:
            path = source / f'bose{shape}_d{bond}_{method}.json'
            if not path.exists():
                log = path.with_suffix('.log')
                content = log.read_text()
                if 'Traceback (most recent call last)' not in content or 'numpy.linalg.LinAlgError:' not in content:
                    raise RuntimeError(f'No completed result or verified numerical failure: {path}')
                history = re.findall(r'lattice LETTA sweep\s+(\d+).*?energy=([-\d.]+)', content)
                failure = dict(case=f'{shape[0]}x{shape[1]} D={bond}', shape=shape,
                    bond=bond, method=method, failed=True,
                    last_completed_sweep=int(history[-1][0]) if history else 0,
                    last_reported_energy=float(history[-1][1]) if history else None,
                    error=content.strip().splitlines()[-1])
                failures.append(failure)
                records.append(failure)
                shutil.copy2(log, output/log.name)
                continue
            data = json.loads(path.read_text())
            assert data['options']['cbe_baseline_guard_fraction'] == 0
            assert data['config']['sweeps'] == 100
            assert data['options']['cbe_energy_refinement_max_iterations'] == 3
            r, fits = data['record'], data['fits']
            assert abs(data['physical_energy']-r['energy']) < 1e-7
            assert r['energy'] >= data['exact_energy']-1e-7
            fingerprints.add(r['initial_state_fingerprint'])
            inner = [s for f in fits for s in f.get('lsmr', [])]
            counts = Counter(s['stop'] for s in inner)
            records.append(dict(case=f'{shape[0]}x{shape[1]} D={bond}', shape=shape,
                bond=bond, method=method, energy=r['energy'], exact=data['exact_energy'],
                error=r['energy']-data['exact_energy'], seconds=r['elapsed_seconds'],
                max_sweep_energy_increase=max(0., max(np.diff(r['sweep_energies']), default=0.)),
                sweeps=r['sweeps'], converged=r['converged'], accepted=r['cbe_accepted'],
                attempts=r['cbe_updates'], general_fits=len(fits),
                raw_trim_above_old=sum(u['cbe_trimmed_energy'] is not None
                    and u['cbe_trimmed_energy'] > u['cbe_old_energy']+1e-9
                    for u in data['updates']),
                raw_trims=sum(u['cbe_trimmed_energy'] is not None for u in data['updates']),
                accepted_incumbent=r['cbe_incumbent_refinements_selected'],
                inner_solves=len(inner), inner_stops=dict(counts),
                fit_seconds=sum(f['seconds'] for f in fits),
                nonlinear_budget_exits=sum(f.get('status') == (1 if method.startswith('grassmann_newton') else 0)
                                           for f in fits if 'status' in f),
                phase_seconds=r['cbe_phase_seconds']))
            all_data.append(data)
        assert len(fingerprints) == 1
    with gzip.open(output/'full_runs.json.gz', 'wt') as f:
        json.dump(all_data, f)
    (output/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
    (output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    lines = ['# Local comparison of correlated-metric CBE compression', '',
        'Bose–Hubbard: t=1, U=4, chemical potential=2, maximum occupancy=2; open lattice boundaries.',
        'Seed 731; identical initial tensors within each case; maximum 100 directional sweeps; energy-density stopping tolerance 1e-12.',
        'Strict acceptance: a CBE candidate must beat both the previous energy and the ordinary one-site update. All CBE arms retain the existing three-pass energy refinement and coupled-energy safeguards.',
        'Only the nonseparable physical-metric norm-compression solver changes. Exact separable fits and negligible-loss shortcuts remain in use.',
        'Each local process uses one BLAS/OpenMP thread. Times are single instrumented runs, not repeated timing estimates.', '',
        '## End-to-end results', '',
        '| Case | Compression | Energy | Error vs exact | Sweeps | Seconds | Accepted / attempted |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r.get('failed'):
            lines.append(f"| {r['case']} | {LABELS[r['method']]} | FAILED | — | {r['last_completed_sweep']} completed before failure | — | — |")
            continue
        lines.append(f"| {r['case']} | {LABELS[r['method']]} | {r['energy']:.12f} | {r['error']:.5g} | {r['sweeps']} | {r['seconds']:.2f} | {r['accepted']}/{r['attempts']} |")
    lines += ['', 'Negative errors smaller than 1e-7 are numerical noise, not energies below the true ground state.',
              'A sweep is one directional pass; early stopping gives different completed counts. Equal-sweep trajectories and elapsed times are retained in full_runs.json.gz.', '',
              '## Solver details', '',
              'ALS keeps the production loss-change tolerance 1e-10 while varying outer and inner iteration caps independently. The 200/2000 replay arm uses the same stopping tolerance.',
              'Variable-projection TRF and LM and joint-factor TRF use analytic Jacobians, at most 100 function evaluations, and ftol=xtol=gtol=1e-11. Replay-only tight arms use 1000 evaluations and 1e-13.',
              'Grassmann-chart Newton removes factor-basis redundancy, uses an analytic Schur-complement Hessian including residual curvature, and scipy trust-exact with 100 iterations and gtol=1e-11; its tight replay uses 1000 and 1e-13. The fixed chart does not cover subspaces orthogonal to the initial subspace.',
              'Newton and the smaller-side variable-projection arm transpose rectangular problems and permute the full metric when necessary. This changes the subspace parameterization without changing the physical objective. The other variable-projection arms always eliminate the right factor.',
              'Balanced variants refactor the resulting rank-D product as U sqrt(S) and sqrt(S) Vh before returning it to CBE. This changes factor conditioning while preserving the compressed matrix up to rounding. Raw versions are retained as diagnostic controls.',
              'Replay also includes 40 ALS iterations with dense SVD least-squares inner solves, to distinguish inner iterative-solve accuracy from alternating-optimization accuracy.',
              'All use the same physical-metric spectral cutoff 1e-10. Variable projection uses a relative 1e-12 inner pseudoinverse cutoff. Neither tolerance is a guarantee of global optimality.',
              'Nonlinear references materialize the active-site metric square root and Jacobians; they are intended for small local benchmarks. They preserve rank on the original factors. Production defaults were not changed.', '',
              '## Convergence diagnostics', '',
              '| Case | Method | General fits | Inner LSMR solves | LSMR limit reached | Nonlinear budget exits |',
              '|---|---|---:|---:|---:|---:|']
    for r in records:
        if r['method'] != 'one' and not r.get('failed'):
            lines.append(f"| {r['case']} | {LABELS[r['method']]} | {r['general_fits']} | {r['inner_solves']} | {r['inner_stops'].get(7,0)} | {r['nonlinear_budget_exits']} |")
    lines += ['', '## Numerical stability and energy after raw compression', '',
              '| Case | Method | Maximum sweep energy increase | Raw trims above old energy | Accepted incumbent refinements |',
              '|---|---|---:|---:|---:|']
    for r in records:
        if r['shape'] == '32' and not r.get('failed'):
            lines.append(f"| {r['case']} | {LABELS[r['method']]} | {r['max_sweep_energy_increase']:.6g} | {r['raw_trim_above_old']}/{r['raw_trims']} | {r['accepted_incumbent']} |")
    for failure in failures:
        lines += ['', f"Failed arm: {failure['case']}, {LABELS[failure['method']]}; last completed sweep {failure['last_completed_sweep']}, last reported energy {failure['last_reported_energy']}. {failure['error']}. The copied log contains the traceback. This is not a completed 100-sweep energy."]
    replay_path = source/'replay.json'
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        (output/'replay.json').write_text(json.dumps(replay, indent=2)+'\n')
        groups = defaultdict(dict)
        for r in replay:
            groups[r['snapshot']][r['method']] = r
        selected = [g for g in groups.values() if g['baseline']['relative_loss'] > 1e-12
                    and g['baseline']['loss'] > 1e-14]
        lines += ['', '## Identical-problem replay', '',
            f'{len(groups)} saved baseline general-metric problems, sampled with a deterministic reservoir (up to 12 per case). {len(selected)} have baseline relative loss above 1e-12 and absolute loss above 1e-14 and enter the ratios below; near-zero-loss cases remain in replay.json.',
            'All solvers start from the same ordinary SVD factorization on each problem. Ratios compare identical targets and physical metrics; end-to-end fit losses from different paths are not directly compared.', '',
            '| Method | Median loss / baseline | Median fit time (ms) | Median relative projected gradient |',
            '|---|---:|---:|---:|']
        for method in next(iter(groups.values())):
            values = [g[method] for g in selected]
            ratios = [g[method]['loss']/g['baseline']['loss'] for g in selected]
            lines.append(f"| {LABELS.get(method,method)} | {np.median(ratios):.6g} | {1000*np.median([r['seconds'] for r in values]):.3g} | {np.median([r['projected_gradient_relative'] for r in values]):.3g} |")
        hard = [g for name, g in groups.items() if name.startswith('bose32_')
                and g['baseline']['loss'] > 1e-14]
        lines += ['', '### 3×2 replay separately', '',
                  'Ratios need not improve monotonically across solver variants: changing an inner solve or parameterization changes the path through a nonconvex problem.', '',
                  '| Method | Median loss / baseline | Fits improved >10% | Fits worsened >10% |',
                  '|---|---:|---:|---:|']
        for method in next(iter(groups.values())):
            ratios = np.array([g[method]['loss']/g['baseline']['loss'] for g in hard])
            lines.append(f"| {LABELS.get(method,method)} | {np.median(ratios):.6g} | {sum(ratios<.9)}/{len(ratios)} | {sum(ratios>1.1)}/{len(ratios)} |")
    lines += ['', '## Reproduction', '',
        'Run from the repository root with PYTHONPATH=. and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=1:', '',
        '```bash',
        'python -m pyqed._letta_one_site_opt.benchmarks.run_compression_suite --output /private/tmp/letta-compression-strict-20260925 --sweeps 100',
        'python -m pyqed._letta_one_site_opt.benchmarks.compression_accuracy replay --source /private/tmp/letta-compression-strict-20260925/snapshots --output /private/tmp/letta-compression-strict-20260925',
        '```', '',
        'The suite skips already completed files; use a new directory for an independent repeat. Incomplete logs do not count as completed results.',
        'An earlier exploratory-acceptance pilot lives in /private/tmp/letta-compression-20260925 and is excluded from this report. Its interrupted run is not a result.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    palette = plt.get_cmap('tab20')
    colors = {method: palette(i) for i, method in enumerate(('one',)+METHODS)}
    plotted = ('one', 'baseline', 'als40_40', 'als4_400', 'als40_400',
               'joint_trf', 'grassmann_newton_balanced',
               'varpro_lm_balanced', 'varpro_trf_small_balanced')
    for col, (shape, bond) in enumerate([('22', 2), ('32', 3)]):
        data = [d for d in all_data if d['config']['shape'].replace(',','') == shape
                and d['config']['bond'] == bond]
        for d in data:
            r, method = d['record'], d['config']['method']
            if method not in plotted:
                continue
            energy = np.array(r['sweep_energies'])
            axes[0,col].plot(np.arange(1,len(energy)+1), energy, label=LABELS[method], lw=1.4, color=colors[method])
            axes[1,col].plot(r['sweep_elapsed_seconds'], energy, label=LABELS[method], lw=1.4, color=colors[method])
        axes[0,col].set_title(f'{shape[0]}×{shape[1]}, D={bond}' + (' · late-energy zoom' if col == 0 else ''))
        axes[0,col].set_xlabel('Directional sweep')
        axes[1,col].set_xlabel('Elapsed solver time (s)')
        for row in range(2):
            axes[row,col].set_ylabel('Energy')
            axes[row,col].ticklabel_format(axis='y', style='plain', useOffset=False)
            if col == 0:
                axes[row,col].set_ylim(-11.69712, -11.6955)
            else:
                axes[row,col].axhline(data[0]['exact_energy'], color='black', ls=':', lw=1,
                                     label='Exact diagonalization')
            axes[row,col].grid(alpha=.2)
    axes[0,1].legend(fontsize=6.5, loc='upper right', ncol=2)
    fig.savefig(output/'convergence.png', dpi=180)
    fig.savefig(output/'convergence.svg')
    print(output/'report.md')


if __name__ == '__main__':
    main()

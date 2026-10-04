"""Collect a frozen local comparison without treating rejected sweeps as accepted."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'one-site': '#0072B2', 'cbe': '#D55E00', 'two-site': '#009E73'}
NAMES = {'one-site': 'One-site', 'cbe': 'Strict CBE', 'two-site': 'Two-site'}


def collect(plan_path, output):
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    batch_path = plan_path.parent / 'batch.json'
    batch = json.loads(batch_path.read_text()) if batch_path.exists() else {}
    coordinator_alive = None
    if batch.get('pid'):
        try:
            os.kill(batch['pid'], 0)
            coordinator_alive = True
        except ProcessLookupError:
            coordinator_alive = False
        except PermissionError:
            pass
    rows, curves = [], {}
    for task in plan['tasks']:
        path = plan_path.parent / 'results' / task['case'] / f"{task['algorithm']}__{task['profile']}.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        history = result.get('history', [])
        # The observer runs before rollback. Rejected trial energies must not
        # become points on the accepted-state trajectory.
        energy = result.get('initial_energy')
        trajectory = [(0, 0., energy)] if energy is not None else []
        rejected = []
        for item in history:
            if item['fresh_energy_check_passed']:
                energy = item['energy']
            else:
                rejected.append(item['sweep'])
            trajectory.append((item['sweep'], item['solver_seconds'], energy))
        status = result.get('status', 'queued')
        if status in ('running', 'starting') and coordinator_alive is False:
            status = 'interrupted: coordinator exited'
        if status == 'completed':
            status = ('converged' if result['converged'] else
                      'sweep cap' if 'MAXIMUM SWEEPS' in result['message'] else 'stopped: ' + result['message'])
        if result:
            assert result['plan_sha256'] == plan_hash, f'Wrong plan: {path}'
            if 'initial_tensor_hash' in result:
                assert result['initial_tensor_hash'] == task['initial_state']['tensor_hash'], f'Wrong start: {path}'
        endpoint = result.get('energy', energy)
        at20 = next((p[2] for p in trajectory if p[0] == 20), None)
        if at20 is None and status == 'converged':
            at20 = endpoint  # Constant continuation of an already converged run.
        row = dict(case=task['case'], model=task['model'], shape=task['shape'], bond_dim=task['bond_dim'],
                   algorithm=task['algorithm'], status=status, max_sweeps=task['max_sweeps'],
                   sweeps=len(history), energy_at_20=at20, endpoint_energy=endpoint,
                   solver_seconds=result.get('solver_seconds', history[-1]['solver_seconds'] if history else None),
                   rejected_sweeps=rejected, initial_tensor_hash=result.get('initial_tensor_hash'),
                   result=str(path))
        rows.append(row)
        curves[task['index']] = trajectory
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for model in dict.fromkeys(t['model'] for t in plan['tasks']):
        cases = list(dict.fromkeys(t['case'] for t in plan['tasks'] if t['model'] == model))
        fig, axes = plt.subplots(3, len(cases), figsize=(17, 10), squeeze=False, layout='constrained')
        for col, case in enumerate(cases):
            group = [t for t in plan['tasks'] if t['case'] == case]
            for task in group:
                curve = curves[task['index']]
                if not curve:
                    continue
                sweep, seconds, energies = zip(*curve)
                row = next(r for r in rows if r['case'] == case and r['algorithm'] == task['algorithm'])
                display_status = row['status']
                if 'FRESH ENERGY CHECK' in display_status:
                    display_status = 'energy check failed'
                elif display_status.startswith('interrupted'):
                    display_status = 'interrupted'
                label = f"{NAMES[task['algorithm']]} ({display_status})"
                opts = dict(color=COLORS[task['algorithm']], label=label, linewidth=1.4)
                axes[0, col].plot(sweep, energies, **opts)
                late = [p for p in curve if p[0] >= 20]
                if late:
                    late_opts = dict(opts, marker='o', markersize=4) if len(late) == 1 else opts
                    axes[1, col].plot([p[0] for p in late], [p[2] for p in late], **late_opts)
                    axes[2, col].plot([p[1]/60 for p in late], [p[2] for p in late], **late_opts)
                for n in row['rejected_sweeps']:
                    point = next(p for p in curve if p[0] == n)
                    axes[1, col].plot(n, point[2], 'x', color=opts['color'], markersize=7)
            task = group[0]
            shape = sorted(task['shape'])
            axes[0, col].set_title(f"{shape[0]}×{shape[1]}, D={task['bond_dim']}")
            axes[0, col].set_xlim(0, 20)
            axes[0, col].set_xlabel('Sweep (first 20)')
            # Keep the first-20 panel useful even when later energies improve.
            early = [p[2] for t in group for p in curves[t['index']] if 1 <= p[0] <= 20]
            if early:
                span = max(max(early)-min(early), 1e-5)
                axes[0, col].set_ylim(min(early)-.06*span, max(early)+.06*span)
            axes[1, col].set_xlabel('Sweep (20 onward)')
            axes[2, col].set_xlabel('Solver minutes (sweep 20 onward)')
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis='y', useOffset=False)
                ax.set_ylabel('Total energy')
            if axes[1, col].lines:
                axes[1, col].legend(fontsize=7, loc='best')
        fig.suptitle(f"{model.replace('_', ' ').title()}: identical starts; seed 731\n"
                     'One-site / CBE: 500-sweep cap; two-site: 20-sweep cap. × marks a rolled-back sweep.')
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{model}.{ext}', dpi=160)
        plt.close(fig)
    lines = ['# Local LETTA comparison', '', f"Updated: {datetime.now(timezone.utc).isoformat()}", '',
             f"Coordinator: {batch.get('status', 'unknown')}; alive: {coordinator_alive}; heartbeat (UTC): {batch.get('heartbeat', 'unavailable')}.", '',
             'Bose–Hubbard and Heisenberg; 3×3 and 3×4; D=2,3; seed 731. '
             'The 3×4 lattice is traversed in the equivalent 4×3 orientation. '
             'Every method uses the same saved initial tensors within each case.', '',
             'One-site and strict CBE have a 500-sweep cap; two-site has a 20-sweep cap. '
             'Stopping tolerance is 1e-12 in energy change per site per directional sweep. '
             'This stopping criterion does not certify a global minimum. '
             'CBE/two-site use ALS4/LSMR400 compression and at most 32 inner energy-refinement rounds. '
             'A sweep is one directional pass, not a left-right pair.', '',
             'E(20) is shown only after reaching 20 sweeps, or after earlier reported convergence. '
             'Running endpoints are provisional. Rejected trial energies are excluded from accepted-state curves. '
             'Solver time excludes the benchmark observer; timings include competition between up to two single-thread workers.', '',
             '| Case | Method | Status | Sweeps | E(20) | Latest accepted E | Solver min |',
             '|---|---|---|---:|---:|---:|---:|']
    def fmt(x, digits=10):
        return '—' if x is None else f'{x:.{digits}f}'
    for row in rows:
        lines.append(f"| {row['case']} | {NAMES[row['algorithm']]} | {row['status']} | {row['sweeps']} | "
                     f"{fmt(row['energy_at_20'])} | {fmt(row['endpoint_energy'])} | "
                     f"{fmt(None if row['solver_seconds'] is None else row['solver_seconds']/60, 2)} |")
    for model in dict.fromkeys(t['model'] for t in plan['tasks']):
        lines.extend(['', f'![{model}]({model}.png)'])
    lines.extend(['', f'Frozen plan: `{plan_path}`', '',
                  'The eight earlier 100-sweep one-site runs and four two-sweep pilots are separate preliminary runs; '
                  'they are not mixed into these curves.'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(Counter(r['status'] for r in rows)), indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.plan, args.output)

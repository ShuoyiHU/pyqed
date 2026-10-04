"""Snapshot 3x12 cluster histories and plot completed accepted sweeps only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

MODELS = ['ising', 'heisenberg', 'bose_hubbard', 'fermi_hubbard']
NAMES = dict(ising='Ising', heisenberg='Heisenberg', bose_hubbard='Bose–Hubbard', fermi_hubbard='Fermi–Hubbard')
COLORS = {'one-site': '#333333', 'cbe': '#0072B2', 'two-site': '#D55E00'}
METHODS = {'one-site': 'One-site', 'cbe': 'CBE', 'two-site': 'Two-site'}


def snapshot(root, output):
    work = []
    # Reading plans gives exact result paths without depending on mount glob caches.
    for name in ['20261002-141429-17022', '20261002-141432-22226']:
        run = root / 'runs' / name
        raw = (run / 'plan.json').read_bytes()
        job = (run / 'job_id.txt').read_text().strip()
        plan = json.loads(raw)
        save = output / 'raw' / name
        save.mkdir(parents=True, exist_ok=True)
        (save / 'plan.json').write_bytes(raw)
        for task in plan['tasks']:
            if sorted(task['shape']) == [3, 12]:
                work.append((run, job, task, hashlib.sha256(raw).hexdigest(), save))

    def read(item):
        run, job, task, plan_hash, save = item
        label = task['algorithm'] + '__' + task['profile']
        path = run / 'results' / task['case'] / (label + '.json')
        result_bytes = path.read_bytes()
        record = json.loads(result_bytes)
        assert record['plan_sha256'] == plan_hash
        assert record['initial_tensor_hash'] == task['initial_state']['tensor_hash']
        dest = save / task['case']
        dest.mkdir(exist_ok=True)
        (dest / (label + '.json')).write_text(json.dumps(record, indent=2))
        error = (run / 'logs' / f'{job}_{task["index"]}.err').read_text()
        (dest / (label + '.err')).write_text(error)
        message = record.get('message') or ''
        if 'oom_kill' in error:
            status = 'OOM'
        elif 'CANCELLED' in error:
            status = 'cancelled'
        elif record['status'] == 'failed' or 'Traceback' in error:
            status = 'exception'
        elif 'ENERGY CHECK' in message:
            status = 'energy check'
        elif record['status'] == 'completed':
            status = 'converged' if record.get('converged') else 'cap'
        else:
            status = 'unfinished'
        history = record.get('history', [])
        accepted, rejected = [], []
        for n, row in enumerate(history):
            # The benchmark observer runs before the solver's whole-sweep rollback.
            reject = (not row.get('fresh_energy_check_passed', True)
                      or (n == len(history)-1 and status == 'energy check'))
            (rejected if reject else accepted).append(row)
        if record['status'] == 'completed' and accepted:
            assert abs(accepted[-1]['energy'] - record['energy']) < 1e-7, path
        return dict(task=task, job=job, status=status, accepted=accepted, rejected=rejected,
                    final_energy=record.get('energy'), updated=record.get('updated'),
                    source_sha256=hashlib.sha256(result_bytes).hexdigest())

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read, work))


def style(task):
    return dict(color=COLORS[task['algorithm']],
                linestyle='--' if task['profile'].startswith('als40-') else '-', linewidth=1.7)


def label(run):
    task = run['task']
    profile = '' if task['algorithm'] == 'one-site' else (' / ALS40' if task['profile'].startswith('als40-') else ' / ALS4')
    count = run['accepted'][-1]['sweep'] if run['accepted'] else 0
    return f"{METHODS[task['algorithm']]}{profile}: {count} ({run['status']})"


def panel(ax, group, *, zoom=False):
    for run in group:
        rows = run['accepted']
        opts = style(run['task'])
        if not rows:
            ax.plot([], [], label=label(run) + ' — no sweep', **opts)
            continue
        x, y = [r['sweep'] for r in rows], [r['energy'] for r in rows]
        ax.plot(x, y, label=label(run), **opts)
        marker = ('x' if run['status'] in ('energy check', 'exception', 'OOM', 'cancelled') else
                  'o' if run['status'] == 'unfinished' else 's')
        ax.plot(x[-1], y[-1], marker=marker, markersize=6, markerfacecolor='white',
                color=opts['color'], markeredgewidth=1.2, linestyle='none')
    model = group[0]['task']['model']
    all_rows = [r for run in group for r in run['accepted']]
    last = max((r['sweep'] for r in all_rows), default=1)
    if zoom:
        lo, hi = (1, min(last, 15)) if model == 'fermi_hubbard' else (10, last)
        visible = [r['energy'] for r in all_rows if lo <= r['sweep'] <= hi]
        ax.set_xlim(lo, max(lo+1, hi) + max((hi-lo)*.015, .2))
        if visible:
            span = max(max(visible)-min(visible), 1e-8)
            ax.set_ylim(min(visible)-.1*span, max(visible)+.1*span)
        ax.set_title(f"Seed {group[0]['task']['seed']} · {'early sweeps' if model == 'fermi_hubbard' else 'energy zoom, sweep ≥ 10'}", fontsize=11)
    else:
        ax.set_xlim(1, last+max(last*.015, 1))
        ax.set_title(f"Seed {group[0]['task']['seed']} · all completed sweeps", fontsize=11)
    ax.set_xlabel('Directional sweep number')
    ax.set_ylabel('Total energy')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = snapshot(args.root, args.output)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M CST (UTC+8)')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    with (args.output / 'source_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','D','seed','method','profile','status','sweep','energy','accepted','solver_seconds'])
        writer.writeheader()
        for run in runs:
            t = run['task']
            for accepted, rows in [(True, run['accepted']), (False, run['rejected'])]:
                for r in rows:
                    writer.writerow(dict(model=t['model'],D=t['bond_dim'],seed=t['seed'],method=t['algorithm'],
                        profile=t['profile'],status=run['status'],sweep=r['sweep'],energy=r['energy'],
                        accepted=accepted,solver_seconds=r['solver_seconds']))
    summary = [dict(task=r['task'],status=r['status'],accepted_sweeps=len(r['accepted']),
                    last_energy=r['accepted'][-1]['energy'] if r['accepted'] else None,
                    rejected_sweeps=[h['sweep'] for h in r['rejected']],updated=r['updated']) for r in runs]
    (args.output/'summary.json').write_text(json.dumps(dict(snapshot=stamp,runs=summary),indent=2))
    footer = ('Lines show accepted states only; × stopped/failed, ○ unfinished, □ sweep cap. '
              'Unfinished is file status, not a live Slurm query.\n'
              'ALS4/ALS40: compression rounds; both use LSMR400. Each sweep is one directional pass. No smoothing or extrapolation.')
    with PdfPages(args.output/'energy_vs_sweep_3x12.pdf') as pdf:
        for model in MODELS:
            fig, axes = plt.subplots(2,2,figsize=(14,9.8))
            for row, seed in enumerate([731,1735]):
                group = [r for r in runs if r['task']['model']==model and r['task']['seed']==seed]
                panel(axes[row,0],group)
                panel(axes[row,1],group,zoom=True)
                axes[row,0].legend(fontsize=8,loc='best',framealpha=.85)
            bond = 3 if model=='fermi_hubbard' else 4
            fig.suptitle(f'{NAMES[model]} · 3×12 · D={bond}\nEnergy versus completed sweep · {stamp}',fontsize=15,y=.985)
            fig.text(.5,.018,footer,ha='center',va='bottom',fontsize=8)
            fig.subplots_adjust(left=.095,right=.98,bottom=.12,top=.88,wspace=.29,hspace=.36)
            for ext in ['png','pdf','svg']:
                fig.savefig(args.output/f'{model}_3x12.{ext}',dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes=plt.subplots(4,2,figsize=(13,16))
    for row, model in enumerate(MODELS):
        for col, seed in enumerate([731,1735]):
            group=[r for r in runs if r['task']['model']==model and r['task']['seed']==seed]
            panel(axes[row,col],group)
            axes[row,col].set_title(f'{NAMES[model]} · D={group[0]["task"]["bond_dim"]} · seed {seed}')
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=METHODS[m]) for m in COLORS]
    handles += [Line2D([0],[0],color='#777',ls='-',label='ALS4'),Line2D([0],[0],color='#777',ls='--',label='ALS40')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=5,frameon=False)
    fig.suptitle(f'3×12 cluster calculations · all completed accepted sweeps\n{stamp}',fontsize=16,y=.992)
    fig.text(.5,.018,footer,ha='center',fontsize=8)
    fig.subplots_adjust(left=.1,right=.98,bottom=.08,top=.91,wspace=.28,hspace=.42)
    fig.savefig(args.output/'overview_3x12.png',dpi=150)
    plt.close(fig)
    (args.output/'caption.md').write_text('Energy versus completed directional sweep for the 3×12 cluster tests. '+stamp+'.\n\n'+footer+'\n\n'
        'Seeds731 and1735 are shown separately, without averaging. The figures include all saved completed sweeps from unfinished jobs. '
        'Rejected trials are recorded in source_data.csv but excluded from plotted trajectories. The × marker is at the last accepted state, '
        'not at the rejected candidate. Jobs with no completed sweep appear only in the legend. '
        'Right-hand panels zoom the energy range from sweep10 onward; Fermi–Hubbard instead shows the first15 sweeps because CBE has completed few sweeps. '
        'All panels use linear axes; no energy offsets are subtracted. Different methods have different costs per sweep.\n')
    print('Snapshot',stamp,'records',len(runs),'statuses',dict(Counter(r['status'] for r in runs)))
    print('Accepted points',sum(len(r['accepted']) for r in runs),'rejected points',sum(len(r['rejected']) for r in runs))


if __name__=='__main__':
    main()

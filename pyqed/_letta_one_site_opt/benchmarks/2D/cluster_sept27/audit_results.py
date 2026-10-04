"""Retain cluster reports and compare completed, matched-case solver results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = (source/'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    (output/'plan.json').write_bytes(plan_bytes)
    reports, files, rows = [], [], []
    for task in plan['tasks']:
        relative = Path('results')/task['case']/(task['algorithm']+'__'+task['profile']+'.json')
        file = source/relative
        data = file.read_bytes() if file.exists() else b'{}'
        result = json.loads(data)
        if result and result['task'] != task:
            raise ValueError(f'Task mismatch: {relative}')
        reports.append(result)
        files.append(dict(path=str(relative), sha256=sha(data), bytes=len(data)))
        history = result.get('history', [])
        used, reasons = Counter(), Counter()
        for sweep in history:
            used.update(sweep['compression']['used_solvers'])
            reasons.update(sweep['compression']['fallback_reasons'])
        energy = result.get('energy')
        scale = max(1., abs(energy or 0.))
        jumps = [h['energy_change'] for h in history if h['energy_change'] > 1e-7*scale]
        rows.append(dict(index=task['index'], case=task['case'], model=task['model'],
            algorithm=task['algorithm'], profile=task['profile'], status=result.get('status','missing'),
            energy=energy, solver_seconds=result.get('solver_seconds'), sweeps=len(history),
            converged=result.get('converged'), max_rss_gib=result.get('max_rss_gib'),
            physical_energy_error=abs(energy-result['physical_energy']) if energy is not None else None,
            plan_matches=result.get('plan_sha256')==sha(plan_bytes),
            bundle_matches=result.get('runtime',{}).get('bundle_sha256')==plan['bundle_sha256'],
            initial_tensor_hash=result.get('initial_tensor_hash'),
            final_tensor_file_exists=file.with_name(file.stem+'__final.npz').is_file(),
            material_upward_sweeps=len(jumps), max_upward_sweep=max(jumps,default=0.),
            used_solvers=json.dumps(dict(used),sort_keys=True),
            fallback_reasons=json.dumps(dict(reasons),sort_keys=True),
            fits=sum(used.values()), nonlinear_fits=sum(v for k,v in used.items()
                if k in ('variable-projection','joint-ls','grassmann-newton')),
            fallbacks=sum(reasons.values()), error=result.get('error')))
    with gzip.open(output/'all_reports.json.gz','wt') as stream:
        json.dump(reports,stream)
    baseline = {r['case']:r for r in rows if r['algorithm']=='one-site' and r['status']=='completed'}
    grouped = defaultdict(list)
    for r in rows:
        base = baseline.get(r['case'])
        done = r['status']=='completed' and base is not None
        r['same_initial_hash_as_one_site'] = r['initial_tensor_hash']==base['initial_tensor_hash'] if base else None
        r['energy_minus_one_site'] = r['energy']-base['energy'] if done else None
        r['runtime_over_one_site'] = r['solver_seconds']/base['solver_seconds'] if done else None
        grouped[r['algorithm'],r['profile']].append(r)
    table(output/'jobs.csv',rows)
    profiles=[]
    for (algorithm,profile),group in grouped.items():
        done=[r for r in group if r['status']=='completed']
        profiles.append(dict(algorithm=algorithm,profile=profile,completed=len(done),
            failed=sum(r['status']=='failed' for r in group),
            interrupted=sum(r['status']=='interrupted' for r in group),
            better_than_one_site=sum(r['energy_minus_one_site'] < -1e-7 for r in done),
            worse_than_one_site=sum(r['energy_minus_one_site'] > 1e-7 for r in done),
            median_runtime_ratio=median(r['runtime_over_one_site'] for r in done) if done else None,
            runs_with_upward_sweeps=sum(r['material_upward_sweeps']>0 for r in done),
            nonlinear_fits=sum(r['nonlinear_fits'] for r in done),
            all_fits=sum(r['fits'] for r in done),fallbacks=sum(r['fallbacks'] for r in done)))
    table(output/'profiles.csv',profiles)
    # Keep a per-model profile table, so a win on one model cannot hide a loss on another.
    model_profiles=[]
    for model in sorted({r['model'] for r in rows}):
        for (algorithm,profile),group in grouped.items():
            done=[r for r in group if r['model']==model and r['status']=='completed']
            if done:
                model_profiles.append(dict(model=model,algorithm=algorithm,profile=profile,
                    completed=len(done),better=sum(r['energy_minus_one_site'] < -1e-7 for r in done),
                    worse=sum(r['energy_minus_one_site'] > 1e-7 for r in done),
                    median_runtime_ratio=median(r['runtime_over_one_site'] for r in done)))
    table(output/'model_profiles.csv',model_profiles)
    snapshot=dict(collected_utc=datetime.now(timezone.utc).isoformat(),source=str(source),
        counts=dict(Counter(r['status'] for r in rows)),plan_sha256=sha(plan_bytes),
        files=files,status_source='retained result files, not live Slurm',
        upward_sweep_threshold='1e-7 * max(1, abs(final_energy))')
    (output/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    plot(reports,output)
    print(json.dumps({k:v for k,v in snapshot.items() if k!='files'},indent=2))
    print(json.dumps(profiles,indent=2))


def plot(reports,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    completed=[r for r in reports if r.get('status')=='completed']
    profiles=sorted({r['task']['profile'] for r in completed if r['task']['algorithm']!='one-site'})
    colors=dict(zip(profiles,plt.get_cmap('tab10').colors))
    for model in sorted({r['task']['model'] for r in completed}):
        cases=sorted({r['task']['case'] for r in completed if r['task']['model']==model})
        with PdfPages(output/f'{model}.pdf') as pdf:
            for case in cases:
                fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
                for row,algorithm in enumerate(('cbe','two-site')):
                    selected=[r for r in completed if r['task']['case']==case and r['task']['algorithm'] in ('one-site',algorithm)]
                    for r in selected:
                        h=r['history']; base=r['task']['algorithm']=='one-site'; p=r['task']['profile']
                        label='one-site' if base else p
                        kw=dict(color='black' if base else colors[p],ls='--' if base else '-',lw=1.3,label=label)
                        axes[row,0].plot([x['sweep'] for x in h],[x['energy'] for x in h],**kw)
                        axes[row,1].plot([x['solver_seconds'] for x in h],[x['energy'] for x in h],**kw)
                    for col in range(2):
                        ax=axes[row,col]; ax.set_ylabel(f'{algorithm}: total energy'); ax.grid(alpha=.2)
                        ax.set_xlabel('Directional sweep' if col==0 else 'Solver seconds (log scale)')
                        if col: ax.set_xscale('log')
                    axes[row,0].legend(fontsize=7,ncol=2)
                fig.suptitle(case+'\nCompleted runs only; requested profiles may contain ALS fallbacks')
                pdf.savefig(fig)
                plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    audit(a.source,a.output)

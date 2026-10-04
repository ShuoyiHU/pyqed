"""Open virtual boundaries with the unchanged periodic Hubbard Hamiltonian."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..open_boundary import OpenState, OpenContractions, OpenOneSiteOptions, open_one_site
from ..periodic import hubbard_ring_terms, bose_hubbard_ring_terms
from .periodic_hubbard import save


def run_case(output, model, length, dimension, variant, sweeps=500, seed=731, warm_start=None):
    output.mkdir(parents=True,exist_ok=True)
    path = output/f'L{length}_D{dimension}_{variant}_seed{seed}.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report['status']=='finished':
            return report
        raise FileExistsError(f'use a new output directory for incomplete run {path}')
    numbers = (0,1,2) if model=='bose' else (0,1,1,2)
    initial = OpenState.random(length,dimension,seed=seed,particle_numbers=numbers)
    state = initial if variant=='mps' else initial.with_nn_ties(wrap=variant=='letta_wrap')
    if warm_start is not None:
        parent = json.loads(warm_start.read_text())
        if (variant != 'letta_wrap' or parent['method'] != 'letta'
                or parent['length'] != length or parent['bond_dimension'] != dimension
                or parent['model']['name'] != model or parent['status'] != 'finished'):
            raise ValueError('warm start must be matching finished no-wrap LETTA')
        with np.load(warm_start.with_suffix('.npz')) as checkpoint:
            tensors = [checkpoint[f'tensor_{i}'].copy() for i in range(length)]
        tensors[-1] = np.repeat(tensors[-1][:,None],len(numbers),axis=1)
        state = OpenState(tensors,'letta',parent['charges'],numbers,wrap_tie=True)
    terms = bose_hubbard_ring_terms(length) if model=='bose' else hubbard_ring_terms(length)
    options = OpenOneSiteOptions(max_sweeps=sweeps)
    source = Path(__file__).resolve().parents[1]
    report = dict(status='running',length=length,bond_dimension=dimension,method=variant,seed=seed,
                  model=dict(name=model,t=1.,U=4.,chemical_potential=0.,particles=length,
                             max_occupancy=2 if model=='bose' else None,
                             physical_boundary='periodic',virtual_boundary='open',wrap_tie=state.wrap_tie),
                  charges=[q.tolist() for q in state.charges],particle_numbers=list(numbers),
                  actual_bond_dimensions=[len(q) for q in state.charges],
                  parameters_per_site=[int(state.mask(i).sum()) for i in range(length)],
                  initialization='same random open MPS for all three variants' if warm_start is None else 'optimized no-wrap LETTA embedded exactly',
                  parent_checkpoint=str(warm_start) if warm_start is not None else None,
                  initial_mps_sha256=hashlib.sha256(b''.join(a.tobytes() for a in initial.tensors)).hexdigest() if warm_start is None else None,
                  options=asdict(options),
                  source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ('open_boundary.py','periodic.py','solver.py')},
                  history=[dict(sweep=0,energy=OpenContractions(state,terms).energy(),elapsed_seconds=0.)])
    if warm_start is not None and abs(report['history'][0]['energy']-parent['final_energy']) > 1e-10:
        raise AssertionError('warm-start embedding changed the energy')
    save(path,report)
    def observer(row,current):
        report['history'].append(row)
        save(path,report)
        if row['sweep']%25==0 or row['converged']:
            print(f'{model} L{length} D{dimension} {variant}: {row["sweep"]} E={row["energy"]:.12f} '
                  f't={row["elapsed_seconds"]:.1f}s cond={row["max_equilibrated_metric_condition"]:.3g}',flush=True)
    try:
        final,history,converged = open_one_site(state,terms,options,observer)
        np.savez_compressed(path.with_suffix('.npz'),**{f'tensor_{i}':a for i,a in enumerate(final.tensors)})
        if model=='fermi':
            from .periodic_hubbard_validation import physical_check
            validation = physical_check(final)
        else:
            from .periodic_bose_validation import sector_hamiltonian
            configs,h = sector_hamiltonian(length)
            v = final.amplitudes(configs)
            norm = np.vdot(v,v).real
            hv = h@v
            e = float((np.vdot(v,hv)/norm).real)
            validation = dict(physical_energy=e,physical_norm=float(norm),
                              energy_variance=float(np.vdot(hv-e*v,hv-e*v).real/norm))
        validation['contraction_error'] = abs(validation['physical_energy']-history[-1]['energy'])
        if validation['contraction_error']>1e-8:
            raise AssertionError(f'independent energy check failed: {validation}')
        report.update(status='finished',history=history,converged=converged,sweeps=history[-1]['sweep'],
                      final_energy=history[-1]['energy'],elapsed_seconds=history[-1]['elapsed_seconds'],
                      stop_reason='energy plateau' if converged else 'sweep cap',validation=validation)
        save(path,report)
        print('FINISHED',path.name,report['final_energy'],flush=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save(path,report)
        raise
    return report


def collect_plot(output, closed):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = [json.loads(p.read_text()) for p in list(output.glob('*/*seed*.json'))
               + list(output.glob('warm/*/*seed*.json'))]
    rows = []
    for r in reports:
        olddir = closed/r['model']['name']
        kind = 'mps' if r['method']=='mps' else 'letta'
        oldpath = olddir/f'L{r["length"]}_D{r["bond_dimension"]}_{kind}_seed{r["seed"]}.json'
        old = json.loads(oldpath.read_text()) if oldpath.exists() else None
        row = dict(model=r['model']['name'],length=r['length'],D=r['bond_dimension'],method=r['method'],
                   status=r['status'],initialization=r['initialization'],energy=r.get('final_energy'),sweeps=r.get('sweeps'),
                   stop_reason=r.get('stop_reason'),seconds=r.get('elapsed_seconds'),
                   closed_energy=old.get('final_energy') if old else None,
                   energy_minus_closed=r['final_energy']-old['final_energy'] if old and r.get('final_energy') else None,
                   physical_check_error=r.get('validation',{}).get('contraction_error'))
        rows.append(row)
    save(output/'summary.json',rows)
    if rows:
        with (output/'summary.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    styles = {'mps':('#2474a6','-','Open MPS'),'letta':('#db8b23','-','Open LETTA, no wrap tie'),
              'letta_wrap':('#228452','-','Open LETTA, wrap tie')}
    for model in sorted({r['model']['name'] for r in reports}):
        for length in sorted({r['length'] for r in reports}):
            selected = [r for r in reports if r['model']['name']==model and r['length']==length]
            if not selected:continue
            bonds = sorted({r['bond_dimension'] for r in selected})
            fig,axes = plt.subplots(3,len(bonds),figsize=(4*len(bonds),10),squeeze=False,constrained_layout=True)
            for col,d in enumerate(bonds):
                series = [(r,*(('#222222','-.','Open LETTA wrap, no-wrap start') if r.get('parent_checkpoint')
                               else styles[r['method']])) for r in sorted(selected,key=lambda r:list(styles).index(r['method']))
                          if r['bond_dimension']==d]
                for kind,color in [('mps','#2474a6'),('letta','#884f99')]:
                    p = closed/model/f'L{length}_D{d}_{kind}_seed731.json'
                    if p.exists():
                        series.append((json.loads(p.read_text()),color,'--',f'Closed {kind.upper()} (paper gauge)'))
                for r,color,style,label in series:
                    e = np.array([h['energy'] for h in r['history']])
                    x = np.array([h['sweep'] for h in r['history']])
                    if not r.get('converged'):label += ' [cap]' if r['status']=='finished' else ' [running]'
                    axes[0,col].plot(x,e,style,color=color,label=label,lw=1.4)
                    delta = e-e[-1]
                    valid = delta>0
                    axes[1,col].plot(x,e,style,color=color,lw=1.4)
                    axes[2,col].plot(x[valid],delta[valid],style,color=color,lw=1.4)
                reference_path = closed.parent/('2026-10-03-periodic-bose-hubbard' if model=='bose' else '2026-10-03-periodic-hubbard')/'references.json'
                if reference_path.exists():
                    for ref in json.loads(reference_path.read_text()):
                        if ref['length']==length:
                            axes[0,col].axhline(ref['energy'],ls=':',color='0.4',lw=1,label='Exact periodic fixed-N')
                support_path = output/'support_references.json'
                if support_path.exists():
                    for ref in json.loads(support_path.read_text()):
                        if ref['model']==model and ref['length']==length and d in ref['bond_dimensions']:
                            axes[0,col].axhline(ref['support_energy_lower_bound'],ls='-.',color='#b94444',
                                               lw=1,label='Open charge-support bound')
                axes[0,col].set_title(f'D = {d}')
                axes[0,col].legend(fontsize=7)
                finals = [r['history'][-1]['energy'] for r,_,_,_ in series]
                low,high = min(finals),max(finals)
                margin = max(.15*(high-low),1e-4)
                axes[1,col].set_ylim(low-margin,high+margin)
                axes[1,col].set_title('Final-energy detail',fontsize=10)
                axes[2,col].set_yscale('log')
                for ax in axes[:,col]:
                    ax.set_xlabel('Sweep (one pass)');ax.grid(alpha=.2)
                axes[0,col].set_ylabel('Energy / t')
                axes[1,col].set_ylabel('Energy / t')
                axes[2,col].set_ylabel(r'$(E_k-E_{\mathrm{last}})/t$')
            fig.suptitle(f'{"Bose–Hubbard (n_max=2)" if model=="bose" else "Fermionic Hubbard"}: L=N={length}, t=1, U=4, μ=0\n'
                         'Physical Hamiltonian periodic in every run; open vs closed are different variational families')
            for ext in ('png','pdf'):
                fig.savefig(output/f'{model}_L{length}_boundaries.{ext}',dpi=160)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','plot'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['fermi','bose'],default='fermi')
    p.add_argument('--lengths',type=int,nargs='+',default=[5,10])
    p.add_argument('--bonds',type=int,nargs='+',default=[2,3,4,6])
    p.add_argument('--methods',nargs='+',choices=['mps','letta','letta_wrap'],default=['mps','letta','letta_wrap'])
    p.add_argument('--sweeps',type=int,default=500)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--closed',type=Path,default=Path('docs/benchmarks/2026-10-03-periodic-paper-gauge'))
    args=p.parse_args()
    if args.mode=='plot':
        collect_plot(args.output,args.closed)
    else:
        for length in args.lengths:
            for d in args.bonds:
                for variant in args.methods:
                    run_case(args.output/args.model,args.model,length,d,variant,args.sweeps,warm_start=args.warm_start)


if __name__=='__main__':main()

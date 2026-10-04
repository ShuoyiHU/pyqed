"""Same-start periodic Hubbard one-site comparison; exact sector ED is reference only."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from pyqed._letta_one_site_opt.periodic import (
    PeriodicState, PeriodicOneSiteOptions, RingContractions,
    hubbard_ring_terms, bose_hubbard_ring_terms, periodic_one_site,
)


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sector_reference(length, hopping=1., interaction=4.):
    """Sparse occupation-basis ED in minimal |Sz|, never used by the optimizer.

    SU(2) symmetry puts a member of each spin multiplet in this sector, so it
    contains the lowest energy at the specified total electron count.
    """
    up, down = (length+1)//2, length//2
    bits = []
    for us in itertools.combinations(range(length), up):
        ub = sum(1 << (2*i) for i in us)
        for ds in itertools.combinations(range(length), down):
            bits.append(ub + sum(1 << (2*i+1) for i in ds))
    lookup = {b: i for i, b in enumerate(bits)}
    rows, columns, values = [], [], []
    for column, occupations in enumerate(bits):
        rows.append(column); columns.append(column)
        values.append(interaction * sum((occupations >> (2*i)) & 3 == 3 for i in range(length)))
        for i in range(length):
            j = (i+1) % length
            for spin in (0, 1):
                for target, source in ((2*i+spin, 2*j+spin), (2*j+spin, 2*i+spin)):
                    if not occupations & (1 << source) or occupations & (1 << target):
                        continue
                    intermediate = occupations ^ (1 << source)
                    parity = ((occupations & ((1 << source)-1)).bit_count()
                              + (intermediate & ((1 << target)-1)).bit_count())
                    rows.append(lookup[intermediate | (1 << target)])
                    columns.append(column)
                    values.append(-hopping * (-1)**parity)
    h = sparse.coo_matrix((values, (rows, columns)), shape=(len(bits), len(bits))).tocsr()
    rng = np.random.default_rng(8675309)
    e, v = eigsh(h, k=1, which='SA', tol=1e-12, v0=rng.normal(size=len(bits)))
    residual = float(np.linalg.norm(h@v[:, 0]-e[0]*v[:, 0]))
    return dict(length=length, electrons=length, nup=up, ndown=down,
                energy=float(e[0]), residual=residual, sector_dimension=len(bits))


def run_case(output, length, dimension, kind, seed=731, sweeps=500, warm_start=None,
             model='fermi', max_occupancy=2, gauge_floor=1e-6, gauge_method='marginal'):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'L{length}_D{dimension}_{kind}_seed{seed}.json'
    if path.exists():
        report = json.loads(path.read_text())
        if report.get('status') == 'finished':
            print('Already finished:', path.name, flush=True)
            return report
        raise FileExistsError(f'Incomplete existing run: {path}; use a new directory')
    numbers = tuple(range(max_occupancy+1)) if model == 'bose' else (0, 1, 1, 2)
    initial = PeriodicState.random(length, dimension, seed, particle_numbers=numbers)
    if warm_start is not None:
        parent = json.loads(warm_start.read_text())
        if parent['method'] != 'mps' or parent['length'] != length:
            raise ValueError('paired warm starts require an MPS of the same length')
        with np.load(warm_start.with_suffix('.npz')) as data:
            initial = PeriodicState([data[f'tensor_{i}'] for i in range(length)], 'mps',
                                    np.array(parent['charges']), numbers).padded_start(dimension, seed=seed)
    state = initial if kind == 'mps' else initial.with_nn_ties()
    terms = (bose_hubbard_ring_terms(length, max_occupancy=max_occupancy)
             if model == 'bose' else hubbard_ring_terms(length))
    options = PeriodicOneSiteOptions(max_sweeps=sweeps, gauge_floor=gauge_floor,
                                    gauge_method=gauge_method)
    source = Path(__file__).resolve().parents[1] / 'periodic.py'
    report = dict(status='running', length=length, bond_dimension=dimension, method=kind,
                  seed=seed, model=dict(name=model, t=1., U=4., particles=length,
                                      max_occupancy=max_occupancy if model=='bose' else None,
                                      chemical_potential=0.,
                                      physical_boundary='periodic', virtual_boundary='periodic'),
                  charges=state.charges.tolist(), particle_numbers=list(numbers), options=asdict(options),
                  parameters_per_site=int(state.mask.sum()),
                  initialization='random MPS' if warm_start is None else 'D4 MPS padded with noise=1e-3',
                  parent_checkpoint=None if warm_start is None else str(warm_start),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_hashes={name:hashlib.sha256((source.parent/name).read_bytes()).hexdigest()
                                 for name in ('periodic.py', 'solver.py', 'contractions.py')},
                  environment=dict(python=platform.python_version(), numpy=np.__version__),
                  initial_mps_sha256=hashlib.sha256(b''.join(a.tobytes() for a in initial.tensors)).hexdigest(),
                  history=[dict(sweep=0, energy=RingContractions(state, terms).energy(), elapsed_seconds=0.)])
    save(path, report)
    def observer(row, current):
        report['history'].append(row)
        save(path, report)
        if row['sweep'] % 10 == 0 or row['converged']:
            print(f'L={length} D={dimension} {kind} sweep={row["sweep"]} E={row["energy"]:.12f} '
                  f't={row["elapsed_seconds"]:.1f}s', flush=True)
    try:
        final, history, converged = periodic_one_site(state, terms, options, observer)
        np.savez_compressed(path.with_suffix('.npz'), **{f'tensor_{i}': a for i,a in enumerate(final.tensors)})
        report.update(status='finished', history=history, converged=converged,
                      final_energy=history[-1]['energy'], sweeps=history[-1]['sweep'],
                      elapsed_seconds=history[-1]['elapsed_seconds'],
                      stop_reason='energy density stable for three sweeps' if converged else '500-sweep cap' if sweeps==500 else 'sweep cap')
        save(path, report)
        print('FINISHED', path.name, report['final_energy'], report['sweeps'], flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        save(path, report)
        raise
    return report


def plot_results(output, include_warm_starts=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = [json.loads(p.read_text()) for p in output.rglob('L*_seed*.json')]
    reports = [r for r in reports if r.get('history')]
    if not include_warm_starts:
        reports = [r for r in reports if not r.get('parent_checkpoint')]
    colors = {2: '#2673b8', 3: '#e68632', 4: '#30966a', 6: '#9b5ba5'}
    references = json.loads((output/'references.json').read_text()) if (output/'references.json').exists() else []
    support_path = output/'support_references.json'
    support_references = json.loads(support_path.read_text()) if support_path.exists() else []
    for length in sorted({r['length'] for r in reports}):
        bosonic = next(r for r in reports if r['length']==length)['model'].get('name') == 'bose'
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for r in sorted(reports, key=lambda r:(r['bond_dimension'], r['method'])):
            if r['length'] != length: continue
            x = np.array([h['sweep'] for h in r['history']]); e = np.array([h['energy'] for h in r['history']])
            color = colors.get(r['bond_dimension'])
            style = '-' if r['method']=='letta' else '--'
            label = f'{r["method"].upper()} D={r["bond_dimension"]}'
            if r.get('parent_checkpoint'):
                color = '#202020'
                label += ' (D4 start)'
            if r.get('status') != 'finished': label += ' (running)'
            elif not r.get('converged'): label += ' (cap)'
            axes[0].plot(x,e,style,color=color,label=label,lw=1.7)
            delta = e-e[-1]
            # Zero has no logarithm. Do not invent a floor or take absolute values.
            valid = delta > 0
            axes[1].plot(x[valid],delta[valid],style,color=color,label=label,lw=1.7)
        for ref in references:
            if ref['length']==length:
                axes[0].axhline(ref['energy'],color='0.3',lw=1,ls=':',label='Exact fixed-N ground state')
        for ref in support_references:
            if ref['length']==length and ref['bond_dimension']==3:
                axes[0].axhline(ref['support_energy_lower_bound'],color='0.55',lw=1,ls='-.',
                                label='Charge-support bound (D=3,4,6)')
        axes[0].set_ylabel('Energy / t'); axes[1].set_ylabel(r'$(E_k-E_{\mathrm{final\ round}})/t$')
        axes[1].set_yscale('log')
        for ax in axes:
            ax.set_xlabel('One-site sweep (one pass through all sites)')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8,ncol=2)
        if bosonic and any(r['length']==length and r['bond_dimension']>=3 for r in reports):
            detail = axes[0].inset_axes([.24, .17, .72, .35])
            selected = [r for r in reports if r['length']==length and r['bond_dimension']>=3]
            for r in selected:
                detail.plot([h['sweep'] for h in r['history']],
                            [h['energy'] for h in r['history']],
                            '-' if r['method']=='letta' else '--',
                            color=colors.get(r['bond_dimension']), lw=1.2)
            low = min(r['history'][-1]['energy'] for r in selected)
            high = max(r['history'][-1]['energy'] for r in selected)
            for ref in references:
                if ref['length']==length:
                    low=min(low,ref['energy'])
                    detail.axhline(ref['energy'],color='0.3',ls=':',lw=1)
            for ref in support_references:
                if ref['length']==length and ref['bond_dimension']==3:
                    detail.axhline(ref['support_energy_lower_bound'],color='0.55',ls='-.',lw=1)
            margin=max((high-low)*.15,1e-4)
            detail.set_ylim(low-margin,high+margin)
            detail.set_title('Low-energy detail (D ≥ 3)', fontsize=8)
            detail.tick_params(labelsize=7)
            detail.ticklabel_format(useOffset=False,axis='y')
            detail.grid(alpha=.2)
        name = 'Bose–Hubbard' if bosonic else 'Hubbard'
        cutoff = next(r for r in reports if r['length']==length)['model'].get('max_occupancy')
        suffix = f', n_max={cutoff}' if bosonic else ''
        fig.suptitle(f'Periodic {name}: L={length}, N={length}, t=1, U=4{suffix} | periodic virtual bonds')
        prefix = 'periodic_bose_hubbard' if bosonic else 'periodic_hubbard'
        fig.savefig(output/f'{prefix}_L{length}.png',dpi=180)
        fig.savefig(output/f'{prefix}_L{length}.pdf')
        plt.close(fig)
    save(output/'summary.json',[{k:r.get(k) for k in ('length','bond_dimension','method','charges','initialization','status','final_energy','sweeps','converged','elapsed_seconds')} for r in reports])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','reference','plot'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[5,10])
    p.add_argument('--bonds',type=int,nargs='+',default=[2,3,4,6])
    p.add_argument('--methods',nargs='+',choices=['mps','letta'],default=['mps','letta'])
    p.add_argument('--sweeps',type=int,default=500)
    p.add_argument('--seed',type=int,default=731)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--model', choices=['fermi', 'bose'], default='fermi')
    p.add_argument('--max-occupancy', type=int, default=2)
    p.add_argument('--gauge-floor', type=float, default=1e-6)
    p.add_argument('--gauge-method', choices=['marginal','paper'], default='marginal')
    p.add_argument('--main-only', action='store_true', help='omit auxiliary warm starts from plots')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='reference':
        references=[]
        for length in args.lengths:
            if args.model == 'bose':
                from .periodic_bose_validation import sector_reference as bose_reference
                r=bose_reference(length, max_occupancy=args.max_occupancy)
            else:
                r=sector_reference(length)
            references.append(r);save(args.output/'references.json',references)
            print(json.dumps(r),flush=True)
    elif args.mode=='plot': plot_results(args.output, include_warm_starts=not args.main_only)
    else:
        for length in args.lengths:
            for bond in args.bonds:
                for method in args.methods:
                    run_case(args.output,length,bond,method,args.seed,args.sweeps,args.warm_start,
                             args.model,args.max_occupancy,args.gauge_floor,args.gauge_method)


if __name__=='__main__':main()

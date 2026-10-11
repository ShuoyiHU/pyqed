"""Matched SU(2) Hubbard LETTA/CBE audit, with independent per-step energy checks.

Run with PYTHONPATH=. and single-thread BLAS. Outputs default to /private/tmp.
This benchmark uses reduced operators only; plotting is reproducible from JSON.
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import argparse,json,pickle,time
import numpy as np

def initial_mps(n,electrons,spin,cap,convention,seed,allocation_only=False):
    from pyqed.mps.nonabelian.states import (build_random_reduced_spatial_mps,
                                            spatial_target_sector,_fuse_spatial_sectors)
    base=build_random_reduced_spatial_mps(n,target_sector=spatial_target_sector(electrons,spin),bond_multiplicity=1,seed=seed)
    physical=tuple(base[0].qns[1]);sectors=[tuple(base[0].qns[0])]+[tuple(s.qns[2]) for s in base]
    if max(map(len,sectors))>cap:
        # A union of complete fusion paths cannot create disconnected sectors.
        completions=[None]*(n+1);completions[n]=Counter({sectors[-1][0]:1})
        for i in reversed(range(n)):
            completions[i]=Counter({q:sum(completions[i+1][out] for p in physical
                for out in _fuse_spatial_sectors(q,p)) for q in sectors[i]})
        allowed=[set() for _ in range(n+1)];rng=np.random.default_rng(seed)
        for _ in range(cap):
            q=sectors[0][0];allowed[0].add(q)
            for i in range(n):
                choices=[out for p in physical for out in _fuse_spatial_sectors(q,p)
                    if completions[i+1][out]>0]
                weights=np.asarray([completions[i+1][out] for out in choices],dtype=float)
                q=choices[rng.choice(len(choices),p=weights/weights.sum())]
                allowed[i+1].add(q)
        sectors=[tuple(q for q in layer if q in allowed[i]) for i,layer in enumerate(sectors)]
    forward=[Counter({sectors[0][0]:1})]
    for i in range(n):
        counts=Counter()
        for q,r in forward[-1].items():
            for p in physical:
                for out in _fuse_spatial_sectors(q,p):
                    if out in sectors[i+1]:counts[out]+=r
        forward.append(counts)
    backward=[None]*(n+1);backward[n]=Counter({sectors[-1][0]:1})
    for i in reversed(range(n)):
        backward[i]=Counter({q:sum(backward[i+1][out] for p in physical for out in _fuse_spatial_sectors(q,p)) for q in sectors[i]})
    layouts=[Counter({sectors[0][0]:1})]
    for cut in range(1,n):
        counts=Counter({q:1 for q in sectors[cut]});weight=lambda q:1 if convention=='multiplets' else q.irrep.dim
        used=sum(weight(q) for q in counts)
        if used>cap:raise ValueError(f'Cap {cap} cannot retain one copy of each reachable sector at cut {cut}; adaptive sector selection is needed')
        while True:
            changed=False
            for q in sectors[cut]:
                if counts[q]<min(forward[cut][q],backward[cut][q]) and used+weight(q)<=cap:
                    counts[q]+=1;used+=weight(q);changed=True
            if not changed:break
        layouts.append(counts)
    layouts.append(Counter({sectors[-1][0]:1}))
    dimensions=[dict(multiplets=sum(x.values()),magnetic=sum(q.irrep.dim*d for q,d in x.items())) for x in layouts[1:-1]]
    if allocation_only:return dimensions
    rng=np.random.default_rng(seed);sites=[]
    for i in range(n):
        left,right=layouts[i:i+2];data={}
        for ql,dl in left.items():
            for qp in physical:
                for qr in _fuse_spatial_sectors(ql,qp):
                    if qr in right:data[ql,qp,qr]=rng.normal(size=(dl,1,right[qr]))/np.sqrt(max(1,dl*right[qr]))
        qns=[[q for q,d in left.items() for _ in range(d)],list(physical),[q for q,d in right.items() for _ in range(d)]]
        sites.append(type(base[0])(data=data,qns=qns,dirs=[-1,1,1],metadata={'physical_basis':'fully_reduced_su2'}))
    dimensions=[dict(multiplets=sum(x.values()),magnetic=sum(q.irrep.dim*d for q,d in x.items())) for x in layouts[1:-1]]
    return sites,dimensions


def integrals(width,length,c):
    n=width*length;h=np.zeros((n,n));g=np.zeros((n,)*4)
    edges=[]
    for x in range(length):
        for y in range(width):
            i=x*width+y
            if y+1<width:edges.append((i,i+1))
            if x+1<length:edges.append((i,i+width))
            if x+1<length:
                for dy in (-1,1):
                    if 0<=y+dy<width:
                        j=i+width+dy;h[i,j]=h[j,i]=-c['t_diagonal']
    for i,j in edges:h[i,j]=h[j,i]=-c['t']
    for i in range(n):g[i,i,i,i]=c['U']
    return h,g,edges


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ('ours','cbe'):
        data=json.loads((output/f'{name}.json').read_text())
        ax.plot(range(1,len(data['history'])+1),data['history'],'.-',label=name)
    ax.set(xlabel='Directional sweep',ylabel='Total energy',title='Matched reduced Hubbard audit')
    ax.legend();ax.grid(alpha=.2);fig.savefig(output/'energy_vs_sweep.png',dpi=180)


def main(a):
    if a.plot_only:
        plot(a.output);return
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA,LETTADMROptions,letta_dmrg
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    import pyqed._letta_one_site_opt.reduced_cbe as cbe

    a.output.mkdir(parents=True,exist_ok=False)
    n=2*a.length;ne=n-1
    h,g,edges=integrals(2,a.length,dict(U=12.,t=1.,t_diagonal=-.25))
    problem=ElectronicProblem(h,g,((ne+1)//2,(ne-1)//2));mpo=problem.su2_mpo()
    sites,_=initial_mps(n,ne,1,a.cap,'multiplets',4)
    state=ReducedLatticeLETTA.from_mps(sites,symmetry=problem.symmetry('su2'),
        neighborhoods=tuple((i,)+tuple(j for left,j in edges if left==i) for i in range(n)))
    options=LETTADMROptions(max_sweeps=a.sweeps,tolerance=1e-8,eigensolver_tolerance=1e-10,
        metric_tolerance=1e-12,matrix_free=True,dense_solver_threshold=32,gauge_mode='frontier',verbosity=1,
        cbe_projection_max_iterations=a.projection_iterations,
        cbe_refinement_max_iterations=32,cbe_energy_refinement_max_iterations=32,
        cbe_expansion_dimension=1,cbe_baseline_guard_fraction=0.)
    (a.output/'config.json').write_text(json.dumps(dict(length=a.length,cap=a.cap,sweeps=a.sweeps,
        projection_iterations=a.projection_iterations)))
    original=cbe.reduced_cbe_site
    count=0
    def audit(state,ham,site,direction,bond,opts):
        nonlocal count
        count+=1;before=_energy(state,ham,stable=True);bonds=list(state.bond_dimensions);start=time.perf_counter()
        update=original(state,ham,site,direction,bond,opts)
        actual=_energy(state,ham,stable=True)
        record=dict(step=count,site=site,direction=direction,before=before,after=actual,
                    seconds=time.perf_counter()-start,bonds_before=bonds,bonds_after=list(state.bond_dimensions))
        for key,value in vars(update).items():
            if key.startswith('cbe_') or key in ('accepted','relative_residual','local_converged'):
                record[key]=value
        with (a.output/'updates.jsonl').open('a') as f:f.write(json.dumps(record,default=str)+'\n')
        if update.cbe_baseline_energy is not None:
            assert actual<=update.cbe_baseline_energy+1e-9,record
        return update
    try:
        cbe.reduced_cbe_site=audit
        for method in ('ours','cbe'):
            start=time.perf_counter()
            result=letta_dmrg(mpo,state=state,options=replace(options,cbe_enabled=method=='cbe'))
            with (a.output/f'{method}.pkl').open('wb') as f:pickle.dump(result.state,f)
            (a.output/f'{method}.json').write_text(json.dumps(dict(history=[r.energy for r in result.history],
                seconds=time.perf_counter()-start,converged=result.converged)))
    finally:cbe.reduced_cbe_site=original
    plot(a.output)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--length',type=int,default=3)
    parser.add_argument('--cap',type=int,default=20)
    parser.add_argument('--sweeps',type=int,default=8)
    parser.add_argument('--projection-iterations',type=int,default=100)
    parser.add_argument('--output',type=Path,default=Path('/private/tmp/su2-cbe-guard-audit'))
    parser.add_argument('--plot-only',action='store_true')
    main(parser.parse_args())

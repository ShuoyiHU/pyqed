"""Matched two- and four-leg Hubbard CBE validation batch."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import initial_mps,start_memory_log

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

def main(a):
    plan=json.loads((a.run/'plan.json').read_text());c=plan['config'];task=plan['tasks'][a.index]
    import pyqed
    root=Path(pyqed.__file__).resolve().parents[1]
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text()) if (root/'SOURCE_MANIFEST.json').exists() else None
    if not a.smoke:
        if manifest is None or manifest['revision']!=c['source_revision']:raise ValueError('Wrong source snapshot')
        for name,digest in manifest['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed source: {name}')
    out=a.run/f'task_{a.index:02d}';out.mkdir(exist_ok=False)
    from letta_phase_diagnostics import install
    start_memory_log(out)
    width,length,cap,holes=(2,3,20,1) if a.smoke else (task['width'],task['length'],task['cap'],task['holes'])
    n=width*length;ne=n-holes;spin=ne%2
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA,LETTADMROptions,letta_dmrg
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    h,g,edges=integrals(width,length,c)
    problem=ElectronicProblem(h,g,((ne+spin)//2,(ne-spin)//2));mpo=problem.su2_mpo()
    neighborhoods=tuple((i,)+tuple(j for l,j in edges if l==i) for i in range(n))
    record=dict(task=task,config=c,source=str(root),width=width,length=length,cap=cap,electrons=ne,two_s=spin)
    sites,dims=initial_mps(n,ne,spin,cap,'multiplets',c['seed'])
    record['initial_tensor_sha256']=hashlib.sha256(b''.join(v.tobytes() for s in sites for v in s.data.values())).hexdigest()
    state=ReducedLatticeLETTA.from_mps(sites,symmetry=problem.symmetry('su2'),neighborhoods=neighborhoods)
    record['initial_energy']=float(_energy(state,mpo,stable=True))
    (out/'metadata.json').write_text(json.dumps(record,indent=2))
    print('Initial energy',record['initial_energy'],flush=True)
    install(out)
    kwargs=dict(max_sweeps=1 if a.smoke else c['max_sweeps'],tolerance=c['energy_tolerance'],
        eigensolver_tolerance=task['stationarity_tolerance'],metric_tolerance=c['metric_tolerance'],
        matrix_free=True,dense_solver_threshold=32,gauge_mode='frontier',verbosity=1)
    start=time.perf_counter()
    result=letta_dmrg(mpo,state=state,options=LETTADMROptions(**kwargs,cbe_enabled=True,
        cbe_selector='exact',cbe_expansion_dimension=1,cbe_baseline_guard_fraction=0.,
        cbe_projection_max_iterations=c['projection_max_iterations'],
        cbe_refinement_max_iterations=c['als_max_iterations'],
        cbe_energy_refinement_max_iterations=c['energy_refinement_max_iterations']))
    if max(result.state.bond_dimensions)>cap:raise ValueError('Bond cap exceeded')
    record.update(energy=float(result.energy),history=[float(r.energy) for r in result.history],
        converged=bool(result.converged),solve_seconds=time.perf_counter()-start)
    with (out/'state.tmp').open('wb') as f:pickle.dump(result.state,f)
    (out/'state.tmp').replace(out/'state.pkl')
    (out/'result.json').write_text(json.dumps(record,indent=2))
    from plot_results import plot_task
    plot_task(out,record)
    print(json.dumps({k:record[k] for k in ['energy','converged','solve_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--smoke',action='store_true');main(p.parse_args())

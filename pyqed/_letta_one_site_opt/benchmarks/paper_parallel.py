"""Matched replay for arXiv:2103.09976 shared-memory changes.

Synthetic block dimensions are internal reduced multiplicities, not D_mul.
Optionally time complete tiny chemistry stages from identical seeded states.
Run with one BLAS thread. Compilation and warm-up are excluded from medians.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from pyqed._letta_two_site_opt.sector_operator import SectorOperator
from pyqed._letta_two_site_opt.shared_sector_products import SharedSectorProducts


def configure(backend,threads):
    os.environ['PYQED_SHARED_PRODUCTS']='0' if backend=='legacy' else '1'
    os.environ['PYQED_SHARED_PRODUCT_BACKEND']='python' if backend=='python' else 'compiled'
    os.environ['PYQED_SECTOR_THREADS']=str(threads)
    os.environ['PYQED_PRODUCT_CACHE_MB']='64'
    os.environ['PYQED_PRODUCT_DISK_MB']='0'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension',type=int,default=96)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--whole-sweeps',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(123);n=args.dimension
    shapes={i:(n,2,n) for i in range(4)}
    left=rng.normal(size=(4,n,n));right=rng.normal(size=(6,n,n))
    terms=[(b,k,left,rng.normal(size=(4,6,2,2)),right) for b in shapes for k in shapes]
    t=perf_counter();op=SectorOperator(shapes,terms);preparation=perf_counter()-t
    vector=rng.normal(size=op.layout.size);rows=[]
    configure('legacy',1);reference=op.apply_packed(vector)
    for backend in ['legacy','python','compiled']:
        for threads in [1,2,4]:
            configure(backend,threads)
            t=perf_counter();out=op.apply_packed(vector);cold=perf_counter()-t
            np.testing.assert_allclose(out,reference,rtol=2e-12,atol=2e-10)
            samples=[]
            for _ in range(args.repeats):
                t=perf_counter();op.apply_packed(vector);samples.append(perf_counter()-t)
            rows.append(dict(backend=backend,threads=threads,first_action_seconds=cold,
                median_action_seconds=float(np.median(samples)),samples_seconds=samples,
                max_action_error=float(np.max(abs(out-reference)))))
    plan=SharedSectorProducts(op,64*1024**2)
    _,_,work=plan._compiled_plan(4)
    report=dict(internal_block_dimension=n,physical_copies=2,routes=len(op.routes),
        naive_MAC=2*len(op.routes)*n**3,prepared_work=work,
        shared_arena_bytes_real=plan.ram_elements*8,
        workspace_bytes_per_worker_real=op.workspace_size*8,
        preparation_seconds=preparation,actions=rows)
    if args.whole_sweeps:
        from pyqed._letta_one_site_opt.qchem import ElectronicProblem
        from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
        h=np.array([[.1,-.7,.2],[-.7,.2,-.4],[.2,-.4,.3]])
        g=np.zeros((3,)*4)
        for i in range(3):g[i,i,i,i]=4.
        p=ElectronicProblem(h,g,(1,1),.37)
        def replay():
            t=perf_counter();job=QCLETTA(p,4,seed=71);times={'preparation':perf_counter()-t}
            for name,fn in [('two_site',lambda:job.two_site(2)),('one_site_mps',lambda:job.polish(2)),
                            ('tie',lambda:job.tie([(0,2)])),('one_site_letta',lambda:job.optimize_letta(2))]:
                t=perf_counter();fn();times[name]=perf_counter()-t
            return times,[row['total_energy'] for row in job.history]
        results=[];reference_energies=None
        for backend in ['legacy','python','compiled']:
            configure(backend,2);replay()  # Warm structural/compiler caches.
            runs=[]
            for _ in range(3):
                times,energies=replay();runs.append(times)
                if reference_energies is None:reference_energies=energies
                np.testing.assert_allclose(energies,reference_energies,atol=2e-10,rtol=0.)
            results.append(dict(backend=backend,threads=2,
                median_seconds={key:float(np.median([r[key] for r in runs])) for key in runs[0]},
                final_energy_Ha=energies[-1],directional_sweeps_per_stage=2))
        report['tiny_chemistry_stages']=results
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

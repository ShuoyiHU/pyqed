"""Reproducible SU(2) CAS MPS -> orbital-information ties -> LETTA example.

Run from the repository root with PYTHONPATH=. and single-thread BLAS.
See examples/qchem/README_letta_qc.md for Fe(CO)5 and checkpoint continuation.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
from pyqed._letta_one_site_opt.orbital_information import conditional_entropies

FECO5 = '''
Fe     0.0000033513     0.0000000000     0.0000000000
C      0.0000120362     0.0000000000     1.7875478540
C      0.0000120362     0.0000000000    -1.7875478540
C     -0.8972187835    -1.5540061679     0.0000000000
C     -0.8972187835     1.5540061679     0.0000000000
C      1.7944036865     0.0000000000     0.0000000000
O      0.0000196369     0.0000000000     2.9326301797
O      0.0000196369     0.0000000000    -2.9326301797
O     -1.4707073864    -2.5472855808     0.0000000000
O     -1.4707073864     2.5472855808     0.0000000000
O      2.9413819559     0.0000000000     0.0000000000
'''


def make_problem(args):
    if args.integrals:
        with np.load(args.integrals) as data:
            p=ElectronicProblem(data['h1'],data['eri'],tuple(data['nelec']),float(data['ecore']),
                                tuple(data['orbital_order']) if 'orbital_order' in data else None)
        return p,dict(integrals=str(args.integrals),orbital_labels='zero-based chain positions')
    from pyscf import ao2mo,gto,scf,mcscf
    atom=FECO5 if args.molecule=='feco5' else ('H 0 0 0; H 0 0 0.74' if args.molecule=='h2' else 'H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2')
    basis=args.basis or ('6-31g' if args.molecule=='feco5' else 'sto-6g')
    ncas=args.ncas or (10 if args.molecule=='feco5' else (2 if args.molecule=='h2' else 4))
    nelecas=args.nelecas if args.nelecas is not None else ncas
    mol=gto.M(atom=atom,basis=basis,unit='Angstrom',spin=0,verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.max_cycle=100;mf.kernel()
    if not mf.converged:raise RuntimeError('RHF failed to converge')
    cas=mcscf.CASCI(mf,ncas,nelecas)
    h1,core=cas.get_h1eff();eri=ao2mo.restore(1,cas.get_h2eff(),ncas)
    return ElectronicProblem(h1,eri,cas.nelecas,core),dict(molecule=args.molecule,basis=basis,
        active_mo_indices_one_based=list(range(cas.ncore+1,cas.ncore+ncas+1)),
        selection='contiguous canonical RHF window; no orbital optimization',rhf_energy=float(mf.e_tot))


def plot(report, output, reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    history=report['history'];time=np.cumsum([r['seconds'] for r in history])/60
    for stage in dict.fromkeys(r['stage'] for r in history):
        mask=np.array([r['stage']==stage for r in history]);energy=np.array([r['total_energy'] for r in history])[mask]
        if reference is not None:energy=(energy-reference)*1000
        for ax,x in zip(axes,[np.arange(1,len(history)+1),time]):ax.plot(x[mask],energy,'o-',ms=3,label=stage)
    for ax in axes:
        ax.set_ylabel('Energy above FCI (mEh)' if reference is not None else 'Total energy (Eh)')
        ax.ticklabel_format(axis='y',useOffset=False);ax.grid(alpha=.2);ax.legend()
    axes[0].set_xlabel('Cumulative directional sweep');axes[1].set_xlabel('Recorded solver time (minutes)')
    fig.suptitle(f"CAS LETTA: D_mul={report['bond_dim']}, ties={report['edges']}")
    fig.savefig(output/'energies.png',dpi=180);fig.savefig(output/'energies.pdf');plt.close(fig)
    if report['information']:
        info=report['information'];mi=np.array(info['mutual_information'])
        cond=conditional_entropies(info['single_entropy'],mi)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,matrix,title,cmap in zip(axes,[mi,cond,cond.T],['Mutual information','S(i|j)','S(j|i)'],['magma','magma_r','magma_r']):
            limits={} if title=='Mutual information' else dict(vmin=np.nanmin(cond),vmax=np.nanmax(cond))
            im=ax.imshow(matrix,origin='lower',cmap=cmap,**limits);ax.set(title=title,xlabel='Chain position j (zero-based)',ylabel='Chain position i (zero-based)')
            fig.colorbar(im,ax=ax,label='nats')
        fig.savefig(output/'orbital_information.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecule',choices=['h2','h4','feco5'],default='h2')
    parser.add_argument('--integrals',type=Path)
    parser.add_argument('--basis');parser.add_argument('--ncas',type=int);parser.add_argument('--nelecas',type=int)
    parser.add_argument('--bond-dim',type=int,default=30)
    parser.add_argument('--two-site-sweeps',type=int,default=39)
    parser.add_argument('--polish-sweeps',type=int,default=600)
    parser.add_argument('--letta-sweeps',type=int,default=600)
    parser.add_argument('--tie-selection',choices=['conditional','mi'],default='conditional')
    parser.add_argument('--ties',type=int,default=2,help='Maximum disjoint pairs for MI selection only')
    parser.add_argument('--edges',help='Explicit zero-based chain pairs, e.g. 1-5,2-6,3-9')
    parser.add_argument('--allow-unconverged-mps',action='store_true',help='Explicitly permit tying before the MPS stationarity audit passes')
    parser.add_argument('--reference',action='store_true',help='Small-system FCI validation only; at most 12 orbitals')
    parser.add_argument('--output',type=Path,default=Path('letta_qc_output'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    problem,metadata=make_problem(args)
    np.savez(args.output/'integrals.npz',h1=problem.h1,eri=problem.eri,nelec=problem.nelec,
             ecore=problem.ecore,orbital_order=problem.orbital_order)
    reference=None;vector=None
    if args.reference:
        if problem.norb>12:raise ValueError('FCI validation is restricted to at most 12 orbitals')
        from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import cas_reference
        reference,vector=cas_reference(problem)
    job=QCLETTA(problem,args.bond_dim,checkpoint=args.output/'checkpoint.pkl')
    job.two_site(args.two_site_sweeps).polish(args.polish_sweeps)
    if not job.success and not args.allow_unconverged_mps:
        report=job.report();(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        plot(report,args.output,reference)
        raise RuntimeError('MPS polishing is not converged; checkpoint retained. Continue polishing or explicitly allow a pilot MPS.')
    edges=None if args.edges is None else [tuple(map(int,p.split('-'))) for p in args.edges.split(',')]
    job.tie(edges,selection=args.tie_selection,count=args.ties).optimize_letta(args.letta_sweeps)
    report=job.report();report['problem']=metadata
    if reference is not None:
        from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import physical_diagnostics
        report['reference_energy']=reference
        report['diagnostics']=physical_diagnostics(job.state,problem,reference,vector)
        if abs(report['diagnostics']['total_energy']-job.energy)>1e-8:
            raise RuntimeError('Independent PySCF energy disagrees with solver')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(report,args.output,reference)
    print(json.dumps(dict(energy=job.energy,converged=job.success,message=job.message,
                         parameters=report['parameters'],output=str(args.output)),indent=2))

if __name__=='__main__':main()

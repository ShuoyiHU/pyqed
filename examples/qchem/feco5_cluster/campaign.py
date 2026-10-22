"""Fe(CO)5 cluster campaign; run from an immutable, source-hashed bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(os.environ.get('FECO_ROOT',Path(__file__).resolve().parent))
OUT=Path(os.environ.get('OUTPUT_ROOT',ROOT/'results'))

def read(p): return json.loads(Path(p).read_text())
def config(): return read(ROOT/'config.json')
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
def checkpoint(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    with t.open('wb') as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    t.replace(p)
def signature():
    h=hashlib.sha256()
    for name in ['config.json','geometry.xyz','campaign.py','source_manifest.json']:
        h.update((ROOT/name).read_bytes())
    return h.hexdigest()
def verify_source():
    manifest=read(ROOT/'source_manifest.json')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source mismatch: '+name)
    import pyqed
    if Path(pyqed.__file__).resolve()!=(ROOT/'source/pyqed/__init__.py').resolve():
        raise RuntimeError('Wrong imported pyqed: '+str(pyqed.__file__))
    return manifest['commit']

def frontier_groups(energies,indices,count,tolerance,occupied=False):
    """Exact-size knapsack over complete energy clusters, maximizing frontier proximity.

    This is an energy-based near-degeneracy heuristic, not point-group labeling.
    It may skip a closer singleton/pair to avoid splitting another cluster.
    """
    import numpy as np
    indices=list(indices);groups=[]
    for i in indices:
        if not groups or energies[i]-energies[groups[-1][0]]>tolerance:groups.append([i])
        else:groups[-1].append(i)
    dp={0:(0.,[])}
    for group in groups:
        cost=float(sum((-1 if occupied else 1)*energies[i] for i in group))
        for n,(old,chosen) in list(dp.items()):
            k=n+len(group)
            if k<=count and (k not in dp or old+cost<dp[k][0]):dp[k]=(old+cost,chosen+group)
    if count not in dp:raise ValueError('Requested CAS splits energy groups; choose another size')
    selected=sorted(dp[count][1]);chosen=set(selected)
    assert all(not chosen.intersection(g) or set(g)<=chosen for g in groups)
    return selected,groups

def preflight():
    import importlib.metadata as meta
    from pyscf import gto
    import numpy as np
    import scipy
    atom='\n'.join((ROOT/'geometry.xyz').read_text().splitlines()[2:]);rows=[]
    for b in config()['bases']:
        m=gto.M(atom=atom,basis=b,spin=0,charge=0,verbose=0)
        assert m.nelectron==96 and not m.has_ecp()
        rows.append(dict(basis=b,nao=m.nao_nr(),dense_AO_eri_GiB=8*m.nao_nr()**4/2**30))
    report=dict(commit=verify_source(),signature=signature(),python=sys.version,
                versions={x:meta.version(x) for x in ['numpy','scipy','pyscf']},bases=rows,
                counts=dict(preparations=5,mps=80,letta=20))
    print(json.dumps(report,indent=2));return report

def prepare(basis_index):
    import numpy as np
    from scipy.linalg import fractional_matrix_power
    from pyscf import gto,scf,ao2mo
    c=config();folder=OUT/c['bases'][basis_index];folder.mkdir(parents=True,exist_ok=True)
    done=folder/'prepared.json'
    if done.exists():
        if read(done)['signature']!=signature():raise RuntimeError('Changed inputs: use new OUTPUT_ROOT')
        return
    atom='\n'.join((ROOT/'geometry.xyz').read_text().splitlines()[2:])
    mol=gto.M(atom=atom,basis=c['bases'][basis_index],spin=0,charge=0,symmetry=False,
              verbose=4,max_memory=int(os.environ.get('SCF_MEMORY_MB','24000')))
    mf=scf.RHF(mol);mf.direct_scf=True;mf.conv_tol=1e-12;mf.max_cycle=200
    mf.chkfile=str(folder/'rhf.chk');mf.kernel()
    if not mf.converged:mf=mf.newton();mf.max_cycle=100;mf.kernel()
    if not mf.converged:raise RuntimeError('RHF failed to converge')
    stable=False
    for _ in range(4):
        orbitals,_,stable,_=mf.stability(internal=True,external=False,return_status=True)
        if stable:break
        mf.kernel(dm0=mf.make_rdm1(orbitals,mf.mo_occ))
        if not mf.converged:raise RuntimeError('Stability restart failed')
    if not stable:raise RuntimeError('RHF internally unstable')
    order=np.argsort(mf.mo_energy,kind='stable');C=mf.mo_coeff[:,order];e=mf.mo_energy[order];occ=mf.mo_occ[order]
    if not (np.all(occ[:48]==2) and np.all(occ[48:]==0)):raise RuntimeError('Non-Aufbau RHF')
    S=mf.get_ovlp();np.testing.assert_allclose(C.T@S@C,np.eye(len(e)),atol=1e-9)
    np.savez_compressed(folder/'orbitals.npz',mo_coeff=C,mo_energy=e,mo_occ=occ)
    L=fractional_matrix_power(S,.5)@C;labels=mol.ao_labels()
    save(folder/'orbital_diagnostics.json',[dict(MO=i+1,energy=float(e[i]),occupation=float(occ[i]),
        leading_AOs=[dict(label=labels[k],weight=float(abs(L[k,i])**2)) for k in np.argsort(-abs(L[:,i])**2)[:10]]) for i in range(len(e))])
    for ne,no in c['spaces']:
        folder_cas=folder/f'cas_{ne}_{no}';folder_cas.mkdir(exist_ok=True)
        active_occ,og=frontier_groups(e,range(48),ne//2,c['degeneracy_tolerance_Eh'],True)
        active_vir,vg=frontier_groups(e,range(48,len(e)),no-ne//2,c['degeneracy_tolerance_Eh'])
        active=active_occ+active_vir;core=sorted(set(range(48))-set(active_occ))
        core_C=C[:,core];active_C=C[:,active];dm=2*core_C@core_C.T
        h=mf.get_hcore();v=mf.get_veff(mol,dm)
        h1=active_C.T@(h+v)@active_C;ecore=float(mol.energy_nuc()+np.einsum('ij,ji',dm,h+.5*v))
        # AO-to-MO transform uses disk, never a full nao**4 AO array.
        scratch=folder_cas/'transform.h5'
        ao2mo.kernel(mol,active_C,str(scratch),dataname='eri_mo',max_memory=mol.max_memory)
        with ao2mo.load(str(scratch),'eri_mo') as eri_file:eri=ao2mo.restore(1,np.asarray(eri_file),no)
        scratch.unlink()
        # The RHF determinant in this frozen-core Hamiltonian must reproduce RHF.
        nocc=ne//2
        determinant=ecore+2*np.trace(h1[:nocc,:nocc])+sum(2*eri[i,i,j,j]-eri[i,j,j,i] for i in range(nocc) for j in range(nocc))
        if abs(determinant-mf.e_tot)>1e-8:raise RuntimeError('Frozen-core RHF energy mismatch')
        tmp=folder_cas/'integrals.tmp.npz';np.savez_compressed(tmp,h1=h1,eri=eri,ecore=ecore,nelec=(nocc,nocc),orbital_order=np.arange(no));tmp.replace(folder_cas/'integrals.npz')
        save(folder_cas/'model.json',dict(signature=signature(),basis=c['bases'][basis_index],nelec=ne,norb=no,
            active_MOs_one_based=[i+1 for i in active],core_MOs_one_based=[i+1 for i in core],
            energy_groups_one_based=[[i+1 for i in g] for g in og+vg if set(g).intersection(active)],
            active_mo_energies=e[active].tolist(),selection=c['selection'],rhf_energy=float(mf.e_tot),
            rhf_reconstructed=float(determinant),rhf_internal_stable=True,
            integral_sha256=hashlib.sha256((folder_cas/'integrals.npz').read_bytes()).hexdigest()))
    save(done,dict(signature=signature(),rhf_energy=float(mf.e_tot),rhf_internal_stable=True))

def load_case(basis_index,space_index):
    import numpy as np
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    c=config();ne,no=c['spaces'][space_index];folder=OUT/c['bases'][basis_index]/f'cas_{ne}_{no}'
    model=read(folder/'model.json')
    if model['signature']!=signature():raise RuntimeError('Case input/source signature mismatch')
    if hashlib.sha256((folder/'integrals.npz').read_bytes()).hexdigest()!=model['integral_sha256']:raise RuntimeError('Integral hash mismatch')
    with np.load(folder/'integrals.npz') as z:p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
    return folder,p

def plot(folder,rows,stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:return
    energy=[r['total_energy'] for r in rows];elapsed=[];t=0.
    for r in rows:t+=r['seconds'];elapsed.append(t)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    ax[0].plot(range(1,len(rows)+1),energy,'.-');ax[0].set_xlabel('Directional sweep')
    ax[1].plot(elapsed,energy,'.-');ax[1].set_xlabel('Accumulated sweep time (s)')
    for a in ax:a.set_ylabel('Total energy (Eh)');a.ticklabel_format(useOffset=False,axis='y')
    fig.suptitle(stage)
    fig.savefig(folder/'progress.png',dpi=160);fig.savefig(folder/'progress.pdf');plt.close(fig)

def mps(basis_index,space_index,cap_index):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA,ReducedFrontier
    from pyqed._letta_one_site_opt.qchem import ElectronicProblem
    from pyqed._letta_one_site_opt.qchem_workflow import parameter_counts
    from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG
    c=config();folder,p=load_case(basis_index,space_index);cap=c['mps_caps'][cap_index]
    folder=folder/f'mps_D{cap}';folder.mkdir(exist_ok=True);path=folder/'checkpoint.pkl'
    history=[]
    if path.exists():
        with path.open('rb') as f:old=pickle.load(f)
        if old['signature']!=signature():raise RuntimeError('Checkpoint signature mismatch')
        if old.get('converged'):return
        state=old['state'];history=old['history']
    else:state=ReducedLatticeLETTA.random((1,p.norb),symmetry=p.symmetry('su2'),neighborhoods=tuple((i,) for i in range(p.norb)),multiplets_per_sector=1,seed=c['seed'],real=True)
    sites=ReducedFrontier.from_state(state).to_mps(state)
    # Exclude the large core offset from convergence subtraction and add it once in reports.
    electronic=ElectronicProblem(p.h1,p.eri,p.nelec,0.)
    solver=ReducedDMRG(electronic,sites,max_bond=cap,threads=1)
    remaining=c['mps_max_sweeps']-len(history)
    if remaining<1:raise RuntimeError('MPS sweep cap reached without convergence; inspect before extending')
    direction='rl' if history and history[-1]['direction']=='lr' else 'lr'
    def record(row):
        history.append(dict(stage='two_site',sweep=len(history)+1,direction=row['direction'],
            electronic_energy=row['energy'],total_energy=row['energy']+p.ecore,
            seconds=row['preparation_seconds']+row['solve_seconds']+row['verification_seconds'],
            max_residual=row['max_residual'],**parameter_counts(solver.state)))
        payload=dict(signature=signature(),state=solver.state,history=history,converged=False,bond_dim=cap)
        checkpoint(path,payload);save(folder/'progress.json',dict(history=history,converged=False))
        print(json.dumps(history[-1]),flush=True)
    solver.run(max_sweeps=remaining,energy_tolerance=c['energy_tolerance'],residual_tolerance=c['residual_tolerance'],callback=record,start_direction=direction)
    checkpoint(path,dict(signature=signature(),state=solver.state,history=history,converged=solver.success,bond_dim=cap))
    save(folder/'result.json',dict(signature=signature(),history=history,converged=solver.success,
        message=solver.message,energy=solver.energy+p.ecore,preparation_seconds=solver.setup_seconds,
        parameters=parameter_counts(solver.state),engine=c['mps_engine']))
    plot(folder,history,f'MPS D_mul={cap}')
    if not solver.success:raise RuntimeError('MPS not converged at sweep cap; dependent LETTA held')

def letta(basis_index,space_index):
    from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    c=config();case,p=load_case(basis_index,space_index);folder=case/'letta_D30';folder.mkdir(exist_ok=True)
    source=case/'mps_D30/checkpoint.pkl'
    with source.open('rb') as f:parent=pickle.load(f)
    if parent['signature']!=signature() or not parent['converged']:raise RuntimeError('Requires matching converged D30 source')
    path=folder/'checkpoint.pkl'
    if path.exists():job=QCLETTA.load(p,path)
    else:
        job=QCLETTA(p,30,seed=c['seed'],checkpoint=path);job.state=parent['state'];job.history=list(parent['history'])
        before=float(_energy(job.state,job.hamiltonian,stable=True));job.energy=before+p.ecore
        job.tie(selection='conditional')
        after=float(_energy(job.state,job.hamiltonian,stable=True))
        if abs(after-before)>1e-9:raise RuntimeError('Tying changed embedded MPS energy')
        save(folder/'ties.json',dict(edges_zero_based=job.edges,edges_MO_one_based=[[read(case/'model.json')['active_MOs_one_based'][i] for i in edge] for edge in job.edges],embedding_energy_change=after-before,information=job.information))
    if job.stages.get('letta',{}).get('converged'):return
    remaining=c['letta_max_sweeps']-sum(r['stage']=='letta' for r in job.history)
    if remaining<1:raise RuntimeError('LETTA sweep cap reached; inspect before extending')
    job.optimize_letta(remaining);save(folder/'result.json',job.report())
    plot(folder,[r for r in job.history if r['stage']=='letta'],'LETTA D_mul=30')
    if not job.success:raise RuntimeError('LETTA reached sweep cap without convergence')

def collect():
    import csv
    rows=[]
    for b in config()['bases']:
        for ne,no in config()['spaces']:
            case=OUT/b/f'cas_{ne}_{no}'
            for name in [f'mps_D{d}' for d in config()['mps_caps']]+['letta_D30']:
                p=case/name/'result.json'
                if p.exists():
                    r=read(p);rows.append(dict(basis=b,CAS=f'{ne},{no}',method=name,energy=r['energy'],converged=r.get('converged',r.get('success')),sweeps=len([x for x in r['history'] if x['stage']==('letta' if name.startswith('letta') else 'two_site')])))
    if rows:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['preflight','prepare','mps','letta','collect']);a.add_argument('--basis',type=int,default=0);a.add_argument('--space',type=int,default=0);a.add_argument('--cap',type=int,default=0);args=a.parse_args()
    verify_source()
    if args.stage=='preflight':preflight()
    elif args.stage=='prepare':prepare(args.basis)
    elif args.stage=='mps':mps(args.basis,args.space,args.cap)
    elif args.stage=='letta':letta(args.basis,args.space)
    else:collect()

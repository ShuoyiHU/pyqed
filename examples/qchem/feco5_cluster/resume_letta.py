"""Explicitly start LETTA from the final saved MPS, including a capped source.

Kept separate from campaign.py so running jobs and input signatures stay valid.
The original MPS checkpoint and its convergence status are never modified.
"""
import argparse
import hashlib
import pickle
from campaign import config,load_case,signature,save,read,plot,verify_source

def run(basis_index,space_index,accept_unconverged=False):
    from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    c=config();case,p=load_case(basis_index,space_index);folder=case/'letta_D30';folder.mkdir(exist_ok=True)
    source=case/'mps_D30/checkpoint.pkl'
    with source.open('rb') as f:parent=pickle.load(f)
    if parent['signature']!=signature() or parent['bond_dim']!=30:
        raise RuntimeError('Requires matching D30 source')
    if not parent['history']:raise RuntimeError('Source has no completed sweep')
    if not parent['converged'] and not accept_unconverged:
        raise RuntimeError('Use --accept-unconverged to authorize the last saved MPS state')
    save(folder/'source_acceptance.json',dict(source=str(source),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mps_converged=bool(parent['converged']),
        accepted_unconverged=bool(accept_unconverged and not parent['converged']),
        last_sweep=parent['history'][-1]))
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

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis',type=int,required=True)
    parser.add_argument('--space',type=int,required=True)
    parser.add_argument('--accept-unconverged',action='store_true')
    args=parser.parse_args()
    verify_source()
    run(args.basis,args.space,args.accept_unconverged)

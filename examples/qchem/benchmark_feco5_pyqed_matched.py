"""Native pyqed DMRG -> one-site polish -> NN LETTA, with matched stopping."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
from time import perf_counter
import numpy as np
from pyqed._letta_one_site_opt import LETTADMROptions,letta_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem,embed_ties
from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import (
    biased_initial_state,dmrg_run,physical_diagnostics,apply_cas_vector,
    apply_mpo_vector,state_fingerprint,write_json,
)
from pyqed.mps import cpp_davidson,abelian_direct


def counters():
    return {**abelian_direct.abelian_svd_kernel_stats(),
            **abelian_direct.abelian_environment_advance_payload_stats()}


def polish(mpo,state,method,folder,cap):
    history=[];stable=0;previous=float(state.expectation(mpo));t=perf_counter()
    for i in range(200):
        start=perf_counter()
        result=letta_dmrg(mpo,state=state,options=LETTADMROptions(max_sweeps=1,
            tolerance=1e-10/state.nsites,eigensolver_tolerance=1e-10,
            eigensolver_max_iterations=200,metric_tolerance=1e-12,
            energy_increase_tolerance=1e-11,gauge_mode='frontier',
            dense_solver_threshold=64,start_direction='lr' if i%2==0 else 'rl'))
        state=result.state;energy=float(result.energy);change=abs(energy-previous)
        if energy>previous+1e-9:raise AssertionError(('Variational energy increase',method,i,energy,previous))
        stable=stable+1 if change<=1e-10 else 0
        history.append(dict(pass_index=i+1,energy=energy,change=change,seconds=perf_counter()-start))
        previous=energy
        print(method,json.dumps(history[-1]),flush=True)
        if stable>=2:break
    elapsed=perf_counter()-t
    return state,dict(stage_wall_seconds=elapsed,sweeps=len(history),energy_stationary=stable>=2,
        electronic_energy=energy,history=history)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    for n,cap in [(4,12),(6,16)]:
        folder=args.root/f'cas_{n}_{n}';z=np.load(folder/'integrals.npz')
        p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
        e=ElectronicProblem(p.h1,p.eri,p.nelec,0.)
        ref=json.loads((folder/'reference.json').read_text())['energy'];v=np.load(folder/'reference_vector.npy')
        start=perf_counter();mpo=e.mpo(backend='symbolic');build=perf_counter()-start
        initial=biased_initial_state(p,cap,71,'hf-biased');fingerprint=state_fingerprint(initial)
        check=perf_counter()
        action_errors=[float(np.linalg.norm(apply_mpo_vector(mpo,x)-apply_cas_vector(e,x))) for x in [v,initial.state_vector()]]
        assert max(action_errors)<1e-8
        validation_setup=perf_counter()-check
        report=dict(cas=[n,n],basis='sto-6g',D=cap,dimension_unit='Abelian bond states',seed=71,
            symmetry='N and Sz, not spin SU(2)',platform=platform.platform(),threads=1,
            initial_fingerprint=fingerprint,integral_sha256=hashlib.sha256((folder/'integrals.npz').read_bytes()).hexdigest(),
            shared_mpo_build_seconds=build,operator_validation_seconds=validation_setup,operator_action_errors=action_errors,
            convergence='two successive total-energy changes <= 1e-10 Eh for both one-site stages',
            local_eigensolver_tolerance=1e-10,dmrg_two_site_budget=24,one_site_budget=200,
            cpp_available=cpp_davidson.CPP_DAVIDSON_AVAILABLE,cpp_build_error=cpp_davidson.CPP_DAVIDSON_BUILD_ERROR,
            records={})
        path=folder/f'matched_D{cap}.json';write_json(path,report)
        before=counters();start=perf_counter();mps,two=dmrg_run(mpo,initial,cap,max_sweeps=24);two_wall=perf_counter()-start
        two['stage_wall_seconds']=two_wall;two['kernel_counter_delta']={k:v-before.get(k,0) for k,v in counters().items() if isinstance(v,int)}
        report['records']['dmrg_two_site']=two;write_json(path,report)
        check=perf_counter();two['diagnostics']=physical_diagnostics(mps,p,ref,v);two['validation_seconds']=perf_counter()-check
        if abs(two['diagnostics']['total_energy']-(two['solver_electronic_energy']+p.ecore))>1e-8:raise AssertionError('DMRG physical mismatch')
        for method in ['mps_polish','letta_nn']:
            if method=='mps_polish':start_state=mps
            else:
                start_state=embed_ties(mps,tie_neighborhoods(n,nearest=True))
                distance=float(np.linalg.norm(start_state.state_vector()-mps.state_vector()))
                report['embedding_distance']=distance;assert distance<1e-10
            state,row=polish(mpo,start_state,method,folder,cap)
            check=perf_counter();diag=physical_diagnostics(state,p,ref,v);row['validation_seconds']=perf_counter()-check
            row.update(diagnostics=diag,parameter_count=state.parameter_count,bond_dimensions=state.bond_dimensions)
            assert max(state.bond_dimensions)<=cap
            assert diag['energy_error']>=-1e-8 and diag['sector_leakage']<1e-10
            assert abs(diag['total_energy']-(row['electronic_energy']+p.ecore))<1e-8
            if method=='mps_polish':mps=state
            else:assert diag['total_energy']<=report['records']['mps_polish']['diagnostics']['total_energy']+1e-9
            report['records'][method]=row
            with (folder/f'matched_D{cap}_{method}.pkl').open('wb') as f:pickle.dump(state,f)
            write_json(path,report)
        report['mps_pipeline_seconds']=two_wall+report['records']['mps_polish']['stage_wall_seconds']
        report['letta_pipeline_seconds']=report['mps_pipeline_seconds']+report['records']['letta_nn']['stage_wall_seconds']
        assert state_fingerprint(initial)==fingerprint
        write_json(path,report)
        print('FINAL',n,cap,report['mps_pipeline_seconds'],report['letta_pipeline_seconds'],flush=True)

if __name__=='__main__':main()

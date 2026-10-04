"""Matched local pyqed DMRG/LETTA pilot; no block2 or reference-space solve."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
from time import perf_counter
import numpy as np
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import (
    biased_initial_state, dmrg_run, letta_run, physical_diagnostics,
    apply_cas_vector, apply_mpo_vector, state_fingerprint, write_json,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--D',type=int,required=True)
    parser.add_argument('--passes',type=int,default=200)
    args=parser.parse_args();folder=args.case;cap=args.D
    start=perf_counter();z=np.load(folder/'integrals.npz')
    p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
    e=ElectronicProblem(p.h1,p.eri,p.nelec,0.)
    ref=json.loads((folder/'reference.json').read_text())['energy'];v=np.load(folder/'reference_vector.npy')
    setup=perf_counter()-start
    start=perf_counter();mpo=e.mpo(backend='symbolic');build=perf_counter()-start
    # Check the shared operator on the exact reference and a random fixed-N,Sz vector.
    initial=biased_initial_state(p,cap,71,'hf-biased')
    v0=initial.state_vector()
    errs=[float(np.linalg.norm(apply_mpo_vector(mpo,x)-apply_cas_vector(e,x))) for x in [v,v0]]
    if max(errs)>1e-8:raise AssertionError(('Hamiltonian mismatch',errs))
    report=dict(basis='sto-6g',cas=[sum(p.nelec),p.norb],D=cap,dimension_unit='Abelian bond states',
        symmetry='U(1) N x U(1) Sz; represented as Nalpha,Nbeta in LETTA',seed=71,
        initialization='same HF-biased random MPS for independent DMRG and LETTA starts',
        threads=1,platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        integral_sha256=hashlib.sha256((folder/'integrals.npz').read_bytes()).hexdigest(),
        initial_fingerprint=state_fingerprint(initial),fci_energy=ref,load_seconds=setup,shared_mpo_build_seconds=build,
        shared_mpo_bonds=mpo.bond_dimensions,operator_action_errors=errs,records={})
    write_json(folder/f'pyqed_D{cap}.json',report)
    for method in ['dmrg','letta_cold','letta_warm']:
        start_state=initial if method!='letta_warm' else dmrg_state
        start=perf_counter()
        if method=='dmrg':state,record=dmrg_run(mpo,start_state,cap,max_sweeps=args.passes)
        else:state,record=letta_run(mpo,start_state,cap,max_sweeps=args.passes,method='nn')
        elapsed=perf_counter()-start
        check=perf_counter();diagnostic=physical_diagnostics(state,p,ref,v);diag_seconds=perf_counter()-check
        if diagnostic['energy_error'] < -1e-8 or diagnostic['sector_leakage']>1e-10 or max(state.bond_dimensions)>cap:
            raise AssertionError(('Physical validation failed',diagnostic,state.bond_dimensions))
        difference=abs(diagnostic['total_energy']-(record['solver_electronic_energy']+p.ecore))
        if difference>1e-8:raise AssertionError(('Solver physical mismatch',difference))
        if method=='dmrg':dmrg_state=state
        if method=='letta_warm' and diagnostic['total_energy']>report['records']['dmrg']['diagnostics']['total_energy']+1e-9:
            raise AssertionError('Warm LETTA increased physical energy')
        record.update(diagnostics=diagnostic,stage_wall_seconds=elapsed,independent_validation_seconds=diag_seconds,
                      initial_fingerprint=state_fingerprint(start_state),physical_energy_discrepancy=difference)
        report['records'][method]=record
        with (folder/f'D{cap}_{method}.pkl').open('wb') as f:pickle.dump(state,f)
        write_json(folder/f'pyqed_D{cap}.json',report)
        print('RESULT',method,'error',diagnostic['energy_error'],'spin',diagnostic['spin_squared'],
              'sweeps',record['sweeps'],'converged',record['converged'],'wall',elapsed,flush=True)
    if state_fingerprint(initial)!=report['initial_fingerprint']:raise AssertionError('Shared initial state mutated')

if __name__=='__main__':main()

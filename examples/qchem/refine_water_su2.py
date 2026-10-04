"""Continue nonstationary saved water runs; preserve their initial reports."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from benchmark_water_su2 import (ElectronicProblem, ReducedLatticeLETTA, ReducedFrontier,
    tie_neighborhoods, reference, solve, diagnostics, write_json, diagnostic_state_vector)


def refine(root, max_passes):
    for folder in sorted(root.glob('cas_*')):
        paths = []
        for path in folder.glob('D*.json'):
            r = json.loads(path.read_text())
            if not r['dmrg']['polish']['energy_stationary'] or not r['letta']['optimization']['energy_stationary']:
                paths.append(path)
        if not paths:
            continue
        data = np.load(folder/'integrals.npz')
        p = ElectronicProblem(data['h1'],data['eri'],tuple(data['nelec']),float(data['ecore']),tuple(data['orbital_order']))
        h = p.su2_mpo();h.native_mpo(p.symmetry('su2').physical_basis)
        exact,vector = reference(p)
        for path in sorted(paths,key=lambda p:int(p.stem[1:])):
            r=json.loads(path.read_text());cap=r['cap']
            archive=folder/f'initial_D{cap}.json'
            if not archive.exists():write_json(archive,r)
            if not r['dmrg']['polish']['energy_stationary']:
                with (folder/f'D{cap}_dmrg.pkl').open('rb') as f:mps=pickle.load(f)
                mps,run=solve(mps,h,cap,'mps_polish',max_passes,1e-10,folder)
                r['dmrg']=dict(**diagnostics(mps,p,exact,vector),optimization=r['dmrg']['optimization'],
                    initial_polish=r['dmrg']['polish'],polish=run)
                with (folder/f'D{cap}_dmrg.pkl').open('wb') as f:pickle.dump(mps,f)
                nn=ReducedLatticeLETTA.from_mps(ReducedFrontier.from_state(mps).to_mps(mps),
                    symmetry=p.symmetry('su2'),neighborhoods=tie_neighborhoods(p.norb,nearest=True))
                r['embedding_distance']=float(np.linalg.norm(diagnostic_state_vector(nn)-diagnostic_state_vector(mps)))
            else:
                with (folder/f'D{cap}_letta.pkl').open('rb') as f:nn=pickle.load(f)
            nn,run=solve(nn,h,cap,'letta_nn',max_passes,1e-10,folder)
            if run['energy'] > r['letta']['total_energy']+1e-10:
                with (folder/f'D{cap}_letta.pkl').open('rb') as f:old_nn=pickle.load(f)
                old_nn,old_run=solve(old_nn,h,cap,'letta_nn',max_passes,1e-10,folder)
                if old_run['energy'] < run['energy']:
                    nn,run=old_nn,old_run
            r['letta']=dict(**diagnostics(nn,p,exact,vector),initial_optimization=r['letta']['optimization'],optimization=run)
            assert r['letta']['total_energy'] <= r['dmrg']['total_energy']+1e-9
            assert abs(r['dmrg']['total_energy']-r['dmrg']['polish']['energy']) < 1e-8
            assert abs(r['letta']['total_energy']-run['energy']) < 1e-8
            r['additional_refinement_pass_limit']=max_passes
            write_json(path,r)
            with (folder/f'D{cap}_letta.pkl').open('wb') as f:pickle.dump(nn,f)
            print(f'REFINED {folder.name} D={cap}: MPS {r["dmrg"]["energy_error"]:.4e}, LETTA {r["letta"]["energy_error"]:.4e}',flush=True)
        write_json(folder/'results.json',sorted((json.loads(p.read_text()) for p in folder.glob('D*.json')),key=lambda r:r['cap']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path,nargs='?',default=Path('examples/qchem/water_su2_benchmark'))
    parser.add_argument('--max-passes',type=int,default=200);args=parser.parse_args();refine(args.root,args.max_passes)

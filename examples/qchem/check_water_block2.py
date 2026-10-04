"""Independent block2 SU(2) DMRG check using the saved water integrals."""
import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


def run(root,cases,caps):
    for case in cases:
        folder=root/f'cas_{case.replace(",","_")}'
        data=np.load(folder/'integrals.npz');n=len(data['h1']);ne=int(sum(data['nelec']))
        output=folder/'block2_check.json'
        records=json.loads(output.read_text()) if output.exists() else []
        for cap in caps:
            if any(r['cap']==cap for r in records):continue
            with tempfile.TemporaryDirectory(prefix='water_block2_',dir='/private/tmp') as scratch:
                driver=DMRGDriver(scratch=scratch,stack_mem=256<<20,symm_type=SymmetryTypes.SU2,n_threads=1,fp_codec_cutoff=0.)
                driver.bw.b.Random.rand_seed(71)
                driver.initialize_system(n_sites=n,n_elec=ne,spin=0,orb_sym=[0]*n)
                mpo=driver.get_qc_mpo(data['h1'],data['eri'],ecore=float(data['ecore']),reorder=None,iprint=0)
                identity=driver.get_identity_mpo()
                ket=driver.get_random_mps(tag='KET',bond_dim=cap,nroots=1)
                t=perf_counter()
                driver.dmrg(mpo,ket,n_sweeps=24,bond_dims=[cap],noises=[1e-6,1e-7]+[0.]*22,
                    thrds=[1e-20],tol=1e-11,cutoff=0.,iprint=0)
                ket,_=driver.adjust_mps(ket,dot=1)
                energy=driver.dmrg(mpo,ket,n_sweeps=100,bond_dims=[cap],noises=[0.],thrds=[1e-20],tol=1e-11,cutoff=0.,iprint=0)
                norm=driver.expectation(ket,identity,ket)
                physical=float(driver.expectation(ket,mpo,ket)/norm)
                ket.info.load_mutable()
                dimensions=[int(x.n_states_total) for x in ket.info.left_dims]
                assert max(dimensions)<=cap,(cap,dimensions)
                assert abs(physical-energy)<1e-8,(physical,energy)
                r=dict(cap=cap,energy=physical,norm=float(norm),multiplet_dimensions=dimensions,
                    one_site_energy=float(energy),seconds=perf_counter()-t,seed=71)
                records.append(r)
                output.write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(dict(case=case,**r)),flush=True)
                driver.finalize()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('examples/qchem/water_su2_benchmark'))
    parser.add_argument('--cas',nargs='+',default=['4,4','6,6','8,6','8,8']);parser.add_argument('--caps',type=int,nargs='+',default=[3,4,8,12,16,18,24,32,48,64])
    a=parser.parse_args();run(a.root,a.cas,a.caps)

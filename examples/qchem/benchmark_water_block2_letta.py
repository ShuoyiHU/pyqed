"""H2O dimension comparison with two-seed block2 MPS initialization of LETTA.

Run in the existing block2-pyscf environment. Imports reduced tensors directly;
full vectors are post-import/post-optimization diagnostics only.
"""
import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from pyblock2.driver.core import DMRGDriver,SymmetryTypes
from pyblock2.algebra.io import MPSTools,TensorTools
from benchmark_water_su2 import (ElectronicProblem,ReducedLatticeLETTA,ReducedFrontier,
    tie_neighborhoods,reference,diagnostics,solve,write_json,apply_cas_vector,diagnostic_state_vector)
from pyqed._letta_one_site_opt.reduced_symmetry import SpinChargeSector,SU2Irrep
from pyqed._letta_two_site_opt import LETTATwoSiteOptions
from pyqed._letta_two_site_opt.reduced_solver import (_schmidt_split_untied,
    _merge_pair_blocks,_shrink_source_blocks,_retained_bond_sectors)

GRIDS={'4_4':[2,3,4,5,6], '6_6':[2,3,4,6,8,10,12,14,15,16,17,18,20],
       '8_6':[2,4,6,8,9,10,11,12], '8_8':[2,4,8,12,16,20,24,32,40,48,64]}


def import_left_canonical(ket,problem):
    """Unfuse a singlet block2 MPS with center at the last spatial orbital."""
    n=problem.norb
    if ket.center!=n-1 or ket.dot!=1 or ket.info.target.twos!=0:
        raise ValueError('import requires a one-site singlet MPS centered at the right edge')
    # block2's K-form last core can have a vacuum ket label rather than -target.
    # Explicitly unfuse its (left x physical) bra, instead of interpreting its
    # rank-two sparse matrix as an already unfused boundary tensor.
    ket.info.load_left_dims(n-1)
    left=ket.info.left_dims[n-1];basis=ket.info.basis[n-1]
    fused=basis.__class__.tensor_product_ref(left,basis,ket.info.left_dims_fci[n])
    connection=basis.__class__.get_connection_info(left,basis,fused)
    ket.load_tensor(n-1)
    last=TensorTools.from_block2_fused(ket.tensors[n-1],left,basis,fused,connection)
    ket.unload_tensor(n-1);fused.deallocate()
    converted=MPSTools.from_block2(ket);converted.tensors[-1]=last
    sym=problem.symmetry('su2');cores=[];bonds=[]
    def label(q):return SpinChargeSector(int(q.n),SU2Irrep(int(q.twos)))
    for i,tensor in enumerate(converted.tensors):
        blocks={};right={}
        for block in tensor.blocks:
            key=tuple(map(label,block.q_labels));a=np.array(block.reduced,copy=True)
            if i==0:key=(sym.identity,)+key;a=a[None,:,:]
            if i==n-1:key=key+(sym.sector,);a=a[:,:,None]
            blocks[key]=a;right[key[-1]]=a.shape[-1]
        cores.append(blocks)
        if i<n-1:bonds.append(tuple(q for q in sorted(right) for _ in range(right[q])))
    return ReducedLatticeLETTA((1,n),sym,cores,bond_sectors=bonds,
        neighborhoods=tuple((i,) for i in range(n)),normalize=False)


def compact(state,cap):
    """Remove redundant MPS multiplicities using only reduced norm Schmidt splits."""
    state=state.copy();loss=0.
    for i in reversed(range(state.nsites-1)):
        sites=tuple(ReducedFrontier.from_state(state).to_mps(state))
        split=_schmidt_split_untied(SimpleNamespace(left_site=i),
            _merge_pair_blocks(sites[i],sites[i+1]),sites,state,cap,'rl',
            LETTATwoSiteOptions(metric_tolerance=1e-12))
        retained=dict(split.retained_multiplicities)
        bonds=list(state.bond_sectors);bonds[i]=_retained_bond_sectors(bonds[i],retained)
        state.bond_sectors=tuple(bonds)
        state.tensors[i]=_shrink_source_blocks(split.left_blocks,retained,'left')
        state.tensors[i+1]=_shrink_source_blocks(split.right_blocks,retained,'right')
        state.tensors=state._validate_tensors(state.tensors)
        loss+=split.discarded_weight
    state.normalize()
    return state,float(loss)


def block2_state(p,cap,seed):
    with tempfile.TemporaryDirectory(prefix='water_su2_b2_',dir='/private/tmp') as scratch:
        driver=DMRGDriver(scratch=scratch,stack_mem=256<<20,symm_type=SymmetryTypes.SU2,n_threads=1,fp_codec_cutoff=0.)
        driver.bw.b.Random.rand_seed(seed)
        driver.initialize_system(n_sites=p.norb,n_elec=sum(p.nelec),spin=0,orb_sym=[0]*p.norb)
        mpo=driver.get_qc_mpo(p.h1.copy(),p.eri.copy(),ecore=p.ecore,reorder=None,iprint=0)
        identity=driver.get_identity_mpo();ket=driver.get_random_mps(tag='KET',bond_dim=cap)
        start=perf_counter()
        driver.dmrg(mpo,ket,n_sweeps=24,bond_dims=[cap],noises=[1e-6,1e-7]+[0.]*22,
            thrds=[1e-20],tol=1e-11,cutoff=0.,iprint=0)
        ket,_=driver.adjust_mps(ket,dot=1)
        driver.dmrg(mpo,ket,n_sweeps=100,bond_dims=[cap],noises=[0.],thrds=[1e-20],tol=1e-11,cutoff=0.,iprint=0)
        if ket.center==0:
            driver.dmrg(mpo,ket,n_sweeps=1,bond_dims=[cap],noises=[0.],thrds=[1e-20],tol=0.,cutoff=0.,iprint=0,forward=True)
        energy=float(driver.expectation(ket,mpo,ket)/driver.expectation(ket,identity,ket))
        state=import_left_canonical(ket,p)
        vector=diagnostic_state_vector(state);norm=float(np.vdot(vector,vector).real)
        imported=float(np.vdot(vector,apply_cas_vector(p,vector)).real/norm)
        assert abs(norm-1)<1e-10 and abs(imported-energy)<1e-9,(norm,energy,imported)
        state,loss=compact(state,cap)
        v=diagnostic_state_vector(state);compact_energy=float(np.vdot(v,apply_cas_vector(p,v)).real/np.vdot(v,v).real)
        distance=float(np.linalg.norm(v-vector))
        assert loss<1e-12 and abs(compact_energy-energy)<1e-9 and distance<1e-6,(loss,distance,compact_energy,energy)
        assert max(state.bond_dimensions)<=cap
        record=dict(seed=seed,energy=energy,imported_energy=imported,import_norm=norm,
                    compact_energy=compact_energy,compact_distance=distance,discarded_weight=loss,seconds=perf_counter()-start)
        driver.finalize()
    return state,record


def run(source,root,cases,caps,max_passes,dmrg_only=False):
    for case in cases:
        folder=root/f'cas_{case}';folder.mkdir(parents=True,exist_ok=True)
        for name in ('integrals.npz','model.json'):
            if not (folder/name).exists():shutil.copy2(source/f'cas_{case}'/name,folder/name)
        z=np.load(folder/'integrals.npz');p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
        exact,vector=reference(p);res=float(np.linalg.norm(apply_cas_vector(p,vector)-exact*vector))
        assert res<2e-10,res
        write_json(folder/'reference.json',dict(energy=exact,residual_norm=res))
        h=p.su2_mpo();h.native_mpo(p.symmetry('su2').physical_basis)
        for cap in caps or GRIDS[case]:
            path=folder/(f'dmrg_only_D{cap}.json' if dmrg_only else f'D{cap}.json')
            if path.exists():continue
            candidates=[block2_state(p,cap,seed) for seed in (71,72)]
            selected=min(candidates,key=lambda r:r[1]['energy'])
            mps,polish=solve(selected[0],h,cap,'mps_polish',max_passes,1e-10,folder)
            md=diagnostics(mps,p,exact,vector)
            assert abs(md['total_energy']-polish['energy'])<1e-8
            if dmrg_only:
                initialization=dict(energy=selected[1]['energy'],seconds=sum(x[1]['seconds'] for x in candidates),
                    candidates=[x[1] for x in candidates],method='block2 SU2 two-site then one-site, best of seeds 71 and 72')
                row=dict(cap=cap,dimension_unit='SU2 multiplets',fci_energy=exact,reference_version=2,
                    seed=selected[1]['seed'],initialization=initialization['method'],
                    dmrg=dict(**md,optimization=initialization,polish=polish))
                write_json(path,row)
                with (folder/f'dmrg_only_D{cap}.pkl').open('wb') as f:pickle.dump(mps,f)
                print(f'DMRG_ONLY {case} D={cap} error={md["energy_error"]:.6e}',flush=True)
                continue
            tied=ReducedLatticeLETTA.from_mps(ReducedFrontier.from_state(mps).to_mps(mps),
                symmetry=p.symmetry('su2'),neighborhoods=tie_neighborhoods(p.norb,nearest=True))
            embedding=float(np.linalg.norm(diagnostic_state_vector(tied)-diagnostic_state_vector(mps)))
            nn,run=solve(tied,h,cap,'letta_nn',max_passes,1e-10,folder)
            nd=diagnostics(nn,p,exact,vector)
            assert embedding<1e-11 and nn.bond_sectors==mps.bond_sectors
            assert nd['total_energy']<=md['total_energy']+1e-9 and abs(nd['total_energy']-run['energy'])<1e-8
            initialization=dict(energy=selected[1]['energy'],seconds=sum(x[1]['seconds'] for x in candidates),
                candidates=[x[1] for x in candidates],method='block2 SU2 two-site then one-site, best of seeds 71 and 72')
            row=dict(cap=cap,dimension_unit='SU2 multiplets',fci_energy=exact,reference_version=2,
                seed=selected[1]['seed'],initialization=initialization['method'],embedding_distance=embedding,
                dmrg=dict(**md,optimization=initialization,polish=polish),letta=dict(**nd,optimization=run))
            write_json(path,row)
            for method,state in [('dmrg',mps),('letta',nn)]:
                with (folder/f'D{cap}_{method}.pkl').open('wb') as f:pickle.dump(state,f)
            print(f'FINAL {case} D={cap} MPS={md["energy_error"]:.6e} LETTA={nd["energy_error"]:.6e}',flush=True)
        write_json(folder/'results.json',sorted((json.loads(f.read_text()) for f in folder.glob('D*.json')),key=lambda r:r['cap']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('examples/qchem/water_su2_benchmark'))
    parser.add_argument('--output',type=Path,default=Path('examples/qchem/water_su2_comparison_final'))
    parser.add_argument('--cas',nargs='+',default=list(GRIDS));parser.add_argument('--caps',type=int,nargs='+');parser.add_argument('--max-passes',type=int,default=100);parser.add_argument('--dmrg-only',action='store_true')
    a=parser.parse_args();run(a.source,a.output,a.cas,a.caps,a.max_passes,a.dmrg_only)

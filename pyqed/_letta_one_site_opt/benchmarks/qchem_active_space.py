"""CAS(6,6)/CAS(8,8) LETTA comparisons with pyqed two-site Abelian DMRG.

PySCF supplies RHF integrals and an independent CASCI reference. Optimization
uses the same pyqed symbolic MPO for both solvers. Diagnostics apply the CAS
Hamiltonian in the fixed-electron determinant space; no full dense H is built.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import platform

import numpy as np

from .. import LatticeLETTA, LETTADMROptions, letta_dmrg
from ..qchem import ElectronicProblem, OCCUPATIONS, initial_state, embed_ties
from ..orbital_ordering import (tie_neighborhoods, graph_diagnostics,
    orbital_mutual_information, correlation_order, select_long_range_ties)


CASES = {
    'lif_eq': dict(atom='Li 0 0 0; F 0 0 1.56', basis='6-31g', ncas=6, nelecas=6),
    'lif_stretched': dict(atom='Li 0 0 0; F 0 0 3.0', basis='6-31g', ncas=6, nelecas=6),
    'n2_eq': dict(atom='N 0 0 0; N 0 0 1.10', basis='6-31g', ncas=6, nelecas=6),
    'n2_stretched': dict(atom='N 0 0 0; N 0 0 2.0', basis='6-31g', ncas=6, nelecas=6),
    'water': dict(atom='O 0 0 0; H 0 0 0.96; H 0.9294217 0 -0.24038', basis='6-31g', ncas=8, nelecas=8),
}


def active_problem(case):
    from pyscf import ao2mo, gto, mcscf, scf
    specification = CASES[case]
    mol = gto.M(atom=specification['atom'], basis=specification['basis'], unit='Angstrom', spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF did not converge')
    cas = mcscf.CASCI(mf, specification['ncas'], specification['nelecas'])
    h1, core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), cas.ncas)
    problem = ElectronicProblem(h1, eri, cas.nelecas, core)
    metadata = dict(case=case, **specification, unit='Angstrom', frozen_orbitals=cas.ncore,
        active_mo_indices=list(range(cas.ncore, cas.ncore+cas.ncas)),
        mo_energies=mf.mo_energy.tolist(), orbital_basis='RHF canonical, contiguous occupied/virtual active window',
        orbital_optimization=False, rhf_total_energy=float(mf.e_tot), core_energy=core,
        orbital_selection='Freeze the lowest (total electrons - active electrons)/2 RHF orbitals; take the next ncas orbitals',
        integral_sha256=hashlib.sha256(h1.tobytes()+eri.tobytes()).hexdigest())
    return problem, metadata


def determinant_map(problem):
    from pyscf import fci
    n = problem.norb
    alpha = fci.cistring.make_strings(range(n), problem.nelec[0])
    beta = fci.cistring.make_strings(range(n), problem.nelec[1])
    flats = np.empty((len(alpha),len(beta)),dtype=int)
    signs = np.empty_like(flats)
    for ia,a in enumerate(alpha):
        aa = [(int(a)>>i)&1 for i in range(n)]
        for ib,b in enumerate(beta):
            bb = [(int(b)>>i)&1 for i in range(n)]
            flats[ia,ib] = sum((aa[i]+2*bb[i])*4**(n-1-i) for i in range(n))
            signs[ia,ib] = (-1)**sum(bb[i]*aa[j] for i in range(n) for j in range(i+1,n))
    return flats, signs


def cas_reference(problem):
    from pyscf import fci
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 200
    energy, ci = solver.kernel(problem.h1,problem.eri,problem.norb,problem.nelec,ecore=problem.ecore)
    if not solver.converged:
        raise RuntimeError('CASCI reference did not converge')
    flats, signs = determinant_map(problem)
    vector = np.zeros(4**problem.norb)
    vector[flats] = ci*signs
    return float(energy), vector


def diagnostic_state_vector(state, *, max_sites=14):
    """Reference-only reconstruction through local CG tensors and MPS contraction.

    This avoids enumerating every virtual-index path for every determinant.
    Magnetic tensors/full vectors are used here only for benchmark validation.
    """
    from ..reduced_state import ReducedLatticeLETTA
    if not isinstance(state, ReducedLatticeLETTA):
        return state.state_vector()
    if state.nsites > max_sites:
        raise ValueError('full-state reconstruction is for small diagnostics only')
    from ..reduced_frontier import ReducedFrontier
    from ..reduced_contraction import expand_reduced_mps_site
    from ..reduced_symmetry import _sector_irrep
    from pyqed.mps.nonabelian.coupling import ordered_two_m_values
    value = np.ones((1, 1))
    for site in ReducedFrontier.from_state(state).to_mps(state):
        tensor = expand_reduced_mps_site(site)
        value = np.tensordot(value, tensor, axes=([-1], [0])).reshape(-1, tensor.shape[-1])
    target = _sector_irrep(state.symmetry.sector)
    return value[:, ordered_two_m_values(target).index(target.two_j)].copy()


def physical_diagnostics(state, problem, reference_energy, reference_vector):
    from pyscf import fci
    vector = diagnostic_state_vector(state)
    vector /= np.linalg.norm(vector)
    flats, signs = determinant_map(problem)
    ci = vector[flats]*signs
    sector_norm = float(np.vdot(ci,ci).real)
    eri = fci.direct_spin1.absorb_h1e(problem.h1,problem.eri,problem.norb,problem.nelec,.5)
    # Some LETTA tensors are complex dtype even when their imaginary parts vanish.
    def apply(values):
        return fci.direct_spin1.contract_2e(eri,np.asarray(values,order='C'),problem.norb,problem.nelec)
    action = apply(ci.real) + 1j*apply(ci.imag) if np.iscomplexobj(ci) else apply(ci)
    electronic = float(np.vdot(ci,action).real/sector_norm)
    residual = float(np.linalg.norm(action-electronic*ci)/np.sqrt(sector_norm))
    overlap = float(abs(np.vdot(reference_vector,vector))**2)
    spin_squared=0.
    for component in (ci.real,ci.imag):
        weight=float(np.linalg.norm(component)**2)
        if weight>1e-25:
            spin_squared+=weight*fci.spin_op.spin_square(component/np.sqrt(weight),problem.norb,problem.nelec)[0]
    spin_squared=float(spin_squared/sector_norm)
    energy = electronic+problem.ecore
    return dict(total_energy=energy, energy_error=energy-reference_energy,
                residual_norm=residual, energy_variance=residual**2,
                sector_leakage=max(0.,1-sector_norm), fci_overlap=overlap, spin_squared=spin_squared)


def biased_initial_state(problem, cap, seed, mode='hf-biased'):
    state=initial_state(problem,max_bond_dim=cap,seed=seed)
    if mode=='random':
        return state
    if mode!='hf-biased':
        raise ValueError('initialization must be hf-biased or random')
    na,nb=problem.nelec
    occupations=[(int(i<na),int(i<nb)) for i in problem.orbital_order]
    cumulative=np.cumsum(occupations,axis=0)
    path=[0]+[list(charges).index(tuple(cumulative[i])) for i,charges in enumerate(state.bond_charges)]+[0]
    for i,a in enumerate(state.tensors):
        a*=.1
        physical=occupations[i][0]+2*occupations[i][1]
        a[path[i],physical,path[i+1]]=1.
    state.normalize()
    return state


def apply_mpo_vector(mpo,vector):
    """Reference action with O(chi * 4^n) storage, never a 4^n by 4^n matrix."""
    x=np.asarray(vector).reshape(1,1,-1)
    for w in mpo.factors:
        x=np.einsum('lrpq,lbqk->rbpk',w,x.reshape(w.shape[0],x.shape[1],4,-1),optimize=True)
        x=x.reshape(w.shape[1],-1,x.shape[-1])
    return x.ravel()


def apply_cas_vector(problem, vector):
    """Independent PySCF action in the fixed-electron sector (validation only)."""
    from pyscf import fci
    flats, signs = determinant_map(problem)
    ci = np.asarray(vector)[flats] * signs
    eri = fci.direct_spin1.absorb_h1e(
        problem.h1, problem.eri, problem.norb, problem.nelec, .5)
    def apply(component):
        return fci.direct_spin1.contract_2e(
            eri, np.asarray(component, order='C'), problem.norb, problem.nelec)
    action = apply(ci.real)
    if np.iscomplexobj(ci):
        action = action + 1j * apply(ci.imag)
    out = np.zeros_like(vector)
    out[flats] = (action + problem.ecore * ci) * signs
    return out


def state_fingerprint(state):
    digest=hashlib.sha256()
    for a in state.tensors:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def dmrg_run(mpo, start, bond_dim, *, max_sweeps=20):
    from pyqed.mps.dmrg import DMRG
    from pyqed.mps.mps import MPS,dense_to_symmetric_mpo,symmetric_to_dense
    from pyqed.mps.symmetry import SymmetryManager
    mgr=SymmetryManager(['charge','sz'])
    maps=[dict(enumerate(mgr.phys_qns)) for _ in range(start.nsites)]
    begin=perf_counter()
    sym_mpo=dense_to_symmetric_mpo(mpo.factors,maps,tol=1e-13)
    conversion_seconds=perf_counter()-begin
    def callback(**row):
        print(f"DMRG D={bond_dim} sweep={row.get('sweep')} E_elec={row.get('energy')} trunc={row.get('truncation')}",flush=True)
    na,nb=start.symmetry.sector
    solver=DMRG(sym_mpo,bond_dim,init_guess=MPS([a.copy() for a in start.tensors]),
        symmetry=True,sym_mgr=mgr,target_qn=mgr.get_target_qn(na+nb,na-nb),site_qn_maps=maps,
        nsweeps=max_sweeps,not_conv_err=False,noise=0.,sweep_tol=1e-10,
        davidson_tol=1e-10,davidson_max_iter=150,sweep_callback=callback)
    begin=perf_counter();solver.run();elapsed=perf_counter()-begin
    dense=symmetric_to_dense(solver.ground_state,maps)
    # Follow the same sector ordering/degeneracy expansion as symmetric_to_dense.
    # Preserve the adapted DMRG bond charges for a subsequent LETTA warm start.
    bonds = []
    for tensor in solver.ground_state.factors[:-1]:
        sizes = Counter(tensor.qns[1])
        for key, block in tensor.data.items():
            sizes[key[1]] = max(sizes[key[1]], block.shape[1])
        charges = []
        for charge, sz in dict.fromkeys(tensor.qns[1]):
            charges.extend([((charge+sz)//2, (charge-sz)//2)] * sizes[(charge,sz)])
        bonds.append(tuple(charges))
    state=LatticeLETTA((1,start.nsites),4,dense.factors,
                       neighborhoods=tie_neighborhoods(start.nsites),
                       symmetry=start.symmetry,bond_charges=tuple(bonds))
    params=sum(block.size for a in solver.ground_state.factors for block in a.data.values())
    history=[dict(sweep=int(r['sweep']),direction=r.get('direction'),
                  local_energy=float(r['energy']),truncation=float(r.get('truncation') or 0.),
                  seconds=r.get('sweep_seconds')) for r in solver.sweep_history]
    return state,dict(method='pyqed_dmrg_two_site',solver='pyqed.mps.dmrg.DMRG',
        solver_electronic_energy=float(solver.e_tot),seconds=elapsed,
        symmetry_conversion_seconds=conversion_seconds,parameter_count=int(params),
        stored_entries=state.dense_parameter_count,bond_dimensions=state.bond_dimensions,
        converged=solver.converged,
        sweeps=sum(r['direction'] in {'lr','rl','h2-local'} for r in history),
        recenter_updates=sum(r['direction'].startswith('recenter') for r in history),history=history,
        history_energy_note='two-site local energies before truncation; final reported energy recomputed from returned state')


def letta_run(mpo,start,bond_dim,*,max_sweeps=20,method='nn',edges=()):
    sites=tie_neighborhoods(start.nsites,edges if method in {'direct','carried'} else (),
                           nearest=method!='mps',carry=method=='carried')
    state=embed_ties(start,sites)
    begin=perf_counter()
    result=letta_dmrg(mpo,state=state,options=LETTADMROptions(
        max_sweeps=max_sweeps,tolerance=1e-10,metric_tolerance=1e-12,
        eigensolver_tolerance=1e-10,eigensolver_max_iterations=200,
        matrix_free=True,dense_solver_threshold=64,gauge_mode='frontier',verbosity=1))
    elapsed=perf_counter()-begin
    kinds=Counter(u.metric_kind for s in result.history for u in s.updates)
    return result.state,dict(method='letta_'+method,solver='letta_dmrg',seconds=elapsed,
        solver_electronic_energy=result.energy,parameter_count=result.state.parameter_count,
        stored_entries=result.state.dense_parameter_count,bond_dimensions=result.state.bond_dimensions,
        converged=result.converged,sweep_stagnation_converged=result.converged,
        sweeps=result.sweeps,metric_kinds=dict(kinds),
        max_local_residual=max(u.residual_norm for s in result.history for u in s.updates),
        hamiltonian_applications=sum(u.hamiltonian_applications for s in result.history for u in s.updates),
        graph=graph_diagnostics(sites),neighborhoods=sites,
        history=[dict(sweep=s.sweep,direction=s.direction,energy=s.energy,seconds=s.elapsed_seconds) for s in result.history])


def write_json(path,report):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run(case,*,output,bond_dims=(16,32),methods=('dmrg','nn'),max_sweeps=20,seed=731,order='natural',initialization='hf-biased'):
    import pyscf,scipy
    begin=perf_counter();problem,metadata=active_problem(case)
    reference,vector=cas_reference(problem)
    setup_seconds=perf_counter()-begin
    entropy,weights=orbital_mutual_information(vector,problem.norb)
    permutation=correlation_order(weights) if order=='correlation' else tuple(range(problem.norb))
    if order not in {'natural','correlation'}:
        raise ValueError('order must be natural or correlation')
    problem=problem.reordered(permutation)
    if order=='correlation':
        reference,vector=cas_reference(problem)
    electronic=ElectronicProblem(problem.h1,problem.eri,problem.nelec,0.)
    begin=perf_counter();mpo=electronic.mpo(backend='symbolic');build_seconds=perf_counter()-begin
    independent_action = apply_cas_vector(electronic, vector)
    reference_residual = float(np.linalg.norm(independent_action-(reference-problem.ecore)*vector))
    reference_action_error=float(np.linalg.norm(apply_mpo_vector(mpo,vector)-independent_action))
    if reference_action_error>1e-8:
        raise AssertionError('MPO disagrees with independent CASCI reference action')
    edges=select_long_range_ties(weights[np.ix_(permutation,permutation)],max_edges=1)
    report=dict(**metadata,order_name=order,orbital_order=permutation,selected_long_edges=edges,
        fci_total_energy=reference,seed=seed,max_sweeps=max_sweeps,initialization=initialization,
        fci_mpo_action_error=reference_action_error,
        fci_reference_residual=reference_residual,
        integral_cutoff=0.,mpo_builder='pyqed AutoMPO Hopcroft-Karp',mpo_bond_dimensions=mpo.bond_dimensions,
        setup_and_reference_seconds=setup_seconds,mpo_build_seconds=build_seconds,
        mpi_ranks=1,blas_threads=1,dmrg_noise=0.,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__),
        ordering_affinity='FCI orbital MI oracle' if order=='correlation' else None,
        one_orbital_entropies=entropy.tolist(),mutual_information=weights.tolist(),records=[])
    print(f"PREPARED {case} CAS({sum(problem.nelec)},{problem.norb}) FCI={reference:.12f} MPO={mpo.bond_dimensions}",flush=True)
    write_json(output,report)
    for dim in bond_dims:
        initial=biased_initial_state(problem,dim,seed,initialization)
        initial_energy=float(initial.expectation(mpo))+problem.ecore
        fingerprint=state_fingerprint(initial)
        for method in methods:
            print(f"START {case} {order} {method} D={dim}",flush=True)
            if method=='dmrg':
                state,row=dmrg_run(mpo,initial,dim,max_sweeps=max_sweeps)
            elif method=='warm-nn':
                warm, warm_row = dmrg_run(mpo,initial,dim,max_sweeps=max_sweeps)
                warm_energy = float(warm.expectation(mpo))+problem.ecore
                embedded = embed_ties(warm,tie_neighborhoods(problem.norb,nearest=True))
                embedding_error = float(np.linalg.norm(embedded.state_vector()-warm.state_vector()))
                embedding_energy_error = abs(float(embedded.expectation(mpo))+problem.ecore-warm_energy)
                if embedding_error>1e-12 or embedding_energy_error>1e-10:
                    raise AssertionError('DMRG-to-LETTA embedding did not preserve the state')
                state,row=letta_run(mpo,warm,dim,max_sweeps=max_sweeps)
                row.update(method='letta_nn_from_dmrg',warm_start_energy=warm_energy,
                           warm_start_parameters=warm_row['parameter_count'],
                           warm_start_seconds=warm_row['seconds'],
                           warm_start_bond_charges=warm.bond_charges,
                           embedding_vector_error=embedding_error,
                           embedding_energy_error=embedding_energy_error,
                           warm_start_fingerprint=state_fingerprint(warm))
            elif method in {'mps','nn','direct','carried'}:
                state,row=letta_run(mpo,initial,dim,max_sweeps=max_sweeps,method=method,edges=edges)
            else:
                raise ValueError('unknown method')
            if state_fingerprint(initial) != fingerprint:
                raise AssertionError('solver mutated the shared initial state')
            begin=perf_counter()
            diagnostic=physical_diagnostics(state,problem,reference,vector)
            contracted=float(state.expectation(mpo))+problem.ecore
            row.update(diagnostic,bond_cap=dim,initial_energy=initial_energy,initial_fingerprint=fingerprint,
                mpo_energy_consistency=abs(contracted-diagnostic['total_energy']),
                solver_energy_consistency=abs(row['solver_electronic_energy']+problem.ecore-diagnostic['total_energy']),
                diagnostic_seconds=perf_counter()-begin)
            if row['energy_error'] < -1e-8 or row['sector_leakage']>1e-9 or row['mpo_energy_consistency']>1e-8:
                raise AssertionError('independent physical validation failed')
            if method=='warm-nn':
                row['energy_change_from_dmrg']=row['total_energy']-warm_energy
                if row['energy_change_from_dmrg']>1e-9:
                    raise AssertionError('LETTA optimization worsened its embedded DMRG state')
            report['records'].append(row);write_json(output,report)
            print(f"DONE {case} {method} D={dim} error={row['energy_error']:.5e} residual={row['residual_norm']:.5e} parameters={row['parameter_count']} seconds={row['seconds']:.2f}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--bond-dims',type=int,nargs='+',default=[16,32])
    parser.add_argument('--methods',nargs='+',choices=['dmrg','mps','nn','direct','carried','warm-nn'],default=['dmrg','nn'])
    parser.add_argument('--max-sweeps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=731)
    parser.add_argument('--order',choices=['natural','correlation'],default='natural')
    parser.add_argument('--initialization',choices=['hf-biased','random'],default='hf-biased')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    return run(args.case,output=args.output,bond_dims=args.bond_dims,methods=args.methods,
               max_sweeps=args.max_sweeps,seed=args.seed,order=args.order,initialization=args.initialization)


if __name__=='__main__':
    main()

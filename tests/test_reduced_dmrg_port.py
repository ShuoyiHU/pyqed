import numpy as np
import pytest
from pyqed._letta_one_site_opt.benchmarks.su2_cbe_guard_audit import initial_mps, integrals
from pyqed._letta_one_site_opt import ReducedLatticeLETTA
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, OCCUPATIONS
from pyqed.mps.nonabelian.canonical import right_canonicalize_sites, left_canonicalize_sites, mixed_canonicalize_sites
from pyqed.mps.nonabelian.reduced_dmrg import ReducedDMRG
from test_letta_qchem import determinant_hamiltonian


def setup(ne):
    h,g,_=integrals(2,2,dict(U=4.,t=1.,t_diagonal=0.))
    p=ElectronicProblem(h,g,((ne+1)//2,ne//2),ecore=.31)
    sites,_=initial_mps(4,ne,ne%2,30,'multiplets',7)
    return p,sites


def state(p,sites):
    return ReducedLatticeLETTA.from_mps(sites,symmetry=p.symmetry('su2'),neighborhoods=tuple((i,) for i in range(4)))


@pytest.mark.parametrize('ne',[3,4])
@pytest.mark.parametrize('gauge',[left_canonicalize_sites,right_canonicalize_sites,lambda s:mixed_canonicalize_sites(s,2)])
def test_qr_preserves_complex_wavefunction(ne,gauge):
    p,sites=setup(ne)
    rng=np.random.default_rng(91)
    for site in sites:
        site.data={k:v+1j*.1*rng.normal(size=v.shape) for k,v in site.data.items()}
    before=state(p,sites).state_vector()
    original=[{k:v.copy() for k,v in s.data.items()} for s in sites]
    output=gauge(sites)
    np.testing.assert_allclose(state(p,output).state_vector(),before,atol=2e-13,rtol=2e-13)
    for i,site in enumerate(sites):
        assert all(np.array_equal(v,original[i][k]) for k,v in site.data.items())
    assert all(len(s.qns[2])<=len(old.qns[2]) for s,old in zip(output,sites))


@pytest.mark.parametrize('ne',[3,4])
def test_engine_matches_independent_ed_and_preserves_input(ne):
    p,sites=setup(ne)
    before=state(p,sites).state_vector()
    solve=ReducedDMRG(p,sites,max_bond=30).run(max_sweeps=4)
    configurations=np.array(list(np.ndindex(4,4,4,4)))
    allowed=np.flatnonzero(np.all(OCCUPATIONS[configurations].sum(axis=1)==p.nelec,axis=1))
    dense=determinant_hamiltonian(p)[np.ix_(allowed,allowed)]
    exact=np.linalg.eigvalsh(dense)[0]
    assert solve.energy==pytest.approx(exact,abs=2e-10)
    assert solve.success
    assert all(r['max_residual']<=1.01e-10 for r in solve.history)
    np.testing.assert_array_equal(state(p,sites).state_vector(),before)
    tied=ReducedLatticeLETTA.from_mps(solve.sites,symmetry=p.symmetry('su2'),neighborhoods=((0,1),(1,2),(2,3),(3,)))
    np.testing.assert_allclose(tied.state_vector(),solve.state.state_vector(),atol=1e-13)


def test_rejects_unconverged_local_root():
    p,sites=setup(4)
    from pyqed.mps.nonabelian.sweep_engine import load_engine
    load_engine()  # A missing build must not count as solver rejection.
    with pytest.raises(RuntimeError, match='not accepted'):
        ReducedDMRG(p,sites).run(max_sweeps=1,max_iterations=1,max_space=2,residual_tolerance=1e-15)


def test_davidson_small_independent_correction_reaches_requested_tolerance():
    h,g,_=integrals(2,3,dict(U=4.,t=1.,t_diagonal=0.))
    p=ElectronicProblem(h,g,(3,3))
    sites,_=initial_mps(6,6,0,30,'multiplets',7)
    solve=ReducedDMRG(p,sites).run(max_sweeps=2,residual_tolerance=1e-10)
    assert all(row['max_residual']<1e-10 for row in solve.history)


def test_general_integrals_doublet_matches_spin_selected_reference():
    from test_letta_qchem import integrals as random_integrals
    from pyscf import fci
    p=ElectronicProblem(*random_integrals(3),(2,1),ecore=.17)
    sites,_=initial_mps(3,3,1,30,'multiplets',7)
    solve=ReducedDMRG(p,sites).run(max_sweeps=4)
    oracle=fci.addons.fix_spin_(fci.direct_spin1.FCI(),shift=10.,ss=.75)
    oracle.conv_tol=1e-12
    exact,ci=oracle.kernel(p.h1,p.eri,3,p.nelec,ecore=p.ecore)
    assert fci.spin_op.spin_square(ci,3,p.nelec)[0]==pytest.approx(.75,abs=1e-10)
    assert solve.energy==pytest.approx(exact,abs=2e-10)


def test_truncated_state_energy_checked_after_schmidt_split():
    p,sites=setup(4)
    solve=ReducedDMRG(p,sites,max_bond=3).run(max_sweeps=2)
    vector=solve.state.state_vector()
    independent=np.vdot(vector,determinant_hamiltonian(p)@vector)/np.vdot(vector,vector)
    assert solve.energy==pytest.approx(independent.real,abs=2e-11)
    assert all(len(s.qns[2])<=3 for s in solve.sites[:-1])


def test_complex_davidson_against_dense_reference():
    from pyqed.mps.nonabelian.sweep_engine import load_engine
    p,_=setup(4)
    owner=load_engine().SU2MovingEnvironment(p.h1,p.eri,4,two_s=0,cutoff=0.)
    rng=np.random.default_rng(38)
    matrix=rng.normal(size=(24,24))+1j*rng.normal(size=(24,24))
    matrix=(matrix+matrix.conj().T)/2
    owner.install_local_operator('reference',(matrix,),[0],[0],24)
    result=owner.local_davidson('reference',np.diag(matrix),rng.normal(size=24),1e-11,300,24,False)
    assert result['converged']
    assert result['energy']==pytest.approx(np.linalg.eigvalsh(matrix)[0],abs=1e-11)
    assert np.linalg.norm(matrix@result['vector']-result['energy']*result['vector'])<1e-11


def test_restart_can_continue_with_right_to_left_pass():
    p,sites=setup(4)
    first=ReducedDMRG(p,sites,max_bond=3).run(max_sweeps=1)
    resumed=ReducedDMRG(p,first.sites,max_bond=3).run(max_sweeps=1,start_direction='rl')
    uninterrupted=ReducedDMRG(p,sites,max_bond=3).run(max_sweeps=2)
    assert resumed.history[0]['direction']=='rl'
    assert resumed.energy==pytest.approx(uninterrupted.energy,abs=2e-10)


def test_owned_svd_provider_survives_workspace_release():
    """Exercise owned SVD directly, including Linux's explicit LAPACK provider."""
    from pyqed.mps.nonabelian.sweep_engine import load_engine
    from pyqed.mps.nonabelian.reduced_dmrg import _export_sites
    from pyqed._letta_one_site_opt.reduced_solver import _energy
    p,sites=setup(4)
    owner=load_engine().SU2MovingEnvironment(p.h1,p.eri,4,two_s=0,ecore=p.ecore,cutoff=0.)
    owner.install_mps(right_canonicalize_sites(sites))
    energies=[]
    for direction in ['lr','rl']:
        owner.release_workspaces()
        owner.begin_half_sweep(direction,4)
        owner.prepare_owned_half_sweep()
        records=owner.execute_owned_half_sweep(max_bond=30,max_bond_mode='reduced',
            cutoff=0.,davidson_tolerance=1e-10,max_iterations=300,max_space=96,
            accept_unconverged=False)
        owner.finish_half_sweep()
        assert all(r['solve']['converged'] for r in records)
        energies.append(float(_energy(state(p,_export_sites(owner)),p.su2_mpo(),stable=True)))
    configurations=np.array(list(np.ndindex(4,4,4,4)))
    allowed=np.flatnonzero(np.all(OCCUPATIONS[configurations].sum(axis=1)==p.nelec,axis=1))
    exact=np.linalg.eigvalsh(determinant_hamiltonian(p)[np.ix_(allowed,allowed)])[0]
    np.testing.assert_allclose(energies,exact,atol=2e-10,rtol=0.)

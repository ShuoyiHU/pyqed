"""Transfer diagnostics, exact embedding and checkpointed reduced solver regression."""
import pickle
import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.qchem_workflow import QCLETTA,compact_hamiltonian,parameter_counts
from pyqed._letta_one_site_opt.orbital_information import orbital_information,strongest_disjoint,conditional_entropies,negative_conditional_ties
from pyqed._letta_one_site_opt.orbital_ordering import orbital_mutual_information
from pyqed._letta_one_site_opt import ReducedLatticeLETTA,ReducedFrontier
from pyqed._letta_one_site_opt.reduced_contraction import expand_reduced_mps_site


def problem(n=4):
    h=np.diag(np.arange(n)*.1);h+=np.diag(np.full(n-1,-.5),1)+np.diag(np.full(n-1,-.5),-1)
    eri=np.zeros((n,)*4)
    for i in range(n):eri[i,i,i,i]=1.
    return ElectronicProblem(h,eri,(n//2,n//2),3.1)


@pytest.mark.parametrize('seed',[71,72])
def test_transfers_match_complex_fermionic_reference(seed):
    p=problem()
    state=ReducedLatticeLETTA.random((1,4),symmetry=p.symmetry('su2'),neighborhoods=((0,),(1,),(2,),(3,)),multiplets_per_sector=2,seed=seed,real=False)
    sites=[expand_reduced_mps_site(s) for s in ReducedFrontier.from_state(state).to_mps(state)]
    single,mi,_=orbital_information(sites)
    expected,weights=orbital_mutual_information(state.state_vector(),4)
    np.testing.assert_allclose(single,expected,atol=1e-11)
    np.testing.assert_allclose(mi,weights,atol=1e-11)
    cond=conditional_entropies(single,mi)
    off=~np.eye(4,dtype=bool)
    np.testing.assert_allclose((cond-cond.T)[off],(single[:,None]-single[None,:])[off],atol=1e-12)
    assert np.isnan(cond.diagonal()).all()


def test_selection_and_validation():
    mi=np.array([[0,9,8,0],[9,0,0,7],[8,0,0,6],[0,7,6,0.]])
    assert strongest_disjoint(mi)[0]==[(0,1),(2,3)]
    assert strongest_disjoint(mi,0)[0]==[]
    assert strongest_disjoint(np.zeros((4,4)),3)[0]==[]
    with pytest.raises(ValueError):strongest_disjoint(mi,-1)
    with pytest.raises(ValueError):orbital_information([np.zeros((1,4,1))])
    with pytest.raises(ValueError):orbital_information([np.zeros((1,3,1))])


def test_compaction_preserves_operator():
    p=problem(2);original=p.su2_mpo();basis=p.symmetry('su2').physical_basis
    mpo=original.native_mpo(basis);compact=compact_hamiltonian(original,basis)
    assert compact.native_mpo(basis) is mpo
    def matrix(factors):
        out=np.ones((1,1,1))
        for core in factors:
            a=np.asarray(core.as_dense() if hasattr(core,'as_dense') else core)
            out=np.einsum('aij,abkl->bikjl',out,a).reshape(a.shape[1],out.shape[1]*4,out.shape[2]*4)
        return out[0]
    np.testing.assert_allclose(matrix(original.canonical_factors),matrix(compact.canonical_factors),atol=2e-12)


def test_workflow_checkpoint_embedding_and_physical_energy(tmp_path):
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import cas_reference,physical_diagnostics,diagnostic_state_vector
    p=problem();job=QCLETTA(p,4,checkpoint=tmp_path/'state.pkl')
    job.two_site(2).polish(2)
    original=diagnostic_state_vector(job.state);sectors=job.state.bond_sectors
    before=parameter_counts(job.state)['reduced_coefficients']
    job.tie([(0,2)])
    assert job.state.bond_sectors==sectors
    np.testing.assert_allclose(diagnostic_state_vector(job.state),original,atol=1e-11)
    assert parameter_counts(job.state)['reduced_coefficients']>before
    job.optimize_letta(2)
    reference,vector=cas_reference(p)
    diagnostics=physical_diagnostics(job.state,p,reference,vector)
    assert abs(diagnostics['total_energy']-job.energy)<1e-9
    assert abs(diagnostics['spin_squared'])<1e-10
    restored=QCLETTA.load(p,tmp_path/'state.pkl')
    assert restored.history==job.history
    assert restored.edges==job.edges
    restored.optimize_letta(1)
    assert restored.history[-1]['sweep']==3
    assert restored.history[-1]['direction']=='lr'
    wrong=ElectronicProblem(p.h1,p.eri,p.nelec,p.ecore+1)
    with pytest.raises(ValueError,match='checkpoint'):QCLETTA.load(wrong,tmp_path/'state.pkl')
    # Interrupt after a real completed sweep: the saved state must match that sweep.
    save=restored.save
    def interrupt():
        save()
        raise RuntimeError('simulated interruption')
    restored.save=interrupt
    with pytest.raises(RuntimeError,match='simulated'):restored.optimize_letta(2)
    recovered=QCLETTA.load(p,tmp_path/'state.pkl')
    assert recovered.history[-1]['sweep']==4
    d=physical_diagnostics(recovered.state,p,reference,vector)
    assert abs(d['total_energy']-recovered.energy)<1e-9


def test_negative_conditional_default_rule():
    single=np.array([.2,.3,.001])
    mi=np.array([[0,.25,0],[.25,0,0],[0,0,0.]])
    # Only S(0|1) is negative; zero and tiny positive values do not select ties.
    assert negative_conditional_ties(single,mi)==[(0,1)]
    assert negative_conditional_ties([.2,.2],[[0,.2],[.2,0]])==[]
    assert negative_conditional_ties([0.,0.],np.zeros((2,2)))==[]

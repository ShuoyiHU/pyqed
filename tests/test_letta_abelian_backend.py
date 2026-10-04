"""Exact Abelian/reduced coordinate conversion and shared symmetry solvers."""
from dataclasses import replace
import numpy as np
import pytest
from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.symmetry import AbelianSymmetry
from pyqed._letta_one_site_opt.abelian_backend import AbelianReducedMap, abelian_dmrg
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, initial_state, embed_ties
from pyqed._letta_compression import MetricCompressionOptions
from pyqed._letta_two_site_opt import LETTATwoSiteOptions
from test_letta_qchem import integrals, determinant_hamiltonian


@pytest.mark.parametrize('charges,target', [((0,1,0),1), ((-1,1),0), (((1,0),(0,1),(0,0)),(1,1))])
def test_noncontiguous_charge_copies_and_arbitrary_ties_round_trip(charges,target):
    moduli = (None,None) if isinstance(target,tuple) else None
    sym=AbelianSymmetry(charges,target,moduli)
    s=LatticeLETTA.random((1,4),physical_dim=len(charges),bond_dim=5,symmetry=sym,seed=31,real=False)
    ties=((0,3),(1,0),(2,1),(3,2))
    # Construct legal random arrays in the new tie layout.
    rng=np.random.default_rng(33)
    tensors=[rng.normal(size=(a.shape[0],len(charges),len(charges),a.shape[-1]))+
             1j*rng.normal(size=(a.shape[0],len(charges),len(charges),a.shape[-1])) for a in s.tensors]
    s=LatticeLETTA((1,4),len(charges),tensors,neighborhoods=ties,symmetry=sym,bond_charges=s.bond_charges)
    mapping=AbelianReducedMap(sym)
    reduced=mapping.to_reduced(s)
    restored=mapping.from_reduced(reduced)
    np.testing.assert_allclose(restored.state_vector(),s.state_vector(),atol=2e-12)
    for a,b in zip(restored.tensors,s.tensors):np.testing.assert_array_equal(a,b)
    assert restored.bond_charges==s.bond_charges
    assert reduced.symmetry_violation()==0.
    assert tuple(restored.site_neighborhood(i) for i in range(4))==ties


@pytest.mark.parametrize('symmetry',['n','nalpha_nbeta'])
def test_native_abelian_mpo_keeps_all_charge_components(symmetry):
    from test_letta_qchem_symmetry import dense_mpo
    p=ElectronicProblem(*integrals(3),(2,1),.2)
    mapping=AbelianReducedMap(p.symmetry(symmetry))
    h=mapping.hamiltonian(p.mpo())
    compiled=h.native_mpo(mapping.reduced_symmetry.physical_basis)
    actual=dense_mpo(compiled.component_factors())
    # Compare in the grouped local basis used by reduced tensors.
    order=mapping.physical_order
    indices=np.arange(4**3).reshape((4,)*3)[np.ix_(order,order,order)].reshape(-1)
    expected=determinant_hamiltonian(p)[np.ix_(indices,indices)]
    np.testing.assert_allclose(actual,expected,atol=2e-11)
    assert len(compiled.operator_basis.sectors[0].labels)==(2 if symmetry=='n' else 3)


@pytest.mark.parametrize('method',['one-site','cbe','two-site'])
@pytest.mark.parametrize('symmetry',['n','nalpha_nbeta'])
def test_shared_abelian_methods_match_hubbard_fci(method,symmetry):
    from pyscf import fci
    n=3;h1=-np.eye(n,k=1)-np.eye(n,k=-1);eri=np.zeros((n,)*4)
    for i in range(n):eri[i,i,i,i]=3.
    p=ElectronicProblem(h1,eri,(2,1),.13)
    s=embed_ties(initial_state(p,max_bond_dim=8,symmetry=symmetry),((0,1),(1,2),(2,)))
    compression=MetricCompressionOptions(als_max_iterations=10,lsmr_max_iterations=100)
    opts=(LETTATwoSiteOptions(max_sweeps=5,reduced_sector_growth=True,
        energy_refinement_max_iterations=4,compression=compression) if method=='two-site' else
        LETTADMROptions(max_sweeps=6,cbe_enabled=method=='cbe',compression=compression,
                       cbe_energy_refinement_max_iterations=4))
    result=abelian_dmrg(p.mpo(),state=s,options=opts,bond_dim=8 if method=='two-site' else None)
    exact=fci.direct_spin1.kernel(h1,eri,n,p.nelec,ecore=p.ecore)[0]
    assert result.energy==pytest.approx(exact,abs=2e-9)
    assert isinstance(result.state,LatticeLETTA)
    assert result.state.symmetry==s.symmetry
    assert result.state.symmetry_violation()==0.
    assert result.energy==pytest.approx(np.vdot(result.state.state_vector(),
        determinant_hamiltonian(p)@result.state.state_vector()).real,abs=2e-9)


def test_public_cbe_discovers_absent_u1_sector_without_pair_eigensolve(monkeypatch):
    import pyqed._letta_two_site_opt.reduced_solver as two
    n=2;p=ElectronicProblem(np.array([[0.,-1.],[-1.,0.]]),np.zeros((2,)*4),(1,1))
    a=np.zeros((1,4,4,3));b=np.zeros((3,4,1));a[0,3,0,0]=1.;b[0,0,0]=1.
    s=LatticeLETTA((1,2),4,[a,b],symmetry=p.symmetry('n'),bond_charges=((2,2,2),))
    def forbidden(*args,**kwargs):raise AssertionError('Abelian CBE called pair eigensolve')
    monkeypatch.setattr(two,'_solve_local_problem',forbidden)
    result=letta_dmrg(p.mpo(),state=s,options=LETTADMROptions(cbe_enabled=True,max_sweeps=5,
        cbe_expansion_dimension=2,cbe_energy_refinement_max_iterations=4))
    assert result.energy==pytest.approx(-2.,abs=2e-9)
    assert result.state.symmetry_violation()==0.
    assert any(u.cbe_expansion_dimension for row in result.history for u in row.updates)


@pytest.mark.parametrize('model_name', ['bose_hubbard', 'heisenberg'])
@pytest.mark.parametrize('method', ['one-site', 'cbe', 'two-site'])
def test_condensed_models_use_native_abelian_blocks(model_name, method, monkeypatch):
    from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
    from pyqed._letta_one_site_opt.reduced_state import ReducedLatticeLETTA
    n = 4 if model_name == 'heisenberg' else 3
    model = build_model(model_name, '1d', n, **({'mu': 0.} if model_name == 'bose_hubbard' else {}))
    d = model.physical_dim
    charges, target = ((1,-1),0) if model_name == 'heisenberg' else ((0,1,2),3)
    sym = AbelianSymmetry(charges,target)
    s = LatticeLETTA.random((1,n),physical_dim=d,bond_dim=6,symmetry=sym,real=False,seed=111)
    # Independent sum of explicit tensor products from the model terms.
    dense = np.zeros((d**n,d**n),dtype=complex)
    for term in model.terms:
        product = np.array([[1.]])
        for i in range(n): product=np.kron(product,term.operators.get(i,np.eye(d)))
        dense += term.coefficient*product
    states = list(np.ndindex(*((d,)*n)))
    indices = [i for i,c in enumerate(states) if sum(charges[j] for j in c)==target]
    exact = np.linalg.eigvalsh(dense[np.ix_(indices,indices)])[0]
    def forbidden(*a,**kw):raise AssertionError('native Abelian sweep expanded the global state')
    monkeypatch.setattr(ReducedLatticeLETTA,'state_vector',forbidden)
    opts=(LETTATwoSiteOptions(max_sweeps=5,reduced_sector_growth=True,
        energy_refinement_max_iterations=3) if method=='two-site' else
        LETTADMROptions(max_sweeps=6,cbe_enabled=method=='cbe',cbe_energy_refinement_max_iterations=3))
    result=abelian_dmrg(model.mpo,state=s,options=opts,bond_dim=6 if method=='two-site' else None)
    assert result.energy==pytest.approx(exact,abs=2e-9)
    assert result.state.symmetry_violation()==0.


def test_second_u1_conservation_is_checked_before_optimization():
    from pyqed._letta_one_site_opt import LatticeMPO
    p=ElectronicProblem(np.zeros((1,1)),np.zeros((1,)*4),(1,0))
    spin_flip=np.zeros((4,4));spin_flip[1,2]=spin_flip[2,1]=1.
    h=LatticeMPO((spin_flip[None,None],))
    # This operator preserves particle number but breaks Nalpha and Nbeta.
    AbelianReducedMap(p.symmetry('n')).hamiltonian(h)
    with pytest.raises(ValueError,match='not a charge-conserving'):
        AbelianReducedMap(p.symmetry('nalpha_nbeta')).hamiltonian(h)


@pytest.mark.parametrize('solver', ['als','variable-projection','joint-ls','grassmann-newton'])
def test_abelian_cbe_exposes_shared_compression_controls(solver):
    p=ElectronicProblem(np.array([[0.,-1.],[-1.,0.]]),np.zeros((2,)*4),(1,1))
    a=np.zeros((1,4,4,3));b=np.zeros((3,4,1));a[0,3,0,0]=1.;b[0,0,0]=1.
    s=LatticeLETTA((1,2),4,[a,b],symmetry=p.symmetry('n'),bond_charges=((2,2,2),))
    compression=MetricCompressionOptions(solver=solver,max_iterations=12,als_max_iterations=2,lsmr_max_iterations=7)
    result=letta_dmrg(p.mpo(),state=s,options=LETTADMROptions(max_sweeps=1,cbe_enabled=True,
        cbe_energy_refinement_max_iterations=2,compression=compression))
    reports=[d for row in result.history for u in row.updates for d in u.cbe_compression_diagnostics]
    assert reports and all(d['requested_solver']==solver for d in reports)
    for report in reports:
        if report['used_solver']=='als':
            assert report['iterations']<=2
            assert all(r['iterations']<=7 for r in report['linear_solves'])
    assert result.energy < 0.

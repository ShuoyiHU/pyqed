"""Electronic signs, sectors, gauge support and independent PySCF references."""
from itertools import permutations

import numpy as np
import pytest

from pyqed._letta_one_site_opt import canonicalize_frontier, IdentityEnvironmentCache, letta_dmrg, LETTADMROptions
from pyqed._letta_one_site_opt.qchem import ElectronicProblem, OCCUPATIONS, initial_state, embed_ties, sector_bond_charges
from pyqed._letta_one_site_opt.orbital_ordering import (
    tie_neighborhoods, graph_diagnostics, fermionic_reorder_vector,
    orbital_mutual_information, correlation_order, ordering_cost, select_long_range_ties)


def integrals(n=3):
    rng = np.random.default_rng(431)
    h = rng.normal(size=(n,n)); h = (h+h.T)/2
    factors = rng.normal(size=(3,n,n)); factors = (factors+factors.transpose(0,2,1))/2
    g = np.einsum('xpq,xrs->pqrs', factors, factors)
    return h, g


def determinant_hamiltonian(problem):
    """Independent occupation-bit action; no Kronecker/JW helper from production."""
    n = problem.norb
    configurations = tuple(np.ndindex(*((4,)*n)))
    bits = [tuple(OCCUPATIONS[list(c)].ravel()) for c in configurations]
    lookup = {b:i for i,b in enumerate(bits)}
    matrix = np.eye(4**n)*problem.ecore
    terms = []
    for p,q in np.ndindex(n,n):
        for spin in (0,1):
            terms.append((problem.h1[p,q], [(2*p+spin,1),(2*q+spin,0)]))
    for p,q,r,s in np.ndindex(n,n,n,n):
        for spin,tau in np.ndindex(2,2):
            terms.append((.5*problem.eri[p,q,r,s], [(2*p+spin,1),(2*r+tau,1),(2*s+tau,0),(2*q+spin,0)]))
    for col,b in enumerate(bits):
        for coefficient,ops in terms:
            occupied=list(b); value=coefficient
            for k,create in reversed(ops):
                if occupied[k]==create:
                    value=0.; break
                value *= (-1)**sum(occupied[:k]); occupied[k]=create
            if value:
                matrix[lookup[tuple(occupied)],col] += value
    return matrix


def test_integral_mpo_matches_independent_determinant_action_and_conserves_particles():
    h,g=integrals()
    p=ElectronicProblem(h,g,(2,1),.37)
    dense=p.mpo().to_dense()
    expected=determinant_hamiltonian(p)
    np.testing.assert_allclose(dense,expected,atol=2e-12,rtol=0.)
    np.testing.assert_allclose(dense,dense.T,atol=2e-12,rtol=0.)
    counts=np.array([OCCUPATIONS[list(c)].sum(axis=0) for c in np.ndindex(4,4,4)])
    for spin in range(2):
        np.testing.assert_allclose(dense*(counts[:,spin,None]-counts[None,:,spin]),0.,atol=2e-12)


def test_fermionic_reordering_matches_permuted_integrals_for_all_three_site_orders():
    h,g=integrals()
    p=ElectronicProblem(h,g,(2,1))
    dense=p.mpo().to_dense()
    for order in permutations(range(3)):
        transform=np.column_stack([fermionic_reorder_vector(v,order) for v in np.eye(64)])
        np.testing.assert_allclose(p.reordered(order).mpo().to_dense(),transform@dense@transform.T,atol=2e-12)
        np.testing.assert_allclose(transform.T@transform,np.eye(64),atol=0)


def test_charge_complete_initialization_and_exact_shared_embedding():
    h,g=integrals(4)
    p=ElectronicProblem(h,g,(2,2))
    with pytest.raises(ValueError,match='at least 9'):
        initial_state(p,max_bond_dim=8)
    mps=initial_state(p,max_bond_dim=9)
    assert mps.bond_dimensions==(4,9,4)
    for sites in [tie_neighborhoods(4,nearest=True),tie_neighborhoods(4,[(0,3)],nearest=True),
                  tie_neighborhoods(4,[(0,3)],nearest=True,carry=True)]:
        tied=embed_ties(mps,sites)
        np.testing.assert_allclose(tied.state_vector(),mps.state_vector(),atol=2e-15)
        assert tied.symmetry_violation()==0
        for configuration,amplitude in zip(np.ndindex(4,4,4,4),tied.state_vector()):
            if not np.array_equal(OCCUPATIONS[list(configuration)].sum(axis=0),(2,2)):
                assert amplitude==0


def test_nn_gauge_preserves_wavefunction_and_is_identity_on_supported_metric():
    h,g=integrals(4)
    mps=initial_state(ElectronicProblem(h,g,(2,2)),max_bond_dim=9)
    tied=embed_ties(mps,tie_neighborhoods(4,nearest=True))
    original=tied.state_vector()
    for center in range(4):
        state=tied.copy(); canonicalize_frontier(state,center)
        np.testing.assert_allclose(state.state_vector(),original,atol=3e-14)
        cache=IdentityEnvironmentCache(state)
        metric=cache.effective_metric(cache.build_left_environments()[center],cache.build_right_environments()[center+1],center).to_dense()
        allowed=np.flatnonzero(state.symmetry_mask(center).ravel())
        values=np.linalg.eigvalsh(metric[np.ix_(allowed,allowed)])
        np.testing.assert_allclose(values[values>1e-10],1.,atol=2e-12)


def test_direct_and_carried_graph_costs_and_gauge_conditions():
    nn=graph_diagnostics(tie_neighborhoods(4,nearest=True))
    direct=graph_diagnostics(tie_neighborhoods(4,[(0,3)],nearest=True))
    carried=graph_diagnostics(tie_neighborhoods(4,[(0,3)],nearest=True,carry=True))
    assert nn['all_cuts_conditional'] and nn['max_frontier_width']==1
    assert not direct['all_cuts_conditional'] and direct['max_frontier_width']==2
    assert carried['all_cuts_conditional'] and carried['max_frontier_width']==2
    assert carried['max_hamiltonian_frontier_configurations']==256


def test_fermionic_mi_covaries_under_permutation_and_orders_strong_pairs():
    rng=np.random.default_rng(442)
    vector=rng.normal(size=256)
    configs=np.array(list(np.ndindex(4,4,4,4)))
    vector[~np.all(OCCUPATIONS[configs].sum(axis=1)==(2,2),axis=1)]=0
    entropy,weights=orbital_mutual_information(vector,4)
    for order in [(0,2,1,3),(3,1,0,2)]:
        e,w=orbital_mutual_information(fermionic_reorder_vector(vector,order),4)
        np.testing.assert_allclose(e,entropy[list(order)],atol=2e-14)
        np.testing.assert_allclose(w,weights[np.ix_(order,order)],atol=2e-14)
    affinity=np.array([[0.,0.,2.,0.],[0.,0.,0.,1.],[2.,0.,0.,0.],[0.,1.,0.,0.]])
    order=correlation_order(affinity)
    assert ordering_cost(affinity,order)==3.
    assert select_long_range_ties(affinity,max_edges=1)==((0,2),)
    assert select_long_range_ties(affinity,max_frontier_width=1)==()


def test_independent_pyscf_fci_energy_and_ci_basis():
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.benchmarks.qchem_ground_state import fci_reference
    h,g=integrals()
    problem=ElectronicProblem(h,g,(2,1),.31)
    e,v=fci_reference(problem)
    np.testing.assert_allclose(problem.mpo().to_dense()@v,e*v,atol=2e-10)
    allowed=[i for i,c in enumerate(np.ndindex(4,4,4)) if np.array_equal(OCCUPATIONS[list(c)].sum(axis=0),(2,1))]
    exact=np.linalg.eigvalsh(determinant_hamiltonian(problem)[np.ix_(allowed,allowed)])[0]
    assert abs(e-exact)<1e-10


@pytest.mark.parametrize('case',['h2_equilibrium','h2_stretched'])
def test_molecular_nn_ground_state_is_fci_without_dense_solver_path(case,monkeypatch):
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.benchmarks.qchem_ground_state import molecular_problem,fci_reference
    from pyqed._letta_one_site_opt import LatticeLETTA,LatticeMPO
    problem,_=molecular_problem(case)
    exact,_=fci_reference(problem)
    state=embed_ties(initial_state(problem,max_bond_dim=4),tie_neighborhoods(2,nearest=True))
    mpo=problem.mpo()
    def forbidden(*args,**kwargs):
        raise AssertionError('dense Hilbert-space helper entered solver')
    with monkeypatch.context() as m:
        m.setattr(LatticeLETTA,'state_vector',forbidden)
        m.setattr(LatticeLETTA,'local_frame',forbidden)
        m.setattr(LatticeMPO,'to_dense',forbidden)
        result=letta_dmrg(mpo,state=state,options=LETTADMROptions(max_sweeps=8,gauge_mode='frontier'))
    assert abs(result.energy-exact)<1e-10
    assert result.state.symmetry_violation()==0


def test_input_validation_and_zero_hamiltonian():
    h,g=integrals(2)
    with pytest.raises(ValueError,match='symmetric'):
        ElectronicProblem(np.array([[1.,2.],[3.,4.]]),g,(1,1))
    with pytest.raises(ValueError,match='real integrals'):
        ElectronicProblem(h.astype(complex),g,(1,1))
    with pytest.raises(ValueError,match='symmetries'):
        ElectronicProblem(h,np.arange(16.).reshape(2,2,2,2),(1,1))
    with pytest.raises(ValueError,match='storage budget'):
        ElectronicProblem(h,g,(1,1)).mpo(max_storage_bytes=1)
    with pytest.raises(ValueError,match='permutation'):
        ElectronicProblem(h,g,(1,1)).reordered((0,0))
    with pytest.raises(ValueError,match='distinct'):
        tie_neighborhoods(2,[(0,0)])
    with pytest.raises(ValueError,match='budgets'):
        select_long_range_ties(np.eye(3),max_frontier_width=0)
    np.testing.assert_array_equal(ElectronicProblem(np.zeros((1,1)),np.zeros((1,)*4),(0,0)).mpo().to_dense(),np.zeros((4,4)))
    assert sector_bond_charges(1,(0,0),1)==()


def test_h4_nn_benchmark_checks_fci_gauge_and_same_initial_state():
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.benchmarks.qchem_ground_state import run_case
    report=run_case('h4_chain',max_bond_dim=9,max_sweeps=20)
    mps,nn=report['records']
    assert abs(mps['initial_energy']-nn['initial_energy'])<1e-12
    assert mps['energy_error']>1e-3
    assert abs(nn['energy_error'])<1e-10 and nn['residual_norm']<1e-8
    assert nn['parameter_count']>mps['parameter_count']
    assert nn['sector_leakage']==0.
    assert all(a['support_identity_error']<1e-10 for a in nn['metric_audit'])
    assert max(np.diff(nn['energies']),default=0.)<1e-10
    assert nn['fci_reference_residual']<1e-10

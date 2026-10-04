"""CAS integrals and independent comparison of both tensor-network solvers."""
import numpy as np
import pytest

from pyqed._letta_one_site_opt.qchem import ElectronicProblem,initial_state
from pyqed._letta_one_site_opt.benchmarks.qchem_ground_state import molecular_problem
from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import (
    active_problem,cas_reference,physical_diagnostics,dmrg_run,
    biased_initial_state,apply_cas_vector,apply_mpo_vector,determinant_map)


def test_symbolic_mpo_matches_svd_and_independent_casci():
    pytest.importorskip('pyscf')
    problem,_=molecular_problem('h4_chain')
    symbolic=problem.mpo(backend='symbolic')
    np.testing.assert_allclose(symbolic.to_dense(),problem.mpo().to_dense(),atol=1e-12)
    energy,vector=cas_reference(problem)
    np.testing.assert_allclose(symbolic.to_dense()@vector,energy*vector,atol=1e-10)


def test_lif_631g_cas66_includes_frozen_core_and_six_orbitals():
    pytest.importorskip('pyscf')
    problem,metadata=active_problem('lif_eq')
    assert problem.norb==6 and problem.nelec==(3,3)
    assert metadata['frozen_orbitals']==3
    assert metadata['basis']=='6-31g'
    reference,vector=cas_reference(problem)
    class ReferenceState:
        def state_vector(self):
            return vector.copy()
    diagnostic=physical_diagnostics(ReferenceState(),problem,reference,vector)
    assert abs(diagnostic['energy_error'])<1e-10
    assert diagnostic['residual_norm']<1e-6
    assert diagnostic['sector_leakage']<1e-12


def test_pyqed_dmrg_two_site_returns_physical_ground_state():
    pytest.importorskip('pyscf')
    problem,_=molecular_problem('h4_chain')
    electronic=ElectronicProblem(problem.h1,problem.eri,problem.nelec,0.)
    mpo=electronic.mpo(backend='symbolic')
    initial=initial_state(problem,max_bond_dim=16)
    reference,vector=cas_reference(problem)
    state,row=dmrg_run(mpo,initial,16,max_sweeps=8)
    diagnostic=physical_diagnostics(state,problem,reference,vector)
    assert abs(diagnostic['energy_error'])<1e-9
    assert diagnostic['residual_norm']<1e-7
    assert abs(state.expectation(mpo)+problem.ecore-diagnostic['total_energy'])<1e-10
    assert abs(row['solver_electronic_energy']+problem.ecore-reference)<1e-9


def test_cas66_symbolic_action_matches_complex_ci_without_dense_hamiltonian(monkeypatch):
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.operators import LatticeMPO
    problem,_=active_problem('lif_eq')
    problem=problem.reordered((5,2,4,0,3,1))
    def forbidden(*args,**kwargs):
        raise AssertionError('dense Hamiltonian must not be built')
    monkeypatch.setattr(LatticeMPO,'to_dense',forbidden)
    mpo=problem.mpo(backend='symbolic')
    indices,_=determinant_map(problem)
    rng=np.random.default_rng(93)
    vector=np.zeros(4**problem.norb,dtype=complex)
    vector[indices]=rng.normal(size=indices.shape)+1j*rng.normal(size=indices.shape)
    vector/=np.linalg.norm(vector)
    np.testing.assert_allclose(apply_mpo_vector(mpo,vector),apply_cas_vector(problem,vector),atol=1e-11)


def test_hf_biased_cas66_dmrg_reaches_singlet_ground_state():
    pytest.importorskip('pyscf')
    from pyqed._letta_one_site_opt.qchem import embed_ties
    from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
    problem,_=active_problem('lif_eq')
    # Verify that original occupied-orbital labels survive ordering.
    problem=problem.reordered((5,2,4,0,3,1))
    initial=biased_initial_state(problem,64,731)
    occupation=[3 if i<3 else 0 for i in problem.orbital_order]
    address=sum(p*4**(problem.norb-1-i) for i,p in enumerate(occupation))
    assert abs(initial.state_vector()[address])**2 > .99
    tied=embed_ties(initial,tie_neighborhoods(problem.norb,nearest=True))
    np.testing.assert_allclose(initial.state_vector(),tied.state_vector(),atol=1e-13)
    mpo=ElectronicProblem(problem.h1,problem.eri,problem.nelec,0.).mpo(backend='symbolic')
    reference,vector=cas_reference(problem)
    state,_=dmrg_run(mpo,initial,64,max_sweeps=12)
    diagnostic=physical_diagnostics(state,problem,reference,vector)
    assert abs(diagnostic['energy_error'])<1e-9
    assert diagnostic['residual_norm']<1e-7
    assert abs(diagnostic['spin_squared'])<1e-8


def test_dmrg_adapted_sectors_embed_exactly_and_letta_cannot_worsen_start():
    """Equal total D with different fixed charge multiplicities is not nesting."""
    pytest.importorskip('pyscf')
    from collections import Counter
    from pyqed._letta_one_site_opt.qchem import OCCUPATIONS,embed_ties
    from pyqed._letta_one_site_opt.orbital_ordering import tie_neighborhoods
    from pyqed._letta_one_site_opt.benchmarks.qchem_active_space import letta_run
    problem,_=active_problem('lif_eq')
    mpo=ElectronicProblem(problem.h1,problem.eri,problem.nelec,0.).mpo(backend='symbolic')
    initial=biased_initial_state(problem,32,731)
    mps,_=dmrg_run(mpo,initial,32,max_sweeps=8)
    vector=mps.state_vector()
    # Fix the NN frontier label (orbital 3) to empty and the left charge to
    # (2,2). Every state with the old allocation has conditional rank <= 3.
    charges=np.array([OCCUPATIONS[list(c)].sum(axis=0) for c in np.ndindex(4,4,4)])
    block=vector.reshape(64,4,16)[np.all(charges==(2,2),axis=1),0,:]
    rank=np.count_nonzero(np.linalg.svd(block,compute_uv=False)>1e-10)
    assert rank>Counter(initial.bond_charges[2])[(2,2)]
    tied=embed_ties(mps,tie_neighborhoods(problem.norb,nearest=True))
    assert tied.bond_charges==mps.bond_charges
    np.testing.assert_allclose(tied.state_vector(),vector,atol=1e-12)
    before=float(mps.expectation(mpo))
    assert abs(tied.expectation(mpo)-before)<1e-10
    optimized,_=letta_run(mpo,mps,32,max_sweeps=2)
    assert float(optimized.expectation(mpo))<=before+1e-9

import numpy as np
import pytest
from pyqed._letta_one_site_opt import ReducedLatticeLETTA, ReducedPhysicalBasis, ReducedSymmetry, su2_heisenberg_mpo
from pyqed._letta_one_site_opt.spin_fusion import coupled_pair_basis, fuse_su2_pairs
from pyqed._letta_one_site_opt.reduced_solver import _energy


def dense_mpo(factors):
    value = np.ones((1, 1, 1))
    for w in factors:
        w = np.asarray(w.as_dense() if hasattr(w, 'as_dense') else w)
        value = np.einsum('aij,abkl->bikjl', value, w).reshape(w.shape[1], value.shape[1]*w.shape[2], value.shape[2]*w.shape[3])
    return value[0]


def test_pair_basis_unitarity_and_spin_channels():
    pair = coupled_pair_basis(ReducedPhysicalBasis.spatial_orbital())
    assert pair.basis.reduced_dim == 10
    assert pair.basis.dense_dim == 16
    np.testing.assert_allclose(pair.transform.conj().T @ pair.transform, np.eye(16), atol=1e-14)
    singles = [channel for channel in pair.channels if channel[0].label == channel[1].label == 'single']
    assert {channel[2].irrep.two_j for channel in singles} == {0, 2}


@pytest.mark.parametrize('basis,charge,spin', [(ReducedPhysicalBasis.spin_half(),0,0), (ReducedPhysicalBasis.spatial_orbital(),3,1)])
def test_fusion_preserves_complex_state_and_native_energy(basis, charge, spin):
    symmetry = ReducedSymmetry.su2(basis, target_charge=charge, target_two_j=spin)
    state = ReducedLatticeLETTA.random((1,4), symmetry=symmetry, seed=7, real=False, neighborhoods=tuple((i,) for i in range(4)))
    if charge:
        from pyqed._letta_one_site_opt.qchem import ElectronicProblem
        h = np.diag([-0.7]*3, 1) + np.diag([-0.7]*3, -1)
        g = np.zeros((4,)*4)
        for i in range(4): g[i,i,i,i] = 4.
        ham = ElectronicProblem(h,g,(2,1)).su2_mpo()
    else:
        ham = su2_heisenberg_mpo(4)
    blocked, blocked_ham = fuse_su2_pairs(state, ham, neighborhoods=((0,1),(1,)))
    pair = coupled_pair_basis(basis)
    transform = np.kron(pair.transform,pair.transform)
    np.testing.assert_allclose(transform @ blocked.state_vector(),state.state_vector(), atol=2e-13)
    np.testing.assert_allclose(dense_mpo(blocked_ham.canonical_factors), transform.conj().T @ dense_mpo(ham.canonical_factors) @ transform, atol=2e-13)
    assert abs(_energy(state,ham,stable=True)-_energy(blocked,blocked_ham,stable=True)) < 1e-10
    assert blocked.bond_sectors == (state.bond_sectors[1],)
    assert blocked_ham.native_mpo(blocked.physical_basis).relative_reconstruction_error < 1e-10


def test_reject_tied_input_instead_of_silently_dropping_ties():
    state = ReducedLatticeLETTA.random((1,4),symmetry=ReducedSymmetry.su2(ReducedPhysicalBasis.spin_half(),target_two_j=0),seed=1)
    with pytest.raises(ValueError,match='untied'):
        fuse_su2_pairs(state,su2_heisenberg_mpo(4))


def test_mixed_real_complex_blocks_and_no_global_projection(monkeypatch):
    basis = ReducedPhysicalBasis.spin_half()
    state = ReducedLatticeLETTA.random((1,4),symmetry=ReducedSymmetry.su2(basis,target_two_j=0),seed=11,real=False,neighborhoods=tuple((i,) for i in range(4)))
    for tensor in state.tensors:
        key = next(iter(tensor))
        tensor[key] = tensor[key].real.copy()
    expected = state.state_vector()
    ham = su2_heisenberg_mpo(4)
    energy = _energy(state,ham,stable=True)
    def forbidden(*args,**kwargs):
        raise AssertionError('global wavefunction reconstruction used in active path')
    with monkeypatch.context() as patch:
        patch.setattr(ReducedLatticeLETTA,'state_vector',forbidden)
        blocked, bh = fuse_su2_pairs(state,ham)
        assert abs(_energy(blocked,bh,stable=True)-energy)<1e-11
    u=coupled_pair_basis(basis).transform
    np.testing.assert_allclose(np.kron(u,u) @ blocked.state_vector(),expected,atol=1e-13)


def test_spin_fusion_ties_increase_rank_at_fixed_backbone_and_preserve_singlet():
    pair = coupled_pair_basis(ReducedPhysicalBasis.spin_half())
    symmetry = ReducedSymmetry.su2(pair.basis,target_two_j=0)
    ranks=[]
    for tied in (False,True):
        neighborhoods=tuple((i,i+1) if tied and i<3 else (i,) for i in range(4))
        state=ReducedLatticeLETTA.random((1,4),symmetry=symmetry,seed=2,neighborhoods=neighborhoods,normalize=False)
        sectors=pair.basis.sectors  # J=0 and J=1, one multiplet each.
        tensors=[{key:v for key,v in tensor.items() if key[0] in sectors and key[2] in sectors} for tensor in state.tensors]
        state=ReducedLatticeLETTA((1,4),symmetry,tensors,bond_sectors=(sectors,)*3,neighborhoods=neighborhoods)
        assert state.bond_dimensions==(2,2,2)
        vector=state.state_vector()
        ranks.append(np.linalg.matrix_rank(vector.reshape(16,16),tol=1e-10))
        # Independent magnetic-basis validation: total S+ annihilates the singlet.
        transform=pair.transform
        for _ in range(3): transform=np.kron(transform,pair.transform)
        spin_vector=transform @ vector
        raising=np.array([[0.,1.],[0.,0.]])
        total=np.zeros((256,256))
        for site in range(8):
            total+=np.kron(np.kron(np.eye(2**site),raising),np.eye(2**(7-site)))
        assert np.linalg.norm(total @ spin_vector)<1e-12
    assert ranks==[4,8]

"""Direct Hubbard MPO versus independent fermionic action and reduced compiler."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed._letta_one_site_opt.reduced_symmetry import ReducedPhysicalBasis
from test_letta_qchem import determinant_hamiltonian
from test_letta_qchem_symmetry import dense_mpo

@pytest.mark.parametrize('n',[1,2,3,4])
@pytest.mark.parametrize('zero',[False,True])
def test_sparse_hubbard_matrix(n,zero,monkeypatch):
    rng=np.random.default_rng(824)
    h=rng.normal(size=(n,n));h=(h+h.T)/2
    g=np.zeros((n,)*4);ii=np.arange(n);g[ii,ii,ii,ii]=rng.uniform(.4,4,n)
    if zero:h[:]=0;g[:]=0
    p=ElectronicProblem(h,g,(1,0),0. if zero else .31)
    monkeypatch.setenv('PYQED_HUBBARD_MPO','sparse')
    sparse=p.su2_mpo().native_mpo(ReducedPhysicalBasis.spatial_orbital())
    expected=determinant_hamiltonian(p)
    np.testing.assert_allclose(dense_mpo(sparse.component_factors()),expected,atol=2e-12,rtol=2e-12)
    monkeypatch.setenv('PYQED_HUBBARD_MPO','compiled')
    compiled=p.su2_mpo().native_mpo(ReducedPhysicalBasis.spatial_orbital())
    np.testing.assert_allclose(dense_mpo(compiled.component_factors()),expected,atol=2e-11,rtol=2e-12)

def test_hubbard_bypasses_chain_factorization(monkeypatch):
    from pyqed._letta_one_site_opt.reduced_mpo_compile import SpinTensorMPO
    def forbidden(*args,**kwargs):raise AssertionError('Hubbard used chain QR/SVD compiler')
    monkeypatch.setattr(SpinTensorMPO,'compile',forbidden)
    monkeypatch.setenv('PYQED_HUBBARD_MPO','sparse')
    h=np.diag([-1.,-1.],1)+np.diag([-1.,-1.],-1)
    g=np.zeros((3,)*4);i=np.arange(3);g[i,i,i,i]=4
    ham=ElectronicProblem(h,g,(2,1)).su2_mpo()
    mpo=ham.native_mpo(ReducedPhysicalBasis.spatial_orbital())
    assert all(a.metadata['operator_construction']=='sparse_hubbard' for a in mpo.sites)

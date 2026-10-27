"""Sparse QC construction must preserve dense reference operator cores."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed.mps.nonabelian.sparse_symbolic_mpo import SparseSymbolicMPO

@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('offset', [0., .37])
def test_sparse_chemistry_matches_dense_cores(monkeypatch, n, offset):
    rng=np.random.default_rng(923)
    h=rng.normal(size=(n,n));h=(h+h.T)/2
    # Pair-factorized ERIs exercise all index coincidences and exchange signs.
    factors=rng.normal(size=(3,n,n));factors=(factors+factors.transpose(0,2,1))/2
    eri=np.einsum('apq,ars->pqrs',factors,factors)
    problem=ElectronicProblem(h,eri,(n//2,n//2),offset)
    monkeypatch.setenv('PYQED_QC_SYMBOLIC_STORAGE','dense')
    dense=problem._build_su2_mpo()
    monkeypatch.setenv('PYQED_QC_SYMBOLIC_STORAGE','sparse')
    sparse=problem._build_su2_mpo()
    for reference,actual in zip(dense.canonical_factors,sparse.canonical_factors):
        assert isinstance(actual,SparseSymbolicMPO)
        assert not actual.dense_blocks and not actual.reduced_terms
        np.testing.assert_allclose(actual.as_dense(),reference.as_dense(),atol=2e-14,rtol=1e-14)
    basis=problem.symmetry('su2').physical_basis
    a=dense.native_mpo(basis);b=sparse.native_mpo(basis)
    # Local reconstruction above fixes the complete operator; compiled spectra
    # and reconstruction bounds independently check the downstream compiler.
    assert [len(s.qns[2]) for s in a.sites]==[len(s.qns[2]) for s in b.sites]
    assert b.relative_reconstruction_error<1e-11


def test_overwrite_qr_preserves_compiled_operator(monkeypatch):
    import pyqed._letta_one_site_opt.reduced_mpo_compile as compiler

    rng = np.random.default_rng(991)
    factors = rng.normal(size=(4, 4, 4))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    h = rng.normal(size=(4, 4))
    problem = ElectronicProblem((h + h.T) / 2,
        np.einsum('apq,ars->pqrs', factors, factors), (2, 2), .37)
    cores = problem.su2_mpo().canonical_factors
    basis = problem.symmetry('su2').physical_basis
    actual = compiler.SpinTensorMPO.compile(cores, basis)
    monkeypatch.setattr(compiler, 'scipy_qr',
        lambda a, **kwargs: np.linalg.qr(a, mode='reduced'))
    reference = compiler.SpinTensorMPO.compile(cores, basis)
    for a, b in zip(actual.sites, reference.sites):
        assert a.data.keys() == b.data.keys()
        for key in a.data:
            np.testing.assert_allclose(a.data[key], b.data[key],
                                       atol=2e-13, rtol=2e-13)
    assert actual.relative_reconstruction_error < 1e-11

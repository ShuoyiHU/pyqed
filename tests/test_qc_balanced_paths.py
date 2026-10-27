"""Full small-system operators validate exact coefficient relocation."""
import numpy as np
import pytest
from pyqed._letta_one_site_opt.qchem import ElectronicProblem
from pyqed.mps.nonabelian.models import _dense_matrix_from_local_mpo


@pytest.mark.parametrize('n', [2, 3, 4])
@pytest.mark.parametrize('storage', ['dense', 'sparse'])
def test_balanced_paths_preserve_full_hamiltonian(monkeypatch, n, storage):
    rng = np.random.default_rng(923)
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    factors = rng.normal(size=(3, n, n))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    problem = ElectronicProblem(h, np.einsum('apq,ars->pqrs', factors, factors),
                                (n // 2, n // 2), .37)
    monkeypatch.setenv('PYQED_QC_SYMBOLIC_STORAGE', storage)
    monkeypatch.setenv('PYQED_QC_AUTOMPO_LAYOUT', 'prefix')
    reference = problem._build_su2_mpo()
    monkeypatch.setenv('PYQED_QC_AUTOMPO_LAYOUT', 'balanced')
    actual = problem._build_su2_mpo()
    # Explicit determinant-space reference is confined to this small test.
    expected = _dense_matrix_from_local_mpo(reference.canonical_factors)
    observed = _dense_matrix_from_local_mpo(actual.canonical_factors)
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-13)
    compiled = actual.native_mpo(problem.symmetry('su2').physical_basis)
    assert compiled.relative_reconstruction_error < 1e-11


@pytest.mark.parametrize('family', [None, (), (None,), 'P', ('P', None)])
def test_balanced_paths_accumulate_duplicates_and_keep_family_sums(family):
    from pyqed.mps.nonabelian.builder import AutoMPO
    from pyqed.mps.nonabelian.operators import spatial_number, spatial_parity
    number, parity = spatial_number(), spatial_parity()
    auto = AutoMPO([number.phys_out_leg] * 3)
    for weight in [.5, .5, -.25]:
        auto.add_term((0, number), (1, parity), (2, number),
                      coeff=weight, family=family)
    auto.add_term((0, number), (1, parity), (2, number), coeff=.3, family='Q')
    expected = 1.05 * np.kron(np.kron(number.as_dense(), parity.as_dense()), number.as_dense())
    actual = _dense_matrix_from_local_mpo(auto.build(sparse=True, balanced=True))
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_operator_caches_do_not_survive_a_build():
    from pyqed.mps.nonabelian.builder import AutoMPO
    from pyqed.mps.nonabelian.operators import spatial_create_up, spatial_annihilate_up
    create, destroy = spatial_create_up(), spatial_annihilate_up()
    auto = AutoMPO([create.phys_out_leg] * 2)
    auto.add_term((0, create), (1, destroy))
    first = _dense_matrix_from_local_mpo(auto.build(sparse=True, balanced=True))
    np.testing.assert_allclose(first, np.kron(create.as_dense(), destroy.as_dense()))
    creation_blocks = {key: value.copy() for key, value in create.blocks.items()}
    destruction_blocks = {key: value.copy() for key, value in destroy.blocks.items()}
    create.blocks.clear()
    create.blocks.update(destruction_blocks)
    destroy.blocks.clear()
    destroy.blocks.update(creation_blocks)
    second = _dense_matrix_from_local_mpo(auto.build(sparse=True, balanced=True))
    np.testing.assert_allclose(second, np.kron(create.as_dense(), destroy.as_dense()))
    assert not np.array_equal(first, second)

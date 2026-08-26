import numpy as np

from pyqed.qchem.dmrg.dmrg import (
    _build_spatial_fermion_operators,
    _build_spatial_s2_matrix,
)
from pyqed.qchem.dmrg.tddmrg import _mpo_to_dense_matrix
from pyqed.qchem.gdvr.tddmrg import TDDMRG, build_gdvr_spatial_hamiltonian_mpo


class _ToyMolecule:
    def __init__(self):
        self.z = np.array([-0.5, 0.75])
        self.shapes = {"Nz": 2, "M": 1, "size": 2}
        self.hcore = np.array([[-0.8, 0.13], [0.13, -0.25]])
        self.eri_j = [
            [np.array([[0.70]]), np.array([[0.31]])],
            [np.array([[0.31]]), np.array([[0.55]])],
        ]
        self.eri_k = [[block.copy() for block in row] for row in self.eri_j]
        self.nelec = 2
        self.spin = 0

    def nuclear_repulsion_energy(self):
        return 0.0


class _ToyRHF:
    def __init__(self):
        self.mol = _ToyMolecule()
        self.mo_coeff = np.eye(2)
        self.mo_energy = np.array([-0.8, -0.25])
        self.mo_occ = np.array([2.0, 0.0])
        self.dm = np.diag(self.mo_occ)
        self.e_tot = -1.0

    def energy_nuc(self):
        return self.mol.nuclear_repulsion_energy()

    def get_hcore(self):
        return self.mol.hcore

    def dipole(self, basis="ao"):
        z = np.diag(self.mol.z)
        return np.array([np.zeros_like(z), np.zeros_like(z), -z])


def test_direct_gdvr_build_applies_requested_spin_penalty():
    mf = _ToyRHF()
    shift = 0.37
    bare_mpo, _ = build_gdvr_spatial_hamiltonian_mpo(mf.mol)
    unpenalized = TDDMRG(mf).build()

    sadmrg = TDDMRG(mf)
    sadmrg.fix_spin(ss=0.0, shift=shift)
    sadmrg.build()

    bare = _mpo_to_dense_matrix(bare_mpo)
    penalized = _mpo_to_dense_matrix(sadmrg.H)
    s2 = _build_spatial_s2_matrix(_build_spatial_fermion_operators(2))
    np.testing.assert_allclose(penalized - bare, shift * s2, atol=1.0e-12)
    assert sadmrg._hamiltonian_mpo_cache_key != unpenalized._hamiltonian_mpo_cache_key

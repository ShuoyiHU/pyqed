import unittest

import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG, SymmetryManager, gen_hf_config
from pyqed.qchem.hf import RHF


class SectorCompleteInitialGuessTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.mol = Molecule(
            atom="N 0 0 0; N 0 0 1.1",
            unit="angstrom",
            basis="sto-3g",
        )
        self.mol.build(driver="gbasis")
        self.mf = RHF(self.mol).run(verbose=0)

    def test_hf_sector_noise_keeps_hf_path_dominant_and_opens_other_sectors(self):
        dmrg = DMRG(
            self.mf,
            ncas=6,
            nelecas=6,
            D=64,
            init_guess="hf_sector_noise",
            symmetry="sz",
            verbose=0,
        )
        dmrg.sym_mgr = SymmetryManager(["charge", "sz"])
        guess = dmrg.get_initial_guess_symmetric("hf_sector_noise")

        hf_config = gen_hf_config(6, 12)
        left_qn = dmrg.sym_mgr.get_vac_qn()
        for site, tensor in enumerate(guess):
            state = "occ" if hf_config[site] else "emp"
            physical_qn = dmrg.sym_mgr.get_phys_qn(site, state)
            right_qn = dmrg.sym_mgr.combine(left_qn, physical_qn)
            hf_key = (left_qn, right_qn, physical_qn)
            self.assertIn(hf_key, tensor.data)

            hf_amplitude = abs(tensor.data[hf_key][0, 0, 0])
            noise_amplitudes = [
                float(np.max(np.abs(block)))
                for key, block in tensor.data.items()
                if key != hf_key
            ]
            self.assertTrue(noise_amplitudes)
            self.assertGreater(hf_amplitude, 10.0 * max(noise_amplitudes))
            left_qn = right_qn

    def test_hf_sector_noise_recovers_n2_cas66_correlation(self):

        dmrg = DMRG(
            self.mf,
            ncas=6,
            nelecas=6,
            D=64,
            init_guess="hf_sector_noise",
            symmetry="sz",
            dmrg_performance="generic",
            verbose=0,
        )
        dmrg.run(nsweeps=20, sweep_tol=1.0e-9)

        self.assertLess(dmrg.e_tot, self.mf.e_tot - 0.1)
        self.assertGreater(max(dmrg.dmrg.ground_state.bond_orders()), 4)
        self.assertAlmostEqual(dmrg.e_tot, -107.62310178052438, places=7)


if __name__ == "__main__":
    unittest.main()
